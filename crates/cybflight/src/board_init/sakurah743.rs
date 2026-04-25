use cybflight_drivers::baro::dps310::Dps310;
use cybflight_drivers::baro::icp20100::Icp20100;
use cybflight_drivers::gps::UbloxM10;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::led::Led;
use cybflight_drivers::mag::{Ist8310, Qmc5883l};
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_executor::{SendSpawner, Spawner};
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Duration, Timer, with_timeout};
use static_cell::StaticCell;

use crate::bsp;
use crate::hal;
use crate::motors::{DshotQuadConfig, MotorTimerConfig};
use crate::sensors::baro::BaroReader;
use crate::sensors::gps::GpsRunner;
use crate::sensors::imu::{ImuReader, SpiBusMtx, icm_reader_task};
use crate::sensors::mag::{I2cBusMtx, MagReader};
use crate::status;
use crate::usb_serial;
use hal::gpio::{AfType, Flex, OutputType, Speed};
use hal::spi::{self, Spi};
use hal::time::Hertz;
use hal::timer::low_level::Timer as LLTimer;

// Single interrupt struct covering all buffered serial UARTs used as role
// candidates. board_init dispatches based on the bsp::PORT_* constants
// defined in the BSP port mapping table; LLVM eliminates dead match arms
// since the discriminants are compile-time constants.
hal::bind_interrupts!(struct SerialIrqs {
    UART4 => hal::usart::BufferedInterruptHandler<hal::peripherals::UART4>;
    USART3 => hal::usart::BufferedInterruptHandler<hal::peripherals::USART3>;
});

// Bind I2C1 interrupts for onboard IST8310
hal::bind_interrupts!(struct I2c1Irqs {
    I2C1_EV => hal::i2c::EventInterruptHandler<hal::peripherals::I2C1>;
    I2C1_ER => hal::i2c::ErrorInterruptHandler<hal::peripherals::I2C1>;
});

// Bind I2C2 interrupts for external mag (QMC5883L)
hal::bind_interrupts!(struct I2c2Irqs {
    I2C2_EV => hal::i2c::EventInterruptHandler<hal::peripherals::I2C2>;
    I2C2_ER => hal::i2c::ErrorInterruptHandler<hal::peripherals::I2C2>;
});

// Bind USART1 interrupt for ESP bridge (DMA UART)
hal::bind_interrupts!(struct Usart1Irqs {
    USART1 => hal::usart::InterruptHandler<hal::peripherals::USART1>;
});

/// Board initialization.
///
/// Spawner routing:
/// - `spawner` (thread): LED, USB, RC reader, mag, baro, GPS, ESP bridge.
///   These tolerate latency; they run alongside the mission planner.
/// - `ctrl_spawner` (interrupt P10): **IMU readers**. Same executor as
///   the downstream attitude filter, ESKF, INDI and MPC — so the whole
///   sensor→estimation→control chain preempts any thread work (notably
///   the trajectory planner's BFGS solve). The SPI bus mutex type
///   (`SpiBusMtx`) was upgraded from `NoopRawMutex` to
///   `CriticalSectionRawMutex` to make the task `Send` for the
///   interrupt-executor spawn.
/// - `high_spawner` (interrupt P6): DShot motor output.
pub async fn init(
    spawner: &Spawner,
    ctrl_spawner: &SendSpawner,
    high_spawner: &SendSpawner,
    board: bsp::Board,
) {
    // --- Load vehicle parameters from flash (or defaults) ---
    crate::params::init_from_flash(board.internal_flash);

    // --- LEDs: turn off led1 & led2, use led0 for status ---
    let mut led1 = Led::new(board.leds.led1, false);
    let mut led2 = Led::new(board.leds.led2, false);
    led1.off();
    led2.off();
    let led0 = Led::new(board.leds.led0, false);
    spawner.spawn(status::task(led0)).unwrap();

    // --- USB CDC serial ---
    spawner
        .spawn(usb_serial::task(
            board.usb.usb_otg_fs,
            board.usb.dp,
            board.usb.dm,
        ))
        .unwrap();

    // Wait for power to stabilize before touching SPI devices.
    Timer::after_millis(100).await;

    let mut spi_config = spi::Config::default();
    spi_config.frequency = Hertz(1_000_000);
    spi_config.mode = spi::MODE_3;

    // --- IMU1: ICM42688P on SPI4 (PE12/13/14, CS=PE11, DRDY=PB2) ---
    static SPI4_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi4 = Spi::new(
        board.spi.spi4,
        board.spi.spi4_sck,
        board.spi.spi4_mosi,
        board.spi.spi4_miso,
        board.spi.spi4_tx_dma,
        board.spi.spi4_rx_dma,
        spi_config,
    );
    let spi4_bus = SPI4_BUS.init(Mutex::new(spi4));
    let dev4 = SpiDevice::new(spi4_bus, board.sensors.gyro1_cs);

    let mut delay = embassy_time::Delay;
    match Icm426xx::new(dev4, board.sensors.gyro1_drdy, &mut delay).await {
        Ok(imu1) => {
            defmt::info!("IMU1 init OK");
            // IMU reader → control executor (P10). Fresh gyro/accel
            // samples preempt any thread-executor work, including the
            // trajectory planner's BFGS solve.
            ctrl_spawner
                .spawn(icm_reader_task(
                    ImuReader::new(imu1, board.sensors.gyro1_align, 80.0, 200.0),
                    &crate::sensors::IMU_1,
                ))
                .unwrap_or_else(|e| defmt::error!("Failed to spawn IMU1 reader task: {}", e));
        }
        Err(e) => defmt::error!("IMU1 init failed: {}", e),
    }

    // --- SerialRx: bsp::PORT_SERIAL_RX selects the UART. ---
    // To move SerialRx to a different UART: change PORT_SERIAL_RX in bsp/sakurah743/src/lib.rs.
    // LLVM eliminates the dead match arms since PORT_SERIAL_RX is a compile-time constant.
    #[cfg(feature = "rx_crsf")]
    match bsp::PORT_SERIAL_RX {
        bsp::SerialPortId::Uart4 => {
            static TX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            static RX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            let tx_buf = &mut TX_BUF.init([0u8; 128])[..];
            let rx_buf = &mut RX_BUF.init([0u8; 128])[..];
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 420_000;
            match hal::usart::BufferedUart::new(
                board.serial.uart4,
                board.serial.uart4_rx,
                board.serial.uart4_tx,
                tx_buf, rx_buf, SerialIrqs, uart_config,
            ) {
                Ok(uart) => {
                    defmt::info!("CRSF UART4 init OK");
                    // CRSF parser on ctrl_spawner (P10): disarm-switch
                    // edge detection lives here (sensors/rc.rs:91-108),
                    // so running on P10 means the emergency-disarm path
                    // cannot be blocked by a thread-resident planner
                    // solve. Worst-case disarm latency ~2-3 ms.
                    ctrl_spawner
                        .spawn(crate::sensors::rc::crsf_runner::crsf_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn CRSF task: {}", e));
                }
                Err(e) => defmt::error!("CRSF UART4 init failed: {}", e),
            }
        }
        _ => defmt::warn!("CRSF: PORT_SERIAL_RX is not a supported SerialRx port on this board"),
    }

    #[cfg(feature = "rx_ghst")]
    match bsp::PORT_SERIAL_RX {
        bsp::SerialPortId::Uart4 => {
            static TX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            static RX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            let tx_buf = &mut TX_BUF.init([0u8; 128])[..];
            let rx_buf = &mut RX_BUF.init([0u8; 128])[..];
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 420_000;
            match hal::usart::BufferedUart::new_half_duplex(
                board.serial.uart4,
                board.serial.uart4_tx,
                SerialIrqs,
                tx_buf, rx_buf, uart_config,
                hal::usart::HalfDuplexReadback::NoReadback,
            ) {
                Ok(uart) => {
                    defmt::info!("GHST UART4 half-duplex init OK");
                    // GHST parser on ctrl_spawner — see CRSF rationale.
                    ctrl_spawner
                        .spawn(crate::sensors::rc::ghst_runner::ghst_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn GHST task: {}", e));
                }
                Err(e) => defmt::error!("GHST UART4 init failed: {}", e),
            }
        }
        _ => defmt::warn!("GHST: PORT_SERIAL_RX is not a supported SerialRx port on this board"),
    }

    // --- GPS: bsp::PORT_GPS selects the UART. ---
    match bsp::PORT_GPS {
        bsp::SerialPortId::Usart3 => {
            defmt::info!("GPS: starting USART3 init");
            static GPS_TX_BUF: StaticCell<[u8; 256]> = StaticCell::new();
            static GPS_RX_BUF: StaticCell<[u8; 256]> = StaticCell::new();
            let tx_buf = &mut GPS_TX_BUF.init([0u8; 256])[..];
            let rx_buf = &mut GPS_RX_BUF.init([0u8; 256])[..];
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 115_200;
            match hal::usart::BufferedUart::new(
                board.serial.usart3,
                board.serial.usart3_rx,
                board.serial.usart3_tx,
                tx_buf, rx_buf, SerialIrqs, uart_config,
            ) {
                Ok(uart) => {
                    defmt::info!("GPS: USART3 OK, sending CFG-VALSET...");
                    let mut delay = embassy_time::Delay;
                    match with_timeout(Duration::from_secs(3), UbloxM10::new(uart, &mut delay)).await {
                        Ok(Ok(gps)) => {
                            defmt::info!("GPS u-blox M10 init OK");
                            spawner
                                .spawn(crate::sensors::gps::ublox_gps_task(GpsRunner::new(gps)))
                                .unwrap_or_else(|e| defmt::error!("Failed to spawn GPS task: {}", e));
                        }
                        Ok(Err(e)) => defmt::warn!("GPS init failed: {}", e),
                        Err(_) => defmt::warn!("GPS init timed out (no module?)"),
                    }
                }
                Err(e) => defmt::error!("GPS USART3 init failed: {}", e),
            }
        }
        _ => defmt::warn!("GPS: PORT_GPS is not a supported GPS port on this board"),
    }

    // --- I2C2 shared bus (PB10 SCL, PB11 SDA) for external QMC5883L ---
    defmt::info!("I2C2: starting init");
    {
        static I2C2_BUS: StaticCell<I2cBusMtx> = StaticCell::new();
        let mut i2c_config = hal::i2c::Config::default();
        i2c_config.frequency = Hertz(400_000);
        let i2c2 = hal::i2c::I2c::new(
            board.i2c.i2c2,
            board.i2c.i2c2_scl,
            board.i2c.i2c2_sda,
            I2c2Irqs,
            board.i2c.i2c2_tx_dma,
            board.i2c.i2c2_rx_dma,
            i2c_config,
        );
        defmt::info!("I2C2: bus created at 400 kHz");
        let i2c2_bus: &'static I2cBusMtx = I2C2_BUS.init(Mutex::<NoopRawMutex, _>::new(i2c2));

        // QMC5883L external compass (addr 0x0D)
        defmt::info!("I2C2: probing QMC5883L at 0x0D...");
        let mut probe_dev = I2cDevice::new(i2c2_bus);
        if Qmc5883l::probe(&mut probe_dev).await {
            defmt::info!("I2C2: QMC5883L found, initializing...");
            let dev = I2cDevice::new(i2c2_bus);
            let mut delay = embassy_time::Delay;
            match Qmc5883l::new(dev, &mut delay).await {
                Ok(mag) => {
                    defmt::info!("QMC5883L init OK — spawning task");
                    spawner
                        .spawn(crate::sensors::mag::qmc5883l_mag_task(MagReader::new(
                            mag,
                            bsp_types::SensorAlign::Cw180Deg,
                            nalgebra::Vector3::zeros(),
                        )))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn QMC5883L task: {}", e));
                }
                Err(e) => defmt::warn!("QMC5883L init failed: {}", e),
            }
        } else {
            defmt::warn!("QMC5883L not detected on I2C2 (addr 0x0D)");
        }
    }

    // --- I2C1 shared bus (PB6 SCL, PB7 SDA) for ICP20100 baro1 + IST8310 mag ---
    defmt::info!("I2C1: starting init for ICP20100 + IST8310");
    {
        static I2C1_BUS: StaticCell<I2cBusMtx> = StaticCell::new();
        let mut i2c_config = hal::i2c::Config::default();
        i2c_config.frequency = Hertz(400_000);
        let i2c1 = hal::i2c::I2c::new(
            board.i2c.i2c1,
            board.i2c.i2c1_scl,
            board.i2c.i2c1_sda,
            I2c1Irqs,
            board.i2c.i2c1_tx_dma,
            board.i2c.i2c1_rx_dma,
            i2c_config,
        );
        defmt::info!("I2C1: bus created at 400 kHz");

        // I2C1 has no external pullups (ArduPilot hwdef: PULLUP flag on PB6/PB7).
        // Embassy I2C sets Pull::None by default, so enable internal pullups via GPIO register.
        hal::pac::GPIOB.pupdr().modify(|w| {
            w.set_pupdr(6, hal::pac::gpio::vals::Pupdr::PULL_UP); // PB6 SCL
            w.set_pupdr(7, hal::pac::gpio::vals::Pupdr::PULL_UP); // PB7 SDA
        });

        let i2c1_bus: &'static I2cBusMtx = I2C1_BUS.init(Mutex::<NoopRawMutex, _>::new(i2c1));

        // Allow pullups and bus to settle before probing
        Timer::after_millis(5).await;

        // Init all I2C1 devices before spawning tasks (avoid bus contention)
        let mut baro1_driver = None;
        let mut mag_int_driver = None;

        // ICP20100 barometer (addr 0x63) — init directly, no separate probe
        defmt::info!(
            "I2C1: initializing ICP20100 at {:#x}...",
            bsp::sensors::BARO_1_I2C_ADDR
        );
        {
            let dev = I2cDevice::new(i2c1_bus);
            let mut delay = embassy_time::Delay;
            match Icp20100::new(dev, bsp::sensors::BARO_1_I2C_ADDR, &mut delay).await {
                Ok(baro) => {
                    defmt::info!("ICP20100 init OK");
                    baro1_driver = Some(baro);
                }
                Err(e) => defmt::warn!("ICP20100 init failed: {}", e),
            }
        }

        // IST8310 internal magnetometer (addr 0x0E)
        defmt::info!(
            "I2C1: probing IST8310 at {:#x}...",
            bsp::sensors::MAG_I2C_ADDR
        );
        {
            let mut probe_dev = I2cDevice::new(i2c1_bus);
            let mut delay = embassy_time::Delay;
            if Ist8310::probe(&mut probe_dev, bsp::sensors::MAG_I2C_ADDR, &mut delay).await {
                defmt::info!("I2C1: IST8310 found, initializing...");
                let dev = I2cDevice::new(i2c1_bus);
                let mut delay = embassy_time::Delay;
                match Ist8310::new(dev, bsp::sensors::MAG_I2C_ADDR, &mut delay).await {
                    Ok(mag) => {
                        defmt::info!("IST8310 init OK");
                        mag_int_driver = Some(mag);
                    }
                    Err(e) => defmt::warn!("IST8310 init failed: {}", e),
                }
            } else {
                defmt::warn!(
                    "IST8310 not detected on I2C1 (addr {:#x})",
                    bsp::sensors::MAG_I2C_ADDR
                );
            }
        }

        // Spawn I2C1 tasks after all devices are initialized
        if let Some(baro) = baro1_driver {
            defmt::info!("Spawning baro1 task");
            spawner
                .spawn(crate::sensors::baro::icp20100_baro_task(
                    BaroReader::new(baro),
                    &crate::sensors::BARO_1,
                ))
                .unwrap_or_else(|e| defmt::error!("Failed to spawn ICP20100 baro1 task: {}", e));
        }
        if let Some(mag) = mag_int_driver {
            defmt::info!("Spawning mag int task");
            spawner
                .spawn(crate::sensors::mag::ist8310_mag_task(MagReader::new(
                    mag,
                    bsp_types::SensorAlign::Cw180Deg,
                    nalgebra::Vector3::zeros(),
                )))
                .unwrap_or_else(|e| defmt::error!("Failed to spawn IST8310 task: {}", e));
        }
    }

    // --- SPI1 shared bus (PA5/6/7) for BARO_2 (DPS310, CS=PC5) + IMU2 (IIM42652, CS=PA4) ---
    defmt::info!("SPI1: starting init for DPS310 baro2 + IIM42652 IMU2");
    {
        use crate::sensors::imu::SpiBusMtx;
        static SPI1_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
        let spi1 = Spi::new(
            board.spi.spi1,
            board.spi.spi1_sck,
            board.spi.spi1_mosi,
            board.spi.spi1_miso,
            board.spi.spi1_tx_dma,
            board.spi.spi1_rx_dma,
            spi_config,
        );
        let spi1_bus = SPI1_BUS.init(Mutex::new(spi1));

        // DPS310 barometer (CS=PC5)
        let mut probe_dev = SpiDevice::new(spi1_bus, board.sensors.baro2_cs);
        if Dps310::probe_spi(&mut probe_dev).await {
            defmt::info!("SPI1: DPS310 found, initializing...");
            let mut delay = embassy_time::Delay;
            match Dps310::new_spi(probe_dev, &mut delay).await {
                Ok(baro) => {
                    defmt::info!("DPS310 (SPI) init OK — spawning baro2 task");
                    spawner
                        .spawn(crate::sensors::baro::dps310_spi_baro_task(
                            BaroReader::new(baro),
                            &crate::sensors::BARO_2,
                        ))
                        .unwrap_or_else(|e| {
                            defmt::error!("Failed to spawn DPS310 baro2 task: {}", e)
                        });
                }
                Err(e) => defmt::warn!("DPS310 (SPI) init failed: {}", e),
            }
        } else {
            defmt::warn!("DPS310 not detected on SPI1");
        }

        // IMU2: IIM42652 (CS=PA4, DRDY=PC4) — same Icm426xx driver as IMU1
        let dev_imu2 = SpiDevice::new(spi1_bus, board.sensors.gyro2_cs);
        let mut delay = embassy_time::Delay;
        match Icm426xx::new(dev_imu2, board.sensors.gyro2_drdy, &mut delay).await {
            Ok(imu2) => {
                defmt::info!("IMU2 (IIM42652) init OK");
                // IMU2 reader → control executor (see IMU1 rationale).
                ctrl_spawner
                    .spawn(icm_reader_task(
                        ImuReader::new(imu2, board.sensors.gyro2_align, 80.0, 200.0),
                        &crate::sensors::IMU_2,
                    ))
                    .unwrap_or_else(|e| defmt::error!("Failed to spawn IMU2 reader task: {}", e));
            }
            Err(e) => defmt::warn!("IMU2 (IIM42652) init failed: {}", e),
        }
    }

    // --- ESP bridge: bsp::PORT_ESP_BRIDGE selects the UART. ---
    match bsp::PORT_ESP_BRIDGE {
        bsp::SerialPortId::Usart1 => {
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 921_600;

            match hal::usart::Uart::new(
                board.serial.usart1,
                board.serial.usart1_rx,
                board.serial.usart1_tx,
                Usart1Irqs,
                board.motors.dma1_ch4,
                board.motors.dma1_ch5,
                uart_config,
            ) {
                Ok(uart) => {
                    let (tx, rx) = uart.split();
                    defmt::info!("ESP bridge USART1 init OK (DMA)");
                    spawner
                        .spawn(crate::comm::esp_bridge::esp_bridge_rx_task(rx))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn ESP bridge RX: {}", e));
                    spawner
                        .spawn(crate::comm::esp_bridge::esp_bridge_tx_task(tx))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn ESP bridge TX: {}", e));
                }
                Err(e) => defmt::error!("ESP bridge USART1 init failed: {}", e),
            }
        }
        _ => defmt::warn!(
            "ESP bridge: PORT_ESP_BRIDGE is not a supported port on this board"
        ),
    }

    // --- DShot motor output ---

    // Enable RCC for motor timers (Timer::new enables clock + reset)
    let timer5 = LLTimer::new(board.motors.tim5);
    let timer3 = LLTimer::new(board.motors.tim3);
    let tim5_regs = timer5.regs_gp16();
    let tim3_regs = timer3.regs_gp16();
    // Prevent drop from disabling RCC clocks
    core::mem::forget(timer5);
    core::mem::forget(timer3);

    // Configure GPIO as timer AF (board-specific pins + AF numbers)
    let af = AfType::output(OutputType::PushPull, Speed::Low);
    macro_rules! pin_af {
        ($pin:expr, $af_num:expr) => {{
            let mut flex = Flex::new($pin);
            flex.set_as_af_unchecked($af_num, af);
            core::mem::forget(flex);
        }};
    }
    pin_af!(board.motors.m1, 2); // PA2 AF2 = TIM5_CH3
    pin_af!(board.motors.m2, 2); // PA3 AF2 = TIM5_CH4
    pin_af!(board.motors.m3, 2); // PB1 AF2 = TIM3_CH4
    pin_af!(board.motors.m4, 2); // PB0 AF2 = TIM3_CH3

    let dshot_config = DshotQuadConfig {
        motors: [
            MotorTimerConfig {
                timer_regs: tim5_regs,
                channel_index: 2,
                dma_request: 57,
                gpio_port: hal::pac::GPIOA,
                gpio_pin: 2,
                af_number: 2,
            }, // M1: PA2 TIM5_CH3
            MotorTimerConfig {
                timer_regs: tim5_regs,
                channel_index: 3,
                dma_request: 58,
                gpio_port: hal::pac::GPIOA,
                gpio_pin: 3,
                af_number: 2,
            }, // M2: PA3 TIM5_CH4
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 3,
                dma_request: 26,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 1,
                af_number: 2,
            }, // M3: PB1 TIM3_CH4
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 2,
                dma_request: 25,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 0,
                af_number: 2,
            }, // M4: PB0 TIM3_CH3
        ],
        timers: [tim5_regs, tim3_regs],
        timer_count: 2,
    };

    high_spawner
        .spawn(crate::motors::dshot::dshot_task(
            dshot_config,
            board.motors.dma1_ch0,
            board.motors.dma1_ch1,
            board.motors.dma1_ch2,
            board.motors.dma1_ch3,
        ))
        .unwrap_or_else(|_| defmt::error!("Failed to spawn DShot task"));
}
