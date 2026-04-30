use cybflight_drivers::baro::dps310::Dps310;
#[cfg(feature = "est_pos_gps")]
use cybflight_drivers::gps::Ublox;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use cybflight_drivers::imu::{probe_imu_raw, DetectedImu};
use cybflight_drivers::led::Led;
use cybflight_drivers::mag::Qmc5883l;
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_executor::{SendSpawner, Spawner};
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::Timer;
#[cfg(feature = "est_pos_gps")]
use embassy_time::{with_timeout, Duration};
use static_cell::StaticCell;

use crate::bsp;
use crate::hal;
use crate::motors::{DshotQuadConfig, MotorTimerConfig};
use crate::sensors::baro::BaroReader;
#[cfg(feature = "est_pos_gps")]
use crate::sensors::gps::GpsRunner;
use crate::sensors::imu::{icm_reader_task, mpu_reader_task, ImuReader, SpiBusMtx};
use crate::sensors::mag::{I2cBusMtx, MagReader};
use crate::sensors::power::{PowerMonitor, power_task};
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
    USART1 => hal::usart::BufferedInterruptHandler<hal::peripherals::USART1>;
    USART2 => hal::usart::BufferedInterruptHandler<hal::peripherals::USART2>;
    UART4  => hal::usart::BufferedInterruptHandler<hal::peripherals::UART4>;
});

// Bind I2C1 interrupts for DPS310 baro + QMC5883L external mag
hal::bind_interrupts!(struct I2c1Irqs {
    I2C1_EV => hal::i2c::EventInterruptHandler<hal::peripherals::I2C1>;
    I2C1_ER => hal::i2c::ErrorInterruptHandler<hal::peripherals::I2C1>;
});

// Bind USART6 interrupt for ESP bridge (DMA UART)
hal::bind_interrupts!(struct Usart6Irqs {
    USART6 => hal::usart::InterruptHandler<hal::peripherals::USART6>;
});

/// Board initialization (see `sakurah743::init` for routing rationale).
/// IMU readers run on `ctrl_spawner` (interrupt P10) so the
/// sensor→estimation→control chain is never starved by thread work.
pub async fn init(
    spawner: &Spawner,
    ctrl_spawner: &SendSpawner,
    high_spawner: &SendSpawner,
    board: bsp::Board,
) {
    // --- Load vehicle parameters from flash (or defaults) ---
    crate::params::init_from_flash(board.internal_flash);

    // --- LED: only 1 LED, use it for status ---
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

    // --- IMU1 on SPI2 (PB13/14/15, CS=PB12, DRDY=PD0) ---
    // Probe WHO_AM_I on raw bus before wrapping in Mutex/SpiDevice.
    let mut spi2 = Spi::new(
        board.spi.spi2,
        board.spi.spi2_sck,
        board.spi.spi2_mosi,
        board.spi.spi2_miso,
        board.spi.spi2_tx_dma,
        board.spi.spi2_rx_dma,
        spi_config,
    );
    let mut cs = board.sensors.gyro1_cs;
    let detected = probe_imu_raw(&mut spi2, &mut cs).await;
    defmt::info!("IMU probe: {:?}", detected);

    // Wrap bus in shared mutex now that probe is done.
    static SPI2_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi2_bus = SPI2_BUS.init(Mutex::new(spi2));
    let dev2 = SpiDevice::new(spi2_bus, cs);

    let mut delay = embassy_time::Delay;
    const ACCEL_CUTOFF_HZ: f32 = 20.0;
    const GYRO_CUTOFF_HZ: f32 = 150.0;

    match detected {
        Ok(DetectedImu::Icm42605)
        | Ok(DetectedImu::Icm42622P)
        | Ok(DetectedImu::Icm42688P)
        | Ok(DetectedImu::Iim42652)
        | Ok(DetectedImu::Iim42653) => {
            match Icm426xx::new(dev2, board.sensors.gyro1_drdy, &mut delay).await {
                Ok(imu1) => {
                    defmt::info!("IMU1 init OK (ICM)");
                    // IMU reader → control executor so fresh samples
                    // preempt thread work (e.g. planner BFGS solve).
                    ctrl_spawner
                        .spawn(icm_reader_task(
                            ImuReader::new(
                                imu1,
                                board.sensors.gyro1_align,
                                ACCEL_CUTOFF_HZ,
                                GYRO_CUTOFF_HZ,
                            ),
                            &crate::sensors::IMU_1,
                        ))
                        .unwrap();
                }
                Err(e) => defmt::error!("IMU1 ICM init failed: {}", e),
            }
        }
        Ok(DetectedImu::Mpu6000) | Ok(DetectedImu::Mpu6500) => {
            match Mpu6x00::new(dev2, board.sensors.gyro1_drdy, &mut delay).await {
                Ok(imu1) => {
                    defmt::info!("IMU1 init OK (MPU)");
                    // IMU reader → control executor (see ICM branch).
                    ctrl_spawner
                        .spawn(mpu_reader_task(
                            ImuReader::new(
                                imu1,
                                board.sensors.gyro1_align,
                                ACCEL_CUTOFF_HZ,
                                GYRO_CUTOFF_HZ,
                            ),
                            &crate::sensors::IMU_1,
                        ))
                        .unwrap();
                }
                Err(e) => defmt::error!("IMU1 MPU init failed: {}", e),
            }
        }
        Err(id) => {
            defmt::error!("Unknown IMU: WHO_AM_I={:#x}", id);
        }
    }

    static TX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
    static RX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
    let tx_buf = &mut TX_BUF.init([0u8; 128])[..];
    let rx_buf = &mut RX_BUF.init([0u8; 128])[..];
    let mut rc_uart_cfg = hal::usart::Config::default();
    rc_uart_cfg.baudrate = 420_000;

    // --- SerialRx: bsp::PORT_SERIAL_RX selects the UART. ---
    // To move SerialRx to USART1: change PORT_SERIAL_RX in bsp/foxeerh743/src/lib.rs.
    // LLVM eliminates the dead match arms since PORT_SERIAL_RX is a compile-time constant.
    #[cfg(feature = "rx_crsf")]
    match bsp::PORT_SERIAL_RX {
        bsp::SerialPortId::Usart2 => {
            match hal::usart::BufferedUart::new(
                board.serial.usart2,
                board.serial.usart2_rx,
                board.serial.usart2_tx,
                tx_buf,
                rx_buf,
                SerialIrqs,
                rc_uart_cfg,
            ) {
                Ok(uart) => {
                    defmt::info!("CRSF USART2 init OK (T2=PA2, R2=PA3)");
                    // CRSF parser on ctrl_spawner (P10) — arm-switch edge
                    // detection lives inside the parser, so this keeps
                    // emergency-disarm response sub-5 ms even while the
                    // thread executor is blocked by a planner solve.
                    ctrl_spawner
                        .spawn(crate::sensors::rc::crsf_runner::crsf_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn CRSF task: {}", e));
                }
                Err(e) => defmt::error!("CRSF USART2 init failed: {}", e),
            }
        }
        bsp::SerialPortId::Usart1 => {
            match hal::usart::BufferedUart::new(
                board.serial.usart1,
                board.serial.usart1_rx,
                board.serial.usart1_tx,
                tx_buf,
                rx_buf,
                SerialIrqs,
                rc_uart_cfg,
            ) {
                Ok(uart) => {
                    defmt::info!("CRSF USART1 init OK (T1=PA9, R1=PA10)");
                    // CRSF parser on ctrl_spawner (P10) — arm-switch edge
                    // detection lives inside the parser, so this keeps
                    // emergency-disarm response sub-5 ms even while the
                    // thread executor is blocked by a planner solve.
                    ctrl_spawner
                        .spawn(crate::sensors::rc::crsf_runner::crsf_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn CRSF task: {}", e));
                }
                Err(e) => defmt::error!("CRSF USART1 init failed: {}", e),
            }
        }
        _ => defmt::warn!("CRSF: PORT_SERIAL_RX is not a supported SerialRx port on this board"),
    }

    #[cfg(feature = "rx_ghst")]
    match bsp::PORT_SERIAL_RX {
        bsp::SerialPortId::Usart2 => {
            match hal::usart::BufferedUart::new_half_duplex(
                board.serial.usart2,
                board.serial.usart2_tx,
                SerialIrqs,
                tx_buf,
                rx_buf,
                rc_uart_cfg,
                hal::usart::HalfDuplexReadback::NoReadback,
            ) {
                Ok(uart) => {
                    defmt::info!("GHST USART2 half-duplex init OK (T2 pad = PA2)");
                    // GHST parser on ctrl_spawner — see CRSF rationale.
                    ctrl_spawner
                        .spawn(crate::sensors::rc::ghst_runner::ghst_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn GHST task: {}", e));
                }
                Err(e) => defmt::error!("GHST USART2 init failed: {}", e),
            }
        }
        bsp::SerialPortId::Usart1 => {
            match hal::usart::BufferedUart::new_half_duplex(
                board.serial.usart1,
                board.serial.usart1_tx,
                SerialIrqs,
                tx_buf,
                rx_buf,
                rc_uart_cfg,
                hal::usart::HalfDuplexReadback::NoReadback,
            ) {
                Ok(uart) => {
                    defmt::info!("GHST USART1 half-duplex init OK (T1 pad = PA9)");
                    // GHST parser on ctrl_spawner — see CRSF rationale.
                    ctrl_spawner
                        .spawn(crate::sensors::rc::ghst_runner::ghst_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn GHST task: {}", e));
                }
                Err(e) => defmt::error!("GHST USART1 init failed: {}", e),
            }
        }
        _ => defmt::warn!("GHST: PORT_SERIAL_RX is not a supported SerialRx port on this board"),
    }

    // --- GPS: bsp::PORT_GPS selects the UART. ---
    //
    // Gated on `est_pos_gps` so non-GPS builds skip UART setup and the
    // 3-second u-blox-init timeout entirely.
    #[cfg(feature = "est_pos_gps")]
    match bsp::PORT_GPS {
        bsp::SerialPortId::Uart4 => {
            defmt::info!("GPS: starting UART4 init");
            static GPS_TX_BUF: StaticCell<[u8; 256]> = StaticCell::new();
            static GPS_RX_BUF: StaticCell<[u8; 256]> = StaticCell::new();
            let tx_buf = &mut GPS_TX_BUF.init([0u8; 256])[..];
            let rx_buf = &mut GPS_RX_BUF.init([0u8; 256])[..];
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 115_200;
            match hal::usart::BufferedUart::new(
                board.serial.uart4,
                board.serial.uart4_rx,
                board.serial.uart4_tx,
                tx_buf,
                rx_buf,
                SerialIrqs,
                uart_config,
            ) {
                Ok(uart) => {
                    defmt::info!("GPS: UART4 OK, sending CFG-VALSET...");
                    let mut delay = embassy_time::Delay;
                    match with_timeout(Duration::from_secs(3), Ublox::new(uart, &mut delay))
                        .await
                    {
                        Ok(Ok(gps)) => {
                            defmt::info!("GPS u-blox init OK");
                            spawner
                                .spawn(crate::sensors::gps::ublox_gps_task(GpsRunner::new(gps)))
                                .unwrap_or_else(|e| {
                                    defmt::error!("Failed to spawn GPS task: {}", e)
                                });
                        }
                        Ok(Err(e)) => defmt::warn!("GPS init failed: {}", e),
                        Err(_) => defmt::warn!("GPS init timed out (no module?)"),
                    }
                }
                Err(e) => defmt::error!("GPS UART4 init failed: {}", e),
            }
        }
        _ => defmt::warn!("GPS: PORT_GPS is not a supported GPS port on this board"),
    }

    // --- I2C1 shared bus (PB8 SCL, PB9 SDA) for DPS310 baro + QMC5883L ---
    defmt::info!("I2C1: starting init");
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
        let i2c1_bus: &'static I2cBusMtx = I2C1_BUS.init(Mutex::<NoopRawMutex, _>::new(i2c1));

        // QMC5883L external compass (addr 0x0D)
        defmt::info!("I2C1: probing QMC5883L at 0x0D...");
        let mut probe_dev = I2cDevice::new(i2c1_bus);
        if Qmc5883l::probe(&mut probe_dev).await {
            defmt::info!("I2C1: QMC5883L found, initializing...");
            let dev = I2cDevice::new(i2c1_bus);
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
            defmt::warn!("QMC5883L not detected on I2C1 (addr 0x0D)");
        }

        // DPS310 barometer (addr 0x76) on same I2C1 bus
        defmt::info!(
            "I2C1: probing DPS310 at {:#x}...",
            bsp::sensors::BARO_1_I2C_ADDR
        );
        let mut probe_dev = I2cDevice::new(i2c1_bus);
        if Dps310::probe_i2c(&mut probe_dev, bsp::sensors::BARO_1_I2C_ADDR).await {
            defmt::info!("I2C1: DPS310 found, initializing...");
            let dev = I2cDevice::new(i2c1_bus);
            let mut delay = embassy_time::Delay;
            match Dps310::new_i2c(dev, bsp::sensors::BARO_1_I2C_ADDR, &mut delay).await {
                Ok(baro) => {
                    defmt::info!("DPS310 (I2C) init OK — spawning baro1 task");
                    spawner
                        .spawn(crate::sensors::baro::dps310_i2c_baro_task(
                            BaroReader::new(baro),
                            &crate::sensors::BARO_1,
                        ))
                        .unwrap_or_else(|e| {
                            defmt::error!("Failed to spawn DPS310 baro1 task: {}", e)
                        });
                }
                Err(e) => defmt::warn!("DPS310 (I2C) init failed: {}", e),
            }
        } else {
            defmt::warn!(
                "DPS310 not detected on I2C1 (addr {:#x})",
                bsp::sensors::BARO_1_I2C_ADDR
            );
        }
    }

    // --- ESP bridge: bsp::PORT_ESP_BRIDGE selects the UART. ---
    match bsp::PORT_ESP_BRIDGE {
        bsp::SerialPortId::Usart6 => {
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 921_600;

            match hal::usart::Uart::new(
                board.serial.usart6,
                board.serial.usart6_rx,
                board.serial.usart6_tx,
                Usart6Irqs,
                board.motors.dma1_ch4,
                board.motors.dma1_ch5,
                uart_config,
            ) {
                Ok(uart) => {
                    let (tx, rx) = uart.split();
                    defmt::info!("ESP bridge USART6 init OK (DMA)");
                    spawner
                        .spawn(crate::comm::esp_bridge::esp_bridge_rx_task(rx))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn ESP bridge RX: {}", e));
                    spawner
                        .spawn(crate::comm::esp_bridge::esp_bridge_tx_task(tx))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn ESP bridge TX: {}", e));
                }
                Err(e) => defmt::error!("ESP bridge USART6 init failed: {}", e),
            }
        }
        _ => defmt::warn!(
            "ESP bridge: PORT_ESP_BRIDGE is not a supported port on this board"
        ),
    }

    // --- DShot motor output ---

    // Enable RCC for motor timer (Timer::new enables clock + reset)
    let timer3 = LLTimer::new(board.motors.tim3);
    let tim3_regs = timer3.regs_gp16();
    // Prevent drop from disabling RCC clock
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
    pin_af!(board.motors.m1, 2); // PB4 AF2 = TIM3_CH1
    pin_af!(board.motors.m2, 2); // PB5 AF2 = TIM3_CH2
    pin_af!(board.motors.m3, 2); // PB0 AF2 = TIM3_CH3
    pin_af!(board.motors.m4, 2); // PB1 AF2 = TIM3_CH4

    let dshot_config = DshotQuadConfig {
        motors: [
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 0,
                dma_request: 23,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 4,
                af_number: 2,
            }, // M1: PB4 TIM3_CH1
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 1,
                dma_request: 24,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 5,
                af_number: 2,
            }, // M2: PB5 TIM3_CH2
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 2,
                dma_request: 25,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 0,
                af_number: 2,
            }, // M3: PB0 TIM3_CH3
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 3,
                dma_request: 26,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 1,
                af_number: 2,
            }, // M4: PB1 TIM3_CH4
        ],
        timers: [tim3_regs, tim3_regs], // Only 1 unique timer; timers[1] unused padding
        timer_count: 1,
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

    // --- Power monitoring (ADC3) ---
    // Placed after DShot so motor control starts even if ADC init has issues.
    let power_mon = PowerMonitor::new(
        board.adc.adc3,
        board.adc.vbat,
        board.adc.curr,
        bsp::POWER_CAL,
    );
    spawner
        .spawn(power_task(power_mon))
        .unwrap_or_else(|_| defmt::error!("Failed to spawn power task"));
}
