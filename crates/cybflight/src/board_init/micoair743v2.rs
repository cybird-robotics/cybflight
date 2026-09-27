//! Board init for the MicoAir743v2 (BMI270-only configuration).
//!
//! Wires the BMI270 gyro (SPI3), SPL06 baro (I2C2), QMC5883L mag (I2C1), a
//! 4-motor quad DShot on TIM1, microSD blackbox, USB CDC, RC, ESP bridge and
//! power monitoring. See `bsp-micoair743v2` for the pin map.
//!
//! Divergences from `sakurah743.rs`:
//!   * single BMI270 IMU on SPI3 (no dual ICM, no SPI2 BMI088);
//!   * SPL06 baro on I2C2, QMC5883L mag on I2C1 (mag/baro on *separate* buses,
//!     and the internal/external bus roles are swapped vs Sakura);
//!   * motors M1–M4 on **TIM1** — an advanced-control timer, so `BDTR.MOE`
//!     must be enabled or the outputs stay Hi-Z (the DShot driver does not do
//!     this itself);
//!   * no CAN, no PINIO, no second baro;
//!   * the WS2812 arm-LED is **not** wired here — its driver is hard-bound to
//!     TIM1 (now the motor timer) and the board's LED pad is TIM4_CH3; a
//!     generic WS2812 retarget is a follow-up (see docs/architecture.md TODO).

use cybflight_drivers::baro::spl06::Spl06;
#[cfg(feature = "est_pos_gps")]
use cybflight_drivers::gps::GpsDriver;
use cybflight_drivers::imu::bmi270::Bmi270;
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
use crate::sensors::gps::{err_kind, GpsHealth, GpsRunner, GPS_HEALTH};
use crate::sensors::imu::{bmi270_reader_task, ImuReader, SpiBusMtx};
use crate::sensors::mag::{I2cBusMtx, MagReader};
use crate::sensors::power::{power_task, AdcInput, PowerMonitor};
use crate::status;
use crate::usb_serial;
use hal::gpio::{AfType, Flex, OutputType, Speed};
use hal::spi::{self, Spi};
use hal::time::Hertz;
use hal::timer::low_level::Timer as LLTimer;

// Buffered serial UARTs used as role candidates (SerialRx on USART6, GPS on
// USART3). board_init dispatches on the bsp::PORT_* constants.
hal::bind_interrupts!(struct SerialIrqs {
    USART6 => hal::usart::BufferedInterruptHandler<hal::peripherals::USART6>;
    USART3 => hal::usart::BufferedInterruptHandler<hal::peripherals::USART3>;
    USART2 => hal::usart::BufferedInterruptHandler<hal::peripherals::USART2>;
});

// I2C1 — QMC5883L mag (+ external connector).
hal::bind_interrupts!(struct I2c1Irqs {
    I2C1_EV => hal::i2c::EventInterruptHandler<hal::peripherals::I2C1>;
    I2C1_ER => hal::i2c::ErrorInterruptHandler<hal::peripherals::I2C1>;
});

// I2C2 — SPL06 baro.
hal::bind_interrupts!(struct I2c2Irqs {
    I2C2_EV => hal::i2c::EventInterruptHandler<hal::peripherals::I2C2>;
    I2C2_ER => hal::i2c::ErrorInterruptHandler<hal::peripherals::I2C2>;
});

// USART1 — ESP bridge (DMA UART).
hal::bind_interrupts!(struct Usart1Irqs {
    USART1 => hal::usart::InterruptHandler<hal::peripherals::USART1>;
});

// SDMMC1 — blackbox storage.
hal::bind_interrupts!(struct SdmmcIrqs {
    SDMMC1 => hal::sdmmc::InterruptHandler<hal::peripherals::SDMMC1>;
});

/// Board initialization. See `sakurah743::init` for the spawner-routing
/// rationale (thread / ctrl P10 / high P6).
pub async fn init(
    spawner: &Spawner,
    ctrl_spawner: &SendSpawner,
    high_spawner: &SendSpawner,
    board: bsp::Board,
) {
    // --- Load vehicle parameters from flash (or defaults) ---
    crate::params::init_from_flash(board.internal_flash);

    // Sensor topology params (IMU LPF cutoffs, mag hard-iron) — from the
    // vehicle YAML bake plus any flash overrides. Captured once at init.
    let sensor_params = crate::params::get().sensors;
    let accel_cutoff_hz = sensor_params.imu_accel_lpf_hz;
    let gyro_cutoff_hz = sensor_params.imu_gyro_lpf_hz;
    let mag_hard_iron =
        nalgebra::Vector3::from(crate::params::get().airframe.install.mag_hard_iron);

    // --- LEDs: led0 for status, others off ---
    let mut led1 = Led::new(board.leds.led1, false);
    let mut led2 = Led::new(board.leds.led2, false);
    led1.off();
    led2.off();
    let led0 = Led::new(board.leds.led0, false);
    spawner.spawn(status::task(led0)).unwrap();

    // --- Arm LED (WS2812 on PD14 / TIM4_CH3 / AF2 / DMA1_CH7) ---
    // Seed the live atomic from flash, then spawn the WS2812 task. TIM4 is a
    // general-purpose timer (no BDTR/MOE needed); enable its RCC clock and keep
    // it alive (forget) before handing the GP16 regs to the driver.
    crate::arm_led::ARM_LED_ENABLED.store(
        crate::params::get().system.arm_led_enabled,
        core::sync::atomic::Ordering::Relaxed,
    );
    let led_timer = LLTimer::new(board.motors.tim4);
    let led_tim_regs = led_timer.regs_gp16();
    core::mem::forget(led_timer);
    let arm_led_strip = crate::arm_led::ws2812::Ws2812::new(
        led_tim_regs,
        board.motors.led_strip,
        2,  // AF2 = TIM4_CH3
        2,  // CH3
        board.motors.dma1_ch7,
        31, // DMAMUX request: TIM4_CH3
    );
    spawner
        .spawn(crate::arm_led::task(arm_led_strip))
        .unwrap_or_else(|e| defmt::error!("Failed to spawn arm LED task: {}", e));

    // --- USB CDC serial ---
    spawner
        .spawn(usb_serial::task(
            board.usb.usb_otg_fs,
            board.usb.dp,
            board.usb.dm,
        ))
        .unwrap();

    // --- Blackbox / flight-data-recorder (microSD via SDMMC1, 4-bit) ---
    if bsp::HAS_BLACKBOX_STORAGE {
        let mut sdmmc_cfg = hal::sdmmc::Config::default();
        sdmmc_cfg.data_transfer_timeout = 5_000_000;
        let sdmmc = hal::sdmmc::Sdmmc::new_4bit(
            board.sdio.sdio,
            SdmmcIrqs,
            board.sdio.ck,
            board.sdio.cmd,
            board.sdio.d0,
            board.sdio.d1,
            board.sdio.d2,
            board.sdio.d3,
            sdmmc_cfg,
        );
        let store = crate::blackbox::SdmmcBlockStore::new(sdmmc);
        if let Err(e) = spawner.spawn(crate::blackbox::blackbox_task(store)) {
            defmt::error!("blackbox: spawn failed: {:?}", defmt::Debug2Format(&e));
        }
    }

    // Wait for power to stabilize before touching SPI/I2C devices.
    Timer::after_millis(100).await;

    // --- IMU1: BMI270 on SPI3 (PB3/PB4/PD6, CS=PA15, DRDY=PB7) ---
    // MODE3 per hwdef. 8 MHz is well within the BMI270's 10 MHz limit and
    // gives ample headroom for 3.2 kHz reads; drop to ~1 MHz if a marginal
    // board drops the microcode upload.
    let mut spi_config = spi::Config::default();
    spi_config.frequency = Hertz(8_000_000);
    spi_config.mode = spi::MODE_3;

    static SPI3_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi3 = Spi::new(
        board.spi.spi3,
        board.spi.spi3_sck,
        board.spi.spi3_mosi,
        board.spi.spi3_miso,
        board.spi.spi3_tx_dma,
        board.spi.spi3_rx_dma,
        spi_config,
    );
    let spi3_bus = SPI3_BUS.init(Mutex::new(spi3));
    let dev3 = SpiDevice::new(spi3_bus, board.sensors.gyro1_cs);

    match Bmi270::new(dev3, board.sensors.gyro1_drdy, embassy_time::Delay).await {
        Ok(imu1) => {
            defmt::info!("IMU1 (BMI270) init OK");
            ctrl_spawner
                .spawn(bmi270_reader_task(
                    ImuReader::new(
                        imu1,
                        board.sensors.gyro1_align,
                        accel_cutoff_hz,
                        gyro_cutoff_hz,
                    ),
                    &crate::sensors::IMU_1,
                    Some(&crate::sensors::IMU_1_RAW),
                ))
                .unwrap_or_else(|e| defmt::error!("Failed to spawn IMU1 reader task: {}", e));
        }
        Err(e) => defmt::error!("IMU1 (BMI270) init failed: {}", e),
    }

    // --- SerialRx (CRSF/GHST): bsp::PORT_SERIAL_RX selects the UART (USART6). ---
    #[cfg(feature = "rx_crsf")]
    match bsp::PORT_SERIAL_RX {
        bsp::SerialPortId::Usart6 => {
            static TX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            static RX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            let tx_buf = &mut TX_BUF.init([0u8; 128])[..];
            let rx_buf = &mut RX_BUF.init([0u8; 128])[..];
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 420_000;
            match hal::usart::BufferedUart::new(
                board.serial.usart6,
                board.serial.usart6_rx,
                board.serial.usart6_tx,
                tx_buf,
                rx_buf,
                SerialIrqs,
                uart_config,
            ) {
                Ok(uart) => {
                    defmt::info!("CRSF USART6 init OK");
                    ctrl_spawner
                        .spawn(crate::sensors::rc::crsf_runner::crsf_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn CRSF task: {}", e));
                }
                Err(e) => defmt::error!("CRSF USART6 init failed: {}", e),
            }
        }
        _ => defmt::warn!("CRSF: PORT_SERIAL_RX is not a supported SerialRx port on this board"),
    }

    #[cfg(feature = "rx_ghst")]
    match bsp::PORT_SERIAL_RX {
        bsp::SerialPortId::Usart6 => {
            static TX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            static RX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
            let tx_buf = &mut TX_BUF.init([0u8; 128])[..];
            let rx_buf = &mut RX_BUF.init([0u8; 128])[..];
            let mut uart_config = hal::usart::Config::default();
            uart_config.baudrate = 420_000;
            match hal::usart::BufferedUart::new_half_duplex(
                board.serial.usart6,
                board.serial.usart6_tx,
                SerialIrqs,
                tx_buf,
                rx_buf,
                uart_config,
                hal::usart::HalfDuplexReadback::NoReadback,
            ) {
                Ok(uart) => {
                    defmt::info!("GHST USART6 half-duplex init OK");
                    ctrl_spawner
                        .spawn(crate::sensors::rc::ghst_runner::ghst_task(uart))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn GHST task: {}", e));
                }
                Err(e) => defmt::error!("GHST USART6 init failed: {}", e),
            }
        }
        _ => defmt::warn!("GHST: PORT_SERIAL_RX is not a supported SerialRx port on this board"),
    }

    // --- GPS: bsp::PORT_GPS selects the UART (USART3). Gated on est_pos_gps. ---
    #[cfg(feature = "est_pos_gps")]
    {
        #[cfg(not(feature = "gps_unicore"))]
        const GPS_BAUD: u32 = 230_400;
        #[cfg(feature = "gps_unicore")]
        const GPS_BAUD: u32 = 115_200;
        GPS_HEALTH.lock(|c| c.set(GpsHealth::Initializing));
        match bsp::PORT_GPS {
            bsp::SerialPortId::Usart3 => {
                static GPS_TX_BUF: StaticCell<[u8; 256]> = StaticCell::new();
                static GPS_RX_BUF: StaticCell<[u8; 1024]> = StaticCell::new();
                let tx_buf = &mut GPS_TX_BUF.init([0u8; 256])[..];
                let rx_buf = &mut GPS_RX_BUF.init([0u8; 1024])[..];
                let mut uart_config = hal::usart::Config::default();
                uart_config.baudrate = GPS_BAUD;
                match hal::usart::BufferedUart::new(
                    board.serial.usart3,
                    board.serial.usart3_rx,
                    board.serial.usart3_tx,
                    tx_buf,
                    rx_buf,
                    SerialIrqs,
                    uart_config,
                ) {
                    Ok(uart) => {
                        let mut delay = embassy_time::Delay;
                        match with_timeout(Duration::from_secs(30), GpsDriver::new(uart, &mut delay))
                            .await
                        {
                            Ok(Ok(gps)) => {
                                defmt::info!("GPS u-blox init OK");
                                spawner
                                    .spawn(crate::sensors::gps::gps_task(GpsRunner::new(gps)))
                                    .unwrap_or_else(|e| {
                                        defmt::error!("Failed to spawn GPS task: {}", e)
                                    });
                            }
                            Ok(Err(e)) => {
                                GPS_HEALTH.lock(|c| c.set(GpsHealth::InitFailed(err_kind(&e))));
                                defmt::warn!("GPS init failed: {}", e);
                            }
                            Err(_) => {
                                GPS_HEALTH.lock(|c| c.set(GpsHealth::InitTimedOut));
                                defmt::warn!("GPS init timed out (no module?)");
                            }
                        }
                    }
                    Err(e) => {
                        GPS_HEALTH.lock(|c| c.set(GpsHealth::UartInitFailed));
                        defmt::error!("GPS USART3 init failed: {}", e);
                    }
                }
            }
            _ => {
                GPS_HEALTH.lock(|c| c.set(GpsHealth::UartInitFailed));
                defmt::warn!("GPS: PORT_GPS is not a supported GPS port on this board");
            }
        }
    }

    // --- I2C1 shared bus (PB8 SCL, PB9 SDA): QMC5883L internal mag (0x0D) ---
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
        let i2c1_bus: &'static I2cBusMtx = I2C1_BUS.init(Mutex::<NoopRawMutex, _>::new(i2c1));

        let mut probe_dev = I2cDevice::new(i2c1_bus);
        if Qmc5883l::probe(&mut probe_dev).await {
            defmt::info!("I2C1: QMC5883L found, initializing...");
            let dev = I2cDevice::new(i2c1_bus);
            let mut delay = embassy_time::Delay;
            match Qmc5883l::new(dev, &mut delay).await {
                Ok(mag) => {
                    defmt::info!("QMC5883L init OK — spawning internal mag task");
                    spawner
                        .spawn(crate::sensors::mag::qmc5883l_mag_int_task(MagReader::new(
                            mag,
                            board.sensors.mag_align,
                            mag_hard_iron,
                        )))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn QMC5883L task: {}", e));
                }
                Err(e) => defmt::warn!("QMC5883L init failed: {}", e),
            }
        } else {
            defmt::warn!("QMC5883L not detected on I2C1 (addr 0x0D)");
        }
    }

    // --- I2C2 shared bus (PB10 SCL, PB11 SDA): SPL06 baro (0x77) ---
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
        let i2c2_bus: &'static I2cBusMtx = I2C2_BUS.init(Mutex::<NoopRawMutex, _>::new(i2c2));

        let mut probe_dev = I2cDevice::new(i2c2_bus);
        if Spl06::probe_i2c(&mut probe_dev, bsp::sensors::BARO_1_I2C_ADDR).await {
            defmt::info!("I2C2: SPL06 found, initializing...");
            let dev = I2cDevice::new(i2c2_bus);
            let mut delay = embassy_time::Delay;
            match Spl06::new_i2c(dev, bsp::sensors::BARO_1_I2C_ADDR, &mut delay).await {
                Ok(baro) => {
                    defmt::info!("SPL06 init OK — spawning baro1 task");
                    spawner
                        .spawn(crate::sensors::baro::spl06_i2c_baro_task(
                            BaroReader::new(baro),
                            &crate::sensors::BARO_1,
                        ))
                        .unwrap_or_else(|e| defmt::error!("Failed to spawn SPL06 baro task: {}", e));
                }
                Err(e) => defmt::warn!("SPL06 init failed: {}", e),
            }
        } else {
            defmt::warn!("SPL06 not detected on I2C2 (addr 0x77)");
        }
    }

    // --- ESP bridge: bsp::PORT_ESP_BRIDGE selects the UART (USART1). ---
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
        _ => defmt::warn!("ESP bridge: PORT_ESP_BRIDGE is not a supported port on this board"),
    }

    // --- DShot motor output (quad on TIM1, M1–M4) ---
    //
    // TIM1 is an advanced-control timer: its outputs are gated by BDTR.MOE,
    // which the (general-purpose) DShot driver never sets. Enable it here once
    // — it persists through the driver's per-frame reconfiguration (which only
    // touches CCMR/CCER/CCR/DIER/CR1, never BDTR).
    let timer1 = LLTimer::new(board.motors.tim1);
    let tim1_regs = timer1.regs_gp16();
    timer1.regs_advanced().bdtr().modify(|w| w.set_moe(true));
    core::mem::forget(timer1); // keep RCC clock + MOE alive

    // Configure GPIO as timer AF (PE9/11/13/14 → TIM1 CH1–CH4, AF1).
    let af = AfType::output(OutputType::PushPull, Speed::Low);
    macro_rules! pin_af {
        ($pin:expr, $af_num:expr) => {{
            let mut flex = Flex::new($pin);
            flex.set_as_af_unchecked($af_num, af);
            core::mem::forget(flex);
        }};
    }
    pin_af!(board.motors.m1, 1); // PE14 AF1 = TIM1_CH4
    pin_af!(board.motors.m2, 1); // PE13 AF1 = TIM1_CH3
    pin_af!(board.motors.m3, 1); // PE11 AF1 = TIM1_CH2
    pin_af!(board.motors.m4, 1); // PE9  AF1 = TIM1_CH1

    let dshot_config = DshotQuadConfig {
        motors: [
            MotorTimerConfig {
                timer_regs: tim1_regs,
                channel_index: 3,
                dma_request: 14, // TIM1_CH4
                gpio_port: hal::pac::GPIOE,
                gpio_pin: 14,
                af_number: 1,
            }, // M1: PE14 TIM1_CH4
            MotorTimerConfig {
                timer_regs: tim1_regs,
                channel_index: 2,
                dma_request: 13, // TIM1_CH3
                gpio_port: hal::pac::GPIOE,
                gpio_pin: 13,
                af_number: 1,
            }, // M2: PE13 TIM1_CH3
            MotorTimerConfig {
                timer_regs: tim1_regs,
                channel_index: 1,
                dma_request: 12, // TIM1_CH2
                gpio_port: hal::pac::GPIOE,
                gpio_pin: 11,
                af_number: 1,
            }, // M3: PE11 TIM1_CH2
            MotorTimerConfig {
                timer_regs: tim1_regs,
                channel_index: 0,
                dma_request: 11, // TIM1_CH1
                gpio_port: hal::pac::GPIOE,
                gpio_pin: 9,
                af_number: 1,
            }, // M4: PE9 TIM1_CH1
        ],
        timers: [tim1_regs, tim1_regs],
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
    // MICOAIR743V2: VBAT=PC0 (ADC3_INP10), CURR=PC1 (ADC3_INP11).
    let power_mon = PowerMonitor::new(
        board.adc.adc3,
        board.adc.vbat,
        board.adc.curr,
        AdcInput {
            gpioc_pin: 0,
            channel: 10,
        },
        AdcInput {
            gpioc_pin: 1,
            channel: 11,
        },
        bsp::POWER_CAL,
    );
    spawner
        .spawn(power_task(power_mon))
        .unwrap_or_else(|_| defmt::error!("Failed to spawn power task"));
}
