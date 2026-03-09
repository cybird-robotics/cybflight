use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use cybflight_drivers::imu::{DetectedImu, probe_imu_raw};
use cybflight_drivers::led::Led;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_executor::{SendSpawner, Spawner};
use embassy_sync::mutex::Mutex;
use embassy_time::Timer;
use static_cell::StaticCell;

use crate::bsp;
use crate::hal;
use crate::motors::{DshotQuadConfig, MotorTimerConfig};
use crate::sensors::imu::{ImuReader, SpiBusMtx, icm_reader_task, mpu_reader_task};
use crate::status;
use crate::usb_serial;
use hal::gpio::{AfType, Flex, OutputType, Speed};
use hal::spi::{self, Spi};
use hal::time::Hertz;
use hal::timer::low_level::Timer as LLTimer;

// Bind USART1 interrupt for SerialRx (CRSF/GHST) — BF default: SERIALRX_UART = USART1
hal::bind_interrupts!(struct Usart1Irqs {
    USART1 => hal::usart::BufferedInterruptHandler<hal::peripherals::USART1>;
});

pub async fn init(spawner: &Spawner, high_spawner: &SendSpawner, board: bsp::Board) {
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
                    spawner
                        .spawn(icm_reader_task(ImuReader::new(
                            imu1,
                            board.sensors.gyro1_align,
                            ACCEL_CUTOFF_HZ,
                            GYRO_CUTOFF_HZ,
                        )))
                        .unwrap();
                }
                Err(e) => defmt::error!("IMU1 ICM init failed: {}", e),
            }
        }
        Ok(DetectedImu::Mpu6000) | Ok(DetectedImu::Mpu6500) => {
            match Mpu6x00::new(dev2, board.sensors.gyro1_drdy, &mut delay).await {
                Ok(imu1) => {
                    defmt::info!("IMU1 init OK (MPU)");
                    spawner
                        .spawn(mpu_reader_task(ImuReader::new(
                            imu1,
                            board.sensors.gyro1_align,
                            ACCEL_CUTOFF_HZ,
                            GYRO_CUTOFF_HZ,
                        )))
                        .unwrap();
                }
                Err(e) => defmt::error!("IMU1 MPU init failed: {}", e),
            }
        }
        Err(id) => {
            defmt::error!("Unknown IMU: WHO_AM_I={:#x}", id);
        }
    }

    // --- SerialRx: USART1 on PA10 (RX) / PA9 (TX) — BF default ---
    // CRSF: full-duplex (separate TX/RX pins), standard Betaflight behavior.
    // GHST: single-wire half-duplex on TX pin (T1 pad = PA9), per BF SERIAL_BIDIR.
    #[cfg(feature = "rx_crsf")]
    {
        static TX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
        static RX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
        let tx_buf = &mut TX_BUF.init([0u8; 128])[..];
        let rx_buf = &mut RX_BUF.init([0u8; 128])[..];

        let mut uart_config = hal::usart::Config::default();
        uart_config.baudrate = 420_000;

        match hal::usart::BufferedUart::new(
            board.serial.usart1,
            board.serial.usart1_rx,
            board.serial.usart1_tx,
            tx_buf,
            rx_buf,
            Usart1Irqs,
            uart_config,
        ) {
            Ok(uart) => {
                defmt::info!("CRSF USART1 init OK");
                spawner
                    .spawn(crate::sensors::rc::crsf_runner::crsf_task(uart))
                    .unwrap_or_else(|e| defmt::error!("Failed to spawn CRSF task: {}", e));
            }
            Err(e) => defmt::error!("CRSF USART1 init failed: {}", e),
        }
    }

    #[cfg(feature = "rx_ghst")]
    {
        static TX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
        static RX_BUF: StaticCell<[u8; 128]> = StaticCell::new();
        let tx_buf = &mut TX_BUF.init([0u8; 128])[..];
        let rx_buf = &mut RX_BUF.init([0u8; 128])[..];

        let mut uart_config = hal::usart::Config::default();
        uart_config.baudrate = 420_000;

        match hal::usart::BufferedUart::new_half_duplex(
            board.serial.usart1,
            board.serial.usart1_tx,
            Usart1Irqs,
            tx_buf,
            rx_buf,
            uart_config,
            hal::usart::HalfDuplexReadback::NoReadback,
        ) {
            Ok(uart) => {
                defmt::info!("GHST USART1 half-duplex init OK");
                spawner
                    .spawn(crate::sensors::rc::ghst_runner::ghst_task(uart))
                    .unwrap_or_else(|e| defmt::error!("Failed to spawn GHST task: {}", e));
            }
            Err(e) => defmt::error!("GHST USART1 init failed: {}", e),
        }
    }

    // --- DShot motor output ---

    // Enable RCC for motor timer (Timer::new enables clock + reset)
    let timer3 = LLTimer::new(board.motors.tim3);
    let tim3_regs = timer3.regs_gp16();
    // Prevent drop from disabling RCC clock
    core::mem::forget(timer3);

    // Configure GPIO as timer AF (board-specific pins + AF numbers)
    let af = AfType::output(OutputType::PushPull, Speed::VeryHigh);
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
}
