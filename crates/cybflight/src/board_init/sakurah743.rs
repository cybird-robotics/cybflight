use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::led::Led;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_executor::Spawner;
use embassy_sync::mutex::Mutex;
use embassy_time::Timer;
use static_cell::StaticCell;

use crate::bsp;
use crate::hal;
use crate::motors::{DshotQuadConfig, MotorTimerConfig};
use crate::sensors::imu::{ImuReader, SpiBusMtx, icm_reader_task};
use crate::status;
use crate::usb_serial;
use hal::gpio::{AfType, Flex, OutputType, Speed};
use hal::spi::{self, Spi};
use hal::time::Hertz;
use hal::timer::low_level::Timer as LLTimer;

// Bind UART4 interrupt for SerialRx (CRSF/GHST)
hal::bind_interrupts!(struct Uart4Irqs {
    UART4 => hal::usart::BufferedInterruptHandler<hal::peripherals::UART4>;
});

pub async fn init(spawner: &Spawner, board: bsp::Board) {
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
            spawner
                .spawn(icm_reader_task(ImuReader::new(
                    imu1,
                    board.sensors.gyro1_align,
                    80.0,
                    200.0,
                )))
                .unwrap_or_else(|e| defmt::error!("Failed to spawn IMU1 reader task: {}", e));
        }
        Err(e) => defmt::error!("IMU1 init failed: {}", e),
    }

    // TODO: spawn IMU2 when fusion task exists
    // IMU2: IIM42652 on SPI1 (PA5/6/7, CS=PA4, DRDY=PC4)

    // --- SerialRx: UART4 ---
    // CRSF: full-duplex (separate TX/RX pins), standard Betaflight behavior.
    // GHST: single-wire half-duplex on TX pin (T4 pad = PB9), per BF SERIAL_BIDIR.
    #[cfg(feature = "rx_crsf")]
    {
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
            tx_buf,
            rx_buf,
            Uart4Irqs,
            uart_config,
        ) {
            Ok(uart) => {
                defmt::info!("CRSF UART4 init OK");
                spawner
                    .spawn(crate::sensors::rc::crsf_runner::crsf_task(uart))
                    .unwrap_or_else(|e| defmt::error!("Failed to spawn CRSF task: {}", e));
            }
            Err(e) => defmt::error!("CRSF UART4 init failed: {}", e),
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
            board.serial.uart4,
            board.serial.uart4_tx,
            Uart4Irqs,
            tx_buf,
            rx_buf,
            uart_config,
            hal::usart::HalfDuplexReadback::NoReadback,
        ) {
            Ok(uart) => {
                defmt::info!("GHST UART4 half-duplex init OK");
                spawner
                    .spawn(crate::sensors::rc::ghst_runner::ghst_task(uart))
                    .unwrap_or_else(|e| defmt::error!("Failed to spawn GHST task: {}", e));
            }
            Err(e) => defmt::error!("GHST UART4 init failed: {}", e),
        }
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
    let af = AfType::output(OutputType::PushPull, Speed::VeryHigh);
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
            }, // M1: PA2 TIM5_CH3
            MotorTimerConfig {
                timer_regs: tim5_regs,
                channel_index: 3,
                dma_request: 58,
                gpio_port: hal::pac::GPIOA,
                gpio_pin: 3,
            }, // M2: PA3 TIM5_CH4
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 3,
                dma_request: 26,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 1,
            }, // M3: PB1 TIM3_CH4
            MotorTimerConfig {
                timer_regs: tim3_regs,
                channel_index: 2,
                dma_request: 25,
                gpio_port: hal::pac::GPIOB,
                gpio_pin: 0,
            }, // M4: PB0 TIM3_CH3
        ],
        timers: [tim5_regs, tim3_regs],
        timer_count: 2,
    };

    spawner
        .spawn(crate::motors::dshot::dshot_task(
            dshot_config,
            board.motors.dma1_ch0,
            board.motors.dma1_ch1,
            board.motors.dma1_ch2,
            board.motors.dma1_ch3,
        ))
        .unwrap_or_else(|_| defmt::error!("Failed to spawn DShot task"));
}
