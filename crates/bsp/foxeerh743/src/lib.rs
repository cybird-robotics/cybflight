#![no_std]

pub use embassy_stm32 as hal;

use hal::exti;
use hal::gpio::{Level, Output, Pull, Speed};
use hal::{bind_interrupts, Config, Peripherals};

pub use bsp_types::{DmaHint, MotorMeta, SensorAlign, SerialPortId, SerialRole, TimerChannel, TimerId};
pub use cybflight_drivers::beeper::Beeper;

/// Board name from Betaflight target.
pub const BOARD_NAME: &str = "FOXEERH743";
/// Manufacturer ID from Betaflight target.
pub const MANUFACTURER_ID: &str = "FOXE";

/// Betaflight target: beeper is inverted (active-low).
pub const BEEPER_INVERTED: bool = true;

pub const IMU_COUNT: usize = 1;
pub const HAS_BARO: bool = true;
pub const HAS_MAG: bool = false;
pub const HAS_OSD: bool = true;
pub const HAS_FLASH: bool = true;
pub const HAS_SDCARD: bool = false;
pub const LED_COUNT: usize = 1;

pub const DEFAULT_SERIAL_ROLE_MAP: &[(SerialRole, SerialPortId)] = &[
    (SerialRole::SerialRx, SerialPortId::Usart1),
];

/// Sensor identities from Betaflight header.
pub mod sensors {
    /// Betaflight: ICM42688P (also supports MPU6000/MPU6500) on SPI2 (GYRO_1).
    pub const GYRO_1: &str = "ICM42688P";
    /// Betaflight: DPS310 barometer on I2C1.
    pub const BARO: &str = "DPS310";
    /// Betaflight: magnetometer on I2C1, align CW180.
    pub const MAG: &str = "MAG";
}

/// Motor metadata table in motor index order (M1..M8).
///
/// M1  PB4  TIM3 CH1 AF2  (DMA1 Stream0 Req23)
/// M2  PB5  TIM3 CH2 AF2  (DMA1 Stream1 Req24)
/// M3  PB0  TIM3 CH3 AF2  (DMA1 Stream2 Req25)
/// M4  PB1  TIM3 CH4 AF2  (DMA1 Stream3 Req26)
/// M5  PD12 TIM4 CH1 AF2  (DMA1 Stream4 Req29)
/// M6  PD13 TIM4 CH2 AF2  (DMA1 Stream5 Req30)
/// M7  PC8  TIM8 CH3 AF3  (DMA1 Stream6 Req49)
/// M8  PC9  TIM8 CH4 AF3  (DMA1 Stream7 Req50)
pub const MOTOR_META: [MotorMeta; 8] = [
    MotorMeta {
        timer: TimerId::Tim3,
        channel: TimerChannel::Ch1,
        af: 2,
        dma: Some(DmaHint { stream: 0, request: 23 }),
    },
    MotorMeta {
        timer: TimerId::Tim3,
        channel: TimerChannel::Ch2,
        af: 2,
        dma: Some(DmaHint { stream: 1, request: 24 }),
    },
    MotorMeta {
        timer: TimerId::Tim3,
        channel: TimerChannel::Ch3,
        af: 2,
        dma: Some(DmaHint { stream: 2, request: 25 }),
    },
    MotorMeta {
        timer: TimerId::Tim3,
        channel: TimerChannel::Ch4,
        af: 2,
        dma: Some(DmaHint { stream: 3, request: 26 }),
    },
    MotorMeta {
        timer: TimerId::Tim4,
        channel: TimerChannel::Ch1,
        af: 2,
        dma: Some(DmaHint { stream: 4, request: 29 }),
    },
    MotorMeta {
        timer: TimerId::Tim4,
        channel: TimerChannel::Ch2,
        af: 2,
        dma: Some(DmaHint { stream: 5, request: 30 }),
    },
    MotorMeta {
        timer: TimerId::Tim8,
        channel: TimerChannel::Ch3,
        af: 3,
        dma: Some(DmaHint { stream: 6, request: 49 }),
    },
    MotorMeta {
        timer: TimerId::Tim8,
        channel: TimerChannel::Ch4,
        af: 3,
        dma: Some(DmaHint { stream: 7, request: 50 }),
    },
];

/// Servo PWM mapping metadata (from Betaflight target).
pub const SERVO_META: &[(/*pin*/ &str, TimerId, TimerChannel, /*af*/ u8)] = &[
    ("PE5", TimerId::Tim15, TimerChannel::Ch1, 4),
    ("PE6", TimerId::Tim15, TimerChannel::Ch2, 4),
];

/// LED strip pin mapping metadata.
pub const LED_STRIP_META: (/*pin*/ &str, TimerId, TimerChannel, /*af*/ u8, DmaHint) =
    ("PA8", TimerId::Tim1, TimerChannel::Ch1, 1, DmaHint { stream: 0, request: 11 });

// EXTI interrupt required by gyro DRDY pin:
// - PD0 (EXTI0) for GYRO_1
bind_interrupts!(pub struct ExtiIrqs {
    EXTI0 => exti::InterruptHandler<hal::interrupt::typelevel::EXTI0>;
});

/// Onboard LED (PC13).
pub struct Leds {
    pub led0: Output<'static>,
}

pub struct SpiPins {
    // SPI1 (PA5/6/7): OSD (MAX7456)
    pub spi1: hal::Peri<'static, hal::peripherals::SPI1>,
    pub spi1_sck: hal::Peri<'static, hal::peripherals::PA5>,
    pub spi1_miso: hal::Peri<'static, hal::peripherals::PA6>,
    pub spi1_mosi: hal::Peri<'static, hal::peripherals::PA7>,

    // SPI2 (PB13/14/15): gyro
    pub spi2: hal::Peri<'static, hal::peripherals::SPI2>,
    pub spi2_sck: hal::Peri<'static, hal::peripherals::PB13>,
    pub spi2_miso: hal::Peri<'static, hal::peripherals::PB14>,
    pub spi2_mosi: hal::Peri<'static, hal::peripherals::PB15>,
    pub spi2_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH0>,
    pub spi2_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH1>,

    // SPI3 (PC10/11/12): flash (W25Q128FV)
    pub spi3: hal::Peri<'static, hal::peripherals::SPI3>,
    pub spi3_sck: hal::Peri<'static, hal::peripherals::PC10>,
    pub spi3_miso: hal::Peri<'static, hal::peripherals::PC11>,
    pub spi3_mosi: hal::Peri<'static, hal::peripherals::PC12>,
}

pub struct I2cPins {
    pub i2c1: hal::Peri<'static, hal::peripherals::I2C1>,
    pub i2c1_scl: hal::Peri<'static, hal::peripherals::PB8>,
    pub i2c1_sda: hal::Peri<'static, hal::peripherals::PB9>,
}

pub struct SerialPins {
    pub usart1: hal::Peri<'static, hal::peripherals::USART1>,
    pub usart1_tx: hal::Peri<'static, hal::peripherals::PA9>,
    pub usart1_rx: hal::Peri<'static, hal::peripherals::PA10>,

    pub uart4: hal::Peri<'static, hal::peripherals::UART4>,
    pub uart4_tx: hal::Peri<'static, hal::peripherals::PA0>,
    pub uart4_rx: hal::Peri<'static, hal::peripherals::PA1>,

    pub usart6: hal::Peri<'static, hal::peripherals::USART6>,
    pub usart6_tx: hal::Peri<'static, hal::peripherals::PC6>,
    pub usart6_rx: hal::Peri<'static, hal::peripherals::PC7>,

    pub uart7: hal::Peri<'static, hal::peripherals::UART7>,
    pub uart7_tx: hal::Peri<'static, hal::peripherals::PE8>,
    pub uart7_rx: hal::Peri<'static, hal::peripherals::PE7>,

    pub uart8: hal::Peri<'static, hal::peripherals::UART8>,
    pub uart8_tx: hal::Peri<'static, hal::peripherals::PE1>,
    pub uart8_rx: hal::Peri<'static, hal::peripherals::PE0>,
}

pub struct AdcPins {
    pub vbat: hal::Peri<'static, hal::peripherals::PC3>,
    pub curr: hal::Peri<'static, hal::peripherals::PC2>,
    pub rssi: hal::Peri<'static, hal::peripherals::PC5>,
}

pub struct MotorPins {
    // Timers used by motors/servos/ledstrip:
    pub tim3: hal::Peri<'static, hal::peripherals::TIM3>,
    pub tim4: hal::Peri<'static, hal::peripherals::TIM4>,
    pub tim8: hal::Peri<'static, hal::peripherals::TIM8>,
    pub tim15: hal::Peri<'static, hal::peripherals::TIM15>, // servos
    pub tim1: hal::Peri<'static, hal::peripherals::TIM1>,   // LED_STRIP

    // Motor GPIO pins:
    pub m1: hal::Peri<'static, hal::peripherals::PB4>,
    pub m2: hal::Peri<'static, hal::peripherals::PB5>,
    pub m3: hal::Peri<'static, hal::peripherals::PB0>,
    pub m4: hal::Peri<'static, hal::peripherals::PB1>,
    pub m5: hal::Peri<'static, hal::peripherals::PD12>,
    pub m6: hal::Peri<'static, hal::peripherals::PD13>,
    pub m7: hal::Peri<'static, hal::peripherals::PC8>,
    pub m8: hal::Peri<'static, hal::peripherals::PC9>,

    // Additional PWM outs (servos):
    pub s1: hal::Peri<'static, hal::peripherals::PE5>,
    pub s2: hal::Peri<'static, hal::peripherals::PE6>,

    // LED strip pin:
    pub led_strip: hal::Peri<'static, hal::peripherals::PA8>,

    // DMA streams used by Betaflight for motor timer channels:
    pub dma1_ch0: hal::Peri<'static, hal::peripherals::DMA1_CH0>,
    pub dma1_ch1: hal::Peri<'static, hal::peripherals::DMA1_CH1>,
    pub dma1_ch2: hal::Peri<'static, hal::peripherals::DMA1_CH2>,
    pub dma1_ch3: hal::Peri<'static, hal::peripherals::DMA1_CH3>,
    pub dma1_ch4: hal::Peri<'static, hal::peripherals::DMA1_CH4>,
    pub dma1_ch5: hal::Peri<'static, hal::peripherals::DMA1_CH5>,
    pub dma1_ch6: hal::Peri<'static, hal::peripherals::DMA1_CH6>,
    pub dma1_ch7: hal::Peri<'static, hal::peripherals::DMA1_CH7>,
}

pub struct SensorPins {
    // Gyro 1: ICM42688P on SPI2, CS=PB12, DRDY=PD0 (EXTI0)
    pub gyro1_cs: Output<'static>,
    pub gyro1_drdy: exti::ExtiInput<'static>,
    pub gyro1_align: SensorAlign,

    // Mag on I2C1, align CW180
    pub mag_align: SensorAlign,
}

pub struct FlashPins {
    // Flash W25Q128FV on SPI3, CS=PA15
    pub flash_cs: Output<'static>,
}

pub struct OsdPins {
    // MAX7456 on SPI1, CS=PA4
    pub osd_cs: Output<'static>,
}

pub struct UsbPins {
    pub usb_otg_fs: hal::Peri<'static, hal::peripherals::USB_OTG_FS>,
    pub dp: hal::Peri<'static, hal::peripherals::PA12>,
    pub dm: hal::Peri<'static, hal::peripherals::PA11>,
}

bind_interrupts!(pub struct UsbIrqs {
    OTG_FS => hal::usb::InterruptHandler<hal::peripherals::USB_OTG_FS>;
});

pub struct Board {
    pub leds: Leds,
    pub beeper: Beeper<Output<'static>>,

    pub spi: SpiPins,
    pub i2c: I2cPins,
    pub serial: SerialPins,
    pub adc: AdcPins,
    pub motors: MotorPins,
    pub sensors: SensorPins,
    pub flash: FlashPins,
    pub osd: OsdPins,
    pub usb: UsbPins,
}

/// Board clock/power configuration.
///
/// PLL1: HSI (64 MHz) / M=4 * N=60 / P=2 = 480 MHz SYSCLK (VOS0).
/// VCO = 960 MHz, which is within the wide VCO range (192–960 MHz) for H743.
/// AHB at 240 MHz (SYSCLK/2), APBx at 120 MHz (AHB/2).
/// SPI123 kernel clock = PLL1_Q = 240 MHz.
/// USB uses HSI48 (independent of PLL1).
fn board_config() -> Config {
    let mut config = Config::default();
    {
        use hal::rcc::*;
        config.rcc.pll1 = Some(Pll {
            source: PllSource::HSI,
            prediv: PllPreDiv::DIV4,   // 64 / 4 = 16 MHz ref
            mul:    PllMul::MUL60,     // 16 * 60 = 960 MHz VCO
            fracn:  None,
            divp:   Some(PllDiv::DIV2), // 480 MHz SYSCLK
            divq:   Some(PllDiv::DIV4), // 240 MHz for SPI123
            divr:   None,
        });
        config.rcc.sys      = Sysclk::PLL1_P;
        config.rcc.ahb_pre  = AHBPrescaler::DIV2;  // 240 MHz AHB
        config.rcc.apb1_pre = APBPrescaler::DIV2;  // 120 MHz
        config.rcc.apb2_pre = APBPrescaler::DIV2;
        config.rcc.apb3_pre = APBPrescaler::DIV2;
        config.rcc.apb4_pre = APBPrescaler::DIV2;
        config.rcc.hsi48 = Some(Hsi48Config { sync_from_usb: true }); // 48 MHz for USB
        config.rcc.mux.usbsel = mux::Usbsel::HSI48;
        config.rcc.mux.spi123sel = mux::Saisel::PLL1_Q;
    }
    config
}

/// Initialize the HAL and return all board resources.
pub fn init() -> (Board, hal::usart::UartTx<'static, hal::mode::Blocking>) {
    let p: Peripherals = hal::init(board_config());

    // LED (PC13). Start low.
    let leds = Leds {
        led0: Output::new(p.PC13, Level::Low, Speed::Low),
    };

    // Beeper (PD2), inverted => OFF should be High.
    let beeper_pin = Output::new(
        p.PD2,
        if BEEPER_INVERTED { Level::High } else { Level::Low },
        Speed::Low,
    );
    let beeper = Beeper::new(beeper_pin, BEEPER_INVERTED);

    // Sensor CS pins: deasserted high.
    let gyro1_cs = Output::new(p.PB12, Level::High, Speed::VeryHigh);
    let flash_cs = Output::new(p.PA15, Level::High, Speed::VeryHigh);
    let osd_cs = Output::new(p.PA4, Level::High, Speed::VeryHigh);

    // Gyro DRDY pin as EXTI input.
    let gyro1_drdy = exti::ExtiInput::new(p.PD0, p.EXTI0, Pull::None, ExtiIrqs);

    // USB OTG_FS pins (PA11=DM, PA12=DP).
    let usb = UsbPins {
        usb_otg_fs: p.USB_OTG_FS,
        dp: p.PA12,
        dm: p.PA11,
    };

    let spi = SpiPins {
        spi1: p.SPI1,
        spi1_sck: p.PA5,
        spi1_miso: p.PA6,
        spi1_mosi: p.PA7,

        spi2: p.SPI2,
        spi2_sck: p.PB13,
        spi2_miso: p.PB14,
        spi2_mosi: p.PB15,
        spi2_tx_dma: p.DMA2_CH0,
        spi2_rx_dma: p.DMA2_CH1,

        spi3: p.SPI3,
        spi3_sck: p.PC10,
        spi3_miso: p.PC11,
        spi3_mosi: p.PC12,
    };

    let i2c = I2cPins {
        i2c1: p.I2C1,
        i2c1_scl: p.PB8,
        i2c1_sda: p.PB9,
    };

    let serial = SerialPins {
        usart1: p.USART1,
        usart1_tx: p.PA9,
        usart1_rx: p.PA10,

        uart4: p.UART4,
        uart4_tx: p.PA0,
        uart4_rx: p.PA1,

        usart6: p.USART6,
        usart6_tx: p.PC6,
        usart6_rx: p.PC7,

        uart7: p.UART7,
        uart7_tx: p.PE8,
        uart7_rx: p.PE7,

        uart8: p.UART8,
        uart8_tx: p.PE1,
        uart8_rx: p.PE0,
    };

    let adc = AdcPins {
        vbat: p.PC3,
        curr: p.PC2,
        rssi: p.PC5,
    };

    let motors = MotorPins {
        tim3: p.TIM3,
        tim4: p.TIM4,
        tim8: p.TIM8,
        tim15: p.TIM15,
        tim1: p.TIM1,

        m1: p.PB4,
        m2: p.PB5,
        m3: p.PB0,
        m4: p.PB1,
        m5: p.PD12,
        m6: p.PD13,
        m7: p.PC8,
        m8: p.PC9,

        s1: p.PE5,
        s2: p.PE6,

        led_strip: p.PA8,

        dma1_ch0: p.DMA1_CH0,
        dma1_ch1: p.DMA1_CH1,
        dma1_ch2: p.DMA1_CH2,
        dma1_ch3: p.DMA1_CH3,
        dma1_ch4: p.DMA1_CH4,
        dma1_ch5: p.DMA1_CH5,
        dma1_ch6: p.DMA1_CH6,
        dma1_ch7: p.DMA1_CH7,
    };

    let sensors = SensorPins {
        gyro1_cs,
        gyro1_drdy,
        gyro1_align: SensorAlign::Cw0Deg,

        mag_align: SensorAlign::Cw180Deg,
    };

    // USART3 TX (PB10) for defmt serial logging at 921600 baud.
    let mut defmt_uart_config = hal::usart::Config::default();
    defmt_uart_config.baudrate = 921_600;
    let defmt_uart = hal::usart::UartTx::new_blocking(p.USART3, p.PB10, defmt_uart_config)
        .expect("defmt UART init failed");

    (
        Board {
            leds,
            beeper,
            spi,
            i2c,
            serial,
            adc,
            motors,
            sensors,
            flash: FlashPins { flash_cs },
            osd: OsdPins { osd_cs },
            usb,
        },
        defmt_uart,
    )
}
