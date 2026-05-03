#![no_std]

pub use embassy_stm32 as hal;

use hal::exti;
use hal::gpio::{Input, Level, Output, Pull, Speed};
use hal::{bind_interrupts, Config, Peripherals};

pub use bsp_types::{
    DmaHint, MotorMeta, PowerCalibration, SensorAlign, SerialPortId, TimerChannel, TimerId,
};
pub use cybflight_drivers::beeper::Beeper;

/// Board name from Betaflight target.
pub const BOARD_NAME: &str = "SAKURAH743";
/// Manufacturer ID from Betaflight target.
pub const MANUFACTURER_ID: &str = "SSAK";

/// Betaflight target: beeper is inverted (active-low).
pub const BEEPER_INVERTED: bool = true;

pub const IMU_COUNT: usize = 2;
pub const BARO_COUNT: usize = 2;
pub const HAS_MAG: bool = true;
pub const HAS_OSD: bool = false;
pub const HAS_FLASH: bool = false;
pub const HAS_GPS: bool = true;
pub const HAS_SDCARD: bool = true;
/// True if the board exposes any storage backend usable for the
/// blackbox / flight-data-recorder pipeline. On SAKURAH743 this is
/// driven by the onboard microSD slot. Flipping this to `false`
/// compiles out the recorder pipeline entirely (no feature flag
/// required).
pub const HAS_BLACKBOX_STORAGE: bool = HAS_SDCARD;
pub const LED_COUNT: usize = 3;

// =====================================================================
// PORT MAPPING TABLE — single source of truth for UART role assignments.
// Changing a const here is the only BSP edit needed to move that role.
// `board_init/sakurah743.rs` dispatches on these values via `match`;
// LLVM elides dead arms because the discriminants are compile-time consts.
//
// Role        Port    Pins              Notes
// SerialRx    UART4   PB9 TX / PB8 RX   CRSF/GHST receiver
// GPS         USART3  PD8 TX / PD9 RX   u-blox (M8/M9/F9P)
// ESP bridge  USART1  PA9 TX / PA10 RX  WiFi/companion link, DMA
// =====================================================================

/// ADC calibration from `bf_dump.txt`: vbat_scale=170, vbat_divider=5,
/// vbat_multiplier=1, current_meter_scale=250, offset=0.
pub const POWER_CAL: PowerCalibration = PowerCalibration {
    voltage_scale: 170,
    voltage_divider: 5,
    voltage_multiplier: 1,
    current_scale: 250,
    current_offset: 0,
};

/// Serial port role assignments — the single place to reassign a role to a different UART.
/// Changing one constant here is the only edit needed to move that role to a different port.
pub const PORT_SERIAL_RX: SerialPortId = SerialPortId::Uart4;
pub const PORT_GPS: SerialPortId = SerialPortId::Usart3;
pub const PORT_ESP_BRIDGE: SerialPortId = SerialPortId::Usart1;

/// Sensor identities from Betaflight header (no WHOAMI constants here).
pub mod sensors {
    /// Betaflight: ICM42688P on SPI4 (GYRO_1).
    pub const GYRO_1: &str = "ICM42688P";
    /// Betaflight: IIM42652 on SPI1 (GYRO_2).
    pub const GYRO_2: &str = "IIM42652";
    /// Betaflight: ICP20100 barometer on I2C1 addr 0x63.
    pub const BARO_1: &str = "ICP20100";
    pub const BARO_1_I2C_ADDR: u8 = 0x63;
    /// Betaflight: DPS310 barometer on SPI1.
    pub const BARO_2: &str = "DPS310";
    /// Betaflight: IST8310 magnetometer on I2C1 addr 0x0E (14).
    pub const MAG: &str = "IST8310";
    pub const MAG_I2C_ADDR: u8 = 0x0E;
}

/// Motor metadata table in motor index order (M1..M8).
///
/// M1  PA2  TIM5 CH3 AF2  (DMA1 Stream0 Req57)
/// M2  PA3  TIM5 CH4 AF2  (DMA1 Stream1 Req58)
/// M3  PB1  TIM3 CH4 AF2  (DMA1 Stream3 Req26)
/// M4  PB0  TIM3 CH3 AF2  (DMA1 Stream2 Req25)
/// M5  PA1  TIM5 CH2 AF2  (DMA1 Stream5 Req56)
/// M6  PA0  TIM5 CH1 AF2  (DMA1 Stream4 Req55)
/// M7  PE5  TIM15 CH1 AF4 (DMA1 Stream6 Req105)
/// M8  PE6  TIM15 CH2 AF4 (DMA NONE)
pub const MOTOR_META: [MotorMeta; 8] = [
    MotorMeta {
        timer: TimerId::Tim5,
        channel: TimerChannel::Ch3,
        af: 2,
        dma: Some(DmaHint {
            stream: 0,
            request: 57,
        }),
    },
    MotorMeta {
        timer: TimerId::Tim5,
        channel: TimerChannel::Ch4,
        af: 2,
        dma: Some(DmaHint {
            stream: 1,
            request: 58,
        }),
    },
    MotorMeta {
        timer: TimerId::Tim3,
        channel: TimerChannel::Ch4,
        af: 2,
        dma: Some(DmaHint {
            stream: 3,
            request: 26,
        }),
    },
    MotorMeta {
        timer: TimerId::Tim3,
        channel: TimerChannel::Ch3,
        af: 2,
        dma: Some(DmaHint {
            stream: 2,
            request: 25,
        }),
    },
    MotorMeta {
        timer: TimerId::Tim5,
        channel: TimerChannel::Ch2,
        af: 2,
        dma: Some(DmaHint {
            stream: 5,
            request: 56,
        }),
    },
    MotorMeta {
        timer: TimerId::Tim5,
        channel: TimerChannel::Ch1,
        af: 2,
        dma: Some(DmaHint {
            stream: 4,
            request: 55,
        }),
    },
    MotorMeta {
        timer: TimerId::Tim15,
        channel: TimerChannel::Ch1,
        af: 4,
        dma: Some(DmaHint {
            stream: 6,
            request: 105,
        }),
    },
    MotorMeta {
        timer: TimerId::Tim15,
        channel: TimerChannel::Ch2,
        af: 4,
        dma: None,
    },
];

/// Servo PWM mapping metadata (from `timer` output, Betaflight target).
pub const SERVO_META: &[(/*pin*/ &str, TimerId, TimerChannel, /*af*/ u8)] = &[
    ("PD12", TimerId::Tim4, TimerChannel::Ch1, 2),
    ("PD13", TimerId::Tim4, TimerChannel::Ch2, 2),
    ("PD14", TimerId::Tim4, TimerChannel::Ch3, 2),
    ("PD15", TimerId::Tim4, TimerChannel::Ch4, 2),
];

/// LED strip pin mapping metadata.
pub const LED_STRIP_META: (
    /*pin*/ &str,
    TimerId,
    TimerChannel,
    /*af*/ u8,
    DmaHint,
) = (
    "PA8",
    TimerId::Tim1,
    TimerChannel::Ch1,
    1,
    DmaHint {
        stream: 7,
        request: 11,
    },
);

// EXTI interrupts required by gyro DRDY pins:
// - PB2 (EXTI2) for GYRO_1
// - PC4 (EXTI4) for GYRO_2
//
// This is safe to define in BSP as long as only ONE BSP is compiled into firmware.
bind_interrupts!(pub struct ExtiIrqs {
    EXTI2 => exti::InterruptHandler<hal::interrupt::typelevel::EXTI2>;
    EXTI4 => exti::InterruptHandler<hal::interrupt::typelevel::EXTI4>;
});

/// Onboard LEDs (PB3/PB4/PB5). Polarity is not specified in target header; you may need to invert in firmware if needed.
pub struct Leds {
    pub led0: Output<'static>,
    pub led1: Output<'static>,
    pub led2: Output<'static>,
}

pub struct SpiPins {
    // SPI1 (PA5/6/7): gyro2 + baro
    pub spi1: hal::Peri<'static, hal::peripherals::SPI1>,
    pub spi1_sck: hal::Peri<'static, hal::peripherals::PA5>,
    pub spi1_miso: hal::Peri<'static, hal::peripherals::PA6>,
    pub spi1_mosi: hal::Peri<'static, hal::peripherals::PA7>,
    pub spi1_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH0>,
    pub spi1_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH1>,

    // SPI2 (PB13/14/15): external pads
    pub spi2: hal::Peri<'static, hal::peripherals::SPI2>,
    pub spi2_sck: hal::Peri<'static, hal::peripherals::PB13>,
    pub spi2_miso: hal::Peri<'static, hal::peripherals::PB14>,
    pub spi2_mosi: hal::Peri<'static, hal::peripherals::PB15>,
    pub spi2_cs: Output<'static>, // PB12

    // SPI4 (PE12/13/14): gyro1
    pub spi4: hal::Peri<'static, hal::peripherals::SPI4>,
    pub spi4_sck: hal::Peri<'static, hal::peripherals::PE12>,
    pub spi4_miso: hal::Peri<'static, hal::peripherals::PE13>,
    pub spi4_mosi: hal::Peri<'static, hal::peripherals::PE14>,
    pub spi4_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH2>,
    pub spi4_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH3>,
}

pub struct I2cPins {
    pub i2c1: hal::Peri<'static, hal::peripherals::I2C1>,
    pub i2c1_scl: hal::Peri<'static, hal::peripherals::PB6>,
    pub i2c1_sda: hal::Peri<'static, hal::peripherals::PB7>,
    pub i2c1_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH4>,
    pub i2c1_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH5>,

    pub i2c2: hal::Peri<'static, hal::peripherals::I2C2>,
    pub i2c2_scl: hal::Peri<'static, hal::peripherals::PB10>,
    pub i2c2_sda: hal::Peri<'static, hal::peripherals::PB11>,
    pub i2c2_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH6>,
    pub i2c2_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH7>,
}

pub struct SerialPins {
    pub usart1: hal::Peri<'static, hal::peripherals::USART1>,
    pub usart1_tx: hal::Peri<'static, hal::peripherals::PA9>,
    pub usart1_rx: hal::Peri<'static, hal::peripherals::PA10>,

    pub usart3: hal::Peri<'static, hal::peripherals::USART3>,
    pub usart3_tx: hal::Peri<'static, hal::peripherals::PD8>,
    pub usart3_rx: hal::Peri<'static, hal::peripherals::PD9>,

    pub uart4: hal::Peri<'static, hal::peripherals::UART4>,
    pub uart4_tx: hal::Peri<'static, hal::peripherals::PB9>,
    pub uart4_rx: hal::Peri<'static, hal::peripherals::PB8>, // also RX_PPM pin in BF; mutually exclusive usage

    pub usart6: hal::Peri<'static, hal::peripherals::USART6>,
    pub usart6_tx: hal::Peri<'static, hal::peripherals::PC6>,
    pub usart6_rx: hal::Peri<'static, hal::peripherals::PC7>,

    pub uart7: hal::Peri<'static, hal::peripherals::UART7>,
    pub uart7_tx: hal::Peri<'static, hal::peripherals::PE8>,
    pub uart7_rx: hal::Peri<'static, hal::peripherals::PE7>,
    pub uart7_cts: hal::Peri<'static, hal::peripherals::PE10>,
    pub uart7_rts: hal::Peri<'static, hal::peripherals::PE9>,
}

pub struct AdcPins {
    pub adc3: hal::Peri<'static, hal::peripherals::ADC3>,
    pub vbat: hal::Peri<'static, hal::peripherals::PC3>,
    pub curr: hal::Peri<'static, hal::peripherals::PC2>,
    pub ext1: hal::Peri<'static, hal::peripherals::PC0>,
    pub rssi: hal::Peri<'static, hal::peripherals::PC1>,
}

pub struct SdioPins {
    pub sdio: hal::Peri<'static, hal::peripherals::SDMMC1>,
    pub ck: hal::Peri<'static, hal::peripherals::PC12>,
    pub cmd: hal::Peri<'static, hal::peripherals::PD2>,
    pub d0: hal::Peri<'static, hal::peripherals::PC8>,
    pub d1: hal::Peri<'static, hal::peripherals::PC9>,
    pub d2: hal::Peri<'static, hal::peripherals::PC10>,
    pub d3: hal::Peri<'static, hal::peripherals::PC11>,
}

pub struct PinIo {
    pub pinio1: Output<'static>, // PE15
    pub pinio2: Output<'static>, // PD10
    pub pinio3: Output<'static>, // PD11
}

pub struct MotorPins {
    // Timers used by motors/servos/ledstrip:
    pub tim3: hal::Peri<'static, hal::peripherals::TIM3>,
    pub tim5: hal::Peri<'static, hal::peripherals::TIM5>,
    pub tim15: hal::Peri<'static, hal::peripherals::TIM15>,
    pub tim4: hal::Peri<'static, hal::peripherals::TIM4>, // servos
    pub tim1: hal::Peri<'static, hal::peripherals::TIM1>, // LED_STRIP

    // Motor GPIO pins:
    pub m1: hal::Peri<'static, hal::peripherals::PA2>,
    pub m2: hal::Peri<'static, hal::peripherals::PA3>,
    pub m3: hal::Peri<'static, hal::peripherals::PB1>,
    pub m4: hal::Peri<'static, hal::peripherals::PB0>,
    pub m5: hal::Peri<'static, hal::peripherals::PA1>,
    pub m6: hal::Peri<'static, hal::peripherals::PA0>,
    pub m7: hal::Peri<'static, hal::peripherals::PE5>,
    pub m8: hal::Peri<'static, hal::peripherals::PE6>,

    // Additional PWM outs (servos):
    pub s1: hal::Peri<'static, hal::peripherals::PD12>,
    pub s2: hal::Peri<'static, hal::peripherals::PD13>,
    pub s3: hal::Peri<'static, hal::peripherals::PD14>,
    pub s4: hal::Peri<'static, hal::peripherals::PD15>,

    // LED strip pin:
    pub led_strip: hal::Peri<'static, hal::peripherals::PA8>,

    // DMA streams used by Betaflight for motor/ledstrip timer channels (hints for your DShot implementation):
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
    // Gyro 1: ICM42688P on SPI4, CS=PE11, DRDY=PB2 (EXTI2)
    pub gyro1_cs: Output<'static>,
    pub gyro1_drdy: exti::ExtiInput<'static>,
    pub gyro1_align: SensorAlign,

    // Gyro 2: IIM42652 on SPI1, CS=PA4, DRDY=PC4 (EXTI4)
    pub gyro2_cs: Output<'static>,
    pub gyro2_drdy: exti::ExtiInput<'static>,
    pub gyro2_align: SensorAlign,

    // Baro 2: DPS310 on SPI1, CS=PC5
    pub baro2_cs: Output<'static>,

    // Mag: IST8310 on I2C1 addr 0x0E, align CW180
    pub mag_i2c_addr: u8,
    pub mag_align: SensorAlign,
}

pub struct CanPins {
    pub can1: hal::Peri<'static, hal::peripherals::FDCAN1>,
    pub can1_rx: hal::Peri<'static, hal::peripherals::PD0>,
    pub can1_tx: hal::Peri<'static, hal::peripherals::PD1>,
    pub can1_silent: Output<'static>, // PD3, LOW = normal mode
}

pub struct Board {
    pub leds: Leds,
    pub beeper: Beeper<Output<'static>>,

    pub spi: SpiPins,
    pub i2c: I2cPins,
    pub serial: SerialPins,
    pub adc: AdcPins,
    pub sdio: SdioPins,
    pub pinio: PinIo,
    pub motors: MotorPins,
    pub sensors: SensorPins,
    pub can: CanPins,
    pub usb: UsbPins,

    /// Internal flash peripheral for parameter storage.
    pub internal_flash: hal::Peri<'static, hal::peripherals::FLASH>,

    /// USB detect input (PD4). Pull-down matches Betaflight IOCFG_IPD.
    pub usb_detect: Input<'static>,
}

pub struct UsbPins {
    pub usb_otg_fs: hal::Peri<'static, hal::peripherals::USB_OTG_FS>,
    pub dp: hal::Peri<'static, hal::peripherals::PA12>,
    pub dm: hal::Peri<'static, hal::peripherals::PA11>,
}

bind_interrupts!(pub struct UsbIrqs {
    OTG_FS => hal::usb::InterruptHandler<hal::peripherals::USB_OTG_FS>;
});

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
            prediv: PllPreDiv::DIV4, // 64 / 4 = 16 MHz ref
            mul: PllMul::MUL60,      // 16 * 60 = 960 MHz VCO
            fracn: None,
            divp: Some(PllDiv::DIV2), // 480 MHz SYSCLK
            divq: Some(PllDiv::DIV4), // 240 MHz for SPI123
            divr: None,
        });
        config.rcc.sys = Sysclk::PLL1_P;
        config.rcc.ahb_pre = AHBPrescaler::DIV2; // 240 MHz AHB
        config.rcc.apb1_pre = APBPrescaler::DIV2; // 120 MHz
        config.rcc.apb2_pre = APBPrescaler::DIV2;
        config.rcc.apb3_pre = APBPrescaler::DIV2;
        config.rcc.apb4_pre = APBPrescaler::DIV2;
        config.rcc.hsi48 = Some(Hsi48Config {
            sync_from_usb: true,
        }); // USB clock
        config.rcc.mux.usbsel = mux::Usbsel::HSI48;
        config.rcc.mux.spi123sel = mux::Saisel::PLL1_Q;
        config.rcc.mux.adcsel = mux::Adcsel::PER; // PER_CK = HSI 64 MHz (ADC3)
    }
    config
}

/// Initialize the HAL and return all board resources.
///
/// This consumes the singleton peripherals and returns a strongly-typed board mapping.
pub fn init() -> (Board, hal::usart::UartTx<'static, hal::mode::Blocking>) {
    let p: Peripherals = hal::init(board_config());

    // LEDs (PB3/4/5). Start low.
    let leds = Leds {
        led0: Output::new(p.PB3, Level::Low, Speed::Low),
        led1: Output::new(p.PB4, Level::Low, Speed::Low),
        led2: Output::new(p.PB5, Level::Low, Speed::Low),
    };

    // Beeper (PD7), inverted => OFF should be High.
    let beeper_pin = Output::new(
        p.PD7,
        if BEEPER_INVERTED {
            Level::High
        } else {
            Level::Low
        },
        Speed::Low,
    );
    let beeper = Beeper::new(beeper_pin, BEEPER_INVERTED);

    // Sensor CS pins: deasserted high.
    let gyro1_cs = Output::new(p.PE11, Level::High, Speed::VeryHigh);
    let gyro2_cs = Output::new(p.PA4, Level::High, Speed::VeryHigh);
    let baro2_cs = Output::new(p.PC5, Level::High, Speed::VeryHigh);

    // Gyro DRDY pins as EXTI inputs.
    // Pull::None is usually correct for push-pull DRDY lines. Change if your board needs it.
    let gyro1_drdy = exti::ExtiInput::new(p.PB2, p.EXTI2, Pull::None, ExtiIrqs);
    let gyro2_drdy = exti::ExtiInput::new(p.PC4, p.EXTI4, Pull::None, ExtiIrqs);

    // USB detect PD4 — pull-down matches Betaflight IOCFG_IPD.
    let usb_detect = Input::new(p.PD4, Pull::Down);

    // USB OTG_FS pins (PA11=DM, PA12=DP).
    let usb = UsbPins {
        usb_otg_fs: p.USB_OTG_FS,
        dp: p.PA12,
        dm: p.PA11,
    };

    // PINIO outputs (defaults low).
    let pinio = PinIo {
        pinio1: Output::new(p.PE15, Level::Low, Speed::Low),
        pinio2: Output::new(p.PD10, Level::Low, Speed::Low),
        pinio3: Output::new(p.PD11, Level::Low, Speed::Low),
    };

    let spi = SpiPins {
        spi1: p.SPI1,
        spi1_sck: p.PA5,
        spi1_miso: p.PA6,
        spi1_mosi: p.PA7,
        spi1_tx_dma: p.DMA2_CH0,
        spi1_rx_dma: p.DMA2_CH1,

        spi2: p.SPI2,
        spi2_sck: p.PB13,
        spi2_miso: p.PB14,
        spi2_mosi: p.PB15,
        spi2_cs: Output::new(p.PB12, Level::High, Speed::VeryHigh),

        spi4: p.SPI4,
        spi4_sck: p.PE12,
        spi4_miso: p.PE13,
        spi4_mosi: p.PE14,
        spi4_tx_dma: p.DMA2_CH2,
        spi4_rx_dma: p.DMA2_CH3,
    };

    let i2c = I2cPins {
        i2c1: p.I2C1,
        i2c1_scl: p.PB6,
        i2c1_sda: p.PB7,
        i2c1_tx_dma: p.DMA2_CH4,
        i2c1_rx_dma: p.DMA2_CH5,

        i2c2: p.I2C2,
        i2c2_scl: p.PB10,
        i2c2_sda: p.PB11,
        i2c2_tx_dma: p.DMA2_CH6,
        i2c2_rx_dma: p.DMA2_CH7,
    };

    let serial = SerialPins {
        usart1: p.USART1,
        usart1_tx: p.PA9,
        usart1_rx: p.PA10,

        usart3: p.USART3,
        usart3_tx: p.PD8,
        usart3_rx: p.PD9,

        uart4: p.UART4,
        uart4_tx: p.PB9,
        uart4_rx: p.PB8,

        usart6: p.USART6,
        usart6_tx: p.PC6,
        usart6_rx: p.PC7,

        uart7: p.UART7,
        uart7_tx: p.PE8,
        uart7_rx: p.PE7,
        uart7_cts: p.PE10,
        uart7_rts: p.PE9,
    };

    let adc = AdcPins {
        adc3: p.ADC3,
        vbat: p.PC3,
        curr: p.PC2,
        ext1: p.PC0,
        rssi: p.PC1,
    };

    let sdio = SdioPins {
        sdio: p.SDMMC1,
        ck: p.PC12,
        cmd: p.PD2,
        d0: p.PC8,
        d1: p.PC9,
        d2: p.PC10,
        d3: p.PC11,
    };

    let motors = MotorPins {
        tim3: p.TIM3,
        tim5: p.TIM5,
        tim15: p.TIM15,
        tim4: p.TIM4,
        tim1: p.TIM1,

        m1: p.PA2,
        m2: p.PA3,
        m3: p.PB1,
        m4: p.PB0,
        m5: p.PA1,
        m6: p.PA0,
        m7: p.PE5,
        m8: p.PE6,

        s1: p.PD12,
        s2: p.PD13,
        s3: p.PD14,
        s4: p.PD15,

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
        gyro1_align: SensorAlign::Cw180Deg,

        gyro2_cs,
        gyro2_drdy,
        gyro2_align: SensorAlign::Cw180Deg,

        baro2_cs,

        mag_i2c_addr: sensors::MAG_I2C_ADDR,
        mag_align: SensorAlign::Cw180Deg,
    };

    let can = CanPins {
        can1: p.FDCAN1,
        can1_rx: p.PD0,
        can1_tx: p.PD1,
        can1_silent: Output::new(p.PD3, Level::Low, Speed::Low), // LOW = normal mode
    };

    // UART8 TX (PE1) for defmt serial logging at 921600 baud.
    let mut defmt_uart_config = hal::usart::Config::default();
    defmt_uart_config.baudrate = 921_600;
    let defmt_uart = hal::usart::UartTx::new_blocking(p.UART8, p.PE1, defmt_uart_config)
        .expect("defmt UART init failed");

    (
        Board {
            leds,
            beeper,
            spi,
            i2c,
            serial,
            adc,
            sdio,
            pinio,
            motors,
            sensors,
            can,
            usb,
            internal_flash: p.FLASH,
            usb_detect,
        },
        defmt_uart,
    )
}
