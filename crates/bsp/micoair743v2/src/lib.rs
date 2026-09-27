#![no_std]

//! Board support for the **MicoAir743v2** (STM32H743VIT6).
//!
//! Pin mapping derived from the ArduPilot `MicoAir743v2` hwdef and the
//! Betaflight `MICOAIR743V2` target. This board carries a BMI088 + BMI270 dual
//! IMU, an SPL06 baro and a QMC5883L mag; this BSP wires the **BMI270 only**
//! (matching Betaflight's v2 target, which ignores the BMI088). The BMI088 on
//! SPI2 is left for a future dual-IMU expansion — see the SPI2 note below.
//!
//! Notable differences from the SAKURAH743 it replaces:
//!   * gyro is a **BMI270 on SPI3** (Sakura: dual ICM on SPI1/SPI4),
//!   * baro is an **SPL06 on I2C2**, mag a **QMC5883L on I2C1**,
//!   * motors M1–M4 are on **TIM1** (an advanced-control timer — board_init
//!     must enable `BDTR.MOE`),
//!   * **no CAN, no PINIO, no external dataflash**,
//!   * UART8 is the on-board **Bluetooth** module, so defmt logging moves to
//!     **UART5** (PB6).

pub use embassy_stm32 as hal;

use hal::exti;
use hal::gpio::{Level, Output, Pull, Speed};
use hal::{bind_interrupts, Config, Peripherals};

pub use bsp_types::{
    DmaHint, MotorMeta, PowerCalibration, SensorAlign, SerialPortId, TimerChannel, TimerId,
};
pub use cybflight_drivers::beeper::Beeper;

/// Board name from Betaflight target.
pub const BOARD_NAME: &str = "MICOAIR743V2";
/// Manufacturer ID from Betaflight target.
pub const MANUFACTURER_ID: &str = "MICO";

/// Betaflight target (`MICOAIR743` v1 and v2) marks the beeper inverted
/// (active-low). The ArduPilot hwdef implies active-high — confirm on the
/// bench; flip this if the buzzer is silent / stuck on.
pub const BEEPER_INVERTED: bool = true;

/// Single IMU wired: BMI270 (SPI3). The board also has a BMI088 on SPI2 that
/// this BSP does not yet wire (matches Betaflight's v2 target).
pub const IMU_COUNT: usize = 1;
/// Single baro: SPL06 on I2C2.
pub const BARO_COUNT: usize = 1;
pub const HAS_MAG: bool = true;
/// AT7456E OSD exists on SPI1 but we do not implement OSD.
pub const HAS_OSD: bool = false;
pub const HAS_FLASH: bool = false;
pub const HAS_GPS: bool = true;
pub const HAS_SDCARD: bool = true;
/// Storage backend for the blackbox / flight-data-recorder pipeline — driven
/// by the onboard microSD slot. `false` compiles out the recorder entirely.
pub const HAS_BLACKBOX_STORAGE: bool = HAS_SDCARD;
/// STM32H743 D3-domain backup SRAM is always present at the MCU level.
pub const HAS_BACKUP_SRAM: bool = true;
pub const LED_COUNT: usize = 3;

/// Primary gyro output data rate (Hz). The INDI control loop runs once per
/// primary-gyro sample, so this IS the inner-loop rate and must match the
/// BMI270 driver's `sample_rate_hz()`. The BMI270 runs at 3.2 kHz (register
/// read path; 6.4 kHz is FIFO-only). This is why the BMI270 board's control
/// loop runs at 3.2 kHz vs the 8 kHz ICM boards.
pub const PRIMARY_GYRO_ODR_HZ: f32 = 3200.0;

// =====================================================================
// PORT MAPPING TABLE — single source of truth for UART role assignments.
//
// Role        Port    Pins              Notes
// SerialRx    USART6  PC6 TX / PC7 RX   CRSF/GHST receiver (AP/BF: RCIN)
// GPS         USART3  PD8 TX / PD9 RX   u-blox / UM982
// ESP bridge  USART1  PA9 TX / PA10 RX  WiFi/companion link, DMA
// defmt log   UART5   PB6 TX            (UART8 is the on-board Bluetooth)
// =====================================================================

/// ADC calibration. Betaflight `MICOAIR743V2`: vbatscale=211, current_scale=707.
/// vbatresdivval defaults to 10 (211/10 ≈ 21.1 ≈ ArduPilot BATT_VOLT_MULT 21.12).
/// Bench-calibrate `current_scale` against a known load before relying on it.
pub const POWER_CAL: PowerCalibration = PowerCalibration {
    voltage_scale: 211,
    voltage_divider: 10,
    voltage_multiplier: 1,
    current_scale: 707,
    current_offset: 0,
};

/// Serial port role assignments — the single place to reassign a role.
pub const PORT_SERIAL_RX: SerialPortId = SerialPortId::Usart6;
pub const PORT_GPS: SerialPortId = SerialPortId::Usart3;
pub const PORT_ESP_BRIDGE: SerialPortId = SerialPortId::Usart1;

/// Sensor identities (no WHOAMI constants here).
pub mod sensors {
    /// BMI270 6-axis IMU on SPI3 (GYRO_1).
    pub const GYRO_1: &str = "BMI270";
    /// SPL06 barometer on I2C2 addr 0x77.
    pub const BARO_1: &str = "SPL06";
    pub const BARO_1_I2C_ADDR: u8 = 0x77;
    /// QMC5883L magnetometer on I2C1 addr 0x0D.
    pub const MAG: &str = "QMC5883L";
    pub const MAG_I2C_ADDR: u8 = 0x0D;
}

// EXTI interrupt required by the BMI270 DRDY pin PB7 (EXTI line 7 → grouped
// NVIC vector EXTI9_5). Safe to define here as long as only ONE BSP is
// compiled into firmware.
bind_interrupts!(pub struct ExtiIrqs {
    EXTI9_5 => exti::InterruptHandler<hal::interrupt::typelevel::EXTI9_5>;
});

/// Onboard RGB status LEDs (PE3 red / PE2 green / PE4 blue). hwdef drives them
/// active-high (idle low).
pub struct Leds {
    pub led0: Output<'static>,
    pub led1: Output<'static>,
    pub led2: Output<'static>,
}

pub struct SpiPins {
    // SPI3 (PB3 SCK / PB4 MISO / PD6 MOSI): BMI270 gyro.
    pub spi3: hal::Peri<'static, hal::peripherals::SPI3>,
    pub spi3_sck: hal::Peri<'static, hal::peripherals::PB3>,
    pub spi3_miso: hal::Peri<'static, hal::peripherals::PB4>,
    pub spi3_mosi: hal::Peri<'static, hal::peripherals::PD6>,
    pub spi3_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH0>,
    pub spi3_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH1>,
    // NOTE: SPI2 (PD3/PC2/PC3) carries the unused BMI088, SPI1 (PA5/6/7) the
    // unused AT7456E OSD — not wired here.
}

pub struct I2cPins {
    // I2C1 (PB8 SCL / PB9 SDA): QMC5883L mag + external connector.
    pub i2c1: hal::Peri<'static, hal::peripherals::I2C1>,
    pub i2c1_scl: hal::Peri<'static, hal::peripherals::PB8>,
    pub i2c1_sda: hal::Peri<'static, hal::peripherals::PB9>,
    pub i2c1_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH2>,
    pub i2c1_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH3>,

    // I2C2 (PB10 SCL / PB11 SDA): SPL06 baro (internal).
    pub i2c2: hal::Peri<'static, hal::peripherals::I2C2>,
    pub i2c2_scl: hal::Peri<'static, hal::peripherals::PB10>,
    pub i2c2_sda: hal::Peri<'static, hal::peripherals::PB11>,
    pub i2c2_tx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH4>,
    pub i2c2_rx_dma: hal::Peri<'static, hal::peripherals::DMA2_CH5>,
}

pub struct SerialPins {
    pub usart1: hal::Peri<'static, hal::peripherals::USART1>,
    pub usart1_tx: hal::Peri<'static, hal::peripherals::PA9>,
    pub usart1_rx: hal::Peri<'static, hal::peripherals::PA10>,

    // DisplayPort, which we don't implement, so the pins are free).
    pub usart2: hal::Peri<'static, hal::peripherals::USART2>,
    pub usart2_tx: hal::Peri<'static, hal::peripherals::PA2>,
    pub usart2_rx: hal::Peri<'static, hal::peripherals::PA3>,

    pub usart3: hal::Peri<'static, hal::peripherals::USART3>,
    pub usart3_tx: hal::Peri<'static, hal::peripherals::PD8>,
    pub usart3_rx: hal::Peri<'static, hal::peripherals::PD9>,

    pub usart6: hal::Peri<'static, hal::peripherals::USART6>,
    pub usart6_tx: hal::Peri<'static, hal::peripherals::PC6>,
    pub usart6_rx: hal::Peri<'static, hal::peripherals::PC7>,
}

pub struct AdcPins {
    pub adc3: hal::Peri<'static, hal::peripherals::ADC3>,
    pub vbat: hal::Peri<'static, hal::peripherals::PC0>, // ADC3_INP10
    pub curr: hal::Peri<'static, hal::peripherals::PC1>, // ADC3_INP11
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

pub struct MotorPins {
    /// Motors M1–M4 are all on TIM1 (advanced-control timer). board_init must
    /// enable `BDTR.MOE` via `regs_advanced()` or the outputs stay Hi-Z.
    pub tim1: hal::Peri<'static, hal::peripherals::TIM1>,

    pub m1: hal::Peri<'static, hal::peripherals::PE14>, // TIM1_CH4
    pub m2: hal::Peri<'static, hal::peripherals::PE13>, // TIM1_CH3
    pub m3: hal::Peri<'static, hal::peripherals::PE11>, // TIM1_CH2
    pub m4: hal::Peri<'static, hal::peripherals::PE9>,  // TIM1_CH1

    /// WS2812 LED strip on the "LED" pad PD14 = TIM4_CH3 (AF2). TIM4 is a
    /// general-purpose timer (no BDTR/MOE needed) and is otherwise unused on
    /// this board (motors are on TIM1).
    pub tim4: hal::Peri<'static, hal::peripherals::TIM4>,
    pub led_strip: hal::Peri<'static, hal::peripherals::PD14>,

    // Motor CC DMA (DMA1 CH0..CH3).
    pub dma1_ch0: hal::Peri<'static, hal::peripherals::DMA1_CH0>,
    pub dma1_ch1: hal::Peri<'static, hal::peripherals::DMA1_CH1>,
    pub dma1_ch2: hal::Peri<'static, hal::peripherals::DMA1_CH2>,
    pub dma1_ch3: hal::Peri<'static, hal::peripherals::DMA1_CH3>,
    // ESP-bridge USART1 DMA (DMA1 CH4/CH5).
    pub dma1_ch4: hal::Peri<'static, hal::peripherals::DMA1_CH4>,
    pub dma1_ch5: hal::Peri<'static, hal::peripherals::DMA1_CH5>,
    // WS2812 LED-strip CC DMA (DMA1 CH7).
    pub dma1_ch7: hal::Peri<'static, hal::peripherals::DMA1_CH7>,
}

pub struct SensorPins {
    // Gyro 1: BMI270 on SPI3, CS=PA15, DRDY=PB7 (EXTI7).
    pub gyro1_cs: Output<'static>,
    pub gyro1_drdy: exti::ExtiInput<'static>,
    pub gyro1_align: SensorAlign,

    // Mag: QMC5883L on I2C1 addr 0x0D.
    pub mag_i2c_addr: u8,
    pub mag_align: SensorAlign,
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
    pub sdio: SdioPins,
    pub motors: MotorPins,
    pub sensors: SensorPins,
    pub usb: UsbPins,

    /// Internal flash peripheral for parameter storage.
    pub internal_flash: hal::Peri<'static, hal::peripherals::FLASH>,
}

// ---- Clock declarations -------------------------------------------------
//
// These mirror `board_config()` below and exist so drivers that compute
// timings from a clock (DShot bit periods, WS2812 pulse widths, the
// DWT cycle-to-microsecond conversion) can assert against the board they
// are actually built for instead of assuming one. `verify_clocks()` in
// `crate::clocks` re-checks them against the running RCC configuration
// at boot, which is what catches an edit to `board_config` that forgets
// to move these.

/// SYSCLK, in Hz. PLL1_P from the configuration below.
pub const SYSCLK_HZ: u32 = 480_000_000;

/// Kernel clock of the APB2 timers (TIM1/8/15/16/17), in Hz.
///
/// APB2 = AHB/2 = 120 MHz, and the H7 doubles the timer clock whenever
/// the APB prescaler is not 1, so timers see 240 MHz.
pub const APB2_TIMER_HZ: u32 = 240_000_000;

/// Kernel clock of the APB1 timers (TIM2-7/12-14), in Hz. Same doubling
/// rule and the same prescaler, so it matches APB2.
pub const APB1_TIMER_HZ: u32 = 240_000_000;

/// Board clock/power configuration. Identical to SAKURAH743 — both are H743 at
/// 480 MHz SYSCLK with SPI123 on PLL1_Q (240 MHz, covers SPI3) and ADC on
/// PER_CK (HSI 64 MHz, ADC3).
///
/// PLL1: HSI (64 MHz) / M=4 * N=60 / P=2 = 480 MHz SYSCLK (VOS0).
fn board_config() -> Config {
    let mut config = Config::default();
    {
        use hal::rcc::*;
        config.rcc.pll1 = Some(Pll {
            source: PllSource::HSI,
            prediv: PllPreDiv::DIV4,
            mul: PllMul::MUL60,
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
        });
        config.rcc.mux.usbsel = mux::Usbsel::HSI48;
        config.rcc.mux.spi123sel = mux::Saisel::PLL1_Q;
        config.rcc.mux.adcsel = mux::Adcsel::PER; // PER_CK = HSI 64 MHz (ADC3)
    }
    config
}

/// Initialize the HAL and return all board resources plus the defmt UART
/// (UART5 TX = PB6 — UART8/PE1 is the on-board Bluetooth on this board).
pub fn init() -> (Board, hal::usart::UartTx<'static, hal::mode::Blocking>) {
    let p: Peripherals = hal::init(board_config());

    // LEDs (PE3/PE2/PE4). Active-high; start low (off).
    let leds = Leds {
        led0: Output::new(p.PE3, Level::Low, Speed::Low),
        led1: Output::new(p.PE2, Level::Low, Speed::Low),
        led2: Output::new(p.PE4, Level::Low, Speed::Low),
    };

    // Beeper (PD15), inverted => OFF should be High.
    let beeper_pin = Output::new(
        p.PD15,
        if BEEPER_INVERTED {
            Level::High
        } else {
            Level::Low
        },
        Speed::Low,
    );
    let beeper = Beeper::new(beeper_pin, BEEPER_INVERTED);

    // BMI270 CS (PA15), deasserted high. DRDY (PB7) as EXTI input.
    let gyro1_cs = Output::new(p.PA15, Level::High, Speed::VeryHigh);
    let gyro1_drdy = exti::ExtiInput::new(p.PB7, p.EXTI7, Pull::None, ExtiIrqs);

    let usb = UsbPins {
        usb_otg_fs: p.USB_OTG_FS,
        dp: p.PA12,
        dm: p.PA11,
    };

    let spi = SpiPins {
        spi3: p.SPI3,
        spi3_sck: p.PB3,
        spi3_miso: p.PB4,
        spi3_mosi: p.PD6,
        spi3_tx_dma: p.DMA2_CH0,
        spi3_rx_dma: p.DMA2_CH1,
    };

    let i2c = I2cPins {
        i2c1: p.I2C1,
        i2c1_scl: p.PB8,
        i2c1_sda: p.PB9,
        i2c1_tx_dma: p.DMA2_CH2,
        i2c1_rx_dma: p.DMA2_CH3,

        i2c2: p.I2C2,
        i2c2_scl: p.PB10,
        i2c2_sda: p.PB11,
        i2c2_tx_dma: p.DMA2_CH4,
        i2c2_rx_dma: p.DMA2_CH5,
    };

    let serial = SerialPins {
        usart1: p.USART1,
        usart1_tx: p.PA9,
        usart1_rx: p.PA10,

        usart2: p.USART2,
        usart2_tx: p.PA2,
        usart2_rx: p.PA3,

        usart3: p.USART3,
        usart3_tx: p.PD8,
        usart3_rx: p.PD9,

        usart6: p.USART6,
        usart6_tx: p.PC6,
        usart6_rx: p.PC7,
    };

    let adc = AdcPins {
        adc3: p.ADC3,
        vbat: p.PC0,
        curr: p.PC1,
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
        tim1: p.TIM1,
        m1: p.PE14,
        m2: p.PE13,
        m3: p.PE11,
        m4: p.PE9,
        tim4: p.TIM4,
        led_strip: p.PD14,
        dma1_ch0: p.DMA1_CH0,
        dma1_ch1: p.DMA1_CH1,
        dma1_ch2: p.DMA1_CH2,
        dma1_ch3: p.DMA1_CH3,
        dma1_ch4: p.DMA1_CH4,
        dma1_ch5: p.DMA1_CH5,
        dma1_ch7: p.DMA1_CH7,
    };

    let sensors = SensorPins {
        gyro1_cs,
        gyro1_drdy,
        // Identity. The ArduPilot hwdef lists ROTATION_ROLL_180, but that is
        // relative to AP's FRD (z-down) body frame; cybflight uses FLU (z-up,
        // see mahony.rs / position_control), and the FRD→FLU change is itself a
        // 180° roll, so it cancels AP's roll-180 → identity. Confirmed by a
        // level bench read: chip outputs (~0,~0,+g), so Cw0Deg publishes the
        // required (0,0,+g). (Betaflight's MICOAIR743V2 target also defaults to
        // CW0.) NOTE: this fixes the vertical (z) sign; verify the X/Y (roll/
        // pitch/yaw) handedness with a nose-down + roll-right tilt test.
        gyro1_align: SensorAlign::Cw0Deg,

        mag_i2c_addr: sensors::MAG_I2C_ADDR,
        // QMC5883L internal compass: hwdef ROTATION_NONE, but ArduPilot bakes a
        // universal X/Z sign negation into its QMC driver. Reproduce it with a
        // flip about Y = (-x,y,-z) = Cw0DegFlip. VERIFY ON BENCH (mag cal).
        mag_align: SensorAlign::Cw0DegFlip,
    };

    // defmt logging on UART5 TX (PB6) at 921600 baud. (UART8/PE1 is the
    // on-board Bluetooth module, so it can't double as the log UART here.)
    let mut defmt_uart_config = hal::usart::Config::default();
    defmt_uart_config.baudrate = 921_600;
    let defmt_uart = hal::usart::UartTx::new_blocking(p.UART5, p.PB6, defmt_uart_config)
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
            motors,
            sensors,
            usb,
            internal_flash: p.FLASH,
        },
        defmt_uart,
    )
}
