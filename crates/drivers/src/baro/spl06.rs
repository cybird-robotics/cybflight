//! SPL06-001 barometric pressure sensor driver (Goertek).
//!
//! The SPL06 is register- and calibration-compatible with the Infineon DPS310
//! — same WHO_AM_I register (0x0D), same product-id value (0x10), same 18-byte
//! calibration block and the same compensation polynomial. This driver is a
//! close cousin of [`super::dps310`]; the differences are:
//!
//!   * no DPS310 "Infineon temperature errata" register pokes (0x0E/0x0F/0x62)
//!     — the SPL06 has no such fix and writing those registers is undefined,
//!   * no `COEF_SRCE` (0x28) register — the temperature-source bit is set
//!     directly in `TMP_CFG` (SPL06 measures temperature with its external/MEMS
//!     sensor, ArduPilot `AP_Baro_SPL06`),
//!   * a small retry loop around the WHO_AM_I probe to absorb the dual-protocol
//!     cold-start (the first transaction after power-up can be dropped while the
//!     part latches its bus mode).
//!
//! Configured for continuous measurement at 32 Hz with 16x oversampling
//! (matching our DPS310 config and its proven scale factor).
//!
//! Reference: Goertek SPL06-001 datasheet + ArduPilot `AP_Baro_SPL06`,
//! cross-checked against Betaflight's DPS310 driver (which also drives SPL06).

use embedded_hal_async::delay::DelayNs;
use embedded_hal_async::i2c::I2c;
use embedded_hal_async::spi::SpiDevice;

use super::{BaroReading, ReadBaro};

// ---------------------------------------------------------------------------
// Register addresses (identical layout to DPS310)
// ---------------------------------------------------------------------------

const REG_PSR_B2: u8 = 0x00;
const REG_PRS_CFG: u8 = 0x06;
const REG_TMP_CFG: u8 = 0x07;
const REG_MEAS_CFG: u8 = 0x08;
const REG_CFG_REG: u8 = 0x09;
const REG_RESET: u8 = 0x0C;
const REG_PRODUCT_ID: u8 = 0x0D;
const REG_COEF: u8 = 0x10;

/// SPL06-001 product id (read from `REG_PRODUCT_ID`). Numerically identical to
/// the DPS310 — the two parts are indistinguishable by id alone, which is fine
/// here because this board only carries an SPL06. (0x11 = SPA06/SPL07-003, a
/// larger-calibration variant, is rejected.)
const PRODUCT_ID: u8 = 0x10;

// MEAS_CFG status bits
const MEAS_CFG_COEF_RDY: u8 = 0x80;
const MEAS_CFG_SENSOR_RDY: u8 = 0x40;
const MEAS_CFG_PRS_RDY: u8 = 0x10;

/// `TMP_CFG` bit7: select the external (MEMS) temperature sensor. SPL06 has no
/// `COEF_SRCE` register, so unlike the DPS310 we set this bit directly
/// (ArduPilot `AP_Baro_SPL06`).
const TMP_CFG_EXT_SENSOR: u8 = 0x80;

/// PRS_CFG / TMP_CFG: 32 Hz measurement rate, 16x oversampling.
const CFG_RATE_32HZ_OSR16: u8 = 0x54;

// Scaling factor for 16x oversampling (kP = kT, same table value as DPS310).
const SCALE_FACTOR: f32 = 253952.0;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Error<E> {
    Bus(E),
    BadId(u8),
    NotReady,
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::Bus(e) => defmt::write!(f, "Bus({})", e),
            Error::BadId(id) => defmt::write!(f, "BadId({:#x})", id),
            Error::NotReady => defmt::write!(f, "NotReady"),
        }
    }
}

// ---------------------------------------------------------------------------
// Bus abstraction
// ---------------------------------------------------------------------------

#[allow(async_fn_in_trait)]
pub trait Spl06Bus {
    type Error;
    async fn read_reg(&mut self, reg: u8, buf: &mut [u8]) -> Result<(), Self::Error>;
    async fn write_reg(&mut self, reg: u8, val: u8) -> Result<(), Self::Error>;
}

pub struct SpiBusWrapper<SPI> {
    spi: SPI,
}

impl<SPI: SpiDevice> Spl06Bus for SpiBusWrapper<SPI> {
    type Error = SPI::Error;

    async fn read_reg(&mut self, reg: u8, buf: &mut [u8]) -> Result<(), Self::Error> {
        let cmd = [reg | 0x80];
        self.spi
            .transaction(&mut [
                embedded_hal_async::spi::Operation::Write(&cmd),
                embedded_hal_async::spi::Operation::Read(buf),
            ])
            .await
    }

    async fn write_reg(&mut self, reg: u8, val: u8) -> Result<(), Self::Error> {
        self.spi.write(&[reg, val]).await
    }
}

pub struct I2cBusWrapper<I2C> {
    i2c: I2C,
    addr: u8,
}

impl<I2C: I2c> Spl06Bus for I2cBusWrapper<I2C> {
    type Error = I2C::Error;

    async fn read_reg(&mut self, reg: u8, buf: &mut [u8]) -> Result<(), Self::Error> {
        self.i2c.write_read(self.addr, &[reg], buf).await
    }

    async fn write_reg(&mut self, reg: u8, val: u8) -> Result<(), Self::Error> {
        self.i2c.write(self.addr, &[reg, val]).await
    }
}

// ---------------------------------------------------------------------------
// Calibration coefficients (bit layout identical to DPS310)
// ---------------------------------------------------------------------------

struct Spl06Coefs {
    c0: f32,
    c1: f32,
    c00: f32,
    c10: f32,
    c01: f32,
    c11: f32,
    c20: f32,
    c21: f32,
    c30: f32,
}

fn parse_coefs(raw: &[u8; 18]) -> Spl06Coefs {
    // c0: 12-bit signed, raw[0] + high nibble of raw[1]
    let c0 = (((raw[0] as i16) << 4) | ((raw[1] as i16) >> 4)) as i32;
    let c0 = sign_extend(c0, 12);

    // c1: 12-bit signed, low nibble of raw[1] + raw[2]
    let c1 = ((((raw[1] & 0x0F) as i16) << 8) | (raw[2] as i16)) as i32;
    let c1 = sign_extend(c1, 12);

    // c00: 20-bit signed from raw[3..5]
    let c00 = ((raw[3] as i32) << 12) | ((raw[4] as i32) << 4) | ((raw[5] as i32) >> 4);
    let c00 = sign_extend(c00, 20);

    // c10: 20-bit signed from raw[5..7]
    let c10 = (((raw[5] & 0x0F) as i32) << 16) | ((raw[6] as i32) << 8) | (raw[7] as i32);
    let c10 = sign_extend(c10, 20);

    // c01, c11, c20, c21, c30: 16-bit signed
    let c01 = i16::from_be_bytes([raw[8], raw[9]]) as i32;
    let c11 = i16::from_be_bytes([raw[10], raw[11]]) as i32;
    let c20 = i16::from_be_bytes([raw[12], raw[13]]) as i32;
    let c21 = i16::from_be_bytes([raw[14], raw[15]]) as i32;
    let c30 = i16::from_be_bytes([raw[16], raw[17]]) as i32;

    Spl06Coefs {
        c0: c0 as f32,
        c1: c1 as f32,
        c00: c00 as f32,
        c10: c10 as f32,
        c01: c01 as f32,
        c11: c11 as f32,
        c20: c20 as f32,
        c21: c21 as f32,
        c30: c30 as f32,
    }
}

/// Sign-extend an n-bit two's complement value to i32.
fn sign_extend(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (val << shift) >> shift
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub struct Spl06<B> {
    bus: B,
    coefs: Spl06Coefs,
}

impl<SPI: SpiDevice> Spl06<SpiBusWrapper<SPI>> {
    /// Probe for an SPL06 via SPI. Returns true if WHO_AM_I matches (with a
    /// few retries to absorb the dual-protocol cold-start).
    pub async fn probe_spi(spi: &mut SPI) -> bool {
        for _ in 0..5 {
            let mut read = [0u8];
            if spi
                .transaction(&mut [
                    embedded_hal_async::spi::Operation::Write(&[REG_PRODUCT_ID | 0x80]),
                    embedded_hal_async::spi::Operation::Read(&mut read),
                ])
                .await
                .is_ok()
                && read[0] == PRODUCT_ID
            {
                return true;
            }
        }
        false
    }

    /// Create a new SPL06 over SPI.
    pub async fn new_spi(
        spi: SPI,
        delay: &mut impl DelayNs,
    ) -> Result<Spl06<SpiBusWrapper<SPI>>, Error<SPI::Error>> {
        let bus = SpiBusWrapper { spi };
        Spl06::init(bus, delay).await
    }
}

impl<I2C: I2c> Spl06<I2cBusWrapper<I2C>> {
    /// Probe for an SPL06 via I2C. Returns true if WHO_AM_I matches (with a
    /// few retries to absorb the dual-protocol cold-start).
    pub async fn probe_i2c(i2c: &mut I2C, addr: u8) -> bool {
        for _ in 0..5 {
            let mut buf = [0u8];
            if i2c.write_read(addr, &[REG_PRODUCT_ID], &mut buf).await.is_ok()
                && buf[0] == PRODUCT_ID
            {
                return true;
            }
        }
        false
    }

    /// Create a new SPL06 over I2C.
    pub async fn new_i2c(
        i2c: I2C,
        addr: u8,
        delay: &mut impl DelayNs,
    ) -> Result<Spl06<I2cBusWrapper<I2C>>, Error<I2C::Error>> {
        let bus = I2cBusWrapper { i2c, addr };
        Spl06::init(bus, delay).await
    }
}

impl<B: Spl06Bus> Spl06<B> {
    async fn init(mut bus: B, delay: &mut impl DelayNs) -> Result<Self, Error<B::Error>> {
        // 1. Read WHO_AM_I (retry for the dual-protocol cold-start).
        let mut id = [0u8];
        for _ in 0..5 {
            bus.read_reg(REG_PRODUCT_ID, &mut id)
                .await
                .map_err(Error::Bus)?;
            if id[0] == PRODUCT_ID {
                break;
            }
            delay.delay_ms(2).await;
        }
        if id[0] != PRODUCT_ID {
            return Err(Error::BadId(id[0]));
        }

        // 2. Soft reset
        bus.write_reg(REG_RESET, 0x09).await.map_err(Error::Bus)?;
        delay.delay_ms(40).await;

        // 3. Wait for COEF_RDY + SENSOR_RDY
        for _ in 0..10 {
            let mut status = [0u8];
            bus.read_reg(REG_MEAS_CFG, &mut status)
                .await
                .map_err(Error::Bus)?;
            if status[0] & (MEAS_CFG_COEF_RDY | MEAS_CFG_SENSOR_RDY)
                == (MEAS_CFG_COEF_RDY | MEAS_CFG_SENSOR_RDY)
            {
                break;
            }
            delay.delay_ms(10).await;
        }

        // 4. Read calibration coefficients
        let mut coef_raw = [0u8; 18];
        bus.read_reg(REG_COEF, &mut coef_raw)
            .await
            .map_err(Error::Bus)?;
        let coefs = parse_coefs(&coef_raw);

        // 5. Configure for continuous P+T at 32 Hz, 16x OSR.
        Self::configure(&mut bus).await?;
        delay.delay_ms(50).await;

        Ok(Self { bus, coefs })
    }

    /// Write the measurement configuration (shared by `init` and `recover`).
    async fn configure(bus: &mut B) -> Result<(), Error<B::Error>> {
        // PRS_CFG: 32 Hz rate, 16x oversampling.
        bus.write_reg(REG_PRS_CFG, CFG_RATE_32HZ_OSR16)
            .await
            .map_err(Error::Bus)?;
        // TMP_CFG: 32 Hz rate, 16x OSR, external temperature sensor.
        bus.write_reg(REG_TMP_CFG, CFG_RATE_32HZ_OSR16 | TMP_CFG_EXT_SENSOR)
            .await
            .map_err(Error::Bus)?;
        // CFG_REG: P/T bit-shift enabled (required for >8x OSR).
        bus.write_reg(REG_CFG_REG, 0x0C).await.map_err(Error::Bus)?;
        // MEAS_CFG: continuous P+T measurement.
        bus.write_reg(REG_MEAS_CFG, 0x07).await.map_err(Error::Bus)?;
        Ok(())
    }
}

impl<B: Spl06Bus> ReadBaro for Spl06<B>
where
    B::Error: defmt::Format,
{
    type Error = Error<B::Error>;

    async fn read(&mut self) -> Result<BaroReading, Self::Error> {
        // Poll for PRS_RDY
        for _ in 0..50 {
            let mut status = [0u8];
            self.bus
                .read_reg(REG_MEAS_CFG, &mut status)
                .await
                .map_err(Error::Bus)?;
            if status[0] & MEAS_CFG_PRS_RDY != 0 {
                break;
            }
        }

        // Burst read 6 bytes: PSR_B2, PSR_B1, PSR_B0, TMP_B2, TMP_B1, TMP_B0
        let mut buf = [0u8; 6];
        self.bus
            .read_reg(REG_PSR_B2, &mut buf)
            .await
            .map_err(Error::Bus)?;

        // Parse 24-bit two's complement
        let raw_psr = ((buf[0] as i32) << 16) | ((buf[1] as i32) << 8) | (buf[2] as i32);
        let raw_psr = sign_extend(raw_psr, 24);
        let raw_tmp = ((buf[3] as i32) << 16) | ((buf[4] as i32) << 8) | (buf[5] as i32);
        let raw_tmp = sign_extend(raw_tmp, 24);

        // Scale
        let p_sc = raw_psr as f32 / SCALE_FACTOR;
        let t_sc = raw_tmp as f32 / SCALE_FACTOR;

        // Compensated pressure (Pa)
        let c = &self.coefs;
        let pressure_pa = c.c00
            + p_sc * (c.c10 + p_sc * (c.c20 + p_sc * c.c30))
            + t_sc * c.c01
            + t_sc * p_sc * (c.c11 + p_sc * c.c21);

        // Compensated temperature (°C)
        let temp_c = c.c0 * 0.5 + c.c1 * t_sc;

        Ok(BaroReading {
            pressure_pa,
            temp_c,
        })
    }

    async fn recover(&mut self) -> Result<(), Self::Error> {
        // Soft reset
        self.bus
            .write_reg(REG_RESET, 0x09)
            .await
            .map_err(Error::Bus)?;

        // Poll for COEF_RDY + SENSOR_RDY (each I2C/SPI transaction ~100us)
        for _ in 0..400 {
            let mut status = [0u8];
            self.bus
                .read_reg(REG_MEAS_CFG, &mut status)
                .await
                .map_err(Error::Bus)?;
            if status[0] & (MEAS_CFG_COEF_RDY | MEAS_CFG_SENSOR_RDY)
                == (MEAS_CFG_COEF_RDY | MEAS_CFG_SENSOR_RDY)
            {
                break;
            }
        }

        // Re-read coefficients
        let mut coef_raw = [0u8; 18];
        self.bus
            .read_reg(REG_COEF, &mut coef_raw)
            .await
            .map_err(Error::Bus)?;
        self.coefs = parse_coefs(&coef_raw);

        // Reconfigure
        Self::configure(&mut self.bus).await?;

        Ok(())
    }

    fn sample_rate_hz(&self) -> f32 {
        32.0
    }
}
