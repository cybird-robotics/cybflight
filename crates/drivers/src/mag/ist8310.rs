//! IST8310 3-axis magnetometer driver (I2C only).
//!
//! Single-measurement mode at ~100 Hz. Reference: Betaflight compass_ist8310.c
//! + ArduPilot AP_Compass_IST8310.cpp.

use embedded_hal_async::delay::DelayNs;
use embedded_hal_async::i2c::I2c;

use nalgebra::Vector3;

use super::{MagReading, ReadMag};

// ---------------------------------------------------------------------------
// Register addresses
// ---------------------------------------------------------------------------

const REG_WHO_AM_I: u8 = 0x00;
const REG_STAT1: u8 = 0x02;
const REG_DATA_XL: u8 = 0x03;
const REG_CNTRL1: u8 = 0x0A;
const REG_CNTRL2: u8 = 0x0B;
const REG_AVGCNTL: u8 = 0x41;
const REG_PDCNTL: u8 = 0x42;

const WHO_AM_I_VALUE: u8 = 0x10;

// Status bits
const STAT1_DRDY: u8 = 0x01;

// Scaling: 3 mGauss/LSB = 0.3 µT/LSB
const SCALE_UT: f32 = 0.3;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Error<E> {
    I2c(E),
    BadId(u8),
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::I2c(e) => defmt::write!(f, "I2c({})", e),
            Error::BadId(id) => defmt::write!(f, "BadId({:#x})", id),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub struct Ist8310<I2C> {
    i2c: I2C,
    addr: u8,
}

impl<I2C> Ist8310<I2C>
where
    I2C: I2c,
{
    /// Probe for an IST8310 on the bus. Per ArduPilot, reset before reading
    /// WHO_AM_I since the register is writable and can be corrupted by bus noise.
    pub async fn probe(i2c: &mut I2C, addr: u8, delay: &mut impl DelayNs) -> bool {
        // Soft reset
        if i2c.write(addr, &[REG_CNTRL2, 0x01]).await.is_err() {
            return false;
        }
        delay.delay_ms(50).await;

        // Read WHO_AM_I
        let mut buf = [0u8];
        if i2c
            .write_read(addr, &[REG_WHO_AM_I], &mut buf)
            .await
            .is_ok()
        {
            if buf[0] == WHO_AM_I_VALUE {
                return true;
            }
        }

        // Retry once (ArduPilot does this twice)
        if i2c.write(addr, &[REG_CNTRL2, 0x01]).await.is_err() {
            return false;
        }
        delay.delay_ms(50).await;

        if i2c
            .write_read(addr, &[REG_WHO_AM_I], &mut buf)
            .await
            .is_ok()
        {
            buf[0] == WHO_AM_I_VALUE
        } else {
            false
        }
    }

    /// Initialize the IST8310.
    pub async fn new(
        mut i2c: I2C,
        addr: u8,
        delay: &mut impl DelayNs,
    ) -> Result<Self, Error<I2C::Error>> {
        // 1. Soft reset
        i2c.write(addr, &[REG_CNTRL2, 0x01])
            .await
            .map_err(Error::I2c)?;
        delay.delay_ms(50).await;

        // 2. Read WHO_AM_I
        let mut id = [0u8];
        i2c.write_read(addr, &[REG_WHO_AM_I], &mut id)
            .await
            .map_err(Error::I2c)?;
        if id[0] != WHO_AM_I_VALUE {
            return Err(Error::BadId(id[0]));
        }

        // 3. Set averaging: 16x on all axes
        i2c.write(addr, &[REG_AVGCNTL, 0x24])
            .await
            .map_err(Error::I2c)?;

        // 4. Set pulse duration: normal (0xC0)
        i2c.write(addr, &[REG_PDCNTL, 0xC0])
            .await
            .map_err(Error::I2c)?;

        // 5. Trigger first single measurement
        i2c.write(addr, &[REG_CNTRL1, 0x01])
            .await
            .map_err(Error::I2c)?;

        Ok(Self { i2c, addr })
    }
}

impl<I2C> ReadMag for Ist8310<I2C>
where
    I2C: I2c,
    I2C::Error: defmt::Format,
{
    type Error = Error<I2C::Error>;

    async fn read(&mut self) -> Result<MagReading, Self::Error> {
        // Poll STAT1 for DRDY — yield between iterations so other tasks
        // sharing this I2C bus can acquire the mutex.
        for _ in 0..200 {
            let mut status = [0u8];
            self.i2c
                .write_read(self.addr, &[REG_STAT1], &mut status)
                .await
                .map_err(Error::I2c)?;
            if status[0] & STAT1_DRDY != 0 {
                break;
            }
            crate::yield_now().await;
        }

        // Burst read 6 data bytes: X_L, X_H, Y_L, Y_H, Z_L, Z_H
        let mut buf = [0u8; 6];
        self.i2c
            .write_read(self.addr, &[REG_DATA_XL], &mut buf)
            .await
            .map_err(Error::I2c)?;

        let raw_x = i16::from_le_bytes([buf[0], buf[1]]);
        let raw_y = i16::from_le_bytes([buf[2], buf[3]]);
        let raw_z = i16::from_le_bytes([buf[4], buf[5]]);

        // Trigger next single measurement
        self.i2c
            .write(self.addr, &[REG_CNTRL1, 0x01])
            .await
            .map_err(Error::I2c)?;

        Ok(MagReading {
            field_ut: Vector3::new(
                raw_x as f32 * SCALE_UT,
                // Negate Y-axis per Betaflight datasheet correction
                -(raw_y as f32) * SCALE_UT,
                raw_z as f32 * SCALE_UT,
            ),
            // IST8310 has no temperature sensor — report 0
            temp_c: 0.0,
        })
    }

    async fn recover(&mut self) -> Result<(), Self::Error> {
        // Soft reset — poll WHO_AM_I for readiness instead of hard delay
        self.i2c
            .write(self.addr, &[REG_CNTRL2, 0x01])
            .await
            .map_err(Error::I2c)?;

        // Poll until WHO_AM_I is readable again (each I2C transaction ~100us)
        for _ in 0..500 {
            let mut buf = [0u8];
            if self
                .i2c
                .write_read(self.addr, &[REG_WHO_AM_I], &mut buf)
                .await
                .is_ok()
            {
                if buf[0] == WHO_AM_I_VALUE {
                    break;
                }
            }
            crate::yield_now().await;
        }

        // Reconfigure
        self.i2c
            .write(self.addr, &[REG_AVGCNTL, 0x24])
            .await
            .map_err(Error::I2c)?;
        self.i2c
            .write(self.addr, &[REG_PDCNTL, 0xC0])
            .await
            .map_err(Error::I2c)?;
        self.i2c
            .write(self.addr, &[REG_CNTRL1, 0x01])
            .await
            .map_err(Error::I2c)?;

        Ok(())
    }

    fn sample_rate_hz(&self) -> f32 {
        100.0
    }
}
