//! QMC5883L 3-axis magnetometer driver.
//!
//! Generic over `embedded_hal_async::i2c::I2c`. Configures for continuous
//! mode, 200 Hz ODR, 8 Gauss range, 512x oversampling.

use embedded_hal_async::i2c::I2c;

use nalgebra::Vector3;

use super::{MagReading, ReadMag};

// ---------------------------------------------------------------------------
// Register addresses
// ---------------------------------------------------------------------------

const QMC5883L_ADDR: u8 = 0x0D;

const REG_DATA_X_LSB: u8 = 0x00;
const REG_STATUS: u8 = 0x06;
const REG_CONTROL1: u8 = 0x09;
const REG_SET_RESET: u8 = 0x0A;
const REG_SET_RESET_PERIOD: u8 = 0x0B;

// Status register bits
const STATUS_DRDY: u8 = 0x01;

// Control1 register values
// Mode: continuous (bits 1:0 = 01)
// ODR: 200 Hz (bits 3:2 = 11)
// Range: 8G (bit 5:4 = 01)
// OSR: 512 (bits 7:6 = 00)
const CTRL1_CONTINUOUS_200HZ_8G_OSR512: u8 = 0x1D;

// Scaling: 8G range = 3000 LSB/Gauss.
// 1 Gauss = 100 uT → 3000 LSB/Gauss = 30 LSB/uT.
// So factor = 1.0 / 30.0 to get microtesla.
const SCALE_UT: f32 = 1.0 / 30.0;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Error<E> {
    I2c(E),
    NotFound,
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::I2c(e) => defmt::write!(f, "I2c({})", e),
            Error::NotFound => defmt::write!(f, "NotFound"),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub struct Qmc5883l<I2C> {
    i2c: I2C,
}

impl<I2C> Qmc5883l<I2C>
where
    I2C: I2c,
{
    /// Probe for a QMC5883L on the bus. Returns true if found.
    pub async fn probe(i2c: &mut I2C) -> bool {
        let mut buf = [0u8; 1];
        defmt::debug!("QMC5883L: probing addr {:#04x}", QMC5883L_ADDR);
        match i2c.write_read(QMC5883L_ADDR, &[REG_STATUS], &mut buf).await {
            Ok(()) => {
                defmt::debug!("QMC5883L: probe OK, status={:#04x}", buf[0]);
                true
            }
            Err(_) => {
                defmt::debug!("QMC5883L: probe failed (no ACK)");
                false
            }
        }
    }

    /// Initialize the QMC5883L.
    pub async fn new(
        mut i2c: I2C,
        delay: &mut impl embedded_hal_async::delay::DelayNs,
    ) -> Result<Self, Error<I2C::Error>> {
        // Soft reset
        i2c.write(QMC5883L_ADDR, &[REG_SET_RESET, 0x80])
            .await
            .map_err(Error::I2c)?;
        delay.delay_ms(20).await;

        // Set/Reset period register (recommended value)
        i2c.write(QMC5883L_ADDR, &[REG_SET_RESET_PERIOD, 0x01])
            .await
            .map_err(Error::I2c)?;

        // Configure: continuous, 200 Hz, 8G, 512x OSR
        i2c.write(
            QMC5883L_ADDR,
            &[REG_CONTROL1, CTRL1_CONTINUOUS_200HZ_8G_OSR512],
        )
        .await
        .map_err(Error::I2c)?;

        // Enable pointer rollover
        i2c.write(QMC5883L_ADDR, &[REG_SET_RESET, 0x40])
            .await
            .map_err(Error::I2c)?;

        delay.delay_ms(10).await;

        Ok(Self { i2c })
    }
}

impl<I2C> ReadMag for Qmc5883l<I2C>
where
    I2C: I2c,
    I2C::Error: defmt::Format,
{
    type Error = Error<I2C::Error>;

    async fn read(&mut self) -> Result<MagReading, Self::Error> {
        // Poll for DRDY (the I2C transaction itself provides ~100us delay)
        for _ in 0..100 {
            let mut status = [0u8; 1];
            self.i2c
                .write_read(QMC5883L_ADDR, &[REG_STATUS], &mut status)
                .await
                .map_err(Error::I2c)?;
            if status[0] & STATUS_DRDY != 0 {
                break;
            }
        }

        // Burst read 6 data bytes + 2 temp bytes = 8 bytes from register 0x00
        let mut buf = [0u8; 8];
        self.i2c
            .write_read(QMC5883L_ADDR, &[REG_DATA_X_LSB], &mut buf)
            .await
            .map_err(Error::I2c)?;

        let raw_x = i16::from_le_bytes([buf[0], buf[1]]);
        let raw_y = i16::from_le_bytes([buf[2], buf[3]]);
        let raw_z = i16::from_le_bytes([buf[4], buf[5]]);
        let raw_temp = i16::from_le_bytes([buf[6], buf[7]]);

        Ok(MagReading {
            field_ut: Vector3::new(
                raw_x as f32 * SCALE_UT,
                raw_y as f32 * SCALE_UT,
                raw_z as f32 * SCALE_UT,
            ),
            // QMC5883L temp sensor is uncalibrated; ~100 LSB/°C with
            // chip-specific zero offset.  Adding 20 °C gives a rough
            // room-temperature baseline.
            temp_c: raw_temp as f32 / 100.0 + 20.0,
        })
    }

    async fn recover(&mut self) -> Result<(), Self::Error> {
        // Re-initialize control register
        self.i2c
            .write(
                QMC5883L_ADDR,
                &[REG_CONTROL1, CTRL1_CONTINUOUS_200HZ_8G_OSR512],
            )
            .await
            .map_err(Error::I2c)?;
        Ok(())
    }

    fn sample_rate_hz(&self) -> f32 {
        200.0
    }
}
