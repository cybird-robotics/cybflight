//! MPU6000/MPU6500 IMU driver with async DMA SPI and async DRDY.
//!
//! Supports MPU6000 (WHO_AM_I=0x68) and MPU6500 (WHO_AM_I=0x70).
//! Reads accelerometer + gyroscope data on DRDY interrupt and outputs physical units.

use embedded_hal_async::spi::SpiDevice;
use embedded_hal_async::{delay::DelayNs, digital::Wait};
use nalgebra::Vector3;

use super::{ImuReading, ReadImu};

// ---------------------------------------------------------------------------
// Register addresses
// ---------------------------------------------------------------------------

const REG_SMPLRT_DIV: u8 = 0x19;
const REG_CONFIG: u8 = 0x1A;
const REG_GYRO_CONFIG: u8 = 0x1B;
const REG_ACCEL_CONFIG: u8 = 0x1C;
const REG_INT_PIN_CFG: u8 = 0x37;
const REG_INT_ENABLE: u8 = 0x38;
const REG_ACCEL_XOUT_H: u8 = 0x3B;
const REG_SIGNAL_PATH_RESET: u8 = 0x68;
const REG_USER_CTRL: u8 = 0x6A;
const REG_PWR_MGMT_1: u8 = 0x6B;
const REG_PWR_MGMT_2: u8 = 0x6C;
const REG_WHO_AM_I: u8 = 0x75;

// SPI R/W bit
const SPI_READ: u8 = 0x80;

// WHO_AM_I values
const WHOAMI_MPU6000: u8 = 0x68;
const WHOAMI_MPU6500: u8 = 0x70;

// ---------------------------------------------------------------------------
// Scale factors (same FSR as ICM426xx at ±2000dps / ±16g)
// ---------------------------------------------------------------------------

// Gyro: 2000 / 32768 * PI / 180
const GYRO_SCALE_2000DPS: f32 = 2000.0 / 32768.0 * core::f32::consts::PI / 180.0;

// Accel: 16 / 32768 * 9.80665
const ACCEL_SCALE_16G: f32 = 16.0 / 32768.0 * 9.80665;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// MPU6x00 chip variant.
#[derive(Copy, Clone, Debug, Eq, PartialEq, defmt::Format)]
pub enum Variant {
    Mpu6000,
    Mpu6500,
}

/// Driver error type, wrapping the SPI bus error.
#[derive(Debug)]
pub enum Error<E> {
    Spi(E),
    UnknownChip(u8),
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::Spi(e) => defmt::write!(f, "Spi({})", e),
            Error::UnknownChip(id) => defmt::write!(f, "UnknownChip({=u8:#x})", id),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// MPU6000/6500 driver with async DMA SPI and async DRDY.
///
/// Generic over any `embedded-hal-async` SPI device and an async DRDY wait pin.
pub struct Mpu6x00<SPI: SpiDevice, DRDY: Wait> {
    spi: SPI,
    drdy: DRDY,
    variant: Variant,
    temp_scale: f32,
    temp_offset: f32,
}

impl<SPI: SpiDevice, DRDY: Wait> Mpu6x00<SPI, DRDY> {
    /// Initialize the IMU. Performs soft reset, WHO_AM_I identification,
    /// configures ±2000dps / ±16g, and enables DRDY interrupt.
    pub async fn new(
        spi: SPI,
        drdy: DRDY,
        delay: &mut impl DelayNs,
    ) -> Result<Self, Error<SPI::Error>> {
        let mut drv = Self {
            spi,
            drdy,
            variant: Variant::Mpu6000,
            temp_scale: 0.0,
            temp_offset: 0.0,
        };

        // 1. Soft reset
        drv.write_reg(REG_PWR_MGMT_1, 0x80).await?;
        delay.delay_ms(100).await;

        // 2. Signal path reset (gyro + accel + temp)
        drv.write_reg(REG_SIGNAL_PATH_RESET, 0x07).await?;
        delay.delay_ms(100).await;

        // 3. Read WHO_AM_I with retries
        let mut whoami = 0u8;
        for _ in 0..5 {
            whoami = drv.read_reg(REG_WHO_AM_I).await?;
            if whoami != 0x00 && whoami != 0xFF {
                break;
            }
            delay.delay_ms(1).await;
        }

        let variant = match whoami {
            WHOAMI_MPU6000 => Variant::Mpu6000,
            WHOAMI_MPU6500 => Variant::Mpu6500,
            _ => return Err(Error::UnknownChip(whoami)),
        };

        drv.variant = variant;

        defmt::info!(
            "MPU6x00: detected {:?} (WHO_AM_I={=u8:#x})",
            variant,
            whoami
        );

        // 4. Variant-specific configuration (matches Betaflight register order)
        match variant {
            Variant::Mpu6000 => {
                // Clock source: PLL with Z-axis gyro reference
                drv.write_reg(REG_PWR_MGMT_1, 0x03).await?;
                delay.delay_us(15).await;

                // Disable I2C interface (SPI-only mode)
                drv.write_reg(REG_USER_CTRL, 0x10).await?;
                delay.delay_us(15).await;

                // Disable all standby modes
                drv.write_reg(REG_PWR_MGMT_2, 0x00).await?;
                delay.delay_us(15).await;

                // Sample rate divider (0 = no extra division)
                drv.write_reg(REG_SMPLRT_DIV, 0x00).await?;
                delay.delay_us(15).await;

                // Gyro ±2000 dps (FS_SEL=3)
                drv.write_reg(REG_GYRO_CONFIG, 0x18).await?;
                delay.delay_us(15).await;

                // Accel ±16g (AFS_SEL=3)
                drv.write_reg(REG_ACCEL_CONFIG, 0x18).await?;
                delay.delay_us(15).await;

                // INT_ANYRD_2CLEAR, active-high, push-pull
                drv.write_reg(REG_INT_PIN_CFG, 0x10).await?;
                delay.delay_us(15).await;

                // Enable DATA_RDY interrupt
                drv.write_reg(REG_INT_ENABLE, 0x01).await?;
                delay.delay_us(15).await;

                // DLPF config — BW=256Hz (DLPF_CFG=0, 8kHz gyro sample rate)
                drv.write_reg(REG_CONFIG, 0x00).await?;
                delay.delay_us(1).await;

                drv.temp_scale = 1.0 / 340.0;
                drv.temp_offset = 36.53;
            }
            Variant::Mpu6500 => {
                // Wake from sleep before clock config
                drv.write_reg(REG_PWR_MGMT_1, 0x00).await?;
                delay.delay_ms(100).await;

                // Clock source: PLL auto-select
                drv.write_reg(REG_PWR_MGMT_1, 0x01).await?;
                delay.delay_ms(15).await;

                // Gyro ±2000 dps (FS_SEL=3)
                drv.write_reg(REG_GYRO_CONFIG, 0x18).await?;
                delay.delay_ms(15).await;

                // Accel ±16g (AFS_SEL=3)
                drv.write_reg(REG_ACCEL_CONFIG, 0x18).await?;
                delay.delay_ms(15).await;

                // DLPF config — BW=256Hz (DLPF_CFG=0, 8kHz gyro sample rate)
                drv.write_reg(REG_CONFIG, 0x00).await?;
                delay.delay_ms(15).await;

                // Sample rate divider (0 = no extra division)
                drv.write_reg(REG_SMPLRT_DIV, 0x00).await?;
                delay.delay_ms(100).await;

                // INT_ANYRD_2CLEAR, active-high, push-pull
                drv.write_reg(REG_INT_PIN_CFG, 0x10).await?;
                delay.delay_ms(15).await;

                // Enable DATA_RDY interrupt
                drv.write_reg(REG_INT_ENABLE, 0x01).await?;
                delay.delay_ms(15).await;

                // Disable I2C interface (SPI-only mode)
                drv.write_reg(REG_USER_CTRL, 0x10).await?;
                delay.delay_ms(100).await;

                // NOTE: Betaflight uses 1/340 and 36.53 for MPU6500 temperature
                // conversion. The MPU6500 datasheet specifies 1/333.87 and 21.0.
                drv.temp_scale = 1.0 / 340.0;
                drv.temp_offset = 36.53;
            }
        }

        // 5. Wait for startup
        delay.delay_ms(100).await;

        // 6. Drain stale samples
        for _ in 0..10 {
            drv.drdy.wait_for_rising_edge().await.ok();
            let mut buf = [0u8; 15];
            buf[0] = SPI_READ | REG_ACCEL_XOUT_H;
            let _ = drv.spi.transfer_in_place(&mut buf).await;
        }

        Ok(drv)
    }

    /// Returns the detected chip variant.
    pub fn variant(&self) -> Variant {
        self.variant
    }

    // -----------------------------------------------------------------------
    // Low-level SPI helpers (async)
    // -----------------------------------------------------------------------

    async fn read_reg(&mut self, reg: u8) -> Result<u8, Error<SPI::Error>> {
        let mut buf = [SPI_READ | reg, 0x00];
        self.spi
            .transfer_in_place(&mut buf)
            .await
            .map_err(Error::Spi)?;
        Ok(buf[1])
    }

    async fn write_reg(&mut self, reg: u8, val: u8) -> Result<(), Error<SPI::Error>> {
        let buf = [reg, val];
        self.spi.write(&buf).await.map_err(Error::Spi)?;
        Ok(())
    }
}

impl<SPI: SpiDevice, DRDY: Wait> ReadImu for Mpu6x00<SPI, DRDY>
where
    SPI::Error: defmt::Format,
{
    type Error = Error<SPI::Error>;

    /// REG_SMPLRT_DIV = 0x00, REG_CONFIG DLPF_CFG = 0 (disabled) → gyro rate = 8 kHz.
    fn sample_rate_hz(&self) -> f32 {
        8000.0
    }

    /// Wait for DRDY, then burst-read accel + temp + gyro.
    ///
    /// Returns an [`ImuReading`] with physical units in chip-native axis order.
    async fn read(&mut self) -> Result<ImuReading, Self::Error> {
        // Wait for data-ready rising edge
        self.drdy.wait_for_rising_edge().await.ok();

        // Burst read: ACCEL(6) + TEMP(2) + GYRO(6) = 14 bytes from 0x3B
        let mut buf = [0u8; 15];
        buf[0] = SPI_READ | REG_ACCEL_XOUT_H;
        self.spi
            .transfer_in_place(&mut buf)
            .await
            .map_err(Error::Spi)?;
        // Parse big-endian i16 values from buf[1..]
        let raw_ax = i16::from_be_bytes([buf[1], buf[2]]);
        let raw_ay = i16::from_be_bytes([buf[3], buf[4]]);
        let raw_az = i16::from_be_bytes([buf[5], buf[6]]);
        let raw_temp = i16::from_be_bytes([buf[7], buf[8]]);
        let raw_gx = i16::from_be_bytes([buf[9], buf[10]]);
        let raw_gy = i16::from_be_bytes([buf[11], buf[12]]);
        let raw_gz = i16::from_be_bytes([buf[13], buf[14]]);

        Ok(ImuReading {
            accel_m_s2: Vector3::new(
                raw_ax as f32 * ACCEL_SCALE_16G,
                raw_ay as f32 * ACCEL_SCALE_16G,
                raw_az as f32 * ACCEL_SCALE_16G,
            ),
            gyro_rad_s: Vector3::new(
                raw_gx as f32 * GYRO_SCALE_2000DPS,
                raw_gy as f32 * GYRO_SCALE_2000DPS,
                raw_gz as f32 * GYRO_SCALE_2000DPS,
            ),
            temp_c: raw_temp as f32 * self.temp_scale + self.temp_offset,
        })
    }
}
