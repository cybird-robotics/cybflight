//! Bosch BMI270 6-axis IMU driver with async DMA SPI and async DRDY.
//!
//! The BMI270 is materially different from the ICM426xx family in three ways
//! that this driver has to handle carefully:
//!
//!   1. **Microcode upload.** The part boots without a usable gyro until a
//!      328-byte configuration file is streamed into `INIT_DATA`. The upload
//!      handshake (`PWR_CONF`=0 → `INIT_CTRL`=0 → blob → `INIT_CTRL`=1 → poll
//!      `INTERNAL_STATUS`) must complete before any sampling.
//!   2. **Dummy byte on every SPI read.** A register read returns one leading
//!      dummy byte that must be discarded — real data starts at the byte after
//!      the address phase. We do this with a throwaway 1-byte `Read` op.
//!   3. **Little-endian data** (opposite of ICM big-endian), a **separate
//!      temperature register** with its own scale, and a `GYRO_RANGE` quirk
//!      (bit3 `ois_range` must be set even for 2000 dps).
//!
//! Configured for 3200 Hz gyro ODR (DRDY-driven, register read — not FIFO),
//! 1600 Hz accel ODR, ±2000 dps / ±16 g, normal-mode gyro filter (~751 Hz).
//!
//! Reference: Bosch BMI270 datasheet, ArduPilot `AP_InertialSensor_BMI270`,
//! Betaflight `accgyro_spi_bmi270`.

use embedded_hal_async::spi::{Operation, SpiDevice};
use embedded_hal_async::{delay::DelayNs, digital::Wait};

use nalgebra::Vector3;

use super::{ImuReading, ReadImu};

// ---------------------------------------------------------------------------
// Register addresses
// ---------------------------------------------------------------------------

const REG_CHIP_ID: u8 = 0x00;
const REG_INTERNAL_STATUS: u8 = 0x21;
const REG_TEMPERATURE_LSB: u8 = 0x22;
const REG_ACC_DATA_X_LSB: u8 = 0x0C; // accel(6) then gyro(6) are contiguous from here
const REG_ACC_CONF: u8 = 0x40;
const REG_ACC_RANGE: u8 = 0x41;
const REG_GYRO_CONF: u8 = 0x42;
const REG_GYRO_RANGE: u8 = 0x43;
const REG_INT1_IO_CTRL: u8 = 0x53;
const REG_INT_MAP_DATA: u8 = 0x58;
const REG_INIT_CTRL: u8 = 0x59;
const REG_INIT_DATA: u8 = 0x5E;
const REG_PWR_CONF: u8 = 0x7C;
const REG_PWR_CTRL: u8 = 0x7D;
const REG_CMD: u8 = 0x7E;

// SPI read flag (MSB of the address byte).
const SPI_READ: u8 = 0x80;

const CHIP_ID: u8 = 0x24;

// ---------------------------------------------------------------------------
// Configuration values
// ---------------------------------------------------------------------------

const CMD_SOFTRESET: u8 = 0xB6;

// ACC_CONF = filter_perf(1)<<7 | bwp(0=OSR4)<<4 | odr(0x0C = 1600 Hz)
const VAL_ACC_CONF: u8 = (1 << 7) | 0x0C;
// ACC_RANGE = 16 g
const VAL_ACC_RANGE_16G: u8 = 0x03;
// GYRO_CONF = filter_perf(1)<<7 | noise_perf(1)<<6 | bwp(2=NORM)<<4 | odr(0x0D = 3200 Hz)
const VAL_GYRO_CONF: u8 = (1 << 7) | (1 << 6) | (2 << 4) | 0x0D;
// GYRO_RANGE = 2000 dps. Bit3 (ois_range) MUST be set for 2000 dps, otherwise
// the gyro silently scales as 250 dps in prefiltered mode (undocumented Bosch
// behaviour; see Betaflight/ArduPilot comments).
const VAL_GYRO_RANGE_2000DPS: u8 = 0x08;
// INT_MAP_DATA = route data-ready to INT1
const VAL_INT_MAP_DATA_DRDY_INT1: u8 = 0x04;
// INT1_IO_CTRL = active-high, push-pull, output enabled
const VAL_INT1_IO_CTRL: u8 = 0x0A;
// PWR_CONF = disable advanced power save, enable FIFO self-wake
const VAL_PWR_CONF_NORMAL: u8 = 0x02;
// PWR_CTRL = enable gyro + accel + temperature, disable aux
const VAL_PWR_CTRL: u8 = 0x0E;

// ---------------------------------------------------------------------------
// Scale factors
// ---------------------------------------------------------------------------

// Gyro: ±2000 dps full scale → rad/s.
const GYRO_SCALE_2000DPS: f32 = 2000.0 / 32768.0 * core::f32::consts::PI / 180.0;
// Accel: ±16 g full scale → m/s².
const ACCEL_SCALE_16G: f32 = 16.0 / 32768.0 * 9.80665;

// Temperature: degC = raw * 0.002 + 23.0 (ArduPilot). 0x8000 = invalid.
const TEMP_SCALE: f32 = 0.002;
const TEMP_OFFSET: f32 = 23.0;
const TEMP_INVALID: i16 = -32768; // 0x8000

/// Refresh the (slow, ~10 ms) temperature reading once every N samples to
/// avoid an extra SPI transaction on every DRDY.
const TEMP_REFRESH_INTERVAL: u16 = 100;

/// BMI270 configuration file (microcode) — 328 bytes, vendored verbatim from
/// Bosch Sensortec BMI270-Sensor-API `bmi270_maximum_fifo.c` (BSD-3-Clause).
/// Uploaded to INIT_DATA (0x5E) during init; do not modify.
#[rustfmt::skip]
static BMI270_CONFIG_FILE: [u8; 328] = [
    0xc8, 0x2e, 0x00, 0x2e, 0x80, 0x2e, 0x1a, 0x00, 0xc8, 0x2e, 0x00, 0x2e,
    0xc8, 0x2e, 0x00, 0x2e, 0xc8, 0x2e, 0x00, 0x2e, 0xc8, 0x2e, 0x00, 0x2e,
    0xc8, 0x2e, 0x00, 0x2e, 0xc8, 0x2e, 0x00, 0x2e, 0x90, 0x32, 0x21, 0x2e,
    0x59, 0xf5, 0x10, 0x30, 0x21, 0x2e, 0x6a, 0xf5, 0x1a, 0x24, 0x22, 0x00,
    0x80, 0x2e, 0x3b, 0x00, 0xc8, 0x2e, 0x44, 0x47, 0x22, 0x00, 0x37, 0x00,
    0xa4, 0x00, 0xff, 0x0f, 0xd1, 0x00, 0x07, 0xad, 0x80, 0x2e, 0x00, 0xc1,
    0x80, 0x2e, 0x00, 0xc1, 0x80, 0x2e, 0x00, 0xc1, 0x80, 0x2e, 0x00, 0xc1,
    0x80, 0x2e, 0x00, 0xc1, 0x80, 0x2e, 0x00, 0xc1, 0x80, 0x2e, 0x00, 0xc1,
    0x80, 0x2e, 0x00, 0xc1, 0x80, 0x2e, 0x00, 0xc1, 0x80, 0x2e, 0x00, 0xc1,
    0x80, 0x2e, 0x00, 0xc1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x24,
    0xfc, 0xf5, 0x80, 0x30, 0x40, 0x42, 0x50, 0x50, 0x00, 0x30, 0x12, 0x24,
    0xeb, 0x00, 0x03, 0x30, 0x00, 0x2e, 0xc1, 0x86, 0x5a, 0x0e, 0xfb, 0x2f,
    0x21, 0x2e, 0xfc, 0xf5, 0x13, 0x24, 0x63, 0xf5, 0xe0, 0x3c, 0x48, 0x00,
    0x22, 0x30, 0xf7, 0x80, 0xc2, 0x42, 0xe1, 0x7f, 0x3a, 0x25, 0xfc, 0x86,
    0xf0, 0x7f, 0x41, 0x33, 0x98, 0x2e, 0xc2, 0xc4, 0xd6, 0x6f, 0xf1, 0x30,
    0xf1, 0x08, 0xc4, 0x6f, 0x11, 0x24, 0xff, 0x03, 0x12, 0x24, 0x00, 0xfc,
    0x61, 0x09, 0xa2, 0x08, 0x36, 0xbe, 0x2a, 0xb9, 0x13, 0x24, 0x38, 0x00,
    0x64, 0xbb, 0xd1, 0xbe, 0x94, 0x0a, 0x71, 0x08, 0xd5, 0x42, 0x21, 0xbd,
    0x91, 0xbc, 0xd2, 0x42, 0xc1, 0x42, 0x00, 0xb2, 0xfe, 0x82, 0x05, 0x2f,
    0x50, 0x30, 0x21, 0x2e, 0x21, 0xf2, 0x00, 0x2e, 0x00, 0x2e, 0xd0, 0x2e,
    0xf0, 0x6f, 0x02, 0x30, 0x02, 0x42, 0x20, 0x26, 0xe0, 0x6f, 0x02, 0x31,
    0x03, 0x40, 0x9a, 0x0a, 0x02, 0x42, 0xf0, 0x37, 0x05, 0x2e, 0x5e, 0xf7,
    0x10, 0x08, 0x12, 0x24, 0x1e, 0xf2, 0x80, 0x42, 0x83, 0x84, 0xf1, 0x7f,
    0x0a, 0x25, 0x13, 0x30, 0x83, 0x42, 0x3b, 0x82, 0xf0, 0x6f, 0x00, 0x2e,
    0x00, 0x2e, 0xd0, 0x2e, 0x12, 0x40, 0x52, 0x42, 0x00, 0x2e, 0x12, 0x40,
    0x52, 0x42, 0x3e, 0x84, 0x00, 0x40, 0x40, 0x42, 0x7e, 0x82, 0xe1, 0x7f,
    0xf2, 0x7f, 0x98, 0x2e, 0x6a, 0xd6, 0x21, 0x30, 0x23, 0x2e, 0x61, 0xf5,
    0xeb, 0x2c, 0xe1, 0x6f,
];

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Error<E> {
    Spi(E),
    UnknownChip(u8),
    /// `INTERNAL_STATUS` never reported a successful microcode init (low nibble
    /// != 1). Carries the last status byte read.
    InitFailed(u8),
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::Spi(e) => defmt::write!(f, "Spi({})", e),
            Error::UnknownChip(id) => defmt::write!(f, "UnknownChip({=u8:#x})", id),
            Error::InitFailed(s) => defmt::write!(f, "InitFailed(status={=u8:#x})", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// BMI270 driver, generic over any `embedded-hal-async` SPI device, an async
/// DRDY wait pin, and a `DelayNs` source.
///
/// The delay source is **owned** (not borrowed) so that [`ReadImu::recover`],
/// which has no delay parameter, can re-run the microcode-upload handshake on
/// its own — the BMI270 cannot be recovered without the init delays. The
/// drivers crate stays embassy-free; the concrete `DelayNs` (e.g.
/// `embassy_time::Delay`, a ZST) is supplied by the caller.
pub struct Bmi270<SPI: SpiDevice, DRDY: Wait, D: DelayNs> {
    spi: SPI,
    drdy: DRDY,
    delay: D,
    last_temp_c: f32,
    temp_counter: u16,
}

impl<SPI: SpiDevice, DRDY: Wait, D: DelayNs> Bmi270<SPI, DRDY, D> {
    /// Initialize the BMI270: enter SPI mode, soft reset, verify chip id,
    /// upload the microcode, configure ODR/range/interrupt, power on.
    pub async fn new(spi: SPI, drdy: DRDY, delay: D) -> Result<Self, Error<SPI::Error>> {
        let mut drv = Self {
            spi,
            drdy,
            delay,
            last_temp_c: 0.0,
            temp_counter: 0,
        };

        // 1. The part powers up in I2C mode; a CS rising edge latches it into
        //    SPI. Issue a throwaway CHIP_ID read (its value is invalid) to
        //    generate that edge.
        let _ = drv.read_reg(REG_CHIP_ID).await;
        drv.delay.delay_ms(1).await;

        // 2. Soft reset, then re-enter SPI mode with another dummy read.
        drv.write_reg(REG_CMD, CMD_SOFTRESET).await?;
        drv.delay.delay_ms(5).await;
        let _ = drv.read_reg(REG_CHIP_ID).await;
        drv.delay.delay_ms(1).await;

        // 3. Verify CHIP_ID with retries.
        let mut chip_id = 0u8;
        for _ in 0..5 {
            chip_id = drv.read_reg(REG_CHIP_ID).await?;
            if chip_id == CHIP_ID {
                break;
            }
            drv.delay.delay_ms(1).await;
        }
        if chip_id != CHIP_ID {
            return Err(Error::UnknownChip(chip_id));
        }

        // 4. Microcode upload handshake.
        drv.upload_config().await?;

        // 5. Configure accel + gyro.
        drv.write_reg(REG_ACC_CONF, VAL_ACC_CONF).await?;
        drv.write_reg(REG_ACC_RANGE, VAL_ACC_RANGE_16G).await?;
        drv.write_reg(REG_GYRO_CONF, VAL_GYRO_CONF).await?;
        drv.write_reg(REG_GYRO_RANGE, VAL_GYRO_RANGE_2000DPS).await?;

        // 6. Data-ready interrupt on INT1 (active-high push-pull).
        drv.write_reg(REG_INT_MAP_DATA, VAL_INT_MAP_DATA_DRDY_INT1)
            .await?;
        drv.write_reg(REG_INT1_IO_CTRL, VAL_INT1_IO_CTRL).await?;

        // 7. Power: normal power config, enable gyro + accel + temp.
        drv.write_reg(REG_PWR_CONF, VAL_PWR_CONF_NORMAL).await?;
        drv.write_reg(REG_PWR_CTRL, VAL_PWR_CTRL).await?;
        drv.delay.delay_ms(10).await;

        defmt::info!("BMI270: init OK (CHIP_ID={=u8:#x})", chip_id);

        // 8. Drain a few stale samples so the first published reading is fresh.
        for _ in 0..10 {
            drv.drdy.wait_for_rising_edge().await.ok();
            let mut buf = [0u8; 12];
            let _ = drv.read_regs(REG_ACC_DATA_X_LSB, &mut buf).await;
        }

        Ok(drv)
    }

    /// Microcode upload handshake. Strict ordering: disable power-save, begin
    /// init, stream the 328-byte blob in one burst, finish init, then poll
    /// `INTERNAL_STATUS` until the low nibble reads 1 ("init ok").
    async fn upload_config(&mut self) -> Result<(), Error<SPI::Error>> {
        // Disable advanced power save; the part needs >=450us before the next
        // step. INIT_CTRL=0 begins the load.
        self.write_reg(REG_PWR_CONF, 0x00).await?;
        self.delay.delay_ms(1).await;
        self.write_reg(REG_INIT_CTRL, 0x00).await?;

        // Stream the config file: address byte then the blob, one CS-asserted
        // transaction.
        self.spi
            .transaction(&mut [
                Operation::Write(&[REG_INIT_DATA]),
                Operation::Write(&BMI270_CONFIG_FILE),
            ])
            .await
            .map_err(Error::Spi)?;
        self.delay.delay_ms(10).await;

        // Finish init and wait for the part to parse the microcode.
        self.write_reg(REG_INIT_CTRL, 0x01).await?;
        self.delay.delay_ms(20).await;

        let mut status = 0u8;
        for _ in 0..10 {
            status = self.read_reg(REG_INTERNAL_STATUS).await?;
            if status & 0x0F == 1 {
                return Ok(());
            }
            self.delay.delay_ms(5).await;
        }
        Err(Error::InitFailed(status))
    }

    // -----------------------------------------------------------------------
    // Low-level SPI helpers (async)
    //
    // Reads discard the leading dummy byte the BMI270 emits after the address
    // phase: address `Write`, throwaway 1-byte `Read`, then the real data.
    // -----------------------------------------------------------------------

    async fn read_regs(&mut self, reg: u8, buf: &mut [u8]) -> Result<(), Error<SPI::Error>> {
        let mut dummy = [0u8; 1];
        self.spi
            .transaction(&mut [
                Operation::Write(&[reg | SPI_READ]),
                Operation::Read(&mut dummy),
                Operation::Read(buf),
            ])
            .await
            .map_err(Error::Spi)
    }

    async fn read_reg(&mut self, reg: u8) -> Result<u8, Error<SPI::Error>> {
        let mut buf = [0u8; 1];
        self.read_regs(reg, &mut buf).await?;
        Ok(buf[0])
    }

    async fn write_reg(&mut self, reg: u8, val: u8) -> Result<(), Error<SPI::Error>> {
        self.spi.write(&[reg, val]).await.map_err(Error::Spi)
    }

    /// Read the (slow) temperature register and update the cached value.
    async fn refresh_temp(&mut self) -> Result<(), Error<SPI::Error>> {
        let mut buf = [0u8; 2];
        self.read_regs(REG_TEMPERATURE_LSB, &mut buf).await?;
        let raw = i16::from_le_bytes([buf[0], buf[1]]);
        if raw != TEMP_INVALID {
            self.last_temp_c = raw as f32 * TEMP_SCALE + TEMP_OFFSET;
        }
        Ok(())
    }
}

impl<SPI: SpiDevice, DRDY: Wait, D: DelayNs> ReadImu for Bmi270<SPI, DRDY, D>
where
    SPI::Error: defmt::Format,
{
    type Error = Error<SPI::Error>;

    /// Gyro ODR = 3200 Hz (the DRDY cadence). The accel runs at 1600 Hz and is
    /// simply re-sampled on each gyro DRDY.
    fn sample_rate_hz(&self) -> f32 {
        3200.0
    }

    /// Gyro hardware filter is normal-mode at 3200 Hz ODR ⇒ ~751 Hz -3dB.
    fn hardware_lpf_cutoff_hz(&self) -> Option<f32> {
        Some(751.0)
    }

    async fn read(&mut self) -> Result<ImuReading, Self::Error> {
        // Wait for data-ready rising edge (INT1, non-latched ⇒ pulsed).
        self.drdy.wait_for_rising_edge().await.ok();

        // Burst read accel(6) + gyro(6) — contiguous from ACC_DATA_X_LSB.
        let mut buf = [0u8; 12];
        self.read_regs(REG_ACC_DATA_X_LSB, &mut buf).await?;

        // Little-endian i16 (opposite of the ICM driver).
        let raw_ax = i16::from_le_bytes([buf[0], buf[1]]);
        let raw_ay = i16::from_le_bytes([buf[2], buf[3]]);
        let raw_az = i16::from_le_bytes([buf[4], buf[5]]);
        let raw_gx = i16::from_le_bytes([buf[6], buf[7]]);
        let raw_gy = i16::from_le_bytes([buf[8], buf[9]]);
        let raw_gz = i16::from_le_bytes([buf[10], buf[11]]);

        // Temperature lives in a separate, slowly-updating register; refresh it
        // occasionally rather than on every DRDY.
        if self.temp_counter == 0 {
            let _ = self.refresh_temp().await;
            self.temp_counter = TEMP_REFRESH_INTERVAL;
        }
        self.temp_counter -= 1;

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
            temp_c: self.last_temp_c,
        })
    }

    async fn recover(&mut self) -> Result<(), Self::Error> {
        // A BMI270 recovery is a full re-init (the microcode upload cannot be
        // skipped after a reset). Re-run the reset → upload → configure → power
        // sequence with the owned delay; on failure the reader task retries.
        let _ = self.read_reg(REG_CHIP_ID).await;
        self.delay.delay_ms(1).await;
        self.write_reg(REG_CMD, CMD_SOFTRESET).await?;
        self.delay.delay_ms(5).await;
        let _ = self.read_reg(REG_CHIP_ID).await;
        self.delay.delay_ms(1).await;

        let chip_id = self.read_reg(REG_CHIP_ID).await?;
        if chip_id != CHIP_ID {
            return Err(Error::UnknownChip(chip_id));
        }

        self.upload_config().await?;
        self.write_reg(REG_ACC_CONF, VAL_ACC_CONF).await?;
        self.write_reg(REG_ACC_RANGE, VAL_ACC_RANGE_16G).await?;
        self.write_reg(REG_GYRO_CONF, VAL_GYRO_CONF).await?;
        self.write_reg(REG_GYRO_RANGE, VAL_GYRO_RANGE_2000DPS).await?;
        self.write_reg(REG_INT_MAP_DATA, VAL_INT_MAP_DATA_DRDY_INT1)
            .await?;
        self.write_reg(REG_INT1_IO_CTRL, VAL_INT1_IO_CTRL).await?;
        self.write_reg(REG_PWR_CONF, VAL_PWR_CONF_NORMAL).await?;
        self.write_reg(REG_PWR_CTRL, VAL_PWR_CTRL).await?;
        self.delay.delay_ms(10).await;
        Ok(())
    }
}
