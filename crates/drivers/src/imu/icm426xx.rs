//! ICM426xx-family IMU driver with async DMA SPI and async DRDY.
//!
//! Supports ICM42605, ICM42622P, ICM42688P, IIM42652, and IIM42653.
//! Reads accelerometer + gyroscope data on DRDY interrupt and outputs physical units.

use embedded_hal_async::spi::SpiDevice;
use embedded_hal_async::{delay::DelayNs, digital::Wait};

use nalgebra::Vector3;

use super::{ImuReading, ReadImu};

// ---------------------------------------------------------------------------
// Register addresses
// ---------------------------------------------------------------------------

// Bank 0
const REG_DEVICE_CONFIG: u8 = 0x11;
const REG_INT_CONFIG: u8 = 0x14;
const REG_TEMP_DATA1: u8 = 0x1D;
#[allow(dead_code)]
const REG_INT_STATUS: u8 = 0x2D;
const REG_INTF_CONFIG1: u8 = 0x4D;
const REG_PWR_MGMT0: u8 = 0x4E;
const REG_GYRO_CONFIG0: u8 = 0x4F;
const REG_ACCEL_CONFIG0: u8 = 0x50;
const REG_GYRO_CONFIG1: u8 = 0x51;
const REG_GYRO_ACCEL_CONFIG0: u8 = 0x52;
const REG_ACCEL_CONFIG1: u8 = 0x53;
const REG_INT_CONFIG0: u8 = 0x63;
const REG_INT_CONFIG1: u8 = 0x64;
const REG_INT_SOURCE0: u8 = 0x65;
const REG_WHO_AM_I: u8 = 0x75;
const REG_BANK_SEL: u8 = 0x76;

// Bank 1 — gyro anti-alias filter
const REG_GYRO_AAF_DELT: u8 = 0x0C;
const REG_GYRO_AAF_DELTSQR_LO: u8 = 0x0D;
const REG_GYRO_AAF_BITSHIFT: u8 = 0x0E;

// Bank 2 — accel anti-alias filter
const REG_ACCEL_AAF_DELT: u8 = 0x03;
const REG_ACCEL_AAF_DELTSQR_LO: u8 = 0x04;
const REG_ACCEL_AAF_BITSHIFT: u8 = 0x05;

// SPI R/W bit
const SPI_READ: u8 = 0x80;

// ---------------------------------------------------------------------------
// WHO_AM_I values
// ---------------------------------------------------------------------------

const WHOAMI_ICM42605: u8 = 0x42;
const WHOAMI_ICM42622P: u8 = 0x46;
const WHOAMI_ICM42688P: u8 = 0x47;
const WHOAMI_IIM42652: u8 = 0x6F;
const WHOAMI_IIM42653: u8 = 0x56;

// ---------------------------------------------------------------------------
// Scale factors (precomputed)
// ---------------------------------------------------------------------------

// Gyro: FSR / 32768 * PI / 180
// All chips at FS_SEL=000: 2000 dps (ICM42688P/42605/42622P/IIM42652)
const GYRO_SCALE_2000DPS: f32 = 2000.0 / 32768.0 * core::f32::consts::PI / 180.0;
// IIM42653 only: FS_SEL=000 = 4000 dps
const GYRO_SCALE_4000DPS: f32 = 4000.0 / 32768.0 * core::f32::consts::PI / 180.0;

// Accel: FSR / 32768 * 9.80665
// All chips at FS_SEL=000: 16g (ICM42688P/42605/42622P/IIM42652)
const ACCEL_SCALE_16G: f32 = 16.0 / 32768.0 * 9.80665;
// IIM42653 only: FS_SEL=000 = 32g
const ACCEL_SCALE_32G: f32 = 32.0 / 32768.0 * 9.80665;

// Temperature: raw / 132.48 + 25.0
const TEMP_SCALE: f32 = 1.0 / 132.48;
const TEMP_OFFSET: f32 = 25.0;

// ---------------------------------------------------------------------------
// Anti-alias filter (AAF) configuration
//
// The AAF is a hardware low-pass filter that runs before the ADC. It prevents
// high-frequency vibration (e.g. from motors) from aliasing into the digital
// samples — once aliased, no software filter can remove it.
//
// Register values come from the LUT in datasheet DS-000347 section 5.3.
// Both chip families target the same real-world cutoffs, per ODR mode:
//
//   8 kHz ODR:  gyro ~1 kHz  — passes all flight-relevant dynamics
//   1 kHz ODR:  gyro ~258 Hz — Nyquist drops to 500 Hz, so the AAF must
//               narrow below it to keep its anti-aliasing job
//   Accel ~250 Hz in both modes — accel is used for attitude/gravity
//               correction only, and 250 Hz is already below either Nyquist
//
// The ICM-42688P / ICM-42622P family runs the AAF on a 32 MHz internal clock.
// The ICM-42605 / IIM-42652 / IIM-42653 family runs it on an 8 MHz clock.
// Cutoff scales linearly with clock rate, so the 8 MHz family needs 4× larger
// delt values to reach the same real-world frequencies:
//
//   42688P (32 MHz): delt=21 → 997 Hz gyro  | delt=6  → 258 Hz accel
//   42605  (8 MHz):  delt=63 → 995 Hz gyro  | delt=21 → 249 Hz accel
//          (table entry at 32 MHz)  (3979 Hz) |          (997 Hz)
//
// The 1 kHz-mode gyro triplets are therefore the same LUT rows as the accel
// ones (258 Hz at 32 MHz, 997 Hz-row → ~249 Hz real at 8 MHz).
// ---------------------------------------------------------------------------

struct AafConfig {
    delt: u8,
    deltsqr: u16,
    bitshift: u8,
}

// ICM42688P / ICM42622P family (datasheet section 5.3)
const AAF_GYRO_42688: AafConfig = AafConfig {
    delt: 21,
    deltsqr: 440,
    bitshift: 6,
};
// 258 Hz row — gyro AAF in 1 kHz mode, accel AAF in both modes
const AAF_258HZ_42688: AafConfig = AafConfig {
    delt: 6,
    deltsqr: 36,
    bitshift: 10,
};

// ICM42605 / IIM42652 / IIM42653 family (datasheet section 5.3)
const AAF_GYRO_42605: AafConfig = AafConfig {
    delt: 63,
    deltsqr: 3968,
    bitshift: 3,
};
// ~249 Hz real (997 Hz table row at the 8 MHz clock) — gyro AAF in 1 kHz
// mode, accel AAF in both modes
const AAF_249HZ_42605: AafConfig = AafConfig {
    delt: 21,
    deltsqr: 440,
    bitshift: 6,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Output data rate / noise-profile selection (DS-000347 §5.5, §5.6).
///
/// The UI filter block's selectable "low noise" bandwidths only take effect
/// at ODR ≤ 1 kHz — at 8 kHz the UI filter is pinned to ~ODR/4 "low latency",
/// which is why the 8 kHz mode writes the widest setting and the 1 kHz mode
/// is the one that buys the datasheet's best-noise configuration.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default, defmt::Format)]
pub enum OutputDataRate {
    /// ODR 8 kHz: UI filter wide open (`GYRO_ACCEL_CONFIG0 = 0xFF`),
    /// gyro AAF ~1 kHz. Betaflight-parity low-latency setup.
    #[default]
    Odr8kHz,
    /// ODR 1 kHz Low-Noise: UI filter BW code 1 (ODR/4 ≈ 227 Hz, 2nd order),
    /// gyro AAF narrowed to ~258 Hz (Nyquist is 500 Hz).
    Odr1kHzLowNoise,
}

impl OutputDataRate {
    /// Effective sample rate in Hz.
    pub const fn hz(self) -> f32 {
        match self {
            Self::Odr8kHz => 8000.0,
            Self::Odr1kHzLowNoise => 1000.0,
        }
    }

    /// ODR_SEL bits [3:0] of GYRO_CONFIG0 / ACCEL_CONFIG0 (FS_SEL stays 000).
    const fn odr_sel(self) -> u8 {
        match self {
            Self::Odr8kHz => 0x03,
            Self::Odr1kHzLowNoise => 0x06,
        }
    }

    /// GYRO_ACCEL_CONFIG0 (0x52): accel UI BW bits [7:4] | gyro UI BW bits [3:0].
    const fn ui_filter_bw(self) -> u8 {
        match self {
            Self::Odr8kHz => 0xFF,
            Self::Odr1kHzLowNoise => 0x11,
        }
    }
}

/// ICM426xx chip variant.
#[derive(Copy, Clone, Debug, Eq, PartialEq, defmt::Format)]
pub enum Variant {
    Icm42605,
    Icm42622P,
    Icm42688P,
    Iim42652,
    Iim42653,
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

/// ICM426xx driver with async DMA SPI and async DRDY.
///
/// Generic over any `embedded-hal-async` SPI device and an async DRDY wait pin.
pub struct Icm426xx<SPI: SpiDevice, DRDY: Wait> {
    spi: SPI,
    drdy: DRDY,
    variant: Variant,
    odr: OutputDataRate,
    gyro_scale: f32,
    accel_scale: f32,
}

impl<SPI: SpiDevice, DRDY: Wait> Icm426xx<SPI, DRDY> {
    /// Initialize the IMU. Performs soft reset, WHO_AM_I identification,
    /// anti-alias filter configuration, and powers on gyro + accel at the
    /// requested ODR in low-noise mode.
    pub async fn new(
        spi: SPI,
        drdy: DRDY,
        delay: &mut impl DelayNs,
        odr: OutputDataRate,
    ) -> Result<Self, Error<SPI::Error>> {
        let mut drv = Self {
            spi,
            drdy,
            variant: Variant::Icm42688P, // placeholder, overwritten below
            odr,
            gyro_scale: 0.0,
            accel_scale: 0.0,
        };

        // 1. Ensure bank 0
        drv.write_reg(REG_BANK_SEL, 0x00).await?;

        // 2. Soft reset
        drv.write_reg(REG_DEVICE_CONFIG, 0x01).await?;
        delay.delay_ms(1).await;

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
            WHOAMI_ICM42605 => Variant::Icm42605,
            WHOAMI_ICM42622P => Variant::Icm42622P,
            WHOAMI_ICM42688P => Variant::Icm42688P,
            WHOAMI_IIM42652 => Variant::Iim42652,
            WHOAMI_IIM42653 => Variant::Iim42653,
            _ => return Err(Error::UnknownChip(whoami)),
        };

        // FS_SEL=000 means ±2000dps/±16g for all chips except IIM42653 (±4000dps/±32g).
        // IIM42652 max range is ±2000dps/±16g despite being in the IIM family.
        let (gyro_scale, accel_scale) = match variant {
            Variant::Iim42653 => (GYRO_SCALE_4000DPS, ACCEL_SCALE_32G),
            _ => (GYRO_SCALE_2000DPS, ACCEL_SCALE_16G),
        };

        drv.variant = variant;
        drv.gyro_scale = gyro_scale;
        drv.accel_scale = accel_scale;

        defmt::info!(
            "ICM426xx: detected {:?} (WHO_AM_I={=u8:#x})",
            variant,
            whoami
        );

        // 4. Power off sensors during configuration
        drv.write_reg(REG_PWR_MGMT0, 0x00).await?;

        // 5. Configure gyro anti-alias filter (bank 1)
        let (gyro_aaf, accel_aaf) = match (variant, odr) {
            (Variant::Icm42688P | Variant::Icm42622P, OutputDataRate::Odr8kHz) => {
                (AAF_GYRO_42688, AAF_258HZ_42688)
            }
            (Variant::Icm42688P | Variant::Icm42622P, OutputDataRate::Odr1kHzLowNoise) => {
                (AAF_258HZ_42688, AAF_258HZ_42688)
            }
            (_, OutputDataRate::Odr8kHz) => (AAF_GYRO_42605, AAF_249HZ_42605),
            (_, OutputDataRate::Odr1kHzLowNoise) => (AAF_249HZ_42605, AAF_249HZ_42605),
        };

        // AAF register encoding (matching Betaflight):
        //   DELT register: raw value (gyro) or raw << 1 (accel)
        //   DELTSQR_LO: deltSqr[7:0]
        //   BITSHIFT reg: [bitshift:4][deltSqr[11:8]:4]
        drv.write_reg(REG_BANK_SEL, 0x01).await?;
        drv.write_reg(REG_GYRO_AAF_DELT, gyro_aaf.delt).await?;
        drv.write_reg(REG_GYRO_AAF_DELTSQR_LO, (gyro_aaf.deltsqr & 0xFF) as u8)
            .await?;
        drv.write_reg(
            REG_GYRO_AAF_BITSHIFT,
            (gyro_aaf.bitshift << 4) | (gyro_aaf.deltsqr >> 8) as u8,
        )
        .await?;

        // 6. Configure accel anti-alias filter (bank 2)
        // Note: accel delt register requires << 1 shift (Betaflight: aafConfig.delt << 1)
        drv.write_reg(REG_BANK_SEL, 0x02).await?;
        drv.write_reg(REG_ACCEL_AAF_DELT, accel_aaf.delt << 1)
            .await?;
        drv.write_reg(REG_ACCEL_AAF_DELTSQR_LO, (accel_aaf.deltsqr & 0xFF) as u8)
            .await?;
        drv.write_reg(
            REG_ACCEL_AAF_BITSHIFT,
            (accel_aaf.bitshift << 4) | (accel_aaf.deltsqr >> 8) as u8,
        )
        .await?;

        // 7. Back to bank 0
        drv.write_reg(REG_BANK_SEL, 0x00).await?;

        // 8. UI filter bandwidth — 8 kHz: low latency (Betaflight: accel=15<<4
        //    | gyro=15 = 0xFF); 1 kHz: BW code 1 = ODR/4 ≈ 227 Hz both sensors
        //    (a datasheet-bold "low noise" setting, only effective at ODR ≤ 1 kHz)
        drv.write_reg(REG_GYRO_ACCEL_CONFIG0, odr.ui_filter_bw())
            .await?;

        // 8b. 1 kHz mode only: pin the UI filter order to 2nd (the reset
        //     default, written explicitly). The 8 kHz path never wrote these
        //     registers and must stay bit-identical on the wire.
        if odr == OutputDataRate::Odr1kHzLowNoise {
            let gyro_cfg1 = drv.read_reg(REG_GYRO_CONFIG1).await?;
            drv.write_reg(REG_GYRO_CONFIG1, (gyro_cfg1 & !0x0C) | 0x04)
                .await?;
            let accel_cfg1 = drv.read_reg(REG_ACCEL_CONFIG1).await?;
            drv.write_reg(REG_ACCEL_CONFIG1, (accel_cfg1 & !0x18) | 0x08)
                .await?;
        }

        // 9. Interrupt config: LATCHED (bit 2), push-pull, active-high.
        //    Betaflight uses pulsed mode because its EXTI ISR starts the
        //    DMA read itself and is never late. Here the consumer is an
        //    async task on a shared executor; when it is late by more than
        //    its own read time it would miss an 8 µs pulse and lose the
        //    sample outright (measured: ~20 % loss at 8 kHz with INDI on
        //    the same executor). Latched INT1 stays high until the sensor
        //    registers are read (INT_CONFIG0 below), so a late reader
        //    picks the sample up — lateness becomes latency, not loss.
        drv.write_reg(REG_INT_CONFIG, 0x07).await?;

        // 10. UI_DRDY_INT_CLEAR = 0b11: DRDY status (and the latched INT1)
        //     clears on either an INT_STATUS read or a sensor-data read, so
        //     the per-sample path needs no separate status transaction.
        drv.write_reg(REG_INT_CONFIG0, 0x30).await?;

        // 11. Enable DRDY on INT1
        drv.write_reg(REG_INT_SOURCE0, 0x08).await?;

        // 12. INT_CONFIG1: clear async reset (bit 4), set 8us pulse (bit 6),
        //     disable tdeassert (bit 5) — all per Betaflight
        let int_config1 = drv.read_reg(REG_INT_CONFIG1).await?;
        drv.write_reg(
            REG_INT_CONFIG1,
            (int_config1 & !(1 << 4)) | (1 << 6) | (1 << 5),
        )
        .await?;

        // 13. Disable AFSR — prevents gyro stalls (mask 0xC0 = bits [7:6])
        let intf_config1 = drv.read_reg(REG_INTF_CONFIG1).await?;
        drv.write_reg(REG_INTF_CONFIG1, (intf_config1 & 0x3F) | 0x40)
            .await?;

        // 14. Power on gyro + accel in low-noise mode
        // Betaflight: "Turn on gyro and acc on again so ODR and FSR can be configured"
        drv.write_reg(REG_PWR_MGMT0, 0x0F).await?;
        delay.delay_ms(1).await;

        // 15. Gyro: ODR per mode (0x03 = 8 kHz, 0x06 = 1 kHz), FSR=max
        drv.write_reg(REG_GYRO_CONFIG0, odr.odr_sel()).await?;
        delay.delay_ms(15).await;

        // 16. Accel: ODR per mode, FSR=max
        drv.write_reg(REG_ACCEL_CONFIG0, odr.odr_sel()).await?;
        delay.delay_ms(15).await;

        // 17. Wait for gyro/accel startup (up to 200ms from standby per datasheet)
        delay.delay_ms(200).await;

        // Verify config readback
        let gyro_cfg = drv.read_reg(REG_GYRO_CONFIG0).await?;
        let accel_cfg = drv.read_reg(REG_ACCEL_CONFIG0).await?;
        let pwr = drv.read_reg(REG_PWR_MGMT0).await?;
        defmt::info!(
            "ICM426xx readback: GYRO_CFG={=u8:#x} ACCEL_CFG={=u8:#x} PWR={=u8:#x}",
            gyro_cfg,
            accel_cfg,
            pwr
        );

        // 18. Drain stale samples
        for _ in 0..10 {
            drv.drdy.wait_for_high().await.ok();
            let mut buf = [0u8; 15];
            buf[0] = SPI_READ | REG_TEMP_DATA1;
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

impl<SPI: SpiDevice, DRDY: Wait> ReadImu for Icm426xx<SPI, DRDY>
where
    SPI::Error: defmt::Format,
{
    type Error = Error<SPI::Error>;

    /// The ODR programmed into GYRO_CONFIG0 / ACCEL_CONFIG0 at init.
    fn sample_rate_hz(&self) -> f32 {
        self.odr.hz()
    }

    /// In 1 kHz Low-Noise mode the on-chip UI filter is a 2nd-order 227 Hz
    /// low-pass; at 8 kHz it is wide open (~2 kHz) and reported as `None`.
    fn hardware_lpf_cutoff_hz(&self) -> Option<f32> {
        match self.odr {
            OutputDataRate::Odr8kHz => None,
            OutputDataRate::Odr1kHzLowNoise => Some(227.0),
        }
    }

    /// Wait for DRDY, then burst-read accel + gyro + temperature.
    ///
    /// Returns an [`ImuReading`] with physical units in chip-native axis order.
    async fn read(&mut self) -> Result<ImuReading, Self::Error> {
        // Wait for data-ready: level-triggered on the latched INT1, so a
        // reader that arrives late still sees it (init step 9).
        // TODO(verify): a burst that starts late can straddle the next ODR
        // update; confirm in the ICM-42688-P datasheet that the UI data
        // registers are held coherent during a burst read (or scan a raw
        // 8 kHz log for single-sample spikes). The proper end state is the
        // FIFO (atomic packets, lossless, drained at the control rate).
        self.drdy.wait_for_high().await.ok();

        // Single transaction per sample: no INT_STATUS read — INT_CONFIG0
        // = 0x30 clears DRDY on the sensor read itself. Each async DMA
        // transaction costs tens of µs of setup + IRQ + executor wake on
        // top of the bus time, which a 125 µs period cannot afford twice.
        // Burst read: TEMP(2) + ACCEL(6) + GYRO(6) = 14 bytes
        // Wire format: [addr | 0x80] + 14 data bytes = 15 bytes total
        let mut buf = [0u8; 15];
        buf[0] = SPI_READ | REG_TEMP_DATA1;
        self.spi
            .transfer_in_place(&mut buf)
            .await
            .map_err(Error::Spi)?;

        // Parse big-endian i16 values from buf[1..]
        let raw_temp = i16::from_be_bytes([buf[1], buf[2]]);
        let raw_ax = i16::from_be_bytes([buf[3], buf[4]]);
        let raw_ay = i16::from_be_bytes([buf[5], buf[6]]);
        let raw_az = i16::from_be_bytes([buf[7], buf[8]]);
        let raw_gx = i16::from_be_bytes([buf[9], buf[10]]);
        let raw_gy = i16::from_be_bytes([buf[11], buf[12]]);
        let raw_gz = i16::from_be_bytes([buf[13], buf[14]]);

        let gs = self.gyro_scale;
        let as_ = self.accel_scale;

        Ok(ImuReading {
            accel_m_s2: Vector3::new(
                raw_ax as f32 * as_,
                raw_ay as f32 * as_,
                raw_az as f32 * as_,
            ),
            gyro_rad_s: Vector3::new(raw_gx as f32 * gs, raw_gy as f32 * gs, raw_gz as f32 * gs),
            temp_c: raw_temp as f32 * TEMP_SCALE + TEMP_OFFSET,
        })
    }
}
