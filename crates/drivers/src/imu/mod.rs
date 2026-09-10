pub mod bmi270;
pub mod icm426xx;
pub mod mpu6x00;

use nalgebra::Vector3;

/// A single IMU reading in physical units, chip-native axis order.
///
/// Shared output type for all IMU drivers.
#[derive(Clone, Debug)]
pub struct ImuReading {
    /// Acceleration in m/s².
    pub accel_m_s2: Vector3<f32>,
    /// Angular rate in rad/s.
    pub gyro_rad_s: Vector3<f32>,
    /// Die temperature in degrees Celsius.
    pub temp_c: f32,
}

#[allow(async_fn_in_trait)]
/// Async IMU driver trait.
///
/// All IMU drivers implement this to plug into the common reader task.
pub trait ReadImu {
    type Error: defmt::Format;

    /// Wait for the next sample and return it.
    async fn read(&mut self) -> Result<ImuReading, Self::Error>;

    /// Attempt to recover the device after a read error.
    ///
    /// Called automatically by the reader task on error. The default
    /// implementation is a no-op; drivers with a reset pin or
    /// re-initialisation sequence should override it.
    async fn recover(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Output data rate in Hz.
    ///
    /// Must match the rate at which `read()` produces samples so that the
    /// reader task can configure its software filter correctly.
    fn sample_rate_hz(&self) -> f32;

    /// Cutoff frequency (Hz) of any onboard hardware low-pass filter applied
    /// to the signal path before the data register is updated.
    ///
    /// When `Some`, the reader task can skip or reconfigure the software
    /// Butterworth filter to avoid double-filtering. Returns `None` by
    /// default (no hardware filter, or cutoff unknown).
    fn hardware_lpf_cutoff_hz(&self) -> Option<f32> {
        None
    }
}

/// Result of probing the WHO_AM_I register on an SPI bus.
#[derive(Copy, Clone, Debug, Eq, PartialEq, defmt::Format)]
pub enum DetectedImu {
    Icm42605,
    Icm42622P,
    Icm42688P,
    Iim42652,
    Iim42653,
    Mpu6000,
    Mpu6500,
}

/// Probe an IMU on a raw SPI bus by reading WHO_AM_I (register 0x75).
///
/// This must be called **before** wrapping the bus in a `Mutex`/`SpiDevice`,
/// since drivers consume the SPI device in `new()`. Uses manual CS toggle
/// on the raw `SpiBus`.
///
/// Returns `Ok(DetectedImu)` on a recognized WHO_AM_I, or `Err(raw_id)` for
/// unknown values.
pub async fn probe_imu_raw<BUS, CS>(bus: &mut BUS, cs: &mut CS) -> Result<DetectedImu, u8>
where
    BUS: embedded_hal_async::spi::SpiBus,
    CS: embedded_hal::digital::OutputPin,
{
    let mut buf = [0x80 | 0x75, 0x00];
    let _ = cs.set_low();
    let _ = bus.transfer_in_place(&mut buf).await;
    let _ = cs.set_high();

    match buf[1] {
        0x42 => Ok(DetectedImu::Icm42605),
        0x46 => Ok(DetectedImu::Icm42622P),
        0x47 => Ok(DetectedImu::Icm42688P),
        0x6F => Ok(DetectedImu::Iim42652),
        0x56 => Ok(DetectedImu::Iim42653),
        0x68 => Ok(DetectedImu::Mpu6000),
        0x70 => Ok(DetectedImu::Mpu6500),
        other => Err(other),
    }
}
