pub mod qmc5883l;

pub use qmc5883l::Qmc5883l;

use nalgebra::Vector3;

/// A single magnetometer reading in physical units, sensor-native axis order.
#[derive(Clone, Debug)]
pub struct MagReading {
    /// Magnetic field in microtesla.
    pub field_ut: Vector3<f32>,
    /// Die temperature in degrees Celsius.
    pub temp_c: f32,
}

#[allow(async_fn_in_trait)]
/// Async magnetometer driver trait.
///
/// All mag drivers implement this to plug into the common reader task.
pub trait ReadMag {
    type Error: defmt::Format;

    /// Wait for the next sample and return it.
    async fn read(&mut self) -> Result<MagReading, Self::Error>;

    /// Attempt to recover the device after a read error.
    async fn recover(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Output data rate in Hz.
    fn sample_rate_hz(&self) -> f32;
}
