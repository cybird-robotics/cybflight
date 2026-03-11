pub mod dps310;
pub mod icp20100;

pub use dps310::Dps310;
pub use icp20100::Icp20100;

/// A single barometer reading in physical units.
pub struct BaroReading {
    /// Pressure in pascals.
    pub pressure_pa: f32,
    /// Die temperature in degrees Celsius.
    pub temp_c: f32,
}

#[allow(async_fn_in_trait)]
/// Async barometer driver trait.
///
/// All baro drivers implement this to plug into the common reader task.
pub trait ReadBaro {
    type Error: defmt::Format;

    /// Wait for the next sample and return it.
    async fn read(&mut self) -> Result<BaroReading, Self::Error>;

    /// Attempt to recover the device after a read error.
    async fn recover(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Output data rate in Hz.
    fn sample_rate_hz(&self) -> f32;
}
