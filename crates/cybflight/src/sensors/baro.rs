//! Barometer sensor task — reads samples from a baro driver and publishes
//! `BaroSample` messages to the appropriate channel.

use cybflight_drivers::baro::dps310::{Dps310, I2cBusWrapper, SpiBusWrapper};
use cybflight_drivers::baro::icp20100::Icp20100;
use cybflight_drivers::baro::ReadBaro;
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_sync::blocking_mutex::raw::{CriticalSectionRawMutex, NoopRawMutex};
use embassy_sync::pubsub::PubSubChannel;
use embassy_time::{Instant, Timer};

use crate::hal;
use crate::sensors::imu::SpiBus;
use crate::sensors::mag::I2cBus;
use cybflight_msgs as msgs;
use hal::gpio::Output;

// SPI shared-bus mutex on the baro SPI device matches the IMU bus
// (see `sensors::imu::SpiBusMtx`): `CriticalSectionRawMutex` so the
// bus can be safely shared between the IMU reader on the control
// executor (P10) and the DPS310 baro reader on the thread executor.
pub type Dps310SpiDev = Dps310<
    SpiBusWrapper<SpiDevice<'static, CriticalSectionRawMutex, SpiBus, Output<'static>>>,
>;
pub type Dps310I2cDev =
    Dps310<I2cBusWrapper<I2cDevice<'static, NoopRawMutex, I2cBus>>>;
pub type Icp20100Dev = Icp20100<I2cDevice<'static, NoopRawMutex, I2cBus>>;

pub struct BaroReader<D: ReadBaro> {
    baro: D,
}

impl<D: ReadBaro> BaroReader<D> {
    pub fn new(baro: D) -> Self {
        Self { baro }
    }

    pub async fn run(
        &mut self,
        channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::BaroSample, 2, 4, 1>,
    ) -> ! {
        let publisher = channel.immediate_publisher();
        let rate_hz = self.baro.sample_rate_hz();
        defmt::info!("Baro task running — reading at {} Hz", rate_hz);
        let period_ms = (1000.0 / rate_hz) as u64;
        loop {
            // Sleep for one sensor period — data will be ready when we wake,
            // and the bus is free for other tasks sharing it.
            Timer::after_millis(period_ms).await;
            match self.baro.read().await {
                Ok(reading) => {
                    publisher.publish_immediate(msgs::BaroSample {
                        timestamp: Instant::now(),
                        pressure_pa: reading.pressure_pa,
                        temp_c: reading.temp_c,
                    });
                }
                Err(e) => {
                    defmt::warn!("Baro read error: {}", e);
                    if let Err(re) = self.baro.recover().await {
                        defmt::error!("Baro recovery failed: {}", re);
                        Timer::after_millis(100).await;
                    }
                }
            }
        }
    }
}

#[embassy_executor::task]
pub async fn dps310_spi_baro_task(
    mut reader: BaroReader<Dps310SpiDev>,
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::BaroSample, 2, 4, 1>,
) {
    reader.run(channel).await;
}

#[embassy_executor::task]
pub async fn dps310_i2c_baro_task(
    mut reader: BaroReader<Dps310I2cDev>,
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::BaroSample, 2, 4, 1>,
) {
    reader.run(channel).await;
}

#[embassy_executor::task]
pub async fn icp20100_baro_task(
    mut reader: BaroReader<Icp20100Dev>,
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::BaroSample, 2, 4, 1>,
) {
    reader.run(channel).await;
}
