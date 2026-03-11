//! External magnetometer sensor task — reads samples from a QMC5883L and
//! publishes `MagSample` messages to the `MAG_EXT` channel.

use bsp_types::SensorAlign;
use cybflight_drivers::mag::qmc5883l::Qmc5883l;
use cybflight_drivers::mag::ReadMag;
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Instant, Timer};

use crate::apply_alignment;
use crate::hal;
use cybflight_msgs as msgs;

pub type I2cBus = hal::i2c::I2c<'static, hal::mode::Async, hal::i2c::Master>;
pub type I2cBusMtx = Mutex<NoopRawMutex, I2cBus>;
pub type Qmc5883lDev = Qmc5883l<I2cDevice<'static, NoopRawMutex, I2cBus>>;

pub struct MagReader<D: ReadMag> {
    mag: D,
    align: SensorAlign,
}

impl<D: ReadMag> MagReader<D> {
    pub fn new(mag: D, align: SensorAlign) -> Self {
        Self { mag, align }
    }

    pub async fn run(&mut self) -> ! {
        let publisher = super::MAG_EXT.immediate_publisher();
        defmt::info!("Mag task running — reading at {} Hz", self.mag.sample_rate_hz());
        loop {
            match self.mag.read().await {
                Ok(reading) => {
                    let field = apply_alignment(self.align, reading.field_ut);
                    publisher.publish_immediate(msgs::MagSample {
                        timestamp: Instant::now(),
                        field_ut: field,
                        temp_c: reading.temp_c,
                    });
                }
                Err(e) => {
                    defmt::warn!("Mag read error: {}", e);
                    if let Err(re) = self.mag.recover().await {
                        defmt::error!("Mag recovery failed: {}", re);
                        Timer::after_millis(100).await;
                    }
                }
            }
        }
    }
}

#[embassy_executor::task]
pub async fn qmc5883l_mag_task(mut reader: MagReader<Qmc5883lDev>) {
    reader.run().await;
}
