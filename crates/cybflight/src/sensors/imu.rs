use bsp_types::SensorAlign;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Instant, Timer};

use crate::hal;
use crate::{apply_alignment, ImuSample};
use hal::gpio::Output;
use hal::spi::{self, Spi};

use super::FUSED_IMU;

pub type SpiBus = Spi<'static, hal::mode::Async, spi::mode::Master>;
pub type SpiBusMtx = Mutex<NoopRawMutex, SpiBus>;
pub type ImuDev = Icm426xx<
    SpiDevice<'static, NoopRawMutex, SpiBus, Output<'static>>,
    hal::exti::ExtiInput<'static>,
>;

#[embassy_executor::task(pool_size = 2)]
pub async fn imu_reader_task(mut imu: ImuDev, align: SensorAlign) {
    loop {
        match imu.read().await {
            Ok(reading) => {
                let sample = ImuSample {
                    accel: apply_alignment(align, reading.accel),
                    gyro: apply_alignment(align, reading.gyro),
                    temp_c: reading.temp_c,
                    timestamp: Instant::now(),
                };
                let _ = FUSED_IMU.try_send(sample);
            }
            Err(e) => {
                defmt::warn!("IMU read error: {}", e);
                Timer::after_millis(10).await;
            }
        }
    }
}
