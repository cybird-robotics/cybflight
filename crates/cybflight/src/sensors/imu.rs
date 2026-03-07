use bsp_types::SensorAlign;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Instant, Timer};

use crate::hal;
use crate::{apply_alignment, msgs};
use hal::gpio::Output;
use hal::spi::{self, Spi};

use super::FUSED_IMU;

pub type SpiBus = Spi<'static, hal::mode::Async, spi::mode::Master>;
pub type SpiBusMtx = Mutex<NoopRawMutex, SpiBus>;
pub type IcmDev = Icm426xx<
    SpiDevice<'static, NoopRawMutex, SpiBus, Output<'static>>,
    hal::exti::ExtiInput<'static>,
>;
pub type MpuDev = Mpu6x00<
    SpiDevice<'static, NoopRawMutex, SpiBus, Output<'static>>,
    hal::exti::ExtiInput<'static>,
>;

#[embassy_executor::task(pool_size = 2)]
pub async fn icm_reader_task(mut imu: IcmDev, align: SensorAlign) {
    loop {
        match imu.read().await {
            Ok(reading) => {
                let sample = msgs::Imu {
                    accel_m_s2: apply_alignment(align, reading.accel),
                    gyro_rad_s: apply_alignment(align, reading.gyro),
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

#[embassy_executor::task(pool_size = 2)]
pub async fn mpu_reader_task(mut imu: MpuDev, align: SensorAlign) {
    loop {
        match imu.read().await {
            Ok(reading) => {
                let sample = msgs::Imu {
                    accel_m_s2: apply_alignment(align, reading.accel),
                    gyro_rad_s: apply_alignment(align, reading.gyro),
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
