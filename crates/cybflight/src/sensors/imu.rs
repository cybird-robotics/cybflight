use bsp_types::SensorAlign;
use cybflight_core::butterworth::ButterworthFilter;
use cybflight_drivers::imu::ReadImu;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Instant, Timer};
use crate::hal;
use cybflight_msgs as msgs;
use crate::apply_alignment;
use hal::gpio::Output;
use hal::spi::{self, Spi};

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

pub struct ImuReader<D: ReadImu> {
    imu: D,
    align: SensorAlign,
    accel_filter: ButterworthFilter<f32, 3>,
    gyro_filter: ButterworthFilter<f32, 3>,
}

impl<D: ReadImu> ImuReader<D> {
    pub fn new(imu: D, align: SensorAlign, accel_cutoff_hz: f32, gyro_cutoff_hz: f32) -> Self {
        let sample_hz = imu.sample_rate_hz();
        Self {
            imu,
            align,
            accel_filter: ButterworthFilter::new(accel_cutoff_hz, sample_hz, None, None),
            gyro_filter: ButterworthFilter::new(gyro_cutoff_hz, sample_hz, None, None),
        }
    }

    pub async fn run(&mut self) -> ! {
        let publisher = super::RAW_IMU.immediate_publisher();
        loop {
            match self.imu.read().await {
                Ok(reading) => {
                    let accel = apply_alignment(self.align, reading.accel_m_s2);
                    let gyro = apply_alignment(self.align, reading.gyro_rad_s);

                    let af = self.accel_filter.compute(&accel.into());
                    let gf = self.gyro_filter.compute(&gyro.into());

                    publisher.publish_immediate(msgs::Imu {
                        accel_m_s2: af.into(),
                        gyro_rad_s: gf.into(),
                        temp_c: reading.temp_c,
                        timestamp: Instant::now(),
                    });
                }
                Err(e) => {
                    defmt::warn!("IMU read error: {}", e);
                    if let Err(re) = self.imu.recover().await {
                        defmt::error!("IMU recovery failed: {}", re);
                        self.accel_filter.reset_input_output(None, None);
                        self.gyro_filter.reset_input_output(None, None);
                        Timer::after_millis(100).await;
                    }
                }
            }
        }
    }
}

#[embassy_executor::task(pool_size = 2)]
pub async fn icm_reader_task(mut reader: ImuReader<IcmDev>) {
    reader.run().await;
}

#[embassy_executor::task(pool_size = 2)]
pub async fn mpu_reader_task(mut reader: ImuReader<MpuDev>) {
    reader.run().await;
}
