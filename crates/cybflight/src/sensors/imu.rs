use crate::apply_alignment;
use crate::hal;
use air_filters::iir::biquad::DirectForm2;
use air_filters::iir::biquad::{BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType};
use air_filters::Filter;
use bsp_types::SensorAlign;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use cybflight_drivers::imu::ReadImu;
use cybflight_msgs as msgs;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_sync::pubsub::PubSubChannel;
use embassy_time::{Instant, Timer};
use hal::gpio::Output;
use hal::spi::{self, Spi};

pub type SpiBus = Spi<'static, hal::mode::Async, spi::mode::Master>;
/// Shared-bus mutex for SPI IMU/baro buses.
///
/// `CriticalSectionRawMutex` (not `NoopRawMutex`) because the IMU reader
/// task lives on `EXECUTOR_CTRL` (interrupt P10) while other consumers
/// of the same bus (e.g. DPS310 baro on SPI1) live on the thread
/// executor. A `NoopRawMutex` would be undefined behavior in that
/// cross-executor pattern.
///
/// Timing note: the `RawMutex` critical section guards only the
/// `is-locked` flag and runs for a handful of instructions per
/// acquire/release — NOT the full SPI/DMA transfer. The async mutex
/// guard itself is held across the DMA wait with IRQs enabled, so
/// neither the P6 DShot ISR nor the P0–P5 DMA-completion ISRs see any
/// meaningful added latency.
pub type SpiBusMtx = Mutex<CriticalSectionRawMutex, SpiBus>;
pub type IcmDev = Icm426xx<
    SpiDevice<'static, CriticalSectionRawMutex, SpiBus, Output<'static>>,
    hal::exti::ExtiInput<'static>,
>;
pub type MpuDev = Mpu6x00<
    SpiDevice<'static, CriticalSectionRawMutex, SpiBus, Output<'static>>,
    hal::exti::ExtiInput<'static>,
>;

pub struct ImuReader<D: ReadImu> {
    imu: D,
    align: SensorAlign,
    accel_filter: [BiquadFilter<f32, DirectForm2<f32>>; 3],
    gyro_filter: [BiquadFilter<f32, DirectForm2<f32>>; 3],
}

impl<D: ReadImu> ImuReader<D> {
    pub fn new(
        imu: D,
        align: SensorAlign,
        accel_cutoff_hz: f32,
        gyro_cutoff_hz: f32,
    ) -> Self {
        let sample_hz = imu.sample_rate_hz();
        let common_config_options = BiquadFilterConfigBuilder::direct_form_2()
            .sample_frequency_hz(sample_hz)
            .filter_type(BiquadFilterType::LowPass);
        Self {
            imu,
            align,
            accel_filter: core::array::from_fn(|_| {
                BiquadFilter::new(
                    common_config_options
                        .clone()
                        .cutoff_frequency_hz(accel_cutoff_hz)
                        .build()
                        .unwrap_or_else(|_| {
                            defmt::error!(
                                "Got in valid accel filter config: cutoff {}Hz, sample {}Hz",
                                accel_cutoff_hz,
                                sample_hz
                            );
                            panic!();
                        }),
                )
            }),
            gyro_filter: core::array::from_fn(|_| {
                BiquadFilter::new(
                    common_config_options
                        .clone()
                        .cutoff_frequency_hz(gyro_cutoff_hz)
                        .build()
                        .unwrap_or_else(|_| {
                            defmt::error!(
                                "Got in valid gyro filter config: cutoff {}Hz, sample {}Hz",
                                gyro_cutoff_hz,
                                sample_hz
                            );
                            panic!();
                        }),
                )
            }),
        }
    }

    pub async fn run(
        &mut self,
        channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>,
    ) -> ! {
        let publisher = channel.immediate_publisher();

        loop {
            match self.imu.read().await {
                Ok(reading) => {
                    let accel = apply_alignment(self.align, reading.accel_m_s2);
                    let gyro = apply_alignment(self.align, reading.gyro_rad_s);

                    let af = self.accel_filter.apply(accel.into());
                    let gf = self.gyro_filter.apply(gyro.into());

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
                        self.accel_filter
                            .reset([0.0; 3])
                            .expect("accel filter reset failed");
                        self.gyro_filter
                            .reset([0.0; 3])
                            .expect("gyro filter reset failed");
                        Timer::after_millis(100).await;
                    }
                }
            }
        }
    }
}

#[embassy_executor::task(pool_size = 2)]
pub async fn icm_reader_task(
    mut reader: ImuReader<IcmDev>,
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>,
) {
    reader.run(channel).await;
}

#[embassy_executor::task(pool_size = 2)]
pub async fn mpu_reader_task(
    mut reader: ImuReader<MpuDev>,
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>,
) {
    reader.run(channel).await;
}
