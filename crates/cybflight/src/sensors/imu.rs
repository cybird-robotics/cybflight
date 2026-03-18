use core::sync::atomic::{AtomicBool, Ordering};

use crate::apply_alignment;
use crate::hal;
use bsp_types::SensorAlign;
use cybflight_core::butterworth::ButterworthFilter;
use cybflight_drivers::imu::ReadImu;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use cybflight_msgs as msgs;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_sync::blocking_mutex::raw::{CriticalSectionRawMutex, NoopRawMutex};
use embassy_sync::mutex::Mutex;
use embassy_sync::pubsub::PubSubChannel;
use embassy_time::{Instant, Timer};
use hal::gpio::Output;
use hal::spi::{self, Spi};
use nalgebra::Vector3;

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

// ---------------------------------------------------------------------------
// Gyro calibration (BF-style: average at rest, motion rejection)
// ---------------------------------------------------------------------------

/// Calibration duration in microseconds. Matches BF `CALIBRATING_GYRO_TIME_US`.
const GYRO_CAL_DURATION_US: u64 = 1_250_000;

/// Per-axis range (max − min) threshold for motion rejection (rad/s).
/// If any axis exceeds this during calibration, the accumulator resets.
/// BF uses a similar check via `devStandardDeviation`.
const GYRO_CAL_MOTION_THRESHOLD: f32 = 0.1;

/// Accumulates gyro samples at rest and computes a bias offset.
struct GyroCal {
    sum: Vector3<f64>,
    min: Vector3<f32>,
    max: Vector3<f32>,
    count: u32,
    target_samples: u32,
}

impl GyroCal {
    fn new(sample_rate_hz: f32) -> Self {
        let target = (sample_rate_hz * (GYRO_CAL_DURATION_US as f32 / 1_000_000.0)) as u32;
        Self {
            sum: Vector3::zeros(),
            min: Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Vector3::new(f32::MIN, f32::MIN, f32::MIN),
            count: 0,
            target_samples: target,
        }
    }

    fn reset(&mut self) {
        self.sum = Vector3::zeros();
        self.min = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
        self.max = Vector3::new(f32::MIN, f32::MIN, f32::MIN);
        self.count = 0;
    }

    /// Feed one gyro sample. Returns `Some(bias)` when calibration completes.
    fn feed(&mut self, gyro: &Vector3<f32>) -> Option<Vector3<f32>> {
        for i in 0..3 {
            if gyro[i] < self.min[i] {
                self.min[i] = gyro[i];
            }
            if gyro[i] > self.max[i] {
                self.max[i] = gyro[i];
            }
            if self.max[i] - self.min[i] > GYRO_CAL_MOTION_THRESHOLD {
                defmt::warn!("Gyro cal: motion detected (axis {}), restarting", i);
                self.reset();
                return None;
            }
        }

        self.sum += gyro.cast::<f64>();
        self.count += 1;

        if self.count >= self.target_samples {
            Some((self.sum / self.count as f64).cast::<f32>())
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// IMU reader
// ---------------------------------------------------------------------------

pub struct ImuReader<D: ReadImu> {
    imu: D,
    align: SensorAlign,
    accel_filter: ButterworthFilter<f32, 3>,
    gyro_filter: ButterworthFilter<f32, 3>,
    /// Optional flag set to `true` when gyro calibration completes.
    /// Pass `Some(&GYRO_CALIBRATED)` for the primary IMU.
    cal_flag: Option<&'static AtomicBool>,
}

impl<D: ReadImu> ImuReader<D> {
    pub fn new(
        imu: D,
        align: SensorAlign,
        accel_cutoff_hz: f32,
        gyro_cutoff_hz: f32,
        cal_flag: Option<&'static AtomicBool>,
    ) -> Self {
        let sample_hz = imu.sample_rate_hz();
        Self {
            imu,
            align,
            accel_filter: ButterworthFilter::new(accel_cutoff_hz, sample_hz, None, None),
            gyro_filter: ButterworthFilter::new(gyro_cutoff_hz, sample_hz, None, None),
            cal_flag,
        }
    }

    pub async fn run(
        &mut self,
        channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>,
    ) -> ! {
        let publisher = channel.immediate_publisher();
        let sample_rate = self.imu.sample_rate_hz();

        // --- Gyro calibration phase ---
        let mut cal = GyroCal::new(sample_rate);
        if self.cal_flag.is_some() {
            crate::status::STATUS
                .sender()
                .send(crate::status::SystemStatus::Calibrating);
        }
        defmt::info!(
            "Gyro cal: collecting {} samples @ {}Hz (hold still!)",
            cal.target_samples,
            sample_rate as u32,
        );

        let gyro_bias = loop {
            match self.imu.read().await {
                Ok(reading) => {
                    let gyro = apply_alignment(self.align, reading.gyro_rad_s);
                    if let Some(bias) = cal.feed(&gyro) {
                        break bias;
                    }
                    // Publish during calibration (bias=0) so Mahony can start converging
                    let accel = apply_alignment(self.align, reading.accel_m_s2);
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
                    defmt::warn!("IMU read error during cal: {}", e);
                    if let Err(re) = self.imu.recover().await {
                        defmt::error!("IMU recovery failed: {}", re);
                        Timer::after_millis(100).await;
                    }
                }
            }
        };

        defmt::info!(
            "Gyro cal complete: bias=[{}, {}, {}] rad/s",
            gyro_bias.x,
            gyro_bias.y,
            gyro_bias.z,
        );
        if let Some(flag) = self.cal_flag {
            flag.store(true, Ordering::Release);
            crate::status::STATUS
                .sender()
                .send(crate::status::SystemStatus::Disarmed);
        }

        // Reset filters so the step change from bias subtraction doesn't ring
        self.gyro_filter.reset_input_output(None, None);

        // --- Normal operation: subtract bias ---
        loop {
            match self.imu.read().await {
                Ok(reading) => {
                    let accel = apply_alignment(self.align, reading.accel_m_s2);
                    let gyro = apply_alignment(self.align, reading.gyro_rad_s) - gyro_bias;

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
