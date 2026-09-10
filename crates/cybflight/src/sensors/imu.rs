use crate::apply_alignment;
use crate::hal;
use crate::rates::IMU_PUBSUB_CAP;
use super::IMU_PUBSUB_SUBS;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use air_filters::iir::biquad::DirectForm2;
use air_filters::iir::biquad::{BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType};
use air_filters::Filter;
use bsp_types::SensorAlign;
use cybflight_drivers::imu::bmi270::Bmi270;
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use cybflight_drivers::imu::ReadImu;
use cybflight_core::imu_stamp::SampleStamper;

/// Fixed-point shift for the sample stamper's tick grid (see
/// `ImuReader::new`). 2^16 sub-ticks keep the 8 kHz period's rounding
/// error at ~2e-6 relative — far below the 1 % the stamper's PLL tracks.
const STAMP_Q: u32 = 16;
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
pub type Bmi270Dev = Bmi270<
    SpiDevice<'static, CriticalSectionRawMutex, SpiBus, Output<'static>>,
    hal::exti::ExtiInput<'static>,
    embassy_time::Delay,
>;

pub struct ImuReader<D: ReadImu> {
    imu: D,
    align: SensorAlign,
    accel_filter: [BiquadFilter<f32, DirectForm2<f32>>; 3],
    gyro_filter: [BiquadFilter<f32, DirectForm2<f32>>; 3],
    /// On-grid timestamp reconstruction + lost-sample detection; see
    /// [`cybflight_core::imu_stamp`].
    stamper: SampleStamper,
}

/// Samples the primary IMU reader judged lost (never observed) since
/// boot — from timestamp reconstruction, not from any consumer's queue.
/// The first honest loss counter on the IMU path; `imurate` prints it.
pub static IMU1_LOST_SAMPLES: AtomicU32 = AtomicU32::new(0);
/// Secondary-IMU counterpart of [`IMU1_LOST_SAMPLES`].
pub static IMU2_LOST_SAMPLES: AtomicU32 = AtomicU32::new(0);

/// Nyquist-clamp an IMU low-pass cutoff so no parameter value can panic
/// the reader into a flash-persistent boot loop.
///
/// `imu_accel_lpf_hz` / `imu_gyro_lpf_hz` are schema-bounded to 2000 Hz,
/// which is legal at the 8 kHz ICM default but sits *above Nyquist* on an
/// `imu_1khz` build (500 Hz) and on the 3.2 kHz BMI270 board (1600 Hz) —
/// where `BiquadFilterConfigBuilder::build` rejects the config and the
/// call below would panic. Both keys are reboot-flagged, so a single
/// `param set` + `save` over USB is enough to reach that state. Per
/// docs/safety_protocol.md rule 2, a config problem degrades; it never
/// kills the task.
///
/// The 0.4·fs bound is [`cybflight_core::indi::clamp_cutoff_hz`] so the
/// firmware has one Nyquist policy. The *structurally invalid* case is
/// handled here instead of there: INDI degrades a non-finite/≤0 cutoff to
/// 12 Hz, which is right for a sync filter and unflyable for a gyro LPF,
/// so this path falls back to the schema default and lets the shared
/// clamp take it from there.
fn clamp_imu_cutoff_hz(cutoff_hz: f32, sample_hz: f32, default_hz: f32, what: &str) -> f32 {
    let requested = if cutoff_hz.is_finite() && cutoff_hz > 0.0 {
        cutoff_hz
    } else {
        defmt::warn!(
            "IMU {} LPF cutoff {} invalid — using schema default {} Hz",
            what,
            cutoff_hz,
            default_hz
        );
        default_hz
    };
    let clamped = cybflight_core::indi::clamp_cutoff_hz(requested, sample_hz);
    if clamped != requested {
        defmt::warn!(
            "IMU {} LPF cutoff {} Hz too high for {} Hz sampling — clamped to {} Hz",
            what,
            requested,
            sample_hz,
            clamped
        );
    }
    clamped
}

/// Effective (post-clamp) biquad LP cutoffs of the **first** IMU
/// reader constructed — IMU-1 on every board (`board_init` builds the
/// primary before any secondary). Stored as f32 bits; 0 until init.
///
/// Exists so the blackbox can write the true acquisition filter
/// settings into each log's Metadata record: the params request one
/// cutoff, `clamp_imu_cutoff_hz` may substitute another, and only the
/// clamped value describes the data actually on the card.
pub static IMU1_EFFECTIVE_ACCEL_LPF_HZ_BITS: AtomicU32 = AtomicU32::new(0);
/// Gyro counterpart of [`IMU1_EFFECTIVE_ACCEL_LPF_HZ_BITS`].
pub static IMU1_EFFECTIVE_GYRO_LPF_HZ_BITS: AtomicU32 = AtomicU32::new(0);
/// First-wins latch for the two statics above, so a later-constructed
/// IMU-2 reader (same params today, but nothing enforces that) can't
/// overwrite the primary's record.
static IMU1_LPF_STORED: AtomicBool = AtomicBool::new(false);

impl<D: ReadImu> ImuReader<D> {
    pub fn new(
        imu: D,
        align: SensorAlign,
        accel_cutoff_hz: f32,
        gyro_cutoff_hz: f32,
    ) -> Self {
        let sample_hz = imu.sample_rate_hz();
        // Clamp before building: the cutoffs are runtime params while the
        // sample rate is a compile-time build knob, and the two are
        // validated independently. See `clamp_imu_cutoff_hz`.
        let defaults = cybflight_core::params::SensorParams::default();
        let accel_cutoff_hz = clamp_imu_cutoff_hz(
            accel_cutoff_hz,
            sample_hz,
            defaults.imu_accel_lpf_hz,
            "accel",
        );
        let gyro_cutoff_hz =
            clamp_imu_cutoff_hz(gyro_cutoff_hz, sample_hz, defaults.imu_gyro_lpf_hz, "gyro");
        // Record the primary IMU's effective cutoffs for the blackbox
        // metadata record (first constructed reader wins = IMU-1).
        if IMU1_LPF_STORED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            IMU1_EFFECTIVE_ACCEL_LPF_HZ_BITS.store(accel_cutoff_hz.to_bits(), Ordering::Release);
            IMU1_EFFECTIVE_GYRO_LPF_HZ_BITS.store(gyro_cutoff_hz.to_bits(), Ordering::Release);
        }
        let common_config_options = BiquadFilterConfigBuilder::direct_form_2()
            .sample_frequency_hz(sample_hz)
            .filter_type(BiquadFilterType::LowPass);
        Self {
            // Q16 timer ticks: period = TICK_HZ·2^16 / ODR (268 435.456
            // at 32 768 Hz / 8 kHz), so the per-sample path is integer
            // adds/shifts with no µs↔tick conversion or 64-bit divide.
            stamper: SampleStamper::from_period(
                ((embassy_time::TICK_HZ as f64 * (1u64 << STAMP_Q) as f64) / sample_hz as f64)
                    as i64,
            ),
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
        channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
        raw_channel: Option<
            &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
        >,
    ) -> ! {
        let publisher = channel.immediate_publisher();
        // Optional pre-filter mirror for blackbox sysid. `None` on
        // dual-IMU secondary slots and on boards that don't wire raw
        // logging — the per-sample cost is one cmp + branch when not
        // taken.
        let raw_publisher = raw_channel.map(|c| c.immediate_publisher());
        // Blackbox rate divider, honoured HERE rather than in the
        // recorder. `IMU_1_RAW` has exactly one subscriber — the
        // recorder — so thinning it ahead of the channel changes
        // nothing any control task sees, and it is the only placement
        // that actually saves anything: a divider behind the buffer
        // still spends a PubSub slot and a drain-budget iteration on
        // every sample it discards, so it cuts the records the
        // recorder can emit per pass one-for-one with the bytes.
        // See `blackbox::record_set::rate_div` for the measurements.
        let mut raw_ctr: u32 = 0;
        let is_imu1 = core::ptr::eq(channel, &super::IMU_1);
        let lost_counter: &'static AtomicU32 =
            if is_imu1 { &IMU1_LOST_SAMPLES } else { &IMU2_LOST_SAMPLES };
        // ~1 kHz mirror of the filtered primary stream (see
        // `sensors::IMU_1_DECIM`); one counter increment per sample.
        let decim_publisher = is_imu1.then(|| super::IMU_1_DECIM.immediate_publisher());
        let mut decim_ctr: u32 = 0;

        loop {
            match self.imu.read().await {
                Ok(reading) => {
                    // Observation time (data-ready + burst, ~constant
                    // offset) → reconstructed on-grid sample time. Wall
                    // time here carries the reader's wake jitter, which
                    // latched DRDY turns from dropped samples into late
                    // ones; the stamper puts them back on the ODR grid
                    // and counts the ones that really went missing.
                    let observed = (Instant::now().as_ticks() as i64) << STAMP_Q;
                    let stamped = self.stamper.stamp(observed);
                    if stamped.lost != 0 {
                        lost_counter.fetch_add(stamped.lost, Ordering::Relaxed);
                    }
                    let timestamp = Instant::from_ticks((stamped.stamp >> STAMP_Q).max(0) as u64);

                    let accel = apply_alignment(self.align, reading.accel_m_s2);
                    let gyro = apply_alignment(self.align, reading.gyro_rad_s);

                    // Raw publish *before* the biquad apply — analyse.py
                    // style RPM-notch fits need pre-filter samples.
                    // Thinned by the blackbox rate divider (see
                    // `raw_ctr` above); div 1 publishes every sample.
                    if let Some(raw) = &raw_publisher {
                        raw_ctr += 1;
                        if raw_ctr >= crate::blackbox::record_set::rate_div() {
                            raw_ctr = 0;
                            raw.publish_immediate(msgs::Imu {
                                accel_m_s2: accel,
                                gyro_rad_s: gyro,
                                temp_c: reading.temp_c,
                                timestamp,
                            });
                        }
                    }

                    let af = self.accel_filter.apply(accel.into());
                    let gf = self.gyro_filter.apply(gyro.into());

                    let filtered = msgs::Imu {
                        accel_m_s2: af.into(),
                        gyro_rad_s: gf.into(),
                        temp_c: reading.temp_c,
                        timestamp,
                    };
                    if let Some(decim) = &decim_publisher {
                        decim_ctr += 1;
                        if decim_ctr >= crate::rates::IMU_DECIM_DIV {
                            decim_ctr = 0;
                            decim.publish_immediate(filtered.clone());
                        }
                    }
                    publisher.publish_immediate(filtered);
                }
                Err(e) => {
                    defmt::warn!("IMU read error: {}", e);
                    self.stamper.reset();
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
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
    raw_channel: Option<
        &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
    >,
) {
    reader.run(channel, raw_channel).await;
}

#[embassy_executor::task(pool_size = 2)]
pub async fn mpu_reader_task(
    mut reader: ImuReader<MpuDev>,
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
    raw_channel: Option<
        &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
    >,
) {
    reader.run(channel, raw_channel).await;
}

#[embassy_executor::task(pool_size = 2)]
pub async fn bmi270_reader_task(
    mut reader: ImuReader<Bmi270Dev>,
    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
    raw_channel: Option<
        &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1>,
    >,
) {
    reader.run(channel, raw_channel).await;
}
