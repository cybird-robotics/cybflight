pub mod baro;
pub mod gps;
pub mod imu;
pub mod mag;
pub mod power;
pub mod rc;
use crate::rates::IMU_PUBSUB_CAP;
use cybflight_msgs as msgs;

use embassy_sync::{
    blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel, signal::Signal,
    watch::Watch,
};

/// Subscriber slots on every IMU channel (one concrete type for the
/// non-generic reader tasks, so IMU_1_RAW / IMU_2 inherit it).
///
/// IMU_1 budget on a full sakurah743 build: indi_task + eskf +
/// `imu1_stream_task` (permanent) = 3; blackbox recorder during a
/// Mid/Large/Sysid session = 4; one shell transient (`imu1` oneshot or
/// `imurate`) = 5; `dev_telem` esp_bridge = 6. 8 leaves two spare. (The
/// Mahony filter reads the ~1 kHz `IMU_1_DECIM` mirror, not IMU_1.) At
/// the old 6 the recorder lost its slot (no `/imu1` in the log) whenever
/// a shell transient was live at the arm edge.
pub const IMU_PUBSUB_SUBS: usize = 8;

// IMU 1: CAP=`IMU_PUBSUB_CAP` — `rates::IMU_PUBSUB_BUFFER_S` of production
// at this build's ODR, so consumer-stall tolerance is the same wall-clock
// on every build. See `crate::rates::IMU_PUBSUB_CAP` for the numbers.
// SUBS=`IMU_PUBSUB_SUBS` (see its doc for the budget), PUBS=1.
pub static IMU_1: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1> =
    PubSubChannel::new();

// IMU 1 raw (pre-biquad-LP): same CAP/SUBS/PUBS as IMU_1. Mirror of
// IMU_1 emitted before the sensor task's accel/gyro biquads. Subscribed
// by the blackbox recorder in the Sysid + Large tiers; left empty
// on tiers that don't ask for it. Sized identically to IMU_1 so the
// recorder's drain budgets / drop-priority logic stays uniform.
pub static IMU_1_RAW: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1> =
    PubSubChannel::new();

// IMU 2: same sizing as IMU_1. Empty on single-IMU boards (shell prints "no data").
// Shares IMU_1's CAP because `#[embassy_executor::task]` cannot be
// generic, so all three channels must have one concrete type for the
// reader tasks in `sensors::imu` to accept them.
pub static IMU_2: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, IMU_PUBSUB_CAP, IMU_PUBSUB_SUBS, 1> =
    PubSubChannel::new();

// CAP=4, SUBS=6 (attitude_control + nmpc + CRSF telem + esp_bridge + shell stream + spare),
// PUBS=1 (single attitude estimator).
pub static VEHICLE_ATTITUDE: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleAttitude,
    4,
    6,
    1,
> = PubSubChannel::new();

// IMU 1 decimated mirror, ~1 kHz: every `rates::IMU_DECIM_DIV`-th
// *filtered* IMU_1 sample, published by the IMU_1 reader right after
// the full-rate publish (one counter increment per sample). For
// consumers that want ~1 kHz and must not be woken at the ODR
// (Mahony). CAP=32 = 32 ms of stall tolerance at 1 kHz (> the 24 ms
// IMU_1 rings). SUBS=4 (mahony + 3 spare), PUBS=1.
pub static IMU_1_DECIM: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 32, 4, 1> =
    PubSubChannel::new();

// Mahony IMU-only attitude (estimation::mahony_task) — kept separate
// from the ESKF's VEHICLE_ATTITUDE so the two estimates never
// interleave on one channel. ~100 Hz. CAP=4 = 40 ms of blackbox
// drain-stall tolerance, matching the 100 Hz INDI telemetry mirrors
// (CAP=2's 20 ms measurably dropped ~1 % of samples across SD
// page-erase pauses). SUBS=4 (blackbox recorder + shell stream +
// 2 spare), PUBS=1 (the Mahony task).
pub static MAHONY_ATTITUDE: PubSubChannel<CriticalSectionRawMutex, msgs::VehicleAttitude, 4, 4, 1> =
    PubSubChannel::new();

// Vicon pose (position + orientation): PUBS=1 (ESP bridge RX),
// SUBS=4 (control + telemetry + shell + spare).
pub static VICON_POSE: PubSubChannel<CriticalSectionRawMutex, msgs::ViconPose, 4, 4, 1> =
    PubSubChannel::new();

// RC input: CAP=4, SUBS=4 (control + telemetry + 2 spare), PUBS=1 (single RC task).
pub static RC_INPUT: PubSubChannel<CriticalSectionRawMutex, msgs::RcInput, 4, 6, 1> =
    PubSubChannel::new();

// RC link status: CAP=2, SUBS=4 (esp_bridge + shell stream + oneshot + spare), PUBS=1 (single RC task).
pub static RC_LINK_STATUS: PubSubChannel<CriticalSectionRawMutex, msgs::RcLinkStatus, 2, 4, 1> =
    PubSubChannel::new();

// DShot telemetry: CAP=2 (high publish rate, only latest matters),
// SUBS=4 (esp_bridge + shell stream + oneshot + spare), PUBS=1 (dshot_task).
pub static DSHOT_TELEMETRY: PubSubChannel<CriticalSectionRawMutex, msgs::DshotTelemetry, 2, 4, 1> =
    PubSubChannel::new();

// GPS fix: CAP=2 (5 Hz, low rate), SUBS=4 (telemetry + shell + 2 spare), PUBS=1.
pub static GPS_FIX: PubSubChannel<CriticalSectionRawMutex, msgs::GpsFix, 2, 4, 1> =
    PubSubChannel::new();

// Full NAV-PVT fix (local, carries NED velocity + s_acc for estimator use).
// Signal (latest-wins) — 5 Hz cadence, only the ESKF GPS task consumes it,
// and missing a stale fix is preferable to queuing up old ones.
pub static GPS_NAV_PVT: Signal<CriticalSectionRawMutex, gps::GpsNavPvt> = Signal::new();

// Dual-antenna heading (UM982 only; empty on u-blox builds). Signal
// (latest-wins) — consumed by the ESKF GPS task as a yaw-aiding
// (vector/direction) attitude measurement.
pub static GPS_HEADING: Signal<CriticalSectionRawMutex, gps::GpsHeading> = Signal::new();

// Vehicle odometry (ESKF output): CAP=8 (1 kHz),
// SUBS=6 (indi_task + rc_interpreter + esp_bridge + outer_loop + mission_planner + spare),
// PUBS=1.
//
// Every consumer here drains to the latest sample, so 8 slots of jitter
// tolerance is what this channel has always needed. The blackbox does
// NOT read it — it reads the decimated `BLACKBOX_ODOMETRY` mirror below,
// so this channel is no longer sized for a logger that wanted every
// sample.
pub static VEHICLE_ODOMETRY: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleOdometry,
    8,
    6,
    1,
> = PubSubChannel::new();

// Blackbox odometry mirror: the same ESKF samples, decimated to
// `rates::BLACKBOX_ODOM_HZ` (125 Hz) at the publisher so the recorder's
// buffer holds *records* rather than samples it is about to discard.
// CAP=`rates::BLACKBOX_ODOM_PUBSUB_CAP` (25 = 200 ms of the recorder's
// own stall distribution), SUBS=2 (recorder + spare), PUBS=1.
//
// Exists because `VEHICLE_ODOMETRY` cannot be thinned at the source —
// the inner loop and outer loop read it — and thinning behind a PubSub
// costs what it saves. Same shape as `IMU_1_DECIM`. See
// `rates::BLACKBOX_ODOM_DECIM` for the measurements.
pub static BLACKBOX_ODOMETRY: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleOdometry,
    { crate::rates::BLACKBOX_ODOM_PUBSUB_CAP },
    2,
    1,
> = PubSubChannel::new();

// Neighbour fleet state (pos/vel of all drones), uplinked by the GS at 10 Hz.
// A `Watch` (latest-value) rather than a stream: a consumer always reads the
// most recent snapshot non-blockingly via `receiver().try_get()` and judges
// staleness from `NeighborStates::is_stale` (whole link) + per-entry `valid`/
// `age` — so a stopped broadcast or a dropped drone degrades safely instead of
// blocking. Up to 4 receivers (avoidance/formation/shell + spare).
pub static NEIGHBOR_STATES: Watch<CriticalSectionRawMutex, msgs::NeighborStates, 4> = Watch::new();

// External magnetometer: CAP=4 (200 Hz), SUBS=4 (attitude + telemetry + shell + spare), PUBS=1.
pub static MAG_EXT: PubSubChannel<CriticalSectionRawMutex, msgs::MagSample, 4, 4, 1> =
    PubSubChannel::new();

// Internal magnetometer: CAP=4 (100 Hz), SUBS=4, PUBS=1.
pub static MAG_INT: PubSubChannel<CriticalSectionRawMutex, msgs::MagSample, 4, 4, 1> =
    PubSubChannel::new();

// Baro 1: CAP=2, SUBS=4 (altitude + telemetry + shell + spare), PUBS=1.
pub static BARO_1: PubSubChannel<CriticalSectionRawMutex, msgs::BaroSample, 2, 4, 1> =
    PubSubChannel::new();

// Baro 2: same sizing for dual-baro boards.
pub static BARO_2: PubSubChannel<CriticalSectionRawMutex, msgs::BaroSample, 2, 4, 1> =
    PubSubChannel::new();

// Power status: CAP=2 (low-rate, latest matters),
// SUBS=5 (esp_bridge + shell stream + oneshot + indi_task + spare), PUBS=1 (power_task).
pub static POWER_STATUS: PubSubChannel<CriticalSectionRawMutex, msgs::PowerStatus, 2, 5, 1> =
    PubSubChannel::new();

// Power telemetry for the blackbox: the same 100 Hz tick as
// `POWER_STATUS` but carrying the *unfiltered* pack voltage alongside
// the `batt_lpf_hz`-filtered one, so a log can show how much the filter
// lags a throttle-punch IR drop. CAP=8 (80 ms drain-stall tolerance at
// 100 Hz), SUBS=2 (blackbox + spare), PUBS=1 (power_task).
pub static POWER_TELEM: PubSubChannel<CriticalSectionRawMutex, power::PowerTelemetry, 8, 2, 1> =
    PubSubChannel::new();
