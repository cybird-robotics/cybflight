pub mod baro;
pub mod gps;
pub mod imu;
pub mod mag;
pub mod power;
pub mod rc;
use cybflight_msgs as msgs;

use embassy_sync::{
    blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel, signal::Signal,
};

// IMU 1: CAP=4 (small queue, fresh data preferred),
// SUBS=6 (attitude + estimation + attitude_control + esp_bridge + shell stream + oneshot), PUBS=1.
pub static IMU_1: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1> = PubSubChannel::new();

// IMU 2: same sizing as IMU_1. Empty on single-IMU boards (shell prints "no data").
pub static IMU_2: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1> = PubSubChannel::new();

// CAP=4, SUBS=6 (attitude_control + nmpc + CRSF telem + esp_bridge + shell stream + spare),
// PUBS=1 (single attitude estimator).
pub static VEHICLE_ATTITUDE: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleAttitude,
    4,
    6,
    1,
> = PubSubChannel::new();

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

// Vehicle odometry (ESKF output): CAP=8 (1 kHz),
// SUBS=6 (indi_task + rc_interpreter + esp_bridge + outer_loop + mission_planner + spare),
// PUBS=1.
pub static VEHICLE_ODOMETRY: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleOdometry,
    8,
    6,
    1,
> = PubSubChannel::new();

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
