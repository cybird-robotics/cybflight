pub mod attitude;
pub mod baro;
pub mod gps;
pub mod imu;
pub mod mag;
pub mod rc;
use core::sync::atomic::AtomicBool;
use cybflight_msgs as msgs;

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel};

/// Set to `true` when primary gyro calibration completes.
/// Checked by the arming state machine (mirrors BF `ARMING_DISABLED_GYRO_NOT_CALIBRATED`).
pub static GYRO_CALIBRATED: AtomicBool = AtomicBool::new(false);

// IMU 1: CAP=4 (small queue, fresh data preferred), SUBS=4 (attitude + telemetry + shell + spare), PUBS=1.
pub static IMU_1: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 1> =
    PubSubChannel::new();

// IMU 2: same sizing. Empty on single-IMU boards (shell prints "no data").
pub static IMU_2: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 1> =
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

// Vicon pose (position + orientation): PUBS=1 (ESP bridge RX),
// SUBS=4 (control + telemetry + shell + spare).
pub static VICON_POSE: PubSubChannel<CriticalSectionRawMutex, msgs::ViconPose, 4, 4, 1> =
    PubSubChannel::new();

// RC input: CAP=4, SUBS=4 (control + telemetry + 2 spare), PUBS=1 (single RC task).
pub static RC_INPUT: PubSubChannel<CriticalSectionRawMutex, msgs::RcInput, 4, 4, 1> =
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

// Manual control setpoint (thrust + rates from RC sticks): CAP=2 (fresh only),
// SUBS=4 (control + telemetry + shell + spare), PUBS=1 (rc_interpreter).
pub static MANUAL_CONTROL: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::ManualControlSetpoint,
    2,
    4,
    1,
> = PubSubChannel::new();
