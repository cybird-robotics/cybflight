pub mod attitude;
pub mod imu;
pub mod rc;
use cybflight_msgs as msgs;

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel};

// CAP=4: small queue; fresh IMU data is always more valuable than old.
// SUBS=4: Mahony/ESKF + telemetry + 2 spare.
// PUBS=2: matches pool_size = 2 on the reader tasks (dual-IMU boards).
pub static RAW_IMU: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 2> =
    PubSubChannel::new();

// CAP=4, SUBS=4 (control + telemetry + 2 spare), PUBS=1 (single attitude estimator).
pub static VEHICLE_ATTITUDE: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleAttitude,
    4,
    4,
    1,
> = PubSubChannel::new();

// Vehicle odometry (position + velocity): PUBS=1 (single odometry source, e.g. VIO/GPS-EKF),
// SUBS=4 (control + telemetry + 2 spare).
pub static VEHICLE_ODOMETRY: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleOdometry,
    4,
    4,
    1,
> = PubSubChannel::new();

// RC input: CAP=4, SUBS=4 (control + telemetry + 2 spare), PUBS=1 (single RC task).
pub static RC_INPUT: PubSubChannel<CriticalSectionRawMutex, msgs::RcInput, 4, 4, 1> =
    PubSubChannel::new();

// RC link status: CAP=2, SUBS=3 (telemetry + shell + spare), PUBS=1 (single RC task).
pub static RC_LINK_STATUS: PubSubChannel<CriticalSectionRawMutex, msgs::RcLinkStatus, 2, 3, 1> =
    PubSubChannel::new();

// DShot telemetry: CAP=2 (high publish rate, only latest matters),
// SUBS=3 (oneshot + stream + spare), PUBS=1 (dshot_task).
pub static DSHOT_TELEMETRY: PubSubChannel<CriticalSectionRawMutex, msgs::DshotTelemetry, 2, 3, 1> =
    PubSubChannel::new();
