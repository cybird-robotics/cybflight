pub mod attitude;
pub mod imu;
use crate::msgs;

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
