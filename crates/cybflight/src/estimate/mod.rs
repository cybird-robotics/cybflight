pub mod feedthrough_estimate;

use cybflight_msgs as msgs;

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel};

pub static VEHICLE_ODOMETRY: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::VehicleOdometry,
    4,
    6,
    1,
> = PubSubChannel::new();

// pub static OCP_SOLVER_OUTPUT: PubSubChannel<
//     CriticalSectionRawMutex,
//     msgs::OcpSolverOutput,
//     4,
//     4,
//     1,
// > = PubSubChannel::new();

// NMPC position setpoint: CAP=2 (fresh setpoints only), SUBS=2 (nmpc_driver + spare),
// PUBS=1 (single RC-to-setpoint converter, not yet implemented).
// pub static NMPC_SETPOINT: PubSubChannel<CriticalSectionRawMutex, msgs::NmpcSetpoint, 2, 2, 1> =
//     PubSubChannel::new();

pub static ATTITUDE_CONTROL_SETPOINT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::AttitudeControlSetpoint,
    2,
    3,
    1,
> = PubSubChannel::new();
