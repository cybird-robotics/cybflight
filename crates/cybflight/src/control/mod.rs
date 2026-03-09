pub mod nmpc_driver;

use crate::msgs;

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel};

pub static OCP_SOLVER_OUTPUT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::OcpSolverOutput,
    4,
    4,
    1,
> = PubSubChannel::new();

// NMPC position setpoint: CAP=2 (fresh setpoints only), SUBS=2 (nmpc_driver + spare),
// PUBS=1 (single RC-to-setpoint converter, not yet implemented).
pub static NMPC_SETPOINT: PubSubChannel<CriticalSectionRawMutex, msgs::NmpcSetpoint, 2, 2, 1> =
    PubSubChannel::new();
