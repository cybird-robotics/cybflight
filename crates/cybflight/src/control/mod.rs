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
