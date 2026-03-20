// attitude_control is disabled: inner_loop_task is the sole controller.
// pub mod attitude_control;
pub mod failsafe;
pub mod inner_loop;
pub mod flight_mode;
pub mod nmpc_driver;
pub mod rc_interpreter;

use cybflight_msgs as msgs;

use core::cell::Cell;
use embassy_sync::{
    blocking_mutex::{self, raw::CriticalSectionRawMutex},
    pubsub::PubSubChannel,
    signal::Signal,
};
use embassy_time::Instant;

// Flight mode signal: written by rc_interpreter, read by attitude_control.
pub static FLIGHT_MODE: Signal<CriticalSectionRawMutex, flight_mode::FlightMode> = Signal::new();

pub static OCP_SOLVER_OUTPUT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::OcpSolverOutput,
    4,
    4,
    1,
> = PubSubChannel::new();

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

// Auto-mode setpoint: current desired state for autonomous flight.
// Written by rc_interpreter (initial), mission_plan_task, ESP bridge, etc.
pub static AUTO_SETPOINT: Signal<CriticalSectionRawMutex, msgs::VehicleOdometry> = Signal::new();

// Last time the inner loop successfully published a motor command.
// Written by inner_loop, read by failsafe controller_watchdog_task.
pub static LAST_CONTROLLER_PUBLISH: blocking_mutex::Mutex<CriticalSectionRawMutex, Cell<Option<Instant>>> =
    blocking_mutex::Mutex::new(Cell::new(None));
