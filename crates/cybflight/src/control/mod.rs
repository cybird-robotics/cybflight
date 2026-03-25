#[cfg(all(feature = "est_mahony", feature = "est_eskf"))]
compile_error!("features est_mahony and est_eskf are mutually exclusive");
#[cfg(not(any(feature = "est_mahony", feature = "est_eskf")))]
compile_error!("one of est_mahony or est_eskf must be selected");

// attitude_control is folded into inner_loop — one unified control loop.
// INDI controller (est_eskf) replaces rate PIDs + linear allocation.
pub mod failsafe;
// indi module moved to cybflight_core::indi::controller
#[cfg(feature = "est_eskf")]
pub mod indi_task;
pub mod inner_loop;
pub mod nmpc_driver;
pub mod rc_interpreter;

#[cfg(feature = "est_eskf")]
pub mod flight_mode;

use cybflight_msgs as msgs;

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel};

#[cfg(feature = "est_eskf")]
use core::cell::Cell;
#[cfg(feature = "est_eskf")]
use embassy_sync::{blocking_mutex, signal::Signal};
#[cfg(feature = "est_eskf")]
use embassy_time::Instant;

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

pub static ATTITUDE_CONTROL_SETPOINT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::AttitudeControlSetpoint,
    2,
    3,
    1,
> = PubSubChannel::new();

#[cfg(feature = "est_eskf")]
pub static AUTO_SETPOINT: Signal<CriticalSectionRawMutex, msgs::VehicleOdometry> = Signal::new();

// Position control setpoint telemetry: CAP=2, SUBS=3 (esp_bridge + shell + spare), PUBS=1.
#[cfg(feature = "est_eskf")]
pub static POSITION_CONTROL_SETPOINT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::PositionControlSetpoint,
    2,
    3,
    1,
> = PubSubChannel::new();

/// In-flight learning toggle. When `true` AND learner prearm is latched,
/// the RLS learner collects data from raw ESC telemetry. Set by the RC
/// interpreter from the learning switch channel.
#[cfg(feature = "est_eskf")]
pub static LEARNING_ENABLED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Learner prearm switch. When `true` at arm time, the flight is configured
/// for data collection: KF off, G2 zeroed, geometric G1 only. Latched at
/// arm — mid-flight changes are ignored by the INDI task.
#[cfg(feature = "est_eskf")]
pub static LEARNER_PREARM: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

// Last time the control loop published a motor command.
// Written by inner_loop or indi_task (est_eskf), read by failsafe controller watchdog.
#[cfg(feature = "est_eskf")]
pub static LAST_CONTROLLER_PUBLISH: blocking_mutex::Mutex<
    CriticalSectionRawMutex,
    Cell<Option<Instant>>,
> = blocking_mutex::Mutex::new(Cell::new(None));

// Motor command telemetry: published by INDI task, subscribed by ESP bridge.
// CAP=2 (latest only), SUBS=2 (esp_bridge + spare), PUBS=1 (indi_task).
#[cfg(feature = "est_eskf")]
pub static ACTUATOR_MOTORS_TELEM: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::ActuatorMotors,
    2,
    2,
    1,
> = PubSubChannel::new();
