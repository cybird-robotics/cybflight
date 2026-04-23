// ── Controller mode selection ────────────────────────────────────────
//
// outer_rate:      INDI + manual rate control (RC sticks → rate_ref)
// outer_geometric: INDI + cascade pos→att→geometric (��� rate_ref)
// outer_mpc:       INDI + SQP/MPC (→ rate_ref)
//
// Exactly one must be selected. outer_geometric and outer_mpc auto-enable
// est_eskf in Cargo.toml. Mahony attitude filter always runs.

#[cfg(not(any(
    feature = "outer_rate",
    feature = "outer_geometric",
    feature = "outer_mpc",
)))]
compile_error!("one of outer_rate, outer_geometric, or outer_mpc must be selected");

#[cfg(any(
    all(feature = "outer_rate", feature = "outer_geometric"),
    all(feature = "outer_rate", feature = "outer_mpc"),
    all(feature = "outer_geometric", feature = "outer_mpc"),
))]
compile_error!("at most one of outer_rate, outer_geometric, outer_mpc may be selected");

#[cfg(feature = "outer_geometric")]
pub mod cascade_task;
pub mod failsafe;
pub mod indi_task;
// inner_loop.rs (legacy rate PIDs) removed — INDI is the sole inner loop.
#[cfg(feature = "outer_mpc")]
pub mod mission_planner;
#[cfg(feature = "outer_mpc")]
pub mod outer_loop;
pub mod rc_interpreter;

#[cfg(feature = "est_eskf")]
pub mod flight_mode;

use cybflight_msgs as msgs;

use core::cell::Cell;
#[cfg(feature = "outer_mpc")]
use core::cell::RefCell;
use embassy_sync::{
    blocking_mutex::{self, raw::CriticalSectionRawMutex},
    pubsub::PubSubChannel,
    signal::Signal,
};
use embassy_time::Instant;

pub static OCP_SOLVER_OUTPUT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::OcpSolverOutput,
    4,
    4,
    1,
> = PubSubChannel::new();

pub static NMPC_SETPOINT: PubSubChannel<CriticalSectionRawMutex, msgs::NmpcSetpoint, 2, 2, 1> =
    PubSubChannel::new();

pub static ATTITUDE_CONTROL_SETPOINT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::AttitudeControlSetpoint,
    2,
    3,
    2,
> = PubSubChannel::new();

/// Rate command from outer loop to INDI inner loop.
///
/// Published by the active outer-loop task:
/// - `outer_rate`:      rc_interpreter (stick → rate + thrust)
/// - `outer_geometric`: cascade_task (pos→att→geometric → rate + thrust)
/// - `outer_mpc`:       outer_loop (SQP/MPC → rate + thrust)
///
/// Consumed by `indi_task` via `try_take()`. Latest-value semantics.
pub static RATE_COMMAND: Signal<CriticalSectionRawMutex, msgs::AttitudeControlSetpoint> =
    Signal::new();

/// Snapshot of the position setpoint the controller is currently tracking.
#[cfg(feature = "est_eskf")]
#[derive(Copy, Clone, Debug, defmt::Format)]
pub struct ActiveSetpoint {
    pub timestamp: Instant,
    pub position: nalgebra::Vector3<f32>,
    pub yaw_rad: f32,
}

#[cfg(feature = "est_eskf")]
pub static ACTIVE_POSITION_SETPOINT: blocking_mutex::Mutex<
    CriticalSectionRawMutex,
    Cell<Option<ActiveSetpoint>>,
> = blocking_mutex::Mutex::new(Cell::new(None));

#[cfg(feature = "est_eskf")]
pub fn read_active_setpoint() -> Option<ActiveSetpoint> {
    ACTIVE_POSITION_SETPOINT.lock(|cell| cell.get())
}

#[cfg(feature = "est_eskf")]
pub static ACTIVE_SETPOINT_READY: Signal<CriticalSectionRawMutex, ()> = Signal::new();

// ─���───────────────────────────────────────────────────────────────────
// Mission planner (outer_mpc only)
// ────────────────────────────────────────────────��────────────────────

#[cfg(feature = "outer_mpc")]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionState {
    Idle = 0,
    Planning = 1,
    Executing = 2,
}

#[cfg(feature = "outer_mpc")]
impl MissionState {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Planning,
            2 => Self::Executing,
            _ => Self::Idle,
        }
    }
}

#[cfg(feature = "outer_mpc")]
pub static MISSION_STATE: core::sync::atomic::AtomicU8 =
    core::sync::atomic::AtomicU8::new(MissionState::Idle as u8);

#[cfg(feature = "outer_mpc")]
pub static MISSION_ABORT_REQUESTED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "outer_mpc")]
pub static PLAN_REQUEST: Signal<CriticalSectionRawMutex, ()> = Signal::new();

#[cfg(feature = "outer_mpc")]
pub struct MissionTrajectory {
    pub traj: cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial,
    pub t_start: Instant,
    pub total_duration_s: f32,
    /// Solve diagnostics from the planner, propagated verbatim into every
    /// Executing-state `MissionStatus` heartbeat so the ground station can
    /// inspect the last solve at any point during the mission.
    pub solve: msgs::SolveDiagnostics,
}

#[cfg(feature = "outer_mpc")]
pub static MISSION_STATUS: PubSubChannel<CriticalSectionRawMutex, msgs::MissionStatus, 2, 2, 1> =
    PubSubChannel::new();

#[cfg(feature = "outer_mpc")]
pub static MISSION_TRAJECTORY_SLOT: blocking_mutex::Mutex<
    CriticalSectionRawMutex,
    RefCell<Option<MissionTrajectory>>,
> = blocking_mutex::Mutex::new(RefCell::new(None));

#[cfg(feature = "est_eskf")]
pub static POSITION_CONTROL_SETPOINT: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::PositionControlSetpoint,
    2,
    3,
    2,
> = PubSubChannel::new();

/// In-flight learning toggle.
pub static LEARNING_ENABLED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Learner prearm switch.
pub static LEARNER_PREARM: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Last time the control loop published a motor command.
/// Written by indi_task, read by failsafe controller watchdog.
pub static LAST_CONTROLLER_PUBLISH: blocking_mutex::Mutex<
    CriticalSectionRawMutex,
    Cell<Option<Instant>>,
> = blocking_mutex::Mutex::new(Cell::new(None));

/// Motor command telemetry: published by INDI task, subscribed by ESP bridge.
pub static ACTUATOR_MOTORS_TELEM: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::ActuatorMotors,
    2,
    2,
    1,
> = PubSubChannel::new();

/// Processed motor RPM telemetry.
pub static PROCESSED_DSHOT_TELEM: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::DshotTelemetry,
    2,
    2,
    1,
> = PubSubChannel::new();
