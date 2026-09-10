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

// Position source for the ESKF — exactly one must be selected when an outer
// loop that consumes ESKF output is active. The Justfile already composes
// these mutually-exclusively, but enforce in code so a hand-rolled
// `cargo build` cannot silently produce a build that compiles offline
// missions without a defined flight environment.
#[cfg(all(feature = "est_pos_mocap", feature = "est_pos_gps"))]
compile_error!("est_pos_mocap and est_pos_gps are mutually exclusive");

// Both position-consuming outer loops need a position source. Without
// one, `main.rs` spawns no estimation task, so the loop blocks forever on
// a `VEHICLE_ODOMETRY` nothing publishes — while `health.rs` takes its
// `not(any(est_pos_*))` branch and reports `estimator_ready = true` as a
// sentinel, so the arm gate does not block either. That combination used
// to compile clean for `outer_geometric`.
#[cfg(all(
    any(feature = "outer_mpc", feature = "outer_geometric"),
    not(any(feature = "est_pos_mocap", feature = "est_pos_gps"))
))]
compile_error!(
    "outer_mpc / outer_geometric require one of est_pos_mocap (indoor) or est_pos_gps (outdoor)"
);

// NOTE: dual-antenna heading fusion is decided by two `build:` knobs,
// both const in `estimation::eskf_imu_gps`: `GPS_HAS_HEADING`
// (`gps_model: unicore` — the u-blox driver never emits a heading event,
// so the fusion sites are unreachable there) AND `GPS_DUAL_ANTENNA`
// (`gps_dual_antenna: yes` — ANT2 actually fitted). The old compile_error
// here is unnecessary: `BuildYaml::validate` rejects the inconsistent
// pairing at the YAML bake, with a message naming both knobs.

#[cfg(feature = "outer_geometric")]
pub mod cascade_task;
pub mod failsafe;
pub mod indi_task;
// inner_loop.rs (legacy rate PIDs) removed — INDI is the sole inner loop.
#[cfg(feature = "outer_mpc")]
pub mod mission_planner;
#[cfg(feature = "outer_mpc")]
pub mod offline_mission;
#[cfg(feature = "outer_mpc")]
pub mod outer_loop;
pub mod rc_interpreter;
pub use msgs::{
    TrackingError, TRACKING_ERROR_SOURCE_CASCADE, TRACKING_ERROR_SOURCE_INDI,
    TRACKING_ERROR_SOURCE_MPC,
};

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

/// Learned cost-adaptation trace (`/mpc_cost` blackbox topic). The
/// outer loop publishes one message per tick while `mpc_learned_cost`
/// is enabled; silent otherwise.
pub static MPC_COST_ADAPT: PubSubChannel<CriticalSectionRawMutex, msgs::MpcCostAdapt, 4, 2, 1> =
    PubSubChannel::new();

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

/// How the desired yaw evolves over the mission trajectory. Selected
/// per mission by the YAML `headings` / `lookahead` keys; see
/// `offline_mission::MissionProfile`.
#[cfg(feature = "outer_mpc")]
pub enum MissionYawMode {
    /// Hold the yaw setpoint latched at mission entry (no `headings`
    /// in the YAML — the pre-existing behavior).
    Constant(f32),
    /// Track the min-acceleration yaw spline solved through the
    /// mission's `headings` (head = entry yaw, boundary rates zero).
    Schedule(cybflight_core::trajectory_planning::minco_acc::YawTrajectory),
    /// Point from the reference position at τ toward the reference
    /// position at τ + `dt_s` (the mission's `yaw_lookahead_dt_s`),
    /// followed at no more than `max_rate_rad_s` (the mission's
    /// `yaw_lookahead_max_rate_rad_s`) starting from the entry yaw — see
    /// `cybflight_core::trajectory_planning::lookahead_yaw`.
    Lookahead { dt_s: f32, max_rate_rad_s: f32 },
}

#[cfg(feature = "outer_mpc")]
pub struct MissionTrajectory {
    pub traj: cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial,
    pub t_start: Instant,
    pub total_duration_s: f32,
    /// Desired-yaw source for this mission (see [`MissionYawMode`]).
    pub yaw: MissionYawMode,
    /// Flatness-map convention for reference attitude + body-rate
    /// feedforward (see [`offline_mission::FlatnessMap`]).
    pub flatness_map: offline_mission::FlatnessMap,
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

// The launch threshold / debounce and the pre-launch idle throttle are
// now the `rc_launch_us`, `rc_launch_confirm_frames` and
// `indi_idle_norm` parameters (see `RcParams` / `IndiControllerParams`).
// `rc_interpreter` reads the first two from its cached `StickConfig`;
// `indi_task` reads the third with its other INDI params.

/// Sticky "drone has crossed the launch threshold this arm session"
/// latch. Set by `rc_interpreter` after `LAUNCH_CONFIRM_FRAMES`
/// confirm frames; cleared only on the disarm edge. Read by
/// `indi_task` to gate motor output between pre-launch idle and
/// normal closed-loop flight. One-way per arm session — pulling
/// throttle low in flight does NOT clear it (the existing land
/// path handles descent).
#[cfg(any(feature = "outer_mpc", feature = "outer_geometric"))]
pub static LAUNCHED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Last time the control loop published a motor command.
/// Written by indi_task, read by failsafe controller watchdog.
pub static LAST_CONTROLLER_PUBLISH: blocking_mutex::Mutex<
    CriticalSectionRawMutex,
    Cell<Option<Instant>>,
> = blocking_mutex::Mutex::new(Cell::new(None));

/// Motor command telemetry: published by INDI task, subscribed by ESP
/// bridge + blackbox recorder.
///
/// CAP=16 / SUBS=4. Sized for the **sysid** tier, where this mirror
/// publishes at ≥500 Hz: 16 slots ≈ 32 ms of drain-stall tolerance
/// there. CAP is `rates::INDI_TELEM_PUBSUB_CAP` (48 = 97 ms at the
/// sysid tier's 500 Hz, 480 ms at the 100 Hz default). It has been
/// raised twice by measurement: CAP=4 lost ~8 % of sysid samples to
/// the recorder's 10–18 ms CMD25 flush stalls, and CAP=16 lost 33 %
/// once `mpc_rate_hz` 100 cut the recorder to ~2.7 ms slices. SUBS=4 covers ESP
/// bridge + recorder + 2 spare for future shell-stream or sysid
/// telemetry consumers.
pub static ACTUATOR_MOTORS_TELEM: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::ActuatorMotors,
    { crate::rates::INDI_TELEM_PUBSUB_CAP },
    4,
    1,
> = PubSubChannel::new();

/// Control setpoint telemetry mirror of [`RATE_COMMAND`].
///
/// `RATE_COMMAND` is a `Signal` consumed by INDI via `try_take()` —
/// reading it from a logger would race with the inner loop. Instead,
/// the active outer-loop task publishes the same `AttitudeControlSetpoint`
/// to this channel alongside the Signal, leaving INDI's hot path
/// untouched. Subscribed by the blackbox recorder.
///
/// CAP=4 / SUBS=4 — same drain-stall tolerance rationale as
/// `ACTUATOR_MOTORS_TELEM`. PUBS=3 covers all three concrete
/// outer-loop publishers (`rc_interpreter`, `cascade_task`,
/// `outer_loop`); only one is feature-active per build.
pub static CONTROL_SETPOINT_TELEM: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::AttitudeControlSetpoint,
    4,
    4,
    3,
> = PubSubChannel::new();

/// Per-tick controller tracking error. Multi-publisher: the active
/// outer loop (cascade or MPC) publishes pos/vel/attitude error at
/// 50–100 Hz; INDI publishes body-rate error at 100 Hz (decimated
/// from the 8 kHz inner loop). PUBS=3 covers all three concrete
/// publishers regardless of which feature combination is enabled —
/// extra `pub` slots cost nothing on inactive paths.
///
/// CAP=16 — same sysid-tier sizing as `ACTUATOR_MOTORS_TELEM` (this
/// mirror runs at ≥500 Hz there too); SUBS=3 leaves a free slot for a
/// future shell-stream consumer.
pub static TRACKING_ERROR: PubSubChannel<
    CriticalSectionRawMutex,
    TrackingError,
    16,
    3,
    3,
> = PubSubChannel::new();

/// Processed motor RPM telemetry.
pub static PROCESSED_DSHOT_TELEM: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::DshotTelemetry,
    2,
    2,
    1,
> = PubSubChannel::new();

/// Per-motor DShot telemetry health, cumulative since boot.
///
/// Every frame the inner loop observes lands in exactly one bucket, so for
/// each motor `passed + slew_reject + range_reject + no_fresh + no_reply +
/// edt == frames`. `nis_reject` is a *subset* of `passed` (the estimator
/// only runs while armed), not a seventh bucket. Counters are monotonic and
/// never reset — difference two samples for a rate over any window.
///
/// This exists because five very different conditions used to collapse into
/// "no measurement": a normal low-RPM gap, a wiring fault, and three
/// distinct filter rejections. The distinctions that matter:
///
/// * `no_fresh` — the reply decoded cleanly but carried period 0, which
///   this ESC firmware sends to mean "no new commutation since my last
///   reply". **Normal**, and its rate scales inversely with RPM: roughly
///   half of all frames at idle, dozens consecutively during spin-up.
///   Not an error, and not a reason to trip a staleness failsafe.
/// * `no_reply` — nothing decodable came back at all (GCR or checksum
///   failure, or a reply that missed the receive window). This *is* an
///   error rate; healthy wiring should keep it low single-digit percent.
/// * `edt` — an extended-telemetry frame. Should be identically zero:
///   the decoder is called as `interpret(raw, false)`, so if EDT ever gets
///   enabled on an ESC those frames would otherwise be read as eRPM.
///
/// Counters do not accrue during the pre-launch idle bypass, which
/// `continue`s before the telemetry block runs.
#[derive(Clone, Copy, Default, defmt::Format)]
pub struct DshotMotorHealth {
    /// Survived decode, the range gate and the slew filter, and was offered
    /// to the RPM estimator.
    pub passed: u32,
    /// Of `passed`, how many the estimator's NIS gate then rejected.
    /// Only accrues while armed — the estimator idles when disarmed.
    pub nis_reject: u32,
    /// Rejected by the slew outlier filter.
    pub slew_reject: u32,
    /// Rejected by the hard range gate (negative, or above 1.5·ω_max).
    pub range_reject: u32,
    /// ESC reported "no new commutation since my last reply". Expected.
    pub no_fresh: u32,
    /// No decodable reply — GCR/checksum failure, late reply, or wiring.
    pub no_reply: u32,
    /// Extended-telemetry frame. Should always be 0.
    pub edt: u32,
}

/// Snapshot of [`DshotMotorHealth`] for all motors plus the frame count
/// they are measured against.
#[derive(Clone, Copy, defmt::Format)]
pub struct DshotHealth {
    pub timestamp: Instant,
    /// DShot telemetry frames the inner loop has consumed. Shared
    /// denominator for every per-motor bucket.
    pub frames: u32,
    pub motors: [DshotMotorHealth; 4],
}

/// Cumulative DShot telemetry health, published at the same 100 Hz
/// decimation as the rest of the inner-loop telemetry.
///
/// CAP=2 is deliberate and sufficient despite the drain-stall rationale
/// used elsewhere: the payload is cumulative, so a dropped sample loses
/// no information — the next one carries the same totals. SUBS=3 covers
/// the ESP bridge, the recorder, and one spare.
///
/// Defined here rather than in `cybflight-msgs` because nothing off-board
/// consumes it yet; promote it there (and `impl Message`) when it should
/// go on the wire.
pub static DSHOT_HEALTH: PubSubChannel<CriticalSectionRawMutex, DshotHealth, 2, 3, 1> =
    PubSubChannel::new();

/// Processed motor state telemetry (filtered omega + omega_dot + raw).
///
/// CAP=`rates::INDI_TELEM_PUBSUB_CAP` / SUBS=4 — same sysid-tier
/// drain-stall sizing as `ACTUATOR_MOTORS_TELEM`; SUBS covers ESP
/// bridge + recorder + 2 spare.
pub static PROCESSED_MOTOR_STATE: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::MotorStateTelemetry,
    { crate::rates::INDI_TELEM_PUBSUB_CAP },
    4,
    1,
> = PubSubChannel::new();
