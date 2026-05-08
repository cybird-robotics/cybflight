use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU8};

use cybflight_core::eskf::EskfHealth;
use cybflight_msgs as msgs;
use embassy_sync::blocking_mutex::{Mutex, raw::CriticalSectionRawMutex};
use embassy_sync::pubsub::PubSubChannel;
use embassy_sync::signal::Signal;
use embassy_time::Instant;
use nalgebra::Vector3;

pub mod eskf_imu_mocap;
pub mod eskf_imu_gps;
pub mod rpm_estimator;


use crate::ConvertToF32Secs;

/// Estimator phase, exposed via `ESTIMATOR_STATUS` for shell queries.
///
/// All variants are `Copy` so they can be stored in a `Cell` and read
/// from an ISR or any task without allocation.
#[derive(Clone, Copy)]
pub enum EstimatorPhase {
    /// Waiting for the first exteroceptive measurement (mocap pose).
    AwaitingExteroceptive,
    /// ESKF initialised but gyro-bias covariance has not yet converged.
    Converging {
        roll_deg: f32,
        pitch_deg: f32,
        yaw_deg: f32,
        pos: [f32; 3],
        vel: [f32; 3],
        gyro_bias: [f32; 3],
        accel_bias: [f32; 3],
        /// Total exteroceptive position updates that exceeded the
        /// inflation cap and were rejected since boot.
        pos_reject_total: u32,
        /// Total exteroceptive attitude updates rejected since boot.
        att_reject_total: u32,
        /// Total updates whose `R` was inflated to admit a large innovation.
        pos_inflated_total: u32,
        att_inflated_total: u32,
        /// Total absolute-jump rejections (Vicon flips / re-associations).
        jump_total: u32,
        /// u-blox carrier solution from the most recent GPS PVT
        /// (0 = none, 1 = float-RTK, 2 = fixed-RTK). 0 on the mocap
        /// path — interpret "carr_soln=0" as "not applicable" there.
        carr_soln: u8,
        /// Satellites used in the most recent fix. 0 on mocap.
        num_sv: u8,
        /// Reported horizontal accuracy estimate [mm]. 0 on mocap.
        h_acc_mm: u32,
    },
    /// Filter converged — arming is permitted.
    Running {
        roll_deg: f32,
        pitch_deg: f32,
        yaw_deg: f32,
        pos: [f32; 3],
        vel: [f32; 3],
        gyro_bias: [f32; 3],
        accel_bias: [f32; 3],
        pos_reject_total: u32,
        att_reject_total: u32,
        pos_inflated_total: u32,
        att_inflated_total: u32,
        /// Total absolute-jump rejections (Vicon flips / re-associations).
        jump_total: u32,
        /// u-blox carrier solution from the most recent GPS PVT
        /// (0 = none, 1 = float-RTK, 2 = fixed-RTK). 0 on the mocap
        /// path — interpret "carr_soln=0" as "not applicable" there.
        carr_soln: u8,
        /// Satellites used in the most recent fix. 0 on mocap.
        num_sv: u8,
        /// Reported horizontal accuracy estimate [mm]. 0 on mocap.
        h_acc_mm: u32,
    },
}

/// Shared estimator status.  Written by `estimation_task`, read by the shell.
pub static ESTIMATOR_STATUS: Mutex<CriticalSectionRawMutex, Cell<EstimatorPhase>> =
    Mutex::new(Cell::new(EstimatorPhase::AwaitingExteroceptive));

/// `true` once the ESKF gyro-bias covariance has converged.
/// Read by the arming state machine to block arming until the estimator
/// is ready (mirrors the `FAILSAFE_ACTIVE` pattern).
pub static ESTIMATOR_READY: AtomicBool = AtomicBool::new(false);

/// Current ESKF gyro bias estimate (rad/s).
/// Written by estimation_task every predict step. Available for diagnostics
/// and consumers that need the EKF-refined estimate.
pub static ESKF_GYRO_BIAS: Signal<CriticalSectionRawMutex, Vector3<f32>> = Signal::new();

/// Current ESKF accel bias estimate (m/s²).
/// Written by estimation_task every predict step, read by INDI task to
/// bias-correct raw accel for specific force feedback.
pub static ESKF_ACCEL_BIAS: Signal<CriticalSectionRawMutex, Vector3<f32>> = Signal::new();

/// Decimated bias telemetry for the blackbox recorder.
///
/// The Signals above publish at the predict rate (~1 kHz) and use
/// `try_take`-style consumption that races with any logger. Mirror
/// them onto a slow PubSubChannel sized for sysid post-flight bias
/// correction — ~10 Hz is more than enough for biases that drift at
/// seconds-scale.
///
/// CAP=2 — at 10 Hz that's 200 ms of drain-stall tolerance, well
/// above any plausible SD stall. SUBS=4 covers recorder + 3 spare
/// for future GS / shell-stream / sysid-tooling consumers.
/// PUBS=2: both ESKF source tasks declare a publisher; only one is
/// feature-active per build (`est_pos_mocap` vs `est_pos_gps`),
/// but PUBS counts declarations.
pub static ESTIMATOR_BIAS_TELEM: PubSubChannel<
    CriticalSectionRawMutex,
    msgs::EstimatorBias,
    2,
    4,
    2,
> = PubSubChannel::new();

// ---------------------------------------------------------------------------
// Fault detection (cherry-picked from 6380f81; outcome_tag dropped because the
// rewritten guard wrappers use FilterReason directly).
// ---------------------------------------------------------------------------

pub mod fault {
    pub const NAN_RESET: u32 = 1 << 0;
    pub const POS_STALE: u32 = 1 << 1;
    pub const VEL_STALE: u32 = 1 << 2;
    pub const ATT_STALE: u32 = 1 << 3;
    pub const COV_TRACE_BLOWUP: u32 = 1 << 4;
    /// EskfFailsafe's jump-cascade gate fired recently — `converged`
    /// was dropped because consecutive pose/PVT jumps exceeded the
    /// `max_consecutive_jumps` threshold (Vicon flip / RTK ambiguity
    /// loss / multipath wraparound). Held for `CASCADE_HOLD_S` after
    /// the event so the live `health` shell view still sees it for a
    /// few seconds.
    pub const GUARD_JUMP_CASCADE: u32 = 1 << 5;
    /// EskfFailsafe's reject-cascade gate fired recently — consecutive
    /// filter rejections (InverseFailed / InflationCapExceeded /
    /// NaNAfterUpdate) exceeded `max_consecutive_rejects`. Same
    /// hold-down as `GUARD_JUMP_CASCADE`.
    pub const GUARD_REJECT_CASCADE: u32 = 1 << 6;
}

pub mod att_health {
    pub const ACCEL_OK: u8 = 1 << 0;
    pub const GYRO_OK: u8 = 1 << 1;
    pub const NO_RECENT_NAN: u8 = 1 << 2;
    pub const ALL_OK: u8 = ACCEL_OK | GYRO_OK | NO_RECENT_NAN;
}

pub const POS_TIMEOUT_S: f32 = 2.0;
pub const POS_COV_BLOWUP_M2: f32 = 25.0;
pub const NAN_RESET_HOLD_S: f32 = 3.0;
/// How long a guard-cascade event keeps its fault bit asserted.
/// Cascades are momentary: the wrapper records a single timestamp
/// when the gate fires; the bit then auto-clears after this window
/// so a healthy filter doesn't carry stale fault state forever.
/// Match `NAN_RESET_HOLD_S` so all "recently bad" hold-downs share
/// one operator-visible duration.
pub const CASCADE_HOLD_S: f32 = 3.0;

pub static ESKF_FAULTS: AtomicU32 = AtomicU32::new(0);
pub static ESKF_DEGRADED: AtomicBool = AtomicBool::new(false);
/// Severe in-flight faults — polled by `failsafe_task` to force disarm.
pub static ESKF_SEVERE_FAULT: AtomicBool = AtomicBool::new(false);
pub static ATTITUDE_HEALTH: AtomicU8 = AtomicU8::new(0);
pub static ESKF_HEALTH: Mutex<CriticalSectionRawMutex, Cell<EskfHealth>> =
    Mutex::new(Cell::new(EskfHealth {
        nan_resets: 0,
        gate_rejects_pos: 0,
        gate_rejects_vel: 0,
        gate_rejects_att: 0,
        gate_rejects_baro: 0,
        gate_rejects_mag: 0,
        last_nis_pos: 0.0,
        last_nis_vel: 0.0,
        last_nis_att: 0.0,
    }));

pub static ESKF_LAST_POS_UPDATE: Mutex<CriticalSectionRawMutex, Cell<Option<Instant>>> =
    Mutex::new(Cell::new(None));
pub static ESKF_LAST_VEL_UPDATE: Mutex<CriticalSectionRawMutex, Cell<Option<Instant>>> =
    Mutex::new(Cell::new(None));
pub static ESKF_LAST_ATT_UPDATE: Mutex<CriticalSectionRawMutex, Cell<Option<Instant>>> =
    Mutex::new(Cell::new(None));
pub static ESKF_LAST_NAN_RESET: Mutex<CriticalSectionRawMutex, Cell<Option<Instant>>> =
    Mutex::new(Cell::new(None));
/// Last time EskfFailsafe surfaced `Disarmed{JumpCascade}`. Set by
/// the wrapper on the cascade outcome; consumed by `evaluate_faults`
/// to assert `fault::GUARD_JUMP_CASCADE` for `CASCADE_HOLD_S`.
pub static ESKF_LAST_JUMP_CASCADE: Mutex<CriticalSectionRawMutex, Cell<Option<Instant>>> =
    Mutex::new(Cell::new(None));
/// Last time EskfFailsafe surfaced `Disarmed{RejectCascade}`. Same
/// shape as `ESKF_LAST_JUMP_CASCADE`.
pub static ESKF_LAST_REJECT_CASCADE: Mutex<CriticalSectionRawMutex, Cell<Option<Instant>>> =
    Mutex::new(Cell::new(None));

/// Static envelope for "is the body resting / near-still enough that
/// IMU readings should be trusted as the boot reference?" Both
/// thresholds are deliberately generous — the goal is to catch a
/// vehicle that's tumbling or falling, not a vehicle that's lifting
/// off cleanly. Used by `attitude_health_bits`.
pub const ACCEL_ARM_TOL_M_S2: f32 = 1.5;
pub const GYRO_ARM_LIMIT_RAD_S: f32 = 0.5;

/// Compute the `ATTITUDE_HEALTH` bitfield from the latest IMU sample
/// + the most recent NaN-reset timestamp. Called every IMU sample by
/// the wrapper; the result is published to the `ATTITUDE_HEALTH`
/// atomic so the arming gate (and future failsafe consumers) can
/// observe attitude observability without re-deriving it.
///
/// Bit layout: see [`att_health`].
pub fn attitude_health_bits(
    accel_m_s2: &Vector3<f32>,
    gyro_rad_s: &Vector3<f32>,
    last_nan_reset: Option<Instant>,
    now: Instant,
) -> u8 {
    let mut bits = 0u8;
    if (accel_m_s2.norm() - 9.81).abs() < ACCEL_ARM_TOL_M_S2 {
        bits |= att_health::ACCEL_OK;
    }
    if gyro_rad_s.iter().all(|v| v.abs() < GYRO_ARM_LIMIT_RAD_S) {
        bits |= att_health::GYRO_OK;
    }
    let recent_nan = match last_nan_reset {
        Some(t) => now
            .checked_duration_since(t)
            .is_some_and(|d| d.as_secs_f32() < NAN_RESET_HOLD_S),
        None => false,
    };
    if !recent_nan {
        bits |= att_health::NO_RECENT_NAN;
    }
    bits
}

pub struct FaultEvalInputs {
    pub now: Instant,
    pub last_pos_update: Option<Instant>,
    pub last_vel_update: Option<Instant>,
    pub last_att_update: Option<Instant>,
    pub pos_cov_trace: f32,
    pub last_nan_reset: Option<Instant>,
    pub last_jump_cascade: Option<Instant>,
    pub last_reject_cascade: Option<Instant>,
    pub armed: bool,
}

/// Returns `(flags, severe)`. `severe` only goes true while armed.
pub fn evaluate_faults(inputs: &FaultEvalInputs) -> (u32, bool) {
    let mut flags = 0u32;

    let stale = |t: Option<Instant>| match t {
        Some(t) => inputs
            .now
            .checked_duration_since(t)
            .is_some_and(|d| d.as_secs_f32() > POS_TIMEOUT_S),
        None => false,
    };
    if stale(inputs.last_pos_update) {
        flags |= fault::POS_STALE;
    }
    if stale(inputs.last_vel_update) {
        flags |= fault::VEL_STALE;
    }
    if stale(inputs.last_att_update) {
        flags |= fault::ATT_STALE;
    }

    let recent = |t: Option<Instant>, hold_s: f32| match t {
        Some(t) => inputs
            .now
            .checked_duration_since(t)
            .is_some_and(|d| d.as_secs_f32() < hold_s),
        None => false,
    };
    if recent(inputs.last_nan_reset, NAN_RESET_HOLD_S) {
        flags |= fault::NAN_RESET;
    }
    if recent(inputs.last_jump_cascade, CASCADE_HOLD_S) {
        flags |= fault::GUARD_JUMP_CASCADE;
    }
    if recent(inputs.last_reject_cascade, CASCADE_HOLD_S) {
        flags |= fault::GUARD_REJECT_CASCADE;
    }

    if inputs.pos_cov_trace > POS_COV_BLOWUP_M2 {
        flags |= fault::COV_TRACE_BLOWUP;
    }

    // Severe = armed + a fault that means "the estimator's pos/vel
    // is currently untrusted." Cascade events qualify: jump-cascade
    // means the filter is now dead-reckoning on IMU after the
    // measurement source was rejected; reject-cascade means the
    // filter mathematics are misbehaving.
    let severe_bits = fault::POS_STALE
        | fault::VEL_STALE
        | fault::NAN_RESET
        | fault::GUARD_JUMP_CASCADE
        | fault::GUARD_REJECT_CASCADE;
    let severe = inputs.armed && ((flags & severe_bits) != 0);

    (flags, severe)
}
