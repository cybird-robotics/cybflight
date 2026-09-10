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
pub mod mahony_task;
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

/// Bias-telemetry publish decimation relative to the ESKF predict rate
/// (1 kHz / 100 = 10 Hz). Shared by both estimation tasks — biases drift
/// slowly, so the channel runs well below the predict rate.
pub const BIAS_TELEM_DECIMATION: u32 = 100;

/// Upper bound on a single ESKF predict step [s]. A gap larger than this
/// means the IMU stream hiccuped (task starvation, driver recovery); the
/// linearization is only valid over a short interval, so the step is
/// skipped rather than propagated over a stale dt. Shared by both
/// estimation tasks so the GPS and mocap paths cannot drift apart.
pub const MAX_PREDICT_DT_S_FALLBACK: f32 = 0.05;

/// The configured longest IMU gap the estimator will integrate across
/// (`eskf_max_predict_dt_s`), degrading to [`MAX_PREDICT_DT_S_FALLBACK`]
/// if the value is not usable.
///
/// Read once at task start, never per predict step: `params::get()`
/// clones the whole config inside a critical section and the predict
/// path runs at ~1 kHz.
pub fn max_predict_dt_s(p: &cybflight_core::params::FirmwareConfig) -> f32 {
    let v = p.eskf.filter.max_predict_dt_s;
    if v.is_finite() && v > 0.0 {
        v
    } else {
        defmt::warn!("estimation: eskf_max_predict_dt_s unusable — using default");
        MAX_PREDICT_DT_S_FALLBACK
    }
}

/// Fault-annunciation windows from the `eskf.faults` param group.
///
/// `eskf.faults` is reboot-flagged, so this is a one-shot read: callers
/// snapshot it at task start and pass it in. It used to be read inside
/// `evaluate_faults` on the claim that the evaluator "runs at the
/// telemetry cadence" — it does not. `ODOM_DECIMATION` is 1, so it runs
/// at the full predict rate, and `params::get()` clones the whole
/// `FirmwareConfig` inside a critical section, which disables the DShot
/// and INDI interrupts.
pub fn fault_params() -> cybflight_core::params::EskfFaultParams {
    crate::params::get().eskf.faults
}

pub static ESKF_FAULTS: AtomicU32 = AtomicU32::new(0);
pub static ESKF_DEGRADED: AtomicBool = AtomicBool::new(false);
/// Severe estimator faults, as classified by `evaluate_faults`.
///
/// Consumed by the **arm gate** (`health::first_blocker` →
/// `BlockReason::EskfSevereFault`) and by telemetry (`health`,
/// `health_wire`, the `/health` blackbox topic).
///
/// `severe` is by construction `armed && <untrusted-pos/vel bits>`, so
/// this can only ever be true *in flight*. That makes it useless to the
/// arm gate — pre-arm it is always false, and `ESKF_DEGRADED`
/// (`flags != 0 && !severe`) already carries every asserted bit there.
///
/// It is currently telemetry-only. It is deliberately NOT wired to
/// `failsafe_task`: an estimator fault mid-flight leaves attitude
/// control intact and usable, and on a mocap vehicle there is no
/// secondary position source to fall back to, so cutting the motors is
/// not obviously safer than letting the pilot take over. Wiring an
/// auto-disarm here is a policy decision, not a bug fix.
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
    p: &AttitudeHealthParams,
) -> u8 {
    let mut bits = 0u8;
    if (accel_m_s2.norm() - p.gravity_m_s2).abs() < p.arm_accel_tol_m_s2 {
        bits |= att_health::ACCEL_OK;
    }
    if gyro_rad_s.iter().all(|v| v.abs() < p.arm_gyro_limit_rad_s) {
        bits |= att_health::GYRO_OK;
    }
    let recent_nan = match last_nan_reset {
        Some(t) => now
            .checked_duration_since(t)
            .is_some_and(|d| d.as_secs_f32() < p.nan_reset_hold_s),
        None => false,
    };
    if !recent_nan {
        bits |= att_health::NO_RECENT_NAN;
    }
    bits
}

/// The four scalars `attitude_health_bits` needs, lifted out of
/// `FirmwareConfig`.
///
/// This runs on **every IMU sample** — 8 kHz on an `imu_rate: 8khz`
/// vehicle. It used to call `params::get()` inline, which clones the
/// entire ~260-field `FirmwareConfig` inside a `critical_section`, i.e.
/// with the P6 DShot and P10 INDI interrupts masked, 8000 times a
/// second. All three source groups (`site`, `safety`, `eskf`) are
/// reboot-flagged, so snapshotting once at task start is exactly the
/// documented contract.
#[derive(Clone, Copy, Debug)]
pub struct AttitudeHealthParams {
    pub gravity_m_s2: f32,
    pub arm_accel_tol_m_s2: f32,
    pub arm_gyro_limit_rad_s: f32,
    pub nan_reset_hold_s: f32,
}

impl AttitudeHealthParams {
    /// Snapshot from the live config. Call once, at task start.
    pub fn snapshot() -> Self {
        let p = crate::params::get();
        Self {
            gravity_m_s2: p.site.gravity_m_s2,
            arm_accel_tol_m_s2: p.safety.arm_accel_tol_m_s2,
            arm_gyro_limit_rad_s: p.safety.arm_gyro_limit_rad_s,
            nan_reset_hold_s: p.eskf.faults.nan_reset_hold_s,
        }
    }
}

pub struct FaultEvalInputs {
    /// Fault-annunciation windows, snapshotted at task start —
    /// `eskf` is reboot-flagged. Passed in rather than read inside
    /// `evaluate_faults` for the same reason as
    /// [`AttitudeHealthParams`]: this runs at the full predict rate.
    pub fault_params: cybflight_core::params::EskfFaultParams,
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
    let fp = inputs.fault_params;
    let mut flags = 0u32;

    let stale = |t: Option<Instant>| match t {
        Some(t) => inputs
            .now
            .checked_duration_since(t)
            .is_some_and(|d| d.as_secs_f32() > fp.pos_timeout_s),
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
    if recent(inputs.last_nan_reset, fp.nan_reset_hold_s) {
        flags |= fault::NAN_RESET;
    }
    if recent(inputs.last_jump_cascade, fp.cascade_hold_s) {
        flags |= fault::GUARD_JUMP_CASCADE;
    }
    if recent(inputs.last_reject_cascade, fp.cascade_hold_s) {
        flags |= fault::GUARD_REJECT_CASCADE;
    }

    if inputs.pos_cov_trace > fp.cov_blowup_m2 {
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
