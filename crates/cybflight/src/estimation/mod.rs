use core::cell::Cell;
use core::sync::atomic::AtomicBool;

use embassy_sync::blocking_mutex::{Mutex, raw::CriticalSectionRawMutex};
use embassy_sync::signal::Signal;
use nalgebra::Vector3;

pub mod eskf_imu_mocap;
pub mod eskf_imu_gps;
pub mod rpm_estimator;

use cybflight_core::eskf::UpdateOutcome;

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

/// Map an `UpdateOutcome` to a short tag for defmt logging.
/// `defmt` cannot format the enum directly without a `defmt::Format` impl,
/// and the enum lives in `cybflight_core` which has no defmt dep.
fn outcome_tag(outcome: UpdateOutcome) -> &'static str {
    match outcome {
        UpdateOutcome::Accepted { inflated: false } => "accepted",
        UpdateOutcome::Accepted { inflated: true } => "inflated",
        UpdateOutcome::NotInitialized => "not-init",
        UpdateOutcome::InverseFailed => "inv-fail",
        UpdateOutcome::InflationCapExceeded => "cap-exceeded",
        UpdateOutcome::NaNAfterUpdate => "nan",
        UpdateOutcome::JumpRejected => "jump",
    }
}

