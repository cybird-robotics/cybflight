use core::cell::Cell;
use core::sync::atomic::AtomicBool;

use embassy_sync::blocking_mutex::{Mutex, raw::CriticalSectionRawMutex};

pub mod eskf_imu_mocap;

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
    },
}

/// Shared estimator status.  Written by `estimation_task`, read by the shell.
pub static ESTIMATOR_STATUS: Mutex<CriticalSectionRawMutex, Cell<EstimatorPhase>> =
    Mutex::new(Cell::new(EstimatorPhase::AwaitingExteroceptive));

/// `true` once the ESKF gyro-bias covariance has converged.
/// Read by the arming state machine to block arming until the estimator
/// is ready (mirrors the `FAILSAFE_ACTIVE` pattern).
pub static ESTIMATOR_READY: AtomicBool = AtomicBool::new(false);
