use core::cell::Cell;

use embassy_sync::blocking_mutex::{Mutex, raw::CriticalSectionRawMutex};

pub mod eskf_imu_gps;

/// Estimator phase, exposed via `ESTIMATOR_STATUS` for shell queries.
///
/// All variants are `Copy` so they can be stored in a `Cell` and read
/// from an ISR or any task without allocation.
#[derive(Clone, Copy)]
pub enum EstimatorPhase {
    /// Waiting for the first barometer sample.
    AwaitingBaro,
    /// Baro baseline captured; waiting for a good GPS fix (fix_type ≥ 3, sv ≥ 6).
    AwaitingGps {
        imu_ready: bool,
        /// Roll from gravity vector (degrees), `None` until first IMU sample.
        roll_deg: Option<f32>,
        /// Pitch from gravity vector (degrees), `None` until first IMU sample.
        pitch_deg: Option<f32>,
    },
    /// GPS origin set; collecting IMU samples for bias/tilt calibration.
    CalibImu,
    /// Filter is running (Phase 2).
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
    Mutex::new(Cell::new(EstimatorPhase::AwaitingBaro));
