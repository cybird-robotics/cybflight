//! Scenario definitions: initial conditions, setpoint source, pass criteria.
//!
//! A `Scenario` is the unit of test: it fully specifies what the plant
//! starts from, what the controller is chasing, and what counts as success.
//! The runner consumes it; the reporter consumes the runner's output.
//!
//! Hover, point-to-point, and mission scenarios are all constructible here;
//! the runner does not branch on scenario type.

use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use nalgebra::{UnitQuaternion, Vector3};

use crate::trajectory::{HoverSetpoint, MissionSetpoints, SetpointSource};

/// Pass/fail gates evaluated by the runner against the final history.
#[derive(Clone, Debug)]
pub struct PassCriteria {
    /// Maximum allowed final position error [m] after setpoint becomes terminal.
    pub terminal_pos_err_m: f32,
    /// Maximum allowed RMS tracking error across the full scenario [m].
    pub rms_pos_err_m: f32,
    /// Maximum allowed peak tilt [rad].
    pub peak_tilt_rad: f32,
    /// Inclusive axis-aligned geofence bounds. Any violation fails.
    pub geofence_min: Vector3<f32>,
    pub geofence_max: Vector3<f32>,
}

impl Default for PassCriteria {
    fn default() -> Self {
        Self {
            terminal_pos_err_m: 0.15,
            rms_pos_err_m: 0.50,
            peak_tilt_rad: 60.0_f32.to_radians(),
            geofence_min: Vector3::new(-15.0, -15.0, -1.0),
            geofence_max: Vector3::new(15.0, 15.0, 20.0),
        }
    }
}

/// Terminal verdict emitted in the simulation report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Verdict {
    Pass,
    Fail,
}

pub struct Scenario {
    pub name: String,
    pub initial_position: Vector3<f32>,
    pub initial_velocity: Vector3<f32>,
    pub initial_attitude: UnitQuaternion<f32>,
    pub setpoints: Box<dyn SetpointSource>,
    pub pass_criteria: PassCriteria,
    /// Additional wall-time to hold terminal hover after the trajectory ends
    /// (for terminal-error measurement). Ignored when the setpoint source is
    /// infinite (pure hover).
    pub terminal_hold_s: f32,
}

impl Scenario {
    /// Hover in place from the specified pose. `initial_tilt` is applied as
    /// a yaw-free tilt around the (1,1,0) axis so the plant starts off-level.
    pub fn hover(
        name: impl Into<String>,
        position: Vector3<f32>,
        initial_tilt_rad: f32,
    ) -> Self {
        let axis = nalgebra::Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0));
        Self {
            name: name.into(),
            initial_position: position,
            initial_velocity: Vector3::zeros(),
            initial_attitude: UnitQuaternion::from_axis_angle(&axis, initial_tilt_rad),
            setpoints: Box::new(HoverSetpoint::new(position)),
            pass_criteria: PassCriteria::default(),
            terminal_hold_s: 3.0,
        }
    }

    /// Single straight-line goto using the MINCO/BFGS planner.
    pub fn point_to_point(
        name: impl Into<String>,
        start: Vector3<f32>,
        target: Vector3<f32>,
        planner_config: &QuadPlanningConfig,
    ) -> Self {
        let (sp, status) = MissionSetpoints::plan(start, &[target], planner_config);
        assert_convergence("point_to_point", status);
        Self {
            name: name.into(),
            initial_position: start,
            initial_velocity: Vector3::zeros(),
            initial_attitude: UnitQuaternion::identity(),
            setpoints: Box::new(sp),
            pass_criteria: PassCriteria::default(),
            terminal_hold_s: 3.0,
        }
    }

    /// Multi-waypoint mission. The last entry in `targets` becomes the
    /// terminal hover position.
    pub fn mission(
        name: impl Into<String>,
        start: Vector3<f32>,
        targets: &[Vector3<f32>],
        planner_config: &QuadPlanningConfig,
    ) -> Self {
        let (sp, status) = MissionSetpoints::plan(start, targets, planner_config);
        assert_convergence("mission", status);
        Self {
            name: name.into(),
            initial_position: start,
            initial_velocity: Vector3::zeros(),
            initial_attitude: UnitQuaternion::identity(),
            setpoints: Box::new(sp),
            pass_criteria: PassCriteria::default(),
            terminal_hold_s: 3.0,
        }
    }
}

fn assert_convergence(label: &str, status: cybflight_core::trajectory_planning::planner::SolverStatus) {
    use cybflight_core::trajectory_planning::planner::SolverStatus;
    assert!(
        matches!(
            status,
            SolverStatus::Convergence | SolverStatus::Stop | SolverStatus::MaxIterations
        ),
        "{label} planner did not converge: {status:?}"
    );
}
