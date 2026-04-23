//! Scenario definitions: vehicle params, initial conditions, setpoint
//! source, pass criteria.
//!
//! A `Scenario` is the unit of test: it fully specifies what vehicle the
//! plant simulates, what the controller is chasing, and what counts as
//! success. The runner consumes it; the reporter consumes the runner's
//! output.
//!
//! ## Parameter ownership
//!
//! Every scenario owns its own `VehicleParams`. That's the single source
//! of truth for everything downstream:
//!   - `QuadPlant` is constructed from it
//!   - Controllers read their gains / limits from it
//!   - Mission / point-to-point scenarios plan their trajectory through
//!     a `QuadPlanningConfig` derived from it
//!
//! Tests pass `scenario.vehicle_params` into the plant + controller
//! factories so the three stay consistent by construction — you cannot
//! accidentally plan a trajectory against one set of limits and then fly
//! it on a plant with different mass.
//!
//! Default scenarios use `VEHICLE.build()` (canonical host-side params,
//! mirroring the firmware's `QUADROTOR_BODY` / `QUADROTOR_MOTORS`). For
//! tuning sweeps or "what if" tests, use the `*_with_params` variants
//! (explicit `VehicleParams`) or the `tweaked_vehicle` helper for a
//! one-field override.

use cybflight_core::params::VehicleParams;
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use nalgebra::{UnitQuaternion, Vector3};

use crate::plant::VEHICLE;
use crate::sensors::{GpsModel, ImuModel, PerfectImu};
use crate::trajectory::{HoverSetpoint, MissionSetpoints, SetpointSource};

/// Return a fresh copy of the canonical host-side vehicle parameters.
/// Convenience for scenarios that want the defaults without reaching into
/// the `plant` module.
pub fn default_vehicle() -> VehicleParams {
    VEHICLE.build()
}

/// Return default vehicle params with the supplied override applied.
/// Reads cleanly at call sites:
/// ```ignore
/// let vp = tweaked_vehicle(|p| p.body.mass_kg = 0.8);
/// let s = Scenario::mission_with_params("heavy_square", vp, start, &wps);
/// ```
pub fn tweaked_vehicle(f: impl FnOnce(&mut VehicleParams)) -> VehicleParams {
    let mut vp = default_vehicle();
    f(&mut vp);
    vp
}

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
    /// Canonical source of truth for plant + controller + planner
    /// construction. Populated at construction time from either the
    /// defaults or an explicit `VehicleParams` handed to the
    /// `*_with_params` constructor.
    pub vehicle_params: VehicleParams,
    pub initial_position: Vector3<f32>,
    pub initial_velocity: Vector3<f32>,
    pub initial_attitude: UnitQuaternion<f32>,
    pub setpoints: Box<dyn SetpointSource>,
    /// IMU synthesizer; defaults to `PerfectImu`. Swap via
    /// [`Scenario::with_imu`] for noise / bias sweeps.
    pub imu_model: Box<dyn ImuModel>,
    /// Optional GPS synthesizer. When `Some`, the runner activates the
    /// in-sim ESKF (fed by `imu_model` predict + this model's
    /// position/velocity updates) and hands ESKF-derived state to the
    /// controller. When `None`, the runner uses plant ground truth —
    /// this is the existing behaviour and preserves snapshot numbers.
    pub gps_model: Option<Box<dyn GpsModel>>,
    pub pass_criteria: PassCriteria,
    /// Additional wall-time to hold terminal hover after the trajectory
    /// ends (for terminal-error measurement). Ignored when the setpoint
    /// source is infinite (pure hover).
    pub terminal_hold_s: f32,
}

impl Scenario {
    /// Replace the IMU model. Use this on any of the `hover`/
    /// `point_to_point`/`mission` constructors to move off `PerfectImu`.
    pub fn with_imu(mut self, imu: Box<dyn ImuModel>) -> Self {
        self.imu_model = imu;
        self
    }

    /// Attach a GPS model. Presence switches the runner into
    /// ESKF-in-the-loop mode — controllers then read ESKF-estimated
    /// state instead of plant ground truth. Removing this call (the
    /// default `None`) keeps the existing truth-state code path, which
    /// is what the clean snapshot rows assume.
    pub fn with_gps(mut self, gps: Box<dyn GpsModel>) -> Self {
        self.gps_model = Some(gps);
        self
    }
}

// ── Hover ───────────────────────────────────────────────────────────────────

impl Scenario {
    /// Hover in place from the specified pose. `initial_tilt` is applied as
    /// a yaw-free tilt around the (1,1,0) axis so the plant starts off-level.
    pub fn hover(name: impl Into<String>, position: Vector3<f32>, initial_tilt_rad: f32) -> Self {
        Self::hover_with_params(name, default_vehicle(), position, initial_tilt_rad)
    }

    pub fn hover_with_params(
        name: impl Into<String>,
        vehicle_params: VehicleParams,
        position: Vector3<f32>,
        initial_tilt_rad: f32,
    ) -> Self {
        let axis = nalgebra::Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0));
        Self {
            name: name.into(),
            vehicle_params,
            initial_position: position,
            initial_velocity: Vector3::zeros(),
            initial_attitude: UnitQuaternion::from_axis_angle(&axis, initial_tilt_rad),
            setpoints: Box::new(HoverSetpoint::new(position)),
            imu_model: Box::new(PerfectImu),
            gps_model: None,
            pass_criteria: PassCriteria::default(),
            terminal_hold_s: 3.0,
        }
    }
}

// ── Point-to-point ──────────────────────────────────────────────────────────

impl Scenario {
    /// Single straight-line goto using the MINCO/BFGS planner. The
    /// planner config is derived from the default vehicle params.
    pub fn point_to_point(
        name: impl Into<String>,
        start: Vector3<f32>,
        target: Vector3<f32>,
    ) -> Self {
        Self::point_to_point_with_params(name, default_vehicle(), start, target)
    }

    pub fn point_to_point_with_params(
        name: impl Into<String>,
        vehicle_params: VehicleParams,
        start: Vector3<f32>,
        target: Vector3<f32>,
    ) -> Self {
        let cfg = QuadPlanningConfig::from_vehicle_params(&vehicle_params);
        let (sp, status) = MissionSetpoints::plan(start, &[target], &cfg);
        assert_convergence("point_to_point", status);
        Self {
            name: name.into(),
            vehicle_params,
            initial_position: start,
            initial_velocity: Vector3::zeros(),
            initial_attitude: UnitQuaternion::identity(),
            setpoints: Box::new(sp),
            imu_model: Box::new(PerfectImu),
            gps_model: None,
            pass_criteria: PassCriteria::default(),
            terminal_hold_s: 3.0,
        }
    }
}

// ── Mission ─────────────────────────────────────────────────────────────────

impl Scenario {
    /// Multi-waypoint mission. The last entry in `targets` becomes the
    /// terminal hover position. Planner config is derived from the
    /// default vehicle params.
    pub fn mission(name: impl Into<String>, start: Vector3<f32>, targets: &[Vector3<f32>]) -> Self {
        Self::mission_with_params(name, default_vehicle(), start, targets)
    }

    pub fn mission_with_params(
        name: impl Into<String>,
        vehicle_params: VehicleParams,
        start: Vector3<f32>,
        targets: &[Vector3<f32>],
    ) -> Self {
        let cfg = QuadPlanningConfig::from_vehicle_params(&vehicle_params);
        let (sp, status) = MissionSetpoints::plan(start, targets, &cfg);
        assert_convergence("mission", status);
        Self {
            name: name.into(),
            vehicle_params,
            initial_position: start,
            initial_velocity: Vector3::zeros(),
            initial_attitude: UnitQuaternion::identity(),
            setpoints: Box::new(sp),
            imu_model: Box::new(PerfectImu),
            gps_model: None,
            pass_criteria: PassCriteria::default(),
            terminal_hold_s: 3.0,
        }
    }
}

fn assert_convergence(
    label: &str,
    status: cybflight_core::trajectory_planning::planner::SolverStatus,
) {
    use cybflight_core::trajectory_planning::planner::SolverStatus;
    assert!(
        matches!(
            status,
            SolverStatus::Convergence | SolverStatus::Stop | SolverStatus::MaxIterations
        ),
        "{label} planner did not converge: {status:?}"
    );
}
