//! Setpoint sources for the simulation runner.
//!
//! Hover, point-to-point, and multi-waypoint missions all implement the same
//! `SetpointSource` trait. Missions are planned offline via the MINCO/BFGS
//! solver in `cybflight_core::trajectory_planning` and sampled along a
//! `PiecewisePolynomial`. When the trajectory completes the sampler pins the
//! last waypoint (hover-at-end) so the runner can still assess terminal
//! stability without a special-case code path.

use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::planner::{plan_with_workspace, PlannerInput, SolverStatus};
use cybflight_core::trajectory_planning::bfgs_trust::BfgsWorkspace;
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use nalgebra::Vector3;

/// One reference sample handed to the controller at each tick.
#[derive(Clone, Copy, Debug)]
pub struct Setpoint {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub acceleration: Vector3<f32>,
    pub yaw: f32,
    /// Flagged true on the tick at which the trajectory completes. The runner
    /// uses this for early-exit / terminal-metric windows.
    pub terminal: bool,
}

impl Setpoint {
    pub fn hover(position: Vector3<f32>) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
            acceleration: Vector3::zeros(),
            yaw: 0.0,
            terminal: true,
        }
    }
}

pub trait SetpointSource {
    fn sample(&mut self, t: f32) -> Setpoint;
    fn duration_s(&self) -> f32;
}

/// Constant-hover setpoint — used directly for hover scenarios and as the
/// post-trajectory hold for point-to-point / mission scenarios.
pub struct HoverSetpoint {
    pos: Vector3<f32>,
}

impl HoverSetpoint {
    pub fn new(pos: Vector3<f32>) -> Self {
        Self { pos }
    }
}

impl SetpointSource for HoverSetpoint {
    fn sample(&mut self, _t: f32) -> Setpoint {
        Setpoint::hover(self.pos)
    }

    fn duration_s(&self) -> f32 {
        f32::INFINITY
    }
}

/// Mission setpoint source: samples an optimized piecewise-polynomial
/// trajectory; after the trajectory ends, holds at its terminal position.
///
/// Handles both the single-segment (goto) and multi-segment (waypoint) cases
/// via the same sampler — the `SetpointSource` contract doesn't change
/// between hover / p2p / mission, which keeps the runner scenario-agnostic.
pub struct MissionSetpoints {
    traj: PiecewisePolynomial,
    duration_s: f32,
    terminal_pos: Vector3<f32>,
}

impl MissionSetpoints {
    pub fn from_trajectory(traj: PiecewisePolynomial) -> Self {
        let duration_s = traj.total_duration();
        let p = traj.get_pos(duration_s);
        let terminal_pos = Vector3::new(p[0], p[1], p[2]);
        Self {
            traj,
            duration_s,
            terminal_pos,
        }
    }

    /// Plan a trajectory from `start` through `targets` (last entry = hover
    /// at end) using the MINCO/BFGS solver. Returns the planned setpoints
    /// along with the final solver status so tests can assert convergence.
    pub fn plan(
        start: Vector3<f32>,
        targets: &[Vector3<f32>],
        config: &QuadPlanningConfig,
    ) -> (Self, SolverStatus) {
        let start_arr: Vec3 = [start.x, start.y, start.z];
        let targets_arr: Vec<Vec3> = targets.iter().map(|v| [v.x, v.y, v.z]).collect();

        let input = if targets_arr.len() == 1 {
            PlannerInput::goto(start_arr, ZERO3, targets_arr[0])
        } else {
            PlannerInput::waypoints(start_arr, ZERO3, &targets_arr)
        };

        // Own the ~35 KB BFGS workspace on the heap — stack would overflow
        // the default host-thread size on some CI configs.
        let mut ws = Box::new(BfgsWorkspace::new());
        let res = plan_with_workspace(&input, config, &mut ws);
        (Self::from_trajectory(res.trajectory), res.status)
    }
}

impl SetpointSource for MissionSetpoints {
    fn sample(&mut self, t: f32) -> Setpoint {
        if t >= self.duration_s {
            return Setpoint {
                position: self.terminal_pos,
                velocity: Vector3::zeros(),
                acceleration: Vector3::zeros(),
                yaw: 0.0,
                terminal: true,
            };
        }
        let p = self.traj.get_pos(t);
        let v = self.traj.get_vel(t);
        let a = self.traj.get_acc(t);
        Setpoint {
            position: Vector3::new(p[0], p[1], p[2]),
            velocity: Vector3::new(v[0], v[1], v[2]),
            acceleration: Vector3::new(a[0], a[1], a[2]),
            yaw: 0.0,
            terminal: false,
        }
    }

    fn duration_s(&self) -> f32 {
        self.duration_s
    }
}
