//! Trajectory planner: jointly optimizes waypoint positions and segment times
//! subject to velocity, thrust, tilt, and body rate constraints.
//!
//! Uses MINCO min-jerk representation with BFGS trust-region optimization.
//! Yaw is assumed zero during planning (differential flatness at ψ=0).

#[allow(unused_imports)]
use num_traits::Float;

use super::bfgs_trust::{bfgs_trust_optimize, BfgsTrustResult};
use super::cost_eval::CostEvaluator;
use super::minco_jerk::MincoJerk;
use super::penalties::{backward_t, forward_t};
use super::piecewise_polynomial::PiecewisePolynomial;
use super::quad_planning_config::QuadPlanningConfig;
use super::types::*;
use super::MAX_PIECES;

/// Solver convergence status.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverStatus {
    Convergence,
    Stop,
    MaxIterations,
    InvalidValue,
}

impl From<BfgsTrustResult> for SolverStatus {
    fn from(r: BfgsTrustResult) -> Self {
        match r {
            BfgsTrustResult::Convergence => Self::Convergence,
            BfgsTrustResult::Stop => Self::Stop,
            BfgsTrustResult::MaxIterations => Self::MaxIterations,
            BfgsTrustResult::InvalidValue => Self::InvalidValue,
        }
    }
}

/// Input specification for the planner.
pub struct PlannerInput {
    /// Start state [pos, vel, acc].
    pub head: PVA3D,
    /// End state [pos, vel, acc].
    pub tail: PVA3D,
    /// Intermediate waypoint positions (first `num_waypoints` entries valid).
    pub waypoints: [Vec3; MAX_PIECES],
    /// Number of intermediate waypoints (pieces = num_waypoints + 1).
    pub num_waypoints: usize,
    /// Initial time allocation per segment (first `num_waypoints + 1` entries valid).
    pub init_times: [f32; MAX_PIECES],
}

impl PlannerInput {
    /// Build input for a single go-to-position (1 piece, 0 intermediate waypoints).
    ///
    /// Flies directly from `start_pos` to `target_pos` along a single polynomial
    /// segment. Initial segment time is estimated from distance.
    pub fn goto(start_pos: Vec3, start_vel: Vec3, target_pos: Vec3) -> Self {
        let head: PVA3D = [start_pos, start_vel, ZERO3];
        let tail: PVA3D = [target_pos, ZERO3, ZERO3];
        let dist = norm_sq3(sub3(start_pos, target_pos)).sqrt();
        let mut init_times = [0.0f32; MAX_PIECES];
        init_times[0] = (dist / 2.0).max(0.5);
        Self {
            head,
            tail,
            waypoints: [ZERO3; MAX_PIECES],
            num_waypoints: 0,
            init_times,
        }
    }

    /// Build input for a waypoint sequence.
    ///
    /// `targets` contains **all** target positions in order. The last element
    /// becomes the final hover position (tail). All preceding elements become
    /// intermediate waypoints.
    ///
    /// Example: `targets = [A, B, C]` produces the path
    /// `start → A → B → C(hover)` with 2 intermediate waypoints and 3 pieces.
    ///
    /// Panics if `targets` is empty.
    pub fn waypoints(start_pos: Vec3, start_vel: Vec3, targets: &[Vec3]) -> Self {
        assert!(
            !targets.is_empty(),
            "targets must have at least one position"
        );

        let last_idx = targets.len() - 1;
        let tail_pos = targets[last_idx];
        let head: PVA3D = [start_pos, start_vel, ZERO3];
        let tail: PVA3D = [tail_pos, ZERO3, ZERO3];

        let num_intermediate = last_idx.min(MAX_PIECES - 1);
        let mut wps = [ZERO3; MAX_PIECES];
        for i in 0..num_intermediate {
            wps[i] = targets[i];
        }

        let num_segments = (num_intermediate + 1).min(MAX_PIECES);
        let mut init_times = [0.0f32; MAX_PIECES];
        let mut prev = start_pos;
        for i in 0..num_segments {
            let next = if i < num_intermediate {
                wps[i]
            } else {
                tail_pos
            };
            let d = norm_sq3(sub3(prev, next)).sqrt();
            init_times[i] = (d / 2.0).max(0.5);
            prev = next;
        }

        Self {
            head,
            tail,
            waypoints: wps,
            num_waypoints: num_intermediate,
            init_times,
        }
    }
}

/// Result of trajectory optimization.
pub struct PlannerResult {
    /// The optimized piecewise polynomial trajectory.
    pub trajectory: PiecewisePolynomial,
    /// Final objective value.
    pub final_cost: f32,
    /// Total solver iterations.
    pub iterations: usize,
    /// Solver convergence status.
    pub status: SolverStatus,
    /// Optimized segment durations [s] (first `num_pieces` entries valid).
    pub optimized_times: [f32; MAX_PIECES],
    /// Optimized waypoint positions (first `num_pieces - 1` entries valid).
    pub optimized_waypoints: [Vec3; MAX_PIECES],
    /// Number of polynomial pieces in the trajectory.
    pub num_pieces: usize,
}

/// Run trajectory optimization.
///
/// Jointly optimizes waypoint positions and segment times to minimize a
/// weighted sum of trajectory time, energy (jerk integral), and soft
/// constraint penalties (velocity, thrust, tilt, body rate).
pub fn plan(input: &PlannerInput, config: &QuadPlanningConfig) -> PlannerResult {
    let n_wp = input.num_waypoints;
    let n_pieces = n_wp + 1;
    debug_assert!(n_pieces >= 1 && n_pieces <= MAX_PIECES);

    let dim_k = n_pieces;
    let dim_d = 3 * n_wp;
    let dim_total = dim_k + dim_d;

    // Initialize decision vector: x = [K_times, D_waypoints]
    let mut x = [0.0f32; 4 * MAX_PIECES];
    for i in 0..n_pieces {
        x[i] = backward_t(input.init_times[i]);
    }
    for i in 0..n_wp {
        x[dim_k + 3 * i] = input.waypoints[i][0];
        x[dim_k + 3 * i + 1] = input.waypoints[i][1];
        x[dim_k + 3 * i + 2] = input.waypoints[i][2];
    }

    // Build cost evaluator
    let mut evaluator = CostEvaluator::new(config, n_pieces, &input.head, &input.tail);

    // Wrap as closure for the optimizer
    let mut eval_fn = |xv: &[f32], grad: &mut [f32]| -> f32 {
        evaluator.evaluate(xv, grad)
    };

    // Optimize
    let (result, final_cost, iterations) =
        bfgs_trust_optimize(&mut x[..dim_total], &mut eval_fn, &config.planner.bfgs_trust);

    // Extract solution
    let mut opt_times = [0.0f32; MAX_PIECES];
    let mut opt_wp = [ZERO3; MAX_PIECES];
    for i in 0..n_pieces {
        opt_times[i] = forward_t(x[i]);
    }
    for i in 0..n_wp {
        opt_wp[i] = [
            x[dim_k + 3 * i],
            x[dim_k + 3 * i + 1],
            x[dim_k + 3 * i + 2],
        ];
    }

    // Generate final trajectory
    let mut final_minco = MincoJerk::new(&input.head, &input.tail, n_pieces);
    final_minco.solve(&opt_wp[..n_wp], &opt_times[..n_pieces]);

    PlannerResult {
        trajectory: final_minco.get_trajectory(),
        final_cost,
        iterations,
        status: result.into(),
        optimized_times: opt_times,
        optimized_waypoints: opt_wp,
        num_pieces: n_pieces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::quad_planning_config::QuadPlanningConfig;

    #[test]
    fn plan_goto_converges() {
        let config = QuadPlanningConfig::default();
        let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [3.0, 0.0, 1.0]);
        let result = plan(&input, &config);

        assert!(
            result.status == SolverStatus::Convergence || result.status == SolverStatus::Stop,
            "solver did not converge: {:?}",
            result.status
        );
        assert!(result.final_cost.is_finite());
        assert!(result.num_pieces == 1);

        // Trajectory should start near start_pos and end near target_pos
        let p0 = result.trajectory.get_pos(0.0);
        let pf = result.trajectory.get_pos(result.trajectory.total_duration());
        for d in 0..3 {
            assert!(p0[d].is_finite());
            assert!(pf[d].is_finite());
        }
        assert!((p0[0] - 0.0).abs() < 0.1);
        assert!((pf[0] - 3.0).abs() < 0.5);
    }

    #[test]
    fn plan_waypoints_converges() {
        let config = QuadPlanningConfig::default();
        let targets = [[2.0, 0.0, 1.0], [4.0, 2.0, 1.0], [6.0, 0.0, 1.0]];
        let input = PlannerInput::waypoints([0.0, 0.0, 1.0], ZERO3, &targets);
        let result = plan(&input, &config);

        assert!(
            result.status == SolverStatus::Convergence
                || result.status == SolverStatus::Stop
                || result.status == SolverStatus::MaxIterations,
            "unexpected status: {:?}",
            result.status
        );
        assert!(result.final_cost.is_finite());
        assert_eq!(result.num_pieces, 3);

        // Check trajectory evaluates without NaN at multiple points
        let dur = result.trajectory.total_duration();
        for i in 0..=10 {
            let t = dur * i as f32 / 10.0;
            let p = result.trajectory.get_pos(t);
            let v = result.trajectory.get_vel(t);
            for d in 0..3 {
                assert!(p[d].is_finite(), "NaN at t={t}");
                assert!(v[d].is_finite(), "NaN vel at t={t}");
            }
        }
    }
}
