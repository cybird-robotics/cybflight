//! Trajectory planner: jointly optimizes waypoint positions and segment times
//! subject to velocity, thrust, tilt, and body rate constraints.
//!
//! Uses MINCO min-jerk representation with BFGS trust-region optimization.
//! Yaw is assumed zero during planning (differential flatness at ψ=0).

#[allow(unused_imports)]
use num_traits::Float;

use super::bfgs_trust::{
    bfgs_trust_init, bfgs_trust_optimize_budgeted, bfgs_trust_resume, BfgsTrustResult,
    BfgsWorkspace,
};
use super::cost_eval::CostEvaluator;
use super::PlannerMinco;
use super::penalties::{backward_t, forward_t};
use super::piecewise_polynomial::PiecewisePolynomial;
use super::quad_planning_config::QuadPlanningConfig;
use super::types::*;
use super::MAX_PLANNED_PIECES;

/// Solver convergence status.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverStatus {
    /// Gradient test met.
    Convergence,
    /// Stopped after progress: cost stagnation or step tolerance. The
    /// trajectory is the best iterate found and is a valid result.
    Stop,
    MaxIterations,
    InvalidValue,
    /// The trust radius collapsed before any step was accepted: the
    /// returned trajectory is the unoptimized seed. Not a solution.
    TrustRegionCollapsed,
    /// The caller's time-budget callback returned `false` before
    /// convergence. The returned trajectory is built from the last
    /// accepted iterate and should typically be discarded — callers
    /// that requested a budget did so because a timely abort was more
    /// important than a valid trajectory.
    TimeExceeded,
}

impl From<BfgsTrustResult> for SolverStatus {
    fn from(r: BfgsTrustResult) -> Self {
        match r {
            BfgsTrustResult::Convergence => Self::Convergence,
            BfgsTrustResult::Stop => Self::Stop,
            BfgsTrustResult::MaxIterations => Self::MaxIterations,
            BfgsTrustResult::InvalidValue => Self::InvalidValue,
            BfgsTrustResult::TrustRegionCollapsed => Self::TrustRegionCollapsed,
            BfgsTrustResult::TimeExceeded => Self::TimeExceeded,
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
    pub waypoints: [Vec3; MAX_PLANNED_PIECES],
    /// Number of intermediate waypoints (pieces = num_waypoints + 1).
    pub num_waypoints: usize,
    /// Initial time allocation per segment (first `num_waypoints + 1` entries valid).
    pub init_times: [f32; MAX_PLANNED_PIECES],
    /// Ball-shape radius for anchoring each waypoint (stereographic
    /// projection, following the C++ reference `Ball::toP` parameterization).
    ///
    /// Each waypoint is constrained to a ball of this radius around its
    /// specified value. Decision variables live in a stereographic chart:
    /// `P = P̂ + 2·r·D / (‖D‖² + 1)`. Regardless of how large `D` grows,
    /// `‖P − P̂‖ ≤ r`.
    ///
    /// Default: 0.01 m — effectively freezes waypoints at their specified
    /// values while keeping BFGS's Hessian approximation well-conditioned.
    /// Override via [`with_ball_radius`](Self::with_ball_radius) for a
    /// looser corridor.
    pub waypoint_radius: f32,
}

/// Default ball-shape radius [m] — tight enough to pin waypoints to their
/// specified values while allowing the optimizer to see smooth gradients.
pub const DEFAULT_WAYPOINT_RADIUS: f32 = 0.01;

/// Smallest ball radius the stereographic chart tolerates. The gradient
/// scale is `s²/r`, so `r = 0` is `0/0` — NaN gradients that the solver then
/// rejects every step of, returning the unoptimized seed. Non-finite or
/// smaller radii are clamped here rather than propagated.
pub const MIN_WAYPOINT_RADIUS: f32 = 1e-4;

/// Clamp a ball radius into the range the stereographic map is defined on.
#[inline]
pub(crate) fn sanitize_radius(r: f32) -> f32 {
    if r.is_finite() {
        r.max(MIN_WAYPOINT_RADIUS)
    } else {
        DEFAULT_WAYPOINT_RADIUS
    }
}

impl PlannerInput {
    /// Build input for a single go-to-position (1 piece, 0 intermediate waypoints).
    ///
    /// Flies directly from `start_pos` to `target_pos` along a single polynomial
    /// segment. Initial segment time is estimated from distance.
    pub fn goto(
        start_pos: impl Into<Vec3>,
        start_vel: impl Into<Vec3>,
        target_pos: impl Into<Vec3>,
    ) -> Self {
        let start_pos = start_pos.into();
        let start_vel = start_vel.into();
        let target_pos = target_pos.into();
        let head: PVA3D = [start_pos, start_vel, ZERO3];
        let tail: PVA3D = [target_pos, ZERO3, ZERO3];
        let dist = (start_pos - target_pos).norm();
        let mut init_times = [0.0f32; MAX_PLANNED_PIECES];
        // Seed segment time generously so the init trajectory stays inside
        // the body-rate and thrust penalty knees. Starting inside an active
        // penalty region has been observed to trap BFGS: its first step
        // *expands* Tᵢ to relieve the penalty, which poisons the Hessian
        // for subsequent compression steps and the trust region collapses.
        init_times[0] = dist.max(1.0);
        Self {
            head,
            tail,
            waypoints: [ZERO3; MAX_PLANNED_PIECES],
            num_waypoints: 0,
            init_times,
            waypoint_radius: DEFAULT_WAYPOINT_RADIUS,
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
    /// Panics if `targets` is empty. At most `MAX_PLANNED_PIECES - 1`
    /// intermediate waypoints are supported; longer inputs are truncated
    /// (the excess intermediate targets are dropped and the path jumps from
    /// the last kept waypoint straight to the tail). That is a debug
    /// assertion, and [`Self::waypoints_truncated`] reports it in release.
    pub fn waypoints(
        start_pos: impl Into<Vec3>,
        start_vel: impl Into<Vec3>,
        targets: &[Vec3],
    ) -> Self {
        assert!(
            !targets.is_empty(),
            "targets must have at least one position"
        );
        let start_pos = start_pos.into();
        let start_vel = start_vel.into();

        let last_idx = targets.len() - 1;
        let tail_pos = targets[last_idx];
        let head: PVA3D = [start_pos, start_vel, ZERO3];
        let tail: PVA3D = [tail_pos, ZERO3, ZERO3];

        let num_intermediate = last_idx.min(MAX_PLANNED_PIECES - 1);
        debug_assert!(
            last_idx == num_intermediate,
            "PlannerInput::waypoints: {} intermediate targets exceed MAX_PLANNED_PIECES-1 = {}; \
             the excess would be silently dropped",
            last_idx,
            MAX_PLANNED_PIECES - 1
        );
        let mut wps = [ZERO3; MAX_PLANNED_PIECES];
        for i in 0..num_intermediate {
            wps[i] = targets[i];
        }

        let num_segments = (num_intermediate + 1).min(MAX_PLANNED_PIECES);
        let mut init_times = [0.0f32; MAX_PLANNED_PIECES];
        let mut prev = start_pos;
        for i in 0..num_segments {
            let next = if i < num_intermediate {
                wps[i]
            } else {
                tail_pos
            };
            // See the comment in `goto`: seed segment time generously so
            // the init trajectory stays inside the body-rate / thrust
            // penalty knees.
            let d = (prev - next).norm();
            init_times[i] = d.max(1.0);
            prev = next;
        }

        Self {
            head,
            tail,
            waypoints: wps,
            num_waypoints: num_intermediate,
            init_times,
            waypoint_radius: DEFAULT_WAYPOINT_RADIUS,
        }
    }

    /// `true` when `targets` passed to [`Self::waypoints`] had more
    /// intermediate points than the planner can hold, i.e. the returned
    /// input does not visit every requested target.
    pub fn waypoints_truncated(targets_len: usize) -> bool {
        targets_len.saturating_sub(1) > MAX_PLANNED_PIECES - 1
    }

    /// Override the ball-shape radius (default: [`DEFAULT_WAYPOINT_RADIUS`]).
    ///
    /// A larger radius lets the optimizer move waypoints within a wider
    /// corridor; a smaller radius pins them harder to the specified values.
    ///
    /// Non-finite or sub-[`MIN_WAYPOINT_RADIUS`] values are clamped to the
    /// minimum — a zero radius would produce NaN gradients (see the constant).
    pub fn with_ball_radius(mut self, r: f32) -> Self {
        self.waypoint_radius = sanitize_radius(r);
        self
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
    pub optimized_times: [f32; MAX_PLANNED_PIECES],
    /// Optimized waypoint positions (first `num_pieces - 1` entries valid).
    pub optimized_waypoints: [Vec3; MAX_PLANNED_PIECES],
    /// Number of polynomial pieces in the trajectory.
    pub num_pieces: usize,
}

/// Run trajectory optimization.
///
/// Jointly optimizes waypoint positions and segment times to minimize a
/// weighted sum of trajectory time, energy (jerk integral), and soft
/// constraint penalties (velocity, thrust, tilt, body rate).
///
/// Allocates a `BfgsWorkspace` (~55 KiB) on the caller's stack, plus the
/// `CostEvaluator`, a second MINCO solver for the final build, and the
/// returned `PlannerResult` (~15 KiB, dominated by its `MAX_PIECES`-sized
/// trajectory). Host/sim entry point — on an embedded task stack use
/// [`plan_with_workspace`], and note that even then the evaluator and
/// result are stack-resident.
pub fn plan(input: &PlannerInput, config: &QuadPlanningConfig) -> PlannerResult {
    let mut ws = BfgsWorkspace::new();
    plan_with_workspace(input, config, &mut ws)
}

/// Run trajectory optimization using a caller-provided BFGS workspace.
///
/// Identical to [`plan`] but lets the caller place the ~55 KiB workspace
/// in static storage (e.g. `static_cell::StaticCell<BfgsWorkspace>`).
/// The `CostEvaluator`, final MINCO solver and `PlannerResult` are still
/// built on the caller's stack. Safe to call repeatedly
/// with the same workspace — `bfgs_trust_optimize` initializes all state
/// it needs from the workspace on entry.
pub fn plan_with_workspace(
    input: &PlannerInput,
    config: &QuadPlanningConfig,
    ws: &mut BfgsWorkspace,
) -> PlannerResult {
    plan_with_workspace_budgeted(input, config, ws, &mut || true)
}

/// Same as [`plan_with_workspace`] but honors a caller-supplied
/// `keep_going` callback. The solver polls it once per outer iteration;
/// returning `false` aborts the solve with
/// [`SolverStatus::TimeExceeded`] and a (possibly sub-optimal) last-
/// accepted iterate in `trajectory`. Intended for wall-clock deadlines
/// enforced by an embedded caller — e.g. a flight controller that
/// cannot afford a planner thread to exceed a watchdog budget.
pub fn plan_with_workspace_budgeted<K>(
    input: &PlannerInput,
    config: &QuadPlanningConfig,
    ws: &mut BfgsWorkspace,
    keep_going: &mut K,
) -> PlannerResult
where
    K: FnMut() -> bool,
{
    let n_wp = input.num_waypoints;
    let n_pieces = n_wp + 1;
    debug_assert!(n_pieces >= 1 && n_pieces <= MAX_PLANNED_PIECES);

    let dim_k = n_pieces;
    let dim_d = 3 * n_wp;
    let dim_total = dim_k + dim_d;

    // Initialize decision vector: x = [K_times, D_stereographic_coords].
    // D = 0 maps to the nominal waypoint via the stereographic projection.
    let mut x = [0.0f32; 4 * MAX_PLANNED_PIECES];
    for i in 0..n_pieces {
        x[i] = backward_t(input.init_times[i]);
    }
    // D is already zero from the array initializer.

    // Build cost evaluator
    let mut evaluator = CostEvaluator::new(
        config,
        n_pieces,
        &input.head,
        &input.tail,
        &input.waypoints,
        input.waypoint_radius,
    );

    // Wrap as closure for the optimizer
    let mut eval_fn = |xv: &[f32], grad: &mut [f32]| -> f32 { evaluator.evaluate(xv, grad) };

    let (result, final_cost, iterations) = bfgs_trust_optimize_budgeted(
        &mut x[..dim_total],
        &mut eval_fn,
        &config.planner.bfgs_trust,
        ws,
        keep_going,
    );

    // Extract solution via stereographic forward map.
    let r = input.waypoint_radius;
    let mut opt_times = [0.0f32; MAX_PLANNED_PIECES];
    let mut opt_wp = [ZERO3; MAX_PLANNED_PIECES];
    for i in 0..n_pieces {
        opt_times[i] = forward_t(x[i]);
    }
    for i in 0..n_wp {
        let dx = x[dim_k + 3 * i];
        let dy = x[dim_k + 3 * i + 1];
        let dz = x[dim_k + 3 * i + 2];
        let norm_sq = dx * dx + dy * dy + dz * dz;
        let s = 2.0 * r / (norm_sq + 1.0);
        opt_wp[i] = input.waypoints[i] + Vec3::new(dx, dy, dz) * s;
    }

    // Generate final trajectory
    let mut final_minco = PlannerMinco::new(&input.head, &input.tail, n_pieces);
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

// ---------------------------------------------------------------------------
// Resumable planner API — lets an async caller interleave solver bursts with
// `yield_now().await` so cooperatively-scheduled peer tasks (GPS, baro, mag)
// are not starved during a long trajectory solve. The one-shot entry points
// above are thin wrappers around this API.
// ---------------------------------------------------------------------------

/// Per-solve state for a resumable trajectory optimization.
///
/// Constructed by [`plan_init`]; advanced by repeated calls to
/// [`plan_resume`]; consumed by [`plan_finalize`] to extract the
/// `PlannerResult`. Holds its own `CostEvaluator` (which owns the
/// inner MINCO solver) so the async caller only has to persist the
/// session between yields — the workspace and session together carry
/// all state the solver needs to pick up where it left off.
pub struct PlanSession {
    /// Decision vector: `[K_times..., D_stereographic...]`, padded to max size.
    pub x: [f32; 4 * MAX_PLANNED_PIECES],
    /// Active decision-vector length (`dim_k + dim_d`).
    pub dim_total: usize,
    /// Number of polynomial pieces.
    pub n_pieces: usize,
    /// Number of intermediate waypoints (`n_pieces - 1`).
    pub n_wp: usize,
    /// Ball-shape stereographic radius, copied from the input.
    pub waypoint_radius: f32,
    /// Nominal waypoint centers, copied from the input for later forward-map.
    pub nominal_waypoints: [Vec3; MAX_PLANNED_PIECES],
    /// Head/tail boundary conditions (for the final MINCO trajectory build).
    pub head: PVA3D,
    pub tail: PVA3D,
    /// Owned cost evaluator (holds the MINCO solver used at every evaluate).
    evaluator: CostEvaluator,
    /// Terminal status once resume returns one; `None` while still running.
    status: Option<SolverStatus>,
}

/// Initialize a resumable trajectory solve.
///
/// Performs the same setup as [`plan_with_workspace_budgeted`] up through
/// (and including) the first cost/gradient evaluation on the initial
/// guess, then returns. Subsequent [`plan_resume`] calls advance the solver.
pub fn plan_init(
    input: &PlannerInput,
    config: &QuadPlanningConfig,
    ws: &mut BfgsWorkspace,
) -> PlanSession {
    let n_wp = input.num_waypoints;
    let n_pieces = n_wp + 1;
    debug_assert!(n_pieces >= 1 && n_pieces <= MAX_PLANNED_PIECES);

    let dim_k = n_pieces;
    let dim_d = 3 * n_wp;
    let dim_total = dim_k + dim_d;

    // Decision vector: x = [K_times, D_stereographic_coords].
    // D = 0 maps to the nominal waypoint via the stereographic projection.
    let mut x = [0.0f32; 4 * MAX_PLANNED_PIECES];
    for i in 0..n_pieces {
        x[i] = backward_t(input.init_times[i]);
    }

    let mut evaluator = CostEvaluator::new(
        config,
        n_pieces,
        &input.head,
        &input.tail,
        &input.waypoints,
        input.waypoint_radius,
    );

    // Seed the solver: evaluate initial cost, set up Hessian, past-f ring.
    let init_result = {
        let mut eval_fn = |xv: &[f32], grad: &mut [f32]| -> f32 { evaluator.evaluate(xv, grad) };
        bfgs_trust_init(
            &mut x[..dim_total],
            &mut eval_fn,
            &config.planner.bfgs_trust,
            ws,
        )
    };

    let status = init_result.map(SolverStatus::from);

    PlanSession {
        x,
        dim_total,
        n_pieces,
        n_wp,
        waypoint_radius: input.waypoint_radius,
        nominal_waypoints: input.waypoints,
        head: input.head,
        tail: input.tail,
        evaluator,
        status,
    }
}

/// Run up to `max_iters_this_call` outer BFGS iterations against the session.
///
/// - Returns `Some(status)` if the solve has reached a terminal status
///   (Convergence, Stop, MaxIterations, InvalidValue, TimeExceeded).
/// - Returns `None` if the iteration budget for this call was exhausted
///   without terminating. The caller should `yield_now().await` (or do
///   whatever equivalent cooperative hand-off its runtime requires) and
///   then call `plan_resume` again to continue.
///
/// `keep_going` is polled once per outer iteration; returning `false`
/// aborts with `TimeExceeded`. Use this for wall-clock deadlines and
/// emergency-disarm interlocks.
pub fn plan_resume<K>(
    session: &mut PlanSession,
    config: &QuadPlanningConfig,
    ws: &mut BfgsWorkspace,
    keep_going: &mut K,
    max_iters_this_call: usize,
) -> Option<SolverStatus>
where
    K: FnMut() -> bool,
{
    // If init already produced a terminal status (e.g. InvalidValue),
    // report it without running another step.
    if let Some(s) = session.status {
        return Some(s);
    }

    // Split-borrow the session so the closure can take `&mut evaluator`
    // while the outer call takes `&mut x`.
    let PlanSession {
        evaluator,
        x,
        dim_total,
        ..
    } = session;

    let mut eval_fn = |xv: &[f32], grad: &mut [f32]| -> f32 { evaluator.evaluate(xv, grad) };

    let result = bfgs_trust_resume(
        &mut x[..*dim_total],
        &mut eval_fn,
        &config.planner.bfgs_trust,
        ws,
        keep_going,
        max_iters_this_call,
    );

    match result {
        Some(r) => {
            let s: SolverStatus = r.into();
            session.status = Some(s);
            Some(s)
        }
        None => None,
    }
}

/// Build the final trajectory from the session's optimized decision vector
/// and return the `PlannerResult`. Consumes the session.
///
/// `status` should be the terminal status returned by `plan_resume`; it is
/// carried into the result verbatim. `ws` is read for the final cost and
/// iteration count captured in workspace state.
pub fn plan_finalize(
    session: PlanSession,
    ws: &BfgsWorkspace,
    status: SolverStatus,
) -> PlannerResult {
    let r = session.waypoint_radius;
    let dim_k = session.n_pieces;

    // Forward stereographic map: D coords → actual waypoint positions.
    let mut opt_times = [0.0f32; MAX_PLANNED_PIECES];
    let mut opt_wp = [ZERO3; MAX_PLANNED_PIECES];
    for i in 0..session.n_pieces {
        opt_times[i] = forward_t(session.x[i]);
    }
    for i in 0..session.n_wp {
        let dx = session.x[dim_k + 3 * i];
        let dy = session.x[dim_k + 3 * i + 1];
        let dz = session.x[dim_k + 3 * i + 2];
        let norm_sq = dx * dx + dy * dy + dz * dz;
        let s = 2.0 * r / (norm_sq + 1.0);
        opt_wp[i] = session.nominal_waypoints[i] + Vec3::new(dx, dy, dz) * s;
    }

    let mut final_minco = PlannerMinco::new(&session.head, &session.tail, session.n_pieces);
    final_minco.solve(&opt_wp[..session.n_wp], &opt_times[..session.n_pieces]);

    PlannerResult {
        trajectory: final_minco.get_trajectory(),
        final_cost: ws.fx(),
        iterations: ws.iter_count(),
        status,
        optimized_times: opt_times,
        optimized_waypoints: opt_wp,
        num_pieces: session.n_pieces,
    }
}

#[cfg(test)]
mod tests {
    use super::super::quad_planning_config::QuadPlanningConfig;
    use super::*;
    use crate::params::PlannerParams;

    /// Legacy test baseline — the values of the deleted
    /// `QuadPlanningConfig::default()`, pinned so planner tests are
    /// independent of any vehicle definition.
    fn test_config() -> QuadPlanningConfig {
        QuadPlanningConfig {
        mass: 0.55,
        grav: 9.81,
        inertia_kg_m2: [0.0025, 0.0021, 0.0043],
        mass_inv: 1.0 / 0.55,
        max_collective_thrust_n: 4.0 * 8.5,
        min_collective_thrust_n: 2.0,
        max_rate_rad_s: [10.0, 10.0, 6.0],
        planner: PlannerParams::default(),
    }
    }

    #[test]
    fn plan_goto_converges() {
        let config = test_config();
        let input = PlannerInput::goto(Vec3::new(0.0, 0.0, 1.0), ZERO3, Vec3::new(3.0, 0.0, 1.0));
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
        let pf = result
            .trajectory
            .get_pos(result.trajectory.total_duration());
        for d in 0..3 {
            assert!(p0[d].is_finite());
            assert!(pf[d].is_finite());
        }
        assert!((p0[0] - 0.0).abs() < 0.1);
        assert!((pf[0] - 3.0).abs() < 0.5);
    }

    #[test]
    fn plan_waypoints_converges() {
        let config = test_config();
        let targets = [
            Vec3::new(2.0, 0.0, 1.0),
            Vec3::new(4.0, 2.0, 1.0),
            Vec3::new(6.0, 0.0, 1.0),
        ];
        let input = PlannerInput::waypoints(Vec3::new(0.0, 0.0, 1.0), ZERO3, &targets);
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

    /// A zero ball radius made the stereographic gradient 0/0 → NaN, and
    /// the solver then returned the unoptimized seed labelled as
    /// `Convergence`. The radius is now floored, so the solve must reach
    /// essentially the same optimum as the default radius.
    #[test]
    fn zero_ball_radius_is_sanitized() {
        let config = test_config();
        let targets = [
            Vec3::new(2.0, 0.0, 1.0),
            Vec3::new(4.0, 2.0, 1.0),
            Vec3::new(6.0, 0.0, 1.0),
        ];
        let start = Vec3::new(0.0, 0.0, 1.0);
        let reference = plan(&PlannerInput::waypoints(start, ZERO3, &targets), &config);
        let zero = plan(
            &PlannerInput::waypoints(start, ZERO3, &targets).with_ball_radius(0.0),
            &config,
        );
        let nan = plan(
            &PlannerInput::waypoints(start, ZERO3, &targets).with_ball_radius(f32::NAN),
            &config,
        );
        for (label, r) in [("r=0", &zero), ("r=NaN", &nan)] {
            assert!(
                matches!(r.status, SolverStatus::Convergence | SolverStatus::Stop),
                "{label}: {:?}",
                r.status
            );
            assert!(r.final_cost.is_finite());
            // Within a few percent of the default-radius optimum — far from
            // the seed, whose cost is ~2.5× larger.
            let rel = (r.final_cost - reference.final_cost).abs() / reference.final_cost;
            assert!(rel < 0.05, "{label}: cost {} vs reference {}", r.final_cost, reference.final_cost);
        }
    }

    #[test]
    fn waypoints_truncation_is_reported() {
        assert!(!PlannerInput::waypoints_truncated(MAX_PLANNED_PIECES));
        assert!(PlannerInput::waypoints_truncated(MAX_PLANNED_PIECES + 1));
    }
}
