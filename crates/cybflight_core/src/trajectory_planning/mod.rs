pub mod banded_system;
pub mod bfgs_trust;
pub mod cost_eval;
pub mod flatness;
pub mod lookahead_yaw;
pub mod minco_acc;
pub mod minco_jerk;
pub mod minco_snap;
pub mod penalties;
pub mod piecewise_polynomial;
pub mod planner;
pub mod polynomial;
pub mod quad_planning_config;
pub mod sampler;
pub mod types;

/// Maximum number of polynomial pieces in a trajectory.
///
/// Sized for the offline-prebaked missions (up to 105 pieces for the
/// planner exports of the outdoor time-optimal tracks, see
/// `crates/cybflight/src/control/offline_mission.rs`). Online BFGS
/// planning is bounded separately by [`MAX_PLANNED_PIECES`] so that
/// the BFGS workspace (whose Hessian is O(N²)) does not balloon when
/// `MAX_PIECES` grows.
pub const MAX_PIECES: usize = 128;

/// Maximum number of polynomial pieces the BFGS planner will ever
/// optimize over. Sizes the BFGS Hessian and `CostEvaluator` scratch
/// arrays. Decoupled from [`MAX_PIECES`] because the offline path
/// stores trajectories far larger than the BFGS solver was tuned for.
pub const MAX_PLANNED_PIECES: usize = 20;

/// The min-jerk solver the BFGS planner embeds, sized to
/// [`MAX_PLANNED_PIECES`] rather than [`MAX_PIECES`] (~8 KiB instead of
/// ~52 KiB — this single instance used to be 95 % of `CostEvaluator`).
pub type PlannerMinco = minco_jerk::MincoJerkN<
    MAX_PLANNED_PIECES,
    { 6 * MAX_PLANNED_PIECES },
    { banded_system::jerk_storage(MAX_PLANNED_PIECES) },
>;
