pub mod banded_system;
pub mod bfgs_trust;
pub mod cost_eval;
pub mod flatness;
pub mod minco_jerk;
pub mod penalties;
pub mod piecewise_polynomial;
pub mod planner;
pub mod polynomial;
pub mod quad_planning_config;
pub mod types;

/// Maximum number of polynomial pieces in a trajectory.
pub const MAX_PIECES: usize = 20;
