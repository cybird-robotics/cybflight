pub mod banded_system;
pub mod bfgs_trust;
pub mod minco_jerk;
pub mod piecewise_polynomial;
pub mod polynomial;
pub mod quad_planning_config;
pub mod types;

/// Maximum number of polynomial pieces in a trajectory.
pub const MAX_PIECES: usize = 16;
