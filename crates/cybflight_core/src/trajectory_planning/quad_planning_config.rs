use crate::params::PlannerParams;

/// Quadrotor physical + planner parameters used for trajectory optimization.
#[derive(Clone)]
pub struct QuadPlanningConfig {
    /// Vehicle mass [kg].
    pub mass: f32,
    /// Gravitational acceleration [m/s²] (positive, e.g. 9.81).
    pub grav: f32,
    /// Diagonal inertia [Ixx, Iyy, Izz] [kg·m²].
    pub inertia_kg_m2: [f32; 3],
    /// Precomputed 1.0 / mass.
    pub mass_inv: f32,
    /// Maximum collective thrust [N] (sum of all motors).
    pub max_collective_thrust_n: f32,
    /// Minimum collective thrust [N].
    pub min_collective_thrust_n: f32,
    /// Maximum body rate [rad/s] per axis.
    pub max_rate_rad_s: [f32; 3],
    /// Planner tunables (weights, limits, solver params).
    pub planner: PlannerParams,
}

// NOTE: no `Default` — the old impl carried a fourth divergent "default
// vehicle" (mass 0.55, its own inertia set). Construct via
// `from_vehicle_params` (production) or an explicit literal (tests).

impl QuadPlanningConfig {
    /// Construct from firmware vehicle parameters.
    ///
    /// Physical parameters (mass, inertia, thrust bounds) come from
    /// `vp.airframe`, local gravity from `vp.site`. Planner tunables are
    /// copied directly from `vp.trajectory.planner`.
    pub fn from_vehicle_params(vp: &crate::params::FirmwareConfig) -> Self {
        let mass = vp.airframe.body.mass_kg;
        let grav = vp.site.gravity_m_s2;
        let mut max_collective_thrust_n = 0.0_f32;
        for m in &vp.airframe.motors {
            max_collective_thrust_n += m.max_thrust_n;
        }
        // Same diagonal floor as FullQuadModel::from_vehicle_params — the
        // shared array metadata cannot bound diagonal terms separately.
        let inertia_floor = |v: f32| if v.is_finite() && v > 1e-6 { v } else { 1e-6 };
        let inertia_kg_m2 = [
            inertia_floor(vp.airframe.body.inertia_kg_m2[0]), // Ixx
            inertia_floor(vp.airframe.body.inertia_kg_m2[4]), // Iyy
            inertia_floor(vp.airframe.body.inertia_kg_m2[8]), // Izz
        ];
        Self {
            mass,
            grav,
            inertia_kg_m2,
            mass_inv: 1.0 / mass,
            // Same derate the MPC applies (`mpc_thrust_frac`) so planner and
            // controller agree on available thrust; previously the planner
            // assumed the full per-motor sum while the MPC used 0.75×.
            max_collective_thrust_n: max_collective_thrust_n * vp.mpc.thrust_frac,
            // Same 10%-of-hover floor as `QuadModel::from_vehicle_params`
            // (previously a fixed 2.0 N that didn't scale with mass).
            min_collective_thrust_n: mass * grav * 0.1,
            max_rate_rad_s: vp.airframe.body.max_rate_rad_s,
            planner: vp.trajectory.planner.clone(),
        }
    }
}
