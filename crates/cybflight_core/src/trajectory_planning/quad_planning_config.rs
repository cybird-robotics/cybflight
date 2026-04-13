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

impl Default for QuadPlanningConfig {
    fn default() -> Self {
        let mass = 0.55;
        Self {
            mass,
            grav: 9.81,
            inertia_kg_m2: [0.0025, 0.0021, 0.0043],
            mass_inv: 1.0 / mass,
            max_collective_thrust_n: 4.0 * 8.5,
            min_collective_thrust_n: 2.0,
            max_rate_rad_s: [10.0, 10.0, 6.0],
            planner: PlannerParams::default(),
        }
    }
}

impl QuadPlanningConfig {
    /// Construct from firmware vehicle parameters.
    ///
    /// Physical parameters (mass, inertia, thrust bounds) come from `VehicleParams`.
    /// Planner tunables are copied directly from `vp.planner`.
    pub fn from_vehicle_params(vp: &crate::params::VehicleParams) -> Self {
        let mass = vp.body.mass_kg;
        let mut max_collective_thrust_n = 0.0_f32;
        for m in &vp.motors {
            max_collective_thrust_n += m.max_thrust_n;
        }
        let inertia_kg_m2 = [
            vp.body.inertia_kg_m2[0], // Ixx
            vp.body.inertia_kg_m2[4], // Iyy
            vp.body.inertia_kg_m2[8], // Izz
        ];
        Self {
            mass,
            grav: 9.81,
            inertia_kg_m2,
            mass_inv: 1.0 / mass,
            max_collective_thrust_n,
            min_collective_thrust_n: 2.0,
            max_rate_rad_s: vp.body.max_rate_rad_s,
            planner: vp.planner.clone(),
        }
    }
}
