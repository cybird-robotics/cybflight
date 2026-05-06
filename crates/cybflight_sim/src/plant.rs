//! Quadrotor plant: wraps `cybflight_core::mpc::FullQuadModel` RK4 propagation.
//!
//! State layout (NX=13) matches `FullQuadModel`:
//!   [px py pz | qx qy qz qw | vx vy vz | wx wy wz]
//! Control layout (NU=4): per-motor thrust [N], Betaflight QuadX ordering.

use cybflight_core::mixer::{MotorParams, RigidBodyParams, SpinDir};
use cybflight_core::mpc::{FullQuadModel, NU, NX};
use cybflight_core::mpc::quad_model::PosCostMode;
use cybflight_core::params::{
    BfgsTrustParams, ControlGains, IndiControllerParams, IndiEffectivenessParams, LearnerParams,
    MpcParams, PlannerParams, SamplerParams, VehicleParams,
};
use nalgebra::{Quaternion, SVector, UnitQuaternion, Vector3, stack};

/// Canonical host-side vehicle parameters.
///
/// **Schema-stability contract** (see `CLAUDE.md` "Snapshot drift"
/// section): every tuning-sensitive sub-config is constructed here
/// via explicit `Type::new(...)` — never via `Type::default()`.
/// Firmware control authors retune `Default` impls in
/// `cybflight_core::params` (MPC weights, planner caps, INDI rate
/// gains) when flight test reveals a better operating point; the sim
/// must NOT pick those up implicitly, or every retune silently
/// shifts the regression snapshot.
///
/// New tuning fields added upstream will fail to compile this
/// builder, forcing an explicit decision about what the sim should
/// use. To intentionally adopt a new firmware tuning, edit the
/// literals here and regenerate the snapshot in the same commit.
pub const VEHICLE: VehicleParamsBuilder = VehicleParamsBuilder;

pub struct VehicleParamsBuilder;

impl VehicleParamsBuilder {
    pub fn build(&self) -> VehicleParams {
        VehicleParams {
            body: RigidBodyParams {
                mass_kg: 0.55,
                inertia_kg_m2: [0.0021, 0.0, 0.0, 0.0, 0.0018, 0.0, 0.0, 0.0, 0.003],
                max_rate_rad_s: [10.0, 10.0, 6.0],
            },
            motors: [
                MotorParams {
                    position_m: [-0.075, -0.1],
                    spin_dir: SpinDir::Cw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [0.075, -0.1],
                    spin_dir: SpinDir::Ccw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [-0.075, 0.1],
                    spin_dir: SpinDir::Ccw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [0.075, 0.1],
                    spin_dir: SpinDir::Cw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
            ],
            control: ControlGains {
                pos_kp: [4.0, 4.0, 8.0],
                pos_kd: [4.0, 4.0, 6.0],
                att_k_rate: [3.0, 3.0, 1.0],
            },
            // Geometric fallback — no tuning surface; no Default drift.
            indi_effectiveness: IndiEffectivenessParams::zero(),
            indi_controller: IndiControllerParams::new(
                [80.0, 80.0, 80.0],            // rate_gains
                12.0,                           // sync_filter_hz
                [1.0, 1.0, 50.0, 50.0, 50.0, 5.0], // wls_wv
                [1.0, 1.0, 1.0, 1.0],          // wls_wu
                14,                             // motor_pole_count
            ),
            learner: LearnerParams::new(
                20.0,            // fx_filt_hz
                40.0,            // motor_filt_hz
                [0.0, 0.0, 0.0], // acc_offset_m
                100.0,           // rls_gamma
                0.25,            // rls_t_char_s
                0.8,             // zeta_rate
                0.8,             // zeta_attitude
            ),
            mpc: MpcParams::new(
                [500.0, 500.0, 500.0],   // pos_weight
                [10.0, 10.0, 10.0],      // vel_weight
                [5.0, 5.0, 200.0],       // att_weight
                [20.0, 20.0, 20.0],      // rate_weight
                1.0,                     // thrust_weight
                0.05,                    // dt
                1e4,                     // rho
                PosCostMode::Quadratic,  // pos_cost_mode
            ),
            planner: PlannerParams::new(
                100.0,                          // max_vel_m_s
                core::f32::consts::FRAC_PI_3,   // max_tilt_rad
                1.0,                            // weight_time
                0.01,                           // weight_energy
                0.0,                            // weight_pos
                0.0,                            // weight_vel
                0.0,                            // weight_tilt
                10.0,                           // weight_body_rate
                10.0,                           // weight_thrust
                0.01,                           // smoothing_eps
                8,                              // num_check_per_piece
                BfgsTrustParams::default(),
            ),
            sampler: SamplerParams::new(
                0.1,             // max_lag_s
                [1.0, 1.0, 1.0], // axis_weights_sqrt
                0.01,            // search_dt
                100,             // max_search_steps
                0.15,            // radius_of_acceptance
                0.1,             // max_lead_s
            ),
            // Sim doesn't exercise the offline-mission registry; the field
            // exists only to satisfy the firmware's flash schema.
            mission_profile: 0,
            arm_led_enabled: false,
            blackbox_record_set: 0,
        }
    }
}

/// Quadrotor rigid-body simulator.
///
/// Owns the integration timestep and an internal `FullQuadModel` instance
/// configured to match it — the MPC's own `FullQuadModel` (if the controller
/// has one) lives in `CascadeController`/future `MpcController` and runs at a
/// different (coarser) timestep.
pub struct QuadPlant {
    pub dt_sim: f32,
    pub params: VehicleParams,
    model: FullQuadModel,
    state: SVector<f32, NX>,
    time_s: f32,
}

impl QuadPlant {
    pub fn new(params: VehicleParams, dt_sim: f32) -> Self {
        let mut model = FullQuadModel::from_vehicle_params(&params);
        model.dt = dt_sim;

        let mut state = SVector::<f32, NX>::zeros();
        state[6] = 1.0;

        Self {
            dt_sim,
            params,
            model,
            state,
            time_s: 0.0,
        }
    }

    pub fn default_vehicle(dt_sim: f32) -> Self {
        Self::new(VEHICLE.build(), dt_sim)
    }

    /// Overwrite the integrator state. Quaternion is re-normalized.
    pub fn reset(&mut self, pos: Vector3<f32>, vel: Vector3<f32>, attitude: UnitQuaternion<f32>) {
        let q = attitude.into_inner();
        self.state = stack![pos; q.coords; vel; Vector3::zeros()];
        self.time_s = 0.0;
    }

    /// Advance one substep given per-motor thrusts [N]. Thrusts are clamped
    /// to each motor's `[0, max_thrust_n]` bound before integration.
    pub fn step(&mut self, u: &SVector<f32, NU>) {
        let mut u_clamped = *u;
        for (i, f) in u_clamped.iter_mut().enumerate() {
            *f = f.clamp(self.model.u_bounds[i][0], self.model.u_bounds[i][1]);
        }
        self.state = self.model.propagate_rk4(&self.state, &u_clamped);
        self.time_s += self.dt_sim;
    }

    pub fn time_s(&self) -> f32 {
        self.time_s
    }

    pub fn raw_state(&self) -> &SVector<f32, NX> {
        &self.state
    }

    pub fn position(&self) -> Vector3<f32> {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }

    pub fn velocity(&self) -> Vector3<f32> {
        Vector3::new(self.state[7], self.state[8], self.state[9])
    }

    pub fn body_rate(&self) -> Vector3<f32> {
        Vector3::new(self.state[10], self.state[11], self.state[12])
    }

    pub fn attitude(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::from_quaternion(Quaternion::from_vector(
            self.state.fixed_rows::<4>(3).into(),
        ))
    }

    /// Tilt angle between body-z and world-z (radians, ≥0).
    pub fn tilt_rad(&self) -> f32 {
        let q = self.attitude();
        let zb_world_z = q.to_rotation_matrix()[(2, 2)];
        zb_world_z.clamp(-1.0, 1.0).acos()
    }
}
