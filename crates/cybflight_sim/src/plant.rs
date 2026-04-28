//! Quadrotor plant: wraps `cybflight_core::mpc::FullQuadModel` RK4 propagation.
//!
//! State layout (NX=13) matches `FullQuadModel`:
//!   [px py pz | qx qy qz qw | vx vy vz | wx wy wz]
//! Control layout (NU=4): per-motor thrust [N], Betaflight QuadX ordering.

use cybflight_core::mixer::{MotorParams, RigidBodyParams, SpinDir};
use cybflight_core::mpc::{FullQuadModel, NU, NX};
use cybflight_core::params::{
    ControlGains, IndiControllerParams, IndiEffectivenessParams, LearnerParams, MpcParams,
    PlannerParams, SamplerParams, VehicleParams,
};
use nalgebra::{stack, Quaternion, SVector, UnitQuaternion, Vector3};

/// Canonical host-side vehicle parameters. Mirrors the firmware's
/// `QUADROTOR_BODY` / `QUADROTOR_MOTORS` / `DEFAULT_CONTROL_GAINS` in
/// `crates/cybflight/src/vehicle.rs`. Kept in sync manually — when the
/// firmware numbers change, update this too. See the runner's parity test.
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
            indi_effectiveness: IndiEffectivenessParams::default(),
            indi_controller: IndiControllerParams::default(),
            learner: LearnerParams::default(),
            mpc: MpcParams::default(),
            planner: PlannerParams::default(),
            sampler: SamplerParams::default(),
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
