//! Host-side controller implementations.
//!
//! Two controllers are provided:
//!
//! - `MpcController` (default) — `FullSqpSolver` over the 13-state
//!   `FullQuadModel`. Matches the SQP machinery used by the firmware
//!   `outer_loop.rs`, but with per-motor thrust outputs so its commands
//!   feed straight into the plant (no INDI wrapper needed on host). This
//!   is the canonical controller for sim tracking work.
//!
//! - `CascadeController` — position → geometric-attitude → rate-P → mixer,
//!   the PD + feedforward stack from `cybflight-core/tests/control_convergence.rs`.
//!   Kept as a baseline for comparison; the PD position stage is legacy in
//!   the firmware (INDI + MPC is the active path).

use cybflight_core::attitude_control::{
    geometric_controller::GeometricAttitudeController, AttitudeControlSetpoint,
    AttitudeControlState,
};
use cybflight_core::mixer::{LinearAllocator, MotorEffectiveness};
use cybflight_core::mpc::{FullQuadModel, FullQuadProblem, FullSqpSolver, N as MPC_N, NU, NX};
use cybflight_core::params::VehicleParams;
use cybflight_core::position_control::{
    self, pd_ff_control::PositionController, PositionControlSetpoint, PositionControlState,
};
use cybflight_core::trajectory_planning::flatness::reference_quaternion;
use nalgebra::{Quaternion, SVector, UnitQuaternion, Vector3, Vector4};

use crate::trajectory::Setpoint;

/// Controller abstraction.
///
/// The runner samples `horizon_samples()` reference setpoints at stride
/// `horizon_stride_s()` (starting from the current time) and hands them in.
/// Single-shot controllers (cascade) set both to their defaults and just
/// read the first sample; MPC uses the full horizon.
pub trait Controller {
    fn name(&self) -> &'static str;

    /// Number of reference samples the controller wants per tick. Default 1.
    fn horizon_samples(&self) -> usize {
        1
    }

    /// Stride [s] between horizon samples. Default 0 (all samples at `t`).
    fn horizon_stride_s(&self) -> f32 {
        0.0
    }

    fn compute(
        &mut self,
        x: &SVector<f32, NX>,
        horizon: &[Setpoint],
        dt_ctrl: f32,
    ) -> SVector<f32, NU>;
}

// ───────────────────────────────────────────────────────────────────────────
// MPC controller (default)
// ───────────────────────────────────────────────────────────────────────────

/// MPC prediction timestep [s] — must match the model's `dt`.
pub const MPC_DT: f32 = 0.05;
/// Solver execution period [s] (100 Hz, matches firmware `outer_loop.rs`).
pub const MPC_SOLVE_DT: f32 = 0.01;

/// SQP-based MPC controller running `FullSqpSolver` at 100 Hz.
pub struct MpcController {
    solver: Box<FullSqpSolver>,
    problem: FullQuadProblem,
    x_refs: [SVector<f32, NX>; MPC_N + 1],
    u_refs: [SVector<f32, NU>; MPC_N],
    u_warm: [SVector<f32, NU>; MPC_N],
    last_u: SVector<f32, NU>,
    /// Solve every N sim-controller ticks. 1 means solve every tick.
    solve_stride: usize,
    stride_counter: usize,
    grav: f32,
}

impl MpcController {
    pub fn from_params(vp: &VehicleParams) -> Self {
        let mut model = FullQuadModel::from_vehicle_params(vp);
        model.dt = MPC_DT;
        let grav = model.grav;
        let problem = FullQuadProblem::with_rk4(model, MPC_N);

        let hover_per_motor = vp.body.mass_kg * grav / NU as f32;
        let u_ref = SVector::<f32, NU>::from_element(hover_per_motor);

        let mut x_ref = SVector::<f32, NX>::zeros();
        x_ref[6] = 1.0; // qw = 1 (identity quaternion, scalar-last layout)

        Self {
            solver: Box::new(FullSqpSolver::new()),
            problem,
            x_refs: [x_ref; MPC_N + 1],
            u_refs: [u_ref; MPC_N],
            u_warm: [u_ref; MPC_N],
            last_u: u_ref,
            solve_stride: 1,
            stride_counter: 0,
            grav,
        }
    }

    fn fill_reference(&mut self, horizon: &[Setpoint]) {
        debug_assert!(horizon.len() == MPC_N + 1);
        for k in 0..=MPC_N {
            let sp = &horizon[k];
            let acc = [sp.acceleration.x, sp.acceleration.y, sp.acceleration.z];
            let q_ref = reference_quaternion(acc, sp.yaw, self.grav);
            self.x_refs[k][0] = sp.position.x;
            self.x_refs[k][1] = sp.position.y;
            self.x_refs[k][2] = sp.position.z;
            self.x_refs[k][3] = q_ref.i;
            self.x_refs[k][4] = q_ref.j;
            self.x_refs[k][5] = q_ref.k;
            self.x_refs[k][6] = q_ref.w;
            self.x_refs[k][7] = sp.velocity.x;
            self.x_refs[k][8] = sp.velocity.y;
            self.x_refs[k][9] = sp.velocity.z;
            // Body rates (indices 10..13) left at zero — we don't have jerk,
            // and the MPC's own terminal cost drives convergence.
            self.x_refs[k][10] = 0.0;
            self.x_refs[k][11] = 0.0;
            self.x_refs[k][12] = 0.0;
        }
    }
}

impl Controller for MpcController {
    fn name(&self) -> &'static str {
        "mpc"
    }

    fn horizon_samples(&self) -> usize {
        MPC_N + 1
    }

    fn horizon_stride_s(&self) -> f32 {
        MPC_DT
    }

    fn compute(
        &mut self,
        x: &SVector<f32, NX>,
        horizon: &[Setpoint],
        _dt_ctrl: f32,
    ) -> SVector<f32, NU> {
        self.stride_counter += 1;
        if self.stride_counter >= self.solve_stride {
            self.stride_counter = 0;
            self.fill_reference(horizon);

            let _ = self.solver.solve(
                &self.problem,
                x,
                &self.x_refs,
                &self.u_refs,
                &self.u_warm,
                1,
                1e-3,
            );
            let u_bar = self.solver.u_bar();
            let mut u0 = u_bar[0];
            // Clamp to the plant's per-motor envelope.
            for i in 0..NU {
                u0[i] = u0[i].clamp(
                    self.problem.model.u_bounds[i][0],
                    self.problem.model.u_bounds[i][1],
                );
            }
            self.last_u = u0;
            self.u_warm = *u_bar;
        }
        self.last_u
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Cascade controller (legacy baseline)
// ───────────────────────────────────────────────────────────────────────────

pub struct CascadeController {
    pos_ctrl: PositionController<f32>,
    att_ctrl: GeometricAttitudeController<f32>,
    allocator: LinearAllocator<4>,
    rate_kp: Vector3<f32>,
    rate_clamp: Vector3<f32>,
    per_motor_max_n: f32,
    max_collective_n: f32,
    idle_n: f32,
}

impl CascadeController {
    pub fn from_params(vp: &VehicleParams) -> Self {
        let g = &vp.control;
        let per_motor_max = vp.motors.iter().map(|m| m.max_thrust_n).fold(0.0f32, f32::max);
        let max_collective = vp.motors.iter().map(|m| m.max_thrust_n).sum::<f32>();

        let pos_ctrl = PositionController::new(
            Vector3::new(g.pos_kp[0], g.pos_kp[1], g.pos_kp[2]),
            Vector3::new(g.pos_kd[0], g.pos_kd[1], g.pos_kd[2]),
            position_control::VehicleParams {
                mass: vp.body.mass_kg,
                gravity: 9.81,
            },
        );
        let att_ctrl = GeometricAttitudeController::new(
            Vector3::new(g.att_k_rate[0], g.att_k_rate[1], g.att_k_rate[2]),
            Vector3::new(1.0, 1.0, 0.2),
        )
        .with_inertia(vp.body.inertia_matrix());

        let effectiveness = MotorEffectiveness::from_motors(&vp.motors);
        let allocator = LinearAllocator::new(effectiveness);

        Self {
            pos_ctrl,
            att_ctrl,
            allocator,
            rate_kp: Vector3::new(g.rate_kp[0], g.rate_kp[1], g.rate_kp[2]),
            rate_clamp: Vector3::new(0.8, 0.6, 0.15),
            per_motor_max_n: per_motor_max,
            max_collective_n: max_collective,
            idle_n: 0.005 * max_collective,
        }
    }
}

impl Controller for CascadeController {
    fn name(&self) -> &'static str {
        "cascade"
    }

    fn compute(
        &mut self,
        x: &SVector<f32, NX>,
        horizon: &[Setpoint],
        _dt_ctrl: f32,
    ) -> SVector<f32, NU> {
        let sp = &horizon[0];
        let pos = Vector3::new(x[0], x[1], x[2]);
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(x[6], x[3], x[4], x[5]));
        let vel = Vector3::new(x[7], x[8], x[9]);
        let omega = Vector3::new(x[10], x[11], x[12]);

        let pos_out = self.pos_ctrl.compute(
            &PositionControlState {
                position: pos,
                velocity: vel,
                attitude: quat,
            },
            &PositionControlSetpoint {
                position: sp.position,
                velocity: sp.velocity,
                acceleration_ff: sp.acceleration,
                yaw: sp.yaw,
            },
        );

        let att_out = self.att_ctrl.compute(
            &AttitudeControlState {
                attitude_quaternion: quat,
                body_rate_rad_s: omega,
            },
            &AttitudeControlSetpoint {
                attitude_quaternion: Some(pos_out.desired_attitude_quaternion),
                body_rate_rad_s: Vector3::zeros(),
                angular_accel_rad_s2: Vector3::zeros(),
            },
        );

        let rate_err = att_out.body_rate_rad_s - omega;
        let torque = Vector3::new(
            (self.rate_kp.x * rate_err.x).clamp(-self.rate_clamp.x, self.rate_clamp.x),
            (self.rate_kp.y * rate_err.y).clamp(-self.rate_clamp.y, self.rate_clamp.y),
            (self.rate_kp.z * rate_err.z).clamp(-self.rate_clamp.z, self.rate_clamp.z),
        );

        let thrust = pos_out
            .collective_thrust_n
            .max(self.idle_n)
            .min(self.max_collective_n);
        let throttles = self
            .allocator
            .allocate(Vector4::new(thrust, torque.x, torque.y, torque.z));

        Vector4::new(
            throttles[0] * self.per_motor_max_n,
            throttles[1] * self.per_motor_max_n,
            throttles[2] * self.per_motor_max_n,
            throttles[3] * self.per_motor_max_n,
        )
    }
}
