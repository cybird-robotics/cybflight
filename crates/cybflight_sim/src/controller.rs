//! Host-side controller stacks.
//!
//! Three stacks exist, all implementing the same `Controller` trait. The
//! runner drives them at their declared `tick_rate_hz()`.
//!
//! - `MpcIndiController` (default, firmware-match) — `SimpleSqpSolver` over the
//!   10-state `QuadModel` running at 100 Hz, producing (collective thrust,
//!   body-rate reference). An `IndiController` from `cybflight-core` runs at
//!   8 kHz, tracking the rate reference and emitting per-motor commands.
//!   Motor commands feed straight into the plant as forces (no ESC model).
//!
//! - `MpcDirectController` — `FullSqpSolver` over the 13-state `FullQuadModel`
//!   at 100 Hz, emitting per-motor thrusts directly. Useful as an
//!   upper-bound reference: "what would MPC + a perfect inner loop do?"
//!
//! - `CascadeController` — PD position + geometric attitude + rate-P + mixer,
//!   at 100 Hz. Legacy baseline; PD position control is no longer the
//!   firmware's active path. Kept for diagnostic diffs.

use crate::baselines::{
    geometric_controller::GeometricAttitudeController, AttitudeControlSetpoint,
    AttitudeControlState,
};
use cybflight_core::indi::{
    controller::{IndiConfig, IndiController, MotorState, NU as INDI_NU, NV},
    effectiveness::IndiMotorParams,
    linearization::ThrustModel,
};
use cybflight_core::mixer::{LinearAllocator, MotorEffectiveness};
use cybflight_core::mpc::{
    quad_model::{N as SIMPLE_N, NU as SIMPLE_NU, NX as SIMPLE_NX},
    FullQuadModel, FullQuadProblem, FullSqpSolver, QuadModel, SimpleQuadProblem, SimpleSqpSolver,
    N as FULL_N, NU, NX,
};
use cybflight_core::params::VehicleParams;
use cybflight_core::position_control::{
    self, pd_ff_control::PositionController, PositionControlSetpoint, PositionControlState,
};
use cybflight_core::trajectory_planning::flatness::reference_quaternion;
use nalgebra::{stack, vector, Quaternion, SVector, UnitQuaternion, Vector3, Vector4};

use crate::sensors::ImuMeasurement;
use crate::trajectory::Setpoint;

/// Controller stack abstraction. The runner ticks `step()` at `tick_rate_hz`
/// and supplies a fresh horizon of `horizon_samples` setpoints sampled at
/// stride `horizon_stride_s` starting from the current sim time.
pub trait Controller {
    fn name(&self) -> &'static str;

    /// Fast-loop tick rate for this stack. Runner drives `step()` at this rate.
    fn tick_rate_hz(&self) -> f32;

    /// Setpoint samples this controller wants per tick. Default 1.
    fn horizon_samples(&self) -> usize {
        1
    }

    /// Stride [s] between consecutive horizon samples. Default 0.
    fn horizon_stride_s(&self) -> f32 {
        0.0
    }

    /// One tick. Returns per-motor forces [N] to hand to the plant.
    ///
    /// `x` is ground-truth plant state (stand-in for a perfect global
    /// estimator — will be replaced by ESKF output when sensors-in-the-
    /// loop lands). `imu` is the scenario-provided IMU measurement;
    /// ground-truth controllers ignore it, INDI consumes it.
    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU>;
}

const MPC_SOLVE_RATE_HZ: f32 = 100.0;
const INDI_LOOP_HZ: f32 = 8000.0;
const SIMPLE_MPC_DT: f32 = 0.05;
const FULL_MPC_DT: f32 = 0.05;

// ───────────────────────────────────────────────────────────────────────────
// MPC + INDI stack (default — matches firmware topology)
// ───────────────────────────────────────────────────────────────────────────

/// 10-state MPC at 100 Hz + INDI inner loop at 8 kHz.
///
/// Outer MPC emits `u = [thrust_N, wx_sp, wy_sp, wz_sp]`. INDI consumes
/// `(rate_sp, thrust_sp / mass)` and the synthesized body-frame specific
/// force, and emits per-motor normalized throttles. Plant integrates
/// `per_motor_max_N * throttle` every tick.
pub struct MpcIndiController {
    solver: Box<SimpleSqpSolver>,
    problem: SimpleQuadProblem,
    x_refs: [SVector<f32, SIMPLE_NX>; SIMPLE_N + 1],
    u_refs: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
    u_warm: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
    last_mpc_u: SVector<f32, SIMPLE_NU>,
    indi: IndiController,
    mpc_stride: u32,
    tick_counter: u32,
    mass: f32,
    grav: f32,
    per_motor_max_n: f32,
}

impl MpcIndiController {
    pub fn from_params(vp: &VehicleParams) -> Self {
        // Outer MPC (QuadModel, same as firmware outer_loop.rs)
        let mut model = QuadModel::from_vehicle_params(vp);
        model.dt = SIMPLE_MPC_DT;
        let grav = model.grav;
        let problem = SimpleQuadProblem::with_rk4(model, SIMPLE_N);
        let mass = vp.body.mass_kg;
        let per_motor_max_n = vp
            .motors
            .iter()
            .map(|m| m.max_thrust_n)
            .fold(0.0f32, f32::max);

        let hover_thrust_n = mass * grav;
        let u_ref = vector![hover_thrust_n, 0.0, 0.0, 0.0];

        let mut x_ref = SVector::<f32, SIMPLE_NX>::zeros();
        x_ref[6] = 1.0;

        // Inner INDI (same config as firmware indi_task.rs)
        let ic = &vp.indi_controller;
        // Nearly-linear motor curve (k=0.025 is the hard minimum; plant is linear)
        let nonlinearity_k = 0.025_f32;
        let indi_cfg = IndiConfig {
            rate_gains: Vector3::new(ic.rate_gains[0], ic.rate_gains[1], ic.rate_gains[2]),
            sync_filter_hz: ic.sync_filter_hz,
            rate_dot_sg_window_size: 13,
            rate_dot_sg_order: 2,
            rate_dot_sg_target_rate_hz: INDI_LOOP_HZ,
            motors: vp.motors,
            body: vp.body,
            indi_motors: [IndiMotorParams {
                time_const_s: 0.015,
                max_rpm: 40000.0,
                g2_yaw: 0.0,
            }; INDI_NU],
            thrust_model: ThrustModel::Quadratic,
            nonlinearity: SVector::from_element(nonlinearity_k),
            act_limit: SVector::from_element(1.0),
            wls_wv: SVector::<f32, NV>::from_row_slice(&ic.wls_wv),
            wls_wu: SVector::<f32, INDI_NU>::from_row_slice(&ic.wls_wu),
            wls_cond_bound: 3.2768e8,
            wls_theta: 1e-4,
            wls_imax: 1,
            nan_limit: 20,
            rpm_invalid_limit: 50,
            rpm_all_invalid_limit: 50,
            rpm_recovery_count: 10,
            motor_pole_count: ic.motor_pole_count,
        };
        let indi = IndiController::new(&indi_cfg, INDI_LOOP_HZ);

        let mpc_stride = (INDI_LOOP_HZ / MPC_SOLVE_RATE_HZ).round() as u32;

        Self {
            solver: Box::new(SimpleSqpSolver::new()),
            problem,
            x_refs: [x_ref; SIMPLE_N + 1],
            u_refs: [u_ref; SIMPLE_N],
            u_warm: [u_ref; SIMPLE_N],
            last_mpc_u: u_ref,
            indi,
            mpc_stride,
            tick_counter: 0,
            mass,
            grav,
            per_motor_max_n,
        }
    }

    fn fill_reference(&mut self, horizon: &[Setpoint]) {
        debug_assert!(horizon.len() == SIMPLE_N + 1);
        for k in 0..=SIMPLE_N {
            let sp = &horizon[k];
            let q_ref = reference_quaternion(sp.acceleration, sp.yaw, self.grav);
            self.x_refs[k] = stack![sp.position; q_ref.coords; sp.velocity];
        }
    }

    fn solve_mpc(&mut self, x_full: &SVector<f32, NX>, horizon: &[Setpoint]) {
        // Extract the 10-state slice (position, quaternion, velocity) from
        // the plant's 13-state. The remaining 3 (body rates) are not part of
        // the simple MPC's state.
        let x0: SVector<f32, SIMPLE_NX> = x_full.fixed_rows::<SIMPLE_NX>(0).into_owned();
        self.fill_reference(horizon);

        let _ = self.solver.solve(
            &self.problem,
            &x0,
            &self.x_refs,
            &self.u_refs,
            &self.u_warm,
            1,
            1e-3,
        );
        let u_bar = self.solver.u_bar();
        let u0 = u_bar[0];
        let bounds = self.problem.model.u_bounds;
        self.last_mpc_u =
            SVector::<f32, SIMPLE_NU>::from_fn(|i, _| u0[i].clamp(bounds[i][0], bounds[i][1]));
        self.u_warm = *u_bar;
    }
}

impl Controller for MpcIndiController {
    fn name(&self) -> &'static str {
        "mpc_indi"
    }

    fn tick_rate_hz(&self) -> f32 {
        INDI_LOOP_HZ
    }

    fn horizon_samples(&self) -> usize {
        SIMPLE_N + 1
    }

    fn horizon_stride_s(&self) -> f32 {
        SIMPLE_MPC_DT
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        // Outer MPC solve every `mpc_stride` ticks.
        if self.tick_counter % self.mpc_stride == 0 {
            self.solve_mpc(x, horizon);
        }
        self.tick_counter = self.tick_counter.wrapping_add(1);

        let thrust_sp_n = self.last_mpc_u[0];
        let rate_sp = Vector3::new(self.last_mpc_u[1], self.last_mpc_u[2], self.last_mpc_u[3]);
        // Collective thrust setpoint expressed as specific force on body-z.
        let spf_sp_z = thrust_sp_n / self.mass;

        let g2_valid = [false; INDI_NU]; // no RPM telemetry in sim
        let (out, _step_state) = self.indi.step(
            &imu.gyro,
            &imu.accel,
            &rate_sp,
            spf_sp_z,
            true,
            &g2_valid,
            MotorState::Internal,
        );

        // Convert normalized commands [0,1] → per-motor thrust [N]. Plant's
        // motor model is linear (force = cmd * max_thrust); INDI was configured
        // with near-zero nonlinearity so the two match.
        out.motor_commands * self.per_motor_max_n
    }
}

// ───────────────────────────────────────────────────────────────────────────
// MPC direct-to-motor (diagnostic baseline)
// ───────────────────────────────────────────────────────────────────────────

pub struct MpcDirectController {
    solver: Box<FullSqpSolver>,
    problem: FullQuadProblem,
    x_refs: [SVector<f32, NX>; FULL_N + 1],
    u_refs: [SVector<f32, NU>; FULL_N],
    u_warm: [SVector<f32, NU>; FULL_N],
    last_u: SVector<f32, NU>,
    mpc_stride: u32,
    tick_counter: u32,
    grav: f32,
}

impl MpcDirectController {
    pub fn from_params(vp: &VehicleParams) -> Self {
        let mut model = FullQuadModel::from_vehicle_params(vp);
        model.dt = FULL_MPC_DT;
        let grav = model.grav;
        let problem = FullQuadProblem::with_rk4(model, FULL_N);

        let hover_per_motor = vp.body.mass_kg * grav / NU as f32;
        let u_ref = SVector::<f32, NU>::from_element(hover_per_motor);
        let mut x_ref = SVector::<f32, NX>::zeros();
        x_ref[6] = 1.0;

        let mpc_stride = (MPC_SOLVE_RATE_HZ / MPC_SOLVE_RATE_HZ) as u32; // = 1

        Self {
            solver: Box::new(FullSqpSolver::new()),
            problem,
            x_refs: [x_ref; FULL_N + 1],
            u_refs: [u_ref; FULL_N],
            u_warm: [u_ref; FULL_N],
            last_u: u_ref,
            mpc_stride,
            tick_counter: 0,
            grav,
        }
    }

    fn fill_reference(&mut self, horizon: &[Setpoint]) {
        for k in 0..=FULL_N {
            let sp = &horizon[k];
            let q_ref = reference_quaternion(sp.acceleration, sp.yaw, self.grav);
            self.x_refs[k] = stack![sp.position; q_ref.coords; sp.velocity; Vector3::zeros()];
        }
    }
}

impl Controller for MpcDirectController {
    fn name(&self) -> &'static str {
        "mpc_direct"
    }

    fn tick_rate_hz(&self) -> f32 {
        MPC_SOLVE_RATE_HZ
    }

    fn horizon_samples(&self) -> usize {
        FULL_N + 1
    }

    fn horizon_stride_s(&self) -> f32 {
        FULL_MPC_DT
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        _imu: &ImuMeasurement,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        if self.tick_counter % self.mpc_stride == 0 {
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
            let u0 = u_bar[0];
            let bounds = self.problem.model.u_bounds;
            self.last_u =
                SVector::<f32, NU>::from_fn(|i, _| u0[i].clamp(bounds[i][0], bounds[i][1]));
            self.u_warm = *u_bar;
        }
        self.tick_counter = self.tick_counter.wrapping_add(1);
        self.last_u
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Cascade controller (legacy PD+FF baseline)
// ───────────────────────────────────────────────────────────────────────────

/// Rate-loop P-gains for the legacy cascade baseline. Sim-only — the firmware's
/// active inner loop is INDI and does not carry rate PID gains in `ControlGains`.
const CASCADE_RATE_KP: Vector3<f32> = Vector3::new(0.1, 0.08, 0.05);

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
        let per_motor_max = vp
            .motors
            .iter()
            .map(|m| m.max_thrust_n)
            .fold(0.0f32, f32::max);
        let max_collective = vp.motors.iter().map(|m| m.max_thrust_n).sum::<f32>();

        let pos_ctrl = PositionController::new(
            Vector3::new(g.pos_kp[0], g.pos_kp[1], g.pos_kp[2]),
            Vector3::new(g.pos_kd[0], g.pos_kd[1], g.pos_kd[2]),
            position_control::VehicleParams {
                mass: vp.body.mass_kg,
                gravity: 9.81,
            },
        );
        let att_ctrl =
            GeometricAttitudeController::new(g.att_k_rate.into(), Vector3::new(1.0, 1.0, 0.2))
                .with_inertia(vp.body.inertia_matrix());

        let effectiveness = MotorEffectiveness::from_motors(&vp.motors);
        let allocator = LinearAllocator::new(effectiveness);

        Self {
            pos_ctrl,
            att_ctrl,
            allocator,
            rate_kp: CASCADE_RATE_KP,
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

    fn tick_rate_hz(&self) -> f32 {
        MPC_SOLVE_RATE_HZ
    }

    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        _imu: &ImuMeasurement,
        horizon: &[Setpoint],
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

        throttles * self.per_motor_max_n
    }
}
