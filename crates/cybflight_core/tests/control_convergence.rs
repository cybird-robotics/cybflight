//! Integration test: both cascade (PD+geometric) and MPC controllers
//! converge from [0,0,0] to [1,1,1] using the same rigid-body dynamics.
//!
//! The dynamics simulator is the `FullQuadModel` (RK4).  The cascade
//! controller is the ground truth (verified on hardware) — only the MPC
//! side may be adjusted if a mismatch is found.
//!
//! Run with:
//!   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --test control_convergence --release -- --nocapture

extern crate alloc;

use cybflight_core::attitude_control::{
    geometric_controller::GeometricAttitudeController, AttitudeControlSetpoint,
    AttitudeControlState,
};
use cybflight_core::mixer::{LinearAllocator, MotorEffectiveness, MotorParams, SpinDir};
use cybflight_core::mpc::{FullQuadModel, FullQuadProblem, FullSqpSolver, N, NU, NX};
use cybflight_core::position_control::{
    self, pd_ff_control::PositionController, PositionControlSetpoint, PositionControlState,
};
use nalgebra::{Quaternion, SVector, UnitQuaternion, Vector3, Vector4};

// ═══════════════════════════════════════════════════════════════════════════
// Vehicle parameters — single source of truth, matching vehicle.rs
// ═══════════════════════════════════════════════════════════════════════════

const MASS: f32 = 0.55;
const GRAV: f32 = 9.81;
const INERTIA: [f32; 3] = [0.0025, 0.0021, 0.0043];
const MAX_THRUST_N: f32 = 8.5;

fn test_motors() -> [MotorParams; 4] {
    [
        MotorParams {
            position_m: [-0.075, -0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: MAX_THRUST_N,
            torque_coeff_m: 0.022,
        },
        MotorParams {
            position_m: [0.075, -0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: MAX_THRUST_N,
            torque_coeff_m: 0.022,
        },
        MotorParams {
            position_m: [-0.075, 0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: MAX_THRUST_N,
            torque_coeff_m: 0.022,
        },
        MotorParams {
            position_m: [0.075, 0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: MAX_THRUST_N,
            torque_coeff_m: 0.022,
        },
    ]
}

fn test_quad_model(dt: f32) -> FullQuadModel {
    FullQuadModel {
        mass: MASS,
        grav: GRAV,
        dt,
        inertia: INERTIA.into(),
        ..Default::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dynamics — trait-based for easy future substitution
// ═══════════════════════════════════════════════════════════════════════════

trait QuadDynamics {
    /// Propagate state one timestep given per-motor forces [N].
    fn step(&self, x: &SVector<f32, NX>, u: &SVector<f32, NU>) -> SVector<f32, NX>;
}

/// Full nonlinear rigid-body dynamics (RK4).
struct FullDynamics {
    model: FullQuadModel,
}

impl QuadDynamics for FullDynamics {
    fn step(&self, x: &SVector<f32, NX>, u: &SVector<f32, NU>) -> SVector<f32, NX> {
        // propagate_rk4 already normalizes the result internally.
        self.model.propagate_rk4(x, u)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// State helpers
// ═══════════════════════════════════════════════════════════════════════════

fn make_state(
    pos: SVector<f32, 3>,
    quat_xyzw: UnitQuaternion<f32>,
    vel: SVector<f32, 3>,
    omega: SVector<f32, 3>,
) -> SVector<f32, NX> {
    [
        pos[0],
        pos[1],
        pos[2],
        quat_xyzw.i,
        quat_xyzw.j,
        quat_xyzw.k,
        quat_xyzw.w,
        vel[0],
        vel[1],
        vel[2],
        omega[0],
        omega[1],
        omega[2],
    ]
    .into()
}

fn pos_of(x: &SVector<f32, NX>) -> Vector3<f32> {
    Vector3::new(x[0], x[1], x[2])
}

/// Generate 50 deterministic initial orientations with tilts up to ±30°.
///
/// Uses a golden-angle spiral on the 2-sphere for the rotation axis
/// combined with a varying tilt magnitude.
fn generate_initial_orientations(count: usize) -> Vec<UnitQuaternion<f32>> {
    let max_tilt_rad = 30.0_f32.to_radians();
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut quats = Vec::with_capacity(count);

    for i in 0..count {
        let t = (i as f32 + 0.5) / count as f32; // ∈ (0, 1)

        // Fibonacci sphere for axis direction
        let phi = 2.0 * core::f32::consts::PI * (i as f32) / golden_ratio;
        let cos_theta = 1.0 - 2.0 * t;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let axis = Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

        // Tilt magnitude: sweep from small to max
        let angle = max_tilt_rad * (0.2 + 0.8 * t); // 6°..30°

        let q = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle);
        quats.push(q);
    }
    quats
}

// ═══════════════════════════════════════════════════════════════════════════
// Cascade controller wrapper
// ═══════════════════════════════════════════════════════════════════════════

struct CascadeController {
    pos_ctrl: PositionController<f32>,
    att_ctrl: GeometricAttitudeController<f32>,
    allocator: LinearAllocator<4>,
    rate_kp: Vector3<f32>,
    max_thrust: f32,
}

impl CascadeController {
    fn new() -> Self {
        let motors = test_motors();
        let effectiveness = MotorEffectiveness::from_motors(&motors);
        let allocator = LinearAllocator::new(effectiveness);

        let pos_ctrl = PositionController::new(
            Vector3::new(4.0, 4.0, 8.0),
            Vector3::new(4.0, 4.0, 6.0),
            position_control::VehicleParams {
                mass: MASS as f32,
                gravity: GRAV as f32,
            },
        );

        let att_ctrl = GeometricAttitudeController::new(
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(1.0, 1.0, 0.2),
        )
        .with_inertia(nalgebra::Matrix3::from_diagonal(&Vector3::new(
            INERTIA[0] as f32,
            INERTIA[1] as f32,
            INERTIA[2] as f32,
        )));

        Self {
            pos_ctrl,
            att_ctrl,
            allocator,
            rate_kp: Vector3::new(0.1, 0.08, 0.05),
            max_thrust: 4.0 * MAX_THRUST_N,
        }
    }

    fn compute(&self, x: &SVector<f32, NX>, target: &Vector3<f32>) -> SVector<f32, NU> {
        let pos = Vector3::new(x[0] as f32, x[1] as f32, x[2] as f32);
        let vel = Vector3::new(x[7] as f32, x[8] as f32, x[9] as f32);
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            x[6] as f32,
            x[3] as f32,
            x[4] as f32,
            x[5] as f32,
        ));
        let omega = Vector3::new(x[10] as f32, x[11] as f32, x[12] as f32);

        // 1. Position controller → desired attitude + thrust
        let pos_out = self.pos_ctrl.compute(
            &PositionControlState {
                position: pos,
                velocity: vel,
                attitude: quat,
            },
            &PositionControlSetpoint {
                position: *target,
                velocity: Vector3::zeros(),
                acceleration_ff: Vector3::zeros(),
                yaw: 0.0,
            },
        );

        // 2. Attitude controller → rate setpoint
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

        // 3. Rate P controller → torque (clamped)
        let rate_error = att_out.body_rate_rad_s - omega;
        let torque = Vector3::new(
            (self.rate_kp.x * rate_error.x).clamp(-0.8, 0.8),
            (self.rate_kp.y * rate_error.y).clamp(-0.6, 0.6),
            (self.rate_kp.z * rate_error.z).clamp(-0.15, 0.15),
        );

        // 4. Mixer → throttles [0,1] → forces
        let idle_n = 0.005 * self.max_thrust;
        let thrust = pos_out.collective_thrust_n.max(idle_n).min(self.max_thrust);
        let throttles = self
            .allocator
            .allocate(Vector4::new(thrust, torque.x, torque.y, torque.z));

        Vector4::new(
            (throttles[0] * MAX_THRUST_N) as f32,
            (throttles[1] * MAX_THRUST_N) as f32,
            (throttles[2] * MAX_THRUST_N) as f32,
            (throttles[3] * MAX_THRUST_N) as f32,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MPC controller wrapper (runs at MPC_DT, holds control between solves)
// ═══════════════════════════════════════════════════════════════════════════

struct MpcController {
    solver: alloc::boxed::Box<FullSqpSolver>,
    problem: FullQuadProblem,
    x_refs: [SVector<f32, NX>; N + 1],
    u_refs: [SVector<f32, NU>; N],
    u_warm: [SVector<f32, NU>; N],
    last_u: SVector<f32, NU>,
    /// Sim steps between MPC solves.
    solve_period: usize,
    step_counter: usize,
}

/// Prediction timestep (horizon resolution).
const MPC_DT: f32 = 0.05;
/// Solver execution period — 100 Hz (10 ms), faster than 1/MPC_DT.
const MPC_SOLVE_DT: f32 = 0.01;

impl MpcController {
    fn new(target: &Vector3<f32>, sim_dt: f32) -> Self {
        let model = test_quad_model(MPC_DT);
        let problem = FullQuadProblem::with_rk4(model, N);

        let hover_per_motor = MASS * GRAV / 4.0;
        let u_ref = SVector::<f32, NU>::from_element(hover_per_motor);

        let mut x_ref = SVector::<f32, NX>::zeros();
        x_ref[0] = target.x;
        x_ref[1] = target.y;
        x_ref[2] = target.z;
        x_ref[6] = 1.0;

        let solve_period = (MPC_SOLVE_DT / sim_dt).round() as usize;

        Self {
            solver: alloc::boxed::Box::new(FullSqpSolver::new()),
            problem,
            x_refs: [x_ref; N + 1],
            u_refs: [u_ref; N],
            u_warm: [u_ref; N],
            last_u: u_ref,
            solve_period,
            step_counter: 0, // solve on the very first call
        }
    }

    fn compute(&mut self, x: &SVector<f32, NX>) -> SVector<f32, NU> {
        self.step_counter += 1;
        if self.step_counter >= self.solve_period {
            self.step_counter = 0;

            let _result = self.solver.solve(
                &self.problem,
                x,
                &self.x_refs,
                &self.u_refs,
                &self.u_warm,
                1,
                1e-3,
            );

            let u_bar = self.solver.u_bar();
            self.last_u = u_bar[0];

            // No time-shift: solver runs at 100 Hz, faster than 1/dt (20 Hz).
            self.u_warm = *u_bar;
        }

        self.last_u
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Controller enum — switchable via flag
// ═══════════════════════════════════════════════════════════════════════════

enum Controller {
    Cascade(CascadeController),
    Mpc(MpcController),
}

impl Controller {
    fn compute(&mut self, x: &SVector<f32, NX>, target: &Vector3<f32>) -> SVector<f32, NU> {
        match self {
            Controller::Cascade(c) => {
                let t32 = Vector3::new(target.x as f32, target.y as f32, target.z as f32);
                c.compute(x, &t32)
            }
            Controller::Mpc(c) => c.compute(x),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simulation loop
// ═══════════════════════════════════════════════════════════════════════════

struct SimResult {
    final_pos: Vector3<f32>,
    pos_error: f32,
    converged: bool,
    steps: usize,
}

fn run_simulation(
    controller: &mut Controller,
    dynamics: &dyn QuadDynamics,
    x0: &SVector<f32, NX>,
    target: &Vector3<f32>,
    max_steps: usize,
    tol: f32,
) -> SimResult {
    let mut x = *x0;

    let mut converge_count = 0usize;
    let converge_window = 100; // 100 × 0.002 s = 0.2 s sustained convergence

    for step in 0..max_steps {
        let u = controller.compute(&x, target);

        // Clamp motor forces to physical bounds
        let mut u_clamped = u;
        for f in u_clamped.iter_mut() {
            *f = f.clamp(0.0, MAX_THRUST_N as f32);
        }

        x = dynamics.step(&x, &u_clamped);

        let pos = pos_of(&x);
        let err = (pos - target).norm();

        if err < tol {
            converge_count += 1;
        } else {
            converge_count = 0;
        }

        if converge_count >= converge_window {
            return SimResult {
                final_pos: pos,
                pos_error: err,
                converged: true,
                steps: step + 1,
            };
        }

        if pos.norm() > 100.0 || !pos.x.is_finite() {
            break;
        }
    }

    let pos = pos_of(&x);
    SimResult {
        final_pos: pos,
        pos_error: (pos - target).norm(),
        converged: false,
        steps: max_steps,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test constants
// ═══════════════════════════════════════════════════════════════════════════

const SIM_DT: f32 = 0.002; // 500 Hz simulation
const MAX_TIME_S: f32 = 15.0; // 15 s max
const MAX_STEPS: usize = (MAX_TIME_S / SIM_DT) as usize;
const POS_TOL: f32 = 0.15; // convergence tolerance [m]
const NUM_ORIENTATIONS: usize = 50;

fn make_dynamics() -> FullDynamics {
    FullDynamics {
        model: test_quad_model(SIM_DT),
    }
}

/// Build an initial state at the origin with a given orientation.
fn initial_state(q: &UnitQuaternion<f32>) -> SVector<f32, NX> {
    make_state(Vector3::zeros(), *q, Vector3::zeros(), Vector3::zeros())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cascade_converges_50_orientations() {
    let target = Vector3::new(1.0, 1.0, 1.0);
    let dynamics = make_dynamics();
    let orientations = generate_initial_orientations(NUM_ORIENTATIONS);

    let mut failures = Vec::new();

    for (i, q) in orientations.iter().enumerate() {
        let x0 = initial_state(q);
        let mut ctrl = Controller::Cascade(CascadeController::new());
        let result = run_simulation(&mut ctrl, &dynamics, &x0, &target, MAX_STEPS, POS_TOL);

        let angle_deg = q.angle().to_degrees();
        if result.converged {
            println!(
                "  Cascade [{:2}] tilt={:5.1}° → converged in {:5} steps ({:.2}s), err={:.4}m",
                i,
                angle_deg,
                result.steps,
                result.steps as f32 * SIM_DT,
                result.pos_error,
            );
        } else {
            println!(
                "  Cascade [{:2}] tilt={:5.1}° → FAILED, err={:.4}m, pos=({:.2},{:.2},{:.2})",
                i,
                angle_deg,
                result.pos_error,
                result.final_pos.x,
                result.final_pos.y,
                result.final_pos.z,
            );
            failures.push(i);
        }
    }

    assert!(
        failures.is_empty(),
        "Cascade controller failed for {} / {} orientations: {:?}",
        failures.len(),
        NUM_ORIENTATIONS,
        failures,
    );
}

#[test]
fn mpc_converges_50_orientations() {
    let target = Vector3::new(1.0, 1.0, 1.0);
    let dynamics = make_dynamics();
    let orientations = generate_initial_orientations(NUM_ORIENTATIONS);

    let mut failures = Vec::new();

    for (i, q) in orientations.iter().enumerate() {
        let x0 = initial_state(q);
        let mut ctrl = Controller::Mpc(MpcController::new(&target, SIM_DT));
        let result = run_simulation(&mut ctrl, &dynamics, &x0, &target, MAX_STEPS, POS_TOL);

        let angle_deg = q.angle().to_degrees();
        if result.converged {
            println!(
                "  MPC    [{:2}] tilt={:5.1}° → converged in {:5} steps ({:.2}s), err={:.4}m",
                i,
                angle_deg,
                result.steps,
                result.steps as f32 * SIM_DT,
                result.pos_error,
            );
        } else {
            println!(
                "  MPC    [{:2}] tilt={:5.1}° → FAILED, err={:.4}m, pos=({:.2},{:.2},{:.2})",
                i,
                angle_deg,
                result.pos_error,
                result.final_pos.x,
                result.final_pos.y,
                result.final_pos.z,
            );
            failures.push(i);
        }
    }

    assert!(
        failures.is_empty(),
        "MPC controller failed for {} / {} orientations: {:?}",
        failures.len(),
        NUM_ORIENTATIONS,
        failures,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Allocation matrix equivalence test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mpc_alloc_matches_firmware_mixer() {
    let model = test_quad_model(0.01);
    let motors = test_motors();
    let effectiveness = MotorEffectiveness::from_motors(&motors);

    let forces = Vector4::new(2.0, 3.0, 1.5, 4.0);
    let (f_total, tau_x, tau_y, tau_z) = model.alloc(&forces);

    let throttles = nalgebra::SVector::<f32, 4>::new(
        forces[0] as f32 / MAX_THRUST_N,
        forces[1] as f32 / MAX_THRUST_N,
        forces[2] as f32 / MAX_THRUST_N,
        forces[3] as f32 / MAX_THRUST_N,
    );
    let v = effectiveness.g1 * throttles;

    let eps = 1e-6;
    assert!(
        (f_total - v[0] as f32).abs() < eps,
        "thrust mismatch: mpc={f_total}, mixer={}",
        v[0]
    );
    assert!(
        (tau_x - v[1] as f32).abs() < eps,
        "roll mismatch: mpc={tau_x}, mixer={}",
        v[1]
    );
    assert!(
        (tau_y - v[2] as f32).abs() < eps,
        "pitch mismatch: mpc={tau_y}, mixer={}",
        v[2]
    );
    assert!(
        (tau_z - v[3] as f32).abs() < eps,
        "yaw mismatch: mpc={tau_z}, mixer={}",
        v[3]
    );

    println!(
        "Allocation match verified: thrust={f_total:.3}, τ=[{tau_x:.4}, {tau_y:.4}, {tau_z:.4}]"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-call runtime benchmark — FullQuadModel vs QuadModel
// ═══════════════════════════════════════════════════════════════════════════
//
// Measures wall-clock time for `solver.solve(...)` (one full SQP iteration:
// linearise + cost evaluation + backward sweep + forward sweep + re-rollout)
// for both monomorphisations of `SqpSolver`. Run with --release for
// meaningful numbers; debug-mode timings are not representative of the
// embedded target's behaviour.

#[test]
fn bench_solve_runtime() {
    use std::hint::black_box;
    use std::time::Instant;

    use cybflight_core::mpc::quad_model::{N as SIMPLE_N, NU as SIMPLE_NU, NX as SIMPLE_NX};
    use cybflight_core::mpc::{QuadModel, SimpleQuadProblem, SimpleSqpSolver};

    const WARMUP: usize = 200;
    const MEASURE: usize = 5_000;

    let target = Vector3::new(1.0, 1.0, 1.0);

    // ── FullQuadModel (NX=13, NU=4) ────────────────────────────────────────
    let full_model = test_quad_model(0.05);
    let full_problem = FullQuadProblem::with_rk4(full_model, N);
    let mut full_solver = alloc::boxed::Box::new(FullSqpSolver::new());

    let hover_per_motor = MASS * GRAV / 4.0;
    let full_u_ref = SVector::<f32, NU>::from_element(hover_per_motor);
    let mut full_x_ref = SVector::<f32, NX>::from_element(0.0f32);
    full_x_ref[0] = target.x;
    full_x_ref[1] = target.y;
    full_x_ref[2] = target.z;
    full_x_ref[6] = 1.0; // identity quaternion
    let full_x_refs = [full_x_ref; N + 1];
    let full_u_refs = [full_u_ref; N];
    let full_u_warm = [full_u_ref; N];

    let mut full_x0 = SVector::<f32, NX>::from_element(0.0f32);
    full_x0[6] = 1.0; // identity quaternion at origin

    // Warm up (cache-priming, CPU frequency scaling)
    for _ in 0..WARMUP {
        let _ = black_box(full_solver.solve(
            &full_problem,
            &full_x0,
            &full_x_refs,
            &full_u_refs,
            &full_u_warm,
            1,
            1e-3,
        ));
    }
    // Measurement
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let _ = black_box(full_solver.solve(
            &full_problem,
            &full_x0,
            &full_x_refs,
            &full_u_refs,
            &full_u_warm,
            1,
            1e-3,
        ));
    }
    let full_elapsed = t0.elapsed();
    let full_per_call_us = full_elapsed.as_nanos() as f64 / 1000.0 / MEASURE as f64;

    // ── QuadModel (NX=10, NU=4) ────────────────────────────────────────────
    let simple_model = QuadModel {
        dt: 0.05,
        ..Default::default()
    };
    let simple_problem = SimpleQuadProblem::with_rk4(simple_model, SIMPLE_N);
    let mut simple_solver = alloc::boxed::Box::new(SimpleSqpSolver::new());

    // Hover input: collective thrust = m·g, zero body-rate command
    let simple_u_ref = SVector::<f32, { SIMPLE_NU }>::from_row_slice(&[MASS * GRAV, 0.0, 0.0, 0.0]);
    let mut simple_x_ref = SVector::<f32, { SIMPLE_NX }>::zeros();
    simple_x_ref[0] = target.x;
    simple_x_ref[1] = target.y;
    simple_x_ref[2] = target.z;
    simple_x_ref[6] = 1.0; // identity quaternion
    let simple_x_refs = [simple_x_ref; SIMPLE_N + 1];
    let simple_u_refs = [simple_u_ref; SIMPLE_N];
    let simple_u_warm = [simple_u_ref; SIMPLE_N];

    let mut simple_x0 = SVector::<f32, SIMPLE_NX>::zeros();
    simple_x0[6] = 1.0;

    for _ in 0..WARMUP {
        let _ = black_box(simple_solver.solve(
            &simple_problem,
            &simple_x0,
            &simple_x_refs,
            &simple_u_refs,
            &simple_u_warm,
            1,
            1e-3,
        ));
    }
    let t1 = Instant::now();
    for _ in 0..MEASURE {
        let _ = black_box(simple_solver.solve(
            &simple_problem,
            &simple_x0,
            &simple_x_refs,
            &simple_u_refs,
            &simple_u_warm,
            1,
            1e-3,
        ));
    }
    let simple_elapsed = t1.elapsed();
    let simple_per_call_us = simple_elapsed.as_nanos() as f64 / 1000.0 / MEASURE as f64;

    // ── Report ─────────────────────────────────────────────────────────────
    let speedup = full_per_call_us / simple_per_call_us;
    println!();
    println!("─────────────────────────────────────────────────────────────");
    println!(
        "MPC solver per-call runtime ({} iters after {} warm-up)",
        MEASURE, WARMUP
    );
    println!("─────────────────────────────────────────────────────────────");
    println!(
        "  FullQuadModel  (NX={}, NU={}, N={}): {:>7.2} µs/call  →  {:>6.1} kHz max rate",
        NX,
        NU,
        N,
        full_per_call_us,
        1000.0 / full_per_call_us
    );
    println!(
        "  QuadModel      (NX={}, NU={}, N={}): {:>7.2} µs/call  →  {:>6.1} kHz max rate",
        SIMPLE_NX,
        SIMPLE_NU,
        SIMPLE_N,
        simple_per_call_us,
        1000.0 / simple_per_call_us
    );
    println!("  Speedup (full / simple):           {:>7.2}×", speedup);
    println!("─────────────────────────────────────────────────────────────");

    // Sanity assertions: both must complete with positive elapsed time.
    assert!(
        full_per_call_us > 0.0,
        "full solver timing must be positive"
    );
    assert!(
        simple_per_call_us > 0.0,
        "simple solver timing must be positive"
    );
}

// ─── 10-case benchmark with non-trivial tilts + velocities ─────────────────
//
// Same per-call timing methodology as `bench_solve_runtime`, but instead of a
// single hover state, we run 10 distinct initial conditions covering a range
// of tilt magnitudes (5°–50°), tilt axes (pure roll/pitch/yaw + mixed +
// inclined), and velocity vectors. This stresses the solver across a much
// wider portion of its state-space and shows whether per-call cost varies
// with linearization point (it shouldn't — the solver does fixed work per
// call — but the test confirms that empirically).

#[test]
fn bench_solve_runtime_varied() {
    use std::hint::black_box;
    use std::time::Instant;

    use cybflight_core::mpc::quad_model::{N as SIMPLE_N, NU as SIMPLE_NU, NX as SIMPLE_NX};
    use cybflight_core::mpc::{QuadModel, SimpleQuadProblem, SimpleSqpSolver};
    use nalgebra::Unit;

    const WARMUP: usize = 100;
    const MEASURE: usize = 2_000;

    // 10 non-trivial initial conditions: (axis, tilt_rad, velocity).
    // Spans 5°–50° tilt magnitude, varied axes, varied velocity directions.
    let cases: [(Vector3<f32>, f32, Vector3<f32>); 10] = [
        // 0: pure roll, small tilt + small +x velocity
        (
            Vector3::new(1.0, 0.0, 0.0),
            5.0_f32.to_radians(),
            Vector3::new(0.5, 0.0, 0.0),
        ),
        // 1: pure pitch, small tilt + small +y velocity
        (
            Vector3::new(0.0, 1.0, 0.0),
            10.0_f32.to_radians(),
            Vector3::new(0.0, 0.5, 0.0),
        ),
        // 2: pure roll, medium tilt + medium velocity
        (
            Vector3::new(1.0, 0.0, 0.0),
            15.0_f32.to_radians(),
            Vector3::new(1.0, 0.0, 0.3),
        ),
        // 3: pure pitch, medium tilt + larger velocity
        (
            Vector3::new(0.0, 1.0, 0.0),
            20.0_f32.to_radians(),
            Vector3::new(0.0, 1.5, -0.3),
        ),
        // 4: yaw + horizontal velocity
        (
            Vector3::new(0.0, 0.0, 1.0),
            30.0_f32.to_radians(),
            Vector3::new(0.5, 0.5, 0.0),
        ),
        // 5: mixed roll/pitch
        (
            Vector3::new(1.0, 1.0, 0.0).normalize(),
            25.0_f32.to_radians(),
            Vector3::new(1.0, 1.0, 0.5),
        ),
        // 6: asymmetric mixed roll/pitch + bigger velocity
        (
            Vector3::new(1.0, 0.5, 0.0).normalize(),
            35.0_f32.to_radians(),
            Vector3::new(2.0, 1.0, 0.0),
        ),
        // 7: fully 3D axis + 3D velocity
        (
            Vector3::new(1.0, 1.0, 1.0).normalize(),
            40.0_f32.to_radians(),
            Vector3::new(1.5, -0.5, 0.5),
        ),
        // 8: oblique axis + reverse velocity
        (
            Vector3::new(0.5, -1.0, 0.5).normalize(),
            45.0_f32.to_radians(),
            Vector3::new(-1.0, 1.5, 1.0),
        ),
        // 9: extreme tilt + large velocity
        (
            Vector3::new(1.0, -1.0, 0.5).normalize(),
            50.0_f32.to_radians(),
            Vector3::new(2.0, -1.0, -0.5),
        ),
    ];

    // ── Build full-model solver/problem/refs (reused across all cases) ────
    let target = Vector3::new(1.0, 1.0, 1.0);

    let full_model = test_quad_model(0.05);
    let full_problem = FullQuadProblem::with_rk4(full_model, N);
    let mut full_solver = alloc::boxed::Box::new(FullSqpSolver::new());

    let hover_per_motor = MASS * GRAV / 4.0;
    let full_u_ref = SVector::<f32, NU>::from_element(hover_per_motor);
    let mut full_x_ref = SVector::<f32, NX>::zeros();
    full_x_ref[0] = target.x;
    full_x_ref[1] = target.y;
    full_x_ref[2] = target.z;
    full_x_ref[6] = 1.0;
    let full_x_refs = [full_x_ref; N + 1];
    let full_u_refs = [full_u_ref; N];
    let full_u_warm_init = [full_u_ref; N];

    // ── Build simple-model solver/problem/refs ────────────────────────────
    let simple_model = QuadModel {
        dt: 0.05,
        ..Default::default()
    };
    let simple_problem = SimpleQuadProblem::with_rk4(simple_model, SIMPLE_N);
    let mut simple_solver = alloc::boxed::Box::new(SimpleSqpSolver::new());

    let simple_u_ref = SVector::<f32, SIMPLE_NU>::from_row_slice(&[MASS * GRAV, 0.0, 0.0, 0.0]);
    let mut simple_x_ref = SVector::<f32, SIMPLE_NX>::zeros();
    simple_x_ref[0] = target.x;
    simple_x_ref[1] = target.y;
    simple_x_ref[2] = target.z;
    simple_x_ref[6] = 1.0;
    let simple_x_refs = [simple_x_ref; SIMPLE_N + 1];
    let simple_u_refs = [simple_u_ref; SIMPLE_N];
    let simple_u_warm_init = [simple_u_ref; SIMPLE_N];

    // Per-case timing storage
    let mut full_times = [0.0f64; 10];
    let mut simple_times = [0.0f64; 10];

    println!();
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!(
        "MPC solve() per-call runtime — 10 non-trivial initial states ({} measure iters)",
        MEASURE
    );
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:>3}  {:>7}  {:>22}  {:>9}  {:>9}  {:>7}",
        "#", "tilt[°]", "velocity[m/s]", "full[µs]", "simple[µs]", "ratio"
    );
    println!("─────────────────────────────────────────────────────────────────────────────");

    for (i, (axis, angle, vel)) in cases.iter().enumerate() {
        let q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(*axis), *angle);

        // Build initial states for both models from the same physical pose.
        let full_x0 = SVector::<f32, NX>::from_row_slice(&[
            0.0, 0.0, 0.0, // position
            q.i, q.j, q.k, q.w, // quaternion (xyzw)
            vel.x, vel.y, vel.z, // velocity
            0.0, 0.0, 0.0, // body rates (state in full model)
        ]);
        let simple_x0 = SVector::<f32, SIMPLE_NX>::from_row_slice(&[
            0.0, 0.0, 0.0, // position
            q.i, q.j, q.k, q.w, // quaternion (xyzw)
            vel.x, vel.y, vel.z, // velocity
        ]);

        // Reset warm-start each case for a fair "cold from hover" measurement.
        let mut full_u_warm = full_u_warm_init;
        let mut simple_u_warm = simple_u_warm_init;

        // ── FullQuadModel timing ──
        for _ in 0..WARMUP {
            let _ = black_box(full_solver.solve(
                &full_problem,
                &full_x0,
                &full_x_refs,
                &full_u_refs,
                &full_u_warm,
                1,
                1e-3,
            ));
            full_u_warm = *full_solver.u_bar();
        }
        let t0 = Instant::now();
        for _ in 0..MEASURE {
            let _ = black_box(full_solver.solve(
                &full_problem,
                &full_x0,
                &full_x_refs,
                &full_u_refs,
                &full_u_warm,
                1,
                1e-3,
            ));
            full_u_warm = *full_solver.u_bar();
        }
        let full_us = t0.elapsed().as_nanos() as f64 / 1000.0 / MEASURE as f64;

        // ── QuadModel timing ──
        for _ in 0..WARMUP {
            let _ = black_box(simple_solver.solve(
                &simple_problem,
                &simple_x0,
                &simple_x_refs,
                &simple_u_refs,
                &simple_u_warm,
                1,
                1e-3,
            ));
            simple_u_warm = *simple_solver.u_bar();
        }
        let t1 = Instant::now();
        for _ in 0..MEASURE {
            let _ = black_box(simple_solver.solve(
                &simple_problem,
                &simple_x0,
                &simple_x_refs,
                &simple_u_refs,
                &simple_u_warm,
                1,
                1e-3,
            ));
            simple_u_warm = *simple_solver.u_bar();
        }
        let simple_us = t1.elapsed().as_nanos() as f64 / 1000.0 / MEASURE as f64;

        full_times[i] = full_us;
        simple_times[i] = simple_us;

        println!(
            "{:>3}  {:>6.1}°  ({:>5.2},{:>5.2},{:>5.2})  {:>9.2}  {:>9.2}  {:>5.2}×",
            i,
            angle.to_degrees(),
            vel.x,
            vel.y,
            vel.z,
            full_us,
            simple_us,
            full_us / simple_us
        );
    }
    println!("─────────────────────────────────────────────────────────────────────────────");

    // Summary stats
    let mean = |xs: &[f64; 10]| xs.iter().sum::<f64>() / 10.0;
    let stddev =
        |xs: &[f64; 10], m: f64| (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 10.0).sqrt();
    let min = |xs: &[f64; 10]| xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = |xs: &[f64; 10]| xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let full_mean = mean(&full_times);
    let full_std = stddev(&full_times, full_mean);
    let full_min = min(&full_times);
    let full_max = max(&full_times);
    let simple_mean = mean(&simple_times);
    let simple_std = stddev(&simple_times, simple_mean);
    let simple_min = min(&simple_times);
    let simple_max = max(&simple_times);
    let ratio_mean = full_mean / simple_mean;

    println!(
        "Full   :  mean={:.2} µs   σ={:.2} µs   min={:.2}   max={:.2}",
        full_mean, full_std, full_min, full_max
    );
    println!(
        "Simple :  mean={:.2} µs   σ={:.2} µs   min={:.2}   max={:.2}",
        simple_mean, simple_std, simple_min, simple_max
    );
    println!(
        "Ratio  :  full / simple = {:.2}×  (mean of means)",
        ratio_mean
    );
    println!("─────────────────────────────────────────────────────────────────────────────");

    // Sanity assertions
    assert!(full_mean > 0.0, "full solver timing must be positive");
    assert!(simple_mean > 0.0, "simple solver timing must be positive");
    // Per-case timings should be stable: σ/mean < 30 % is comfortable headroom
    // for normal CI noise.
    assert!(
        full_std / full_mean < 0.30,
        "full solver per-case stddev too high: σ={}, mean={}",
        full_std,
        full_mean
    );
    assert!(
        simple_std / simple_mean < 0.30,
        "simple solver per-case stddev too high: σ={}, mean={}",
        simple_std,
        simple_mean
    );
}

// ─── Steady-state error: do both MPCs converge to the target? ──────────────
//
// The other convergence tests use early termination (exit when error stays
// below POS_TOL for 100 steps). The reported err is therefore at the FIRST
// time the threshold was crossed, not at end-of-simulation. This test runs
// each controller for the full 15 s horizon with no early exit, so we can
// see whether the position error continues to decay toward zero (transient)
// or settles at a non-zero floor (finite-horizon MPC bias).

#[test]
fn final_steady_state_error() {
    use cybflight_core::mpc::quad_model::{NU as SIMPLE_NU, NX as SIMPLE_NX};
    use nalgebra::Unit;

    let target = Vector3::new(1.0, 1.0, 1.0);

    // Single representative initial state: 15° tilt around the (1,1,0) axis,
    // zero velocity. Not too extreme so both controllers easily reach the
    // vicinity of the target; the question is what they do *after* they get there.
    let q0 = UnitQuaternion::from_axis_angle(
        &Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0)),
        15.0_f32.to_radians(),
    );

    let max_steps: usize = (15.0 / SIM_DT) as usize; // 7500 steps = 15 s
                                                     // Time checkpoints (in seconds) at which to record position error.
    let checkpoints_s: SVector<f32, 6> = SVector::from_row_slice(&[0.5, 1.0, 2.0, 5.0, 10.0, 15.0]);
    let checkpoint_steps: [usize; 6] = [
        (0.5 / SIM_DT) as usize,
        (1.0 / SIM_DT) as usize,
        (2.0 / SIM_DT) as usize,
        (5.0 / SIM_DT) as usize,
        (10.0 / SIM_DT) as usize,
        (15.0 / SIM_DT) as usize - 1,
    ];

    // ── FullQuadModel run ──────────────────────────────────────────────────
    let dyn_full = make_dynamics();
    let mut x = initial_state(&q0);
    let mut ctrl = MpcController::new(&target, SIM_DT);
    let mut full_errs = SVector::<f32, 6>::zeros();
    let mut cp_idx = 0;
    for step in 0..max_steps {
        let u = ctrl.compute(&x);
        let mut u_clamped = u;
        for f in u_clamped.iter_mut() {
            *f = f.clamp(0.0, MAX_THRUST_N);
        }
        x = dyn_full.step(&x, &u_clamped);
        if cp_idx < checkpoint_steps.len() && step == checkpoint_steps[cp_idx] {
            let pos = Vector3::new(x[0], x[1], x[2]);
            full_errs[cp_idx] = (pos - target).norm();
            cp_idx += 1;
        }
    }
    let full_final_pos = Vector3::new(x[0], x[1], x[2]);

    // ── QuadModel run ──────────────────────────────────────────────────────
    let simple_dyn = SimpleDyn::new(SIM_DT);
    let mut sx: SVector<f32, SIMPLE_NX> =
        SVector::from_row_slice(&[0.0, 0.0, 0.0, q0.i, q0.j, q0.k, q0.w, 0.0, 0.0, 0.0]);
    let mut sctrl = SimpleCtrl::new(&target, SIM_DT);
    let mut simple_errs = SVector::<f32, 6>::zeros();
    let mut cp_idx = 0;
    for step in 0..max_steps {
        let u = sctrl.compute(&sx);
        // Clamp inputs to physical bounds
        let bounds = sctrl.problem.model.u_bounds;
        let mut u_clamped = u;
        for i in 0..SIMPLE_NU {
            u_clamped[i] = u_clamped[i].clamp(bounds[i][0], bounds[i][1]);
        }
        sx = simple_dyn.step(&sx, &u_clamped);
        if cp_idx < checkpoint_steps.len() && step == checkpoint_steps[cp_idx] {
            let pos = Vector3::new(sx[0], sx[1], sx[2]);
            simple_errs[cp_idx] = (pos - target).norm();
            cp_idx += 1;
        }
    }
    let simple_final_pos = Vector3::new(sx[0], sx[1], sx[2]);

    // ── Report ─────────────────────────────────────────────────────────────
    println!();
    println!("──────────────────────────────────────────────────────────────────");
    println!("Steady-state position error: 15° tilt around [1,1,0], target [1,1,1]");
    println!("──────────────────────────────────────────────────────────────────");
    println!("  t [s]   FullQuadModel err [m]   QuadModel err [m]");
    for i in 0..6 {
        println!(
            "  {:>5.1}        {:>10.5}              {:>10.5}",
            checkpoints_s[i], full_errs[i], simple_errs[i]
        );
    }
    println!("──────────────────────────────────────────────────────────────────");
    println!(
        "  Full   final pos = ({:.4}, {:.4}, {:.4})  err = {:.5} m",
        full_final_pos.x,
        full_final_pos.y,
        full_final_pos.z,
        (full_final_pos - target).norm()
    );
    println!(
        "  Simple final pos = ({:.4}, {:.4}, {:.4})  err = {:.5} m",
        simple_final_pos.x,
        simple_final_pos.y,
        simple_final_pos.z,
        (simple_final_pos - target).norm()
    );
    println!("──────────────────────────────────────────────────────────────────");

    // Both controllers must reach the target neighborhood within 0.15 m.
    let full_final_err = (full_final_pos - target).norm();
    let simple_final_err = (simple_final_pos - target).norm();
    assert!(
        full_final_err < 0.15,
        "FullQuadModel final err {} exceeds 0.15 m",
        full_final_err
    );
    assert!(
        simple_final_err < 0.15,
        "QuadModel final err {} exceeds 0.15 m",
        simple_final_err
    );
}

// Local copies of QuadModel-side helpers (the originals live in the
// `simple_quad` submodule below, which isn't reachable from this scope).
struct SimpleDyn {
    model: cybflight_core::mpc::QuadModel,
}
impl SimpleDyn {
    fn new(dt: f32) -> Self {
        Self {
            model: cybflight_core::mpc::QuadModel {
                dt,
                ..Default::default()
            },
        }
    }
    fn step(
        &self,
        x: &SVector<f32, { cybflight_core::mpc::quad_model::NX }>,
        u: &SVector<f32, { cybflight_core::mpc::quad_model::NU }>,
    ) -> SVector<f32, { cybflight_core::mpc::quad_model::NX }> {
        self.model.propagate_rk4(x, u)
    }
}

struct SimpleCtrl {
    solver: alloc::boxed::Box<cybflight_core::mpc::SimpleSqpSolver>,
    problem: cybflight_core::mpc::SimpleQuadProblem,
    x_refs: [SVector<f32, { cybflight_core::mpc::quad_model::NX }>;
        cybflight_core::mpc::quad_model::N + 1],
    u_refs:
        [SVector<f32, { cybflight_core::mpc::quad_model::NU }>; cybflight_core::mpc::quad_model::N],
    u_warm:
        [SVector<f32, { cybflight_core::mpc::quad_model::NU }>; cybflight_core::mpc::quad_model::N],
    last_u: SVector<f32, { cybflight_core::mpc::quad_model::NU }>,
    solve_period: usize,
    step_counter: usize,
}
impl SimpleCtrl {
    fn new(target: &Vector3<f32>, sim_dt: f32) -> Self {
        use cybflight_core::mpc::quad_model::{N as SN, NX as SNX};
        const MPC_DT: f32 = 0.05;
        const MPC_SOLVE_DT: f32 = 0.01;
        let model = cybflight_core::mpc::QuadModel {
            dt: MPC_DT,
            ..Default::default()
        };
        let problem = cybflight_core::mpc::SimpleQuadProblem::with_rk4(model, SN);
        let u_ref = SVector::<f32, { cybflight_core::mpc::quad_model::NU }>::from_row_slice(&[
            MASS * GRAV,
            0.0,
            0.0,
            0.0,
        ]);
        let mut x_ref = SVector::<f32, SNX>::zeros();
        x_ref[0] = target.x;
        x_ref[1] = target.y;
        x_ref[2] = target.z;
        x_ref[6] = 1.0;
        let solve_period = (MPC_SOLVE_DT / sim_dt).round() as usize;
        Self {
            solver: alloc::boxed::Box::new(cybflight_core::mpc::SimpleSqpSolver::new()),
            problem,
            x_refs: [x_ref; SN + 1],
            u_refs: [u_ref; SN],
            u_warm: [u_ref; SN],
            last_u: u_ref,
            solve_period,
            step_counter: 0,
        }
    }
    fn compute(
        &mut self,
        x: &SVector<f32, { cybflight_core::mpc::quad_model::NX }>,
    ) -> SVector<f32, { cybflight_core::mpc::quad_model::NU }> {
        self.step_counter += 1;
        if self.step_counter >= self.solve_period {
            self.step_counter = 0;
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
            self.last_u = u_bar[0];
            self.u_warm = *u_bar;
        }
        self.last_u
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QuadModel (10-state, thrust + body-rate) MPC convergence test
// ═══════════════════════════════════════════════════════════════════════════
//
// This test exercises the SAME `SqpSolver` machinery as the FullQuadModel
// test above but instantiated for the reduced 10-state `QuadModel`. The
// simulator and the MPC's internal model are both `QuadModel`, so this is
// a "model-perfect" test that validates the SQP solver works with the
// 10-state formulation. Model-fidelity vs. full rigid-body dynamics is
// already covered by the FullQuadModel test.

mod simple_quad {
    use cybflight_core::mpc::quad_model::{N as SIMPLE_N, NU as SIMPLE_NU, NX as SIMPLE_NX};
    use cybflight_core::mpc::{QuadModel, SimpleQuadProblem, SimpleSqpSolver};
    use nalgebra::{SVector, UnitQuaternion, Vector3};

    use super::{generate_initial_orientations, GRAV, MASS, NUM_ORIENTATIONS, POS_TOL, SIM_DT};

    /// Build a default `QuadModel` and override its dt for the prediction
    /// horizon (50 ms — same as the FullQuadModel test). All other fields
    /// (mass, weights, bounds) come from `Default::default()`.
    fn make_simple_model(dt: f32) -> QuadModel {
        QuadModel {
            dt,
            ..Default::default()
        }
    }

    /// Initial state at the origin with a given orientation, zero velocity.
    fn initial_state(q: &UnitQuaternion<f32>) -> SVector<f32, SIMPLE_NX> {
        SVector::from_row_slice(&[0.0, 0.0, 0.0, q.i, q.j, q.k, q.w, 0.0, 0.0, 0.0])
    }

    fn pos_of(x: &SVector<f32, SIMPLE_NX>) -> Vector3<f32> {
        Vector3::new(x[0], x[1], x[2])
    }

    /// 500 Hz simulator built on `QuadModel::propagate_rk4`.
    /// Since `QuadModel` already normalizes the quaternion inside its
    /// projection-method RK4, we don't need an external normalize call.
    struct SimpleDynamics {
        model: QuadModel,
    }

    impl SimpleDynamics {
        fn new() -> Self {
            Self {
                model: make_simple_model(SIM_DT),
            }
        }

        fn step(
            &self,
            x: &SVector<f32, SIMPLE_NX>,
            u: &SVector<f32, SIMPLE_NU>,
        ) -> SVector<f32, SIMPLE_NX> {
            self.model.propagate_rk4(x, u)
        }
    }

    /// MPC controller wrapper for the simple (10-state) model.
    struct SimpleMpcController {
        solver: alloc::boxed::Box<SimpleSqpSolver>,
        problem: SimpleQuadProblem,
        x_refs: [SVector<f32, SIMPLE_NX>; SIMPLE_N + 1],
        u_refs: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
        u_warm: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
        last_u: SVector<f32, SIMPLE_NU>,
        solve_period: usize,
        step_counter: usize,
    }

    /// Prediction timestep (matches FullQuadModel test).
    const MPC_DT: f32 = 0.05;
    /// Solver execution period — 100 Hz (matches FullQuadModel test).
    const MPC_SOLVE_DT: f32 = 0.01;

    impl SimpleMpcController {
        fn new(target: &Vector3<f32>, sim_dt: f32) -> Self {
            let model = make_simple_model(MPC_DT);
            let problem = SimpleQuadProblem::with_rk4(model, SIMPLE_N);

            // Hover input: collective thrust = m·g, zero body-rate command.
            let u_ref = SVector::<f32, SIMPLE_NU>::from_row_slice(&[MASS * GRAV, 0.0, 0.0, 0.0]);

            let mut x_ref = SVector::<f32, SIMPLE_NX>::zeros();
            x_ref[0] = target.x;
            x_ref[1] = target.y;
            x_ref[2] = target.z;
            x_ref[6] = 1.0; // identity quaternion: qw = 1

            let solve_period = (MPC_SOLVE_DT / sim_dt).round() as usize;

            Self {
                solver: alloc::boxed::Box::new(SimpleSqpSolver::new()),
                problem,
                x_refs: [x_ref; SIMPLE_N + 1],
                u_refs: [u_ref; SIMPLE_N],
                u_warm: [u_ref; SIMPLE_N],
                last_u: u_ref,
                solve_period,
                step_counter: 0,
            }
        }

        fn compute(&mut self, x: &SVector<f32, SIMPLE_NX>) -> SVector<f32, SIMPLE_NU> {
            self.step_counter += 1;
            if self.step_counter >= self.solve_period {
                self.step_counter = 0;

                let _result = self.solver.solve(
                    &self.problem,
                    x,
                    &self.x_refs,
                    &self.u_refs,
                    &self.u_warm,
                    1,
                    1e-3,
                );

                let u_bar = self.solver.u_bar();
                self.last_u = u_bar[0];
                self.u_warm = *u_bar;
            }

            self.last_u
        }
    }

    struct SimResult {
        final_pos: Vector3<f32>,
        pos_error: f32,
        converged: bool,
        steps: usize,
    }

    fn run_simulation(
        ctrl: &mut SimpleMpcController,
        dynamics: &SimpleDynamics,
        x0: &SVector<f32, SIMPLE_NX>,
        target: &Vector3<f32>,
        max_steps: usize,
    ) -> SimResult {
        let mut x = *x0;
        let mut converge_count = 0usize;
        let converge_window = 100;

        for step in 0..max_steps {
            let u = ctrl.compute(&x);

            // Clamp inputs to physical bounds (model-internal clamp is also
            // applied by the SQP, but the simulator should respect them too).
            let bounds = ctrl.problem.model.u_bounds;
            let mut u_clamped = u;
            for i in 0..SIMPLE_NU {
                u_clamped[i] = u_clamped[i].clamp(bounds[i][0], bounds[i][1]);
            }

            x = dynamics.step(&x, &u_clamped);

            let pos = pos_of(&x);
            let err = (pos - target).norm();

            if err < POS_TOL {
                converge_count += 1;
            } else {
                converge_count = 0;
            }

            if converge_count >= converge_window {
                return SimResult {
                    final_pos: pos,
                    pos_error: err,
                    converged: true,
                    steps: step + 1,
                };
            }

            if pos.norm() > 100.0 || !pos.x.is_finite() {
                break;
            }
        }

        let pos = pos_of(&x);
        SimResult {
            final_pos: pos,
            pos_error: (pos - target).norm(),
            converged: false,
            steps: max_steps,
        }
    }

    #[test]
    fn simple_mpc_converges_50_orientations() {
        let target = Vector3::new(1.0, 1.0, 1.0);
        let dynamics = SimpleDynamics::new();
        let orientations = generate_initial_orientations(NUM_ORIENTATIONS);
        let max_time_s: f32 = 15.0;
        let max_steps: usize = (max_time_s / SIM_DT) as usize;

        let mut failures = alloc::vec::Vec::new();

        for (i, q) in orientations.iter().enumerate() {
            let x0 = initial_state(q);
            let mut ctrl = SimpleMpcController::new(&target, SIM_DT);
            let result = run_simulation(&mut ctrl, &dynamics, &x0, &target, max_steps);

            let angle_deg = q.angle().to_degrees();
            if result.converged {
                println!(
                    "  Simple [{:2}] tilt={:5.1}° → converged in {:5} steps ({:.2}s), err={:.4}m",
                    i,
                    angle_deg,
                    result.steps,
                    result.steps as f32 * SIM_DT,
                    result.pos_error,
                );
            } else {
                println!(
                    "  Simple [{:2}] tilt={:5.1}° → FAILED, err={:.4}m, pos=({:.2},{:.2},{:.2})",
                    i,
                    angle_deg,
                    result.pos_error,
                    result.final_pos.x,
                    result.final_pos.y,
                    result.final_pos.z,
                );
                failures.push(i);
            }
        }

        assert!(
            failures.is_empty(),
            "Simple MPC controller failed for {} / {} orientations: {:?}",
            failures.len(),
            NUM_ORIENTATIONS,
            failures,
        );
    }
}
