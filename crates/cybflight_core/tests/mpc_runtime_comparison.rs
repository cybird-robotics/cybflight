//! Runtime comparison — `SimpleSqpSolver` (10-state `QuadModel`) vs
//! `FullSqpSolver` (13-state `FullQuadModel`), the latter with and without
//! body-rate state constraints.
//!
//! The reduced model commands body rates directly, so the vehicle's rate
//! limits are plain *input* bounds (`u_bounds`). The full model carries the
//! rates as states 10..13, so the same limits require *state* constraints —
//! implemented as relaxed log-barrier stage-cost terms following
//! Frey et al., arXiv:2505.01353v2 (App. A.2: IPM-based SQP ≡ SQP on the
//! log-barrier problem). This file:
//!
//! 1. checks the barrier gradient/Hessian against finite differences,
//! 2. checks the barrier actually restrains the predicted body rates,
//! 3. benches `solve()` wall time for all three solver configurations over a
//!    grid of initial states, in both RTI mode (`max_iters = 1`, the
//!    firmware call pattern) and solve-to-convergence mode.
//!
//! Run with:
//!   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --test mpc_runtime_comparison --release -- --nocapture

extern crate alloc;

use std::hint::black_box;
use std::time::Instant;

use cybflight_core::mpc::quad_model::{
    PosCostMode, N as SIMPLE_N, NU as SIMPLE_NU, NX as SIMPLE_NX,
};
use cybflight_core::mpc::{
    FullQuadModel, FullQuadProblem, FullSqpSolver, MpcProblem, QuadDynamicsModel, QuadModel,
    SimpleQuadProblem, SimpleSqpSolver, SolverResult, SqpSolver, N as FULL_N, NU as FULL_NU,
    NX as FULL_NX,
};
use nalgebra::{SVector, Unit, UnitQuaternion, Vector3, Vector4};

// ═══════════════════════════════════════════════════════════════════════════
// Shared vehicle parameters (pinned, matching control_convergence.rs)
// ═══════════════════════════════════════════════════════════════════════════

const MASS: f32 = 0.55;
const GRAV: f32 = 9.81;
const MPC_DT: f32 = 0.05;
/// Per-axis body-rate limits [rad/s] — identical for the simple model's
/// input bounds and the full model's state constraints.
const RATE_LIM: [f32; 3] = [10.0, 10.0, 6.0];
/// Barrier weight τ / relaxation margin δ used in the constrained config.
const BARRIER_TAU: f32 = 0.1;
const BARRIER_DELTA: f32 = 0.5;
/// Solver iteration/tolerance settings.
const KKT_TOL: f32 = 5e-3;
const CONV_MAX_ITERS: usize = 30;

fn full_model(constrained: bool) -> FullQuadModel {
    FullQuadModel {
        mass: MASS,
        grav: GRAV,
        dt: MPC_DT,
        rate_bounds: [
            [-RATE_LIM[0], RATE_LIM[0]],
            [-RATE_LIM[1], RATE_LIM[1]],
            [-RATE_LIM[2], RATE_LIM[2]],
        ],
        rate_barrier_tau: if constrained { BARRIER_TAU } else { 0.0 },
        rate_barrier_delta: BARRIER_DELTA,
        ..Default::default()
    }
}

/// Reduced model with the same mass/gravity/weights as the full model so the
/// two formulations solve the same physical problem. Rate limits are input
/// bounds here (`u_bounds[1..4]`).
fn simple_model() -> QuadModel {
    QuadModel {
        mass: MASS,
        grav: GRAV,
        dt: MPC_DT,
        u_bounds: [
            [0.0, 4.0 * 8.5],
            [-RATE_LIM[0], RATE_LIM[0]],
            [-RATE_LIM[1], RATE_LIM[1]],
            [-RATE_LIM[2], RATE_LIM[2]],
        ],
        mass_inv: 1.0 / MASS,
        w_pos: [200.0, 200.0, 200.0],
        w_vel: [1.0, 1.0, 1.0],
        w_att: [5.0, 5.0, 200.0],
        w_pos_n: [200.0, 200.0, 200.0],
        w_vel_n: [1.0, 1.0, 1.0],
        w_att_n: [5.0, 5.0, 200.0],
        w_input: Vector4::new(6.0, 1.0, 1.0, 1.0),
        rho: 1e4,
        pos_cost_mode: PosCostMode::Quadratic,
        // Tilt fence off — this suite benchmarks it separately.
        tilt_cos_max: 0.5,
        tilt_barrier_tau: 0.0,
        tilt_barrier_delta: 0.05,
        drag_coeff: [0.0; 3],
        thrust_coeff: 0.0,
        body_drag_coeff: [0.0; 3],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Initial-state grid
// ═══════════════════════════════════════════════════════════════════════════

struct Case {
    label: &'static str,
    pos: Vector3<f32>,
    axis: Vector3<f32>,
    angle_deg: f32,
    vel: Vector3<f32>,
    /// Initial body rates — a state for the full model; the simple model has
    /// no rate state, so this component of the pose is dropped there.
    omega: Vector3<f32>,
}

fn cases() -> [Case; 12] {
    [
        Case {
            label: "hover",
            pos: Vector3::zeros(),
            axis: Vector3::z(),
            angle_deg: 0.0,
            vel: Vector3::zeros(),
            omega: Vector3::zeros(),
        },
        Case {
            label: "tilt10",
            pos: Vector3::zeros(),
            axis: Vector3::x(),
            angle_deg: 10.0,
            vel: Vector3::zeros(),
            omega: Vector3::zeros(),
        },
        Case {
            label: "tilt45",
            pos: Vector3::zeros(),
            axis: Vector3::new(1.0, 1.0, 0.0),
            angle_deg: 45.0,
            vel: Vector3::zeros(),
            omega: Vector3::zeros(),
        },
        Case {
            label: "fast_xlate",
            pos: Vector3::zeros(),
            axis: Vector3::y(),
            angle_deg: 15.0,
            vel: Vector3::new(3.0, 0.0, 0.0),
            omega: Vector3::zeros(),
        },
        Case {
            label: "descend",
            pos: Vector3::zeros(),
            axis: Vector3::z(),
            angle_deg: 0.0,
            vel: Vector3::new(0.0, 0.0, -3.0),
            omega: Vector3::zeros(),
        },
        Case {
            label: "far_offset",
            pos: Vector3::new(-3.0, -3.0, -2.0),
            axis: Vector3::x(),
            angle_deg: 5.0,
            vel: Vector3::zeros(),
            omega: Vector3::zeros(),
        },
        Case {
            label: "mid_rates",
            pos: Vector3::zeros(),
            axis: Vector3::x(),
            angle_deg: 20.0,
            vel: Vector3::zeros(),
            omega: Vector3::new(3.0, 3.0, 1.0),
        },
        Case {
            label: "near_limit",
            pos: Vector3::zeros(),
            axis: Vector3::new(1.0, -1.0, 0.0),
            angle_deg: 30.0,
            vel: Vector3::new(1.0, 0.0, 0.0),
            omega: Vector3::new(9.0, 9.0, 5.0),
        },
        Case {
            label: "over_limit",
            pos: Vector3::zeros(),
            axis: Vector3::new(1.0, 0.5, 0.0),
            angle_deg: 30.0,
            vel: Vector3::zeros(),
            omega: Vector3::new(12.0, -12.0, 7.0),
        },
        Case {
            label: "yaw_spin",
            pos: Vector3::zeros(),
            axis: Vector3::z(),
            angle_deg: 90.0,
            vel: Vector3::zeros(),
            omega: Vector3::new(0.0, 0.0, 5.5),
        },
        Case {
            label: "aggressive",
            pos: Vector3::zeros(),
            axis: Vector3::new(1.0, 1.0, 1.0),
            angle_deg: 45.0,
            vel: Vector3::new(2.0, -1.0, 0.5),
            omega: Vector3::new(6.0, -6.0, 3.0),
        },
        Case {
            label: "tumbling",
            pos: Vector3::zeros(),
            axis: Vector3::new(1.0, -0.5, 0.3),
            angle_deg: 60.0,
            vel: Vector3::new(1.0, 1.0, -1.0),
            omega: Vector3::new(8.0, 8.0, -5.0),
        },
    ]
}

fn case_quat(c: &Case) -> UnitQuaternion<f32> {
    if c.angle_deg == 0.0 {
        UnitQuaternion::identity()
    } else {
        UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(c.axis),
            c.angle_deg.to_radians(),
        )
    }
}

fn full_x0(c: &Case) -> SVector<f32, FULL_NX> {
    let q = case_quat(c);
    SVector::<f32, FULL_NX>::from_row_slice(&[
        c.pos.x, c.pos.y, c.pos.z, q.i, q.j, q.k, q.w, c.vel.x, c.vel.y, c.vel.z, c.omega.x,
        c.omega.y, c.omega.z,
    ])
}

fn simple_x0(c: &Case) -> SVector<f32, SIMPLE_NX> {
    let q = case_quat(c);
    SVector::<f32, SIMPLE_NX>::from_row_slice(&[
        c.pos.x, c.pos.y, c.pos.z, q.i, q.j, q.k, q.w, c.vel.x, c.vel.y, c.vel.z,
    ])
}

// ═══════════════════════════════════════════════════════════════════════════
// Generic timing harness
// ═══════════════════════════════════════════════════════════════════════════

/// Time `measure` cold-start solves (identical inputs each call — the work
/// per call is deterministic). Returns (mean per-call µs, last SolverResult).
#[allow(clippy::too_many_arguments)]
fn time_solve<M, const NX: usize, const NU: usize, const N: usize, const NP1: usize>(
    solver: &mut SqpSolver<NX, NU, N, NP1>,
    problem: &MpcProblem<M, NX, NU>,
    x0: &SVector<f32, NX>,
    x_refs: &[SVector<f32, NX>; NP1],
    u_refs: &[SVector<f32, NU>; N],
    u_init: &[SVector<f32, NU>; N],
    max_iters: usize,
    warmup: usize,
    measure: usize,
) -> (f64, SolverResult)
where
    M: QuadDynamicsModel<NX, NU>,
{
    let mut last = SolverResult {
        cost: 0.0,
        iters: 0,
        converged: false,
        diverged: false,
    };
    for _ in 0..warmup {
        last = black_box(solver.solve(problem, x0, x_refs, u_refs, u_init, max_iters, KKT_TOL));
    }
    let t0 = Instant::now();
    for _ in 0..measure {
        last = black_box(solver.solve(problem, x0, x_refs, u_refs, u_init, max_iters, KKT_TOL));
    }
    let per_call_us = t0.elapsed().as_nanos() as f64 / 1_000.0 / measure as f64;
    (per_call_us, last)
}

/// Max over the predicted horizon (k ≥ 1 — k = 0 is the fixed initial state)
/// of the per-axis body-rate utilization `|ω_i| / lim_i`. > 1.0 means the
/// prediction violates the rate limits somewhere. Returns NaN if the
/// trajectory contains a non-finite value (diverged solve) — `f32::max`
/// would otherwise silently ignore NaNs and report a bogus 0.
fn max_rate_util(x_bar: &[SVector<f32, FULL_NX>; FULL_N + 1]) -> f32 {
    let mut util = 0.0f32;
    for x in x_bar.iter().skip(1) {
        for i in 0..3 {
            let w = x[10 + i];
            if !w.is_finite() {
                return f32::NAN;
            }
            util = util.max(w.abs() / RATE_LIM[i]);
        }
    }
    util
}

// ═══════════════════════════════════════════════════════════════════════════
// Reference / warm-start builders
// ═══════════════════════════════════════════════════════════════════════════

struct FullSetup {
    problem: FullQuadProblem,
    x_refs: [SVector<f32, FULL_NX>; FULL_N + 1],
    u_refs: [SVector<f32, FULL_NU>; FULL_N],
    u_init: [SVector<f32, FULL_NU>; FULL_N],
}

fn full_setup(target: Vector3<f32>, constrained: bool) -> FullSetup {
    let problem = FullQuadProblem::with_rk4(full_model(constrained), FULL_N);
    let hover = SVector::<f32, FULL_NU>::from_element(MASS * GRAV / 4.0);
    let mut x_ref = SVector::<f32, FULL_NX>::zeros();
    x_ref[0] = target.x;
    x_ref[1] = target.y;
    x_ref[2] = target.z;
    x_ref[6] = 1.0;
    FullSetup {
        problem,
        x_refs: [x_ref; FULL_N + 1],
        u_refs: [hover; FULL_N],
        u_init: [hover; FULL_N],
    }
}

struct SimpleSetup {
    problem: SimpleQuadProblem,
    x_refs: [SVector<f32, SIMPLE_NX>; SIMPLE_N + 1],
    u_refs: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
    u_init: [SVector<f32, SIMPLE_NU>; SIMPLE_N],
}

fn simple_setup(target: Vector3<f32>) -> SimpleSetup {
    let problem = SimpleQuadProblem::with_rk4(simple_model(), SIMPLE_N);
    let hover = SVector::<f32, SIMPLE_NU>::from_row_slice(&[MASS * GRAV, 0.0, 0.0, 0.0]);
    let mut x_ref = SVector::<f32, SIMPLE_NX>::zeros();
    x_ref[0] = target.x;
    x_ref[1] = target.y;
    x_ref[2] = target.z;
    x_ref[6] = 1.0;
    SimpleSetup {
        problem,
        x_refs: [x_ref; SIMPLE_N + 1],
        u_refs: [hover; SIMPLE_N],
        u_init: [hover; SIMPLE_N],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Barrier gradient/Hessian vs finite differences
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rate_barrier_matches_finite_differences() {
    let model = full_model(true);
    let mut x = SVector::<f32, FULL_NX>::zeros();
    x[6] = 1.0;
    // Rates spanning both barrier branches: log branch (margin ≫ δ),
    // near-boundary log branch, and quadratic-extension branch (violated).
    for &omega in &[
        [2.0f32, -4.0, 1.0],  // interior
        [9.5, -9.5, 5.5],     // near the limits
        [11.0, -12.0, 6.5],   // beyond the limits (quadratic extension)
    ] {
        x[10] = omega[0];
        x[11] = omega[1];
        x[12] = omega[2];
        let xref = {
            let mut r = SVector::<f32, FULL_NX>::zeros();
            r[6] = 1.0;
            r
        };

        let mut grad = SVector::<f32, FULL_NX>::zeros();
        let mut hess = nalgebra::SMatrix::<f32, FULL_NX, FULL_NX>::zeros();
        let _ = model.state_cost_hess_grad(&x, &xref, &mut grad, &mut hess);

        // Central differences on the three rate states only (quaternion FD
        // would leave the unit sphere; rates are unconstrained coordinates).
        let h = 1e-3f32;
        for i in 10..13 {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let mut g_scratch = SVector::<f32, FULL_NX>::zeros();
            let cp = model.state_cost_grad(&xp, &xref, &mut g_scratch);
            let cm = model.state_cost_grad(&xm, &xref, &mut g_scratch);
            let fd_grad = (cp - cm) / (2.0 * h);
            let rel = (grad[i] - fd_grad).abs() / fd_grad.abs().max(1e-3);
            assert!(
                rel < 2e-2,
                "grad mismatch at ω={omega:?} state {i}: analytic={}, fd={}",
                grad[i],
                fd_grad
            );

            // Hessian diagonal via FD of the gradient.
            let mut gp = SVector::<f32, FULL_NX>::zeros();
            let mut gm = SVector::<f32, FULL_NX>::zeros();
            let _ = model.state_cost_grad(&xp, &xref, &mut gp);
            let _ = model.state_cost_grad(&xm, &xref, &mut gm);
            let fd_hess = (gp[i] - gm[i]) / (2.0 * h);
            let rel_h = (hess[(i, i)] - fd_hess).abs() / fd_hess.abs().max(1e-3);
            assert!(
                rel_h < 2e-2,
                "hess mismatch at ω={omega:?} state {i}: analytic={}, fd={}",
                hess[(i, i)],
                fd_hess
            );
        }
    }
}

// τ = 0 must add nothing: the barrier fields default to off, and the cost /
// gradient / Hessian must be bit-identical to a model that never heard of
// rate bounds. (The barrier *value* has an arbitrary offset — −ln z can be
// negative — so the meaningful signal is the gradient/Hessian.)
#[test]
fn barrier_off_is_byte_identical() {
    let on = full_model(true);
    let off = full_model(false);
    let mut x = SVector::<f32, FULL_NX>::zeros();
    x[6] = 1.0;
    x[10] = 9.9; // just inside the limit — barrier is steep here
    let xref = x;

    let mut g_off = SVector::<f32, FULL_NX>::zeros();
    let mut h_off = nalgebra::SMatrix::<f32, FULL_NX, FULL_NX>::zeros();
    let c_off = off.state_cost_hess_grad(&x, &xref, &mut g_off, &mut h_off);

    let mut g_on = SVector::<f32, FULL_NX>::zeros();
    let mut h_on = nalgebra::SMatrix::<f32, FULL_NX, FULL_NX>::zeros();
    let c_on = on.state_cost_hess_grad(&x, &xref, &mut g_on, &mut h_on);

    // τ = 0 is the plain legacy stage cost: zero error → zero cost, zero grad.
    assert_eq!(c_off, 0.0);
    assert_eq!(g_off, SVector::<f32, FULL_NX>::zeros());
    assert_eq!(h_off[(10, 10)], 2.0 * MPC_DT * 1.0); // w_rate diag only

    // τ > 0 near the bound: positive gradient pushing ω back down, extra
    // positive-definite Hessian mass on the rate diagonal. Expected upper-
    // bound term at margin z = 0.1: log branch −τ/z if z > δ, else quadratic
    // extension τ·(2δ − z)/δ².
    let z = RATE_LIM[0] - 9.9;
    let expect_grad = if z > BARRIER_DELTA {
        BARRIER_TAU / z
    } else {
        BARRIER_TAU * (2.0 * BARRIER_DELTA - z) / (BARRIER_DELTA * BARRIER_DELTA)
    };
    assert!(c_on != c_off, "barrier must change the stage cost value");
    assert!(
        g_on[10] > 0.5 * expect_grad,
        "barrier gradient at ω=9.9 must push down (expected ≈{expect_grad}), got {}",
        g_on[10]
    );
    assert!(
        h_on[(10, 10)] > h_off[(10, 10)],
        "barrier must add Hessian mass on the constrained state"
    );
    // Unconstrained rows untouched.
    assert_eq!(g_on[0], g_off[0]);
    assert_eq!(h_on[(0, 0)], h_off[(0, 0)]);
}

/// The tilt fence (cos-space relaxed log barrier on `1 − 2(qx²+qy²)`) must
/// match central finite differences of its own cost — gradient and Hessian
/// block — in the log branch, near the δ switch, and in the quadratic
/// extension (violated tilt).
#[test]
fn tilt_barrier_matches_finite_differences() {
    let mut model = full_model(false);
    model.tilt_cos_max = 0.5; // 60°
    model.tilt_barrier_tau = 0.5;
    model.tilt_barrier_delta = 0.05;

    let mut xref = SVector::<f32, FULL_NX>::zeros();
    xref[6] = 1.0;

    // Roll-axis tilts: log branch (30°), near the δ switch (58°), violated
    // in the quadratic extension (80°, 120°).
    for tilt_deg in [30.0f32, 58.0, 80.0, 120.0] {
        let half = 0.5 * tilt_deg.to_radians();
        let mut x = SVector::<f32, FULL_NX>::zeros();
        x[3] = libm::sinf(half);
        x[6] = libm::cosf(half);
        // Same state as reference except attitude → all non-tilt cost terms
        // still contribute; the FD check covers the sum, so any mismatch in
        // the tilt term shows up.
        let mut grad = SVector::<f32, FULL_NX>::zeros();
        let mut hess = nalgebra::SMatrix::<f32, FULL_NX, FULL_NX>::zeros();
        let _ = model.state_cost_hess_grad(&x, &xref, &mut grad, &mut hess);

        let h = 1e-4;
        for i in [3usize, 4] {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let mut g_scratch = SVector::<f32, FULL_NX>::zeros();
            let cp = model.state_cost_grad(&xp, &xref, &mut g_scratch);
            let cm = model.state_cost_grad(&xm, &xref, &mut g_scratch);
            let fd_grad = (cp - cm) / (2.0 * h);
            let rel = (grad[i] - fd_grad).abs() / fd_grad.abs().max(1e-3);
            assert!(
                rel < 2e-2,
                "tilt {tilt_deg}°: grad mismatch at q[{i}]: analytic={}, fd={}",
                grad[i],
                fd_grad
            );

            let mut gp = SVector::<f32, FULL_NX>::zeros();
            let mut gm = SVector::<f32, FULL_NX>::zeros();
            let _ = model.state_cost_grad(&xp, &xref, &mut gp);
            let _ = model.state_cost_grad(&xm, &xref, &mut gm);
            let fd_hess = (gp[i] - gm[i]) / (2.0 * h);
            let rel_h = (hess[(i, i)] - fd_hess).abs() / fd_hess.abs().max(1e-3);
            assert!(
                rel_h < 2e-2,
                "tilt {tilt_deg}°: hess mismatch at q[{i}]: analytic={}, fd={}",
                hess[(i, i)],
                fd_hess
            );
        }
        // τ = 0 must be exactly inert at the same state.
        let mut off = model.clone();
        off.tilt_barrier_tau = 0.0;
        let mut g_off = SVector::<f32, FULL_NX>::zeros();
        let mut h_off = nalgebra::SMatrix::<f32, FULL_NX, FULL_NX>::zeros();
        let c_off = off.state_cost_hess_grad(&x, &xref, &mut g_off, &mut h_off);
        let base = full_model(false);
        let mut g_base = SVector::<f32, FULL_NX>::zeros();
        let mut h_base = nalgebra::SMatrix::<f32, FULL_NX, FULL_NX>::zeros();
        let c_base = base.state_cost_hess_grad(&x, &xref, &mut g_base, &mut h_base);
        assert_eq!(c_off, c_base);
        assert_eq!(g_off, g_base);
        assert_eq!(h_off, h_base);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Constraint effectiveness — does the barrier restrain predicted rates?
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn barrier_restrains_predicted_rates() {
    let target = Vector3::new(1.0, 1.0, 1.0);
    let unc = full_setup(target, false);
    let con = full_setup(target, true);
    let mut solver_unc = alloc::boxed::Box::new(FullSqpSolver::new());
    let mut solver_con = alloc::boxed::Box::new(FullSqpSolver::new());

    println!();
    println!("case          unconstrained util   constrained util");
    let mut restrained_any = false;
    for c in cases() {
        let x0 = full_x0(&c);
        let _ = solver_unc.solve(
            &unc.problem, &x0, &unc.x_refs, &unc.u_refs, &unc.u_init, CONV_MAX_ITERS, KKT_TOL,
        );
        let _ = solver_con.solve(
            &con.problem, &x0, &con.x_refs, &con.u_refs, &con.u_init, CONV_MAX_ITERS, KKT_TOL,
        );
        // NB: after the final iteration x_bar is not re-propagated; re-roll
        // the trajectory from the returned u_bar for a consistent readout.
        let roll = |problem: &FullQuadProblem,
                    u_bar: &[SVector<f32, FULL_NU>; FULL_N]|
         -> [SVector<f32, FULL_NX>; FULL_N + 1] {
            let mut xs = [SVector::<f32, FULL_NX>::zeros(); FULL_N + 1];
            xs[0] = x0;
            for k in 0..FULL_N {
                xs[k + 1] = problem.propagate(&xs[k], &u_bar[k]);
            }
            xs
        };
        let util_unc = max_rate_util(&roll(&unc.problem, solver_unc.u_bar()));
        let util_con = max_rate_util(&roll(&con.problem, solver_con.u_bar()));
        println!(
            "{:<12}  {:>18.3}  {:>17.3}{}",
            c.label,
            util_unc,
            util_con,
            if util_unc.is_nan() || util_con.is_nan() {
                "   (diverged — NaN trajectory)"
            } else {
                ""
            }
        );

        // The barrier is a soft constraint (fixed τ, no line search in the
        // SQP), so its equilibrium can sit some percent above 1 on marginal
        // transients — a case in the 1.0–1.5 band is not a pass/fail signal.
        // What must hold: it never *introduces* divergence, and it must
        // strictly reduce strong violations.
        assert!(
            util_unc.is_nan() || !util_con.is_nan(),
            "case {}: barrier introduced divergence (unconstrained was finite)",
            c.label
        );
        if util_unc.is_finite() && util_con.is_finite() && util_unc > 1.5 {
            assert!(
                util_con < util_unc,
                "case {}: constrained solve must reduce rate utilization ({} !< {})",
                c.label,
                util_con,
                util_unc
            );
            restrained_any = true;
        }
    }
    assert!(
        restrained_any,
        "no case pushed the unconstrained solver well past the rate limits — grid too tame"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Runtime benchmark
// ═══════════════════════════════════════════════════════════════════════════

/// Tilt-fence overhead: RTI timing of the flight config (rate barrier on)
/// with and without the tilt constraint, across states covering every
/// barrier branch — hover (log branch, far), 45° (log branch, near), 58°
/// (inside the δ switch), 80° with rates (violated, quadratic extension).
///
/// Prints per-case and mean overhead. Deliberately no timing assertion
/// (CI-flaky); the assertion is that the tilt solves stay finite.
#[test]
fn bench_tilt_barrier_overhead() {
    const WARMUP: usize = 100;
    const MEASURE: usize = 2_000;

    let target = Vector3::new(1.0, 1.0, 1.0);
    let base = full_setup(target, true);
    let mut tilt_model = full_model(true);
    tilt_model.tilt_cos_max = 0.5; // 60°
    tilt_model.tilt_barrier_tau = 0.5;
    tilt_model.tilt_barrier_delta = 0.05;
    let tilt_problem = FullQuadProblem::with_rk4(tilt_model, FULL_N);

    let mut s_off = alloc::boxed::Box::new(FullSqpSolver::new());
    let mut s_on = alloc::boxed::Box::new(FullSqpSolver::new());

    let tilt_cases = [
        Case {
            label: "hover",
            pos: Vector3::zeros(),
            axis: Vector3::z(),
            angle_deg: 0.0,
            vel: Vector3::zeros(),
            omega: Vector3::zeros(),
        },
        Case {
            label: "tilt45",
            pos: Vector3::zeros(),
            axis: Vector3::x(),
            angle_deg: 45.0,
            vel: Vector3::zeros(),
            omega: Vector3::zeros(),
        },
        Case {
            label: "tilt58",
            pos: Vector3::zeros(),
            axis: Vector3::x(),
            angle_deg: 58.0,
            vel: Vector3::zeros(),
            omega: Vector3::zeros(),
        },
        Case {
            label: "tilt80_rates",
            pos: Vector3::zeros(),
            axis: Vector3::x(),
            angle_deg: 80.0,
            vel: Vector3::new(0.0, -1.0, -1.0),
            omega: Vector3::new(3.0, -2.0, 1.0),
        },
    ];

    println!();
    println!("Tilt-fence overhead — RTI (1 iter), rate barrier ON in both columns");
    println!("case            off[µs]      +tilt[µs]   overhead");
    let (mut sum_off, mut sum_on) = (0.0f64, 0.0f64);
    for c in &tilt_cases {
        let x0 = full_x0(c);
        let (t_off, _) = time_solve(
            &mut s_off, &base.problem, &x0, &base.x_refs, &base.u_refs, &base.u_init, 1,
            WARMUP, MEASURE,
        );
        let (t_on, r_on) = time_solve(
            &mut s_on, &tilt_problem, &x0, &base.x_refs, &base.u_refs, &base.u_init, 1,
            WARMUP, MEASURE,
        );
        assert!(
            r_on.cost.is_finite() && s_on.u_bar()[0][0].is_finite(),
            "case {}: tilt-constrained solve diverged",
            c.label
        );
        sum_off += t_off;
        sum_on += t_on;
        println!(
            "{:<12} {:>10.2} {:>13.2} {:>+9.1}%",
            c.label,
            t_off,
            t_on,
            100.0 * (t_on / t_off - 1.0)
        );
    }
    let n = tilt_cases.len() as f64;
    println!(
        "{:<12} {:>10.2} {:>13.2} {:>+9.1}%",
        "MEAN",
        sum_off / n,
        sum_on / n,
        100.0 * (sum_on / sum_off - 1.0)
    );
}

#[test]
fn bench_runtime_comparison() {
    const RTI_WARMUP: usize = 100;
    const RTI_MEASURE: usize = 2_000;
    const CONV_WARMUP: usize = 20;
    const CONV_MEASURE: usize = 300;

    let target = Vector3::new(1.0, 1.0, 1.0);
    let simple = simple_setup(target);
    let unc = full_setup(target, false);
    let con = full_setup(target, true);

    let mut s_solver = alloc::boxed::Box::new(SimpleSqpSolver::new());
    let mut u_solver = alloc::boxed::Box::new(FullSqpSolver::new());
    let mut c_solver = alloc::boxed::Box::new(FullSqpSolver::new());

    let cs = cases();
    let n = cs.len();

    // Column-major result buffers: [solver][case]
    let mut rti_us = vec![[0.0f64; 3]; n];
    let mut conv_us = vec![[0.0f64; 3]; n];
    let mut conv_iters = vec![[0usize; 3]; n];
    let mut conv_ok = vec![[false; 3]; n];
    // Diverged solve: non-finite cost or control — timing still measures
    // real work done, but the result is unusable.
    let mut conv_div = vec![[false; 3]; n];

    for (ci, c) in cs.iter().enumerate() {
        let sx0 = simple_x0(c);
        let fx0 = full_x0(c);

        // ── RTI mode: max_iters = 1 (firmware call pattern) ──
        let (t, _) = time_solve(
            &mut s_solver, &simple.problem, &sx0, &simple.x_refs, &simple.u_refs,
            &simple.u_init, 1, RTI_WARMUP, RTI_MEASURE,
        );
        rti_us[ci][0] = t;
        let (t, _) = time_solve(
            &mut u_solver, &unc.problem, &fx0, &unc.x_refs, &unc.u_refs, &unc.u_init, 1,
            RTI_WARMUP, RTI_MEASURE,
        );
        rti_us[ci][1] = t;
        let (t, _) = time_solve(
            &mut c_solver, &con.problem, &fx0, &con.x_refs, &con.u_refs, &con.u_init, 1,
            RTI_WARMUP, RTI_MEASURE,
        );
        rti_us[ci][2] = t;

        // ── Convergence mode: max_iters = CONV_MAX_ITERS ──
        let (t, r) = time_solve(
            &mut s_solver, &simple.problem, &sx0, &simple.x_refs, &simple.u_refs,
            &simple.u_init, CONV_MAX_ITERS, CONV_WARMUP, CONV_MEASURE,
        );
        conv_us[ci][0] = t;
        conv_iters[ci][0] = r.iters;
        conv_ok[ci][0] = r.converged;
        conv_div[ci][0] = !r.cost.is_finite() || !s_solver.u_bar()[0][0].is_finite();
        let (t, r) = time_solve(
            &mut u_solver, &unc.problem, &fx0, &unc.x_refs, &unc.u_refs, &unc.u_init,
            CONV_MAX_ITERS, CONV_WARMUP, CONV_MEASURE,
        );
        conv_us[ci][1] = t;
        conv_iters[ci][1] = r.iters;
        conv_ok[ci][1] = r.converged;
        conv_div[ci][1] = !r.cost.is_finite() || !u_solver.u_bar()[0][0].is_finite();
        let (t, r) = time_solve(
            &mut c_solver, &con.problem, &fx0, &con.x_refs, &con.u_refs, &con.u_init,
            CONV_MAX_ITERS, CONV_WARMUP, CONV_MEASURE,
        );
        conv_us[ci][2] = t;
        conv_iters[ci][2] = r.iters;
        conv_ok[ci][2] = r.converged;
        conv_div[ci][2] = !r.cost.is_finite() || !c_solver.u_bar()[0][0].is_finite();
    }

    // ── Report ────────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "RTI mode — one SQP iteration per call (max_iters=1), {} calls after {} warm-up",
        RTI_MEASURE, RTI_WARMUP
    );
    println!("  simple = SimpleSqpSolver (NX=10) | full = FullSqpSolver (NX=13) | full+sc = + state constraints");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "case", "simple[µs]", "full[µs]", "full+sc[µs]", "full/simple", "sc/full"
    );
    for (ci, c) in cs.iter().enumerate() {
        println!(
            "{:<12} {:>12.2} {:>12.2} {:>12.2} {:>11.2}× {:>9.3}×",
            c.label,
            rti_us[ci][0],
            rti_us[ci][1],
            rti_us[ci][2],
            rti_us[ci][1] / rti_us[ci][0],
            rti_us[ci][2] / rti_us[ci][1],
        );
    }
    let mean = |col: usize, buf: &Vec<[f64; 3]>| -> f64 {
        buf.iter().map(|r| r[col]).sum::<f64>() / n as f64
    };
    println!(
        "{:<12} {:>12.2} {:>12.2} {:>12.2} {:>11.2}× {:>9.3}×",
        "MEAN",
        mean(0, &rti_us),
        mean(1, &rti_us),
        mean(2, &rti_us),
        mean(1, &rti_us) / mean(0, &rti_us),
        mean(2, &rti_us) / mean(1, &rti_us),
    );

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "Convergence mode — solve to KKT tol {} (max {} iters), {} calls after {} warm-up",
        KKT_TOL, CONV_MAX_ITERS, CONV_MEASURE, CONV_WARMUP
    );
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "{:<12} {:>14} {:>14} {:>16} {:>14}",
        "case", "simple[µs|it]", "full[µs|it]", "full+sc[µs|it]", "sc/full time"
    );
    for (ci, c) in cs.iter().enumerate() {
        let fmt = |us: f64, it: usize, ok: bool, div: bool| {
            let mark = if div {
                "D"
            } else if ok {
                " "
            } else {
                "!"
            };
            format!("{:>8.1}|{:>2}{}", us, it, mark)
        };
        println!(
            "{:<12} {:>14} {:>14} {:>16} {:>13.2}×",
            c.label,
            fmt(conv_us[ci][0], conv_iters[ci][0], conv_ok[ci][0], conv_div[ci][0]),
            fmt(conv_us[ci][1], conv_iters[ci][1], conv_ok[ci][1], conv_div[ci][1]),
            fmt(conv_us[ci][2], conv_iters[ci][2], conv_ok[ci][2], conv_div[ci][2]),
            conv_us[ci][2] / conv_us[ci][1],
        );
    }
    println!(
        "{:<12} {:>11.1}    {:>11.1}    {:>13.1}    {:>13.2}×",
        "MEAN",
        mean(0, &conv_us),
        mean(1, &conv_us),
        mean(2, &conv_us),
        mean(2, &conv_us) / mean(1, &conv_us),
    );
    println!("  ('!' = hit the iteration cap without meeting KKT tolerance; 'D' = diverged, NaN result)");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");

    // Sanity assertions.
    for ci in 0..n {
        for s in 0..3 {
            assert!(
                rti_us[ci][s].is_finite() && rti_us[ci][s] > 0.0,
                "bad RTI timing case {ci} solver {s}"
            );
            assert!(
                conv_us[ci][s].is_finite() && conv_us[ci][s] > 0.0,
                "bad convergence timing case {ci} solver {s}"
            );
        }
    }
}
