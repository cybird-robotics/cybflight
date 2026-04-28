//! Equivalence tests for the SQP solver optimisations applied in this branch:
//!   1. Skip post-`forward_sweep` re-propagation on the last iteration.
//!   2. Exploit the structurally-zero columns of `df/dx` (positions don't
//!      feed back into dynamics) so cols 0..3 of `A` / rows 0..3 of `A^T`
//!      collapse to identity-copies in the backward sweep.
//!   3. Refactor `H_uu` build as `(psi_nz @ bk_nz)` followed by
//!      `bk_nz^T @ tmp` instead of the original 4-deep loop.
//!
//! The test pins behavioural equivalence: a faithful replica of the
//! pre-optimisation solver (`BaselineSolver`, lifted verbatim from `HEAD`'s
//! `sqp_solver.rs`) is run alongside the live `SimpleSqpSolver` on a battery
//! of inputs (hover, tilt, translation, multi-iter, warm-start chains,
//! random fuzz). The optimised `u_bar` must match the baseline within a
//! tight float-reorder tolerance.
//!
//! Run with:
//!   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --test sqp_optimization_equivalence --release -- --nocapture

extern crate alloc;

use alloc::boxed::Box;
use cybflight_core::mpc::{
    QuadDynamicsModel, QuadModel, SimpleQuadProblem, SimpleSqpSolver, quad_model,
};
use nalgebra::{SMatrix, SVector};

const NX: usize = quad_model::NX;
const NU: usize = quad_model::NU;
const N: usize = quad_model::N;
const NP1: usize = N + 1;

// ───────────────────────────────────────────────────────────────────────────
// Baseline solver — verbatim copy of the pre-optimisation SQP solver
// (extracted from `HEAD:crates/cybflight_core/src/mpc/sqp_solver.rs`).
//
// This is intentionally an exact line-for-line replica of the baseline
// implementation: the original triple-nested H_uu build, the dense
// `A^T @ psi` / `at_psi @ A` matmuls (no col-zero exploitation), and the
// always-on post-`forward_sweep` re-propagation. Drift between this and
// the live `SimpleSqpSolver` is exactly the behavioural delta of the
// optimisations in this branch.
// ───────────────────────────────────────────────────────────────────────────

fn cholesky_inv_4x4_baseline<const NU_LOCAL: usize>(
    m: &SMatrix<f32, NU_LOCAL, NU_LOCAL>,
) -> SMatrix<f32, NU_LOCAL, NU_LOCAL> {
    const { assert!(NU_LOCAL == 4) };

    for &reg in &[1e-4, 1e-3, 1e-2] {
        let mut mr = *m;
        for i in 0..4 {
            mr[(i, i)] += reg;
        }
        let mut l = SMatrix::<f32, NU_LOCAL, NU_LOCAL>::zeros();
        let d0 = mr[(0, 0)];
        if d0 <= 0.0 {
            continue;
        }
        l[(0, 0)] = libm::sqrtf(d0);
        let l00i = 1.0 / l[(0, 0)];
        l[(1, 0)] = mr[(1, 0)] * l00i;
        l[(2, 0)] = mr[(2, 0)] * l00i;
        l[(3, 0)] = mr[(3, 0)] * l00i;
        let d1 = mr[(1, 1)] - l[(1, 0)] * l[(1, 0)];
        if d1 <= 0.0 {
            continue;
        }
        l[(1, 1)] = libm::sqrtf(d1);
        let l11i = 1.0 / l[(1, 1)];
        l[(2, 1)] = (mr[(2, 1)] - l[(2, 0)] * l[(1, 0)]) * l11i;
        l[(3, 1)] = (mr[(3, 1)] - l[(3, 0)] * l[(1, 0)]) * l11i;
        let d2 = mr[(2, 2)] - l[(2, 0)] * l[(2, 0)] - l[(2, 1)] * l[(2, 1)];
        if d2 <= 0.0 {
            continue;
        }
        l[(2, 2)] = libm::sqrtf(d2);
        let l22i = 1.0 / l[(2, 2)];
        l[(3, 2)] = (mr[(3, 2)] - l[(3, 0)] * l[(2, 0)] - l[(3, 1)] * l[(2, 1)]) * l22i;
        let d3 = mr[(3, 3)] - l[(3, 0)] * l[(3, 0)] - l[(3, 1)] * l[(3, 1)] - l[(3, 2)] * l[(3, 2)];
        if d3 <= 0.0 {
            continue;
        }
        l[(3, 3)] = libm::sqrtf(d3);

        let mut li = SMatrix::<f32, NU_LOCAL, NU_LOCAL>::zeros();
        li[(0, 0)] = 1.0 / l[(0, 0)];
        li[(1, 1)] = 1.0 / l[(1, 1)];
        li[(2, 2)] = 1.0 / l[(2, 2)];
        li[(3, 3)] = 1.0 / l[(3, 3)];
        li[(1, 0)] = -l[(1, 0)] * li[(0, 0)] * li[(1, 1)];
        li[(2, 0)] = -(l[(2, 0)] * li[(0, 0)] + l[(2, 1)] * li[(1, 0)]) * li[(2, 2)];
        li[(2, 1)] = -l[(2, 1)] * li[(1, 1)] * li[(2, 2)];
        li[(3, 0)] = -(l[(3, 0)] * li[(0, 0)] + l[(3, 1)] * li[(1, 0)] + l[(3, 2)] * li[(2, 0)])
            * li[(3, 3)];
        li[(3, 1)] = -(l[(3, 1)] * li[(1, 1)] + l[(3, 2)] * li[(2, 1)]) * li[(3, 3)];
        li[(3, 2)] = -l[(3, 2)] * li[(2, 2)] * li[(3, 3)];

        let mut result = SMatrix::<f32, NU_LOCAL, NU_LOCAL>::zeros();
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += li[(k, i)] * li[(k, j)];
                }
                result[(i, j)] = s;
            }
        }
        let mut ok = true;
        'check: for i in 0..4 {
            for j in 0..4 {
                if !result[(i, j)].is_finite() {
                    ok = false;
                    break 'check;
                }
            }
        }
        if ok {
            return result;
        }
    }

    let mut diag_max = 1.0f32;
    for i in 0..4 {
        diag_max = diag_max.max(m[(i, i)].abs());
    }
    let s = 1.0 / diag_max;
    let mut r = SMatrix::<f32, NU_LOCAL, NU_LOCAL>::zeros();
    for i in 0..4 {
        r[(i, i)] = s;
    }
    r
}

struct BaselineSolver {
    a: [SMatrix<f32, NX, NX>; N],
    b: [SMatrix<f32, NX, NU>; N],
    q: [SVector<f32, NX>; NP1],
    r: [SVector<f32, NU>; N],
    qm: [SMatrix<f32, NX, NX>; NP1],
    rm: [SVector<f32, NU>; N],
    gain_k: [SMatrix<f32, NU, NX>; N],
    gain_kk: [SVector<f32, NU>; N],
    pp: [SMatrix<f32, NX, NX>; NP1],
    pv: [SVector<f32, NX>; NP1],
    x_bar: [SVector<f32, NX>; NP1],
    u_bar: [SVector<f32, NU>; N],
}

impl BaselineSolver {
    fn new() -> Self {
        Self {
            a: [SMatrix::<f32, NX, NX>::zeros(); N],
            b: [SMatrix::<f32, NX, NU>::zeros(); N],
            q: [SVector::<f32, NX>::zeros(); NP1],
            r: [SVector::<f32, NU>::zeros(); N],
            qm: [SMatrix::<f32, NX, NX>::zeros(); NP1],
            rm: [SVector::<f32, NU>::zeros(); N],
            gain_k: [SMatrix::<f32, NU, NX>::zeros(); N],
            gain_kk: [SVector::<f32, NU>::zeros(); N],
            pp: [SMatrix::<f32, NX, NX>::zeros(); NP1],
            pv: [SVector::<f32, NX>::zeros(); NP1],
            x_bar: [SVector::<f32, NX>::zeros(); NP1],
            u_bar: [SVector::<f32, NU>::zeros(); N],
        }
    }

    fn backward_sweep(&mut self) {
        let bnz_start: usize = <QuadModel as QuadDynamicsModel<NX, NU>>::BNZ_START;
        let bnz_len: usize = <QuadModel as QuadDynamicsModel<NX, NU>>::BNZ_LEN;

        self.pp[N] = self.qm[N];
        self.pv[N] = self.q[N];

        for k in (0..N).rev() {
            let psi = self.pp[k + 1];
            let pv = self.pv[k + 1];
            let bk = self.b[k];

            let mut h_uu = SMatrix::<f32, NU, NU>::from_diagonal(&self.rm[k]);
            for i in 0..NU {
                for j in 0..NU {
                    let mut s = 0.0;
                    for r in 0..bnz_len {
                        let ri = bnz_start + r;
                        for c in 0..bnz_len {
                            let ci = bnz_start + c;
                            s += bk[(ri, i)] * psi[(ri, ci)] * bk[(ci, j)];
                        }
                    }
                    h_uu[(i, j)] += s;
                }
            }

            let ak = self.a[k];
            let mut at_psi = SMatrix::<f32, NX, NX>::zeros();
            for i in 0..NX {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NX {
                        s += ak[(m, i)] * psi[(m, j)];
                    }
                    at_psi[(i, j)] = s;
                }
            }

            let mut h_xu = SMatrix::<f32, NX, NU>::zeros();
            for i in 0..NX {
                for j in 0..NU {
                    let mut s = 0.0;
                    for c in 0..bnz_len {
                        let ci = bnz_start + c;
                        s += at_psi[(i, ci)] * bk[(ci, j)];
                    }
                    h_xu[(i, j)] = s;
                }
            }

            let mut h_u = SVector::<f32, NU>::zeros();
            for i in 0..NU {
                let mut s = self.r[k][i];
                for r in 0..bnz_len {
                    s += bk[(bnz_start + r, i)] * pv[bnz_start + r];
                }
                h_u[i] = s;
            }

            let mut h_x = SVector::<f32, NX>::zeros();
            for i in 0..NX {
                let mut s = self.q[k][i];
                for m in 0..NX {
                    s += ak[(m, i)] * pv[m];
                }
                h_x[i] = s;
            }

            let h_uu_inv = cholesky_inv_4x4_baseline::<NU>(&h_uu);

            for i in 0..NU {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NU {
                        s += h_uu_inv[(i, m)] * h_xu[(j, m)];
                    }
                    self.gain_k[k][(i, j)] = -s;
                }
            }

            for i in 0..NU {
                let mut s = 0.0;
                for m in 0..NU {
                    s += h_uu_inv[(i, m)] * h_u[m];
                }
                self.gain_kk[k][i] = -s;
            }

            let mut at_psi_a = SMatrix::<f32, NX, NX>::zeros();
            for i in 0..NX {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NX {
                        s += at_psi[(i, m)] * ak[(m, j)];
                    }
                    at_psi_a[(i, j)] = s;
                }
            }
            let mut hxu_gk = SMatrix::<f32, NX, NX>::zeros();
            for i in 0..NX {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NU {
                        s += h_xu[(i, m)] * self.gain_k[k][(m, j)];
                    }
                    hxu_gk[(i, j)] = s;
                }
            }
            for i in 0..NX {
                for j in 0..NX {
                    self.pp[k][(i, j)] = self.qm[k][(i, j)] + at_psi_a[(i, j)] + hxu_gk[(i, j)];
                }
            }

            for i in 0..NX {
                let mut s = h_x[i];
                for m in 0..NU {
                    s += h_xu[(i, m)] * self.gain_kk[k][m];
                }
                self.pv[k][i] = s;
            }
        }
    }

    fn forward_sweep(&mut self, x_init: &SVector<f32, NX>, problem: &SimpleQuadProblem) {
        let mut dx = SVector::<f32, NX>::zeros();
        for i in 0..NX {
            dx[i] = x_init[i] - self.x_bar[0][i];
        }
        for k in 0..N {
            let mut du = SVector::<f32, NU>::zeros();
            for i in 0..NU {
                let mut s = self.gain_kk[k][i];
                for j in 0..NX {
                    s += self.gain_k[k][(i, j)] * dx[j];
                }
                du[i] = s;
            }
            let u_old = self.u_bar[k];
            let mut u_new = SVector::<f32, NU>::zeros();
            for i in 0..NU {
                u_new[i] = u_old[i] + du[i];
            }
            self.u_bar[k] = problem.clamp(&u_new);

            let mut du_actual = SVector::<f32, NU>::zeros();
            for i in 0..NU {
                du_actual[i] = self.u_bar[k][i] - u_old[i];
            }
            let mut dx_new = SVector::<f32, NX>::zeros();
            for i in 0..NX {
                let mut s = 0.0;
                for j in 0..NX {
                    s += self.a[k][(i, j)] * dx[j];
                }
                for j in 0..NU {
                    s += self.b[k][(i, j)] * du_actual[j];
                }
                dx_new[i] = s;
            }
            dx = dx_new;
        }
    }

    fn solve(
        &mut self,
        problem: &SimpleQuadProblem,
        x0: &SVector<f32, NX>,
        x_refs: &[SVector<f32, NX>; NP1],
        u_refs: &[SVector<f32, NU>; N],
        u_init: &[SVector<f32, NU>; N],
        max_iters: usize,
        kkt_tol: f32,
    ) {
        self.u_bar = *u_init;

        let mut hess_xx = SMatrix::<f32, NX, NX>::zeros();
        let mut r_diag = SVector::<f32, NU>::zeros();
        let mut grad_x = SVector::<f32, NX>::zeros();
        let mut grad_u = SVector::<f32, NU>::zeros();

        self.x_bar[0] = *x0;
        <QuadModel as QuadDynamicsModel<NX, NU>>::normalize_quat(&mut self.x_bar[0]);
        for k in 0..N {
            self.x_bar[k + 1] = problem.propagate(&self.x_bar[k], &self.u_bar[k]);
        }

        for _iteration in 0..max_iters {
            for k in 0..N {
                let (fx, fu) = problem.linearize(&self.x_bar[k], &self.u_bar[k]);
                self.a[k] = fx;
                self.b[k] = fu;
            }

            for k in 0..N {
                let _ = problem.stage_cost_hess_grad(
                    &self.x_bar[k],
                    &self.u_bar[k],
                    &x_refs[k],
                    &u_refs[k],
                    &mut hess_xx,
                    &mut r_diag,
                    &mut grad_x,
                    &mut grad_u,
                );
                self.qm[k] = hess_xx;
                self.rm[k] = r_diag;
                self.q[k] = grad_x;
                self.r[k] = grad_u;
            }
            let _ = problem.terminal_cost_hess_grad(
                &self.x_bar[N],
                &x_refs[N],
                &mut grad_x,
                &mut hess_xx,
            );
            self.qm[N] = hess_xx;
            self.q[N] = grad_x;

            self.backward_sweep();

            let mut kkt_norm = 0.0f32;
            for k in 0..N {
                for i in 0..NU {
                    kkt_norm = kkt_norm.max(self.gain_kk[k][i].abs());
                }
            }

            self.forward_sweep(x0, problem);

            // Baseline ALWAYS re-propagates (this is the optimisation #1
            // discards on the last iteration).
            self.x_bar[0] = *x0;
            <QuadModel as QuadDynamicsModel<NX, NU>>::normalize_quat(&mut self.x_bar[0]);
            for k in 0..N {
                self.x_bar[k + 1] = problem.propagate(&self.x_bar[k], &self.u_bar[k]);
            }

            if kkt_norm < kkt_tol {
                break;
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────

fn make_problem() -> SimpleQuadProblem {
    let model = QuadModel::default();
    SimpleQuadProblem::with_rk4(model, N)
}

fn quat_xyzw(yaw: f32, pitch: f32, roll: f32) -> [f32; 4] {
    let (cr, sr) = (libm::cosf(roll * 0.5), libm::sinf(roll * 0.5));
    let (cp, sp) = (libm::cosf(pitch * 0.5), libm::sinf(pitch * 0.5));
    let (cy, sy) = (libm::cosf(yaw * 0.5), libm::sinf(yaw * 0.5));
    let qw = cr * cp * cy + sr * sp * sy;
    let qx = sr * cp * cy - cr * sp * sy;
    let qy = cr * sp * cy + sr * cp * sy;
    let qz = cr * cp * sy - sr * sp * cy;
    [qx, qy, qz, qw]
}

fn make_state(p: [f32; 3], yaw: f32, pitch: f32, roll: f32, v: [f32; 3]) -> SVector<f32, NX> {
    let q = quat_xyzw(yaw, pitch, roll);
    SVector::<f32, NX>::from_row_slice(&[p[0], p[1], p[2], q[0], q[1], q[2], q[3], v[0], v[1], v[2]])
}

fn hover_xref(target_pos: [f32; 3]) -> [SVector<f32, NX>; NP1] {
    let mut x = SVector::<f32, NX>::zeros();
    x[0] = target_pos[0];
    x[1] = target_pos[1];
    x[2] = target_pos[2];
    x[6] = 1.0; // qw = 1
    [x; NP1]
}

fn hover_uref(mass: f32) -> [SVector<f32, NU>; N] {
    let hover_thrust = mass * 9.81;
    let u = SVector::<f32, NU>::from_row_slice(&[hover_thrust, 0.0, 0.0, 0.0]);
    [u; N]
}

/// Per-channel diff between two `u_bar` arrays. Returns
/// `(max_abs, max_rel)`. `max_rel` uses each channel's bounds magnitude
/// as the denominator so float reorder on a near-zero rate looks small.
fn ubar_diff(
    a: &[SVector<f32, NU>; N],
    b: &[SVector<f32, NU>; N],
    bounds: &[[f32; 2]; NU],
) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for k in 0..N {
        for i in 0..NU {
            let d = (a[k][i] - b[k][i]).abs();
            let scale = (bounds[i][1] - bounds[i][0]).abs().max(1e-3);
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / scale);
        }
    }
    (max_abs, max_rel)
}

/// Tiny xorshift PRNG for deterministic fuzz inputs (no `rand` dep).
struct Xorshift(u32);
impl Xorshift {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0xDEADBEEF } else { seed })
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Single-shot equivalence check on a given initial state and reference.
fn assert_equiv(
    label: &str,
    x0: &SVector<f32, NX>,
    x_refs: &[SVector<f32, NX>; NP1],
    u_refs: &[SVector<f32, NU>; N],
    u_init: &[SVector<f32, NU>; N],
    max_iters: usize,
    abs_tol: f32,
    rel_tol: f32,
) {
    let problem = make_problem();
    let bounds = problem.model.u_bounds;

    let mut opt = Box::new(SimpleSqpSolver::new());
    let mut base = Box::new(BaselineSolver::new());

    let _ = opt.solve(&problem, x0, x_refs, u_refs, u_init, max_iters, 1e-3);
    base.solve(&problem, x0, x_refs, u_refs, u_init, max_iters, 1e-3);

    let opt_u = *opt.u_bar();
    let base_u = base.u_bar;
    let (max_abs, max_rel) = ubar_diff(&opt_u, &base_u, &bounds);

    println!(
        "[{label}] u_bar max_abs={:.3e}, max_rel={:.3e}, tol_abs={:.0e}, tol_rel={:.0e}",
        max_abs, max_rel, abs_tol, rel_tol
    );
    assert!(
        max_abs <= abs_tol && max_rel <= rel_tol,
        "[{label}] u_bar drift exceeds tolerance: max_abs={:.3e} (tol {:.0e}), max_rel={:.3e} (tol {:.0e})",
        max_abs,
        abs_tol,
        max_rel,
        rel_tol
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

const ABS_TOL: f32 = 5e-4; // ~0.5 mN of thrust or ~5e-4 rad/s on body rate.
const REL_TOL: f32 = 5e-5; // 0.005% of channel range.

#[test]
fn equiv_hover_at_origin() {
    let x0 = make_state([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]);
    let x_refs = hover_xref([0.0, 0.0, 0.0]);
    let u_refs = hover_uref(0.58);
    assert_equiv("hover_at_origin", &x0, &x_refs, &u_refs, &u_refs, 1, ABS_TOL, REL_TOL);
}

#[test]
fn equiv_offset_position_target() {
    let x0 = make_state([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]);
    let x_refs = hover_xref([1.0, -0.5, 2.0]);
    let u_refs = hover_uref(0.58);
    assert_equiv("offset_position_target", &x0, &x_refs, &u_refs, &u_refs, 1, ABS_TOL, REL_TOL);
}

#[test]
fn equiv_tilted_attitude() {
    let cases = [
        ("roll_15deg", 0.0, 0.0, 0.262),
        ("pitch_25deg", 0.0, 0.436, 0.0),
        ("yaw_45deg", 0.785, 0.0, 0.0),
        ("rpy_combined", 0.3, 0.2, 0.1),
        ("near_inverted", 0.0, 1.4, 0.6),
    ];
    for (lab, y, p, r) in cases {
        let x0 = make_state([0.0, 0.0, 0.0], y, p, r, [0.0, 0.0, 0.0]);
        let x_refs = hover_xref([0.0, 0.0, 0.0]);
        let u_refs = hover_uref(0.58);
        assert_equiv(lab, &x0, &x_refs, &u_refs, &u_refs, 1, ABS_TOL, REL_TOL);
    }
}

#[test]
fn equiv_with_velocity() {
    let cases = [
        ("v_up_2", [0.0, 0.0, 2.0]),
        ("v_lateral_3", [3.0, -1.5, 0.0]),
        ("v_diag_5", [4.0, 4.0, 2.0]),
        ("v_extreme_8", [8.0, -6.0, 3.0]),
    ];
    for (lab, v) in cases {
        let x0 = make_state([0.5, -0.2, 0.8], 0.1, 0.05, -0.05, v);
        let x_refs = hover_xref([1.0, 1.0, 1.0]);
        let u_refs = hover_uref(0.58);
        assert_equiv(lab, &x0, &x_refs, &u_refs, &u_refs, 1, ABS_TOL, REL_TOL);
    }
}

#[test]
fn equiv_multi_iter() {
    // max_iters > 1 exercises the inter-iteration handoff. The optimised
    // solver still re-propagates between iterations; only the FINAL
    // re-propagation is skipped. Result must still match baseline.
    let x0 = make_state([0.5, 0.5, 0.0], 0.0, 0.2, 0.1, [0.5, 0.0, 0.0]);
    let x_refs = hover_xref([2.0, 2.0, 1.5]);
    let u_refs = hover_uref(0.58);
    for iters in [2, 3, 5, 10] {
        let lab = alloc::format!("multi_iter_{iters}");
        assert_equiv(&lab, &x0, &x_refs, &u_refs, &u_refs, iters, ABS_TOL, REL_TOL);
    }
}

#[test]
fn equiv_warm_start_chain() {
    // Closed-loop sim: at each tick, advance the plant under the issued
    // u0, refresh `u_warm` from the previous solve, and resolve. Chain
    // length 200 ticks. Diff per tick must stay bounded — float-reorder
    // shouldn't compound into divergence under a stable closed loop.
    let problem = make_problem();
    let bounds = problem.model.u_bounds;
    let mass = problem.model.mass;

    let mut opt = Box::new(SimpleSqpSolver::new());
    let mut base = Box::new(BaselineSolver::new());

    let mut x_opt = make_state([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]);
    let mut x_base = x_opt;
    let x_refs = hover_xref([2.0, 1.0, 1.5]);
    let u_refs = hover_uref(mass);
    let mut u_warm_opt = u_refs;
    let mut u_warm_base = u_refs;

    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;
    for tick in 0..200 {
        let _ = opt.solve(&problem, &x_opt, &x_refs, &u_refs, &u_warm_opt, 1, 1e-3);
        base.solve(&problem, &x_base, &x_refs, &u_refs, &u_warm_base, 1, 1e-3);

        let opt_ub = *opt.u_bar();
        let base_ub = base.u_bar;
        let (a, r) = ubar_diff(&opt_ub, &base_ub, &bounds);
        worst_abs = worst_abs.max(a);
        worst_rel = worst_rel.max(r);

        u_warm_opt = opt_ub;
        u_warm_base = base_ub;

        // Advance plant under the issued u0 (RK4 propagation).
        x_opt = problem.propagate(&x_opt, &opt_ub[0]);
        x_base = problem.propagate(&x_base, &base_ub[0]);

        if tick % 50 == 0 {
            println!(
                "[warm_start_chain] tick={tick} cur_abs={:.3e} cur_rel={:.3e} \
                 |x_opt-x_base|max={:.3e}",
                a,
                r,
                (x_opt - x_base).iter().map(|v| v.abs()).fold(0.0f32, f32::max),
            );
        }
    }
    println!(
        "[warm_start_chain] worst_abs={:.3e}, worst_rel={:.3e}",
        worst_abs, worst_rel
    );
    // Chain tolerance is looser — float reorder accumulates over the closed
    // loop, but the solvers should still track each other to within a few
    // percent of channel range over 200 ticks.
    assert!(
        worst_abs <= 5e-2 && worst_rel <= 5e-3,
        "warm-start chain drift: worst_abs={:.3e}, worst_rel={:.3e}",
        worst_abs,
        worst_rel
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Runtime benchmark — wall-clock comparison of optimised vs baseline solve.
// ───────────────────────────────────────────────────────────────────────────

fn bench_one(label: &str, n_solves: usize, fresh_each_solve: bool) {
    use core::hint::black_box;
    use std::time::Instant;

    let problem = make_problem();
    let mass = problem.model.mass;
    let x_refs = hover_xref([2.0, 1.0, 1.5]);
    let u_refs = hover_uref(mass);

    let mut rng = Xorshift::new(0xBADC0DE);
    let mut x0_set = alloc::vec::Vec::with_capacity(n_solves);
    let mut u_init_set = alloc::vec::Vec::with_capacity(n_solves);
    for _ in 0..n_solves {
        let p = [rng.unit() * 1.0, rng.unit() * 1.0, rng.unit() * 0.5];
        let v = [rng.unit() * 2.0, rng.unit() * 2.0, rng.unit() * 1.0];
        x0_set.push(make_state(p, 0.0, rng.unit() * 0.3, rng.unit() * 0.3, v));
        let mut u_init = u_refs;
        for k in 0..N {
            u_init[k][0] += rng.unit() * 0.5;
            u_init[k][1] += rng.unit() * 0.2;
            u_init[k][2] += rng.unit() * 0.2;
            u_init[k][3] += rng.unit() * 0.1;
        }
        u_init_set.push(u_init);
    }

    let mut opt = Box::new(SimpleSqpSolver::new());
    let mut base = Box::new(BaselineSolver::new());

    // Warmup — populate caches, page-in workspaces.
    for i in 0..16.min(n_solves) {
        let _ = opt.solve(&problem, &x0_set[i], &x_refs, &u_refs, &u_init_set[i], 1, 1e-3);
        base.solve(&problem, &x0_set[i], &x_refs, &u_refs, &u_init_set[i], 1, 1e-3);
    }

    let mut warm_opt = u_refs;
    let t_opt_start = Instant::now();
    for i in 0..n_solves {
        let warm = if fresh_each_solve { &u_init_set[i] } else { &warm_opt };
        let _ = opt.solve(&problem, &x0_set[i], &x_refs, &u_refs, warm, 1, 1e-3);
        warm_opt = *opt.u_bar();
        black_box(&warm_opt);
    }
    let t_opt = t_opt_start.elapsed();

    let mut warm_base = u_refs;
    let t_base_start = Instant::now();
    for i in 0..n_solves {
        let warm = if fresh_each_solve { &u_init_set[i] } else { &warm_base };
        base.solve(&problem, &x0_set[i], &x_refs, &u_refs, warm, 1, 1e-3);
        warm_base = base.u_bar;
        black_box(&warm_base);
    }
    let t_base = t_base_start.elapsed();

    let opt_us = (t_opt.as_secs_f64() * 1e6) / n_solves as f64;
    let base_us = (t_base.as_secs_f64() * 1e6) / n_solves as f64;
    let speedup = base_us / opt_us;
    let saved_pct = (1.0 - opt_us / base_us) * 100.0;
    println!(
        "[bench/{label}] n={n_solves} optimised={opt_us:.2}µs/solve  baseline={base_us:.2}µs/solve  \
         speedup={speedup:.2}x  saved={saved_pct:.1}%"
    );
}

#[test]
fn bench_runtime_vs_baseline() {
    println!("--- SQP runtime: optimised (this branch) vs baseline (HEAD) ---");
    bench_one("fresh_warm",  500,  true);
    bench_one("fresh_warm",  2000, true);
    bench_one("rolled_warm", 500,  false);
    bench_one("rolled_warm", 2000, false);
}

#[test]
fn equiv_fuzz_random() {
    let mut rng = Xorshift::new(0xC0FFEE);
    let problem = make_problem();
    let bounds = problem.model.u_bounds;
    let mass = problem.model.mass;

    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;
    let mut worst_label = alloc::string::String::new();

    let n_cases = 100usize;
    for case in 0..n_cases {
        let p = [rng.unit() * 2.0, rng.unit() * 2.0, rng.unit() * 1.0];
        let yaw = rng.unit() * core::f32::consts::PI;
        let pitch = rng.unit() * 0.6;
        let roll = rng.unit() * 0.6;
        let v = [rng.unit() * 4.0, rng.unit() * 4.0, rng.unit() * 2.0];
        let x0 = make_state(p, yaw, pitch, roll, v);
        let target = [rng.unit() * 3.0, rng.unit() * 3.0, rng.unit() * 2.0 + 0.5];
        let x_refs = hover_xref(target);
        let u_refs = hover_uref(mass);

        // Vary u_init around hover so the warm-start path is exercised.
        let mut u_init = u_refs;
        for k in 0..N {
            u_init[k][0] += rng.unit() * 1.0; // ±1 N
            u_init[k][1] += rng.unit() * 0.5;
            u_init[k][2] += rng.unit() * 0.5;
            u_init[k][3] += rng.unit() * 0.3;
        }

        let mut opt = Box::new(SimpleSqpSolver::new());
        let mut base = Box::new(BaselineSolver::new());
        let _ = opt.solve(&problem, &x0, &x_refs, &u_refs, &u_init, 1, 1e-3);
        base.solve(&problem, &x0, &x_refs, &u_refs, &u_init, 1, 1e-3);

        let (a, r) = ubar_diff(opt.u_bar(), &base.u_bar, &bounds);
        if a > worst_abs {
            worst_abs = a;
            worst_label = alloc::format!(
                "case#{case} p={p:?} ypr=({yaw:.2},{pitch:.2},{roll:.2}) v={v:?}"
            );
        }
        worst_rel = worst_rel.max(r);
    }
    println!(
        "[fuzz_random] n={n_cases}, worst_abs={:.3e}, worst_rel={:.3e}, worst={worst_label}",
        worst_abs, worst_rel
    );
    assert!(
        worst_abs <= ABS_TOL && worst_rel <= REL_TOL,
        "fuzz drift: worst_abs={:.3e}, worst_rel={:.3e}, worst case: {}",
        worst_abs,
        worst_rel,
        worst_label
    );
}
