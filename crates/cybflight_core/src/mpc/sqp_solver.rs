//! SQP solver — multiple-shooting with structured Riccati backward sweep.
//!
//! Sparse-B optimisation: the solver assumes the dynamics input Jacobian
//! `df/du` has only a contiguous block of nonzero rows. The starting row
//! and length are sourced from the model via the
//! [`super::QuadDynamicsModel`] associated constants `BNZ_START` /
//! `BNZ_LEN`, so the same solver code handles both [`super::FullQuadModel`]
//! (rows 7-12) and [`super::QuadModel`] (rows 3-9).
//!
//! All workspace arrays are stack-allocated and sized via const generics
//! `<NX, NU, N, NP1>` (where `NP1 = N + 1` is the trajectory length). No
//! heap allocation.
//!
//! NU is pinned to 4 by [`cholesky_inv_4x4`], which carries an unrolled
//! 4×4 Cholesky inverse. Generalising to other NU is a future refactor.

use super::full_quad_model;
use super::mpc_problem::MpcProblem;
use super::quad_model;
use super::QuadDynamicsModel;
use nalgebra::{SMatrix, SVector};

/// Result metadata from the SQP solver.
///
/// The full trajectory (x_bar, u_bar) remains in the solver workspace and
/// can be accessed via [`SqpSolver::u_bar`] / [`SqpSolver::x_bar`].
#[derive(Clone, Copy)]
pub struct SolverResult {
    pub cost: f32,
    pub iters: usize,
    pub converged: bool,
    /// A non-finite value entered the Riccati recursion (NaN/Inf in the
    /// terminal cost, `H_uu`, gains, or cost-to-go). The offending
    /// iteration's step was **rejected**: `u_bar` still holds the last
    /// valid iterate and `converged` is false. Callers should treat the
    /// solve as failed and reset their warm start — the corruption cause
    /// (bad initial state, cost overflow) usually persists across ticks.
    pub diverged: bool,
}

/// True iff every element of the matrix/vector is finite.
#[inline]
fn all_finite<const R: usize, const C: usize>(m: &SMatrix<f32, R, C>) -> bool {
    m.iter().all(|v| v.is_finite())
}

// ── 4x4 Cholesky inverse ────────────────────────────────────────────────
//
// Const-generic over `NU` so it can be called from inside a const-generic
// solver, but the body assumes `NU == 4`. The compile-time `assert!` makes
// instantiating it with any other dimension a build-time error, and after
// monomorphisation with NU=4 the body's literal indices `[0..3]` produce
// the same machine code as a non-generic 4x4 implementation.

fn cholesky_inv_4x4<const NU: usize>(m: &SMatrix<f32, NU, NU>) -> SMatrix<f32, NU, NU> {
    const { assert!(NU == 4, "cholesky_inv_4x4 only supports NU=4") };

    for &reg in &[1e-4, 1e-3, 1e-2] {
        let mut mr = *m;
        for i in 0..4 {
            mr[(i, i)] += reg;
        }

        let mut l = SMatrix::<f32, NU, NU>::zeros();
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

        // Triangular inverse
        let mut li = SMatrix::<f32, NU, NU>::zeros();
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

        // result = li^T @ li
        let mut result = SMatrix::<f32, NU, NU>::zeros();
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += li[(k, i)] * li[(k, j)];
                }
                result[(i, j)] = s;
            }
        }
        // Check finite
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

    // Fallback: scaled identity. The fold skips non-finite diagonals:
    // `f32::max` already ignores NaN (the 1.0 seed survives), but an Inf
    // diagonal would drive the scale to 0 and return the all-zeros
    // matrix — zero gains, zero KKT norm, a false "converged". The
    // backward sweep additionally bails on a non-finite `H_uu` before
    // ever calling this; the guard keeps the function safe standalone.
    let mut diag_max = 1.0f32;
    for i in 0..4 {
        let v = m[(i, i)].abs();
        if v.is_finite() {
            diag_max = diag_max.max(v);
        }
    }
    let s = 1.0 / diag_max;
    let mut r = SMatrix::<f32, NU, NU>::zeros();
    for i in 0..4 {
        r[(i, i)] = s;
    }
    r
}

// ── QP Workspace ────────────────────────────────────────────────────────

struct QpWorkspace<const NX: usize, const NU: usize, const N: usize, const NP1: usize> {
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
    pub x_bar: [SVector<f32, NX>; NP1],
    pub u_bar: [SVector<f32, NU>; N],
}

impl<const NX: usize, const NU: usize, const N: usize, const NP1: usize>
    QpWorkspace<NX, NU, N, NP1>
{
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

    /// Riccati backward sweep with sparse-B optimisation.
    ///
    /// `BNZ_START` and `BNZ_LEN` are pulled from `M` as compile-time
    /// associated constants — after monomorphisation they fold to literal
    /// integers and the loop bounds match what a model-specific hardcoded
    /// solver would produce.
    ///
    /// Returns `false` (bailing at the offending stage) if a non-finite
    /// value is detected in the recursion — terminal cost, `H_uu`, the
    /// gains, or the cost-to-go. The caller must then treat the whole
    /// step as diverged and NOT run the forward sweep. Detecting this
    /// here rather than only at the KKT norm is load-bearing:
    /// `cholesky_inv_4x4` answers a non-finite `H_uu` with its
    /// scaled-identity fallback, laundering the divergence into finite
    /// (garbage) gains that no downstream NaN check can see. The checks
    /// cost O(NX²) per stage against the sweep's O(NX³) products, on
    /// data still warm in cache.
    /// `n` is the active horizon (`≤ N`, the workspace capacity).
    fn backward_sweep<M>(&mut self, n: usize) -> bool
    where
        M: QuadDynamicsModel<NX, NU>,
    {
        // Capture associated constants into locals so LLVM definitely sees
        // them as compile-time constants in the inner loops below.
        let bnz_start: usize = M::BNZ_START;
        let bnz_len: usize = M::BNZ_LEN;
        // First nonzero column of df/dx. Cols `[0, jx_cs)` of dt·jac_x are
        // zero, so cols 0..jx_cs of A and rows 0..jx_cs of A^T are pure
        // identity — the inner products below collapse to direct copies.
        let jx_cs: usize = M::JAC_X_NZ_COL_START;

        self.pp[n] = self.qm[n];
        self.pv[n] = self.q[n];
        // Terminal cost is the recursion seed — a NaN here (e.g. from a
        // non-finite x_refs[N] or a NaN-propagated x_bar[N]) corrupts
        // every stage below.
        if !all_finite(&self.pp[n]) || !all_finite(&self.pv[n]) {
            return false;
        }

        for k in (0..n).rev() {
            let psi = &self.pp[k + 1];
            let pv = &self.pv[k + 1];

            let bk = &self.b[k];

            // H_uu = diag(rm[k]) + bk_nz^T @ psi_nz @ bk_nz  (NU x NU).
            // Factored as `tmp = psi_nz @ bk_nz` (bnz_len × NU) followed by
            // `bk_nz^T @ tmp` to drop the inner-product cost from
            // O(NU² · bnz_len²) to O(bnz_len² · NU + NU² · bnz_len). The
            // intermediate is sized NX × NU (only its first `bnz_len` rows
            // are used) to avoid a fresh const-generic helper.
            let mut h_uu = SMatrix::<f32, NU, NU>::from_diagonal(&self.rm[k]);
            let mut psi_b = SMatrix::<f32, NX, NU>::zeros();
            for r in 0..bnz_len {
                let ri = bnz_start + r;
                for j in 0..NU {
                    let mut s = 0.0;
                    for c in 0..bnz_len {
                        let ci = bnz_start + c;
                        s += psi[(ri, ci)] * bk[(ci, j)];
                    }
                    psi_b[(r, j)] = s;
                }
            }
            for i in 0..NU {
                for j in 0..NU {
                    let mut s = 0.0;
                    for r in 0..bnz_len {
                        let ri = bnz_start + r;
                        s += bk[(ri, i)] * psi_b[(r, j)];
                    }
                    h_uu[(i, j)] += s;
                }
            }

            // at_psi = A^T @ psi  (NX x NX). Rows `[0, jx_cs)` of A^T are
            // identity rows, so those output rows = the corresponding rows
            // of psi without an inner-product loop.
            let ak = &self.a[k];
            let mut at_psi = SMatrix::<f32, NX, NX>::zeros();
            for i in 0..NX {
                if i < jx_cs {
                    for j in 0..NX {
                        at_psi[(i, j)] = psi[(i, j)];
                    }
                } else {
                    for j in 0..NX {
                        let mut s = 0.0;
                        for m in 0..NX {
                            s += ak[(m, i)] * psi[(m, j)];
                        }
                        at_psi[(i, j)] = s;
                    }
                }
            }

            // h_xu = at_psi[:, BNZ] @ bk_nz  (NX x NU)
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

            // h_u = r[k] + bk_nz^T @ pv_nz
            let mut h_u = SVector::<f32, NU>::zeros();
            for i in 0..NU {
                let mut s = self.r[k][i];
                for r in 0..bnz_len {
                    s += bk[(bnz_start + r, i)] * pv[bnz_start + r];
                }
                h_u[i] = s;
            }

            // h_x = q[k] + A^T @ pv. For i < jx_cs, A^T's i-th row is a
            // pure-identity row, so the dot product collapses to pv[i].
            let mut h_x = SVector::<f32, NX>::zeros();
            for i in 0..NX {
                if i < jx_cs {
                    h_x[i] = self.q[k][i] + pv[i];
                } else {
                    let mut s = self.q[k][i];
                    for m in 0..NX {
                        s += ak[(m, i)] * pv[m];
                    }
                    h_x[i] = s;
                }
            }

            // Pre-cholesky bail: this is the one place a non-finite value
            // can be ALIASED rather than propagated — the inverse's
            // scaled-identity fallback is finite for any input, so a NaN
            // or Inf H_uu must be caught before it disappears into
            // plausible-looking gains.
            if !all_finite(&h_uu) {
                return false;
            }
            let h_uu_inv = cholesky_inv_4x4::<NU>(&h_uu);

            // gain_k[k] = -(h_uu_inv @ h_xu^T)  → (NU x NX)
            for i in 0..NU {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NU {
                        s += h_uu_inv[(i, m)] * h_xu[(j, m)];
                    }
                    self.gain_k[k][(i, j)] = -s;
                }
            }

            // gain_kk[k] = -(h_uu_inv @ h_u)
            for i in 0..NU {
                let mut s = 0.0;
                for m in 0..NU {
                    s += h_uu_inv[(i, m)] * h_u[m];
                }
                self.gain_kk[k][i] = -s;
            }

            // pp[k] = qm[k] + at_psi @ A[k] + h_xu @ gain_k[k]. Cols
            // `[0, jx_cs)` of A are pure-identity columns, so those output
            // columns of `at_psi @ A` equal the corresponding cols of `at_psi`.
            let mut at_psi_a = SMatrix::<f32, NX, NX>::zeros();
            for j in 0..NX {
                if j < jx_cs {
                    for i in 0..NX {
                        at_psi_a[(i, j)] = at_psi[(i, j)];
                    }
                } else {
                    for i in 0..NX {
                        let mut s = 0.0;
                        for m in 0..NX {
                            s += at_psi[(i, m)] * ak[(m, j)];
                        }
                        at_psi_a[(i, j)] = s;
                    }
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

            // pv[k] = h_x + h_xu @ gain_kk[k]
            for i in 0..NX {
                let mut s = h_x[i];
                for m in 0..NU {
                    s += h_xu[(i, m)] * self.gain_kk[k][m];
                }
                self.pv[k][i] = s;
            }

            // Stage-output finiteness: gains feed the forward sweep's
            // Newton step; pp/pv seed stage k−1. Checking all four each
            // stage makes the diverged verdict sound — a NaN confined to
            // pp columns outside the sparse-B block would otherwise ride
            // the recursion for stages before surfacing anywhere the
            // KKT norm or the caller's u0 guard can see it.
            if !all_finite(&self.gain_k[k])
                || !all_finite(&self.gain_kk[k])
                || !all_finite(&self.pp[k])
                || !all_finite(&self.pv[k])
            {
                return false;
            }
        }
        true
    }

    /// Forward sweep: Newton step + update.
    fn forward_sweep<M>(
        &mut self,
        x_init: &SVector<f32, NX>,
        alpha: f32,
        problem: &MpcProblem<M, NX, NU>,
        n: usize,
    ) where
        M: QuadDynamicsModel<NX, NU>,
    {
        let mut dx = SVector::<f32, NX>::zeros();
        for i in 0..NX {
            dx[i] = x_init[i] - self.x_bar[0][i];
        }

        for k in 0..n {
            // du = gain_k[k] @ dx + alpha * gain_kk[k]
            let mut du = SVector::<f32, NU>::zeros();
            for i in 0..NU {
                let mut s = alpha * self.gain_kk[k][i];
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

            // dx = A[k] @ dx + B[k] @ du_actual
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
}

// ── SQP solver ──────────────────────────────────────────────────────────

pub struct SqpSolver<const NX: usize, const NU: usize, const N: usize, const NP1: usize> {
    qp: QpWorkspace<NX, NU, N, NP1>,
}

impl<const NX: usize, const NU: usize, const N: usize, const NP1: usize> SqpSolver<NX, NU, N, NP1> {
    pub fn new() -> Self {
        Self {
            qp: QpWorkspace::new(),
        }
    }

    /// Access the optimal control trajectory after [`solve`].
    pub fn u_bar(&self) -> &[SVector<f32, NU>; N] {
        &self.qp.u_bar
    }

    /// Access the optimal state trajectory after [`solve`].
    pub fn x_bar(&self) -> &[SVector<f32, NX>; NP1] {
        &self.qp.x_bar
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve<M>(
        &mut self,
        problem: &MpcProblem<M, NX, NU>,
        x0: &SVector<f32, NX>,
        x_refs: &[SVector<f32, NX>; NP1],
        u_refs: &[SVector<f32, NU>; N],
        u_init: &[SVector<f32, NU>; N],
        max_iters: usize,
        kkt_tol: f32,
    ) -> SolverResult
    where
        M: QuadDynamicsModel<NX, NU>,
    {
        self.qp.u_bar = *u_init;
        // Active horizon: `problem.n` stages, clamped to the workspace
        // capacity `N`. Stages `n..N` of the workspace are left untouched.
        let n = problem.n.min(N);
        debug_assert!(n >= 1, "SqpSolver::solve: horizon must be at least one stage");

        let mut hess_xx = SMatrix::<f32, NX, NX>::zeros();
        let mut r_diag = SVector::<f32, NU>::zeros();
        let mut grad_x = SVector::<f32, NX>::zeros();
        let mut grad_u = SVector::<f32, NU>::zeros();

        let mut converged = false;
        let mut diverged = false;
        let mut sqp_iter = 0usize;
        let mut final_cost = 0.0;

        // Initial forward propagation (done once before the loop).
        // x_bar[0] is normalized defensively in case the caller's x0 has
        // drifted slightly off the unit sphere; subsequent x_bar[k>0] are
        // automatically normalized inside problem.propagate().
        self.qp.x_bar[0] = *x0;
        M::normalize_quat(&mut self.qp.x_bar[0]);
        for k in 0..n {
            self.qp.x_bar[k + 1] = problem.propagate(&self.qp.x_bar[k], &self.qp.u_bar[k]);
        }

        for iteration in 0..max_iters {
            sqp_iter = iteration + 1;

            // x_bar is already up-to-date (from init or end of previous iteration)

            // Step 1: Linearize
            for k in 0..n {
                let (fx, fu) = problem.linearize(&self.qp.x_bar[k], &self.qp.u_bar[k]);
                self.qp.a[k] = fx;
                self.qp.b[k] = fu;
            }

            // Step 2: Stage costs (accumulate total cost to avoid separate eval_cost pass)
            final_cost = 0.0;
            for k in 0..n {
                final_cost += problem.stage_cost_hess_grad(
                    &self.qp.x_bar[k],
                    &self.qp.u_bar[k],
                    &x_refs[k],
                    &u_refs[k],
                    &mut hess_xx,
                    &mut r_diag,
                    &mut grad_x,
                    &mut grad_u,
                );
                self.qp.qm[k] = hess_xx;
                self.qp.rm[k] = r_diag;
                self.qp.q[k] = grad_x;
                self.qp.r[k] = grad_u;
            }

            // Terminal cost
            final_cost += problem.terminal_cost_hess_grad(
                &self.qp.x_bar[n],
                &x_refs[n],
                &mut grad_x,
                &mut hess_xx,
            );
            self.qp.qm[n] = hess_xx;
            self.qp.q[n] = grad_x;

            // Step 3: Backward Riccati sweep. A `false` return means a
            // non-finite value entered the recursion: the QP subproblem
            // is corrupt and stepping on its gains would move `u_bar` by
            // garbage. Reject the step — `u_bar` keeps the last valid
            // iterate (the warm start on iteration 0), `converged` stays
            // false, and the caller sees `diverged`.
            if !self.qp.backward_sweep::<M>(n) {
                diverged = true;
                break;
            }

            // Step 4: Convergence check (computed before the forward sweep
            // and re-propagation so we can elide the re-propagation when
            // no next iteration will consume it). `kkt_norm` depends only
            // on `gain_kk` from `backward_sweep`, which `forward_sweep`
            // does not modify, so the check is invariant to ordering.
            //
            // NaN-robust accumulation: `f32::max` ignores NaN
            // (`0.0.max(NaN) == 0.0`), which once let an all-NaN
            // `gain_kk` report `kkt_norm = 0` → "converged". The sweep
            // now guarantees finite gains, so the else-branch is defense
            // in depth against any future relaxation of those checks.
            let mut kkt_norm = 0.0f32;
            for k in 0..n {
                for i in 0..NU {
                    let v = self.qp.gain_kk[k][i].abs();
                    if v.is_finite() {
                        kkt_norm = kkt_norm.max(v);
                    } else {
                        kkt_norm = f32::INFINITY;
                    }
                }
            }
            let last_iter = iteration + 1 == max_iters || kkt_norm < kkt_tol;

            // Step 5: Forward sweep (full Newton, alpha=1)
            self.qp.forward_sweep::<M>(x0, 1.0, problem, n);

            // Re-propagate x_bar with updated u_bar — only when another
            // SQP iteration will run and consume it. After the loop exits
            // callers read `u_bar()` (already updated by `forward_sweep`),
            // so leaving `x_bar` inconsistent with the final `u_bar` is
            // harmless. `x_bar[0]` is renormalized in case x0 was non-unit;
            // downstream states are normalized inside `propagate()`.
            if !last_iter {
                self.qp.x_bar[0] = *x0;
                M::normalize_quat(&mut self.qp.x_bar[0]);
                for k in 0..n {
                    self.qp.x_bar[k + 1] = problem.propagate(&self.qp.x_bar[k], &self.qp.u_bar[k]);
                }
            }

            if kkt_norm < kkt_tol {
                converged = true;
                break;
            }
        }

        SolverResult {
            cost: final_cost,
            iters: sqp_iter,
            converged,
            diverged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpc::mpc_problem::SimpleQuadProblem;
    use crate::mpc::quad_model::{PosCostMode, QuadModel, N, NU, NX};

    const NP1: usize = N + 1;

    /// Pinned struct literal (same values as `control_convergence.rs`'s
    /// baseline) — `QuadModel::default()` was deliberately deleted so
    /// tests stay independent of any vehicle definition.
    fn test_model() -> QuadModel {
        let mass = 0.58;
        QuadModel {
            mass,
            grav: 9.81,
            dt: 0.05,
            u_bounds: [[0.0, 48.0], [-10.0, 10.0], [-10.0, 10.0], [-6.0, 6.0]],
            mass_inv: 1.0 / mass,
            w_pos: [200.0, 200.0, 200.0],
            w_vel: [10.0, 10.0, 10.0],
            w_att: [5.0, 5.0, 200.0],
            w_pos_n: [200.0, 200.0, 200.0],
            w_vel_n: [10.0, 10.0, 10.0],
            w_att_n: [5.0, 5.0, 200.0],
            w_input: nalgebra::Vector4::new(1.0, 20.0, 20.0, 20.0),
            rho: 1e4,
            pos_cost_mode: PosCostMode::Quadratic,
            tilt_cos_max: 0.5,
            tilt_barrier_tau: 0.0,
            tilt_barrier_delta: 0.05,
            drag_coeff: [0.0; 3],
            thrust_coeff: 0.0,
            body_drag_coeff: [0.0; 3],
        }
    }

    fn hover_u(model: &QuadModel) -> SVector<f32, NU> {
        SVector::<f32, NU>::from_row_slice(&[model.mass * model.grav, 0.0, 0.0, 0.0])
    }

    fn identity_refs() -> [SVector<f32, NX>; NP1] {
        let mut x = SVector::<f32, NX>::zeros();
        x[6] = 1.0; // qw (scalar-last)
        [x; NP1]
    }

    /// Healthy hover solve: finite inputs must never report `diverged`,
    /// and the hover fixed point converges.
    #[test]
    fn finite_solve_does_not_report_diverged() {
        let model = test_model();
        let hover = hover_u(&model);
        let problem = SimpleQuadProblem::with_rk4(model, N);
        let mut solver = SimpleSqpSolver::new();
        let mut x0 = SVector::<f32, NX>::zeros();
        x0[6] = 1.0;
        let res = solver.solve(
            &problem,
            &x0,
            &identity_refs(),
            &[hover; N],
            &[hover; N],
            10,
            1e-3,
        );
        assert!(!res.diverged);
        assert!(res.converged, "hover from hover should converge");
        assert!(solver.u_bar().iter().all(|u| u.iter().all(|v| v.is_finite())));
    }

    /// Regression for the KKT NaN hole: a NaN initial state NaNs the
    /// linearization and cost, which under the old `f32::max` KKT
    /// accumulation reported `kkt_norm = 0` → `converged: true`. The
    /// sweep must now bail and report `diverged`, leaving `u_bar` at the
    /// (finite) warm start.
    #[test]
    fn nan_x0_reports_diverged_not_converged() {
        let model = test_model();
        let hover = hover_u(&model);
        let problem = SimpleQuadProblem::with_rk4(model, N);
        let mut solver = SimpleSqpSolver::new();
        let mut x0 = SVector::<f32, NX>::zeros();
        x0[0] = f32::NAN;
        x0[6] = 1.0;
        let res = solver.solve(
            &problem,
            &x0,
            &identity_refs(),
            &[hover; N],
            &[hover; N],
            5,
            1e-3,
        );
        assert!(res.diverged, "NaN x0 must be reported as divergence");
        assert!(!res.converged, "a diverged solve must not report converged");
        // Step rejected: the warm start survives untouched and finite.
        assert!(solver.u_bar().iter().all(|u| u.iter().all(|v| v.is_finite())));
    }

    /// Same contract for a NaN landing in the reference trajectory (the
    /// path the firmware once hit via a NaN reference quaternion).
    #[test]
    fn nan_x_ref_reports_diverged_not_converged() {
        let model = test_model();
        let hover = hover_u(&model);
        let problem = SimpleQuadProblem::with_rk4(model, N);
        let mut solver = SimpleSqpSolver::new();
        let mut x0 = SVector::<f32, NX>::zeros();
        x0[6] = 1.0;
        let mut x_refs = identity_refs();
        x_refs[N / 2][1] = f32::NAN;
        let res = solver.solve(&problem, &x0, &x_refs, &[hover; N], &[hover; N], 5, 1e-3);
        assert!(res.diverged);
        assert!(!res.converged);
        assert!(solver.u_bar().iter().all(|u| u.iter().all(|v| v.is_finite())));
    }
}

// ── Concrete monomorphisations ──────────────────────────────────────────

const N_FULL: usize = full_quad_model::N;
const NP1_FULL: usize = N_FULL + 1;
/// `SqpSolver` instantiated for `FullQuadModel` (NX=13, NU=4, N=20).
pub type FullSqpSolver =
    SqpSolver<{ full_quad_model::NX }, { full_quad_model::NU }, N_FULL, NP1_FULL>;

const N_SIMPLE: usize = quad_model::N;
const NP1_SIMPLE: usize = N_SIMPLE + 1;
/// `SqpSolver` instantiated for `QuadModel` (NX=10, NU=4, N=20).
pub type SimpleSqpSolver = SqpSolver<{ quad_model::NX }, { quad_model::NU }, N_SIMPLE, NP1_SIMPLE>;
