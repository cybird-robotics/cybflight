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
use num_traits::Float;

/// Result metadata from the SQP solver.
///
/// The full trajectory (x_bar, u_bar) remains in the solver workspace and
/// can be accessed via [`SqpSolver::u_bar`] / [`SqpSolver::x_bar`].
#[derive(Clone, Copy)]
pub struct SolverResult {
    pub cost: f32,
    pub iters: usize,
    pub converged: bool,
}

// ── 4x4 Cholesky inverse ────────────────────────────────────────────────
//
// Const-generic over `NU` so it can be called from inside a const-generic
// solver, but the body assumes `NU == 4`. The compile-time `assert!` makes
// instantiating it with any other dimension a build-time error, and after
// monomorphisation with NU=4 the body's literal indices `[0..3]` produce
// the same machine code as a non-generic `[[f32; 4]; 4]` implementation.

fn cholesky_inv_4x4<const NU: usize>(m: &[[f32; NU]; NU]) -> [[f32; NU]; NU] {
    const { assert!(NU == 4, "cholesky_inv_4x4 only supports NU=4") };

    for &reg in &[1e-4, 1e-3, 1e-2] {
        let mut mr = *m;
        for i in 0..4 {
            mr[i][i] += reg;
        }

        let mut l = [[0.0f32; NU]; NU];
        let d0 = mr[0][0];
        if d0 <= 0.0 {
            continue;
        }
        l[0][0] = d0.sqrt();
        let l00i = 1.0 / l[0][0];
        l[1][0] = mr[1][0] * l00i;
        l[2][0] = mr[2][0] * l00i;
        l[3][0] = mr[3][0] * l00i;

        let d1 = mr[1][1] - l[1][0] * l[1][0];
        if d1 <= 0.0 {
            continue;
        }
        l[1][1] = d1.sqrt();
        let l11i = 1.0 / l[1][1];
        l[2][1] = (mr[2][1] - l[2][0] * l[1][0]) * l11i;
        l[3][1] = (mr[3][1] - l[3][0] * l[1][0]) * l11i;

        let d2 = mr[2][2] - l[2][0] * l[2][0] - l[2][1] * l[2][1];
        if d2 <= 0.0 {
            continue;
        }
        l[2][2] = d2.sqrt();
        let l22i = 1.0 / l[2][2];
        l[3][2] = (mr[3][2] - l[3][0] * l[2][0] - l[3][1] * l[2][1]) * l22i;

        let d3 = mr[3][3] - l[3][0] * l[3][0] - l[3][1] * l[3][1] - l[3][2] * l[3][2];
        if d3 <= 0.0 {
            continue;
        }
        l[3][3] = d3.sqrt();

        // Triangular inverse
        let mut li = [[0.0f32; NU]; NU];
        li[0][0] = 1.0 / l[0][0];
        li[1][1] = 1.0 / l[1][1];
        li[2][2] = 1.0 / l[2][2];
        li[3][3] = 1.0 / l[3][3];
        li[1][0] = -l[1][0] * li[0][0] * li[1][1];
        li[2][0] = -(l[2][0] * li[0][0] + l[2][1] * li[1][0]) * li[2][2];
        li[2][1] = -l[2][1] * li[1][1] * li[2][2];
        li[3][0] = -(l[3][0] * li[0][0] + l[3][1] * li[1][0] + l[3][2] * li[2][0]) * li[3][3];
        li[3][1] = -(l[3][1] * li[1][1] + l[3][2] * li[2][1]) * li[3][3];
        li[3][2] = -l[3][2] * li[2][2] * li[3][3];

        // result = li^T @ li
        let mut result = [[0.0f32; NU]; NU];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += li[k][i] * li[k][j];
                }
                result[i][j] = s;
            }
        }
        // Check finite
        let mut ok = true;
        'check: for i in 0..4 {
            for j in 0..4 {
                if !result[i][j].is_finite() {
                    ok = false;
                    break 'check;
                }
            }
        }
        if ok {
            return result;
        }
    }

    // Fallback: scaled identity
    let mut diag_max = 1.0f32;
    for i in 0..4 {
        diag_max = diag_max.max(m[i][i].abs());
    }
    let s = 1.0 / diag_max;
    let mut r = [[0.0f32; NU]; NU];
    for i in 0..4 {
        r[i][i] = s;
    }
    r
}

// ── QP Workspace ────────────────────────────────────────────────────────

struct QpWorkspace<const NX: usize, const NU: usize, const N: usize, const NP1: usize> {
    a: [[[f32; NX]; NX]; N],
    b: [[[f32; NU]; NX]; N],
    q: [[f32; NX]; NP1],
    r: [[f32; NU]; N],
    qm: [[[f32; NX]; NX]; NP1],
    rm: [[f32; NU]; N],
    gain_k: [[[f32; NX]; NU]; N],
    gain_kk: [[f32; NU]; N],
    pp: [[[f32; NX]; NX]; NP1],
    pv: [[f32; NX]; NP1],
    pub x_bar: [[f32; NX]; NP1],
    pub u_bar: [[f32; NU]; N],
}

impl<const NX: usize, const NU: usize, const N: usize, const NP1: usize>
    QpWorkspace<NX, NU, N, NP1>
{
    const fn new() -> Self {
        Self {
            a: [[[0.0; NX]; NX]; N],
            b: [[[0.0; NU]; NX]; N],
            q: [[0.0; NX]; NP1],
            r: [[0.0; NU]; N],
            qm: [[[0.0; NX]; NX]; NP1],
            rm: [[0.0; NU]; N],
            gain_k: [[[0.0; NX]; NU]; N],
            gain_kk: [[0.0; NU]; N],
            pp: [[[0.0; NX]; NX]; NP1],
            pv: [[0.0; NX]; NP1],
            x_bar: [[0.0; NX]; NP1],
            u_bar: [[0.0; NU]; N],
        }
    }

    /// Riccati backward sweep with sparse-B optimisation.
    ///
    /// `BNZ_START` and `BNZ_LEN` are pulled from `M` as compile-time
    /// associated constants — after monomorphisation they fold to literal
    /// integers and the loop bounds match what a model-specific hardcoded
    /// solver would produce.
    fn backward_sweep<M>(&mut self)
    where
        M: QuadDynamicsModel<NX, NU>,
    {
        // Capture associated constants into locals so LLVM definitely sees
        // them as compile-time constants in the inner loops below.
        let bnz_start: usize = M::BNZ_START;
        let bnz_len: usize = M::BNZ_LEN;

        self.pp[N] = self.qm[N];
        self.pv[N] = self.q[N];

        for k in (0..N).rev() {
            let psi = &self.pp[k + 1];
            let pv = &self.pv[k + 1];

            let bk = &self.b[k];

            // H_uu = diag(rm[k]) + bk_nz^T @ psi_nz @ bk_nz  (NU x NU)
            let mut h_uu = [[0.0f32; NU]; NU];
            for i in 0..NU {
                h_uu[i][i] = self.rm[k][i];
            }
            for i in 0..NU {
                for j in 0..NU {
                    let mut s = 0.0;
                    for r in 0..bnz_len {
                        let ri = bnz_start + r;
                        for c in 0..bnz_len {
                            let ci = bnz_start + c;
                            s += bk[ri][i] * psi[ri][ci] * bk[ci][j];
                        }
                    }
                    h_uu[i][j] += s;
                }
            }

            // at_psi = A^T @ psi  (NX x NX)
            let ak = &self.a[k];
            let mut at_psi = [[0.0f32; NX]; NX];
            for i in 0..NX {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NX {
                        s += ak[m][i] * psi[m][j];
                    }
                    at_psi[i][j] = s;
                }
            }

            // h_xu = at_psi[:, BNZ] @ bk_nz  (NX x NU)
            let mut h_xu = [[0.0f32; NU]; NX];
            for i in 0..NX {
                for j in 0..NU {
                    let mut s = 0.0;
                    for c in 0..bnz_len {
                        let ci = bnz_start + c;
                        s += at_psi[i][ci] * bk[ci][j];
                    }
                    h_xu[i][j] = s;
                }
            }

            // h_u = r[k] + bk_nz^T @ pv_nz
            let mut h_u = [0.0f32; NU];
            for i in 0..NU {
                let mut s = self.r[k][i];
                for r in 0..bnz_len {
                    s += bk[bnz_start + r][i] * pv[bnz_start + r];
                }
                h_u[i] = s;
            }

            // h_x = q[k] + A^T @ pv
            let mut h_x = [0.0f32; NX];
            for i in 0..NX {
                let mut s = self.q[k][i];
                for m in 0..NX {
                    s += ak[m][i] * pv[m];
                }
                h_x[i] = s;
            }

            let h_uu_inv = cholesky_inv_4x4::<NU>(&h_uu);

            // gain_k[k] = -(h_uu_inv @ h_xu^T)  → (NU x NX)
            for i in 0..NU {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NU {
                        s += h_uu_inv[i][m] * h_xu[j][m];
                    }
                    self.gain_k[k][i][j] = -s;
                }
            }

            // gain_kk[k] = -(h_uu_inv @ h_u)
            for i in 0..NU {
                let mut s = 0.0;
                for m in 0..NU {
                    s += h_uu_inv[i][m] * h_u[m];
                }
                self.gain_kk[k][i] = -s;
            }

            // pp[k] = qm[k] + at_psi @ A[k] + h_xu @ gain_k[k]
            let mut at_psi_a = [[0.0f32; NX]; NX];
            for i in 0..NX {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NX {
                        s += at_psi[i][m] * ak[m][j];
                    }
                    at_psi_a[i][j] = s;
                }
            }
            let mut hxu_gk = [[0.0f32; NX]; NX];
            for i in 0..NX {
                for j in 0..NX {
                    let mut s = 0.0;
                    for m in 0..NU {
                        s += h_xu[i][m] * self.gain_k[k][m][j];
                    }
                    hxu_gk[i][j] = s;
                }
            }
            for i in 0..NX {
                for j in 0..NX {
                    self.pp[k][i][j] = self.qm[k][i][j] + at_psi_a[i][j] + hxu_gk[i][j];
                }
            }

            // pv[k] = h_x + h_xu @ gain_kk[k]
            for i in 0..NX {
                let mut s = h_x[i];
                for m in 0..NU {
                    s += h_xu[i][m] * self.gain_kk[k][m];
                }
                self.pv[k][i] = s;
            }
        }
    }

    /// Forward sweep: Newton step + update.
    fn forward_sweep<M>(&mut self, x_init: &[f32; NX], alpha: f32, problem: &MpcProblem<M, NX, NU>)
    where
        M: QuadDynamicsModel<NX, NU>,
    {
        let mut dx = [0.0f32; NX];
        for i in 0..NX {
            dx[i] = x_init[i] - self.x_bar[0][i];
        }

        for k in 0..N {
            // du = gain_k[k] @ dx + alpha * gain_kk[k]
            let mut du = [0.0f32; NU];
            for i in 0..NU {
                let mut s = alpha * self.gain_kk[k][i];
                for j in 0..NX {
                    s += self.gain_k[k][i][j] * dx[j];
                }
                du[i] = s;
            }

            let u_old = self.u_bar[k];
            let mut u_new = [0.0f32; NU];
            for i in 0..NU {
                u_new[i] = u_old[i] + du[i];
            }
            self.u_bar[k] = problem.clamp(&u_new);

            let mut du_actual = [0.0f32; NU];
            for i in 0..NU {
                du_actual[i] = self.u_bar[k][i] - u_old[i];
            }

            // dx = A[k] @ dx + B[k] @ du_actual
            let mut dx_new = [0.0f32; NX];
            for i in 0..NX {
                let mut s = 0.0;
                for j in 0..NX {
                    s += self.a[k][i][j] * dx[j];
                }
                for j in 0..NU {
                    s += self.b[k][i][j] * du_actual[j];
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

impl<const NX: usize, const NU: usize, const N: usize, const NP1: usize>
    SqpSolver<NX, NU, N, NP1>
{
    pub const fn new() -> Self {
        Self {
            qp: QpWorkspace::new(),
        }
    }

    /// Access the optimal control trajectory after [`solve`].
    pub fn u_bar(&self) -> &[[f32; NU]; N] {
        &self.qp.u_bar
    }

    /// Access the optimal state trajectory after [`solve`].
    pub fn x_bar(&self) -> &[[f32; NX]; NP1] {
        &self.qp.x_bar
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve<M>(
        &mut self,
        problem: &MpcProblem<M, NX, NU>,
        x0: &[f32; NX],
        x_refs: &[[f32; NX]; NP1],
        u_refs: &[[f32; NU]; N],
        u_init: &[[f32; NU]; N],
        max_iters: usize,
        kkt_tol: f32,
    ) -> SolverResult
    where
        M: QuadDynamicsModel<NX, NU>,
    {
        self.qp.u_bar = *u_init;

        let mut hess_xx = [[0.0f32; NX]; NX];
        let mut r_diag = [0.0f32; NU];
        let mut grad_x = [0.0f32; NX];
        let mut grad_u = [0.0f32; NU];

        let mut converged = false;
        let mut sqp_iter = 0usize;
        let mut final_cost = 0.0;

        // Initial forward propagation (done once before the loop).
        // x_bar[0] is normalized defensively in case the caller's x0 has
        // drifted slightly off the unit sphere; subsequent x_bar[k>0] are
        // automatically normalized inside problem.propagate().
        self.qp.x_bar[0] = *x0;
        M::normalize_quat(&mut self.qp.x_bar[0]);
        for k in 0..N {
            self.qp.x_bar[k + 1] = problem.propagate(&self.qp.x_bar[k], &self.qp.u_bar[k]);
        }

        for iteration in 0..max_iters {
            sqp_iter = iteration + 1;

            // x_bar is already up-to-date (from init or end of previous iteration)

            // Step 1: Linearize
            for k in 0..N {
                let (fx, fu) = problem.linearize(&self.qp.x_bar[k], &self.qp.u_bar[k]);
                self.qp.a[k] = fx;
                self.qp.b[k] = fu;
            }

            // Step 2: Stage costs (accumulate total cost to avoid separate eval_cost pass)
            final_cost = 0.0;
            for k in 0..N {
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
                &self.qp.x_bar[N],
                &x_refs[N],
                &mut grad_x,
                &mut hess_xx,
            );
            self.qp.qm[N] = hess_xx;
            self.qp.q[N] = grad_x;

            // Step 3: Backward Riccati sweep
            self.qp.backward_sweep::<M>();

            // Step 4: Convergence check
            let mut kkt_norm = 0.0f32;
            for k in 0..N {
                for i in 0..NU {
                    kkt_norm = kkt_norm.max(self.qp.gain_kk[k][i].abs());
                }
            }

            // Step 5: Forward sweep (full Newton, alpha=1)
            self.qp.forward_sweep::<M>(x0, 1.0, problem);

            // Re-propagate x_bar with updated u_bar.
            // Needed for: (a) x_bar() access by caller, (b) next SQP iteration.
            // x_bar[0] is renormalized in case x0 was non-unit; downstream
            // states are normalized inside propagate().
            self.qp.x_bar[0] = *x0;
            M::normalize_quat(&mut self.qp.x_bar[0]);
            for k in 0..N {
                self.qp.x_bar[k + 1] = problem.propagate(&self.qp.x_bar[k], &self.qp.u_bar[k]);
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
        }
    }
}

// ── Concrete monomorphisations ──────────────────────────────────────────

const N_FULL: usize = full_quad_model::N;
const NP1_FULL: usize = N_FULL + 1;
/// `SqpSolver` instantiated for `FullQuadModel` (NX=13, NU=4, N=20).
pub type FullSqpSolver = SqpSolver<{ full_quad_model::NX }, { full_quad_model::NU }, N_FULL, NP1_FULL>;

const N_SIMPLE: usize = quad_model::N;
const NP1_SIMPLE: usize = N_SIMPLE + 1;
/// `SqpSolver` instantiated for `QuadModel` (NX=10, NU=4, N=20).
pub type SimpleSqpSolver = SqpSolver<{ quad_model::NX }, { quad_model::NU }, N_SIMPLE, NP1_SIMPLE>;
