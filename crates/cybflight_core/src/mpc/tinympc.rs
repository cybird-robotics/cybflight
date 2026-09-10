//! TinyMPC — ADMM-based linear MPC with a cached infinite-horizon LQR.
//!
//! Rust port of the reference solver in <https://github.com/TinyMPC/TinyMPC>
//! (Nguyen, Schoedel, Alavilli, Plancher, Manchester — "TinyMPC: Model-
//! Predictive Control on Resource-Constrained Microcontrollers", ICRA 2024).
//! The port follows `src/tinympc/admm.cpp` step for step:
//!
//! ```text
//! offline : Q̃ = Q + ρI, R̃ = R + ρI
//!           K∞, P∞      ← discrete Riccati iteration on (A, B, Q̃, R̃)
//!           C1 = (R̃ + BᵀP∞B)⁻¹,   C2 = (A − BK∞)ᵀ
//! online  : repeat
//!             backward pass   d_k = C1 (Bᵀ p_{k+1} + r_k)
//!                             p_k = q_k + C2 p_{k+1} − K∞ᵀ r_k
//!             forward pass    u_k = −K∞ x_k − d_k,  x_{k+1} = A x_k + B u_k
//!             slack update    z ← clip(u + y),  v ← clip(x + g)
//!             dual update     y += u − z,       g += x − v
//!             linear cost     r = −R̃ u_ref − ρ(z − y), q = −Q̃ x_ref − ρ(v − g),
//!                             p_N = −P∞ x_ref_N − ρ(v_N − g_N)
//!           until primal/dual residuals < tol (checked every
//!           `check_termination` iterations) or `max_iter`
//! ```
//!
//! The solver is model-agnostic: it takes a fixed discrete-time `(A, B)`
//! pair, diagonal weights and box bounds. [`hover_linear_model`] builds the
//! quadrotor `(A, B)` that makes TinyMPC the linear counterpart of
//! [`super::QuadModel`] (same inputs: collective thrust + body rates), so
//! the two can be compared under identical references and inner loop.
//!
//! Dimensions: `NX` states, `NU` inputs, `N` knot points (`N` states,
//! `N − 1` inputs — the input arrays are sized `N` and the last slot is
//! unused, mirroring the reference's `NHORIZON` layout without needing
//! `generic_const_exprs`).
//!
//! Deviations from the reference, both deliberate:
//! - `update_linear_cost` is also run once **before** the ADMM loop so the
//!   first backward pass already sees the current reference (the reference
//!   code starts the loop with the previous solve's `q`/`r`/`p`).
//! - The Riccati iteration stops on a `‖K − K_prev‖∞` tolerance instead of a
//!   fixed iteration count.

use nalgebra::{SMatrix, SVector};

/// ADMM termination / feature settings (mirrors `TinySettings`).
#[derive(Clone, Copy, Debug)]
pub struct TinySettings {
    pub abs_pri_tol: f32,
    pub abs_dua_tol: f32,
    pub max_iter: usize,
    /// Residuals are evaluated every this many iterations.
    pub check_termination: usize,
    pub en_state_bound: bool,
    pub en_input_bound: bool,
}

impl Default for TinySettings {
    fn default() -> Self {
        Self {
            abs_pri_tol: 1e-3,
            abs_dua_tol: 1e-3,
            max_iter: 100,
            check_termination: 1,
            en_state_bound: false,
            en_input_bound: true,
        }
    }
}

/// Result metadata of one [`TinyMpc::solve`] call.
#[derive(Clone, Copy, Debug)]
pub struct TinyResult {
    pub iters: usize,
    pub converged: bool,
    pub pri_res_state: f32,
    pub pri_res_input: f32,
    pub dua_res_state: f32,
    pub dua_res_input: f32,
}

/// Offline-computed LQR quantities (mirrors `TinyCache`).
#[derive(Clone)]
pub struct TinyCache<const NX: usize, const NU: usize> {
    pub rho: f32,
    pub k_inf: SMatrix<f32, NU, NX>,
    pub p_inf: SMatrix<f32, NX, NX>,
    pub c1: SMatrix<f32, NU, NU>,
    pub c2: SMatrix<f32, NX, NX>,
}

impl<const NX: usize, const NU: usize> TinyCache<NX, NU> {
    /// Discrete-time Riccati iteration on `(A, B, Q + ρI, R + ρI)`.
    ///
    /// `q` / `r` are the **original** diagonal weights; ρ is added here.
    /// Iterates until `‖K − K_prev‖∞ < tol` or `max_iter` (the reference
    /// uses a fixed 5000-iteration loop).
    pub fn compute(
        a: &SMatrix<f32, NX, NX>,
        b: &SMatrix<f32, NX, NU>,
        q: &SVector<f32, NX>,
        r: &SVector<f32, NU>,
        rho: f32,
        tol: f32,
        max_iter: usize,
    ) -> Self {
        let q_tilde = SMatrix::<f32, NX, NX>::from_diagonal(&q.add_scalar(rho));
        let r_tilde = SMatrix::<f32, NU, NU>::from_diagonal(&r.add_scalar(rho));

        let mut p = q_tilde;
        let mut k = SMatrix::<f32, NU, NX>::zeros();
        let mut k_prev = k;
        for _ in 0..max_iter {
            // K = (R̃ + BᵀPB)⁻¹ BᵀPA
            let bt_p = b.transpose() * p;
            let h_uu = r_tilde + bt_p * b;
            let h_uu_inv = h_uu
                .try_inverse()
                .unwrap_or_else(|| SMatrix::<f32, NU, NU>::identity() / h_uu.trace().max(1e-6));
            k = h_uu_inv * (bt_p * a);
            // P = Q̃ + Aᵀ P (A − B K)
            p = q_tilde + a.transpose() * p * (a - b * k);
            // Symmetrize against round-off.
            p = (p + p.transpose()) * 0.5;
            let mut diff = 0.0f32;
            for (x, y) in k.iter().zip(k_prev.iter()) {
                diff = diff.max((x - y).abs());
            }
            if diff < tol {
                break;
            }
            k_prev = k;
        }

        let h_uu = r_tilde + b.transpose() * p * b;
        let c1 = h_uu
            .try_inverse()
            .unwrap_or_else(|| SMatrix::<f32, NU, NU>::identity() / h_uu.trace().max(1e-6));
        let c2 = (a - b * k).transpose();
        Self {
            rho,
            k_inf: k,
            p_inf: p,
            c1,
            c2,
        }
    }
}

/// TinyMPC solver: problem data, cache and ADMM workspace in one struct so
/// consecutive solves warm-start from the previous primal/slack/dual
/// iterates (the reference keeps everything in the global `TinySolver`).
#[derive(Clone)]
pub struct TinyMpc<const NX: usize, const NU: usize, const N: usize> {
    pub settings: TinySettings,
    pub cache: TinyCache<NX, NU>,
    // Problem data.
    pub a: SMatrix<f32, NX, NX>,
    pub b: SMatrix<f32, NX, NU>,
    /// Diagonal of `Q̃ = Q + ρI` (what the reference stores as `work->Q`).
    pub q_diag: SVector<f32, NX>,
    /// Diagonal of `R̃ = R + ρI`.
    pub r_diag: SVector<f32, NU>,
    pub x_min: [SVector<f32, NX>; N],
    pub x_max: [SVector<f32, NX>; N],
    pub u_min: [SVector<f32, NU>; N],
    pub u_max: [SVector<f32, NU>; N],
    pub x_ref: [SVector<f32, NX>; N],
    pub u_ref: [SVector<f32, NU>; N],
    // Workspace (primal / slack / dual / linear cost).
    pub x: [SVector<f32, NX>; N],
    pub u: [SVector<f32, NU>; N],
    q: [SVector<f32, NX>; N],
    r: [SVector<f32, NU>; N],
    p: [SVector<f32, NX>; N],
    d: [SVector<f32, NU>; N],
    v: [SVector<f32, NX>; N],
    vnew: [SVector<f32, NX>; N],
    z: [SVector<f32, NU>; N],
    znew: [SVector<f32, NU>; N],
    g: [SVector<f32, NX>; N],
    y: [SVector<f32, NU>; N],
}

impl<const NX: usize, const NU: usize, const N: usize> TinyMpc<NX, NU, N> {
    /// Build a solver (`tiny_setup` analogue). `q` / `r` are the original
    /// diagonal weights; the LQR cache and `q_diag` / `r_diag` get ρ added.
    /// Bounds default to ±∞ — set [`Self::set_input_bounds`] /
    /// [`Self::set_state_bounds`] afterwards.
    pub fn new(
        a: SMatrix<f32, NX, NX>,
        b: SMatrix<f32, NX, NU>,
        q: SVector<f32, NX>,
        r: SVector<f32, NU>,
        rho: f32,
        settings: TinySettings,
    ) -> Self {
        const { assert!(N >= 2, "TinyMPC needs at least two knot points") };
        let cache = TinyCache::compute(&a, &b, &q, &r, rho, 1e-6, 5000);
        Self {
            settings,
            cache,
            a,
            b,
            q_diag: q.add_scalar(rho),
            r_diag: r.add_scalar(rho),
            x_min: [SVector::from_element(f32::NEG_INFINITY); N],
            x_max: [SVector::from_element(f32::INFINITY); N],
            u_min: [SVector::from_element(f32::NEG_INFINITY); N],
            u_max: [SVector::from_element(f32::INFINITY); N],
            x_ref: [SVector::zeros(); N],
            u_ref: [SVector::zeros(); N],
            x: [SVector::zeros(); N],
            u: [SVector::zeros(); N],
            q: [SVector::zeros(); N],
            r: [SVector::zeros(); N],
            p: [SVector::zeros(); N],
            d: [SVector::zeros(); N],
            v: [SVector::zeros(); N],
            vnew: [SVector::zeros(); N],
            z: [SVector::zeros(); N],
            znew: [SVector::zeros(); N],
            g: [SVector::zeros(); N],
            y: [SVector::zeros(); N],
        }
    }

    /// Same box bounds on every input stage.
    pub fn set_input_bounds(&mut self, lo: SVector<f32, NU>, hi: SVector<f32, NU>) {
        self.u_min = [lo; N];
        self.u_max = [hi; N];
    }

    /// Same box bounds on every state stage.
    pub fn set_state_bounds(&mut self, lo: SVector<f32, NX>, hi: SVector<f32, NX>) {
        self.x_min = [lo; N];
        self.x_max = [hi; N];
    }

    /// Discard the warm start (primal, slack and dual iterates).
    pub fn reset(&mut self) {
        for k in 0..N {
            self.x[k].fill(0.0);
            self.u[k].fill(0.0);
            self.q[k].fill(0.0);
            self.r[k].fill(0.0);
            self.p[k].fill(0.0);
            self.d[k].fill(0.0);
            self.v[k].fill(0.0);
            self.vnew[k].fill(0.0);
            self.z[k].fill(0.0);
            self.znew[k].fill(0.0);
            self.g[k].fill(0.0);
            self.y[k].fill(0.0);
        }
    }

    /// First input of the last solve, clipped to its bounds.
    pub fn u0(&self) -> SVector<f32, NU> {
        SVector::from_fn(|i, _| self.u[0][i].clamp(self.u_min[0][i], self.u_max[0][i]))
    }

    /// Run ADMM from the current warm start with `x0` as the initial state.
    pub fn solve(&mut self, x0: &SVector<f32, NX>) -> TinyResult {
        self.x[0] = *x0;
        // See module docs: refresh the linear cost with the current
        // reference before the first backward pass.
        self.update_linear_cost();

        let mut res = TinyResult {
            iters: 0,
            converged: false,
            pri_res_state: 0.0,
            pri_res_input: 0.0,
            dua_res_state: 0.0,
            dua_res_input: 0.0,
        };
        let check_every = self.settings.check_termination.max(1);

        for i in 0..self.settings.max_iter {
            self.backward_pass_grad();
            self.forward_pass();
            self.update_slack();
            self.update_dual();
            self.update_linear_cost();
            res.iters = i + 1;

            if res.iters % check_every == 0 {
                let rho = self.cache.rho;
                let (mut pri_x, mut pri_u, mut dua_x, mut dua_u) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for k in 0..N {
                    pri_x = pri_x.max((self.x[k] - self.vnew[k]).amax());
                    dua_x = dua_x.max(rho * (self.v[k] - self.vnew[k]).amax());
                }
                for k in 0..N - 1 {
                    pri_u = pri_u.max((self.u[k] - self.znew[k]).amax());
                    dua_u = dua_u.max(rho * (self.z[k] - self.znew[k]).amax());
                }
                res.pri_res_state = pri_x;
                res.pri_res_input = pri_u;
                res.dua_res_state = dua_x;
                res.dua_res_input = dua_u;
                let s = &self.settings;
                if pri_x < s.abs_pri_tol
                    && pri_u < s.abs_pri_tol
                    && dua_x < s.abs_dua_tol
                    && dua_u < s.abs_dua_tol
                {
                    res.converged = true;
                }
            }

            self.v = self.vnew;
            self.z = self.znew;

            if res.converged {
                break;
            }
        }
        res
    }

    /// `d_k = C1 (Bᵀ p_{k+1} + r_k)`, `p_k = q_k + C2 p_{k+1} − K∞ᵀ r_k`.
    fn backward_pass_grad(&mut self) {
        let bt = self.b.transpose();
        let kt = self.cache.k_inf.transpose();
        for k in (0..N - 1).rev() {
            self.d[k] = self.cache.c1 * (bt * self.p[k + 1] + self.r[k]);
            self.p[k] = self.q[k] + self.cache.c2 * self.p[k + 1] - kt * self.r[k];
        }
    }

    /// `u_k = −K∞ x_k − d_k`, `x_{k+1} = A x_k + B u_k`.
    fn forward_pass(&mut self) {
        for k in 0..N - 1 {
            self.u[k] = -(self.cache.k_inf * self.x[k]) - self.d[k];
            self.x[k + 1] = self.a * self.x[k] + self.b * self.u[k];
        }
    }

    /// Project `u + y` / `x + g` onto the box constraints.
    fn update_slack(&mut self) {
        for k in 0..N - 1 {
            let s = self.u[k] + self.y[k];
            self.znew[k] = if self.settings.en_input_bound {
                clip(&s, &self.u_min[k], &self.u_max[k])
            } else {
                s
            };
        }
        for k in 0..N {
            let s = self.x[k] + self.g[k];
            self.vnew[k] = if self.settings.en_state_bound {
                clip(&s, &self.x_min[k], &self.x_max[k])
            } else {
                s
            };
        }
    }

    fn update_dual(&mut self) {
        for k in 0..N - 1 {
            self.y[k] += self.u[k] - self.znew[k];
        }
        for k in 0..N {
            self.g[k] += self.x[k] - self.vnew[k];
        }
    }

    /// Reference form (`Q̃`/`R̃` in the tracking term, `P∞` at the terminal
    /// knot). ρ terms fold the slack/dual iterates into the QP.
    fn update_linear_cost(&mut self) {
        let rho = self.cache.rho;
        for k in 0..N - 1 {
            self.r[k] =
                -self.u_ref[k].component_mul(&self.r_diag) - (self.znew[k] - self.y[k]) * rho;
        }
        for k in 0..N {
            self.q[k] =
                -self.x_ref[k].component_mul(&self.q_diag) - (self.vnew[k] - self.g[k]) * rho;
        }
        let n = N - 1;
        self.p[n] =
            -(self.cache.p_inf.transpose() * self.x_ref[n]) - (self.vnew[n] - self.g[n]) * rho;
    }
}

#[inline]
fn clip<const D: usize>(
    s: &SVector<f32, D>,
    lo: &SVector<f32, D>,
    hi: &SVector<f32, D>,
) -> SVector<f32, D> {
    SVector::from_fn(|i, _| s[i].clamp(lo[i], hi[i]))
}

// ───────────────────────────────────────────────────────────────────────────
// Hover-linearised quadrotor model (linear counterpart of `QuadModel`)
// ───────────────────────────────────────────────────────────────────────────

/// State dimension of the hover-linearised model:
/// `δx = [p(3), θ(3), v(3)]` — position [m], attitude error as a
/// world-frame rotation vector [rad] (`R ≈ I + [θ]×`), velocity [m/s].
pub const HOVER_NX: usize = 9;
/// Input dimension: `δu = [c − m·g, ωx, ωy, ωz]` — collective-thrust
/// deviation from hover [N] and body rates [rad/s]. Same physical inputs as
/// [`super::QuadModel`], so the same inner loop consumes either.
pub const HOVER_NU: usize = 4;

/// Exact zero-order-hold discretisation of [`super::QuadModel`]'s dynamics
/// linearised about hover (ENU/FLU, thrust along +body-z):
///
/// ```text
/// ṗ = v,   θ̇ = ω,   v̇ = [ g·θy, −g·θx, δc/m ]
/// ```
///
/// The continuous `A` is nilpotent (`A³ = 0`), so the four-term matrix
/// exponential series used here is exact.
pub fn hover_linear_model(
    mass: f32,
    grav: f32,
    dt: f32,
) -> (
    SMatrix<f32, HOVER_NX, HOVER_NX>,
    SMatrix<f32, HOVER_NX, HOVER_NU>,
) {
    let mut ac = SMatrix::<f32, HOVER_NX, HOVER_NX>::zeros();
    let mut bc = SMatrix::<f32, HOVER_NX, HOVER_NU>::zeros();
    for i in 0..3 {
        ac[(i, 6 + i)] = 1.0; // ṗ = v
        bc[(3 + i, 1 + i)] = 1.0; // θ̇ = ω
    }
    ac[(6, 4)] = grav; // v̇x = g θy
    ac[(7, 3)] = -grav; // v̇y = −g θx
    bc[(8, 0)] = 1.0 / mass; // v̇z = δc / m

    // A_d = Σ_{k=0}^{3} A^k dt^k / k!,  B_d = Σ_{k=0}^{3} A^k dt^{k+1} / (k+1)! · B
    let eye = SMatrix::<f32, HOVER_NX, HOVER_NX>::identity();
    let mut a_pow = eye;
    let mut a_d = SMatrix::<f32, HOVER_NX, HOVER_NX>::zeros();
    let mut b_int = SMatrix::<f32, HOVER_NX, HOVER_NX>::zeros();
    let mut fact = 1.0f32;
    let mut dt_pow = 1.0f32;
    for k in 0..4 {
        a_d += a_pow * (dt_pow / fact);
        b_int += a_pow * (dt_pow * dt / (fact * (k as f32 + 1.0)));
        a_pow *= ac;
        dt_pow *= dt;
        fact *= k as f32 + 1.0;
    }
    (a_d, b_int * bc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hover_model_is_zoh_exact() {
        let (a, b) = hover_linear_model(1.0, 9.81, 0.1);
        // p_x after dt from unit pitch error: ½ g θy dt² -> x-accel path.
        assert!((a[(0, 4)] - 0.5 * 9.81 * 0.01).abs() < 1e-6);
        assert!((a[(0, 6)] - 0.1).abs() < 1e-6);
        // z-velocity from thrust: dt/m.
        assert!((b[(8, 0)] - 0.1).abs() < 1e-6);
        // x-position from pitch rate: g dt³/6.
        assert!((b[(0, 2)] - 9.81 * 0.001 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn admm_respects_input_bounds_and_tracks() {
        const N: usize = 21;
        let (a, b) = hover_linear_model(1.0, 9.81, 0.05);
        let q = SVector::<f32, HOVER_NX>::from_row_slice(&[
            100.0, 100.0, 100.0, 5.0, 5.0, 50.0, 10.0, 10.0, 10.0,
        ]);
        let r = SVector::<f32, HOVER_NU>::from_element(1.0);
        let mut mpc = TinyMpc::<HOVER_NX, HOVER_NU, N>::new(
            a,
            b,
            q,
            r,
            5.0,
            TinySettings {
                max_iter: 500,
                ..Default::default()
            },
        );
        mpc.set_input_bounds(
            SVector::from_row_slice(&[-9.81, -3.0, -3.0, -3.0]),
            SVector::from_row_slice(&[15.0, 3.0, 3.0, 3.0]),
        );
        // Step 1 m in x, 0.5 m up.
        let mut xr = SVector::<f32, HOVER_NX>::zeros();
        xr[0] = 1.0;
        xr[2] = 0.5;
        mpc.x_ref = [xr; N];
        let res = mpc.solve(&SVector::zeros());
        assert!(res.converged, "ADMM did not converge: {res:?}");
        for k in 0..N - 1 {
            for i in 0..HOVER_NU {
                assert!(mpc.u[k][i] >= mpc.u_min[k][i] - 2e-3);
                assert!(mpc.u[k][i] <= mpc.u_max[k][i] + 2e-3);
            }
        }
        // Moves toward the reference: first pitch rate positive (nose down
        // → +x accel in FLU/ENU), first thrust deviation positive (climb).
        assert!(mpc.u[0][2] > 0.0);
        assert!(mpc.u[0][0] > 0.0);
        let x_end = mpc.x[N - 1];
        assert!(x_end[0] > 0.3 && x_end[2] > 0.2, "x_end = {x_end}");
    }
}
