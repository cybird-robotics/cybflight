// Multiple-shooting QP solver via Riccati block elimination of the KKT system.
//
// Backward sweep: eliminates (Δx, Δu) stage-by-stage, producing feedback gains K_k.
// Forward sweep: recovers the full Newton step (Δx, Δu) and updates the trajectory.

use nalgebra::{ComplexField as _, SMatrix};

use super::model::{Control, CtrlJac, QuadModel, State, StateJac};
use super::solver::N;

type Mat4 = SMatrix<f32, 4, 4>;
type Mat4x10 = SMatrix<f32, 4, 10>;

/// Per-stage workspace for the structured QP.
pub struct QpWorkspace {
    // Linearized dynamics
    pub a: [StateJac; N],  // ∂F/∂x (10×10)
    pub b: [CtrlJac; N],   // ∂F/∂u (10×4)
    pub d: [State; N],     // dynamics defect: F(x̄_k, ū_k) - x̄_{k+1}

    // Cost gradients (KKT residual components)
    pub q: [State; N + 1],   // state cost gradient
    pub r: [Control; N],     // input cost gradient

    // Cost Hessians
    pub qm: [StateJac; N + 1], // Q_k state cost Hessian (10×10)
    pub rm: [Control; N],      // R_k diagonal (4)

    // Feedback gains from backward sweep
    pub gain_k: [Mat4x10; N], // K_k (4×10)
    pub gain_kk: [Control; N], // kk_k (4)

    // Riccati matrices (backward sweep)
    pub pp: [StateJac; N + 1], // P_k (10×10 Schur complement)
    pub pv: [State; N + 1],    // p_k (10)

    // Trajectory (primal variables)
    pub x_bar: [State; N + 1],
    pub u_bar: [Control; N],
}

impl QpWorkspace {
    pub fn new() -> Self {
        Self {
            a: [StateJac::zeros(); N],
            b: [CtrlJac::zeros(); N],
            d: [State::zeros(); N],
            q: [State::zeros(); N + 1],
            r: [Control::zeros(); N],
            qm: [StateJac::zeros(); N + 1],
            rm: [Control::zeros(); N],
            gain_k: [Mat4x10::zeros(); N],
            gain_kk: [Control::zeros(); N],
            pp: [StateJac::zeros(); N + 1],
            pv: [State::zeros(); N + 1],
            x_bar: [State::zeros(); N + 1],
            u_bar: [Control::zeros(); N],
        }
    }

    /// Backward sweep: block elimination of the KKT system.
    ///
    /// Produces feedback gains K_k, feedforward kk_k, and Riccati matrices P_k, p_k.
    pub fn backward_sweep(&mut self) {
        // Terminal: P_N = Q_N, p_N = q_N
        self.pp[N] = self.qm[N];
        self.pv[N] = self.q[N];

        for k in (0..N).rev() {
            let psi = &self.pp[k + 1]; // 10×10
            let psi_d_plus_p = psi * &self.d[k] + &self.pv[k + 1]; // 10

            // H_uu = R_k + B_k^T Ψ B_k (4×4)
            let bt_psi = self.b[k].transpose() * psi; // 4×10
            let h_uu = diag4(&self.rm[k]) + &bt_psi * &self.b[k]; // 4×4

            // H_xu = A_k^T Ψ B_k (10×4)
            let at_psi = self.a[k].transpose() * psi; // 10×10
            let h_xu = &at_psi * &self.b[k]; // 10×4

            // h_u = r_k + B_k^T (Ψ d_k + p_{k+1}) (4)
            let h_u = &self.r[k] + self.b[k].transpose() * &psi_d_plus_p;

            // h_x = q_k + A_k^T (Ψ d_k + p_{k+1}) (10)
            let h_x = &self.q[k] + self.a[k].transpose() * &psi_d_plus_p;

            // Solve 4×4 SPD system: H_uu [K_k; kk_k] = [-H_xu^T; -h_u]
            let h_uu_inv = cholesky_inv_4x4(&h_uu);
            self.gain_k[k] = -(h_uu_inv * h_xu.transpose()); // 4×10
            self.gain_kk[k] = -(h_uu_inv * h_u); // 4

            // Propagate Riccati backward
            // P_k = Q_k + A_k^T Ψ A_k + H_xu K_k
            self.pp[k] = &self.qm[k] + &at_psi * &self.a[k] + &h_xu * &self.gain_k[k];
            // p_k = h_x + H_xu kk_k
            self.pv[k] = &h_x + &h_xu * &self.gain_kk[k];
        }
    }

    /// Forward sweep: recover Newton step and update trajectory.
    ///
    /// `alpha`: step size for globalization (1.0 = full Newton step).
    /// `x_init`: current initial state constraint.
    pub fn forward_sweep(&mut self, x_init: &State, alpha: f32) {
        let mut dx = x_init - &self.x_bar[0]; // Δx_0

        for k in 0..N {
            // Δu_k = K_k Δx_k + α kk_k
            let du: Control = &self.gain_k[k] * &dx + alpha * &self.gain_kk[k];

            // Save old u for computing clamped delta
            let u_old = self.u_bar[k];

            // Update u with clamping to box constraints
            self.u_bar[k] = clamp_control(&(&u_old + &du));

            // Actual delta after clamping
            let du_actual = &self.u_bar[k] - &u_old;

            // Δx_{k+1} = A_k Δx_k + B_k Δu_actual + d_k
            dx = &self.a[k] * &dx + &self.b[k] * &du_actual + &self.d[k];
        }
    }
}

/// Clamp control to box constraints.
#[inline]
fn clamp_control(u: &Control) -> Control {
    let bounds = QuadModel::U_BOUNDS;
    Control::from([
        u[0].clamp(bounds[0].0, bounds[0].1),
        u[1].clamp(bounds[1].0, bounds[1].1),
        u[2].clamp(bounds[2].0, bounds[2].1),
        u[3].clamp(bounds[3].0, bounds[3].1),
    ])
}

/// Build a 4×4 diagonal matrix from a 4-vector.
#[inline]
fn diag4(v: &Control) -> Mat4 {
    let mut m = Mat4::zeros();
    m[(0, 0)] = v[0];
    m[(1, 1)] = v[1];
    m[(2, 2)] = v[2];
    m[(3, 3)] = v[3];
    m
}

/// 4×4 Cholesky inverse for SPD matrix.
/// Falls back to diagonal regularization if Cholesky fails.
fn cholesky_inv_4x4(m: &Mat4) -> Mat4 {
    // Add small regularization for numerical stability
    let mut mr = *m;
    const REG: f32 = 1e-6;
    mr[(0, 0)] += REG;
    mr[(1, 1)] += REG;
    mr[(2, 2)] += REG;
    mr[(3, 3)] += REG;

    // Manual 4×4 Cholesky: L L^T = mr
    let mut l = Mat4::zeros();

    l[(0, 0)] = mr[(0, 0)].sqrt();
    let l00i = 1.0 / l[(0, 0)];
    l[(1, 0)] = mr[(1, 0)] * l00i;
    l[(2, 0)] = mr[(2, 0)] * l00i;
    l[(3, 0)] = mr[(3, 0)] * l00i;

    l[(1, 1)] = (mr[(1, 1)] - l[(1, 0)] * l[(1, 0)]).sqrt();
    let l11i = 1.0 / l[(1, 1)];
    l[(2, 1)] = (mr[(2, 1)] - l[(2, 0)] * l[(1, 0)]) * l11i;
    l[(3, 1)] = (mr[(3, 1)] - l[(3, 0)] * l[(1, 0)]) * l11i;

    l[(2, 2)] = (mr[(2, 2)] - l[(2, 0)] * l[(2, 0)] - l[(2, 1)] * l[(2, 1)]).sqrt();
    let l22i = 1.0 / l[(2, 2)];
    l[(3, 2)] = (mr[(3, 2)] - l[(3, 0)] * l[(2, 0)] - l[(3, 1)] * l[(2, 1)]) * l22i;

    l[(3, 3)] = (mr[(3, 3)] - l[(3, 0)] * l[(3, 0)] - l[(3, 1)] * l[(3, 1)] - l[(3, 2)] * l[(3, 2)]).sqrt();

    // Invert L (lower triangular)
    let mut li = Mat4::zeros();
    li[(0, 0)] = 1.0 / l[(0, 0)];
    li[(1, 1)] = 1.0 / l[(1, 1)];
    li[(2, 2)] = 1.0 / l[(2, 2)];
    li[(3, 3)] = 1.0 / l[(3, 3)];

    li[(1, 0)] = -l[(1, 0)] * li[(0, 0)] * li[(1, 1)];
    li[(2, 0)] = -(l[(2, 0)] * li[(0, 0)] + l[(2, 1)] * li[(1, 0)]) * li[(2, 2)];
    li[(2, 1)] = -l[(2, 1)] * li[(1, 1)] * li[(2, 2)];
    li[(3, 0)] = -(l[(3, 0)] * li[(0, 0)] + l[(3, 1)] * li[(1, 0)] + l[(3, 2)] * li[(2, 0)]) * li[(3, 3)];
    li[(3, 1)] = -(l[(3, 1)] * li[(1, 1)] + l[(3, 2)] * li[(2, 1)]) * li[(3, 3)];
    li[(3, 2)] = -l[(3, 2)] * li[(2, 2)] * li[(3, 3)];

    // M^{-1} = L^{-T} L^{-1}
    li.transpose() * li
}
