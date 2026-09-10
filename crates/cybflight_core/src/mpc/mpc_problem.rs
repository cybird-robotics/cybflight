//! `MpcProblem` — model-agnostic stateless cost/dynamics oracle.
//!
//! `MpcProblem<M, NX, NU>` wraps any model `M` that implements
//! [`super::QuadDynamicsModel`]`<NX, NU>`. The same façade therefore works
//! with [`super::FullQuadModel`] (NX=13, NU=4) or [`super::QuadModel`]
//! (NX=10, NU=4) without code duplication. Concrete instances:
//!
//! - [`FullQuadProblem`]   = `MpcProblem<FullQuadModel, 13, 4>`
//! - [`SimpleQuadProblem`] = `MpcProblem<QuadModel, 10, 4>`

use super::full_quad_model::{self, FullQuadModel};
use super::quad_model::{self, QuadModel};
use super::QuadDynamicsModel;
use nalgebra::{SMatrix, SVector};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Propagation {
    Rk4,
    Euler,
}

/// Stateless oracle that projects a [`QuadDynamicsModel`] into the API the
/// SQP solver consumes (propagation, linearisation, costs, control clamping).
///
/// `NX` and `NU` are const-generic so the array shapes flow into method
/// signatures and are compile-time checked. The same struct therefore wraps
/// any model dimension; pick a model and the type aliases below resolve to
/// the right monomorphisation.
#[derive(Clone)]
pub struct MpcProblem<M, const NX: usize, const NU: usize>
where
    M: QuadDynamicsModel<NX, NU>,
{
    pub model: M,
    /// Active horizon length (stages). The solver clamps it to its
    /// const-generic workspace capacity `N`; `n < N` runs a shorter
    /// horizon in the same workspace.
    pub n: usize,
    pub prop: Propagation,
}

impl<M, const NX: usize, const NU: usize> MpcProblem<M, NX, NU>
where
    M: QuadDynamicsModel<NX, NU>,
{
    pub fn new(model: M, n: usize, prop: Propagation) -> Self {
        Self { model, n, prop }
    }

    /// Convenience constructor: same as `new(model, n, Propagation::Rk4)`.
    pub fn with_rk4(model: M, n: usize) -> Self {
        Self::new(model, n, Propagation::Rk4)
    }

    /// One-step nonlinear propagation, dispatched on the configured integrator.
    pub fn propagate(&self, x: &SVector<f32, NX>, u: &SVector<f32, NU>) -> SVector<f32, NX> {
        match self.prop {
            Propagation::Euler => self.model.propagate_euler(x, u),
            Propagation::Rk4 => self.model.propagate_rk4(x, u),
        }
    }

    /// Returns `(F_x, F_u) = (I + dt·∂f/∂x, dt·∂f/∂u)` (Euler sensitivity).
    pub fn linearize(
        &self,
        x: &SVector<f32, NX>,
        u: &SVector<f32, NU>,
    ) -> (SMatrix<f32, NX, NX>, SMatrix<f32, NX, NU>) {
        self.model.propagate_euler_grad(x, u)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stage_cost_hess_grad(
        &self,
        x: &SVector<f32, NX>,
        u: &SVector<f32, NU>,
        x_ref: &SVector<f32, NX>,
        u_ref: &SVector<f32, NU>,
        hess_xx: &mut SMatrix<f32, NX, NX>,
        r_diag: &mut SVector<f32, NU>,
        grad_x: &mut SVector<f32, NX>,
        grad_u: &mut SVector<f32, NU>,
    ) -> f32 {
        self.model
            .stage_cost_hess_grad(x, u, x_ref, u_ref, hess_xx, r_diag, grad_x, grad_u)
    }

    /// Terminal cost = the model's terminal state cost (no input, no
    /// constraint penalty at N). For `QuadModel` that is the stage state
    /// cost under the terminal weights, which default to the stage ones.
    pub fn terminal_cost_hess_grad(
        &self,
        x: &SVector<f32, NX>,
        x_ref: &SVector<f32, NX>,
        grad_x: &mut SVector<f32, NX>,
        hess_xx: &mut SMatrix<f32, NX, NX>,
    ) -> f32 {
        self.model.terminal_cost_hess_grad(x, x_ref, grad_x, hess_xx)
    }

    /// Sum of stage state-cost + input-cost + constraint penalty over the
    /// horizon, plus the terminal state cost. Used by the solver only for
    /// debugging / line-search; the per-iteration cost is accumulated inside
    /// `SqpSolver::solve` to avoid the second pass.
    #[allow(dead_code)]
    pub fn eval_cost(
        &self,
        x_bar: &[SVector<f32, NX>],
        u_bar: &[SVector<f32, NU>],
        x_refs: &[SVector<f32, NX>],
        u_refs: &[SVector<f32, NU>],
    ) -> f32 {
        let n = self.n;
        let mut gx = SVector::<f32, NX>::zeros();
        let mut gu = SVector::<f32, NU>::zeros();
        let mut gu_con = SVector::<f32, NU>::zeros();
        let mut r_tmp = SVector::<f32, NU>::zeros();
        let mut cost = 0.0;
        for k in 0..n {
            cost += self.model.state_cost_grad(&x_bar[k], &x_refs[k], &mut gx);
            cost += self.model.input_cost_grad(&u_bar[k], &u_refs[k], &mut gu);
            gu_con.fill(0.0);
            r_tmp.fill(0.0);
            cost += self
                .model
                .constraint_hess_grad(&u_bar[k], &mut gu_con, &mut r_tmp);
        }
        cost += self.model.state_cost_grad(&x_bar[n], &x_refs[n], &mut gx);
        cost
    }

    pub fn clamp(&self, u: &SVector<f32, NU>) -> SVector<f32, NU> {
        self.model.clamp_control(u)
    }
}

// ── Concrete monomorphisations ─────────────────────────────────────────────

/// `MpcProblem` instantiated for [`FullQuadModel`] (13-state, per-motor thrust).
pub type FullQuadProblem =
    MpcProblem<FullQuadModel, { full_quad_model::NX }, { full_quad_model::NU }>;

/// `MpcProblem` instantiated for [`QuadModel`] (10-state, thrust + body rates).
pub type SimpleQuadProblem = MpcProblem<QuadModel, { quad_model::NX }, { quad_model::NU }>;
