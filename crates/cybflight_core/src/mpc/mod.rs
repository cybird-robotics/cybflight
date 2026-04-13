pub mod full_quad_model;
pub mod model_utils;
pub mod mpc_problem;
pub mod quad_model;
pub mod sqp_solver;

pub use full_quad_model::{normalize_quat, FullQuadModel, N, NU, NX};
pub use mpc_problem::{FullQuadProblem, MpcProblem, Propagation, SimpleQuadProblem};
pub use quad_model::QuadModel;
pub use sqp_solver::{FullSqpSolver, SimpleSqpSolver, SolverResult, SqpSolver};

use nalgebra::{SMatrix, SVector};

// ───────────────────────────────────────────────────────────────────────────
// Model abstraction trait
// ───────────────────────────────────────────────────────────────────────────
//
// `MpcProblem` is generic over a model type that implements this trait, so
// the same MpcProblem façade can wrap either `FullQuadModel` (NX=13, NU=4)
// or `QuadModel` (NX=10, NU=4) without code duplication.
//
// `NX` and `NU` are const generics so each implementation locks in its own
// dimensions and the array shapes are checked at compile time.

/// Common interface implemented by every quadrotor dynamics model that can
/// drive an [`MpcProblem`]. The signatures are an exact mirror of the inherent
/// methods both [`FullQuadModel`] and [`QuadModel`] already expose, so the
/// `impl` blocks below are pure delegation.
///
/// Three additional items the SQP solver consumes:
/// - `BNZ_START` / `BNZ_LEN` describe the contiguous nonzero-row range of
///   `df/du` (the sparse-B optimization in the Riccati backward sweep).
/// - `normalize_quat` is a static helper that projects the quaternion
///   components of a state vector back onto the unit 3-sphere.
pub trait QuadDynamicsModel<const NX: usize, const NU: usize> {
    /// First nonzero row of `df/du` (contiguous block).
    const BNZ_START: usize;
    /// Number of consecutive nonzero rows in `df/du` starting at `BNZ_START`.
    const BNZ_LEN: usize;

    /// Project the quaternion components of `x` back onto the unit 3-sphere.
    fn normalize_quat(x: &mut SVector<f32, NX>);

    fn propagate_rk4(&self, x: &SVector<f32, NX>, u: &SVector<f32, NU>) -> SVector<f32, NX>;
    fn propagate_euler(&self, x: &SVector<f32, NX>, u: &SVector<f32, NU>) -> SVector<f32, NX>;
    fn propagate_euler_grad(
        &self,
        x: &SVector<f32, NX>,
        u: &SVector<f32, NU>,
    ) -> (SMatrix<f32, NX, NX>, SMatrix<f32, NX, NU>);
    fn state_cost_grad(
        &self,
        x: &SVector<f32, NX>,
        xref: &SVector<f32, NX>,
        grad_x: &mut SVector<f32, NX>,
    ) -> f32;
    fn state_cost_hess_grad(
        &self,
        x: &SVector<f32, NX>,
        xref: &SVector<f32, NX>,
        grad_x: &mut SVector<f32, NX>,
        hess_xx: &mut SMatrix<f32, NX, NX>,
    ) -> f32;
    fn input_cost_grad(
        &self,
        u: &SVector<f32, NU>,
        uref: &SVector<f32, NU>,
        grad_u: &mut SVector<f32, NU>,
    ) -> f32;
    fn constraint_hess_grad(
        &self,
        u: &SVector<f32, NU>,
        grad_u: &mut SVector<f32, NU>,
        r_diag: &mut SVector<f32, NU>,
    ) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn stage_cost_hess_grad(
        &self,
        x: &SVector<f32, NX>,
        u: &SVector<f32, NU>,
        xref: &SVector<f32, NX>,
        uref: &SVector<f32, NU>,
        hess_xx: &mut SMatrix<f32, NX, NX>,
        r_diag: &mut SVector<f32, NU>,
        grad_x: &mut SVector<f32, NX>,
        grad_u: &mut SVector<f32, NU>,
    ) -> f32;
    fn clamp_control(&self, u: &SVector<f32, NU>) -> SVector<f32, NU>;
}

// ── Impl for FullQuadModel (NX=13, NU=4) ───────────────────────────────────
impl QuadDynamicsModel<{ full_quad_model::NX }, { full_quad_model::NU }> for FullQuadModel {
    // Sparse-B layout: rows 7-12 of df/du are non-zero (translational
    // acceleration from per-motor thrusts + body-rate state from differential
    // thrusts).
    const BNZ_START: usize = 7;
    const BNZ_LEN: usize = 6;

    #[inline]
    fn normalize_quat(x: &mut SVector<f32, { full_quad_model::NX }>) {
        full_quad_model::normalize_quat(x);
    }

    #[inline]
    fn propagate_rk4(
        &self,
        x: &SVector<f32, { full_quad_model::NX }>,
        u: &SVector<f32, { full_quad_model::NU }>,
    ) -> SVector<f32, { full_quad_model::NX }> {
        FullQuadModel::propagate_rk4(self, x, u)
    }
    #[inline]
    fn propagate_euler(
        &self,
        x: &SVector<f32, { full_quad_model::NX }>,
        u: &SVector<f32, { full_quad_model::NU }>,
    ) -> SVector<f32, { full_quad_model::NX }> {
        FullQuadModel::propagate_euler(self, x, u)
    }
    #[inline]
    fn propagate_euler_grad(
        &self,
        x: &SVector<f32, { full_quad_model::NX }>,
        u: &SVector<f32, { full_quad_model::NU }>,
    ) -> (
        SMatrix<f32, { full_quad_model::NX }, { full_quad_model::NX }>,
        SMatrix<f32, { full_quad_model::NX }, { full_quad_model::NU }>,
    ) {
        FullQuadModel::propagate_euler_grad(self, x, u)
    }
    #[inline]
    fn state_cost_grad(
        &self,
        x: &SVector<f32, { full_quad_model::NX }>,
        xref: &SVector<f32, { full_quad_model::NX }>,
        grad_x: &mut SVector<f32, { full_quad_model::NX }>,
    ) -> f32 {
        FullQuadModel::state_cost_grad(self, x, xref, grad_x)
    }
    #[inline]
    fn state_cost_hess_grad(
        &self,
        x: &SVector<f32, { full_quad_model::NX }>,
        xref: &SVector<f32, { full_quad_model::NX }>,
        grad_x: &mut SVector<f32, { full_quad_model::NX }>,
        hess_xx: &mut SMatrix<f32, { full_quad_model::NX }, { full_quad_model::NX }>,
    ) -> f32 {
        FullQuadModel::state_cost_hess_grad(self, x, xref, grad_x, hess_xx)
    }
    #[inline]
    fn input_cost_grad(
        &self,
        u: &SVector<f32, { full_quad_model::NU }>,
        uref: &SVector<f32, { full_quad_model::NU }>,
        grad_u: &mut SVector<f32, { full_quad_model::NU }>,
    ) -> f32 {
        FullQuadModel::input_cost_grad(self, u, uref, grad_u)
    }
    #[inline]
    fn constraint_hess_grad(
        &self,
        u: &SVector<f32, { full_quad_model::NU }>,
        grad_u: &mut SVector<f32, { full_quad_model::NU }>,
        r_diag: &mut SVector<f32, { full_quad_model::NU }>,
    ) -> f32 {
        FullQuadModel::constraint_hess_grad(self, u, grad_u, r_diag)
    }
    #[inline]
    fn stage_cost_hess_grad(
        &self,
        x: &SVector<f32, { full_quad_model::NX }>,
        u: &SVector<f32, { full_quad_model::NU }>,
        xref: &SVector<f32, { full_quad_model::NX }>,
        uref: &SVector<f32, { full_quad_model::NU }>,
        hess_xx: &mut SMatrix<f32, { full_quad_model::NX }, { full_quad_model::NX }>,
        r_diag: &mut SVector<f32, { full_quad_model::NU }>,
        grad_x: &mut SVector<f32, { full_quad_model::NX }>,
        grad_u: &mut SVector<f32, { full_quad_model::NU }>,
    ) -> f32 {
        FullQuadModel::stage_cost_hess_grad(self, x, u, xref, uref, hess_xx, r_diag, grad_x, grad_u)
    }
    #[inline]
    fn clamp_control(
        &self,
        u: &SVector<f32, { full_quad_model::NU }>,
    ) -> SVector<f32, { full_quad_model::NU }> {
        FullQuadModel::clamp_control(self, u)
    }
}

// ── Impl for QuadModel (NX=10, NU=4) ───────────────────────────────────────
impl QuadDynamicsModel<{ quad_model::NX }, { quad_model::NU }> for QuadModel {
    // Sparse-B layout: rows 3-9 of df/du are non-zero (quaternion kinematics
    // rows 3-6 driven by body-rate inputs + translational acceleration rows
    // 7-9 driven by collective thrust input).
    const BNZ_START: usize = 3;
    const BNZ_LEN: usize = 7;

    #[inline]
    fn normalize_quat(x: &mut SVector<f32, { quad_model::NX }>) {
        quad_model::normalize_quat(x);
    }

    #[inline]
    fn propagate_rk4(
        &self,
        x: &SVector<f32, { quad_model::NX }>,
        u: &SVector<f32, { quad_model::NU }>,
    ) -> SVector<f32, { quad_model::NX }> {
        QuadModel::propagate_rk4(self, x, u)
    }
    #[inline]
    fn propagate_euler(
        &self,
        x: &SVector<f32, { quad_model::NX }>,
        u: &SVector<f32, { quad_model::NU }>,
    ) -> SVector<f32, { quad_model::NX }> {
        QuadModel::propagate_euler(self, x, u)
    }
    #[inline]
    fn propagate_euler_grad(
        &self,
        x: &SVector<f32, { quad_model::NX }>,
        u: &SVector<f32, { quad_model::NU }>,
    ) -> (
        SMatrix<f32, { quad_model::NX }, { quad_model::NX }>,
        SMatrix<f32, { quad_model::NX }, { quad_model::NU }>,
    ) {
        QuadModel::propagate_euler_grad(self, x, u)
    }
    #[inline]
    fn state_cost_grad(
        &self,
        x: &SVector<f32, { quad_model::NX }>,
        xref: &SVector<f32, { quad_model::NX }>,
        grad_x: &mut SVector<f32, { quad_model::NX }>,
    ) -> f32 {
        QuadModel::state_cost_grad(self, x, xref, grad_x)
    }
    #[inline]
    fn state_cost_hess_grad(
        &self,
        x: &SVector<f32, { quad_model::NX }>,
        xref: &SVector<f32, { quad_model::NX }>,
        grad_x: &mut SVector<f32, { quad_model::NX }>,
        hess_xx: &mut SMatrix<f32, { quad_model::NX }, { quad_model::NX }>,
    ) -> f32 {
        QuadModel::state_cost_hess_grad(self, x, xref, grad_x, hess_xx)
    }
    #[inline]
    fn input_cost_grad(
        &self,
        u: &SVector<f32, { quad_model::NU }>,
        uref: &SVector<f32, { quad_model::NU }>,
        grad_u: &mut SVector<f32, { quad_model::NU }>,
    ) -> f32 {
        QuadModel::input_cost_grad(self, u, uref, grad_u)
    }
    #[inline]
    fn constraint_hess_grad(
        &self,
        u: &SVector<f32, { quad_model::NU }>,
        grad_u: &mut SVector<f32, { quad_model::NU }>,
        r_diag: &mut SVector<f32, { quad_model::NU }>,
    ) -> f32 {
        QuadModel::constraint_hess_grad(self, u, grad_u, r_diag)
    }
    #[inline]
    fn stage_cost_hess_grad(
        &self,
        x: &SVector<f32, { quad_model::NX }>,
        u: &SVector<f32, { quad_model::NU }>,
        xref: &SVector<f32, { quad_model::NX }>,
        uref: &SVector<f32, { quad_model::NU }>,
        hess_xx: &mut SMatrix<f32, { quad_model::NX }, { quad_model::NX }>,
        r_diag: &mut SVector<f32, { quad_model::NU }>,
        grad_x: &mut SVector<f32, { quad_model::NX }>,
        grad_u: &mut SVector<f32, { quad_model::NU }>,
    ) -> f32 {
        QuadModel::stage_cost_hess_grad(self, x, u, xref, uref, hess_xx, r_diag, grad_x, grad_u)
    }
    #[inline]
    fn clamp_control(
        &self,
        u: &SVector<f32, { quad_model::NU }>,
    ) -> SVector<f32, { quad_model::NU }> {
        QuadModel::clamp_control(self, u)
    }
}
