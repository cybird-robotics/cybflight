//! Differential flatness for quadrotors at zero yaw (ψ=0).
//!
//! Forward chain: position derivatives → thrust vector, body axis, body rates.
//! Backward chain: gradient backpropagation through the same transforms.
//!
//! The chain is split into two stages:
//! - [`AlphaState`] from `acc` alone: thrust vector α, body z-axis zB — needed
//!   by the tilt and collective-thrust penalties.
//! - [`FlatnessState`] extends `AlphaState` with jerk-dependent terms (dzB, ω)
//!   — needed only by the body-rate penalty.

#[allow(unused_imports)]
use num_traits::Float;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

use super::types::Vec3;

/// Thrust-vector basis. Computed from acceleration alone — cheap.
///
/// Always well-defined (division by `max(‖α‖, 1e-8)` clamps the singularity).
/// Callers that use ω must additionally check `zb[2] > -0.9` before extending
/// to [`FlatnessState`].
pub struct AlphaState {
    /// α = acc + [0, 0, g].
    pub alpha: Vector3<f32>,
    /// ‖α‖.
    pub norm_alpha: f32,
    /// 1/‖α‖.
    pub inv_norm_alpha: f32,
    /// Body z-axis: α/‖α‖.
    pub zb: Vector3<f32>,
}

/// Full flatness state, including body-rate terms.
pub struct FlatnessState {
    pub alpha: Vector3<f32>,
    pub norm_alpha: f32,
    pub inv_norm_alpha: f32,
    pub zb: Vector3<f32>,
    /// dot(zB, jer).
    pub dot_zb_j: f32,
    /// Body z-axis time derivative: dzB = DN(α)·j / ‖α‖.
    pub dzb: Vector3<f32>,
    /// 1/(1 + zb_z), with singularity guard.
    pub s_inv: f32,
    /// Body rates [ωx, ωy, ωz] at ψ=0.
    pub omega: Vector3<f32>,
}

/// Compute the thrust-vector basis (α, ‖α‖, zB) from acceleration.
#[inline]
pub fn compute_alpha_state(acc: Vec3, gravity: f32) -> AlphaState {
    let alpha = acc + Vector3::new(0.0, 0.0, gravity);
    let norm_alpha = alpha.norm().max(1e-8);
    let inv_norm_alpha = 1.0 / norm_alpha;
    let zb = alpha * inv_norm_alpha;
    AlphaState {
        alpha,
        norm_alpha,
        inv_norm_alpha,
        zb,
    }
}

/// Extend an [`AlphaState`] with body-rate terms.
///
/// Precondition: `alpha.zb[2] > -0.9` (caller must guard; the `1/(1+zb_z)`
/// factor becomes ill-conditioned near inversion).
#[inline]
pub fn extend_to_flatness(alpha: &AlphaState, jer: Vec3) -> FlatnessState {
    let zb = alpha.zb;
    let inv_norm_alpha = alpha.inv_norm_alpha;
    let dot_zb_j = zb.dot(&jer);
    let dzb = (jer - zb * dot_zb_j) * inv_norm_alpha;
    let s_inv = 1.0 / (1.0 + zb[2]).max(0.01);
    let omega = Vector3::new(
        -dzb[1] + s_inv * zb[1] * dzb[2],
        dzb[0] - s_inv * zb[0] * dzb[2],
        s_inv * (zb[1] * dzb[0] - zb[0] * dzb[1]),
    );
    FlatnessState {
        alpha: alpha.alpha,
        norm_alpha: alpha.norm_alpha,
        inv_norm_alpha,
        zb,
        dot_zb_j,
        dzb,
        s_inv,
        omega,
    }
}

/// Reference body-to-world quaternion from the flat output `acc` and a
/// desired yaw `yaw_rad` (natural default `0.0`).
#[inline]
pub fn reference_quaternion(acc: Vec3, yaw_rad: f32, gravity: f32) -> UnitQuaternion<f32> {
    let acc_cmd = acc + Vector3::new(0.0, 0.0, gravity);
    let inv_norm = 1.0 / acc_cmd.norm().max(1e-8);
    let z_b = acc_cmd * inv_norm;

    let s = libm::sinf(yaw_rad);
    let c = libm::cosf(yaw_rad);
    let y_c = Vector3::new(-s, c, 0.0);

    let x_b_unnorm = y_c.cross(&z_b);
    let x_b = x_b_unnorm * (1.0 / x_b_unnorm.norm().max(1e-8));

    let y_b_unnorm = z_b.cross(&x_b);
    let y_b = y_b_unnorm * (1.0 / y_b_unnorm.norm().max(1e-8));

    let r_wb = Matrix3::from_columns(&[x_b, y_b, z_b]);
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_wb))
}

/// Backpropagate body rate gradient through the flatness chain.
///
/// Gradient flow: ∂L/∂ω → ∂L/∂dzB, ∂L/∂zB → ∂L/∂acc, ∂L/∂jer.
/// Accumulates into `grad_acc` and `grad_jer`.
#[inline]
pub fn body_rate_grad_backprop(
    g_omega: &Vector3<f32>,
    fs: &FlatnessState,
    grad_acc: &mut Vector3<f32>,
    grad_jer: &mut Vector3<f32>,
) {
    let zb = fs.zb;
    let dzb = fs.dzb;
    let s_inv = fs.s_inv;
    let inv_norm_alpha = fs.inv_norm_alpha;
    let dot_zb_j = fs.dot_zb_j;

    // grad_ω → grad_dzB via (∂ω/∂dzB)ᵀ
    let g_dzb = Vector3::new(
        g_omega[1] + s_inv * zb[1] * g_omega[2],
        -g_omega[0] - s_inv * zb[0] * g_omega[2],
        s_inv * (zb[1] * g_omega[0] - zb[0] * g_omega[1]),
    );

    // grad_ω → grad_zB via (∂ω/∂zB)ᵀ
    let s2 = s_inv * s_inv;
    let g_zb = Vector3::new(
        -s_inv * (dzb[2] * g_omega[1] + dzb[1] * g_omega[2]),
        s_inv * (dzb[2] * g_omega[0] + dzb[0] * g_omega[2]),
        s2 * (dzb[2] * (zb[0] * g_omega[1] - zb[1] * g_omega[0])
            + (zb[0] * dzb[1] - zb[1] * dzb[0]) * g_omega[2]),
    );

    // DN(α) is the nullspace projector: DN(v) = (v − zB(zB·v)) / ‖α‖
    let dot_zb_gdzb = zb.dot(&g_dzb);
    let dot_zb_gzb = zb.dot(&g_zb);

    // grad_jerk = DN(α) · g_dzb
    let dn_gdzb = (g_dzb - zb * dot_zb_gdzb) * inv_norm_alpha;
    *grad_jer += dn_gdzb;

    // grad_acc from zB path: DN(α) · g_zb
    let dn_gzb = (g_zb - zb * dot_zb_gzb) * inv_norm_alpha;
    *grad_acc += dn_gzb;

    // grad_acc from dzB path: −(∂dzB/∂α)ᵀ · g_dzb
    let dot_dzb_gdzb = dzb.dot(&g_dzb);
    let chain = (dn_gdzb * dot_zb_j + dzb * dot_zb_gdzb + zb * dot_dzb_gdzb) * inv_norm_alpha;
    *grad_acc -= chain;
}
