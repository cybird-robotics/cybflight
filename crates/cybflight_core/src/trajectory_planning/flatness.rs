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
//!
//! Splitting avoids computing ω when only tilt/thrust are active.

#[allow(unused_imports)]
use num_traits::Float;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

use super::types::{dot3, norm_sq3, Vec3};

/// Thrust-vector basis. Computed from acceleration alone — cheap.
///
/// Always well-defined (division by `max(‖α‖, 1e-8)` clamps the singularity).
/// Callers that use ω must additionally check `zb[2] > -0.9` before extending
/// to [`FlatnessState`].
pub struct AlphaState {
    /// α = acc + [0, 0, g].
    pub alpha: Vec3,
    /// ‖α‖.
    pub norm_alpha: f32,
    /// 1/‖α‖.
    pub inv_norm_alpha: f32,
    /// Body z-axis: α/‖α‖.
    pub zb: Vec3,
}

/// Full flatness state, including body-rate terms.
///
/// Duplicates the [`AlphaState`] fields for direct access; extended with
/// jerk-dependent quantities used by the body-rate penalty and its gradient.
pub struct FlatnessState {
    // --- Alpha fields (copied from AlphaState) ---
    pub alpha: Vec3,
    pub norm_alpha: f32,
    pub inv_norm_alpha: f32,
    pub zb: Vec3,
    // --- Body-rate fields ---
    /// dot(zB, jer).
    pub dot_zb_j: f32,
    /// Body z-axis time derivative: dzB = DN(α)·j / ‖α‖.
    pub dzb: Vec3,
    /// 1/(1 + zb_z), with singularity guard.
    pub s_inv: f32,
    /// Body rates [ωx, ωy, ωz] at ψ=0.
    pub omega: Vec3,
}

/// Compute the thrust-vector basis (α, ‖α‖, zB) from acceleration.
#[inline]
pub fn compute_alpha_state(acc: Vec3, gravity: f32) -> AlphaState {
    let alpha = [acc[0], acc[1], acc[2] + gravity];
    let norm_alpha = norm_sq3(alpha).sqrt().max(1e-8);
    let inv_norm_alpha = 1.0 / norm_alpha;
    let zb = [
        alpha[0] * inv_norm_alpha,
        alpha[1] * inv_norm_alpha,
        alpha[2] * inv_norm_alpha,
    ];
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
    let dot_zb_j = dot3(zb, jer);
    let dzb = [
        (jer[0] - zb[0] * dot_zb_j) * inv_norm_alpha,
        (jer[1] - zb[1] * dot_zb_j) * inv_norm_alpha,
        (jer[2] - zb[2] * dot_zb_j) * inv_norm_alpha,
    ];
    let s_inv = 1.0 / (1.0 + zb[2]).max(0.01);
    let omega = [
        -dzb[1] + s_inv * zb[1] * dzb[2],
        dzb[0] - s_inv * zb[0] * dzb[2],
        s_inv * (zb[1] * dzb[0] - zb[0] * dzb[1]),
    ];
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
///
/// Direct port of `QuadManifold::toStateWithTrueYaw` (quaternion branch).
/// Conventions: ENU world / FLU body, `gravity` is positive (e.g. 9.81),
/// so `GVEC = (0, 0, -gravity)` and `accCmd = acc - GVEC = acc + (0, 0, g)`.
///
///   z_B   = accCmd.normalized()
///   x_c   = R_z(yaw) · x̂      y_c = R_z(yaw) · ŷ
///   x_B   = (y_c × z_B).normalized()
///   y_B   = (z_B × x_B).normalized()
///   R_W_B = [x_B  y_B  z_B]    → quaternion
///
/// Floors `‖accCmd‖` and `‖y_c × z_B‖` at 1e-8 to keep the result finite
/// in free-fall and inverted-tilt edge cases; the planner's tilt-limit
/// penalty already keeps trajectories well clear of those regimes.
#[inline]
pub fn reference_quaternion(acc: Vec3, yaw_rad: f32, gravity: f32) -> UnitQuaternion<f32> {
    let acc_cmd = Vector3::new(acc[0], acc[1], acc[2] + gravity);
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
    g_omega: &Vec3,
    fs: &FlatnessState,
    grad_acc: &mut Vec3,
    grad_jer: &mut Vec3,
) {
    let zb = &fs.zb;
    let dzb = &fs.dzb;
    let s_inv = fs.s_inv;
    let inv_norm_alpha = fs.inv_norm_alpha;
    let dot_zb_j = fs.dot_zb_j;

    // grad_ω → grad_dzB via (∂ω/∂dzB)ᵀ
    let g_dzb = [
        g_omega[1] + s_inv * zb[1] * g_omega[2],
        -g_omega[0] - s_inv * zb[0] * g_omega[2],
        s_inv * (zb[1] * g_omega[0] - zb[0] * g_omega[1]),
    ];

    // grad_ω → grad_zB via (∂ω/∂zB)ᵀ
    let s2 = s_inv * s_inv;
    let g_zb = [
        -s_inv * (dzb[2] * g_omega[1] + dzb[1] * g_omega[2]),
        s_inv * (dzb[2] * g_omega[0] + dzb[0] * g_omega[2]),
        s2 * (dzb[2] * (zb[0] * g_omega[1] - zb[1] * g_omega[0])
            + (zb[0] * dzb[1] - zb[1] * dzb[0]) * g_omega[2]),
    ];

    // DN(α) is the nullspace projector: DN(v) = (v − zB(zB·v)) / ‖α‖
    let dot_zb_gdzb = dot3(*zb, g_dzb);
    let dot_zb_gzb = dot3(*zb, g_zb);

    // grad_jerk = DN(α) · g_dzb
    for d in 0..3 {
        grad_jer[d] += (g_dzb[d] - zb[d] * dot_zb_gdzb) * inv_norm_alpha;
    }

    // grad_acc from zB path: DN(α) · g_zb
    for d in 0..3 {
        grad_acc[d] += (g_zb[d] - zb[d] * dot_zb_gzb) * inv_norm_alpha;
    }

    // grad_acc from dzB path: (∂dzB/∂α)ᵀ · g_dzb
    let dn_gdzb = [
        (g_dzb[0] - zb[0] * dot_zb_gdzb) * inv_norm_alpha,
        (g_dzb[1] - zb[1] * dot_zb_gdzb) * inv_norm_alpha,
        (g_dzb[2] - zb[2] * dot_zb_gdzb) * inv_norm_alpha,
    ];
    let dot_dzb_gdzb = dot3(*dzb, g_dzb);
    for d in 0..3 {
        grad_acc[d] -=
            (dot_zb_j * dn_gdzb[d] + dot_zb_gdzb * dzb[d] + dot_dzb_gdzb * zb[d])
                * inv_norm_alpha;
    }
}
