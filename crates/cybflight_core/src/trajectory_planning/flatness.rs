//! Differential flatness for quadrotors at zero yaw (ψ=0).
//!
//! Forward chain: position derivatives → thrust vector, body axis, body rates.
//! Backward chain: gradient backpropagation through the same transforms.
//!
//! All functions are pure math with no planner dependency.

#[allow(unused_imports)]
use num_traits::Float;

use super::types::{dot3, norm_sq3, Vec3, ZERO3};

/// Intermediate flatness quantities at a single trajectory sample point.
///
/// Bundles the shared intermediates that multiple penalties reuse,
/// avoiding redundant computation of zb, dzb, omega, etc.
pub struct FlatnessState {
    /// Thrust vector: α = acc + [0,0,g].
    pub alpha: Vec3,
    /// ‖α‖.
    pub norm_alpha: f32,
    /// 1/‖α‖.
    pub inv_norm_alpha: f32,
    /// Body z-axis: α/‖α‖.
    pub zb: Vec3,
    /// dot(zb, jerk).
    pub dot_zb_j: f32,
    /// Body z-axis time derivative: dzB = DN(α)·j / ‖α‖.
    pub dzb: Vec3,
    /// 1/(1 + zb_z), with singularity guard.
    pub s_inv: f32,
    /// Body rates [ωx, ωy, ωz] at ψ=0.
    pub omega: Vec3,
}

/// Compute flatness state from acceleration and jerk.
///
/// Returns `None` if zb_z <= -0.9 (model undefined near fully inverted).
pub fn compute_flatness_state(acc: Vec3, jer: Vec3, gravity: f32) -> Option<FlatnessState> {
    let alpha = [acc[0], acc[1], acc[2] + gravity];
    let norm_alpha = norm_sq3(alpha).sqrt().max(1e-8);
    let inv_norm_alpha = 1.0 / norm_alpha;
    let zb = [
        alpha[0] * inv_norm_alpha,
        alpha[1] * inv_norm_alpha,
        alpha[2] * inv_norm_alpha,
    ];

    if zb[2] <= -0.9 {
        return None;
    }

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

    Some(FlatnessState {
        alpha,
        norm_alpha,
        inv_norm_alpha,
        zb,
        dot_zb_j,
        dzb,
        s_inv,
        omega,
    })
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
