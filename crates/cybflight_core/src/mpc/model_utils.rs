//! Shared helpers used by both `FullQuadModel` and `QuadModel`.
//!
//! Every function in this module is:
//! - **`#[inline]`** so the compiler folds the call at every use site,
//! - **const-generic over `NX` or `NU`** so the array dimensions stay
//!   compile-time constants and indexing compiles to the same machine
//!   instructions as inlined model-local code,
//! - **pure and allocation-free** — they read inputs, write to caller-owned
//!   buffers, and return at most a small fixed-size value.
//!
//! These three properties guarantee that extracting these helpers does not
//! slow the solver down or change its numerical output. The convergence
//! tests are byte-identical regression gates for this property.
//!
//! Helpers operating on a state vector assume the canonical layout shared
//! by both quadrotor models in this module:
//! - indices 0..3  → world position
//! - indices 3..7  → quaternion (scalar-LAST: qx, qy, qz, qw)
//! - indices 7..10 → world velocity
//! - (FullQuadModel only) indices 10..13 → body angular rates
//!
//! Each helper carries a `const { assert!(NX >= 7) }` (or 10 / 13 as needed)
//! so calling it with an incompatible state size is a compile-time error.

use num_traits::Float;

// ───────────────────────────────────────────────────────────────────────────
// Quaternion projection
// ───────────────────────────────────────────────────────────────────────────

/// Project the quaternion components of a state vector back onto the unit
/// 3-sphere. Cheap (one sqrt + one divide + four multiplies) and idempotent.
///
/// Assumes the quaternion lives at indices 3..7 in the state vector. The
/// const-generic `NX` lets the same function serve both `FullQuadModel`
/// (NX=13) and `QuadModel` (NX=10).
#[inline]
pub fn normalize_quat<const NX: usize>(x: &mut [f32; NX]) {
    const { assert!(NX >= 7, "normalize_quat requires NX >= 7") };
    let qnorm_sq = x[3] * x[3] + x[4] * x[4] + x[5] * x[5] + x[6] * x[6];
    if qnorm_sq > 1e-12 {
        let inv = 1.0 / qnorm_sq.sqrt();
        x[3] *= inv;
        x[4] *= inv;
        x[5] *= inv;
        x[6] *= inv;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tilt-prioritized attitude error and chain-rule Jacobians
// ───────────────────────────────────────────────────────────────────────────

/// Compute the tilt-prioritized attitude error `ea ∈ R³` and the chain-rule
/// Jacobians needed by the quaternion-block cost gradient/Hessian:
///
/// - `ea[3]`        — error vector (roll/pitch/yaw components, scaled)
/// - `de[3][4]`     — `∂ea/∂qa` (3×4)
/// - `dqa_dq[4][4]` — `∂qa/∂q`  (4×4), with sign-flip for half-angle uniqueness
///
/// where `qa = conj(q) ⊗ qref` (the body-frame error quaternion).
///
/// The implementation is identical to the formula in
/// `attitude_control::geometric_controller::TiltPrioritizing` (modulo a
/// sign convention that doesn't affect the squared cost). See the inline
/// comments there for the math derivation.
#[inline]
pub fn attitude_error<const NX: usize>(
    x: &[f32; NX],
    xref: &[f32; NX],
) -> ([f32; 3], [[f32; 4]; 3], [[f32; 4]; 4]) {
    const { assert!(NX >= 7, "attitude_error requires NX >= 7") };
    let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
    let (rx, ry, rz, rw) = (xref[3], xref[4], xref[5], xref[6]);

    let mut qa = [
        -qx * rw + qw * rx + qz * ry - qy * rz,
        -qy * rw - qz * rx + qw * ry + qx * rz,
        -qz * rw + qy * rx - qx * ry + qw * rz,
        qw * rw + qx * rx + qy * ry + qz * rz,
    ];

    let mut sign_flip = 1.0f32;
    if qa[3] < 0.0 {
        for v in qa.iter_mut() {
            *v = -*v;
        }
        sign_flip = -1.0;
    }

    const EPS: f32 = 1e-3;
    let denom = (qa[3] * qa[3] + qa[2] * qa[2] + EPS).sqrt();
    let inv_d = 1.0 / denom;
    let nr = qa[3] * qa[0] - qa[1] * qa[2];
    let np_ = qa[3] * qa[1] + qa[0] * qa[2];
    let ny = qa[2];
    let ea = [2.0 * nr * inv_d, 2.0 * np_ * inv_d, 2.0 * ny * inv_d];

    let inv_d2 = inv_d * inv_d;
    let dd2 = qa[2] * inv_d;
    let dd3 = qa[3] * inv_d;

    let de: [[f32; 4]; 3] = [
        [
            2.0 * qa[3] * inv_d,
            2.0 * (-qa[2]) * inv_d,
            2.0 * (-qa[1] * inv_d - nr * dd2 * inv_d2),
            2.0 * (qa[0] * inv_d - nr * dd3 * inv_d2),
        ],
        [
            2.0 * qa[2] * inv_d,
            2.0 * qa[3] * inv_d,
            2.0 * (qa[0] * inv_d - np_ * dd2 * inv_d2),
            2.0 * (qa[1] * inv_d - np_ * dd3 * inv_d2),
        ],
        [
            0.0,
            0.0,
            2.0 * (inv_d - ny * dd2 * inv_d2),
            2.0 * (-ny * dd3 * inv_d2),
        ],
    ];

    let sf = sign_flip;
    let dqa_dq: [[f32; 4]; 4] = [
        [-rw * sf, -rz * sf, ry * sf, rx * sf],
        [rz * sf, -rw * sf, -rx * sf, ry * sf],
        [-ry * sf, rx * sf, -rw * sf, rz * sf],
        [rx * sf, ry * sf, rz * sf, rw * sf],
    ];

    (ea, de, dqa_dq)
}

// ───────────────────────────────────────────────────────────────────────────
// Position + velocity cost / gradient / Hessian
// ───────────────────────────────────────────────────────────────────────────

/// Add the position and velocity quadratic-cost contributions to `cost` and
/// write their gradients into `grad_x[0..3]` (position) and `grad_x[7..10]`
/// (velocity). Returns the contribution to the running cost so the caller
/// can accumulate it.
#[inline]
pub fn write_pos_vel_cost_grad<const NX: usize>(
    x: &[f32; NX],
    xref: &[f32; NX],
    w_pos: &[f32; 3],
    w_vel: &[f32; 3],
    dt: f32,
    grad_x: &mut [f32; NX],
) -> f32 {
    const { assert!(NX >= 10, "write_pos_vel_cost_grad requires NX >= 10") };
    let mut cost = 0.0;
    for i in 0..3 {
        let ep = x[i] - xref[i];
        cost += dt * ep * ep * w_pos[i];
        grad_x[i] = 2.0 * ep * w_pos[i] * dt;
        let ev = x[7 + i] - xref[7 + i];
        cost += dt * ev * ev * w_vel[i];
        grad_x[7 + i] = 2.0 * ev * w_vel[i] * dt;
    }
    cost
}

/// Write the position and velocity diagonal Hessian entries into
/// `hess_xx[0..3][0..3]` and `hess_xx[7..10][7..10]`. Off-diagonal entries
/// in those blocks are zero (the caller is expected to have zeroed
/// `hess_xx` first).
#[inline]
pub fn write_pos_vel_hess<const NX: usize>(
    w_pos: &[f32; 3],
    w_vel: &[f32; 3],
    dt: f32,
    hess_xx: &mut [[f32; NX]; NX],
) {
    const { assert!(NX >= 10, "write_pos_vel_hess requires NX >= 10") };
    for i in 0..3 {
        hess_xx[i][i] = 2.0 * dt * w_pos[i];
        hess_xx[7 + i][7 + i] = 2.0 * dt * w_vel[i];
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Quaternion cost / gradient / Hessian
// ───────────────────────────────────────────────────────────────────────────

/// Add the quaternion attitude-error quadratic cost to the running total
/// and write the chain-rule gradient `2·dt · dqa_dq^T · de^T · (W_a .* ea)`
/// into `grad_x[3..7]`. Returns the cost contribution.
#[inline]
pub fn write_quat_cost_grad<const NX: usize>(
    ea: &[f32; 3],
    de: &[[f32; 4]; 3],
    dqa_dq: &[[f32; 4]; 4],
    w_att: &[f32; 3],
    dt: f32,
    grad_x: &mut [f32; NX],
) -> f32 {
    const { assert!(NX >= 7, "write_quat_cost_grad requires NX >= 7") };
    let mut cost = 0.0;
    let mut wea = [0.0f32; 3];
    for i in 0..3 {
        cost += dt * ea[i] * ea[i] * w_att[i];
        wea[i] = ea[i] * w_att[i];
    }
    // gqa = de^T @ wea (4-vector)
    let mut gqa = [0.0f32; 4];
    for j in 0..4 {
        for i in 0..3 {
            gqa[j] += de[i][j] * wea[i];
        }
    }
    // grad_x[3..7] = 2·dt · dqa_dq^T @ gqa
    for i in 0..4 {
        let mut s = 0.0;
        for j in 0..4 {
            s += dqa_dq[j][i] * gqa[j];
        }
        grad_x[3 + i] = 2.0 * dt * s;
    }
    cost
}

/// Write the Gauss-Newton quaternion Hessian block
/// `2·dt · J_att^T · diag(W_a) · J_att` into `hess_xx[3..7][3..7]`,
/// where `J_att = de · dqa_dq` (3×4).
#[inline]
pub fn write_quat_hess<const NX: usize>(
    de: &[[f32; 4]; 3],
    dqa_dq: &[[f32; 4]; 4],
    w_att: &[f32; 3],
    dt: f32,
    hess_xx: &mut [[f32; NX]; NX],
) {
    const { assert!(NX >= 7, "write_quat_hess requires NX >= 7") };
    // J_att (3×4) = de @ dqa_dq
    let mut j_att = [[0.0f32; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..4 {
                j_att[i][j] += de[i][k] * dqa_dq[k][j];
            }
        }
    }
    // hess[3..7][3..7] = 2·dt · J_att^T · diag(w_att) · J_att
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..3 {
                s += j_att[k][i] * w_att[k] * j_att[k][j];
            }
            hess_xx[3 + i][3 + j] = 2.0 * dt * s;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Input box constraints
// ───────────────────────────────────────────────────────────────────────────

/// Clamp each component of `u` to its per-channel `[lower, upper]` bound.
#[inline]
pub fn clamp_control<const NU: usize>(u: &[f32; NU], bounds: &[[f32; 2]; NU]) -> [f32; NU] {
    let mut result = *u;
    for i in 0..NU {
        result[i] = result[i].clamp(bounds[i][0], bounds[i][1]);
    }
    result
}

/// Cubic box-constraint penalty + gradient + Hessian-diagonal contribution.
///
/// Adds to (does not overwrite) the existing `grad_u` and `r_diag`. Returns
/// the penalty cost contribution.
#[inline]
pub fn constraint_hess_grad<const NU: usize>(
    u: &[f32; NU],
    bounds: &[[f32; 2]; NU],
    rho: f32,
    grad_u: &mut [f32; NU],
    r_diag: &mut [f32; NU],
) -> f32 {
    let mut penalty = 0.0;
    for i in 0..NU {
        let lb = bounds[i][0];
        let ub = bounds[i][1];
        let lpen = lb - u[i];
        if lpen > 0.0 {
            penalty += rho * lpen * lpen * lpen;
            grad_u[i] += -rho * 3.0 * lpen * lpen;
            r_diag[i] += rho * 6.0 * lpen;
        } else {
            let upen = u[i] - ub;
            if upen > 0.0 {
                penalty += rho * upen * upen * upen;
                grad_u[i] += rho * 3.0 * upen * upen;
                r_diag[i] += rho * 6.0 * upen;
            }
        }
    }
    penalty
}
