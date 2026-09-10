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

use nalgebra::{matrix, SMatrix, SVector, Vector3};

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
pub fn normalize_quat<const NX: usize>(x: &mut SVector<f32, NX>) {
    const { assert!(NX >= 7, "normalize_quat requires NX >= 7") };
    let qnorm_sq = x[3] * x[3] + x[4] * x[4] + x[5] * x[5] + x[6] * x[6];
    if qnorm_sq > 1e-12 {
        let inv = 1.0 / libm::sqrtf(qnorm_sq);
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
    x: &SVector<f32, NX>,
    xref: &SVector<f32, NX>,
) -> (Vector3<f32>, SMatrix<f32, 3, 4>, SMatrix<f32, 4, 4>) {
    const { assert!(NX >= 7, "attitude_error requires NX >= 7") };
    let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
    let (rx, ry, rz, rw) = (xref[3], xref[4], xref[5], xref[6]);

    let qa_raw = nalgebra::Vector4::new(
        -qx * rw + qw * rx + qz * ry - qy * rz,
        -qy * rw - qz * rx + qw * ry + qx * rz,
        -qz * rw + qy * rx - qx * ry + qw * rz,
        qw * rw + qx * rx + qy * ry + qz * rz,
    );
    let (qa, sign_flip) = if qa_raw[3] < 0.0 {
        (-qa_raw, -1.0_f32)
    } else {
        (qa_raw, 1.0_f32)
    };

    const EPS: f32 = 1e-3;
    let denom = libm::sqrtf(qa[3] * qa[3] + qa[2] * qa[2] + EPS);
    let inv_d = 1.0 / denom;
    let nr = qa[3] * qa[0] - qa[1] * qa[2];
    let np_ = qa[3] * qa[1] + qa[0] * qa[2];
    let ny = qa[2];
    let ea = Vector3::new(2.0 * nr * inv_d, 2.0 * np_ * inv_d, 2.0 * ny * inv_d);

    let inv_d2 = inv_d * inv_d;
    let dd2 = qa[2] * inv_d;
    let dd3 = qa[3] * inv_d;

    let de = matrix![
        2.0 * qa[3] * inv_d,
            2.0 * (-qa[2]) * inv_d,
            2.0 * (-qa[1] * inv_d - nr * dd2 * inv_d2),
            2.0 * (qa[0] * inv_d - nr * dd3 * inv_d2);
        2.0 * qa[2] * inv_d,
            2.0 * qa[3] * inv_d,
            2.0 * (qa[0] * inv_d - np_ * dd2 * inv_d2),
            2.0 * (qa[1] * inv_d - np_ * dd3 * inv_d2);
        0.0,
            0.0,
            2.0 * (inv_d - ny * dd2 * inv_d2),
            2.0 * (-ny * dd3 * inv_d2);
    ];

    let sf = sign_flip;
    let dqa_dq = matrix![
        -rw * sf, -rz * sf,  ry * sf, rx * sf;
         rz * sf, -rw * sf, -rx * sf, ry * sf;
        -ry * sf,  rx * sf, -rw * sf, rz * sf;
         rx * sf,  ry * sf,  rz * sf, rw * sf;
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
    x: &SVector<f32, NX>,
    xref: &SVector<f32, NX>,
    w_pos: &[f32; 3],
    w_vel: &[f32; 3],
    dt: f32,
    grad_x: &mut SVector<f32, NX>,
) -> f32 {
    const { assert!(NX >= 10, "write_pos_vel_cost_grad requires NX >= 10") };
    let w_pos_v = Vector3::from(*w_pos);
    let w_vel_v = Vector3::from(*w_vel);
    let pos_err = x.fixed_rows::<3>(0) - xref.fixed_rows::<3>(0);
    let vel_err = x.fixed_rows::<3>(7) - xref.fixed_rows::<3>(7);
    grad_x
        .fixed_rows_mut::<3>(0)
        .copy_from(&(pos_err.component_mul(&w_pos_v) * (2.0 * dt)));
    grad_x
        .fixed_rows_mut::<3>(7)
        .copy_from(&(vel_err.component_mul(&w_vel_v) * (2.0 * dt)));
    dt * (pos_err.component_mul(&pos_err).dot(&w_pos_v)
        + vel_err.component_mul(&vel_err).dot(&w_vel_v))
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
    hess_xx: &mut SMatrix<f32, NX, NX>,
) {
    const { assert!(NX >= 10, "write_pos_vel_hess requires NX >= 10") };
    for i in 0..3 {
        hess_xx[(i, i)] = 2.0 * dt * w_pos[i];
        hess_xx[(7 + i, 7 + i)] = 2.0 * dt * w_vel[i];
    }
}

/// Add the velocity quadratic-cost contribution and write its gradient into
/// `grad_x[7..10]`. Returns the cost contribution.
///
/// Companion to [`write_contour_lag_cost_grad`]: when the position cost is
/// computed by the contour/lag helper, the caller still needs the standard
/// velocity term, and this helper provides it without touching `grad_x[0..3]`.
#[inline]
pub fn write_vel_cost_grad<const NX: usize>(
    x: &SVector<f32, NX>,
    xref: &SVector<f32, NX>,
    w_vel: &[f32; 3],
    dt: f32,
    grad_x: &mut SVector<f32, NX>,
) -> f32 {
    const { assert!(NX >= 10, "write_vel_cost_grad requires NX >= 10") };
    let w_vel_v = Vector3::from(*w_vel);
    let vel_err = x.fixed_rows::<3>(7) - xref.fixed_rows::<3>(7);
    grad_x
        .fixed_rows_mut::<3>(7)
        .copy_from(&(vel_err.component_mul(&w_vel_v) * (2.0 * dt)));
    dt * vel_err.component_mul(&vel_err).dot(&w_vel_v)
}

/// Write the velocity diagonal Hessian entries into `hess_xx[7..10][7..10]`.
/// Companion to [`write_contour_lag_hess`].
#[inline]
pub fn write_vel_hess<const NX: usize>(
    w_vel: &[f32; 3],
    dt: f32,
    hess_xx: &mut SMatrix<f32, NX, NX>,
) {
    const { assert!(NX >= 10, "write_vel_hess requires NX >= 10") };
    for i in 0..3 {
        hess_xx[(7 + i, 7 + i)] = 2.0 * dt * w_vel[i];
    }
}

// ───────────────────────────────────────────────────────────────────────────
// MPCTC contour/lag position cost
// ───────────────────────────────────────────────────────────────────────────
//
// Model Predictive Contouring Tracking Control (MPCTC) replaces the plain
// `w_pos · ‖p − p_ref‖²` position cost with a tangent-decomposed form:
//
//   t̂        = vel_ref / ‖vel_ref‖              (unit tangent of the path)
//   e_lag    = t̂ · (p − p_ref)                  (signed scalar, along path)
//   e_contour= (p − p_ref) − e_lag · t̂          (3-vector, orthogonal to path)
//   J_pos    = w_c · ‖e_contour‖² + w_l · e_lag²
//
// Orthogonal-decomposition identity: when `w_c == w_l`, J_pos = w_c·‖e‖² —
// MPCTC with equal weights is byte-identical to the standard quadratic cost.
//
// When `‖vel_ref‖ < vel_eps` (terminal hover, mission start), t̂ is undefined;
// the cost degenerates to `w_c · ‖e‖²` (set t̂t̂ᵀ = 0, lag term vanishes).
//
// Gradient and Gauss-Newton Hessian:
//
//   M       = w_c·I + (w_l − w_c)·t̂t̂ᵀ          (3×3, dense, PSD)
//   ∂J/∂p   = 2·M·(p − p_ref)
//   ∂²J/∂p² = 2·M

/// Add the MPCTC contour/lag position cost and write its gradient into
/// `grad_x[0..3]`. The tangent direction is read from `xref[7..10]`
/// (which is the trajectory velocity reference written by the sampler).
/// Returns the cost contribution.
#[inline]
pub fn write_contour_lag_cost_grad<const NX: usize>(
    x: &SVector<f32, NX>,
    xref: &SVector<f32, NX>,
    w_contour: f32,
    w_lag: f32,
    vel_eps: f32,
    dt: f32,
    grad_x: &mut SVector<f32, NX>,
) -> f32 {
    const { assert!(NX >= 10, "write_contour_lag_cost_grad requires NX >= 10") };
    let pos_err = x.fixed_rows::<3>(0) - xref.fixed_rows::<3>(0);
    let v_ref = xref.fixed_rows::<3>(7);
    let vnorm_sq = v_ref.dot(&v_ref);

    let (cost, grad3) = if vnorm_sq < vel_eps * vel_eps {
        // Hover fallback: t̂t̂ᵀ = 0, M = w_c·I.
        let cost = dt * w_contour * pos_err.dot(&pos_err);
        let grad3 = pos_err * (2.0 * dt * w_contour);
        (cost, grad3)
    } else {
        let inv = 1.0 / libm::sqrtf(vnorm_sq);
        let t_hat = v_ref * inv;
        let e_lag = t_hat.dot(&pos_err);
        let e_contour_sq = pos_err.dot(&pos_err) - e_lag * e_lag;
        let cost = dt * (w_contour * e_contour_sq + w_lag * e_lag * e_lag);
        // grad = 2·dt · (w_c·e + (w_l − w_c)·e_lag·t̂)
        let grad3 = pos_err * (2.0 * dt * w_contour)
            + t_hat * (2.0 * dt * (w_lag - w_contour) * e_lag);
        (cost, grad3)
    };
    grad_x.fixed_rows_mut::<3>(0).copy_from(&grad3);
    cost
}

/// Write the MPCTC dense 3×3 position Hessian block into
/// `hess_xx[0..3][0..3]`. Off-diagonal entries in that block are written
/// (the block is dense), so the caller is expected to have zeroed the
/// 3×3 block first (typically by zeroing the whole `hess_xx`).
#[inline]
pub fn write_contour_lag_hess<const NX: usize>(
    xref: &SVector<f32, NX>,
    w_contour: f32,
    w_lag: f32,
    vel_eps: f32,
    dt: f32,
    hess_xx: &mut SMatrix<f32, NX, NX>,
) {
    const { assert!(NX >= 10, "write_contour_lag_hess requires NX >= 10") };
    let v_ref = xref.fixed_rows::<3>(7);
    let vnorm_sq = v_ref.dot(&v_ref);
    let two_dt = 2.0 * dt;
    // Diagonal w_c·I always present.
    for i in 0..3 {
        hess_xx[(i, i)] = two_dt * w_contour;
    }
    if vnorm_sq >= vel_eps * vel_eps {
        let inv = 1.0 / libm::sqrtf(vnorm_sq);
        let t_hat = v_ref * inv;
        let coef = two_dt * (w_lag - w_contour);
        for i in 0..3 {
            for j in 0..3 {
                hess_xx[(i, j)] += coef * t_hat[i] * t_hat[j];
            }
        }
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
    ea: &Vector3<f32>,
    de: &SMatrix<f32, 3, 4>,
    dqa_dq: &SMatrix<f32, 4, 4>,
    w_att: &[f32; 3],
    dt: f32,
    grad_x: &mut SVector<f32, NX>,
) -> f32 {
    const { assert!(NX >= 7, "write_quat_cost_grad requires NX >= 7") };
    let w = Vector3::from(*w_att);
    let wea = ea.component_mul(&w);
    // grad_x[3..7] = 2·dt · dqa_dq^T · (de^T · wea)
    let grad_quat = (dqa_dq.transpose() * (de.transpose() * wea)) * (2.0 * dt);
    grad_x.fixed_rows_mut::<4>(3).copy_from(&grad_quat);
    dt * ea.component_mul(ea).dot(&w)
}

/// Write the Gauss-Newton quaternion Hessian block
/// `2·dt · J_att^T · diag(W_a) · J_att` into `hess_xx[3..7][3..7]`,
/// where `J_att = de · dqa_dq` (3×4).
#[inline]
pub fn write_quat_hess<const NX: usize>(
    de: &SMatrix<f32, 3, 4>,
    dqa_dq: &SMatrix<f32, 4, 4>,
    w_att: &[f32; 3],
    dt: f32,
    hess_xx: &mut SMatrix<f32, NX, NX>,
) {
    const { assert!(NX >= 7, "write_quat_hess requires NX >= 7") };
    let j_att = de * dqa_dq; // 3×4
    let w_diag = nalgebra::Matrix3::from_diagonal(&Vector3::from(*w_att));
    let hess_block = j_att.transpose() * w_diag * j_att * (2.0 * dt);
    hess_xx.fixed_view_mut::<4, 4>(3, 3).copy_from(&hess_block);
}

// ───────────────────────────────────────────────────────────────────────────
// Body-rate state box constraints — relaxed logarithmic barrier
// ───────────────────────────────────────────────────────────────────────────
//
// State-constraint support following arXiv:2505.01353v2
// (Frey et al.): an IPM-based SQP is equivalent to SQP applied to the
// log-barrier problem (Appendix A.2, eq. 18–20) — inequality constraints
// `h(x) ≤ 0` are replaced by barrier terms `−τ·log(−h(x))` added to the
// stage cost. In a Gauss-Newton Riccati solver the barrier's gradient and
// Hessian flow into the existing stage-cost vectors `q`/`qm`; the backward
// sweep is untouched. Holding a fixed small τ solves the smoothed KKT
// system (paper eq. 10 / Thm. 3), whose solution → the exactly-constrained
// one as τ → 0.
//
// The exact log barrier is undefined at infeasible iterates (`−h ≤ 0`),
// which an SQP rollout can produce (e.g. an initial state already above the
// rate limit). We therefore use the *relaxed* barrier (Feller & Ebenbauer):
// below a margin δ the log branch switches to its quadratic extension,
//
//   B(z) = −ln(z)                          z > δ
//   B(z) = ½·[((z − 2δ)/δ)² − 1] − ln(δ)   z ≤ δ
//
// where `z` is the constraint margin (`z = −h`, strictly feasible for
// z > 0). Value, gradient, and Hessian are continuous at `z = δ`, and the
// extension is defined for all z, so infeasible iterates see a strong
// quadratic push-back instead of a NaN.

/// Scalar relaxed log-barrier: returns `(B(z), B'(z), B''(z))`.
#[inline]
fn relaxed_log_barrier(z: f32, delta: f32) -> (f32, f32, f32) {
    if z > delta {
        let inv = 1.0 / z;
        (-libm::logf(z), -inv, inv * inv)
    } else {
        let inv_d = 1.0 / delta;
        let t = (z - 2.0 * delta) * inv_d;
        (
            0.5 * (t * t - 1.0) - libm::logf(delta),
            t * inv_d,
            inv_d * inv_d,
        )
    }
}

/// Add the relaxed log-barrier cost for per-axis box bounds on the three
/// body-rate states `x[10..13]` and accumulate its gradient into
/// `grad_x[10..13]` (adds to, does not overwrite). Returns the cost
/// contribution.
///
/// `tau` is the barrier weight (τ in the paper), `delta` the relaxation
/// margin. Callers gate on `tau > 0.0` so the unconstrained path stays
/// byte-identical to the pre-barrier behavior.
#[inline]
pub fn write_rate_barrier_cost_grad<const NX: usize>(
    x: &SVector<f32, NX>,
    bounds: &[[f32; 2]; 3],
    tau: f32,
    delta: f32,
    grad_x: &mut SVector<f32, NX>,
) -> f32 {
    const { assert!(NX >= 13, "write_rate_barrier_cost_grad requires NX >= 13") };
    let mut cost = 0.0;
    for (i, &[lb, ub]) in bounds.iter().enumerate() {
        let w = x[10 + i];
        // Upper bound: h = w − ub ≤ 0 → margin z = ub − w, dz/dw = −1.
        let (b_u, db_u, _) = relaxed_log_barrier(ub - w, delta);
        // Lower bound: h = lb − w ≤ 0 → margin z = w − lb, dz/dw = +1.
        let (b_l, db_l, _) = relaxed_log_barrier(w - lb, delta);
        cost += tau * (b_u + b_l);
        grad_x[10 + i] += tau * (db_l - db_u);
    }
    cost
}

/// Add the relaxed log-barrier Hessian contribution for the body-rate box
/// bounds to the diagonal entries `hess_xx[10..13][10..13]` (adds to, does
/// not overwrite). Companion to [`write_rate_barrier_cost_grad`].
///
/// `d²B/dw² = B''(z)·(dz/dw)² = B''(z)` for both bound directions, and
/// `B'' > 0` everywhere, so the contribution keeps the Gauss-Newton stage
/// Hessian positive definite.
#[inline]
pub fn write_rate_barrier_hess<const NX: usize>(
    x: &SVector<f32, NX>,
    bounds: &[[f32; 2]; 3],
    tau: f32,
    delta: f32,
    hess_xx: &mut SMatrix<f32, NX, NX>,
) {
    const { assert!(NX >= 13, "write_rate_barrier_hess requires NX >= 13") };
    for (i, &[lb, ub]) in bounds.iter().enumerate() {
        let w = x[10 + i];
        let (_, _, d2_u) = relaxed_log_barrier(ub - w, delta);
        let (_, _, d2_l) = relaxed_log_barrier(w - lb, delta);
        hess_xx[(10 + i, 10 + i)] += tau * (d2_u + d2_l);
    }
}

/// Add the relaxed log-barrier cost for a maximum-tilt state constraint and
/// accumulate its gradient into `grad_x[3..5]` (adds to, does not
/// overwrite). Returns the cost contribution.
///
/// The constraint is formulated in **cosine space**: with the scalar-last
/// quaternion `[qx, qy, qz, qw]` at rows 3..7, the world-z component of the
/// body z-axis is `cos(tilt) = 1 − 2(qx² + qy²)` — yaw-decoupled (qz, qw do
/// not appear). The margin is
///
///   `z = (1 − 2(qx² + qy²)) − cos_max_tilt ≥ 0`
///
/// Working in cos space (instead of `acos(·) − θ_max`) avoids the
/// `1/√(1−cos²)` gradient singularity at tilt = 180° — exactly the state a
/// tumble delivers — and keeps every quantity polynomial in the quaternion
/// components. The price is a non-uniform margin in angle (δ in cos units);
/// for a fence constraint that trade is free.
///
/// `∇z = (0, −4qx, −4qy, 0)` on the quaternion rows.
#[inline]
pub fn write_tilt_barrier_cost_grad<const NX: usize>(
    x: &SVector<f32, NX>,
    cos_max_tilt: f32,
    tau: f32,
    delta: f32,
    grad_x: &mut SVector<f32, NX>,
) -> f32 {
    const { assert!(NX >= 7, "write_tilt_barrier_cost_grad requires NX >= 7") };
    let (qx, qy) = (x[3], x[4]);
    let z = 1.0 - 2.0 * (qx * qx + qy * qy) - cos_max_tilt;
    let (b, db, _) = relaxed_log_barrier(z, delta);
    grad_x[3] += tau * db * (-4.0 * qx);
    grad_x[4] += tau * db * (-4.0 * qy);
    tau * b
}

/// Add the tilt-barrier Hessian contribution to the `qx`/`qy` block of
/// `hess_xx` (adds to, does not overwrite). Companion to
/// [`write_tilt_barrier_cost_grad`].
///
/// Both terms of the exact Hessian are kept:
/// `∇²(τB) = τ·B''·∇z∇zᵀ + τ·B'·∇²z` with `∇²z = −4·I₂` on the qx/qy
/// block. `B'' > 0` makes the first term PSD, and `B' < 0` everywhere
/// (log branch and quadratic extension alike) makes `B'·(−4·I₂)` PSD too,
/// so the Gauss-Newton stage Hessian stays positive semi-definite without
/// truncation.
#[inline]
pub fn write_tilt_barrier_hess<const NX: usize>(
    x: &SVector<f32, NX>,
    cos_max_tilt: f32,
    tau: f32,
    delta: f32,
    hess_xx: &mut SMatrix<f32, NX, NX>,
) {
    const { assert!(NX >= 7, "write_tilt_barrier_hess requires NX >= 7") };
    let (qx, qy) = (x[3], x[4]);
    let z = 1.0 - 2.0 * (qx * qx + qy * qy) - cos_max_tilt;
    let (_, db, d2b) = relaxed_log_barrier(z, delta);
    let (gx, gy) = (-4.0 * qx, -4.0 * qy);
    hess_xx[(3, 3)] += tau * (d2b * gx * gx - 4.0 * db);
    hess_xx[(4, 4)] += tau * (d2b * gy * gy - 4.0 * db);
    hess_xx[(3, 4)] += tau * d2b * gx * gy;
    hess_xx[(4, 3)] += tau * d2b * gx * gy;
}

// ───────────────────────────────────────────────────────────────────────────
// Input box constraints
// ───────────────────────────────────────────────────────────────────────────

/// Clamp each component of `u` to its per-channel `[lower, upper]` bound.
#[inline]
pub fn clamp_control<const NU: usize>(
    u: &SVector<f32, NU>,
    bounds: &[[f32; 2]; NU],
) -> SVector<f32, NU> {
    SVector::<f32, NU>::from_fn(|i, _| u[i].clamp(bounds[i][0], bounds[i][1]))
}

/// Cubic box-constraint penalty + gradient + Hessian-diagonal contribution.
///
/// Adds to (does not overwrite) the existing `grad_u` and `r_diag`. Returns
/// the penalty cost contribution.
#[inline]
pub fn constraint_hess_grad<const NU: usize>(
    u: &SVector<f32, NU>,
    bounds: &[[f32; 2]; NU],
    rho: f32,
    grad_u: &mut SVector<f32, NU>,
    r_diag: &mut SVector<f32, NU>,
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

// ───────────────────────────────────────────────────────────────────────────
// Rotor drag (system-identification model, analysis/sysid_mcap.py)
// ───────────────────────────────────────────────────────────────────────────

/// Rotor-speed-proportional drag, the model `analysis/sysid_mcap.py` fits:
/// body-frame specific force `a_b,i = −(c_i/m)·Σω·v_b,i` with
/// `c_i = −m·k_i` the YAML `sim: aero_drag` coefficients [N·s²/(m·rad)].
/// The MPC has no rotor speeds, so `Σω` is taken from the collective
/// thrust it commands, `T = c_T·Σωᵢ²` with equal rotors:
/// `Σω(T) = 2·√(T/c_T)`, `c_T` the per-motor thrust coefficient
/// `max_thrust_n / ω_max²` [N·s²].
///
/// Returns the world-frame drag acceleration and its Jacobians with
/// respect to the world velocity (3×3), the scalar-last quaternion
/// `[x, y, z, w]` (3×4) and the collective thrust (3×1). Disabled
/// (all zero) when `c_t <= 0`.
#[allow(clippy::type_complexity)]
pub fn rotor_drag_accel_jac(
    q_xyzw: [f32; 4],
    v_world: Vector3<f32>,
    thrust_n: f32,
    coeff: &[f32; 3],
    c_t: f32,
    mass_inv: f32,
) -> (Vector3<f32>, SMatrix<f32, 3, 3>, SMatrix<f32, 3, 4>, Vector3<f32>) {
    drag_accel_jac(q_xyzw, v_world, thrust_n, coeff, c_t, &[0.0; 3], mass_inv)
}

/// Rotor drag (linear in `Σω·v_b`, see [`rotor_drag_accel_jac`]) plus
/// quadratic body drag `a_b,i = −(k_q,i/m)·|v_b,i|·v_b,i` with
/// `k_q = ½ρC_dA` per body axis [N·s²/m²] — the term that dominates
/// above ~30 m/s. Either term is disabled by zero coefficients (the rotor
/// term also by `c_t <= 0`). Same Jacobian outputs as the rotor-only
/// helper; the quadratic term contributes to `∂a/∂v` and `∂a/∂q` only.
#[allow(clippy::type_complexity)]
pub fn drag_accel_jac(
    q_xyzw: [f32; 4],
    v_world: Vector3<f32>,
    thrust_n: f32,
    coeff: &[f32; 3],
    c_t: f32,
    quad_coeff: &[f32; 3],
    mass_inv: f32,
) -> (Vector3<f32>, SMatrix<f32, 3, 3>, SMatrix<f32, 3, 4>, Vector3<f32>) {
    let mut a = Vector3::zeros();
    let mut da_dv = SMatrix::<f32, 3, 3>::zeros();
    let mut da_dq = SMatrix::<f32, 3, 4>::zeros();
    let mut da_dt = Vector3::zeros();
    // `!(c_t > 0)` also rejects NaN (a NaN `max_thrust_n/ω_max²` from a
    // bad config must disable the term, not poison the acceleration).
    let rotor_on = c_t > 0.0 && coeff.iter().all(|c| c.is_finite()) && coeff.iter().any(|&c| c != 0.0);
    let quad_on = quad_coeff.iter().all(|c| c.is_finite()) && quad_coeff.iter().any(|&c| c != 0.0);
    if !rotor_on && !quad_on {
        return (a, da_dv, da_dq, da_dt);
    }
    const T_EPS: f32 = 0.05;
    let t = thrust_n.max(T_EPS);
    // Rotor speed from the commanded collective (zero when the rotor term is off).
    let s = if rotor_on { 2.0 * libm::sqrtf(t / c_t) } else { 0.0 };
    // Derivative of the clamped `Σω(t)`: continuous across the floor and
    // bounded by `1/√(T_EPS·c_T)`, so the thrust column of the Jacobian
    // cannot jump sign when the solve dips to zero collective.
    let ds_dt = if rotor_on { 1.0 / libm::sqrtf(t * c_t) } else { 0.0 };

    let (x, y, z, w) = (q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]);
    // R(q) (body → world), scalar-last quaternion, polynomial form.
    let r = nalgebra::Matrix3::new(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    );
    let vb = r.transpose() * v_world;
    let cvec = Vector3::from(*coeff);
    let qvec = Vector3::from(*quad_coeff);
    let k = -mass_inv;
    // f_b,i = k·( s·c_i·v_b,i + k_q,i·|v_b,i|·v_b,i )
    let fb = Vector3::new(
        k * (s * cvec.x * vb.x + qvec.x * vb.x.abs() * vb.x),
        k * (s * cvec.y * vb.y + qvec.y * vb.y.abs() * vb.y),
        k * (s * cvec.z * vb.z + qvec.z * vb.z.abs() * vb.z),
    );
    a = r * fb;

    // ∂f_b/∂v_b = k·diag( s·c_i + 2·k_q,i·|v_b,i| );  ∂a/∂v = R ∂f_b/∂v_b Rᵀ
    let dfb_dvb = nalgebra::Matrix3::from_diagonal(&Vector3::new(
        k * (s * cvec.x + 2.0 * qvec.x * vb.x.abs()),
        k * (s * cvec.y + 2.0 * qvec.y * vb.y.abs()),
        k * (s * cvec.z + 2.0 * qvec.z * vb.z.abs()),
    ));
    da_dv = r * dfb_dvb * r.transpose();

    // ∂a/∂T through Σω(T)
    da_dt = r * Vector3::new(cvec.x * vb.x, cvec.y * vb.y, cvec.z * vb.z) * (k * ds_dt);

    // ∂a/∂q_j = (∂R/∂q_j)·f_b + R·K·(∂R/∂q_j)ᵀ·v,  K = k·s·diag(c).
    // The ∂R/∂q_j are the component derivatives of the *polynomial* R above
    // (the same form `dynamics` evaluates and the thrust Jacobian uses),
    // not the unit-quaternion tangent formulas — the SQP linearizes the
    // polynomial, so its derivatives must be the polynomial's.
    let dr = [
        nalgebra::Matrix3::new(0.0, 2.0 * y, 2.0 * z, 2.0 * y, -4.0 * x, -2.0 * w, 2.0 * z, 2.0 * w, -4.0 * x),
        nalgebra::Matrix3::new(-4.0 * y, 2.0 * x, 2.0 * w, 2.0 * x, 0.0, 2.0 * z, -2.0 * w, 2.0 * z, -4.0 * y),
        nalgebra::Matrix3::new(-4.0 * z, -2.0 * w, 2.0 * x, 2.0 * w, -4.0 * z, 2.0 * y, 2.0 * x, 2.0 * y, 0.0),
        nalgebra::Matrix3::new(0.0, -2.0 * z, 2.0 * y, 2.0 * z, 0.0, -2.0 * x, -2.0 * y, 2.0 * x, 0.0),
    ];
    let r_k = r * dfb_dvb;
    for j in 0..4 {
        let col = dr[j] * fb + r_k * (dr[j].transpose() * v_world);
        for i in 0..3 {
            da_dq[(i, j)] = col[i];
        }
    }
    (a, da_dv, da_dq, da_dt)
}
