//! Penalty functions, time parameterization, and polynomial derivative caching.

#[allow(unused_imports)]
use num_traits::Float;

use super::piecewise_polynomial::PiecewisePolynomial;
use super::types::{fma3, scale3, Vec3};

// ---------------------------------------------------------------------------
// Smoothed L1 penalty
// ---------------------------------------------------------------------------

/// Smoothed L1 penalty function.
///
/// Returns `(f, df)` where `f` is the penalty value and `df` its derivative.
/// - `x < 0` (no violation): `(0, 0)`
/// - `x > mu`: linear tail `(x - mu/2, 1)`
/// - `0 <= x <= mu`: cubic smoothing transition
#[inline]
pub fn smoothed_l1(x: f32, mu: f32) -> (f32, f32) {
    if x < 0.0 {
        (0.0, 0.0)
    } else if x > mu {
        (x - 0.5 * mu, 1.0)
    } else {
        let xdmu = x / mu;
        let sqr = xdmu * xdmu;
        let mumxd2 = mu - 0.5 * x;
        let f = mumxd2 * sqr * xdmu;
        let df = sqr * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
        (f, df)
    }
}

// ---------------------------------------------------------------------------
// Time parameterization (log map)
// ---------------------------------------------------------------------------

/// Forward: K → T = exp(K). Always positive.
#[inline]
pub fn forward_t_log(k: f32) -> f32 {
    k.exp()
}

/// Backward: T → K = ln(T). Clamps T to 1e-5 minimum.
#[inline]
pub fn backward_t_log(t: f32) -> f32 {
    t.max(1e-5).ln()
}

// ---------------------------------------------------------------------------
// Dynamics derivative cache
// ---------------------------------------------------------------------------

/// Cached trajectory derivatives at a sample point, with precomputed powers
/// of t for monomial basis gradient assembly.
pub struct DynDerivatives {
    pub vel: Vec3,
    pub acc: Vec3,
    pub jer: Vec3,
    pub sna: Vec3,
    /// Cached powers: t, t², t³, t⁴.
    pub s1: f32,
    pub s2: f32,
    pub s3: f32,
    pub s4: f32,
}

/// Fused evaluation of vel/acc/jer/snap at local time `t` for segment `seg`.
///
/// Skips position (unused in dynamics penalties). Precomputes powers of t once,
/// avoiding redundant work across 4 derivative evaluations.
#[inline]
pub fn eval_dynamics_derivatives(
    traj: &PiecewisePolynomial,
    seg: usize,
    t: f32,
) -> DynDerivatives {
    let c = &traj.piece(seg).coeffs;

    let s1 = t;
    let s2 = s1 * s1;
    let s3 = s2 * s1;
    let s4 = s2 * s2;

    // Velocity: Σ k·c[k]·t^(k-1) for k=1..5
    let mut vel = scale3(c[1], 1.0);
    vel = fma3(scale3(c[2], 2.0), s1, vel);
    vel = fma3(scale3(c[3], 3.0), s2, vel);
    vel = fma3(scale3(c[4], 4.0), s3, vel);
    vel = fma3(scale3(c[5], 5.0), s4, vel);

    // Acceleration: Σ k(k-1)·c[k]·t^(k-2) for k=2..5
    let mut acc = scale3(c[2], 2.0);
    acc = fma3(scale3(c[3], 6.0), s1, acc);
    acc = fma3(scale3(c[4], 12.0), s2, acc);
    acc = fma3(scale3(c[5], 20.0), s3, acc);

    // Jerk: Σ k(k-1)(k-2)·c[k]·t^(k-3) for k=3..5
    let mut jer = scale3(c[3], 6.0);
    jer = fma3(scale3(c[4], 24.0), s1, jer);
    jer = fma3(scale3(c[5], 60.0), s2, jer);

    // Snap: Σ k(k-1)(k-2)(k-3)·c[k]·t^(k-4) for k=4..5
    let sna = fma3(scale3(c[5], 120.0), s1, scale3(c[4], 24.0));

    DynDerivatives {
        vel,
        acc,
        jer,
        sna,
        s1,
        s2,
        s3,
        s4,
    }
}

// ---------------------------------------------------------------------------
// Monomial basis vectors for jerk order (degree 5, 6 coefficients)
// ---------------------------------------------------------------------------

const JERK_COEFFS: usize = 6;

/// Compute monomial basis vectors for gradient assembly (jerk order, degree 5).
///
/// Returns `(beta_vel, beta_acc, beta_jer)` — arrays of length 6 containing
/// the monomial basis coefficients for each derivative order. These are
/// multiplied element-wise with the derivative gradients and accumulated into
/// the coefficient gradient buffer.
#[inline]
pub fn jerk_basis_vectors(dd: &DynDerivatives) -> ([f32; JERK_COEFFS], [f32; JERK_COEFFS], [f32; JERK_COEFFS]) {
    let mut beta_vel = [0.0f32; JERK_COEFFS];
    let mut beta_acc = [0.0f32; JERK_COEFFS];
    let mut beta_jer = [0.0f32; JERK_COEFFS];

    beta_vel[1] = 1.0;
    beta_vel[2] = 2.0 * dd.s1;
    beta_vel[3] = 3.0 * dd.s2;
    beta_vel[4] = 4.0 * dd.s3;
    beta_vel[5] = 5.0 * dd.s4;

    beta_acc[2] = 2.0;
    beta_acc[3] = 6.0 * dd.s1;
    beta_acc[4] = 12.0 * dd.s2;
    beta_acc[5] = 20.0 * dd.s3;

    beta_jer[3] = 6.0;
    beta_jer[4] = 24.0 * dd.s1;
    beta_jer[5] = 60.0 * dd.s2;

    (beta_vel, beta_acc, beta_jer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothed_l1_no_violation() {
        let (f, df) = smoothed_l1(-1.0, 0.1);
        assert_eq!(f, 0.0);
        assert_eq!(df, 0.0);
    }

    #[test]
    fn smoothed_l1_linear_tail() {
        let (f, df) = smoothed_l1(1.0, 0.1);
        assert!((f - (1.0 - 0.05)).abs() < 1e-6);
        assert!((df - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smoothed_l1_smooth_at_zero() {
        let (f, _df) = smoothed_l1(0.0, 0.1);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn time_log_round_trip() {
        let t = 2.5;
        let k = backward_t_log(t);
        let t2 = forward_t_log(k);
        assert!((t - t2).abs() < 1e-6);
    }
}
