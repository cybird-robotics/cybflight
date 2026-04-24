//! Penalty functions, time parameterization, and polynomial derivative caching.

#[allow(unused_imports)]
use num_traits::Float;

use nalgebra::Vector3;

// ---------------------------------------------------------------------------
// Smoothed L1 penalty
// ---------------------------------------------------------------------------

/// Smoothed L1 penalty function.
///
/// Returns `(f, df)` where `f` is the penalty value and `df` its derivative.
/// - `x < 0` (no violation): `(0, 0)`
/// - `x > mu`: linear tail `(x - mu/2, 1)`
/// - `0 <= x <= mu`: cubic smoothing transition
///
/// For hot-path use, prefer [`smoothed_l1_inv`] with a precomputed `inv_mu`
/// to avoid a per-call `f32` division (14 cycles on Cortex-M7).
#[inline]
pub fn smoothed_l1(x: f32, mu: f32) -> (f32, f32) {
    smoothed_l1_inv(x, mu, 1.0 / mu)
}

/// Smoothed L1 penalty — variant taking `inv_mu = 1.0 / mu` so the division
/// can be hoisted out of a sample loop.
#[inline]
pub fn smoothed_l1_inv(x: f32, mu: f32, inv_mu: f32) -> (f32, f32) {
    if x < 0.0 {
        (0.0, 0.0)
    } else if x > mu {
        (x - 0.5 * mu, 1.0)
    } else {
        let xdmu = x * inv_mu;
        let sqr = xdmu * xdmu;
        let mumxd2 = mu - 0.5 * x;
        let f = mumxd2 * sqr * xdmu;
        let df = sqr * ((-0.5) * xdmu + 3.0 * mumxd2 * inv_mu);
        (f, df)
    }
}

// ---------------------------------------------------------------------------
// Time parameterization (piecewise quadratic map K ↔ T)
// ---------------------------------------------------------------------------

/// Forward: K → T. Always positive.
#[inline]
pub fn forward_t(k: f32) -> f32 {
    if k > 0.0 {
        (0.5 * k + 1.0) * k + 1.0
    } else {
        1.0 / ((0.5 * k - 1.0) * k + 1.0)
    }
}

/// Backward: T → K. Clamps T to 1e-5 minimum to avoid singularity.
#[inline]
pub fn backward_t(t: f32) -> f32 {
    let t = t.max(1e-5);
    if t > 1.0 {
        (2.0 * t - 1.0).sqrt() - 1.0
    } else {
        1.0 - (2.0 / t - 1.0).sqrt()
    }
}

/// Gradient chain rule: ∂L/∂K = ∂T/∂K · ∂L/∂T.
#[inline]
pub fn back_propagate_t(k: f32, grad_t: f32) -> f32 {
    if k > 0.0 {
        grad_t * (k + 1.0)
    } else {
        let den = (0.5 * k - 1.0) * k + 1.0;
        grad_t * (1.0 - k) / (den * den)
    }
}

// ---------------------------------------------------------------------------
// Dynamics derivative cache
// ---------------------------------------------------------------------------

/// Cached trajectory derivatives at a sample point, with precomputed powers
/// of t for monomial basis gradient assembly.
pub struct DynDerivatives {
    pub vel: Vector3<f32>,
    pub acc: Vector3<f32>,
    pub jer: Vector3<f32>,
    pub sna: Vector3<f32>,
    /// Cached powers: t, t², t³, t⁴.
    pub s1: f32,
    pub s2: f32,
    pub s3: f32,
    pub s4: f32,
}

/// Fused evaluation of vel/acc/jer/snap at local time `t` given the 6
/// ascending polynomial coefficients for a piece.
///
/// Skips position (unused in dynamics penalties).
#[inline]
pub fn eval_dynamics_derivatives(coeffs: &[Vector3<f32>], t: f32) -> DynDerivatives {
    let c1 = coeffs[1];
    let c2 = coeffs[2];
    let c3 = coeffs[3];
    let c4 = coeffs[4];
    let c5 = coeffs[5];

    let s1 = t;
    let s2 = s1 * s1;
    let s3 = s2 * s1;
    let s4 = s2 * s2;

    let vel = c1 + c2 * (2.0 * s1) + c3 * (3.0 * s2) + c4 * (4.0 * s3) + c5 * (5.0 * s4);
    let acc = c2 * 2.0 + c3 * (6.0 * s1) + c4 * (12.0 * s2) + c5 * (20.0 * s3);
    let jer = c3 * 6.0 + c4 * (24.0 * s1) + c5 * (60.0 * s2);
    let sna = c4 * 24.0 + c5 * (120.0 * s1);

    DynDerivatives { vel, acc, jer, sna, s1, s2, s3, s4 }
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
    fn time_quadratic_round_trip() {
        for t in [1.5, 2.5, 5.0, 100.0] {
            let k = backward_t(t);
            let t2 = forward_t(k);
            assert!((t - t2).abs() < 1e-4, "T={t}: got {t2}");
        }
        for t in [0.01, 0.1, 0.5, 1.0] {
            let k = backward_t(t);
            let t2 = forward_t(k);
            assert!((t - t2).abs() < 1e-5, "T={t}: got {t2}");
        }
    }

    #[test]
    fn time_quadratic_anchor_and_c1() {
        assert!((forward_t(0.0) - 1.0).abs() < 1e-6);
        assert!((back_propagate_t(0.0, 1.0) - 1.0).abs() < 1e-6);
    }
}
