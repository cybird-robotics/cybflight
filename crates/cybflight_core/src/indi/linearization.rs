// Thrust linearization for INDI motor commands.
//
// Maps between force-proportional motor command u ∈ [0,1] and the
// DShot/PWM command d ∈ [0,1] that produces that force.
//
// Two candidate thrust characteristics are supported, selected by
// `ThrustModel`:
//
//   Quadratic:    u = k·d² + (1−k)·d                 (original indiflight port)
//   SqrtSquared:  u = (k·d + (1−k)·√d)²              (steady-state ω mix, T ∝ ω²)
//
// In both k ∈ [0.025, 0.7]. The Quadratic endpoints are k=0 linear,
// k=1 pure quadratic in d. The SqrtSquared endpoints are k=0 → u=d
// (linear) and k=1 → u=d² (pure quadratic in d); intermediate k blends
// a linear-in-d ω term with a √d-loaded-prop term before squaring.
//
// Both inverses share the same precomputed (a, b, c):
//   a = 1/k,  b = (1−k)² / (4k²),  c = (k−1) / (2k)
//
//   Quadratic:   d = √(a·u + b) + c
//   SqrtSquared: d = (√(a·√u + b) + c)²
//
// Ported from indiflight: src/main/flight/indi.c  (Quadratic)
// Desmos visualization: https://www.desmos.com/calculator/v9q7cxuffs

/// Thrust-to-command model selector.
///
/// Same `k` parameter (clamped to [0.025, 0.7]), different shape. `k` is not
/// directly interchangeable between models — switching models typically
/// requires re-identifying `nonlinearity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ThrustModel {
    /// u = k·d² + (1−k)·d  (default; matches indiflight)
    #[default]
    Quadratic,
    /// u = (k·d + (1−k)·√d)²  (steady-state motor+prop with T ∝ ω²)
    SqrtSquared,
}

/// Pre-computed linearization parameters for one motor.
#[derive(Clone, Copy, Debug)]
pub struct ThrustLinearization {
    k: f32,
    model: ThrustModel,
    a: f32,
    b: f32,
    c: f32,
}

impl ThrustLinearization {
    /// Create linearization parameters from motor nonlinearity factor.
    ///
    /// `nonlinearity` is in [0.0, 1.0]; clamped internally to [0.025, 0.7]
    /// to avoid degenerate curves (div-by-0 at k=0; over-curved at k→1).
    pub fn new(nonlinearity: f32, model: ThrustModel) -> Self {
        let k = nonlinearity.clamp(0.025, 0.7);
        let a = 1.0 / k;
        let b = (k * k - 2.0 * k + 1.0) / (4.0 * k * k);
        let c = (k - 1.0) / (2.0 * k);
        Self { k, model, a, b, c }
    }

    /// Force-proportional u → motor command d (linearization / inverse curve).
    ///
    /// For u outside (0, 1), returns u unchanged (passthrough at boundaries —
    /// both models satisfy d(0)=0 and d(1)=1).
    /// Returns u unchanged if parameters indicate no linearization configured.
    #[inline]
    pub fn linearize(&self, u: f32) -> f32 {
        if self.a < 1.0 || self.b < 0.0 {
            return u;
        }
        if u <= 0.0 || u >= 1.0 {
            return u;
        }
        let sqrt = num_traits::Float::sqrt;
        match self.model {
            ThrustModel::Quadratic => sqrt(self.a * u + self.b) + self.c,
            ThrustModel::SqrtSquared => {
                // Solve u = (k·d + (1−k)·√d)² for d.
                // Substitute x = √d: √u = k·x² + (1−k)·x, then d = x².
                // x shares the same closed form as the Quadratic inverse
                // but with √u in place of u.
                let x = sqrt(self.a * sqrt(u) + self.b) + self.c;
                x * x
            }
        }
    }

    /// Motor command d → force-proportional u (output curve / forward model).
    #[inline]
    pub fn output_curve(&self, d: f32) -> f32 {
        match self.model {
            ThrustModel::Quadratic => self.k * d * d + (1.0 - self.k) * d,
            ThrustModel::SqrtSquared => {
                let sqrt_d = if d > 0.0 {
                    num_traits::Float::sqrt(d)
                } else {
                    0.0
                };
                let s = self.k * d + (1.0 - self.k) * sqrt_d;
                s * s
            }
        }
    }

    /// The nonlinearity factor k.
    pub fn k(&self) -> f32 {
        self.k
    }

    /// The selected thrust model.
    pub fn model(&self) -> ThrustModel {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mid() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic);
        let u = 0.5;
        let d = lin.linearize(u);
        let u_back = lin.output_curve(d);
        assert!((u - u_back).abs() < 1e-6, "roundtrip: u={u}, d={d}, u_back={u_back}");
    }

    #[test]
    fn roundtrip_sweep() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic);
        for i in 1..100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u);
            let u_back = lin.output_curve(d);
            assert!(
                (u - u_back).abs() < 1e-5,
                "roundtrip failed at u={u}: d={d}, u_back={u_back}"
            );
        }
    }

    #[test]
    fn boundary_passthrough() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic);
        assert_eq!(lin.linearize(0.0), 0.0);
        assert_eq!(lin.linearize(1.0), 1.0);
        assert_eq!(lin.output_curve(0.0), 0.0);
        // output_curve(1.0) = k + 1 - k = 1.0
        assert!((lin.output_curve(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linearize_monotonic() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic);
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u);
            assert!(d >= prev, "non-monotonic at u={u}: d={d} < prev={prev}");
            prev = d;
        }
    }

    #[test]
    fn matches_indiflight_values() {
        // k=0.5: A=2.0, B=0.25, C=-0.5
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic);
        assert!((lin.a - 2.0).abs() < 1e-6);
        assert!((lin.b - 0.25).abs() < 1e-6);
        assert!((lin.c - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn sqrtsq_roundtrip_sweep() {
        let lin = ThrustLinearization::new(0.458, ThrustModel::SqrtSquared);
        for i in 1..100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u);
            let u_back = lin.output_curve(d);
            assert!(
                (u - u_back).abs() < 1e-5,
                "sqrtsq roundtrip at u={u}: d={d}, u_back={u_back}"
            );
        }
    }

    #[test]
    fn sqrtsq_boundary_passthrough() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::SqrtSquared);
        assert_eq!(lin.linearize(0.0), 0.0);
        assert_eq!(lin.linearize(1.0), 1.0);
        assert_eq!(lin.output_curve(0.0), 0.0);
        assert!((lin.output_curve(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sqrtsq_monotonic() {
        let lin = ThrustLinearization::new(0.458, ThrustModel::SqrtSquared);
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u);
            assert!(d >= prev, "sqrtsq non-monotonic at u={u}: d={d} < prev={prev}");
            prev = d;
        }
    }

    #[test]
    fn sqrtsq_low_throttle_shape() {
        // The sqrt-squared model has a gentler low-end linear tangent than
        // the quadratic model. At small d, u ≈ (1-k)²·d for sqrt-squared,
        // but u ≈ (1-k)·d for quadratic. For k=0.5: sqrt-sq ≈ 0.25·d,
        // quadratic ≈ 0.5·d — so sqrt-sq gives less force at the same d.
        let q = ThrustLinearization::new(0.5, ThrustModel::Quadratic);
        let s = ThrustLinearization::new(0.5, ThrustModel::SqrtSquared);
        let d = 0.05;
        assert!(s.output_curve(d) < q.output_curve(d));
    }
}
