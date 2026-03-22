// Thrust linearization for INDI motor commands.
//
// Maps between force-proportional motor command u ∈ [0,1] and the
// DShot/PWM command d ∈ [0,1] that produces that force.
//
// The motor thrust characteristic is modelled as:
//   thrust ∝ output_curve(d) = k·d² + (1−k)·d
//
// where k ∈ [0.025, 0.7] is the nonlinearity factor.
// k=0 → linear, k=1 → pure quadratic (thrust ∝ d²).
//
// The inverse (linearization) converts a desired force-proportional u
// to the d that achieves it:
//   d = linearize(u) = √(A·u + B) + C
//
// Ported from indiflight: src/main/flight/indi.c
// Desmos visualization: https://www.desmos.com/calculator/v9q7cxuffs

/// Pre-computed linearization parameters for one motor.
#[derive(Clone, Copy, Debug)]
pub struct ThrustLinearization {
    k: f32,
    a: f32,
    b: f32,
    c: f32,
}

impl ThrustLinearization {
    /// Create linearization parameters from motor nonlinearity factor.
    ///
    /// `nonlinearity` is in [0.0, 1.0] (e.g. 0.5 = 50% nonlinear).
    /// Clamped internally to [0.025, 0.7] to avoid degenerate curves.
    pub fn new(nonlinearity: f32) -> Self {
        let k = nonlinearity.clamp(0.025, 0.7);
        let a = 1.0 / k;
        let b = (k * k - 2.0 * k + 1.0) / (4.0 * k * k);
        let c = (k - 1.0) / (2.0 * k);
        Self { k, a, b, c }
    }

    /// Force-proportional u → motor command d (linearization / inverse curve).
    ///
    /// For u outside (0, 1), returns u unchanged (passthrough at boundaries).
    /// Returns u unchanged if parameters indicate no linearization configured.
    #[inline]
    pub fn linearize(&self, u: f32) -> f32 {
        if self.a < 1.0 || self.b < 0.0 {
            return u;
        }
        if u <= 0.0 || u >= 1.0 {
            return u;
        }
        num_traits::Float::sqrt(self.a * u + self.b) + self.c
    }

    /// Motor command d → force-proportional u (output curve / forward model).
    ///
    /// Models the motor's thrust characteristic: thrust ∝ k·d² + (1−k)·d.
    #[inline]
    pub fn output_curve(&self, d: f32) -> f32 {
        self.k * d * d + (1.0 - self.k) * d
    }

    /// The nonlinearity factor k.
    pub fn k(&self) -> f32 {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mid() {
        let lin = ThrustLinearization::new(0.5);
        let u = 0.5;
        let d = lin.linearize(u);
        let u_back = lin.output_curve(d);
        assert!((u - u_back).abs() < 1e-6, "roundtrip: u={u}, d={d}, u_back={u_back}");
    }

    #[test]
    fn roundtrip_sweep() {
        let lin = ThrustLinearization::new(0.5);
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
        let lin = ThrustLinearization::new(0.5);
        assert_eq!(lin.linearize(0.0), 0.0);
        assert_eq!(lin.linearize(1.0), 1.0);
        assert_eq!(lin.output_curve(0.0), 0.0);
        // output_curve(1.0) = k + 1 - k = 1.0
        assert!((lin.output_curve(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linearize_monotonic() {
        let lin = ThrustLinearization::new(0.5);
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
        let lin = ThrustLinearization::new(0.5);
        assert!((lin.a - 2.0).abs() < 1e-6);
        assert!((lin.b - 0.25).abs() < 1e-6);
        assert!((lin.c - (-0.5)).abs() < 1e-6);
    }
}
