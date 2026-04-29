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

use super::thrust_table::ThrustTable;

/// Grid resolution for `ThrustModel::Table`. Matches the bench-data CSV at
/// `tmp/thrust_map/a2rl_0114.csv` (50×50). Fixed at the type level so a
/// `&'static ThrustTable<TABLE_N>` reference fits in the enum without
/// generics leaking into every caller.
pub const TABLE_N: usize = 50;

/// Thrust-to-command model selector.
///
/// `Quadratic` / `SqrtSquared` are scalar analytic curves parameterized by
/// a single nonlinearity `k`. `Table` is a `(thrust_N, voltage_V) → command`
/// bilinear lookup that compensates for battery sag — see
/// `ThrustTable::lookup`. Switching between any two models typically
/// requires re-identifying the per-motor parameters.
#[derive(Clone, Copy, Debug, Default)]
pub enum ThrustModel {
    /// u = k·d² + (1−k)·d  (default; matches indiflight)
    #[default]
    Quadratic,
    /// u = (k·d + (1−k)·√d)²  (steady-state motor+prop with T ∝ ω²)
    SqrtSquared,
    /// 2D bilinear lookup over a baked thrust map. Carries a static
    /// reference to the table; equality is by pointer identity.
    Table(&'static ThrustTable<TABLE_N>),
}

/// Pre-computed linearization parameters for one motor.
#[derive(Clone, Copy, Debug)]
pub struct ThrustLinearization {
    k: f32,
    model: ThrustModel,
    a: f32,
    b: f32,
    c: f32,
    /// Per-motor maximum thrust (N). Only consumed by `ThrustModel::Table`,
    /// which scales `u ∈ [0,1] → thrust_N` before indexing. Stored
    /// unconditionally to keep `linearize` / `output_curve` branchless on
    /// the analytic path.
    per_motor_max_n: f32,
}

impl ThrustLinearization {
    /// Create linearization parameters from motor nonlinearity factor and
    /// per-motor maximum thrust.
    ///
    /// `nonlinearity` is in [0.0, 1.0]; clamped internally to [0.025, 0.7]
    /// to avoid degenerate curves (div-by-0 at k=0; over-curved at k→1).
    /// `per_motor_max_n` is only used by `ThrustModel::Table`; pass any
    /// finite value (the motor's `max_thrust_n`) for the analytic models.
    pub fn new(nonlinearity: f32, model: ThrustModel, per_motor_max_n: f32) -> Self {
        let k = nonlinearity.clamp(0.025, 0.7);
        let a = 1.0 / k;
        let b = (k * k - 2.0 * k + 1.0) / (4.0 * k * k);
        let c = (k - 1.0) / (2.0 * k);
        Self {
            k,
            model,
            a,
            b,
            c,
            per_motor_max_n,
        }
    }

    /// Force-proportional u → motor command d (linearization / inverse curve).
    ///
    /// For analytic models, `voltage_v` is ignored. For `Table`, `u` is
    /// scaled to per-rotor thrust newtons (`u * per_motor_max_n`) and looked
    /// up at `voltage_v`. All three models return d=0 for u≤0 and d=1 for
    /// u≥1. The Table branch additionally snaps `u·per_motor_max_n ≤
    /// thrust_min_n` to d=0 (sub-idle: the bench data starts at idle thrust,
    /// so anything below it is "motor off") and `u·per_motor_max_n ≥
    /// thrust_max_n` to d=1 (above the table, give full throttle). This
    /// keeps Table behavior aligned with the analytic models so WLS's
    /// "switch a motor off" allocation actually reaches d=0.
    #[inline]
    pub fn linearize(&self, u: f32, voltage_v: f32) -> f32 {
        match self.model {
            ThrustModel::Table(t) => {
                if !u.is_finite() || u <= 0.0 {
                    return 0.0;
                }
                if u >= 1.0 {
                    return 1.0;
                }
                let thrust_n = u * self.per_motor_max_n;
                if thrust_n <= t.thrust_min_n() {
                    return 0.0;
                }
                if thrust_n >= t.thrust_max_n() {
                    return 1.0;
                }
                t.lookup(thrust_n, voltage_v)
            }
            ThrustModel::Quadratic | ThrustModel::SqrtSquared => {
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
                        let x = sqrt(self.a * sqrt(u) + self.b) + self.c;
                        x * x
                    }
                    ThrustModel::Table(_) => u, // unreachable, outer match
                }
            }
        }
    }

    /// Motor command d → force-proportional u (output curve / forward model).
    ///
    /// For analytic models, `voltage_v` is ignored. For `Table`, the
    /// stored map is inverted along the row at `voltage_v` to recover
    /// thrust, then divided by `per_motor_max_n` back into `u ∈ [0,1]`.
    #[inline]
    pub fn output_curve(&self, d: f32, voltage_v: f32) -> f32 {
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
            ThrustModel::Table(t) => {
                // Mirror `linearize`'s boundary semantics: d≤0 → u=0 (motor
                // off), d≥1 → u=1 (saturated). Without the d=0 short-circuit,
                // `invert(0)` returns `thrust_min_n` and we'd estimate a small
                // (~0.005) `u_state` for a fully-off motor, biasing WLS's
                // `du_pref` and `du_min` near zero. Clamp the inverted result
                // to [0,1] so a slight per_motor_max_n / thrust_max_n
                // mismatch can't drive `u_state` outside the valid range.
                if !d.is_finite() || d <= 0.0 {
                    return 0.0;
                }
                if d >= 1.0 {
                    return 1.0;
                }
                if self.per_motor_max_n > 0.0 {
                    (t.invert(d, voltage_v) / self.per_motor_max_n).clamp(0.0, 1.0)
                } else {
                    0.0
                }
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

    /// Per-motor max thrust used for `Table` mode `u ↔ thrust_N` scaling.
    pub fn per_motor_max_n(&self) -> f32 {
        self.per_motor_max_n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Voltage doesn't matter for analytic models, but `linearize` /
    /// `output_curve` still take it. Use a placeholder.
    const V_DC: f32 = 23.0;
    /// Per-motor max thrust; ignored by analytic models.
    const MAX_N: f32 = 10.0;

    #[test]
    fn roundtrip_mid() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic, MAX_N);
        let u = 0.5;
        let d = lin.linearize(u, V_DC);
        let u_back = lin.output_curve(d, V_DC);
        assert!((u - u_back).abs() < 1e-6, "roundtrip: u={u}, d={d}, u_back={u_back}");
    }

    #[test]
    fn roundtrip_sweep() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic, MAX_N);
        for i in 1..100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u, V_DC);
            let u_back = lin.output_curve(d, V_DC);
            assert!(
                (u - u_back).abs() < 1e-5,
                "roundtrip failed at u={u}: d={d}, u_back={u_back}"
            );
        }
    }

    #[test]
    fn boundary_passthrough() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic, MAX_N);
        assert_eq!(lin.linearize(0.0, V_DC), 0.0);
        assert_eq!(lin.linearize(1.0, V_DC), 1.0);
        assert_eq!(lin.output_curve(0.0, V_DC), 0.0);
        // output_curve(1.0) = k + 1 - k = 1.0
        assert!((lin.output_curve(1.0, V_DC) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linearize_monotonic() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic, MAX_N);
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u, V_DC);
            assert!(d >= prev, "non-monotonic at u={u}: d={d} < prev={prev}");
            prev = d;
        }
    }

    #[test]
    fn matches_indiflight_values() {
        // k=0.5: A=2.0, B=0.25, C=-0.5
        let lin = ThrustLinearization::new(0.5, ThrustModel::Quadratic, MAX_N);
        assert!((lin.a - 2.0).abs() < 1e-6);
        assert!((lin.b - 0.25).abs() < 1e-6);
        assert!((lin.c - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn sqrtsq_roundtrip_sweep() {
        let lin = ThrustLinearization::new(0.458, ThrustModel::SqrtSquared, MAX_N);
        for i in 1..100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u, V_DC);
            let u_back = lin.output_curve(d, V_DC);
            assert!(
                (u - u_back).abs() < 1e-5,
                "sqrtsq roundtrip at u={u}: d={d}, u_back={u_back}"
            );
        }
    }

    #[test]
    fn sqrtsq_boundary_passthrough() {
        let lin = ThrustLinearization::new(0.5, ThrustModel::SqrtSquared, MAX_N);
        assert_eq!(lin.linearize(0.0, V_DC), 0.0);
        assert_eq!(lin.linearize(1.0, V_DC), 1.0);
        assert_eq!(lin.output_curve(0.0, V_DC), 0.0);
        assert!((lin.output_curve(1.0, V_DC) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sqrtsq_monotonic() {
        let lin = ThrustLinearization::new(0.458, ThrustModel::SqrtSquared, MAX_N);
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let u = i as f32 / 100.0;
            let d = lin.linearize(u, V_DC);
            assert!(d >= prev, "sqrtsq non-monotonic at u={u}: d={d} < prev={prev}");
            prev = d;
        }
    }

    #[test]
    fn table_routes_through_lookup() {
        // Synthetic linear table: command = thrust / per_motor_max_n,
        // voltage-independent. The enum variant is fixed at TABLE_N, so
        // we build a TABLE_N grid here. Box::leak is fine in a host test.
        const PMAX: f32 = 12.5;
        let mut grid = [[0.0_f32; TABLE_N]; TABLE_N];
        for row in grid.iter_mut() {
            for (col, slot) in row.iter_mut().enumerate() {
                *slot = col as f32 / (TABLE_N as f32 - 1.0);
            }
        }
        let table: &'static ThrustTable<TABLE_N> = Box::leak(Box::new(
            ThrustTable::<TABLE_N>::new(grid, 0.0, PMAX, 20.0, 25.0).unwrap(),
        ));
        let lin = ThrustLinearization::new(0.0, ThrustModel::Table(table), PMAX);
        for u_step in 1..10 {
            let u = u_step as f32 / 10.0;
            let d = lin.linearize(u, 23.0);
            assert!((d - u).abs() < 1e-3, "linear table linearize: u={u} → d={d}");
            let u_back = lin.output_curve(d, 23.0);
            assert!(
                (u_back - u).abs() < 1e-2,
                "linear table roundtrip: u={u} → d={d} → u_back={u_back}"
            );
        }
    }

    #[test]
    fn sqrtsq_low_throttle_shape() {
        // The sqrt-squared model has a gentler low-end linear tangent than
        // the quadratic model. At small d, u ≈ (1-k)²·d for sqrt-squared,
        // but u ≈ (1-k)·d for quadratic. For k=0.5: sqrt-sq ≈ 0.25·d,
        // quadratic ≈ 0.5·d — so sqrt-sq gives less force at the same d.
        let q = ThrustLinearization::new(0.5, ThrustModel::Quadratic, MAX_N);
        let s = ThrustLinearization::new(0.5, ThrustModel::SqrtSquared, MAX_N);
        let d = 0.05;
        assert!(s.output_curve(d, V_DC) < q.output_curve(d, V_DC));
    }
}
