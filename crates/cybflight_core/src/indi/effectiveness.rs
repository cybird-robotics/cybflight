// INDI effectiveness matrices (G1, G2) in acceleration space.
//
// G1 maps motor commands u ∈ [0,1] to pseudo-controls:
//   ν = G1 · u,  where ν = [fx, fy, fz, α_roll, α_pitch, α_yaw]
//
// Force rows are specific force (N/kg = m/s²).
// Torque rows are angular acceleration (rad/s²) via I⁻¹ · τ.
//
// G2 captures rate-dependent effectiveness (reaction torque from motor
// angular acceleration). Only the torque rows (3..6) are non-zero for
// standard multirotors.
//
// See docs/indi_effectiveness.tex for the full derivation.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use crate::mixer::{MotorParams, RigidBodyParams};

/// INDI effectiveness model for an N-motor vehicle.
///
/// Stores G1 (6×N, acceleration space) and G2 (3×N, angular acceleration
/// per motor angular acceleration), plus per-motor parameters for the
/// G2 scaler computation.
pub struct IndiEffectiveness<const N: usize> {
    /// G1 effectiveness matrix (6×N). Columns are motors; rows are
    /// [fx, fy, fz, roll_accel, pitch_accel, yaw_accel].
    pub g1: SMatrix<f32, 6, N>,

    /// G2 rate-dependent effectiveness (3×N). Rows are
    /// [roll_accel, pitch_accel, yaw_accel] per motor angular acceleration.
    /// For standard quads, only the yaw row is non-zero.
    pub g2: SMatrix<f32, 3, N>,

    /// Pre-computed G2 scaler per motor: ω_max² / (2 · τ_motor).
    pub g2_scaler: SVector<f32, N>,

    /// Maximum motor angular speed (rad/s) per motor.
    pub max_omega: SVector<f32, N>,
}

/// Per-motor INDI parameters not in the base MotorParams.
#[derive(Clone, Copy, Debug)]
pub struct IndiMotorParams {
    /// Motor time constant (seconds). Typical: 0.020–0.030 s.
    pub time_const_s: f32,

    /// Maximum motor RPM.
    pub max_rpm: f32,

    /// G2 yaw effectiveness (from system identification).
    /// Sign follows G1 yaw: positive for CW motors in FLU.
    /// Set to 0.0 if unknown.
    pub g2_yaw: f32,
}

impl<const N: usize> IndiEffectiveness<N> {
    /// Build the INDI effectiveness model from motor geometry + body params.
    ///
    /// Derives the 6×N G1 matrix in acceleration space (FLU frame) from
    /// the physical motor parameters and rigid body inertia. See
    /// `docs/indi_effectiveness.tex` Eq. (10) for the derivation.
    ///
    /// # Panics
    /// Panics if the inertia tensor is singular.
    pub fn new(
        motors: &[MotorParams; N],
        body: &RigidBodyParams,
        indi_params: &[IndiMotorParams; N],
    ) -> Self {
        let inertia_inv = body
            .inertia_matrix()
            .try_inverse()
            .expect("indi: inertia tensor is singular");

        let mut g1 = SMatrix::<f32, 6, N>::zeros();
        let mut g2 = SMatrix::<f32, 3, N>::zeros();
        let mut g2_scaler = SVector::<f32, N>::zeros();
        let mut max_omega = SVector::<f32, N>::zeros();

        for (i, (m, ip)) in motors.iter().zip(indi_params.iter()).enumerate() {
            let [px, py] = m.position_m;
            let t = m.max_thrust_n;
            let spin_sign = m.spin_dir as i32 as f32;

            // Force rows: only fz for standard multirotor (thrust along body +z in FLU)
            g1[(2, i)] = t / body.mass_kg;

            // Torque vector in FLU (matches mixer.rs derivation)
            let torque = Vector3::new(
                py * t,                           // roll torque (N·m)
                -px * t,                          // pitch torque (N·m)
                spin_sign * m.torque_coeff_m * t, // yaw torque (N·m)
            );

            // Angular acceleration = I⁻¹ · τ
            let ang_accel = inertia_inv * torque;
            g1[(3, i)] = ang_accel[0]; // roll accel (rad/s²)
            g1[(4, i)] = ang_accel[1]; // pitch accel (rad/s²)
            g1[(5, i)] = ang_accel[2]; // yaw accel (rad/s²)

            // G2: rate-dependent effectiveness (from system ID)
            // For standard quads, only yaw is non-zero.
            g2[(2, i)] = ip.g2_yaw;

            // G2 scaler: ω_max² / (2 · τ_motor)
            let omega_max = ip.max_rpm / 60.0 * core::f32::consts::TAU;
            max_omega[i] = omega_max;
            g2_scaler[i] = 0.5 * omega_max * omega_max / ip.time_const_s;
        }

        Self {
            g1,
            g2,
            g2_scaler,
            max_omega,
        }
    }

    /// Build the combined G1+G2 effectiveness matrix for the current timestep.
    ///
    /// `omega_fs` is the filtered motor speed (rad/s) per motor.
    /// `g2_valid` indicates per-motor whether G2 should be active (RPM valid).
    ///
    /// Returns the 6×N combined matrix used as B in the WLS problem.
    pub fn combined_g1g2(
        &self,
        omega_fs: &SVector<f32, N>,
        g2_valid: &[bool; N],
    ) -> SMatrix<f32, 6, N> {
        let mut g1g2 = self.g1;

        for i in 0..N {
            if !g2_valid[i] {
                continue;
            }

            // omega_inv with threshold at 10% of max to avoid division by zero
            let inv_thresh = 0.1 * self.max_omega[i];
            let omega_inv = if num_traits::Float::abs(omega_fs[i]) > inv_thresh {
                1.0 / omega_fs[i]
            } else {
                1.0 / inv_thresh
            };

            // Add G2 contribution to torque rows (3, 4, 5)
            for j in 0..3 {
                g1g2[(j + 3, i)] += self.g2_scaler[i] * omega_inv * self.g2[(j, i)];
            }
        }

        g1g2
    }

    /// Replace G1, G2, and motor parameters from learned values.
    ///
    /// Validates all values before applying. Returns `true` if the update was
    /// applied, `false` if validation failed (effectiveness unchanged).
    ///
    /// Validation checks:
    /// - All G1/G2 values must be finite and |value| ≤ 1e4
    /// - max_omega must be in (0, 20000] rad/s
    /// - time_const_s must be in [0.005, 0.5]
    pub fn update_from_learned(
        &mut self,
        g1: &SMatrix<f32, 6, N>,
        g2: &SMatrix<f32, 3, N>,
        max_omega: &SVector<f32, N>,
        time_const_s: &SVector<f32, N>,
    ) -> bool {
        // Magnitude bound: ~30× the largest geometric G1 entry for a typical
        // micro-quad. Anything beyond this is a diverged RLS, not a real vehicle.
        const G_MAG_MAX: f32 = 1e4;
        // Max plausible motor speed: ~191k RPM mechanical.
        const OMEGA_MAX: f32 = 20_000.0;

        let g_valid = |&v: &f32| v.is_finite() && v.abs() <= G_MAG_MAX;
        if !g1.iter().all(g_valid) || !g2.iter().all(g_valid) {
            return false;
        }
        for i in 0..N {
            if !max_omega[i].is_finite() || max_omega[i] <= 0.0 || max_omega[i] > OMEGA_MAX {
                return false;
            }
            if !time_const_s[i].is_finite() || time_const_s[i] < 0.005 || time_const_s[i] > 0.5 {
                return false;
            }
        }

        // Apply
        self.g1 = *g1;
        self.g2 = *g2;
        for i in 0..N {
            self.max_omega[i] = max_omega[i];
            // Recompute G2 scaler: ω_max² / (2 · τ)
            self.g2_scaler[i] = 0.5 * max_omega[i] * max_omega[i] / time_const_s[i];
        }

        true
    }

    /// Inertia-inverse helper: convert physical torque column to acceleration space.
    ///
    /// Useful if you need to add custom effectiveness rows not covered by
    /// `from_motors` (e.g. tilting rotors with lateral force).
    pub fn torque_to_accel(inertia_inv: &Matrix3<f32>, torque: &Vector3<f32>) -> Vector3<f32> {
        inertia_inv * torque
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixer::SpinDir;

    fn test_body() -> RigidBodyParams {
        RigidBodyParams {
            mass_kg: 0.55,
            inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
            max_rate_rad_s: [10.0, 10.0, 6.0],
        }
    }

    fn test_motors() -> [MotorParams; 4] {
        [
            MotorParams {
                position_m: [-0.075, -0.1],
                spin_dir: SpinDir::Cw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
            MotorParams {
                position_m: [0.075, -0.1],
                spin_dir: SpinDir::Ccw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
            MotorParams {
                position_m: [-0.075, 0.1],
                spin_dir: SpinDir::Ccw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
            MotorParams {
                position_m: [0.075, 0.1],
                spin_dir: SpinDir::Cw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
        ]
    }

    fn test_indi_params() -> [IndiMotorParams; 4] {
        [IndiMotorParams {
            time_const_s: 0.025,
            max_rpm: 40000.0,
            g2_yaw: 0.0,
        }; 4]
    }

    #[test]
    fn g1_fz_positive() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params());
        for i in 0..4 {
            assert!(
                eff.g1[(2, i)] > 0.0,
                "fz should be positive (upward) in FLU"
            );
        }
        let expected_fz = 8.5 / 0.55;
        assert!((eff.g1[(2, 0)] - expected_fz).abs() < 1e-4);
    }

    #[test]
    fn g1_roll_signs() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params());
        // Right motors (M0, M1, py < 0): negative roll
        assert!(eff.g1[(3, 0)] < 0.0, "M0 right → negative roll");
        assert!(eff.g1[(3, 1)] < 0.0, "M1 right → negative roll");
        // Left motors (M2, M3, py > 0): positive roll
        assert!(eff.g1[(3, 2)] > 0.0, "M2 left → positive roll");
        assert!(eff.g1[(3, 3)] > 0.0, "M3 left → positive roll");
    }

    #[test]
    fn g1_pitch_signs() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params());
        // Rear motors (M0, M2, px < 0): positive pitch (nose down in FLU)
        assert!(eff.g1[(4, 0)] > 0.0, "M0 rear → positive pitch");
        assert!(eff.g1[(4, 2)] > 0.0, "M2 rear → positive pitch");
        // Front motors (M1, M3, px > 0): negative pitch
        assert!(eff.g1[(4, 1)] < 0.0, "M1 front → negative pitch");
        assert!(eff.g1[(4, 3)] < 0.0, "M3 front → negative pitch");
    }

    #[test]
    fn g1_yaw_signs() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params());
        // CW motors (M0, M3): positive yaw in FLU
        assert!(eff.g1[(5, 0)] > 0.0, "M0 CW → positive yaw in FLU");
        assert!(eff.g1[(5, 3)] > 0.0, "M3 CW → positive yaw in FLU");
        // CCW motors (M1, M2): negative yaw
        assert!(eff.g1[(5, 1)] < 0.0, "M1 CCW → negative yaw in FLU");
        assert!(eff.g1[(5, 2)] < 0.0, "M2 CCW → negative yaw in FLU");
    }

    #[test]
    fn g1_numerical_values() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params());
        // M0: rear-right, CW. From indi_effectiveness.tex Section 6:
        let tol = 0.5; // allow rounding
        assert!((eff.g1[(2, 0)] - 15.45).abs() < tol, "fz M0");
        assert!((eff.g1[(3, 0)] - (-340.0)).abs() < tol, "roll M0");
        assert!((eff.g1[(4, 0)] - 303.6).abs() < tol, "pitch M0");
        assert!((eff.g1[(5, 0)] - 43.5).abs() < tol, "yaw M0");
    }

    #[test]
    fn combined_g1g2_without_rpm() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params());
        let omega_fs = SVector::<f32, 4>::zeros();
        let g2_valid = [false; 4];
        let combined = eff.combined_g1g2(&omega_fs, &g2_valid);
        // With G2 disabled, combined should equal G1
        for i in 0..4 {
            for j in 0..6 {
                assert!((combined[(j, i)] - eff.g1[(j, i)]).abs() < 1e-6);
            }
        }
    }

    fn test_indi_params_with_g2() -> [IndiMotorParams; 4] {
        // CW motors (M0, M3): positive G2 yaw in FLU
        // CCW motors (M1, M2): negative G2 yaw in FLU
        [
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: 0.001,
            },
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: -0.001,
            },
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: -0.001,
            },
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: 0.001,
            },
        ]
    }

    #[test]
    fn g2_stored_correctly() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params_with_g2());
        // G2 yaw row should match input signs
        assert!(eff.g2[(2, 0)] > 0.0, "M0 CW → positive G2 yaw in FLU");
        assert!(eff.g2[(2, 1)] < 0.0, "M1 CCW → negative G2 yaw");
        assert!(eff.g2[(2, 2)] < 0.0, "M2 CCW → negative G2 yaw");
        assert!(eff.g2[(2, 3)] > 0.0, "M3 CW → positive G2 yaw");
        // Roll/pitch G2 should be zero (standard quad)
        for i in 0..4 {
            assert_eq!(eff.g2[(0, i)], 0.0, "G2 roll should be zero");
            assert_eq!(eff.g2[(1, i)], 0.0, "G2 pitch should be zero");
        }
    }

    #[test]
    fn g2_scaler_positive() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params_with_g2());
        // G2 scaler = ω_max² / (2·τ), always positive
        for i in 0..4 {
            assert!(eff.g2_scaler[i] > 0.0, "G2 scaler should be positive");
        }
        // Verify value: ω_max = 40000/60 * 2π ≈ 4188.8, τ = 0.025
        // scaler = 4188.8² / (2 * 0.025) = 17546177 / 0.05 ≈ 350923540
        assert!(eff.g2_scaler[0] > 1e8, "G2 scaler magnitude");
    }

    #[test]
    fn combined_g1g2_with_g2_active() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params_with_g2());
        let hover_omega = 20000.0f32 / 60.0 * core::f32::consts::TAU;
        let omega_fs = SVector::<f32, 4>::from_element(hover_omega);
        let g2_valid = [true; 4];

        let combined = eff.combined_g1g2(&omega_fs, &g2_valid);

        // Force rows (0-2) should be unchanged (G2 only affects torque rows)
        for i in 0..4 {
            for j in 0..3 {
                assert!(
                    (combined[(j, i)] - eff.g1[(j, i)]).abs() < 1e-6,
                    "G2 should not affect force row {j}"
                );
            }
        }

        // Torque rows (3-5) should differ from G1 for yaw (row 5)
        for i in 0..4 {
            let yaw_diff = (combined[(5, i)] - eff.g1[(5, i)]).abs();
            assert!(
                yaw_diff > 1e-6,
                "G2 should modify yaw row for motor {i}: diff={yaw_diff}"
            );

            // G2 contribution sign: same as G2 yaw sign (scaler and omega_inv are positive)
            let g2_contribution = combined[(5, i)] - eff.g1[(5, i)];
            let expected_sign = eff.g2[(2, i)].signum();
            assert_eq!(
                g2_contribution.signum(),
                expected_sign,
                "G2 yaw contribution sign wrong for motor {i}"
            );
        }
    }

    #[test]
    fn combined_g1g2_partial_validity() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params_with_g2());
        let hover_omega = 20000.0f32 / 60.0 * core::f32::consts::TAU;
        let omega_fs = SVector::<f32, 4>::from_element(hover_omega);

        // Only M0 and M1 have valid RPM
        let g2_valid = [true, true, false, false];
        let combined = eff.combined_g1g2(&omega_fs, &g2_valid);

        // M0, M1: yaw should differ from G1
        assert!((combined[(5, 0)] - eff.g1[(5, 0)]).abs() > 1e-6);
        assert!((combined[(5, 1)] - eff.g1[(5, 1)]).abs() > 1e-6);
        // M2, M3: yaw should equal G1 (G2 disabled)
        assert!((combined[(5, 2)] - eff.g1[(5, 2)]).abs() < 1e-6);
        assert!((combined[(5, 3)] - eff.g1[(5, 3)]).abs() < 1e-6);
    }

    #[test]
    fn combined_g1g2_omega_inv_threshold() {
        let eff = IndiEffectiveness::new(&test_motors(), &test_body(), &test_indi_params_with_g2());
        // Very low omega: should clamp to 1/inv_thresh, not divide by zero
        let omega_fs = SVector::<f32, 4>::from_element(1.0); // near zero
        let g2_valid = [true; 4];
        let combined = eff.combined_g1g2(&omega_fs, &g2_valid);
        // Should not panic and all values should be finite
        for i in 0..4 {
            for j in 0..6 {
                assert!(combined[(j, i)].is_finite(), "non-finite at ({j},{i})");
            }
        }
    }
}
