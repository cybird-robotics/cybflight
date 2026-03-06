use core::time::Duration;
use nalgebra as na;
use num_traits::NumCast;

/// Errors returned by [`Mahony`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MahonyError {
    /// The accelerometer norm is at or below the configured minimum, making
    /// normalization unreliable. The orientation estimate is unchanged.
    ZeroAcceleration,
    /// A gain vector contained a negative component.
    NegativeGain,
    /// A norm threshold was zero or negative.
    NonPositiveThreshold,
}

/// Mahony nonlinear complementary filter on SO(3).
///
/// Fuses gyroscope and accelerometer measurements (IMU mode) and, optionally,
/// a magnetometer (MARG mode) to produce a continuous orientation estimate.
/// Gyroscope drift is compensated online via an integral term.
///
/// Orientation is expressed in ENU (East-North-Up): the accelerometer
/// reference direction is `[0, 0, 1]` (specific force points up at rest) and
/// the magnetometer horizontal component is placed in Y (North).
///
/// # Reference
///
/// R. Mahony, T. Hamel and J.-M. Pflimlin, "Nonlinear Complementary Filters
/// on the Special Orthogonal Group," *IEEE Transactions on Automatic Control*,
/// vol. 53, no. 5, pp. 1203–1218, June 2008.
pub struct Mahony<T> {
    kp: na::Vector3<T>,
    ki: na::Vector3<T>,
    orientation: na::UnitQuaternion<T>,
    gyro_bias: na::Vector3<T>,
    min_accel_norm_sq: T,
    min_mag_norm_sq: T,
}

impl<T: na::RealField + NumCast + Copy> Mahony<T> {
    /// Creates a filter with default gains and initialised to the identity orientation.
    pub fn new() -> Self {
        Self::with_gains(
            na::Vector3::from_element(T::one()),
            na::Vector3::from_element(T::from(0.3).unwrap()),
        )
    }

    /// Creates a filter with specified gains and initialised to the identity orientation.
    pub fn with_gains(kp: na::Vector3<T>, ki: na::Vector3<T>) -> Self {
        Self::with_gains_and_initial_orientation(kp, ki, na::UnitQuaternion::identity())
    }

    /// Creates a filter with default gains and a known starting orientation.
    pub fn with_initial_orientation(initial_orientation: na::UnitQuaternion<T>) -> Self {
        Self::with_gains_and_initial_orientation(
            na::Vector3::from_element(T::one()),
            na::Vector3::from_element(T::from(0.3).unwrap()),
            initial_orientation,
        )
    }

    /// Creates a filter with a known starting orientation.
    pub fn with_gains_and_initial_orientation(
        kp: na::Vector3<T>,
        ki: na::Vector3<T>,
        initial_orientation: na::UnitQuaternion<T>,
    ) -> Self {
        debug_assert!(kp.min() >= T::zero(), "kp must be non-negative");
        debug_assert!(ki.min() >= T::zero(), "ki must be non-negative");
        Self {
            kp,
            ki,
            orientation: initial_orientation,
            gyro_bias: na::Vector3::zeros(),
            min_accel_norm_sq: T::from(0.01 * 9.81 * 9.81).unwrap(),
            min_mag_norm_sq: T::from(0.01 * 50.0 * 50.0).unwrap(),
        }
    }

    /// Advances the orientation estimate by one timestep.
    ///
    /// # Parameters
    ///
    /// - `gyro`: angular velocity in the body frame (rad/s).
    /// - `accel`: specific force in the body frame (m/s²). Under quasi-static
    ///   conditions this is approximately opposite to gravity.
    /// - `mag`: optional magnetometer reading in the body frame. When `Some`
    ///   and above [`min_mag_norm`](Self::min_mag_norm), the horizontal
    ///   component corrects heading. Otherwise only accelerometer correction
    ///   is applied.
    /// - `dt`: elapsed time since the previous call.
    ///
    /// # Errors
    ///
    /// Returns [`MahonyError::ZeroAcceleration`] if `‖accel‖ ≤ min_accel_norm`,
    /// leaving the orientation estimate unchanged.
    pub fn update(
        &mut self,
        gyro: na::Vector3<T>,
        accel: na::Vector3<T>,
        mag: Option<na::Vector3<T>>,
        dt: Duration,
    ) -> Result<na::UnitQuaternion<T>, MahonyError> {
        let accel_norm_sq = accel.norm_squared();
        if accel_norm_sq <= self.min_accel_norm_sq {
            return Err(MahonyError::ZeroAcceleration);
        }
        let dt = T::from(dt.as_secs_f64()).unwrap();

        // Normalize accelerometer to get measured specific-force direction.
        let v_a = accel / accel_norm_sq.sqrt();

        // Rotating the ENU reference [0, 0, 1] by q⁻¹ gives the third column
        // of Rᵀ — world z expressed in the body frame.
        let v_0i = na::Vector3::new(T::zero(), T::zero(), T::one());
        let v_hat_a = self.orientation.inverse() * v_0i;

        // Error term: cross product of measured vs estimated direction (Eq. 32c / 48a).
        let mut omega_mes = v_a.cross(&v_hat_a);

        if let Some(mag) = mag {
            let mag_norm_sq = mag.norm_squared();
            if mag_norm_sq > self.min_mag_norm_sq {
                let v_m = mag / mag_norm_sq.sqrt();
                // Project into world frame, flatten inclination to retain only
                // the yaw-relevant horizontal component (ENU: horizontal → Y).
                let h = self.orientation * v_m;
                // ‖[0, ‖h_xy‖, h_z]‖ = ‖h‖ = ‖v_m‖ = 1, so no normalisation needed.
                let v_hat_m =
                    self.orientation.inverse() * na::Vector3::new(T::zero(), h.x.hypot(h.y), h.z);
                omega_mes += v_m.cross(&v_hat_m);
            }
        }

        // Bias integrator: ḃ = −kᵢ · ω_mes (Eq. 48c).
        self.gyro_bias -= self.ki.component_mul(&omega_mes) * dt;

        // Corrected angular rate: ω = gyro − b̂ + kₚ · ω_mes (Eq. 48b).
        let omega_corrected = gyro - self.gyro_bias + self.kp.component_mul(&omega_mes);

        // Integrate orientation on SO(3) via the exponential map.
        let delta_q = na::UnitQuaternion::from_scaled_axis(omega_corrected * dt);
        self.orientation = self.orientation * delta_q;
        Ok(self.orientation)
    }

    /// Returns the current orientation estimate.
    pub fn orientation(&self) -> na::UnitQuaternion<T> {
        self.orientation
    }

    /// Overrides the current orientation estimate.
    ///
    /// Useful for re-initialising the filter from an external attitude source.
    pub fn set_orientation(&mut self, orientation: na::UnitQuaternion<T>) {
        self.orientation = orientation;
    }

    /// Returns the proportional gain vector.
    ///
    /// Each component scales the instantaneous error correction applied to the
    /// corresponding body axis.
    pub fn kp(&self) -> na::Vector3<T> {
        self.kp
    }

    /// Sets the proportional gain vector.
    ///
    /// # Errors
    ///
    /// Returns [`MahonyError::NegativeGain`] if any component is negative.
    pub fn set_kp(&mut self, kp: na::Vector3<T>) -> Result<(), MahonyError> {
        if kp.iter().any(|&v| v < T::zero()) {
            return Err(MahonyError::NegativeGain);
        }
        self.kp = kp;
        Ok(())
    }

    /// Returns the integral gain vector.
    ///
    /// Each component scales the bias drift correction applied to the
    /// corresponding body axis. Setting a component to zero disables bias
    /// estimation on that axis.
    pub fn ki(&self) -> na::Vector3<T> {
        self.ki
    }

    /// Sets the integral gain vector.
    ///
    /// # Errors
    ///
    /// Returns [`MahonyError::NegativeGain`] if any component is negative.
    pub fn set_ki(&mut self, ki: na::Vector3<T>) -> Result<(), MahonyError> {
        if ki.iter().any(|&v| v < T::zero()) {
            return Err(MahonyError::NegativeGain);
        }
        self.ki = ki;
        Ok(())
    }

    /// Returns the current gyroscope bias estimate (rad/s).
    pub fn gyro_bias(&self) -> na::Vector3<T> {
        self.gyro_bias
    }

    /// Overrides the gyroscope bias estimate.
    ///
    /// Can be used to seed the filter with a bias from a prior calibration.
    pub fn set_gyro_bias(&mut self, bias: na::Vector3<T>) {
        self.gyro_bias = bias;
    }

    /// Returns the minimum accelerometer norm threshold (m/s²).
    ///
    /// Updates are rejected when `‖accel‖ ≤` this value.
    pub fn min_accel_norm(&self) -> T {
        self.min_accel_norm_sq.sqrt()
    }

    /// Sets the minimum accelerometer norm threshold (m/s²).
    ///
    /// # Errors
    ///
    /// Returns [`MahonyError::NonPositiveThreshold`] if `norm ≤ 0`.
    pub fn set_min_accel_norm(&mut self, norm: T) -> Result<(), MahonyError> {
        if norm <= T::zero() {
            return Err(MahonyError::NonPositiveThreshold);
        }
        self.min_accel_norm_sq = norm * norm;
        Ok(())
    }

    /// Returns the minimum magnetometer norm threshold (µT).
    ///
    /// The magnetometer correction is skipped when `‖mag‖ ≤` this value,
    /// allowing graceful degradation to IMU-only mode during magnetic
    /// disturbances.
    pub fn min_mag_norm(&self) -> T {
        self.min_mag_norm_sq.sqrt()
    }

    /// Sets the minimum magnetometer norm threshold (µT).
    ///
    /// # Errors
    ///
    /// Returns [`MahonyError::NonPositiveThreshold`] if `norm ≤ 0`.
    pub fn set_min_mag_norm(&mut self, norm: T) -> Result<(), MahonyError> {
        if norm <= T::zero() {
            return Err(MahonyError::NonPositiveThreshold);
        }
        self.min_mag_norm_sq = norm * norm;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const DT: Duration = Duration::from_millis(10);

    fn filter(kp: f64, ki: f64) -> Mahony<f64> {
        Mahony::with_gains(na::Vector3::from_element(kp), na::Vector3::from_element(ki))
    }

    /// Rotation angle (rad) between two orientations.
    ///
    /// Uses `|q₁ · q₂| = cos(θ/2)`, handling the q / −q ambiguity via `.abs()`.
    fn angle_between(a: na::UnitQuaternion<f64>, b: na::UnitQuaternion<f64>) -> f64 {
        let dot = a.coords.dot(&b.coords).abs().min(1.0);
        2.0 * dot.acos()
    }

    // --- Error conditions ---

    #[test]
    fn zero_accel_is_rejected() {
        let mut f = filter(1.0, 0.0);
        assert_eq!(
            f.update(na::Vector3::zeros(), na::Vector3::zeros(), None, DT),
            Err(MahonyError::ZeroAcceleration),
        );
    }

    #[test]
    fn below_threshold_accel_is_rejected() {
        let mut f = filter(1.0, 0.0);
        // Default min_accel_norm ≈ 0.981 m/s²; 0.01 m/s² is well below.
        let tiny = na::Vector3::new(0.01_f64, 0.0, 0.0);
        assert_eq!(
            f.update(na::Vector3::zeros(), tiny, None, DT),
            Err(MahonyError::ZeroAcceleration),
        );
    }

    #[test]
    fn set_min_accel_norm_changes_threshold() {
        let mut f = filter(1.0, 0.0);
        let accel = na::Vector3::new(0.0_f64, 0.0, 5.0);
        // Tighten: reject 5 m/s².
        f.set_min_accel_norm(6.0).unwrap();
        assert_eq!(
            f.update(na::Vector3::zeros(), accel, None, DT),
            Err(MahonyError::ZeroAcceleration),
        );
        // Relax: accept 5 m/s².
        f.set_min_accel_norm(4.0).unwrap();
        assert!(f.update(na::Vector3::zeros(), accel, None, DT).is_ok());
    }

    #[test]
    fn set_kp_negative_is_rejected() {
        let mut f = filter(1.0, 0.0);
        assert_eq!(
            f.set_kp(na::Vector3::new(-0.1_f64, 1.0, 1.0)),
            Err(MahonyError::NegativeGain),
        );
    }

    #[test]
    fn set_ki_negative_is_rejected() {
        let mut f = filter(1.0, 0.3);
        assert_eq!(
            f.set_ki(na::Vector3::new(1.0_f64, 1.0, -0.1)),
            Err(MahonyError::NegativeGain),
        );
    }

    #[test]
    fn set_min_accel_norm_nonpositive_is_rejected() {
        let mut f = filter(1.0, 0.0);
        assert_eq!(
            f.set_min_accel_norm(0.0_f64),
            Err(MahonyError::NonPositiveThreshold)
        );
        assert_eq!(
            f.set_min_accel_norm(-1.0_f64),
            Err(MahonyError::NonPositiveThreshold)
        );
    }

    #[test]
    fn set_min_mag_norm_nonpositive_is_rejected() {
        let mut f = filter(1.0, 0.0);
        assert_eq!(
            f.set_min_mag_norm(0.0_f64),
            Err(MahonyError::NonPositiveThreshold)
        );
        assert_eq!(
            f.set_min_mag_norm(-1.0_f64),
            Err(MahonyError::NonPositiveThreshold)
        );
    }

    // --- Dynamics ---

    #[test]
    fn identity_at_rest_is_unchanged() {
        // With identity orientation and accel = [0,0,9.81], the measured and
        // estimated gravity directions coincide: omega_mes = 0. With zero gyro
        // the orientation must not move.
        let mut f = filter(1.0, 0.3);
        let accel = na::Vector3::new(0.0_f64, 0.0, 9.81);
        f.update(na::Vector3::zeros(), accel, None, DT).unwrap();
        let err = angle_between(f.orientation(), na::UnitQuaternion::identity());
        assert!(err < 1e-12, "angle error = {err}");
    }

    #[test]
    fn pure_gyro_integration_is_exact() {
        // With kp = ki = 0 the filter is a pure SO(3) integrator.
        // 100 × 10 ms at 1 rad/s around Z must give exactly 1 rad around Z.
        // Each step uses from_scaled_axis (exact exponential map), so
        // floating-point accumulation error over 100 steps is O(100ε) ≈ 2e-14.
        let mut f = Mahony::with_gains(na::Vector3::zeros(), na::Vector3::zeros());
        let gyro = na::Vector3::new(0.0_f64, 0.0, 1.0);
        let accel = na::Vector3::new(0.0_f64, 0.0, 9.81);
        for _ in 0..100 {
            f.update(gyro, accel, None, DT).unwrap();
        }
        let expected = na::UnitQuaternion::from_axis_angle(&na::Vector3::z_axis(), 1.0_f64);
        // Compare coordinates directly: acos amplifies machine-epsilon differences
        // to ~sqrt(ε) in angle space. Actual coord error after 100 multiplications
        // is O(100ε) ≈ 2e-14, well within 1e-12.
        let coord_err = (f.orientation().coords - expected.coords).norm();
        assert!(coord_err < 1e-12, "coord error = {coord_err}");
    }

    #[test]
    fn accel_correction_converges_from_tilt() {
        // Start 30° rolled around X (π/6 rad error). Feed the correct ENU
        // gravity [0,0,9.81] with zero gyro input. The P-only corrector drives
        // tan(θ/2) ∝ exp(−kp·t), so after 5 s (kp=1) the residual is
        // tan(π/12)·exp(−5) ≈ 0.18% of the initial angle.
        let q_wrong = na::UnitQuaternion::from_axis_angle(&na::Vector3::x_axis(), PI / 6.0);
        let mut f = Mahony::with_gains_and_initial_orientation(
            na::Vector3::from_element(1.0),
            na::Vector3::zeros(),
            q_wrong,
        );
        let accel = na::Vector3::new(0.0_f64, 0.0, 9.81);
        let initial_err = angle_between(f.orientation(), na::UnitQuaternion::identity());
        for _ in 0..500 {
            f.update(na::Vector3::zeros(), accel, None, DT).unwrap();
        }
        let final_err = angle_between(f.orientation(), na::UnitQuaternion::identity());
        assert!(
            final_err < 0.01 * initial_err,
            "expected <1% of initial error; initial = {initial_err:.4} rad, \
             final = {final_err:.4} rad",
        );
    }

    #[test]
    fn output_quaternion_stays_unit() {
        // from_scaled_axis produces exact unit quaternions, but verifying
        // across 200 steps with non-trivial inputs guards against regression.
        let mut f = filter(1.0, 0.3);
        let gyro = na::Vector3::new(0.1_f64, 0.2, 0.3);
        let accel = na::Vector3::new(0.5_f64, 0.5, 9.8);
        for _ in 0..200 {
            let q = f.update(gyro, accel, None, DT).unwrap();
            let norm = q.coords.norm();
            assert!((norm - 1.0).abs() < 1e-12, "quaternion norm = {norm}");
        }
    }

    #[test]
    fn mag_below_threshold_is_ignored() {
        // A magnetometer reading below min_mag_norm must produce identical
        // results to passing None.
        let gyro = na::Vector3::new(0.1_f64, 0.05, 0.02);
        let accel = na::Vector3::new(0.0_f64, 0.0, 9.81);
        // Default min_mag_norm ≈ 7.07 µT; 0.001 µT is far below.
        let low_mag = Some(na::Vector3::new(0.001_f64, 0.0, 0.0));
        let mut f_none = filter(1.0, 0.3);
        let mut f_low = filter(1.0, 0.3);
        for _ in 0..100 {
            f_none.update(gyro, accel, None, DT).unwrap();
            f_low.update(gyro, accel, low_mag, DT).unwrap();
        }
        // The mag branch is not taken in either case (norm far below threshold),
        // so both filters execute the same code path and must be bit-for-bit identical.
        assert_eq!(
            f_none.orientation().coords,
            f_low.orientation().coords,
            "mag below threshold must not affect the result",
        );
    }
}
