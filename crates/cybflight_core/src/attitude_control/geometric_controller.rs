use super::{AttitudeControlOutput, AttitudeControlSetpoint, AttitudeControlState};
use core::option::{Option, Option::Some};
use core::{default::Default, marker::Copy};

use crate::rotation::vee;
use nalgebra as na;
use num_traits::{float::FloatCore, NumCast};

pub enum AttitudeErrorLaw {
    GeometricSO3,
    QuaternionBased,
    TiltPrioritizing,
}

fn evaluate_attitude_error<T: na::RealField + Copy + NumCast>(
    q: &na::UnitQuaternion<T>,
    q_des: &na::UnitQuaternion<T>,
    error_law: AttitudeErrorLaw,
) -> na::Vector3<T> {
    let qe = q.inverse() * q_des;
    // Warning! Lee's convention for Geometric tracking Control sets the rate
    // setpoint to the NEGATIVE of the attitude error, while the
    // quaternion-based/tilt prioritizing control methods more often write the
    // rate setpoint equals a certain expression (not clearly labeled as an
    // attitude error). Therefore, the quaternion derived expressions are
    // explicitly negated to match Lee's convention.
    match error_law {
        AttitudeErrorLaw::GeometricSO3 => {
            // Geometric Tracking Control of a Quadrotor UAV on SE(3), Lee, Leok,
            // and McClamroch
            let rot_fb = q.to_rotation_matrix();
            let rot_des = q_des.to_rotation_matrix();

            vee(&((rot_des.transpose() * rot_fb).matrix()
                - (rot_fb.transpose() * rot_des).matrix()))
                * T::from(0.5).unwrap()
        }
        AttitudeErrorLaw::QuaternionBased => {
            // Nonlinear Quadrocopter Attitude Control Technical Report,
            // Brescianini, Hehn, D'Andrea

            // -2.0 is explicitly negated
            -qe.vector() * T::from(2.0).unwrap().copysign(qe.w)
        }

        AttitudeErrorLaw::TiltPrioritizing => {
            // Tilt-Prioritized Quadrocopter Attitude Control, Brescianini, D'Andrea
            let qt_w_sq = qe.w.powi(2) + qe.k.powi(2);
            if qt_w_sq < T::from(1e-6).unwrap() {
                // When w_sq = cos(thrust_error_angle / 2) == 0, thrust_error_angle =
                // 180 degrees, aka thrust_vec_body and thrust_vec_target are directly
                // opposite. Just rotating by att_target_to_body corrects the thrust
                let mut angle_error = -na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                    T::zero(),
                    qe.i,
                    qe.j,
                    T::zero(),
                ))
                .scaled_axis();
                angle_error.z = T::zero();
                angle_error
            } else {
                let qt_w = qt_w_sq.sqrt();
                let i_qt_w = T::one() / qt_w;
                let qe_w_by_qt_w = i_qt_w * qe.w;
                let qe_z_by_qt_w = i_qt_w * qe.k;

                // We deviate from Brescianini's original law:
                //
                // kp_xy * vec(qe) + kp_z *sgn(qe_0) * vec(qe_yaw)
                //
                // By using the full quaternionic log map to extract the rotation
                // error vector, which is, correctly, the magnitude of the angle error
                // about the axis. Instead of taking the vector part and scaling the
                // error by sine of the angle error
                let mut angle_error = -na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                    qt_w,
                    qe_w_by_qt_w * qe.i - qe_z_by_qt_w * qe.j,
                    qe_w_by_qt_w * qe.j + qe_z_by_qt_w * qe.i,
                    T::zero(),
                ))
                .scaled_axis();
                angle_error.z = -T::from(2).unwrap()
                    * (if qe.w < T::zero() {
                        T::atan2(-qe.k, -qe.w)
                    } else {
                        T::atan2(qe.k, qe.w)
                    });
                angle_error
            }
        }
    }
}

pub struct GeometricAttitudeController<T> {
    k_ang_rate: na::Vector3<T>,
    k_ang_torque: na::Vector3<T>,
    attitude_error_law: AttitudeErrorLaw,
    max_body_rate: na::Vector3<T>,
    enable_exact_linearization: bool,
    inertia: Option<na::Matrix3<T>>,
}

impl<T: na::RealField + Copy + FloatCore> Default for GeometricAttitudeController<T> {
    fn default() -> Self {
        Self {
            k_ang_rate: na::Vector3::new(T::one(), T::one(), T::from(0.5).unwrap()),
            k_ang_torque: na::Vector3::new(T::one(), T::one(), T::from(0.2).unwrap()),
            attitude_error_law: AttitudeErrorLaw::TiltPrioritizing,
            max_body_rate: na::Vector3::new(
                T::from(360.0).unwrap().to_radians(),
                T::from(360.0).unwrap().to_radians(),
                T::from(180.0).unwrap().to_radians(),
            ),
            enable_exact_linearization: false,
            inertia: Some(na::Matrix3::identity()),
        }
    }
}

impl<T: na::RealField + Copy + NumCast + FloatCore> GeometricAttitudeController<T> {
    pub fn new(k_ang_rate: na::Vector3<T>, k_ang_torque: na::Vector3<T>) -> Self {
        Self {
            k_ang_rate,
            k_ang_torque,
            attitude_error_law: AttitudeErrorLaw::GeometricSO3,
            max_body_rate: na::Vector3::new(
                T::from(360.0).unwrap().to_radians(),
                T::from(360.0).unwrap().to_radians(),
                T::from(180.0).unwrap().to_radians(),
            ),
            enable_exact_linearization: false,
            inertia: Some(na::Matrix3::identity()),
        }
    }
    pub fn with_attitude_error_law(mut self, law: AttitudeErrorLaw) -> Self {
        self.attitude_error_law = law;
        self
    }

    pub fn with_max_body_rate(mut self, max_body_rate: na::Vector3<T>) -> Self {
        self.max_body_rate = max_body_rate;
        self
    }

    pub fn with_exact_linearization(mut self, enable: bool) -> Self {
        self.enable_exact_linearization = enable;
        self
    }

    pub fn with_inertia(mut self, inertia: na::Matrix3<T>) -> Self {
        self.inertia = Some(inertia);
        self
    }

    pub fn compute(
        &self,
        state: &AttitudeControlState<T>,
        setpoint: &AttitudeControlSetpoint<T>,
    ) -> AttitudeControlOutput<T> {
        let AttitudeControlState {
            attitude_quaternion: q,
            body_rate_rad_s: rate_fb,
        } = state;

        let AttitudeControlSetpoint {
            attitude_quaternion,
            body_rate_rad_s: ref_body_rate,
            angular_accel_rad_s2: ref_angular_accel,
        } = setpoint;

        let rot_fb = q.to_rotation_matrix();
        let (angle_error, out_body_rate, accel_ref_body, rate_ref_body) =
            if let Some(q_des) = attitude_quaternion {
                // Convention check: Lee defines e_R such that positive error drives
                // negative moment e_R = 0.5 * (R_d^T * R - R^T * R_d) Using standard
                // rotation matrix math is often safer than quaternion shortcuts to match
                // the paper exactly.
                let rot_des = q_des.to_rotation_matrix();

                let angle_error = evaluate_attitude_error(q, q_des, AttitudeErrorLaw::GeometricSO3);

                // May be used as a computed 'body_rate_setpoint', e.g., in
                // mavros_controllers. In this case the body_rate field in ref is ignored
                let rate_ref_comp = self
                    .k_ang_rate
                    .component_mul(&angle_error)
                    .zip_map(&self.max_body_rate, |r, max_r| {
                        na::RealField::clamp(r, -max_r, max_r)
                    });

                // Precompute the transformation from desired body frame to current body frame
                let rot_des_to_curr = rot_fb.transpose() * rot_des;

                // Resolve body rate reference and angular acceleration in current body frame.
                let accel_ref_body = rot_des_to_curr * ref_angular_accel;
                let rate_ref_body = rot_des_to_curr * ref_body_rate;
                (
                    Some(angle_error), // Angle error is only well defined when attitude setpoint is provided
                    -rate_ref_comp,    // Feedback attitude control can give a rate setpoint
                    accel_ref_body,
                    rate_ref_body,
                )
            } else {
                (None, *ref_body_rate, *ref_angular_accel, *ref_body_rate)
            };

        let angle_error_feedback = angle_error.map_or(na::Vector3::<T>::zeros(), |e| {
            -self.k_ang_torque.component_mul(&e)
        });

        let out_torque = if let Some(inertia) = self.inertia {
            let out_torque = angle_error_feedback + inertia * accel_ref_body;
            let inertia_by_rate = inertia * rate_fb;
            let gyro_term = rate_fb.cross(&inertia_by_rate);
            // Gyroscopic / Transport Toggle
            out_torque
                + if self.enable_exact_linearization {
                    // Exact Linearization (Lee 2010): cancel natural gyro dynamics
                    // (w x Jw) and transport theorem cross-term J(w x w_ref)
                    gyro_term - inertia_by_rate.cross(&rate_ref_body)
                } else {
                    gyro_term
                }
        } else {
            angle_error_feedback
        };

        AttitudeControlOutput {
            body_rate_rad_s: out_body_rate,
            torque_n_m: out_torque,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Geometric tracking controller (position + attitude → thrust + body rates)
// ───────────────────────────────────────────────────────────────────────────

/// Gains and limits of [`GeometricTrackingController`]. Field names follow
/// `position_controller_params.h` of `uzh-rpg/rpg_quadrotor_control`.
#[derive(Clone, Copy, Debug)]
pub struct GeometricTrackingParams {
    /// Horizontal position gain [1/s²].
    pub kpxy: f32,
    /// Horizontal velocity gain [1/s].
    pub kdxy: f32,
    /// Vertical position gain [1/s²].
    pub kpz: f32,
    /// Vertical velocity gain [1/s].
    pub kdz: f32,
    /// Roll/pitch attitude-error gain [1/s].
    pub krp: f32,
    /// Yaw attitude-error gain [1/s].
    pub kyaw: f32,
    pub pxy_error_max: f32,
    pub vxy_error_max: f32,
    pub pz_error_max: f32,
    pub vz_error_max: f32,
    /// Floor on the mass-normalised collective thrust [m/s²].
    pub min_normalized_thrust: f32,
    /// Gravitational acceleration magnitude [m/s²].
    pub gravity: f32,
}

impl Default for GeometricTrackingParams {
    /// `parameters/default.yaml` of the reference implementation.
    fn default() -> Self {
        Self {
            kpxy: 10.0,
            kdxy: 4.0,
            kpz: 15.0,
            kdz: 6.0,
            krp: 12.0,
            kyaw: 5.0,
            pxy_error_max: 0.6,
            vxy_error_max: 1.0,
            pz_error_max: 0.3,
            vz_error_max: 0.75,
            min_normalized_thrust: 1.0,
            gravity: 9.81,
        }
    }
}

/// Full-state estimate consumed by [`GeometricTrackingController`].
#[derive(Clone, Copy, Debug)]
pub struct GeometricTrackingState {
    pub position: na::Vector3<f32>,
    pub velocity: na::Vector3<f32>,
    /// Body → world.
    pub orientation: na::UnitQuaternion<f32>,
    pub bodyrates: na::Vector3<f32>,
}

/// One flat-output reference sample (world frame).
#[derive(Clone, Copy, Debug)]
pub struct GeometricTrackingReference {
    pub position: na::Vector3<f32>,
    pub velocity: na::Vector3<f32>,
    pub acceleration: na::Vector3<f32>,
    pub jerk: na::Vector3<f32>,
    pub snap: na::Vector3<f32>,
    pub heading: f32,
    pub heading_rate: f32,
    pub heading_acceleration: f32,
}

impl Default for GeometricTrackingReference {
    fn default() -> Self {
        Self {
            position: na::Vector3::zeros(),
            velocity: na::Vector3::zeros(),
            acceleration: na::Vector3::zeros(),
            jerk: na::Vector3::zeros(),
            snap: na::Vector3::zeros(),
            heading: 0.0,
            heading_rate: 0.0,
            heading_acceleration: 0.0,
        }
    }
}

/// Command produced by [`GeometricTrackingController::run`].
#[derive(Clone, Copy, Debug)]
pub struct GeometricTrackingCommand {
    /// Mass-normalised collective thrust [m/s²] along body z.
    pub collective_thrust_per_mass: f32,
    /// Desired attitude (body → world).
    pub orientation: na::UnitQuaternion<f32>,
    /// Body-rate command = reference feedforward + attitude feedback [rad/s].
    pub bodyrates: na::Vector3<f32>,
    /// Reference angular acceleration feedforward [rad/s²].
    pub angular_accelerations: na::Vector3<f32>,
}

/// Geometric tracking controller: PD position control with differential-
/// flatness feedforward and tilt-prioritised attitude feedback, emitting a
/// (collective thrust, body-rate) command for a body-rate inner loop.
///
/// Port of `PositionController` in
/// <https://github.com/uzh-rpg/rpg_quadrotor_control/tree/master/control/position_controller>
/// (`use_rate_mode = true`, no aerodynamic compensation), i.e. the
/// controller of Faessler, Franchi, Scaramuzza, "Differential Flatness of
/// Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of
/// High-Speed Trajectories", RA-L 2018, with the drag terms off:
///
/// ```text
/// a_des = Kp·sat(p_ref − p) + Kd·sat(v_ref − v) + a_ref + g·ẑ
/// c     = max(a_des · z_B, c_min)                          (thrust / mass)
/// R_des = [x_B, y_B, z_B],  z_B = a_des/‖a_des‖, x_B ⊥ y_C (heading frame)
/// ω_ff  from (a_ref, j_ref, s_ref, ψ, ψ̇, ψ̈) by flatness inversion
/// ω_fb  = 2·diag(k_rp, k_rp, k_yaw)·sgn(q_e.w)·vec(q_e),  q_e = q⁻¹ q_des
/// ω_cmd = ω_ff + ω_fb
/// ```
///
/// Frames: world ENU (z up, gravity −g·ẑ), body FLU, quaternion body → world.
#[derive(Clone, Copy, Debug)]
pub struct GeometricTrackingController {
    pub params: GeometricTrackingParams,
}

impl GeometricTrackingController {
    const ALMOST_ZERO: f32 = 0.001;
    const ALMOST_ZERO_THRUST: f32 = 0.01;

    pub fn new(params: GeometricTrackingParams) -> Self {
        Self { params }
    }

    #[inline]
    fn almost_zero(v: f32) -> bool {
        v.abs() < Self::ALMOST_ZERO
    }

    #[inline]
    fn almost_zero_thrust(v: f32) -> bool {
        v.abs() < Self::ALMOST_ZERO_THRUST
    }

    /// One control step (`PositionController::run`, rate mode).
    pub fn run(
        &self,
        state: &GeometricTrackingState,
        reference: &GeometricTrackingReference,
    ) -> GeometricTrackingCommand {
        let g_vec = na::Vector3::new(0.0, 0.0, -self.params.gravity);
        let reference_inputs = self.compute_nominal_reference_inputs(reference, &state.orientation);

        let pid_error_acc = self.compute_pid_error_acc(state, reference);
        // a_des = e_acc + a_ref − g_vec  (g_vec points down, so this adds +g·ẑ)
        let desired_acceleration = pid_error_acc + reference.acceleration - g_vec;

        let collective_thrust_per_mass =
            self.compute_desired_collective_normalized_thrust(&state.orientation, &desired_acceleration);
        let desired_attitude =
            self.compute_desired_attitude(&desired_acceleration, reference.heading, &state.orientation);
        let feedback_bodyrates = self.compute_feedback_bodyrates(&desired_attitude, &state.orientation);

        GeometricTrackingCommand {
            collective_thrust_per_mass,
            orientation: desired_attitude,
            bodyrates: reference_inputs.bodyrates + feedback_bodyrates,
            angular_accelerations: reference_inputs.angular_accelerations,
        }
    }

    /// Feedforward (orientation, thrust, body rates, angular accelerations)
    /// from the flat reference — `computeNominalReferenceInputs`.
    pub fn compute_nominal_reference_inputs(
        &self,
        r: &GeometricTrackingReference,
        attitude_estimate: &na::UnitQuaternion<f32>,
    ) -> GeometricTrackingCommand {
        let q_heading = na::UnitQuaternion::from_axis_angle(&na::Vector3::z_axis(), r.heading);
        let x_c = q_heading * na::Vector3::x();
        let y_c = q_heading * na::Vector3::y();
        let g_vec = na::Vector3::new(0.0, 0.0, -self.params.gravity);
        let des_acc = r.acceleration - g_vec;

        let q_w_b = self.compute_desired_attitude(&des_acc, r.heading, attitude_estimate);
        let x_b = q_w_b * na::Vector3::x();
        let y_b = q_w_b * na::Vector3::y();
        let z_b = q_w_b * na::Vector3::z();

        let thrust = des_acc.norm();
        let mut w = na::Vector3::<f32>::zeros();
        if !Self::almost_zero_thrust(thrust) {
            w.x = -1.0 / thrust * y_b.dot(&r.jerk);
            w.y = 1.0 / thrust * x_b.dot(&r.jerk);
        }
        let yc_x_zb = y_c.cross(&z_b).norm();
        if !Self::almost_zero(yc_x_zb) {
            w.z = 1.0 / yc_x_zb * (r.heading_rate * x_c.dot(&x_b) + w.y * y_c.dot(&z_b));
        }

        let mut wd = na::Vector3::<f32>::zeros();
        if !Self::almost_zero_thrust(thrust) {
            let thrust_dot = z_b.dot(&r.jerk);
            wd.x = -1.0 / thrust * (y_b.dot(&r.snap) + 2.0 * thrust_dot * w.x - thrust * w.y * w.z);
            wd.y = 1.0 / thrust * (x_b.dot(&r.snap) - 2.0 * thrust_dot * w.y - thrust * w.x * w.z);
        }
        if !Self::almost_zero(yc_x_zb) {
            wd.z = 1.0 / yc_x_zb
                * (r.heading_acceleration * x_c.dot(&x_b)
                    + 2.0 * r.heading_rate * w.z * x_c.dot(&y_b)
                    - 2.0 * r.heading_rate * w.y * x_c.dot(&z_b)
                    - w.x * w.y * y_c.dot(&y_b)
                    - w.x * w.z * y_c.dot(&z_b)
                    + wd.y * y_c.dot(&z_b));
        }

        GeometricTrackingCommand {
            collective_thrust_per_mass: thrust,
            orientation: q_w_b,
            bodyrates: w,
            angular_accelerations: wd,
        }
    }

    /// `computePIDErrorAcc`: saturated PD acceleration error in world frame.
    pub fn compute_pid_error_acc(
        &self,
        s: &GeometricTrackingState,
        r: &GeometricTrackingReference,
    ) -> na::Vector3<f32> {
        let p = &self.params;
        let pe = r.position - s.position;
        let ve = r.velocity - s.velocity;
        let ex = pe.x.clamp(-p.pxy_error_max, p.pxy_error_max);
        let ey = pe.y.clamp(-p.pxy_error_max, p.pxy_error_max);
        let ez = pe.z.clamp(-p.pz_error_max, p.pz_error_max);
        let evx = ve.x.clamp(-p.vxy_error_max, p.vxy_error_max);
        let evy = ve.y.clamp(-p.vxy_error_max, p.vxy_error_max);
        let evz = ve.z.clamp(-p.vz_error_max, p.vz_error_max);
        na::Vector3::new(
            p.kpxy * ex + p.kdxy * evx,
            p.kpxy * ey + p.kdxy * evy,
            p.kpz * ez + p.kdz * evz,
        )
    }

    /// `computeDesiredCollectiveMassNormalizedThrust`: project `a_des` on the
    /// *current* body z-axis, floored at `min_normalized_thrust`.
    pub fn compute_desired_collective_normalized_thrust(
        &self,
        attitude_estimate: &na::UnitQuaternion<f32>,
        desired_acc: &na::Vector3<f32>,
    ) -> f32 {
        let body_z = attitude_estimate * na::Vector3::z();
        desired_acc.dot(&body_z).max(self.params.min_normalized_thrust)
    }

    /// `computeDesiredAttitude`: z_B along `a_des`, x_B from the heading
    /// frame with the robust fallback at the 90°-roll singularity.
    pub fn compute_desired_attitude(
        &self,
        desired_acceleration: &na::Vector3<f32>,
        reference_heading: f32,
        attitude_estimate: &na::UnitQuaternion<f32>,
    ) -> na::UnitQuaternion<f32> {
        let q_heading =
            na::UnitQuaternion::from_axis_angle(&na::Vector3::z_axis(), reference_heading);
        let x_c = q_heading * na::Vector3::x();
        let y_c = q_heading * na::Vector3::y();
        let z_b = if Self::almost_zero(desired_acceleration.norm()) {
            // Free fall: keep the estimated thrust direction.
            attitude_estimate * na::Vector3::z()
        } else {
            desired_acceleration.normalize()
        };
        let x_b = Self::compute_robust_body_x_axis(&y_c.cross(&z_b), &x_c, &y_c, attitude_estimate);
        let y_b = z_b.cross(&x_b).normalize();
        let rot = na::Rotation3::from_matrix_unchecked(na::Matrix3::from_columns(&[x_b, y_b, z_b]));
        na::UnitQuaternion::from_rotation_matrix(&rot)
    }

    /// `computeRobustBodyXAxis`.
    fn compute_robust_body_x_axis(
        x_b_prototype: &na::Vector3<f32>,
        x_c: &na::Vector3<f32>,
        y_c: &na::Vector3<f32>,
        attitude_estimate: &na::UnitQuaternion<f32>,
    ) -> na::Vector3<f32> {
        if Self::almost_zero(x_b_prototype.norm()) {
            // y_C ∥ z_B: every x_B lies in the x_C–z_C plane; project the
            // estimated body x-axis into it.
            let x_b_estimated = attitude_estimate * na::Vector3::x();
            let x_b_projected = x_b_estimated - x_b_estimated.dot(y_c) * y_c;
            if Self::almost_zero(x_b_projected.norm()) {
                *x_c
            } else {
                x_b_projected.normalize()
            }
        } else {
            x_b_prototype.normalize()
        }
    }

    /// `computeFeedBackControlBodyrates`: tilt-prioritised quaternion-error
    /// feedback, `ω = 2·K·sgn(q_e.w)·vec(q_e)`.
    pub fn compute_feedback_bodyrates(
        &self,
        desired_attitude: &na::UnitQuaternion<f32>,
        attitude_estimate: &na::UnitQuaternion<f32>,
    ) -> na::Vector3<f32> {
        let q_e = attitude_estimate.inverse() * desired_attitude;
        let s = if q_e.w >= 0.0 { 2.0 } else { -2.0 };
        na::Vector3::new(
            s * self.params.krp * q_e.i,
            s * self.params.krp * q_e.j,
            s * self.params.kyaw * q_e.k,
        )
    }
}

#[cfg(test)]
mod tracking_tests {
    use super::*;

    #[test]
    fn hover_reference_gives_hover_command() {
        let c = GeometricTrackingController::new(GeometricTrackingParams::default());
        let s = GeometricTrackingState {
            position: na::Vector3::new(0.0, 0.0, 1.0),
            velocity: na::Vector3::zeros(),
            orientation: na::UnitQuaternion::identity(),
            bodyrates: na::Vector3::zeros(),
        };
        let r = GeometricTrackingReference {
            position: na::Vector3::new(0.0, 0.0, 1.0),
            ..Default::default()
        };
        let cmd = c.run(&s, &r);
        assert!((cmd.collective_thrust_per_mass - 9.81).abs() < 1e-5);
        assert!(cmd.bodyrates.norm() < 1e-6);
        assert!(cmd.orientation.angle() < 1e-6);
    }

    #[test]
    fn position_error_tilts_toward_target_and_feedback_rate_follows() {
        let c = GeometricTrackingController::new(GeometricTrackingParams::default());
        let s = GeometricTrackingState {
            position: na::Vector3::zeros(),
            velocity: na::Vector3::zeros(),
            orientation: na::UnitQuaternion::identity(),
            bodyrates: na::Vector3::zeros(),
        };
        let r = GeometricTrackingReference {
            position: na::Vector3::new(1.0, 0.0, 0.0),
            ..Default::default()
        };
        let cmd = c.run(&s, &r);
        // a_des = (kpxy·0.6, 0, g): body z tilts toward +x → positive pitch
        // rate (FLU: nose down = +y rotation moves z_B toward +x).
        let z_b = cmd.orientation * na::Vector3::z();
        assert!(z_b.x > 0.3 && z_b.z > 0.5, "z_b = {z_b}");
        assert!(cmd.bodyrates.y > 0.0);
        assert!(cmd.bodyrates.x.abs() < 1e-5);
        // Thrust projects the (larger) desired acceleration onto current z.
        assert!((cmd.collective_thrust_per_mass - 9.81).abs() < 1e-4);
    }

    #[test]
    fn feedforward_rates_match_flatness_for_a_pitching_reference() {
        let c = GeometricTrackingController::new(GeometricTrackingParams::default());
        // Level flight with jerk along +x: z_B rotates toward +x → ω_y = j_x / g.
        let r = GeometricTrackingReference {
            jerk: na::Vector3::new(2.0, 0.0, 0.0),
            ..Default::default()
        };
        let ff = c.compute_nominal_reference_inputs(&r, &na::UnitQuaternion::identity());
        assert!((ff.bodyrates.y - 2.0 / 9.81).abs() < 1e-5, "{}", ff.bodyrates);
        assert!(ff.bodyrates.x.abs() < 1e-6 && ff.bodyrates.z.abs() < 1e-6);
    }
}
