use crate::rotation::vee;

use super::{AttitudeControlOutput, AttitudeControlSetpoint, AttitudeControlState};
use core::option::{Option, Option::Some};
use core::{default::Default, marker::Copy};

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
    k_rate_torque: na::Vector3<T>,
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
            k_rate_torque: na::Vector3::new(
                T::from(0.4).unwrap(),
                T::from(0.4).unwrap(),
                T::from(0.2).unwrap(),
            ),
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
    pub fn new(
        k_ang_rate: na::Vector3<T>,
        k_ang_torque: na::Vector3<T>,
        k_rate_torque: na::Vector3<T>,
    ) -> Self {
        Self {
            k_ang_rate,
            k_ang_torque,
            k_rate_torque,
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
        let (angle_error, out_body_rate, rate_error, accel_ref_body, rate_ref_body) =
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
                let rate_error = rate_fb - rate_ref_body;
                (
                    Some(angle_error), // Angle error is only well defined when attitude setpoint is provided
                    -rate_ref_comp,    // Feedback attitude control can give a rate setpoint
                    rate_error,
                    accel_ref_body,
                    rate_ref_body,
                )
            } else {
                let rate_error = rate_fb - ref_body_rate;
                (
                    None,
                    *ref_body_rate,
                    rate_error,
                    *ref_angular_accel,
                    *ref_body_rate,
                )
            };

        let angle_error_feedback = angle_error.map_or(na::Vector::zeros(), |e| {
            -self.k_ang_torque.component_mul(&e)
        });

        let out_torque = if let Some(inertia) = self.inertia {
            let out_torque = angle_error_feedback - self.k_rate_torque.component_mul(&rate_error)
                + inertia * accel_ref_body;
            let inertia_by_rate = inertia * rate_fb;
            let gyro_term = rate_fb.cross(&inertia_by_rate);
            // 2. Gyroscopic / Transport Toggle
            out_torque
                + if self.enable_exact_linearization {
                    // Strategy A: Exact Linearization (Lee 2010)
                    // Cancel natural gyro dynamics (w x Jw)
                    // Handle transport theorem cross-term J(w x w_ref)
                    gyro_term - inertia_by_rate.cross(&rate_ref_body)
                } else {
                    gyro_term
                }
        } else {
            angle_error_feedback - self.k_rate_torque.component_mul(&rate_error)
        };

        AttitudeControlOutput {
            body_rate_rad_s: out_body_rate,
            torque_n_m: out_torque,
        }
    }
}
