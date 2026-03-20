// Cascaded SE(3) geometric position controller (outer loop).
//
// Implements the position/velocity tracking from Lee 2010:
//   "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
//
// This module computes a desired attitude and collective thrust from position
// and velocity errors. The resulting attitude setpoint feeds the existing
// `GeometricAttitudeController` as the inner loop.
//
// # Coordinate frame
// World frame: ENU / FLU — z = up, gravity = [0, 0, −g].

use super::{PositionControlOutput, PositionControlSetpoint, PositionControlState, VehicleParams};
use core::marker::Copy;
use nalgebra as na;
use num_traits::{NumCast, float::FloatCore};

/// SE(3) geometric position controller (outer loop only).
///
/// Computes desired attitude + collective thrust from position/velocity PD
/// control. The inner attitude loop is handled by `GeometricAttitudeController`.
pub struct PositionController<T> {
    /// Position proportional gain (diagonal).
    kp: na::Vector3<T>,
    /// Velocity derivative gain (diagonal).
    kd: na::Vector3<T>,
    /// Maximum position error per axis [m] before clamping.
    p_err_max: na::Vector3<T>,
    /// Maximum velocity error per axis [m/s] before clamping.
    v_err_max: na::Vector3<T>,
    /// Vehicle parameters.
    vehicle: VehicleParams<T>,
}

impl<T: na::RealField + Copy + FloatCore> Default for PositionController<T> {
    fn default() -> Self {
        Self {
            kp: na::Vector3::new(
                T::from(6.0).unwrap(),
                T::from(6.0).unwrap(),
                T::from(8.0).unwrap(),
            ),
            kd: na::Vector3::new(
                T::from(4.0).unwrap(),
                T::from(4.0).unwrap(),
                T::from(5.0).unwrap(),
            ),
            p_err_max: na::Vector3::new(
                T::from(0.6).unwrap(),
                T::from(0.6).unwrap(),
                T::from(0.3).unwrap(),
            ),
            v_err_max: na::Vector3::new(
                T::from(1.0).unwrap(),
                T::from(1.0).unwrap(),
                T::from(1.0).unwrap(),
            ),
            vehicle: VehicleParams {
                mass: T::from(0.55).unwrap(),
                gravity: T::from(9.81).unwrap(),
            },
        }
    }
}

impl<T: na::RealField + Copy + NumCast + FloatCore> PositionController<T> {
    pub fn new(kp: na::Vector3<T>, kd: na::Vector3<T>, vehicle: VehicleParams<T>) -> Self {
        let default = Self::default();
        Self {
            kp,
            kd,
            p_err_max: default.p_err_max,
            v_err_max: default.v_err_max,
            vehicle,
        }
    }

    pub fn with_kp(mut self, kp: na::Vector3<T>) -> Self {
        self.kp = kp;
        self
    }

    pub fn with_kd(mut self, kd: na::Vector3<T>) -> Self {
        self.kd = kd;
        self
    }

    pub fn with_vehicle(mut self, vehicle: VehicleParams<T>) -> Self {
        self.vehicle = vehicle;
        self
    }

    pub fn with_error_limits(
        mut self,
        p_err_max: na::Vector3<T>,
        v_err_max: na::Vector3<T>,
    ) -> Self {
        self.p_err_max = p_err_max;
        self.v_err_max = v_err_max;
        self
    }

    /// Compute the desired attitude and collective thrust.
    ///
    /// Algorithm (Lee 2010, Section IV):
    /// 1. PD + feedforward: a_des = Kp*(p_ref − p) + Kd*(v_ref − v) + a_ff + g
    /// 2. Desired thrust direction: z_b_des = a_des / |a_des|
    /// 3. Desired rotation from (z_b_des, yaw_ref)
    /// 4. Collective thrust: F = m * (a_des · R * e_z)
    pub fn compute(
        &self,
        state: &PositionControlState<T>,
        setpoint: &PositionControlSetpoint<T>,
    ) -> PositionControlOutput<T> {
        let pos_error = (setpoint.position - state.position)
            .zip_map(&self.p_err_max, |e, m| na::RealField::clamp(e, -m, m));
        let vel_error = (setpoint.velocity - state.velocity)
            .zip_map(&self.v_err_max, |e, m| na::RealField::clamp(e, -m, m));

        // Gravity compensation vector (FLU/ENU: z-up)
        let g_vec = na::Vector3::new(T::zero(), T::zero(), self.vehicle.gravity);

        // Desired acceleration in world frame
        let a_des = self.kp.component_mul(&pos_error)
            + self.kd.component_mul(&vel_error)
            + setpoint.acceleration_ff
            + g_vec;

        let a_des_norm = a_des.norm();

        // Desired body z-axis (thrust direction)
        let z_b_des = if a_des_norm > T::from(1e-6).unwrap() {
            a_des / a_des_norm
        } else {
            na::Vector3::z()
        };

        // Construct desired rotation matrix from z_b_des and yaw reference.
        // x_c = [cos(yaw), sin(yaw), 0] — heading vector projected onto horizontal
        let (sin_yaw, cos_yaw) = setpoint.yaw.sin_cos();
        let x_c = na::Vector3::new(cos_yaw, sin_yaw, T::zero());

        // y_b_des = (z_b_des × x_c) / |z_b_des × x_c|
        let y_b_cross = z_b_des.cross(&x_c);
        let y_b_norm = y_b_cross.norm();
        let y_b_des = if y_b_norm > T::from(1e-6).unwrap() {
            y_b_cross / y_b_norm
        } else {
            // Degenerate case: z_b_des parallel to x_c, pick arbitrary y
            na::Vector3::y()
        };

        let x_b_des = y_b_des.cross(&z_b_des);

        let rot_des = na::Rotation3::from_matrix_unchecked(na::Matrix3::from_columns(&[
            x_b_des, y_b_des, z_b_des,
        ]));
        let desired_attitude_quaternion = na::UnitQuaternion::from_rotation_matrix(&rot_des);

        // Collective thrust: project desired acceleration onto current body z-axis
        let body_z = state.attitude * na::Vector3::z();
        let collective_thrust_n = self.vehicle.mass * a_des.dot(&body_z);

        // Clamp thrust to non-negative (can't push down)
        let collective_thrust_n = if collective_thrust_n < T::zero() {
            T::zero()
        } else {
            collective_thrust_n
        };

        PositionControlOutput {
            desired_attitude_quaternion,
            desired_body_rate_rad_s: na::Vector3::zeros(),
            collective_thrust_n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    fn default_controller() -> PositionController<f32> {
        PositionController::default()
    }

    #[test]
    fn hover_at_origin() {
        let ctrl = default_controller();
        let state = PositionControlState::default();
        let setpoint = PositionControlSetpoint::default();

        let out = ctrl.compute(&state, &setpoint);

        // At hover: desired attitude should be identity (no tilt needed)
        let angle = out.desired_attitude_quaternion.angle();
        assert!(
            angle < 1e-4,
            "expected near-identity attitude, got angle={angle}"
        );

        // Thrust should equal weight: m*g = 0.55 * 9.81 = 5.3955 N
        assert_approx_eq!(f32, out.collective_thrust_n, 0.55 * 9.81, epsilon = 0.01);
    }

    #[test]
    fn position_error_tilts_attitude() {
        let ctrl = default_controller();
        // Drone at origin, target 1m to the right (+y in FLU)
        let state = PositionControlState::default();
        let setpoint = PositionControlSetpoint {
            position: na::Vector3::new(0.0, 1.0, 0.0),
            ..Default::default()
        };

        let out = ctrl.compute(&state, &setpoint);

        // Desired attitude should have some roll (tilt toward +y)
        let angle = out.desired_attitude_quaternion.angle();
        assert!(angle > 0.01, "expected nonzero tilt, got angle={angle}");

        // Thrust should be greater than hover weight due to tilt compensation
        assert!(out.collective_thrust_n > 0.0);
    }

    #[test]
    fn yaw_reference_rotates_heading() {
        let ctrl = default_controller();
        let state = PositionControlState::default();
        let setpoint = PositionControlSetpoint {
            yaw: core::f32::consts::FRAC_PI_2, // 90 degrees
            ..Default::default()
        };

        let out = ctrl.compute(&state, &setpoint);

        // The desired yaw should be ~90 degrees
        let (_, _, yaw) = out.desired_attitude_quaternion.euler_angles();
        assert_approx_eq!(f32, yaw, core::f32::consts::FRAC_PI_2, epsilon = 0.01);
    }
}
