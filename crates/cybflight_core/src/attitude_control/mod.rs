use core::default::Default;
use nalgebra as na;

pub struct AttitudeControlState<T> {
    pub attitude_quaternion: na::UnitQuaternion<T>,
    pub body_rate_rad_s: na::Vector3<T>,
}

impl<T: na::RealField> Default for AttitudeControlState<T> {
    fn default() -> Self {
        Self {
            attitude_quaternion: na::UnitQuaternion::identity(),
            body_rate_rad_s: na::Vector3::zeros(),
        }
    }
}

pub struct AttitudeControlSetpoint<T> {
    pub attitude_quaternion: Option<na::UnitQuaternion<T>>,
    pub body_rate_rad_s: na::Vector3<T>,
    pub angular_accel_rad_s2: na::Vector3<T>,
}

impl<T: na::RealField> Default for AttitudeControlSetpoint<T> {
    fn default() -> Self {
        Self {
            attitude_quaternion: None,
            body_rate_rad_s: na::Vector3::zeros(),
            angular_accel_rad_s2: na::Vector3::zeros(),
        }
    }
}

pub struct AttitudeControlOutput<T> {
    pub body_rate_rad_s: na::Vector3<T>,
    pub torque_n_m: na::Vector3<T>,
}

pub mod geometric_controller;
