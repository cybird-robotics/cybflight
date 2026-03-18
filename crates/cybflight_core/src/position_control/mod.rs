use core::default::Default;
use nalgebra as na;

/// Vehicle parameters needed by the position controller.
pub struct VehicleParams<T> {
    /// Vehicle mass [kg].
    pub mass: T,
    /// Gravitational acceleration magnitude [m/s²].
    pub gravity: T,
}

/// Full state feedback for the position controller.
pub struct PositionControlState<T> {
    pub position: na::Vector3<T>,
    pub velocity: na::Vector3<T>,
    pub attitude: na::UnitQuaternion<T>,
}

impl<T: na::RealField> Default for PositionControlState<T> {
    fn default() -> Self {
        Self {
            position: na::Vector3::zeros(),
            velocity: na::Vector3::zeros(),
            attitude: na::UnitQuaternion::identity(),
        }
    }
}

/// Reference setpoint for the position controller.
pub struct PositionControlSetpoint<T> {
    pub position: na::Vector3<T>,
    pub velocity: na::Vector3<T>,
    /// Feedforward acceleration in world frame [m/s²].
    pub acceleration_ff: na::Vector3<T>,
    /// Desired yaw angle [rad].
    pub yaw: T,
}

impl<T: na::RealField> Default for PositionControlSetpoint<T> {
    fn default() -> Self {
        Self {
            position: na::Vector3::zeros(),
            velocity: na::Vector3::zeros(),
            acceleration_ff: na::Vector3::zeros(),
            yaw: T::zero(),
        }
    }
}

/// Output of the position controller — feeds into the attitude inner loop.
pub struct PositionControlOutput<T> {
    /// Desired attitude quaternion (world → body).
    pub desired_attitude: na::UnitQuaternion<T>,
    /// Feedforward body rate [rad/s] (zero for now).
    pub desired_body_rate: na::Vector3<T>,
    /// Scalar collective thrust along body z [N].
    pub collective_thrust: T,
}

pub mod pd_ff_control;
