use cybflight_core::vehicle_model::VehicleModel;
use nalgebra as na;

pub struct QuadrotorModel {}

impl VehicleModel for QuadrotorModel {
    type Scalar = f32;

    fn mass(&self) -> f32 {
        1.5
    }

    fn inertia_ixx(&self) -> f32 {
        0.02
    }

    fn inertia_iyy(&self) -> f32 {
        0.02
    }

    fn inertia_izz(&self) -> f32 {
        0.04
    }

    fn front_motor_position(&self) -> na::Vector2<f32> {
        na::Vector2::new(0.1, 0.1)
    }

    fn rear_motor_position(&self) -> na::Vector2<f32> {
        na::Vector2::new(0.1, 0.1)
    }

    fn torque_constant(&self) -> f32 {
        0.01
    }

    fn motor_time_constant_up(&self) -> f32 {
        0.02
    }

    fn max_thrust_per_motor(&self) -> f32 {
        // ~600 g thrust per motor for a 5" prop — placeholder, calibrate from test stand.
        5.9
    }
}

pub static QUADROTOR: QuadrotorModel = QuadrotorModel {};
