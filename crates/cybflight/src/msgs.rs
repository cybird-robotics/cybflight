use embassy_time::Instant;
use nalgebra::{SVector, UnitQuaternion, Vector3};

trait Message: defmt::Format + Clone + Sync + 'static {}

#[derive(Clone, defmt::Format)]
pub struct Imu {
    pub timestamp: Instant,
    pub accel_m_s2: Vector3<f32>,
    pub gyro_rad_s: Vector3<f32>,
    pub temp_c: f32,
}

impl Message for Imu {}

#[derive(Clone, defmt::Format)]
pub struct VehicleAttitude {
    pub timestamp: Instant,
    pub orientation: UnitQuaternion<f32>,
}

impl Message for VehicleAttitude {}

#[derive(Clone, defmt::Format)]
pub struct Pose {
    pub position: Vector3<f32>,
    pub orientation: UnitQuaternion<f32>,
}

impl Message for Pose {}

#[derive(Clone, defmt::Format)]
pub struct Twist {
    pub linear: Vector3<f32>,
    pub angular: Vector3<f32>,
}

impl Message for Twist {}

#[derive(Clone, defmt::Format)]
pub struct VehicleOdometry {
    pub timestamp: Instant,
    pub pose: Pose,
    pub twist: Twist,
}

impl Message for VehicleOdometry {}

const OCP_OUTPUT_SIZE: usize = 4;
#[derive(Clone, defmt::Format)]
pub struct OcpSolverOutput {
    pub timestamp: Instant,
    pub command: SVector<f32, OCP_OUTPUT_SIZE>,
    pub iterations: i32,
    pub converged: bool,
    pub solve_time_us: u64,
}
