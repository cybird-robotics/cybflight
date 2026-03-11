#![no_std]

pub mod dshot;

use embassy_time::Instant;
use nalgebra::{SVector, UnitQuaternion, Vector3};

pub trait Message: defmt::Format + Clone + Sync + 'static {}

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

#[derive(Clone, defmt::Format)]
pub struct RcInput {
    pub timestamp: Instant,
    /// Channel values in PWM microseconds [988..2012].
    pub channels: [u16; 16],
    /// Number of valid channels.
    pub channel_count: u8,
}

impl Message for RcInput {}

#[derive(Clone, defmt::Format)]
pub struct RcLinkStatus {
    pub timestamp: Instant,
    /// RSSI in dBm (negative value).
    pub rssi_dbm: i16,
    /// Link quality percentage [0..100].
    pub link_quality: u8,
    /// Signal-to-noise ratio in dB.
    pub snr: i8,
    /// RF mode index (protocol-specific).
    pub rf_mode: u8,
}

impl Message for RcLinkStatus {}

/// NMPC position setpoint.  Published by the RC-input-to-setpoint converter
/// (not yet implemented); the NMPC driver falls back to a hardcoded hover
/// target until a publisher exists.
#[derive(Clone, defmt::Format)]
pub struct NmpcSetpoint {
    pub timestamp: Instant,
    /// Target position in world frame [m].
    pub position: Vector3<f32>,
}

impl Message for NmpcSetpoint {}

#[derive(Clone, defmt::Format)]
pub struct DshotMotorTelemetry {
    pub value: dshot::TelemetryValue,
    pub raw: Option<u16>,
}

#[derive(Clone, defmt::Format)]
pub struct DshotTelemetry {
    pub timestamp: Instant,
    pub motors: [DshotMotorTelemetry; 4],
}
impl Message for DshotTelemetry {}

#[derive(Clone, defmt::Format)]
pub struct GpsFix {
    pub timestamp: Instant,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_msl_mm: i32,
    pub ground_speed_mm_s: u32,
    pub heading_mot_1e5: i32,
    pub fix_type: u8,
    pub num_sv: u8,
    pub h_acc_mm: u32,
    pub v_acc_mm: u32,
    pub pdop: u16,
}

impl Message for GpsFix {}

#[derive(Clone, defmt::Format)]
pub struct MagSample {
    pub timestamp: Instant,
    pub field_ut: Vector3<f32>,
    pub temp_c: f32,
}

impl Message for MagSample {}

#[derive(Clone, defmt::Format)]
pub struct BaroSample {
    pub timestamp: Instant,
    pub pressure_pa: f32,
    pub temp_c: f32,
}

impl Message for BaroSample {}

const OCP_OUTPUT_SIZE: usize = 4;
#[derive(Clone, defmt::Format)]
pub struct OcpSolverOutput {
    pub timestamp: Instant,
    pub command: SVector<f32, OCP_OUTPUT_SIZE>,
    pub iterations: i32,
    pub converged: bool,
    pub solve_time_us: u64,
}
impl Message for OcpSolverOutput {}
