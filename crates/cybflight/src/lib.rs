#![no_std]

pub mod status;
pub mod usb_serial;

use bsp_types::SensorAlign;
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, channel::Channel};
use embassy_time::Instant;
use nalgebra::Vector3;

/// A message carrying an IMU sample tagged with its source index.
pub struct ImuMessage {
    pub source: u8,
    pub sample: ImuSample,
}

/// Static channel for IMU tasks to publish samples for USB streaming.
/// Capacity 8: USB CDC is slower than IMU; oldest samples are dropped via `try_send`.
pub static IMU_CHANNEL: Channel<CriticalSectionRawMutex, ImuMessage, 8> = Channel::new();

/// A timestamped, board-aligned IMU sample ready for downstream processing.
pub struct ImuSample {
    /// Acceleration in m/s^2, rotated into the board reference frame.
    pub accel: Vector3<f32>,
    /// Angular rate in rad/s, rotated into the board reference frame.
    pub gyro: Vector3<f32>,
    /// Die temperature in degrees Celsius.
    pub temp_c: f32,
    /// Timestamp captured immediately after the burst read completes.
    pub timestamp: Instant,
}

/// Rotate a 3-axis sensor vector according to the board-defined alignment.
///
/// Follows the Betaflight `boardalignment.c:alignSensorViaRotation()` convention.
/// TODO: Use cyblib::math for this, match ROS conventions
pub fn apply_alignment(align: SensorAlign, v: Vector3<f32>) -> Vector3<f32> {
    match align {
        SensorAlign::Default | SensorAlign::Cw0Deg => v,
        SensorAlign::Cw90Deg => Vector3::new(v.y, -v.x, v.z),
        SensorAlign::Cw180Deg => Vector3::new(-v.x, -v.y, v.z),
        SensorAlign::Cw270Deg => Vector3::new(-v.y, v.x, v.z),
        SensorAlign::Cw0DegFlip => Vector3::new(-v.x, v.y, -v.z),
        SensorAlign::Cw90DegFlip => Vector3::new(v.y, v.x, -v.z),
        SensorAlign::Cw180DegFlip => Vector3::new(v.x, -v.y, -v.z),
        SensorAlign::Cw270DegFlip => Vector3::new(-v.y, -v.x, -v.z),
    }
}
