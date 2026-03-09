#![no_std]

#[cfg(feature = "board_sakurah743")]
pub use bsp_sakurah743 as bsp;
#[cfg(feature = "board_foxeerh743")]
pub use bsp_foxeerh743 as bsp;

pub use bsp::hal;

pub mod board_init;
pub mod serial_logger;
pub mod control;
pub mod motors;
pub mod msgs;
pub mod platform;
pub mod sensors;
pub mod shell;
pub mod status;
pub mod usb_serial;

/// Firmware version from `Cargo.toml`.
pub const BUILD_VERSION: &str = env!("CARGO_PKG_VERSION");
/// UTC build timestamp set by `build.rs`, e.g. `"2026-03-07 14:30Z"`.
pub const BUILD_TIMESTAMP: &str = match option_env!("BUILD_TIMESTAMP") {
    Some(s) => s,
    None => "unknown",
};
/// Short git commit hash set by `build.rs`, e.g. `"abc1234"`.
pub const GIT_HASH: &str = match option_env!("GIT_HASH") {
    Some(s) => s,
    None => "unknown",
};

use bsp_types::SensorAlign;
use embassy_time::Instant;
use nalgebra::Vector3;

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
