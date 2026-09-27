#![no_std]

#[cfg(feature = "board_sakurah743")]
pub use bsp_sakurah743 as bsp;
#[cfg(feature = "board_foxeerh743")]
pub use bsp_foxeerh743 as bsp;
#[cfg(feature = "board_micoair743v2")]
pub use bsp_micoair743v2 as bsp;

pub use bsp::hal;

pub mod clocks;
pub mod arm_led;
pub mod blackbox;
pub mod board_init;
pub mod comm;
pub mod serial_logger;
pub mod control;
pub mod estimation;
pub mod health;
pub mod motors;
pub use cybflight_msgs as msgs;
pub mod params;
pub mod platform;
pub mod rates;
pub mod reset_cause;
#[cfg(feature = "postmortem")]
pub mod postmortem;
pub mod sensors;
pub mod shell;
pub mod status;
pub mod thrust_tables;
pub mod usb_serial;
pub mod vehicle;
pub mod watchdog;



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
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubChannel};
use embassy_time::Duration;
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

pub static ARM_DISARM: PubSubChannel<CriticalSectionRawMutex, msgs::ArmDisarm, 4, 4, 1> =
    PubSubChannel::new();

/// Subscribe to a `PubSubChannel`, or park the calling task forever if the
/// channel's compile-time `SUBS` capacity is exhausted.
///
/// The previous `.subscriber().unwrap()` / `.expect()` pattern panics when
/// `SUBS` is hit, which on embassy translates to a fatal executor abort and
/// usually a boot loop. That's the wrong response for an "eyes-only" PR
/// that adds a new reader: one mis-sized const should not lose the FCU.
///
/// This macro logs to defmt **and** to the USB shell, then parks the
/// calling async task on `core::future::pending()`. The rest of the
/// firmware keeps flying — the only function lost is whatever this task
/// was doing — and an operator with the shell open immediately sees
/// which channel is starved and that the SUBS const needs raising.
///
/// Usage (in an async task body only):
/// ```ignore
/// let mut imu_sub = subscribe_or_park!(sensors::IMU_1, "IMU_1");
/// ```
///
/// `loop { pending().await }` has type `!` so it coerces to the
/// subscriber type; the bind site sees a regular subscriber on the happy
/// path and never executes anything past it on the failure path.
#[macro_export]
macro_rules! subscribe_or_park {
    ($channel:expr, $name:literal) => {
        match ($channel).subscriber() {
            Ok(s) => s,
            Err(_) => {
                ::defmt::error!(
                    "{}: subscriber slot exhausted — task parked",
                    $name
                );
                $crate::shell::shell_err(concat!(
                    $name,
                    ": subscriber slot exhausted — task parked"
                ));
                loop {
                    ::core::future::pending::<()>().await;
                }
            }
        }
    };
}

pub trait ConvertToF32Secs {
    fn as_secs_f32(&self) -> f32;
}

impl<T> ConvertToF32Secs for T
where
    T: Into<Duration> + Copy,
{
    fn as_secs_f32(&self) -> f32 {
        let d: Duration = (*self).into();
        d.as_micros() as f32 * 1e-6
    }
}
