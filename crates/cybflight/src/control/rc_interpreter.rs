//! RC interpreter: subscribes to raw RC channels and publishes
//! `AUTO_SETPOINT` (position commands) derived from stick inputs.
//!
//! # Design intent
//!
//! This module is a **stand-in for ground-station "goto ENU coordinate"
//! commands**. Sticks select a target position in the inertial (ENU) frame
//! relative to the arming origin; the drone flies there and holds.
//!
//! Stick mapping is **yaw-independent**: pitch moves the target along world +X,
//! roll along world +Y, regardless of which way the drone is pointing. This
//! matches the semantics of a position waypoint, not a velocity command.
//!
//! ## VELOCITY COMMAND stub (end of file)
//! A velocity-integrated alternative loop body is retained as a commented stub
//! for reference. It is not needed under the goto-coordinate model.

use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3};

use cybflight_core::rc::rc_mapping::ChannelCalibration;
use cybflight_msgs as msgs;

use crate::sensors::{RC_INPUT, VEHICLE_ODOMETRY};

// ── POSITION DIAL constants (active) ────────────────────────────────────────

/// Lateral half-range: centered stick ±1 → ±XY_HALF_RANGE m from origin.
const XY_HALF_RANGE: f32 = 0.5;
/// Throttle full range: stick 0–1 → 0–Z_RANGE m above arming altitude.
const Z_RANGE: f32 = 1.0;
/// Dead-band: skip publishing if target moved less than this [m].
const POSITION_THRESHOLD: f32 = 0.01;

/// Extract yaw angle from a unit quaternion (ZYX Euler convention).
fn extract_yaw(q: &UnitQuaternion<f32>) -> f32 {
    let (_roll, _pitch, yaw) = q.euler_angles();
    yaw
}

#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    let mut rc_sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");
    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("rc_interpreter: VEHICLE_ODOMETRY subscriber");

    // Pitch/roll: centered (stick center = zero world-frame offset).
    // Throttle: low = 0, high = +1 (not centered — absolute altitude dial).
    let pitch_cal = ChannelCalibration::centered(1);
    let roll_cal = ChannelCalibration::centered(0);
    let throttle_cal = ChannelCalibration::throttle(2);

    // Phase 1: Wait for first odometry to establish arming origin.
    let odom = odom_sub.next_message_pure().await;
    let origin = odom.pose.position;
    let yaw = extract_yaw(&odom.pose.orientation);
    let yaw_quat = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);

    // Publish initial setpoint at origin.
    super::AUTO_SETPOINT.signal(msgs::VehicleOdometry {
        timestamp: Instant::now(),
        pose: msgs::Pose {
            position: origin,
            orientation: yaw_quat,
        },
        twist: msgs::Twist {
            linear: Vector3::zeros(),
            angular: Vector3::zeros(),
        },
    });

    // ── POSITION DIAL loop (active) ──────────────────────────────────────────

    // Phase 2: Map RC stick inputs to absolute position offsets from origin.
    let mut last_target = origin;

    loop {
        let rc = rc_sub.next_message_pure().await;

        let dx = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16) * XY_HALF_RANGE;
        let dy = roll_cal.normalize(rc.channels[roll_cal.index] as i16) * XY_HALF_RANGE;
        let dz = throttle_cal.normalize(rc.channels[throttle_cal.index] as i16) * Z_RANGE;

        // Sticks map directly to world-frame (ENU) offsets from arming origin.
        // Pitch → +X (East), Roll → +Y (North) — yaw-independent.
        let target = origin + Vector3::new(dx, dy, dz);

        if (target - last_target).norm() < POSITION_THRESHOLD {
            continue;
        }
        last_target = target;

        super::AUTO_SETPOINT.signal(msgs::VehicleOdometry {
            timestamp: Instant::now(),
            pose: msgs::Pose {
                position: target,
                orientation: yaw_quat,
            },
            twist: msgs::Twist {
                linear: Vector3::zeros(), // no feedforward velocity in position-dial mode
                angular: Vector3::zeros(),
            },
        });
    }
}
