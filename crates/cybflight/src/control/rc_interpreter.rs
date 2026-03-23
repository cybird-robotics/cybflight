//! RC interpreter: maps raw RC channels to a control setpoint.
//!
//! Which setpoint is published depends on the active control mode feature:
//!
//! | Feature         | Publishes to    | Semantics                          |
//! |-----------------|-----------------|------------------------------------|
//! | `est_mahony`   | `MANUAL_CONTROL`| Stick → tilt angle / yaw rate      |
//! | `est_eskf` | `AUTO_SETPOINT` | Stick → ENU position offset [m]    |

use embassy_time::Instant;

use cybflight_msgs as msgs;

use crate::sensors::RC_INPUT;

// ── MANUAL MODE ───────────────────────────────────────────────────────────────

#[cfg(feature = "est_mahony")]
use crate::sensors::MANUAL_CONTROL;

#[cfg(feature = "est_mahony")]
use cybflight_core::rc::rc_mapping::{ChannelSetting, RcMapper, RcSettings};

/// Stick scaling for manual mode.
///
/// Roll/pitch → tilt angle setpoint [rad]; yaw → heading rate [rad/s].
/// When the attitude estimate is not yet valid the attitude controller uses
/// these values as body rates (rate fallback), keeping the craft response
/// gentle during startup.
#[cfg(feature = "est_mahony")]
fn manual_settings() -> RcSettings {
    RcSettings {
        roll: ChannelSetting::new(60.0_f32.to_radians(), 0.0, 0.02),
        pitch: ChannelSetting::new(60.0_f32.to_radians(), 0.0, 0.02),
        yaw: ChannelSetting::new(90.0_f32.to_radians(), 0.0, 0.02),
        throttle: ChannelSetting::default(),
    }
}

#[cfg(feature = "est_mahony")]
#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    let mut sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");
    let pub_ = MANUAL_CONTROL.immediate_publisher();
    let mapper = RcMapper::aetr(manual_settings());

    loop {
        let rc = sub.next_message_pure().await;
        let tr = mapper.map(&rc.channels);
        pub_.publish_immediate(msgs::ManualControlSetpoint {
            timestamp: Instant::now(),
            thrust: tr.thrust,
            roll_rate: tr.roll_rate,
            pitch_rate: tr.pitch_rate,
            yaw_rate: tr.yaw_rate,
        });
        defmt::debug!(
            "RC: ch[0..4]={} {} {} {} {}, setpoint: thrust={} roll={} pitch={} yaw={}",
            rc.channels[0],
            rc.channels[1],
            rc.channels[2],
            rc.channels[3],
            rc.channels[4],
            tr.thrust,
            tr.roll_rate,
            tr.pitch_rate,
            tr.yaw_rate
        );
    }
}

// ── POSITION MODE ────────────────────────────────────────────────────────────

#[cfg(feature = "est_eskf")]
use crate::sensors::VEHICLE_ODOMETRY;
#[cfg(feature = "est_eskf")]
use cybflight_core::rc::rc_mapping::ChannelCalibration;
#[cfg(feature = "est_eskf")]
use nalgebra::{UnitQuaternion, Vector3};

/// Lateral half-range: centered stick ±1 → ±XY_HALF_RANGE m from origin.
#[cfg(feature = "est_eskf")]
const XY_HALF_RANGE: f32 = 0.5;
/// Throttle full range: stick 0–1 → 0–Z_RANGE m above arming altitude.
#[cfg(feature = "est_eskf")]
const Z_RANGE: f32 = 1.0;
/// Dead-band: skip publishing if target moved less than this [m].
#[cfg(feature = "est_eskf")]
const POSITION_THRESHOLD: f32 = 0.01;

#[cfg(feature = "est_eskf")]
fn extract_yaw(q: &UnitQuaternion<f32>) -> f32 {
    let (_roll, _pitch, yaw) = q.euler_angles();
    yaw
}

#[cfg(feature = "est_eskf")]
#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    let mut rc_sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");
    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("rc_interpreter: VEHICLE_ODOMETRY subscriber");

    let pitch_cal = ChannelCalibration::centered(1);
    let roll_cal = ChannelCalibration::centered(0);
    let throttle_cal = ChannelCalibration::throttle(2);

    // Phase 1: Wait for first odometry to establish arming origin.
    let odom = odom_sub.next_message_pure().await;
    let origin = odom.pose.position;
    let yaw = extract_yaw(&odom.pose.orientation);
    let yaw_quat = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);

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

    // Phase 2: Map sticks to ENU position offsets from arming origin.
    // Yaw-independent: pitch → world +X, roll → world +Y.
    let mut last_target = origin;

    loop {
        let rc = rc_sub.next_message_pure().await;

        let dx = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16) * XY_HALF_RANGE;
        let dy = roll_cal.normalize(rc.channels[roll_cal.index] as i16) * XY_HALF_RANGE;
        let dz = throttle_cal.normalize(rc.channels[throttle_cal.index] as i16) * Z_RANGE;

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
                linear: Vector3::zeros(),
                angular: Vector3::zeros(),
            },
        });
    }

    // ── VELOCITY COMMAND stub ────────────────────────────────────────────────
    // (see previous version of this file for the commented-out velocity loop body)
}
