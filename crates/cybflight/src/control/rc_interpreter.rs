//! RC interpreter: maps raw RC channels to a control setpoint.
//!
//! Which setpoint is published depends on the active control mode feature:
//!
//! | Feature            | Publishes to                | Semantics                      |
//! |--------------------|-----------------------------|--------------------------------|
//! | `outer_rate`       | `RATE_COMMAND`              | Stick → body rate + thrust     |
//! | `outer_geometric`  | `ACTIVE_POSITION_SETPOINT`  | Stick → ENU position offset    |
//! | `outer_mpc`        | `ACTIVE_POSITION_SETPOINT`  | Stick → ENU position offset    |

use embassy_time::Instant;

use crate::sensors::RC_INPUT;

// ── RATE MODE ────────────────────────────────────────────────────────────────

#[cfg(feature = "outer_rate")]
use cybflight_core::rc::rc_mapping::ChannelCalibration as RateChannelCalibration;

#[cfg(feature = "outer_rate")]
#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    use crate::vehicle::QUADROTOR_BODY;

    let mut rc_sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");

    let pitch_cal = RateChannelCalibration::centered(1);
    let roll_cal = RateChannelCalibration::centered(0);
    let throttle_cal = RateChannelCalibration::throttle(2);
    let yaw_cal = RateChannelCalibration::centered(3);

    let min_channels: u8 = {
        let mut m = pitch_cal
            .index
            .max(roll_cal.index)
            .max(throttle_cal.index)
            .max(yaw_cal.index);
        m += 1;
        m as u8
    };

    /// Max body rate for roll/pitch [rad/s] (~460 deg/s).
    const MAX_RATE_RP: f32 = 8.0;
    /// Max body rate for yaw [rad/s] (~230 deg/s).
    const MAX_RATE_YAW: f32 = 4.0;
    const RATE_DEADBAND: f32 = 0.05;
    const THROTTLE_DEADBAND: f32 = 0.05;

    const LEARN_TOGGLE_CHANNEL: usize = 6;
    const LEARNER_PREARM_CHANNEL: usize = 7;
    const SWITCH_THRESHOLD: u16 = 1500;

    let hover_thrust_n = QUADROTOR_BODY.mass_kg * 9.81;

    defmt::info!("RC interpreter: rate mode started");

    loop {
        let mut rc = rc_sub.next_message_pure().await;
        while let Some(newer) = rc_sub.try_next_message_pure() {
            rc = newer;
        }

        if rc.channel_count < min_channels {
            continue;
        }

        let learn_on = rc.channel_count > LEARN_TOGGLE_CHANNEL as u8
            && rc.channels[LEARN_TOGGLE_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNING_ENABLED.store(learn_on, core::sync::atomic::Ordering::Release);

        let prearm_on = rc.channel_count > LEARNER_PREARM_CHANNEL as u8
            && rc.channels[LEARNER_PREARM_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNER_PREARM.store(prearm_on, core::sync::atomic::Ordering::Release);

        let roll_norm = roll_cal.normalize(rc.channels[roll_cal.index] as i16);
        let pitch_norm = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16);
        let yaw_norm = yaw_cal.normalize(rc.channels[yaw_cal.index] as i16);
        let throttle_norm = throttle_cal.normalize(rc.channels[throttle_cal.index] as i16);

        let roll_cmd = apply_deadband(roll_norm, RATE_DEADBAND);
        let pitch_cmd = apply_deadband(pitch_norm, RATE_DEADBAND);
        let yaw_cmd = apply_deadband(yaw_norm, RATE_DEADBAND);
        let throttle_cmd = if throttle_norm < THROTTLE_DEADBAND {
            0.0
        } else {
            throttle_norm
        };

        let rate_ref = nalgebra::Vector3::new(
            roll_cmd * MAX_RATE_RP,
            pitch_cmd * MAX_RATE_RP,
            yaw_cmd * MAX_RATE_YAW,
        );
        // Throttle 0→1 maps to 0→2×hover thrust (mid-stick ≈ hover).
        let collective_thrust_n = throttle_cmd * 2.0 * hover_thrust_n;

        super::RATE_COMMAND.signal(cybflight_msgs::AttitudeControlSetpoint {
            timestamp: Instant::now(),
            collective_thrust_n,
            attitude_quaternion: nalgebra::UnitQuaternion::identity(),
            body_rate_rad_s: rate_ref,
            torque_n_m: nalgebra::Vector3::zeros(),
        });
    }
}

#[cfg(feature = "outer_rate")]
#[inline]
fn apply_deadband(s: f32, deadband: f32) -> f32 {
    let a = s.abs();
    if a <= deadband {
        0.0
    } else {
        let scaled = (a - deadband) / (1.0 - deadband);
        let scaled = if scaled > 1.0 { 1.0 } else { scaled };
        if s >= 0.0 { scaled } else { -scaled }
    }
}

// ── POSITION MODE ────────────────────────────────────────────────────────────

#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
use crate::sensors::VEHICLE_ODOMETRY;
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
use cybflight_core::rc::rc_mapping::ChannelCalibration;
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
use nalgebra::Vector3;

/// Lateral half-range: centered stick ±1 → ±XY_HALF_RANGE m from origin.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const XY_HALF_RANGE: f32 = 0.5;
/// Throttle full range: stick 0–1 → 0–Z_RANGE m above arming altitude.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const Z_RANGE: f32 = 1.0;
/// Dead-band: skip publishing if target moved less than this [m].
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const POSITION_THRESHOLD: f32 = 0.01;

#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
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

    // Phase 1: Wait for ESKF convergence, then capture origin from converged odometry.
    while !crate::estimation::ESTIMATOR_READY.load(core::sync::atomic::Ordering::Acquire) {
        embassy_time::Timer::after_millis(100).await;
    }
    let origin = loop {
        let odom = odom_sub.next_message_pure().await;
        let p = odom.pose.position;
        if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
            break p;
        }
        defmt::warn!("rc_interpreter: discarding non-finite odometry during origin capture");
    };

    // Seed the shared setpoint cell and fire the readiness signal. Downstream
    // consumers (cascade_task or outer_loop) wait on ACTIVE_SETPOINT_READY
    // before entering their main loops.
    super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
        cell.set(Some(super::ActiveSetpoint {
            timestamp: Instant::now(),
            position: origin,
            yaw_rad: 0.0,
        }));
    });
    super::ACTIVE_SETPOINT_READY.signal(());

    // Phase 2: Map sticks to ENU position offsets from arming origin.
    // Yaw-independent: pitch → world +X, roll → world +Y.
    let mut last_target = origin;

    /// RC channel index for the learning toggle switch (0-indexed).
    /// Channel 6 (7th channel). >1500 µs = learning data collection active.
    const LEARN_TOGGLE_CHANNEL: usize = 6;
    /// RC channel index for the learner prearm switch (0-indexed).
    /// Channel 7 (8th channel). >1500 µs = learner prearm active.
    const LEARNER_PREARM_CHANNEL: usize = 7;
    const SWITCH_THRESHOLD: u16 = 1500;

    loop {
        let rc = rc_sub.next_message_pure().await;

        // Learning toggle: channel 6 > 1500 → enable RLS data collection
        let learn_on = rc.channel_count > LEARN_TOGGLE_CHANNEL as u8
            && rc.channels[LEARN_TOGGLE_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNING_ENABLED.store(learn_on, core::sync::atomic::Ordering::Release);

        // Learner prearm: channel 7 > 1500 → configure next arm for learning
        let prearm_on = rc.channel_count > LEARNER_PREARM_CHANNEL as u8
            && rc.channels[LEARNER_PREARM_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNER_PREARM.store(prearm_on, core::sync::atomic::Ordering::Release);

        let dx = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16) * XY_HALF_RANGE;
        let dy = roll_cal.normalize(rc.channels[roll_cal.index] as i16) * XY_HALF_RANGE;
        let dz = throttle_cal.normalize(rc.channels[throttle_cal.index] as i16) * Z_RANGE;

        let target = origin + Vector3::new(dx, dy, dz);

        if (target - last_target).norm() < POSITION_THRESHOLD {
            continue;
        }
        last_target = target;

        super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
            cell.set(Some(super::ActiveSetpoint {
                timestamp: Instant::now(),
                position: target,
                yaw_rad: 0.0,
            }));
        });
    }
}
