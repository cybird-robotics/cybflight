//! Cascade outer-loop task: position→attitude→rate controller, ticking
//! at `cascade_rate_hz` (default 100 Hz, reboot-flagged).
//!
//! Reads ESKF odometry and the shared position setpoint, runs the PD
//! position controller and geometric attitude controller, and publishes
//! body-rate + collective-thrust commands to `RATE_COMMAND` for the INDI
//! inner loop to consume.
//!
//! Gated on `cfg(feature = "outer_geometric")`.

#![cfg(feature = "outer_geometric")]

use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    position_control::{self, pd_ff_control},
};
use embassy_time::{Duration, Instant, Ticker};
use nalgebra::{UnitQuaternion, Vector3};

use crate::msgs;
use crate::sensors::VEHICLE_ODOMETRY;

/// Fallback odometry-staleness window, used only if the configured
/// `cascade_odom_stale_s` is not usable. The live value comes from the
/// `cascade` group.
const ODOM_STALE_TIMEOUT_FALLBACK: Duration = Duration::from_millis(100);

/// Legal `cascade_rate_hz` range, mirrored from the schema metadata.
/// Floor: the INDI inner loop's 100 ms `CMD_STALE_TIMEOUT` must cover
/// ≥ 2.5 outer periods or a healthy outer loop reads as stale and trips
/// the failsafe. Ceiling: the cascade is cheap (no solver), so it can
/// run much faster than the MPC before straining the executor.
const CASCADE_RATE_HZ_RANGE: (u16, u16) = (25, 500);

fn odom_is_valid(odom: &msgs::VehicleOdometry) -> bool {
    let fin = |v: &Vector3<f32>| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let q = odom.pose.orientation.as_vector();
    fin(&odom.pose.position)
        && q.x.is_finite()
        && q.y.is_finite()
        && q.z.is_finite()
        && q.w.is_finite()
        && fin(&odom.twist.linear)
        && fin(&odom.twist.angular)
}

#[embassy_executor::task]
pub async fn cascade_task() {
    let params = crate::params::get();
    let g = &params.cascade;

    let pc = pd_ff_control::PositionController::new(
        Vector3::new(g.pos_kp[0], g.pos_kp[1], g.pos_kp[2]),
        Vector3::new(g.pos_kd[0], g.pos_kd[1], g.pos_kd[2]),
        position_control::VehicleParams {
            mass: params.airframe.body.mass_kg,
            gravity: params.site.gravity_m_s2,
        },
    )
    // Error clamps from the `cascade` group. They were previously the
    // controller's own `Default` literals, which `new()` copied and
    // nothing could reach — so the bound on every position and velocity
    // error the cascade acts on was invisible to the operator.
    .with_error_limits(
        Vector3::from(g.pos_err_max),
        Vector3::from(g.vel_err_max),
    );
    // Clamp the commanded body rate to the airframe's own declared
    // capability (`max_rate_r/p/y`) rather than the constructor's
    // 360/360/180 deg/s placeholder. Those params are what every other
    // consumer treats as the vehicle's rate envelope, so leaving the
    // cascade on a different, invisible number let it command rates the
    // airframe was never declared able to hold.
    let ac = geometric_controller::GeometricAttitudeController::new(
        Vector3::new(g.att_k_rate[0], g.att_k_rate[1], g.att_k_rate[2]),
        Vector3::from(params.cascade.att_k_torque),
    )
    .with_inertia(params.airframe.body.inertia_matrix())
    .with_max_body_rate(Vector3::from(params.airframe.body.max_rate_rad_s));

    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("cascade: VEHICLE_ODOMETRY subscriber");
    let pos_pub = super::POSITION_CONTROL_SETPOINT.immediate_publisher();
    let tracking_err_pub = super::TRACKING_ERROR.immediate_publisher();
    let ctrl_sp_pub = super::CONTROL_SETPOINT_TELEM
        .publisher()
        .expect("cascade: CONTROL_SETPOINT_TELEM publisher");

    // Wait for ESKF convergence + setpoint seed from rc_interpreter.
    super::ACTIVE_SETPOINT_READY.wait().await;
    while !crate::estimation::ESTIMATOR_READY.load(core::sync::atomic::Ordering::Acquire) {
        embassy_time::Timer::after_millis(100).await;
    }

    let mut pos_state = position_control::PositionControlState::<f32>::default();
    let mut att_state = attitude_control::AttitudeControlState::<f32>::default();
    let mut att_ref = attitude_control::AttitudeControlSetpoint::<f32>::default();

    // Tick rate from `cascade_rate_hz`. Reboot-flagged (the whole cascade
    // group is): read once at task start. Range-checked at every write
    // path, but a programmatic `params::set()` is not — clamp defensively
    // instead of dividing by an out-of-range value.
    let rate_hz = {
        let raw = params.cascade.rate_hz;
        let clamped = raw.clamp(CASCADE_RATE_HZ_RANGE.0, CASCADE_RATE_HZ_RANGE.1);
        if clamped != raw {
            defmt::warn!(
                "cascade: cascade_rate_hz {} out of range — clamped to {}",
                raw,
                clamped
            );
        }
        clamped as u32
    };
    defmt::info!("Cascade outer loop task started ({} Hz)", rate_hz);

    // Odometry-staleness window from `cascade_odom_stale_s`. Same
    // defensive treatment as the tick rate: a non-finite or non-positive
    // value would otherwise become a zero-length window that rejects
    // every sample and starves the loop.
    let odom_stale_timeout = {
        let raw = params.cascade.odom_stale_s;
        if raw.is_finite() && raw > 0.0 {
            Duration::from_micros((raw * 1.0e6) as u64)
        } else {
            defmt::warn!(
                "cascade: cascade_odom_stale_s not usable — falling back to default"
            );
            ODOM_STALE_TIMEOUT_FALLBACK
        }
    };

    let mut ticker = Ticker::every(Duration::from_micros(1_000_000 / rate_hz as u64));
    // Position-setpoint telemetry decimation, ~10 Hz at any tick rate.
    let pos_pub_decim: u32 = (rate_hz / 10).max(1);
    let mut pub_counter: u32 = 0;

    loop {
        ticker.next().await;
        let now = Instant::now();

        // Drain latest valid odometry.
        let mut got_odom = false;
        while let Some(o) = odom_sub.try_next_message_pure() {
            if odom_is_valid(&o) {
                if o.timestamp <= now && now.duration_since(o.timestamp) <= odom_stale_timeout {
                    pos_state.position = o.pose.position;
                    pos_state.velocity = o.twist.linear;
                    pos_state.attitude = o.pose.orientation;
                    att_state.attitude_quaternion = o.pose.orientation;
                    att_state.body_rate_rad_s = o.twist.angular;
                    got_odom = true;
                }
            }
        }

        if !got_odom {
            continue;
        }

        // Read latest position setpoint.
        let pos_setpoint = if let Some(sp) = super::read_active_setpoint() {
            position_control::PositionControlSetpoint {
                position: sp.position,
                velocity: Vector3::zeros(),
                yaw: sp.yaw_rad,
                ..Default::default()
            }
        } else {
            continue;
        };

        // Position controller → desired attitude + thrust.
        let pc_out = pc.compute(&pos_state, &pos_setpoint);
        let collective_thrust_n = pc_out.collective_thrust_n;
        att_ref.attitude_quaternion = Some(pc_out.desired_attitude_quaternion);
        att_ref.body_rate_rad_s = pc_out.desired_body_rate_rad_s;

        // Tracking error against the live position setpoint. Raw
        // (unclamped) so the host can compare against the
        // controller's `p_err_max`/`v_err_max` to detect saturation.
        let pos_err = pos_setpoint.position - pos_state.position;
        let vel_err = pos_setpoint.velocity - pos_state.velocity;
        tracking_err_pub.publish_immediate(super::TrackingError {
            timestamp: now,
            pos_err,
            vel_err,
            attitude_err: Vector3::zeros(),
            body_rate_err: Vector3::zeros(),
            source: super::TRACKING_ERROR_SOURCE_CASCADE,
        });

        // Attitude controller → rate reference.
        let AttitudeControlOutput {
            body_rate_rad_s: rate_ref,
            torque_n_m: _,
        } = ac.compute(&att_state, &att_ref);

        // Publish rate command to INDI.
        let publish_time = Instant::now();
        let setpoint = msgs::AttitudeControlSetpoint {
            timestamp: publish_time,
            collective_thrust_n,
            attitude_quaternion: att_ref
                .attitude_quaternion
                .unwrap_or(UnitQuaternion::identity()),
            body_rate_rad_s: rate_ref,
            torque_n_m: Vector3::zeros(),
        };
        super::RATE_COMMAND.signal(setpoint.clone());
        ctrl_sp_pub.publish_immediate(setpoint);

        // Position telemetry (decimated to ~10 Hz).
        pub_counter += 1;
        if pub_counter % pos_pub_decim == 0 {
            pos_pub.publish_immediate(msgs::PositionControlSetpoint {
                timestamp: publish_time,
                position: pos_setpoint.position,
                velocity: pos_setpoint.velocity,
                yaw: pos_setpoint.yaw,
            });
        }
    }
}
