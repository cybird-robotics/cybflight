//! Cascade outer-loop task: 100 Hz position→attitude→rate controller.
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
use crate::vehicle::QUADROTOR_BODY;

const ODOM_STALE_TIMEOUT: Duration = Duration::from_millis(100);

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
    let g = &params.control;

    let pc = pd_ff_control::PositionController::new(
        Vector3::new(g.pos_kp[0], g.pos_kp[1], g.pos_kp[2]),
        Vector3::new(g.pos_kd[0], g.pos_kd[1], g.pos_kd[2]),
        position_control::VehicleParams {
            mass: QUADROTOR_BODY.mass_kg,
            gravity: 9.81,
        },
    );
    let ac = geometric_controller::GeometricAttitudeController::new(
        Vector3::new(g.att_k_rate[0], g.att_k_rate[1], g.att_k_rate[2]),
        Vector3::new(1.0, 1.0, 0.2),
    )
    .with_inertia(QUADROTOR_BODY.inertia_matrix());

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

    defmt::info!("Cascade outer loop task started (100 Hz)");

    let mut ticker = Ticker::every(Duration::from_millis(10));
    let mut pub_counter: u32 = 0;

    loop {
        ticker.next().await;
        let now = Instant::now();

        // Drain latest valid odometry.
        let mut got_odom = false;
        while let Some(o) = odom_sub.try_next_message_pure() {
            if odom_is_valid(&o) {
                if o.timestamp <= now && now.duration_since(o.timestamp) <= ODOM_STALE_TIMEOUT {
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

        // Position telemetry (decimated to 10 Hz).
        pub_counter += 1;
        if pub_counter % 10 == 0 {
            pos_pub.publish_immediate(msgs::PositionControlSetpoint {
                timestamp: publish_time,
                position: pos_setpoint.position,
                velocity: pos_setpoint.velocity,
                yaw: pos_setpoint.yaw,
            });
        }
    }
}
