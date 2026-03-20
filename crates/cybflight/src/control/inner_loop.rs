use cybflight_core::{
    attitude_control::{self, AttitudeControlOutput, geometric_controller},
    mixer::LinearAllocator,
    position_control::{self, pd_ff_control},
};
use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3, Vector4};

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs,
    sensors::VEHICLE_ODOMETRY,
    vehicle::{QUADROTOR_BODY, quadrotor_allocator},
};

/// Extract yaw angle from a unit quaternion (ZYX Euler convention).
fn extract_yaw(q: &UnitQuaternion<f32>) -> f32 {
    let (_roll, _pitch, yaw) = q.euler_angles();
    yaw
}

/// Defense-in-depth: reject odometry with any non-finite state.
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

/// Rate feedback gain for the inner rate loop (P-only).
/// Matches the old `k_rate_torque` from the self-contained geometric controller.
const K_RATE_TORQUE: Vector3<f32> = Vector3::new(0.3, 0.25, 0.15);

pub struct InnerLoop<const N: usize> {
    pc: pd_ff_control::PositionController<f32>,
    ac: geometric_controller::GeometricAttitudeController<f32>,
    allocator: LinearAllocator<N>,
}

impl<const N: usize> InnerLoop<N> {
    pub fn new(allocator: LinearAllocator<N>) -> Self {
        Self {
            pc: pd_ff_control::PositionController::default().with_vehicle(
                position_control::VehicleParams {
                    mass: QUADROTOR_BODY.mass_kg,
                    gravity: 9.81,
                },
            ),
            ac: geometric_controller::GeometricAttitudeController::new(
                Vector3::new(1.0, 1.0, 0.5),
                Vector3::new(1.0, 1.0, 0.2),
            )
            .with_inertia(QUADROTOR_BODY.inertia_matrix()),
            allocator,
        }
    }

    pub async fn run(&mut self) -> ! {
        let mut odom_sub = VEHICLE_ODOMETRY.subscriber().unwrap();
        let publisher = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
        let max_thrust_n = self.allocator.max_collective_thrust_n();

        let mut pos_state = position_control::PositionControlState::<f32>::default();
        let mut att_state = attitude_control::AttitudeControlState::<f32>::default();
        let mut att_ref = attitude_control::AttitudeControlSetpoint::<f32>::default();
        let mut collective_thrust_n: f32 = 0.0;
        let mut last_odom_time: Option<Instant> = None;

        // Wait for first AUTO_SETPOINT before entering the control loop.
        let sp = super::AUTO_SETPOINT.wait().await;
        let mut pos_setpoint = position_control::PositionControlSetpoint {
            position: sp.pose.position,
            velocity: sp.twist.linear,
            yaw: extract_yaw(&sp.pose.orientation),
            ..Default::default()
        };

        const ODOM_STALE_TIMEOUT_MS: u64 = 100;

        loop {
            // 1. Await VEHICLE_ODOMETRY (100 Hz from ESKF) — drives the loop.
            let odom = odom_sub.next_message_pure().await;

            // 2. Staleness + validity guard: skip control when estimate is unreliable.
            let now = Instant::now();
            let stale = match last_odom_time {
                Some(t) => now.duration_since(t).as_millis() > ODOM_STALE_TIMEOUT_MS,
                None => false, // First message — not stale
            };
            last_odom_time = Some(now);

            if stale || !odom_is_valid(&odom) {
                continue;
            }

            att_state.body_rate_rad_s = odom.twist.angular;
            att_state.attitude_quaternion = odom.pose.orientation;

            // 2. Check for new auto-mode setpoint (Signal: consume if available).
            if let Some(sp) = super::AUTO_SETPOINT.try_take() {
                pos_setpoint.position = sp.pose.position;
                pos_setpoint.velocity = sp.twist.linear;
                pos_setpoint.yaw = extract_yaw(&sp.pose.orientation);
            }

            // 3. Position controller (runs every odom tick at 500 Hz).
            pos_state.position = odom.pose.position;
            pos_state.velocity = odom.twist.linear;
            pos_state.attitude = odom.pose.orientation;

            let pc_out = self.pc.compute(&pos_state, &pos_setpoint);
            collective_thrust_n = pc_out.collective_thrust_n;
            att_ref.attitude_quaternion = Some(pc_out.desired_attitude_quaternion);
            att_ref.body_rate_rad_s = pc_out.desired_body_rate_rad_s;

            // 4. Attitude controller: outer (attitude → rate setpoint + feedforward torque).
            let AttitudeControlOutput {
                body_rate_rad_s: rate_ref,
                torque_n_m: ff_torque,
            } = self.ac.compute(&att_state, &att_ref);

            // 5. Rate feedback: P-only rate error → torque.
            let rate_error = att_state.body_rate_rad_s - rate_ref;
            let rate_torque = K_RATE_TORQUE.component_mul(&rate_error);
            let torque_n_m = ff_torque - rate_torque;
            let body_rate_rad_s = rate_ref;

            // 6. Non-finite guard: skip frame if controller produced NaN/Inf.
            if !collective_thrust_n.is_finite()
                || !torque_n_m.x.is_finite()
                || !torque_n_m.y.is_finite()
                || !torque_n_m.z.is_finite()
            {
                defmt::warn!("inner_loop: non-finite controller output, skipping frame");
                continue;
            }

            // 7. Clamp thrust: idle floor for gyroscopic stability, cap at motor capacity.
            let idle_thrust_n = 0.005 * max_thrust_n;
            let effective_thrust_n = collective_thrust_n.max(idle_thrust_n).min(max_thrust_n);

            // 8. Allocate thrust + torque to motor throttles.
            let demand = Vector4::new(effective_thrust_n, torque_n_m.x, torque_n_m.y, torque_n_m.z);
            let throttles = self.allocator.allocate(demand);

            let motor_commands = [
                msgs::NormalizedThrottle::new_saturating(throttles[0]),
                msgs::NormalizedThrottle::new_saturating(throttles[1]),
                msgs::NormalizedThrottle::new_saturating(throttles[2]),
                msgs::NormalizedThrottle::new_saturating(throttles[3]),
            ];
            ACTUATOR_MOTORS.signal(msgs::ActuatorMotors {
                timestamp: Instant::now(),
                motor_commands,
            });

            // 9. Publish telemetry.
            publisher.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: Instant::now(),
                collective_thrust_n: effective_thrust_n,
                attitude_quaternion: att_ref
                    .attitude_quaternion
                    .unwrap_or(UnitQuaternion::identity()),
                body_rate_rad_s,
                torque_n_m,
            });

            // 10. Yield to other tasks.
            embassy_futures::yield_now().await;
        }
    }
}

#[embassy_executor::task]
pub async fn inner_loop_task() {
    let mut driver = InnerLoop::new(quadrotor_allocator());
    driver.run().await;
}
