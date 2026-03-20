use core::time::Duration;

use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    mixer::LinearAllocator,
    position_control::{self, pd_ff_control},
};
use discrete_pid::{
    pid::{PidConfigBuilder, PidController},
    time::Micros,
};
use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3, Vector4};

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs,
    sensors::VEHICLE_ODOMETRY,
    vehicle::{quadrotor_allocator, QUADROTOR_BODY},
};

/// If no AUTO_SETPOINT update for this long, freeze at current position.
/// Defense-in-depth: catches stale RC data even if failsafe task is delayed.
const RC_STALE_TIMEOUT_MS: u64 = 250;

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

/// Nominal odometry loop period (~100 Hz from ESKF).
const RATE_PID_SAMPLE_TIME: Duration = Duration::from_millis(2);

struct Pids {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
}

struct RatePids {
    roll: PidController<Micros, f32>,
    pitch: PidController<Micros, f32>,
    yaw: PidController<Micros, f32>,
}

impl RatePids {
    fn new(k_roll: Pids, k_pitch: Pids, k_yaw: Pids) -> Self {
        let make = |k: Pids| {
            let Pids { kp, ki, kd } = k;
            let config = PidConfigBuilder::<f32>::default()
                .kp(kp)
                .ki(ki)
                .kd(kd)
                .sample_time(RATE_PID_SAMPLE_TIME)
                .build()
                .expect("rate PID config invalid");
            PidController::new_uninit(config)
        };
        Self {
            roll: make(k_roll),
            pitch: make(k_pitch),
            yaw: make(k_yaw),
        }
    }

    fn compute(&mut self, rate_fb: Vector3<f32>, rate_ref: Vector3<f32>) -> Vector3<f32> {
        let ts = Micros(Instant::now().as_micros());
        Vector3::new(
            self.roll.compute(rate_fb.x, rate_ref.x, ts, None),
            self.pitch.compute(rate_fb.y, rate_ref.y, ts, None),
            self.yaw.compute(rate_fb.z, rate_ref.z, ts, None),
        )
    }
}

pub struct InnerLoop<const N: usize> {
    pc: pd_ff_control::PositionController<f32>,
    ac: geometric_controller::GeometricAttitudeController<f32>,
    rate_pids: RatePids,
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
                Vector3::new(6.0, 6.0, 1.5), // k_ang_rate  [roll, pitch, yaw]
                Vector3::new(1.0, 1.0, 0.2), // k_ang_torque (unused — discarded below)
            )
            .with_inertia(QUADROTOR_BODY.inertia_matrix()),
            rate_pids: RatePids::new(
                Pids {
                    kp: 0.3,
                    ki: 0.0,
                    kd: 0.0,
                },
                Pids {
                    kp: 0.25,
                    ki: 0.0,
                    kd: 0.0,
                },
                Pids {
                    kp: 0.15,
                    ki: 0.00,
                    kd: 0.0,
                },
            ),
            allocator,
        }
    }

    pub async fn run(&mut self) -> ! {
        let mut odom_sub = VEHICLE_ODOMETRY.subscriber().unwrap();
        let att_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
        let pos_pub = super::POSITION_CONTROL_SETPOINT.immediate_publisher();
        let max_thrust_n = self.allocator.max_collective_thrust_n();

        let mut pos_state = position_control::PositionControlState::<f32>::default();
        let mut att_state = attitude_control::AttitudeControlState::<f32>::default();
        let mut att_ref = attitude_control::AttitudeControlSetpoint::<f32>::default();
        let mut last_odom_time: Option<Instant> = None;

        // Wait for first AUTO_SETPOINT before entering the control loop.
        let sp = super::AUTO_SETPOINT.wait().await;
        let mut pos_setpoint = position_control::PositionControlSetpoint {
            position: sp.pose.position,
            velocity: sp.twist.linear,
            yaw: extract_yaw(&sp.pose.orientation),
            ..Default::default()
        };
        let mut last_setpoint_time = Instant::now();

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

            // 3. Check for new auto-mode setpoint (Signal: consume if available).
            if let Some(sp) = super::AUTO_SETPOINT.try_take() {
                pos_setpoint.position = sp.pose.position;
                pos_setpoint.velocity = sp.twist.linear;
                pos_setpoint.yaw = extract_yaw(&sp.pose.orientation);
                last_setpoint_time = now;
            }

            // RC stale guard: skip control entirely when setpoint is stale.
            // Failsafe controller_watchdog_task will disarm if this persists.
            if now.duration_since(last_setpoint_time).as_millis() > RC_STALE_TIMEOUT_MS {
                continue;
            }

            // 4. Position controller (runs every odom tick).
            pos_state.position = odom.pose.position;
            pos_state.velocity = odom.twist.linear;
            pos_state.attitude = odom.pose.orientation;

            let pc_out = self.pc.compute(&pos_state, &pos_setpoint);
            let collective_thrust_n = pc_out.collective_thrust_n;
            att_ref.attitude_quaternion = Some(pc_out.desired_attitude_quaternion);
            att_ref.body_rate_rad_s = pc_out.desired_body_rate_rad_s;

            // 4. Attitude controller: outer (attitude error → rate setpoint).
            let AttitudeControlOutput {
                body_rate_rad_s: rate_ref,
                torque_n_m: _,
            } = self.ac.compute(&att_state, &att_ref);

            const MAX_ROLL_TORQUE_N_M: f32 = 0.8;
            const MAX_PITCH_TORQUE_N_M: f32 = 0.6;
            const MAX_YAW_TORQUE_N_M: f32 = 0.15;

            // 5. Rate feedback: PID rate error → torque.
            let rate_torque = self.rate_pids.compute(att_state.body_rate_rad_s, rate_ref);
            let torque_n_m = Vector3::new(
                rate_torque[0].clamp(-MAX_ROLL_TORQUE_N_M, MAX_ROLL_TORQUE_N_M),
                rate_torque[1].clamp(-MAX_PITCH_TORQUE_N_M, MAX_PITCH_TORQUE_N_M),
                rate_torque[2].clamp(-MAX_YAW_TORQUE_N_M, MAX_YAW_TORQUE_N_M),
            );
            let body_rate_rad_s = rate_ref;

            // 7. Non-finite guard: skip frame if controller produced NaN/Inf.
            if !collective_thrust_n.is_finite()
                || !torque_n_m.x.is_finite()
                || !torque_n_m.y.is_finite()
                || !torque_n_m.z.is_finite()
            {
                defmt::warn!("inner_loop: non-finite controller output, skipping frame");
                continue;
            }

            // 8. Clamp thrust: idle floor for gyroscopic stability, cap at motor capacity.
            let idle_thrust_n = 0.005 * max_thrust_n;
            let effective_thrust_n = collective_thrust_n.max(idle_thrust_n).min(max_thrust_n);

            // 9. Allocate thrust + torque to motor throttles.
            let demand = Vector4::new(effective_thrust_n, torque_n_m.x, torque_n_m.y, torque_n_m.z);
            let throttles = self.allocator.allocate(demand);

            let motor_commands = [
                msgs::NormalizedThrottle::new_saturating(throttles[0]),
                msgs::NormalizedThrottle::new_saturating(throttles[1]),
                msgs::NormalizedThrottle::new_saturating(throttles[2]),
                msgs::NormalizedThrottle::new_saturating(throttles[3]),
            ];
            let publish_time = Instant::now();
            ACTUATOR_MOTORS.signal(msgs::ActuatorMotors {
                timestamp: publish_time,
                motor_commands,
            });

            // Stamp heartbeat for controller watchdog.
            super::LAST_CONTROLLER_PUBLISH.lock(|c| c.set(Some(publish_time)));

            // 10. Publish telemetry.
            att_pub.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: publish_time,
                collective_thrust_n: effective_thrust_n,
                attitude_quaternion: att_ref
                    .attitude_quaternion
                    .unwrap_or(UnitQuaternion::identity()),
                body_rate_rad_s,
                torque_n_m,
            });
            pos_pub.publish_immediate(msgs::PositionControlSetpoint {
                timestamp: publish_time,
                position: pos_setpoint.position,
                velocity: pos_setpoint.velocity,
                yaw: pos_setpoint.yaw,
                collective_thrust_n,
            });

            // 11. Yield to other tasks.
            embassy_futures::yield_now().await;
        }
    }
}

#[embassy_executor::task]
pub async fn inner_loop_task() {
    let mut driver = InnerLoop::new(quadrotor_allocator());
    driver.run().await;
}
