//! Unified control loop: position + attitude + rate, IMU-driven at 8 kHz.
//!
//! The active estimator feature determines which control layers are engaged:
//!
//! | Feature      | Estimator       | Setpoint source | Control layers           |
//! |--------------|-----------------|-----------------|--------------------------|
//! | `est_mahony` | Mahony filter   | RC sticks       | attitude + rate          |
//! | `est_eskf`   | ESKF + Vicon    | ENU position    | position + attitude + rate |
//!
//! The IMU always drives the loop at 8 kHz. Estimator state is drained
//! non-blockingly each tick. The position controller (`est_eskf`) runs
//! decimated to ~100 Hz via an IMU-tick counter.

use core::time::Duration;

#[cfg(feature = "est_eskf")]
use cybflight_core::position_control::{self, pd_ff_control};
use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    mixer::LinearAllocator,
    params::PidGains,
};

use discrete_pid::{
    pid::{PidConfigBuilder, PidController},
    time::Micros,
};
use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3, Vector4};

#[cfg(feature = "est_mahony")]
use crate::sensors::MANUAL_CONTROL;
use crate::{motors::ACTUATOR_MOTORS, msgs, sensors};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Rate PID sample time matches the IMU loop period (8 kHz).
const RATE_PID_SAMPLE_TIME: Duration = Duration::from_micros(125);

/// RC staleness timeout: zero setpoints after this long without a new frame.
#[cfg(feature = "est_mahony")]
const RC_STALE_TIMEOUT_MS: u64 = 250;

/// Odometry staleness timeout: skip control when ESKF output is this old.
#[cfg(feature = "est_eskf")]
const ODOM_STALE_TIMEOUT_MS: u64 = 100;

/// Position controller runs every N IMU ticks ≈ 100 Hz (8000 / 80).
#[cfg(feature = "est_eskf")]
const POS_CTRL_DECIMATION: u32 = 80;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_yaw(q: &UnitQuaternion<f32>) -> f32 {
    let (_, _, yaw) = q.euler_angles();
    yaw
}

/// Reject odometry with any non-finite component.
#[cfg(feature = "est_eskf")]
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

// ── Shared types ──────────────────────────────────────────────────────────────

struct RatePids {
    roll: PidController<Micros, f32>,
    pitch: PidController<Micros, f32>,
    yaw: PidController<Micros, f32>,
}

impl RatePids {
    fn new(k_roll: PidGains, k_pitch: PidGains, k_yaw: PidGains) -> Self {
        let make = |k: PidGains| {
            let PidGains { kp, ki, kd } = k;
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

    fn compute(&mut self, fb: Vector3<f32>, ref_: Vector3<f32>) -> Vector3<f32> {
        let ts = Micros(Instant::now().as_micros());
        Vector3::new(
            self.roll.compute(fb.x, ref_.x, ts, None),
            self.pitch.compute(fb.y, ref_.y, ts, None),
            self.yaw.compute(fb.z, ref_.z, ts, None),
        )
    }
}

// ── Controller struct ─────────────────────────────────────────────────────────

pub struct ControlLoop<const N: usize> {
    ac: geometric_controller::GeometricAttitudeController<f32>,
    rate_pids: RatePids,
    allocator: LinearAllocator<N>,
    /// Gate: skip control until the estimator has produced its first output.
    state_valid: bool,

    // ── est_mahony: RC stick state + angle-mode integration ──────────────────
    // roll/pitch carry tilt angle commands [rad]; yaw is a heading rate [rad/s].
    // These are applied as raw body rates when state_valid is still false.
    #[cfg(feature = "est_mahony")]
    roll: f32,
    #[cfg(feature = "est_mahony")]
    pitch: f32,
    #[cfg(feature = "est_mahony")]
    yaw: f32,
    /// Accumulated yaw heading reference [rad], seeded from Mahony on first frame.
    #[cfg(feature = "est_mahony")]
    yaw_ref: Option<f32>,

    // ── est_eskf: position controller ───────────────────────────────────────
    #[cfg(feature = "est_eskf")]
    pc: pd_ff_control::PositionController<f32>,
    #[cfg(feature = "est_eskf")]
    pos_state: position_control::PositionControlState<f32>,
    #[cfg(feature = "est_eskf")]
    pos_setpoint: position_control::PositionControlSetpoint<f32>,
    #[cfg(feature = "est_eskf")]
    collective_thrust_n: f32,
    #[cfg(feature = "est_eskf")]
    pos_ctrl_counter: u32,
}

// ── Construction ──────────────────────────────────────────────────────────────

#[cfg(feature = "est_mahony")]
impl<const N: usize> ControlLoop<N> {
    pub fn new(
        params: &cybflight_core::params::VehicleParams,
        allocator: LinearAllocator<N>,
    ) -> Self {
        let g = &params.control;
        Self {
            ac: geometric_controller::GeometricAttitudeController::new(
                Vector3::new(g.att_k_rate[0], g.att_k_rate[1], g.att_k_rate[2]),
                Vector3::new(1.0, 1.0, 0.2), // k_ang_torque
            )
            .with_inertia(params.body.inertia_matrix()),
            rate_pids: RatePids::new(g.rate_pid(0), g.rate_pid(1), g.rate_pid(2)),
            allocator,
            state_valid: false,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            yaw_ref: None,
        }
    }
}

#[cfg(feature = "est_eskf")]
impl<const N: usize> ControlLoop<N> {
    pub fn new(allocator: LinearAllocator<N>) -> Self {
        let params = crate::params::get();
        let g = &params.control;
        Self {
            ac: geometric_controller::GeometricAttitudeController::new(
                Vector3::new(g.att_k_rate[0], g.att_k_rate[1], g.att_k_rate[2]),
                Vector3::new(1.0, 1.0, 0.2), // k_ang_torque
            )
            .with_inertia(params.body.inertia_matrix()),
            rate_pids: RatePids::new(g.rate_pid(0), g.rate_pid(1), g.rate_pid(2)),
            allocator,
            state_valid: false,
            pc: pd_ff_control::PositionController::new(
                Vector3::new(g.pos_kp[0], g.pos_kp[1], g.pos_kp[2]),
                Vector3::new(g.pos_kd[0], g.pos_kd[1], g.pos_kd[2]),
                position_control::VehicleParams {
                    mass: params.body.mass_kg,
                    gravity: 9.81,
                },
            ),
            pos_state: position_control::PositionControlState::default(),
            pos_setpoint: position_control::PositionControlSetpoint::default(),
            collective_thrust_n: 0.0,
            pos_ctrl_counter: 0,
        }
    }
}

// ── Task entry points ─────────────────────────────────────────────────────────

#[cfg(feature = "est_mahony")]
#[embassy_executor::task(pool_size = 2)]
pub async fn control_loop_task() {
    let params = crate::params::get();
    let effectiveness = cybflight_core::mixer::MotorEffectiveness::from_motors(&params.motors);
    let allocator = LinearAllocator::new(effectiveness);
    let mut driver = ControlLoop::new(&params, allocator);
    driver.run().await;
}

#[cfg(feature = "est_eskf")]
#[embassy_executor::task]
pub async fn control_loop_task() {
    let mut driver = ControlLoop::new(crate::vehicle::quadrotor_allocator());
    driver.run().await;
}

// ── Unified control loop ──────────────────────────────────────────────────────

#[cfg(any(feature = "est_mahony", feature = "est_eskf"))]
impl<const N: usize> ControlLoop<N> {
    pub async fn run(&mut self) -> ! {
        // ── Subscriptions ────────────────────────────────────────────────────
        let mut rate_sub = sensors::IMU_1.subscriber().unwrap();
        let att_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
        let max_thrust_n = self.allocator.max_collective_thrust_n();

        #[cfg(feature = "est_mahony")]
        let mut att_sub = sensors::VEHICLE_ATTITUDE.subscriber().unwrap();
        #[cfg(feature = "est_mahony")]
        let mut rc_sub = MANUAL_CONTROL.subscriber().unwrap();
        #[cfg(feature = "est_mahony")]
        let mut last_rc_time = Instant::now();
        #[cfg(feature = "est_mahony")]
        let mut thrust_normalized = 0.0_f32;
        #[cfg(feature = "est_mahony")]
        let mut prev_time = Instant::now();

        #[cfg(feature = "est_eskf")]
        let mut odom_sub = sensors::VEHICLE_ODOMETRY.subscriber().unwrap();
        #[cfg(feature = "est_eskf")]
        let pos_pub = super::POSITION_CONTROL_SETPOINT.immediate_publisher();
        #[cfg(feature = "est_eskf")]
        let mut last_odom_time: Option<Instant> = None;

        let mut att_ref = attitude_control::AttitudeControlSetpoint::default();
        let mut att_state = attitude_control::AttitudeControlState::default();

        // est_eskf: seed the position setpoint from the first AUTO_SETPOINT
        // before entering the main loop (blocks until rc_interpreter publishes).
        #[cfg(feature = "est_eskf")]
        {
            let sp = super::AUTO_SETPOINT.wait().await;
            self.pos_setpoint = position_control::PositionControlSetpoint {
                position: sp.pose.position,
                velocity: sp.twist.linear,
                yaw: extract_yaw(&sp.pose.orientation),
                ..Default::default()
            };
        }

        loop {
            // 1. Await IMU (8 kHz) — drives the loop.
            let imu = rate_sub.next_message_pure().await;
            att_state.body_rate_rad_s = imu.gyro_rad_s;

            let now = Instant::now();
            #[cfg(feature = "est_mahony")]
            let dt_s = {
                let dt = now.duration_since(prev_time).as_micros() as f32 * 1e-6;
                prev_time = now;
                dt
            };

            // 2. Drain estimator state non-blockingly.
            //    est_mahony: pull attitude from Mahony filter.
            //    est_eskf:   pull full state from ESKF; enforce odometry freshness.
            #[cfg(feature = "est_mahony")]
            while let Some(att) = att_sub.try_next_message_pure() {
                att_state.attitude_quaternion = att.orientation;
                self.state_valid = true;
            }

            #[cfg(feature = "est_eskf")]
            {
                let mut got_fresh = false;
                while let Some(odom) = odom_sub.try_next_message_pure() {
                    if odom_is_valid(&odom) {
                        att_state.attitude_quaternion = odom.pose.orientation;
                        self.pos_state.position = odom.pose.position;
                        self.pos_state.velocity = odom.twist.linear;
                        self.pos_state.attitude = odom.pose.orientation;
                        got_fresh = true;
                        self.state_valid = true;
                    }
                }
                if got_fresh {
                    last_odom_time = Some(now);
                }
                // Skip control when odometry is stale (ESKF diverged or Vicon lost).
                if let Some(t) = last_odom_time {
                    if now.duration_since(t).as_millis() > ODOM_STALE_TIMEOUT_MS {
                        continue;
                    }
                }
            }

            // Skip control until the estimator has produced its first output.
            if !self.state_valid {
                continue;
            }

            // 3. Setpoint source.
            //    est_mahony: RC sticks → angle-mode attitude setpoint.
            //    est_eskf:   AUTO_SETPOINT → position controller → attitude setpoint.
            #[cfg(feature = "est_mahony")]
            {
                let mut latest_rc = None;
                while let Some(rc) = rc_sub.try_next_message_pure() {
                    last_rc_time = now;
                    latest_rc = Some(rc);
                }
                let rc_stale = now.duration_since(last_rc_time).as_millis() > RC_STALE_TIMEOUT_MS;

                if let Some(rc) = latest_rc {
                    self.roll = rc.roll_rate;
                    self.pitch = rc.pitch_rate;
                    self.yaw = rc.yaw_rate;
                    thrust_normalized = rc.thrust;
                }
                if rc_stale {
                    self.roll = 0.0;
                    self.pitch = 0.0;
                    self.yaw = 0.0;
                    thrust_normalized = 0.0;
                }

                let yaw_ref = self
                    .yaw_ref
                    .get_or_insert_with(|| extract_yaw(&att_state.attitude_quaternion));
                *yaw_ref += self.yaw * dt_s;
                att_ref.attitude_quaternion = Some(UnitQuaternion::from_euler_angles(
                    self.roll, self.pitch, *yaw_ref,
                ));
                att_ref.body_rate_rad_s = Vector3::new(0.0, 0.0, self.yaw);
            }

            #[cfg(feature = "est_eskf")]
            {
                if let Some(sp) = super::AUTO_SETPOINT.try_take() {
                    self.pos_setpoint = position_control::PositionControlSetpoint {
                        position: sp.pose.position,
                        velocity: sp.twist.linear,
                        yaw: extract_yaw(&sp.pose.orientation),
                        ..Default::default()
                    };
                }
                self.pos_ctrl_counter += 1;
                if self.pos_ctrl_counter >= POS_CTRL_DECIMATION {
                    self.pos_ctrl_counter = 0;
                    let pc_out = self.pc.compute(&self.pos_state, &self.pos_setpoint);
                    self.collective_thrust_n = pc_out.collective_thrust_n;
                    att_ref.attitude_quaternion = Some(pc_out.desired_attitude_quaternion);
                    att_ref.body_rate_rad_s = pc_out.desired_body_rate_rad_s;
                }
            }

            // 4. Geometric attitude controller: attitude error → body-rate setpoint.
            //    The feedforward torque term is discarded: the position/angle controller
            //    already provides a good rate_ref, and adding outer_torque would
            //    double-apply the attitude correction.
            let AttitudeControlOutput {
                body_rate_rad_s: rate_ref,
                ..
            } = self.ac.compute(&att_state, &att_ref);

            // 5. Rate PID: rate error → torque demand.
            let rate_torque = self.rate_pids.compute(att_state.body_rate_rad_s, rate_ref);

            // 6. Torque output.
            //    est_mahony: use rate_torque directly (angle mode is always on post-state_valid).
            //    est_eskf:   clamp per-axis to protect motors during position transients.
            #[cfg(feature = "est_mahony")]
            let torque_n_m = rate_torque;

            #[cfg(feature = "est_eskf")]
            let torque_n_m = {
                const MAX_ROLL: f32 = 0.8;
                const MAX_PITCH: f32 = 0.6;
                const MAX_YAW: f32 = 0.15;
                Vector3::new(
                    rate_torque.x.clamp(-MAX_ROLL, MAX_ROLL),
                    rate_torque.y.clamp(-MAX_PITCH, MAX_PITCH),
                    rate_torque.z.clamp(-MAX_YAW, MAX_YAW),
                )
            };

            // 7. Collective thrust.
            //    est_mahony: RC throttle (normalized) → Newtons, with idle floor.
            //    est_eskf:   position controller output, clamped to motor range;
            //                non-finite guard skips the frame rather than crashing.
            #[cfg(feature = "est_mahony")]
            let collective_thrust_n = {
                const IDLE_THROTTLE: f32 = 0.005;
                thrust_normalized.max(IDLE_THROTTLE) * max_thrust_n
            };

            #[cfg(feature = "est_eskf")]
            let collective_thrust_n = {
                if !self.collective_thrust_n.is_finite()
                    || !torque_n_m.x.is_finite()
                    || !torque_n_m.y.is_finite()
                    || !torque_n_m.z.is_finite()
                {
                    defmt::warn!("control_loop: non-finite output — skipping frame");
                    continue;
                }
                let idle_n = 0.005 * max_thrust_n;
                self.collective_thrust_n.max(idle_n).min(max_thrust_n)
            };

            // 8. Allocate thrust + torque to per-motor throttle commands.
            let demand = Vector4::new(
                collective_thrust_n,
                torque_n_m.x,
                torque_n_m.y,
                torque_n_m.z,
            );
            let throttles = self.allocator.allocate(demand);

            // 9. Motor commands.
            //    est_mahony: direct throttle (linear thrust assumption).
            //    est_eskf:   sqrt linearisation (motor thrust ∝ throttle²).
            #[cfg(feature = "est_mahony")]
            let motor_commands = [
                msgs::NormalizedThrottle::new_saturating(throttles[0]),
                msgs::NormalizedThrottle::new_saturating(throttles[1]),
                msgs::NormalizedThrottle::new_saturating(throttles[2]),
                msgs::NormalizedThrottle::new_saturating(throttles[3]),
            ];
            #[cfg(feature = "est_eskf")]
            let motor_commands = [
                msgs::NormalizedThrottle::new_saturating(libm::sqrtf(throttles[0])),
                msgs::NormalizedThrottle::new_saturating(libm::sqrtf(throttles[1])),
                msgs::NormalizedThrottle::new_saturating(libm::sqrtf(throttles[2])),
                msgs::NormalizedThrottle::new_saturating(libm::sqrtf(throttles[3])),
            ];

            let publish_time = Instant::now();
            ACTUATOR_MOTORS.signal(msgs::ActuatorMotors {
                timestamp: publish_time,
                motor_commands,
            });

            // est_eskf: stamp heartbeat for the failsafe controller watchdog.
            #[cfg(feature = "est_eskf")]
            super::LAST_CONTROLLER_PUBLISH.lock(|c| c.set(Some(publish_time)));

            // 10. Publish telemetry.
            att_pub.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: publish_time,
                collective_thrust_n,
                attitude_quaternion: att_ref
                    .attitude_quaternion
                    .unwrap_or(UnitQuaternion::identity()),
                body_rate_rad_s: rate_ref,
                torque_n_m,
            });
            #[cfg(feature = "est_eskf")]
            pos_pub.publish_immediate(msgs::PositionControlSetpoint {
                timestamp: publish_time,
                position: self.pos_setpoint.position,
                velocity: self.pos_setpoint.velocity,
                yaw: self.pos_setpoint.yaw,
            });

            embassy_futures::yield_now().await;
        }
    }
}
