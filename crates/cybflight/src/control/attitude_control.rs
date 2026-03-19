use core::time::Duration;

use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    mixer::LinearAllocator,
};
use discrete_pid::{
    pid::{PidConfigBuilder, PidController},
    time::Micros,
};
use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3, Vector4};

/// If no MANUAL_CONTROL update for this long, zero thrust and rates.
/// Defense-in-depth: catches stale RC data even if failsafe task is delayed.
const RC_STALE_TIMEOUT_MS: u64 = 250;

/// Nominal IMU loop period. Used as sample_time for the rate PIDs so that when
/// I/D terms are later enabled, their gains are time-invariant at this rate.
const RATE_PID_SAMPLE_TIME: Duration = Duration::from_micros(125); // 8 kHz

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs,
    sensors::{self, MANUAL_CONTROL},
};

/// Per-axis rate PID controllers (roll, pitch, yaw).
///
/// `input`   = measured body rate (rad/s)
/// `setpoint`= rate reference from the outer attitude loop (rad/s)
/// `output`  = rate torque contribution (N·m)
///
/// With ki = kd = 0, `output = kp * (setpoint − input) = −kp * rate_error`,
/// which is exactly the old `−k_rate_torque * rate_error` term from the
/// geometric controller when `kp[axis] == k_rate_torque[axis]`.
struct RatePids {
    roll: PidController<Micros, f32>,
    pitch: PidController<Micros, f32>,
    yaw: PidController<Micros, f32>,
}

impl RatePids {
    fn new(kp_roll: f32, kp_pitch: f32, kp_yaw: f32) -> Self {
        let make = |kp: f32| {
            let config = PidConfigBuilder::<f32>::default()
                .kp(kp)
                .ki(0.0)
                .kd(0.0)
                .sample_time(RATE_PID_SAMPLE_TIME)
                .build()
                .expect("rate PID config invalid");
            PidController::new_uninit(config)
        };
        Self {
            roll: make(kp_roll),
            pitch: make(kp_pitch),
            yaw: make(kp_yaw),
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

pub struct AttitudeControl<const N: usize> {
    ac: geometric_controller::GeometricAttitudeController<f32>,
    rate_pids: RatePids,
    allocator: LinearAllocator<N>,
}

impl<const N: usize> AttitudeControl<N> {
    pub fn new(
        body: &cybflight_core::mixer::RigidBodyParams,
        allocator: LinearAllocator<N>,
    ) -> Self {
        Self {
            ac: geometric_controller::GeometricAttitudeController::new(
                Vector3::new(1.0, 1.0, 0.5), // k_ang_rate  [roll, pitch, yaw]
                Vector3::new(1.0, 1.0, 0.2), // k_ang_torque [roll, pitch, yaw]
            )
            .with_inertia(body.inertia_matrix()),
            // kp values match the old k_rate_torque for identical behavior when I=D=0
            rate_pids: RatePids::new(0.3, 0.25, 0.15),
            allocator,
        }
    }

    pub async fn run(&mut self) -> ! {
        let mut rate_sub = sensors::IMU_1.subscriber().unwrap();
        let mut att_sub = sensors::VEHICLE_ATTITUDE.subscriber().unwrap();
        let mut rc_sub = MANUAL_CONTROL.subscriber().unwrap();
        let publisher = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
        let max_thrust_n = self.allocator.max_collective_thrust_n();

        // Stub reference: hover at z = 1 m, level attitude, zero velocity.
        // TODO: subscribe to super::ATTITUDE_SETPOINT for a live target.
        let mut att_ref = attitude_control::AttitudeControlSetpoint::default();
        let mut state = attitude_control::AttitudeControlState::default();

        // Normalized thrust [0, 1] from the RC interpreter.
        let mut thrust_normalized = 0.0_f32;
        // Track when we last received a valid RC setpoint for stale-data detection.
        let mut last_rc_time = Instant::now();

        loop {
            // Await the highest-rate input (IMU at 8 kHz) to drive the loop.
            let imu = rate_sub.next_message_pure().await;
            state.body_rate_rad_s = imu.gyro_rad_s;

            // Drain attitude and RC non-blockingly so they never starve.
            while let Some(att) = att_sub.try_next_message_pure() {
                state.attitude_quaternion = att.orientation;
            }
            while let Some(rc) = rc_sub.try_next_message_pure() {
                att_ref.body_rate_rad_s = [rc.roll_rate, rc.pitch_rate, rc.yaw_rate].into();
                thrust_normalized = rc.thrust;
                last_rc_time = Instant::now();
            }

            if Instant::now().duration_since(last_rc_time).as_millis() > RC_STALE_TIMEOUT_MS {
                thrust_normalized = 0.0;
                att_ref.body_rate_rad_s = Vector3::zeros();
            }

            // Outer loop: attitude error → rate setpoint + feedforward torque.
            let AttitudeControlOutput {
                body_rate_rad_s: rate_ref,
                torque_n_m: outer_torque,
            } = self.ac.compute(&state, &att_ref);

            // Inner loop: per-axis rate PID → rate feedback torque.
            let rate_torque = self.rate_pids.compute(state.body_rate_rad_s, rate_ref);
            let torque_n_m = outer_torque + rate_torque;

            // Convert normalized thrust [0, 1] → total Newtons, then allocate.
            // When thrust is near zero, send a small idle command so motors keep
            // spinning for gyroscopic stability. The DShot driver enforces a
            // separate anti-stall floor as a safety net.
            const IDLE_THROTTLE: f32 = 0.005; // 0.5% normalized
            let effective_thrust = if thrust_normalized < IDLE_THROTTLE {
                IDLE_THROTTLE
            } else {
                thrust_normalized
            };
            let total_thrust_n = effective_thrust * max_thrust_n;
            let demand = Vector4::new(total_thrust_n, torque_n_m.x, torque_n_m.y, torque_n_m.z);
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

            publisher.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: Instant::now(),
                collective_thrust_n: total_thrust_n,
                attitude_quaternion: att_ref
                    .attitude_quaternion
                    .unwrap_or(UnitQuaternion::identity()),
                body_rate_rad_s: rate_ref,
                torque_n_m,
            });

            embassy_futures::yield_now().await;
        }
    }
}

#[embassy_executor::task(pool_size = 2)]
pub async fn attitude_control_task() {
    let params = crate::params::get();
    let effectiveness = cybflight_core::mixer::MotorEffectiveness::from_motors(&params.motors);
    let allocator = LinearAllocator::new(effectiveness);
    let mut driver = AttitudeControl::new(&params.body, allocator);
    driver.run().await;
}
