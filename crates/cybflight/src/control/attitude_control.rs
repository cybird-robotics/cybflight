use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    mixer::LinearAllocator,
};
use embassy_futures::select::{select3, Either3};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector4};

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs,
    sensors::{self, MANUAL_CONTROL},
    vehicle::quadrotor_allocator,
};

pub struct AttitudeControl<const N: usize> {
    ac: geometric_controller::GeometricAttitudeController<f32>,
    allocator: LinearAllocator<N>,
}

impl<const N: usize> AttitudeControl<N> {
    pub fn new(allocator: LinearAllocator<N>) -> Self {
        Self {
            ac: geometric_controller::GeometricAttitudeController::default(),
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

        loop {
            match select3(
                att_sub.next_message(),
                rate_sub.next_message(),
                rc_sub.next_message(),
            )
            .await
            {
                Either3::First(att) => match att {
                    WaitResult::Message(msgs::VehicleAttitude { orientation, .. }) => {
                        state.attitude_quaternion = orientation;
                    }
                    WaitResult::Lagged(n) => {
                        defmt::warn!("Attitude control: dropped {} attitude updates", n);
                        continue;
                    }
                },

                Either3::Second(rate) => match rate {
                    WaitResult::Message(msgs::Imu { gyro_rad_s, .. }) => {
                        state.body_rate_rad_s = gyro_rad_s;
                    }
                    WaitResult::Lagged(n) => {
                        defmt::warn!("Attitude control: dropped {} IMU updates", n);
                        continue;
                    }
                },

                Either3::Third(rc) => {
                    match rc {
                        WaitResult::Message(msgs::ManualControlSetpoint {
                            timestamp: _,
                            thrust,
                            roll_rate,
                            pitch_rate,
                            yaw_rate,
                        }) => {
                            att_ref.body_rate_rad_s = [roll_rate, pitch_rate, yaw_rate].into();
                            thrust_normalized = thrust;
                        }
                        WaitResult::Lagged(n) => {
                            defmt::warn!("Attitude control: dropped {} RC updates", n);
                            continue;
                        }
                    };
                }
            };

            let AttitudeControlOutput {
                body_rate_rad_s,
                torque_n_m,
            } = self.ac.compute(&state, &att_ref);

            // Convert normalized thrust [0, 1] → total Newtons, then allocate.
            let total_thrust_n = thrust_normalized * max_thrust_n;
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
                body_rate_rad_s,
                torque_n_m,
            });

            embassy_futures::yield_now().await;
        }
    }
}

#[embassy_executor::task(pool_size = 2)]
pub async fn attitude_control_task() {
    let mut driver = AttitudeControl::new(quadrotor_allocator());
    driver.run().await;
}
