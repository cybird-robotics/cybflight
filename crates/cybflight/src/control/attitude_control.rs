use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    vehicle_model::{motor_thrust_to_throttle, thrust_torque_to_motor_thrusts, ThrustTorque, VehicleModel},
};
use embassy_futures::select::{select3, Either3};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::UnitQuaternion;

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs,
    sensors::{self, MANUAL_CONTROL},
    vehicle::QUADROTOR,
};

pub struct AttitudeControl<'a, Mdl: VehicleModel<Scalar = f32>> {
    ac: geometric_controller::GeometricAttitudeController<f32>,
    mdl: &'a Mdl,
}

impl<'a, Mdl: VehicleModel<Scalar = f32>> AttitudeControl<'a, Mdl> {
    pub fn new(mdl: &'a Mdl) -> Self {
        Self {
            ac: geometric_controller::GeometricAttitudeController::default(),
            mdl,
        }
    }

    pub async fn run(&mut self) -> ! {
        let mut rate_sub = sensors::IMU_1.subscriber().unwrap();
        let mut att_sub = sensors::VEHICLE_ATTITUDE.subscriber().unwrap();
        let mut rc_sub = MANUAL_CONTROL.subscriber().unwrap();
        let publisher = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
        let max_thrust = self.mdl.max_thrust_per_motor();

        // Stub reference: hover at z = 1 m, level attitude, zero velocity.
        // TODO: subscribe to super::ATTITUDE_SETPOINT for a live target.
        let mut att_ref = attitude_control::AttitudeControlSetpoint::default();
        let mut state = attitude_control::AttitudeControlState::default();

        let mut collective_thrust_n = 0.0;
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
                        defmt::warn!("NMPC: dropped {} attitude updates", n);
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
                            collective_thrust_n = thrust;
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

            // Convert RC normalized thrust [0,1] to total force (N), then allocate.
            let total_thrust_n = collective_thrust_n * 4.0 * max_thrust;
            let thrusts_n = thrust_torque_to_motor_thrusts(
                &ThrustTorque {
                    collective_thrust_n: total_thrust_n,
                    torque_n_m,
                },
                self.mdl,
            );

            // Per-motor thrust (N) → normalized throttle [0,1] via linear thrust curve.
            let motor_commands = [
                msgs::NormalizedThrottle::new_saturating(motor_thrust_to_throttle(thrusts_n[0], max_thrust)),
                msgs::NormalizedThrottle::new_saturating(motor_thrust_to_throttle(thrusts_n[1], max_thrust)),
                msgs::NormalizedThrottle::new_saturating(motor_thrust_to_throttle(thrusts_n[2], max_thrust)),
                msgs::NormalizedThrottle::new_saturating(motor_thrust_to_throttle(thrusts_n[3], max_thrust)),
            ];
            ACTUATOR_MOTORS.signal(msgs::ActuatorMotors {
                timestamp: Instant::now(),
                motor_commands,
            });

            publisher.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: Instant::now(),
                collective_thrust_n,
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
    let mut driver = AttitudeControl::new(&QUADROTOR);
    driver.run().await;
}
