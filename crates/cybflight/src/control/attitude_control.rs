use cybflight_core::attitude_control::{self, geometric_controller};
use embassy_futures::select::{select3, Either3};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::UnitQuaternion;

use crate::{
    msgs,
    sensors::{self, MANUAL_CONTROL},
};

pub struct AttitudeControl {
    ac: geometric_controller::GeometricAttitudeController<f32>,
}

impl AttitudeControl {
    pub fn new() -> Self {
        Self {
            ac: geometric_controller::GeometricAttitudeController::default(),
        }
    }

    pub async fn run(&mut self) -> ! {
        let mut rate_sub = sensors::IMU_1.subscriber().unwrap();
        let mut att_sub = sensors::VEHICLE_ATTITUDE.subscriber().unwrap();
        let mut rc_sub = MANUAL_CONTROL.subscriber().unwrap();
        let publisher = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();

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

            let output = self.ac.compute(&state, &att_ref);

            publisher.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: Instant::now(),
                collective_thrust_n,
                attitude_quaternion: att_ref
                    .attitude_quaternion
                    .unwrap_or(UnitQuaternion::identity()),
                body_rate_rad_s: output.body_rate_rad_s,
                torque_n_m: output.torque_n_m,
            });

            embassy_futures::yield_now().await;
        }
    }
}

impl Default for AttitudeControl {
    fn default() -> Self {
        Self::new()
    }
}

#[embassy_executor::task(pool_size = 2)]
pub async fn attitude_control_task() {
    let mut driver = AttitudeControl::new();
    driver.run().await;
}
