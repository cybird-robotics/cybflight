use core::time::Duration;

use super::VEHICLE_ODOMETRY;
use crate::sensors;
use cybflight_core::mahony::{Mahony, MahonyError};
use cybflight_msgs as msgs;
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra as na;

pub struct FeedthroughEstimate<T> {
    position: na::Vector3<T>,
    attitude: na::UnitQuaternion<T>,
    velocity: na::Vector3<T>,
    acceleration: na::Vector3<T>,
    angular_rate: na::Vector3<T>,
}

impl<T: na::RealField + Copy> FeedthroughEstimate<T> {
    pub fn new() -> Self {
        Self {
            position: na::Vector3::zeros(),
            attitude: na::UnitQuaternion::identity(),
            velocity: na::Vector3::zeros(),
            acceleration: na::Vector3::zeros(),
            angular_rate: na::Vector3::zeros(),
        }
    }

    /// Reset all state to initial values.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// IMU prediction step (runs at 8 kHz).
    pub fn predict(&mut self, accel: na::Vector3<T>, gyro: na::Vector3<T>) {
        self.acceleration = accel;
        self.angular_rate = gyro;
    }

    /// VICON correction step (runs at ~100 Hz when VICON data arrives).
    pub fn correct_pose(&mut self, position: na::Vector3<T>, attitude: na::UnitQuaternion<T>) {
        self.position = position;
        self.attitude = attitude;
        self.velocity = na::Vector3::zeros();
    }
}

#[embassy_executor::task]
pub async fn feedthrough_estimate() {
    let mut imu_sub = sensors::IMU_1.subscriber().unwrap();
    let mut pose_sub = sensors::VICON_POSE.subscriber().unwrap();
    let att_pub = sensors::VEHICLE_ATTITUDE.immediate_publisher();
    let odom_pub = VEHICLE_ODOMETRY.immediate_publisher();

    let mut estimator: FeedthroughEstimate<f32> = FeedthroughEstimate::<f32>::new();
    let mut mahony: Mahony<f32> = Mahony::new();
    let mut prev_timestamp: Option<Instant> = None;

    let mut vicon_received = false;
    loop {
        // Await IMU at 8 kHz — this drives the loop rate.
        let sample = match imu_sub.next_message().await {
            WaitResult::Message(m) => m,
            WaitResult::Lagged(n) => {
                defmt::warn!("FeedthroughEstimate: dropped {} IMU samples", n);
                prev_timestamp = None;
                continue;
            }
        };

        // Compute dt for Mahony.
        let dt = match prev_timestamp {
            Some(prev) => {
                let emb_dt = sample.timestamp.duration_since(prev);
                emb_dt.into()
            }
            None => Duration::from_millis(1),
        };
        prev_timestamp = Some(sample.timestamp);

        // Run Mahony IMU-driven attitude update.
        let mahony_orientation = match mahony.update(sample.gyro_rad_s, sample.accel_m_s2, None, dt)
        {
            Ok(orientation) => Some(orientation),
            Err(MahonyError::ZeroAcceleration) => {
                defmt::debug!("FeedthroughEstimate: accel norm below threshold, skipping Mahony");
                None
            }
            Err(_) => {
                defmt::warn!("FeedthroughEstimate: unexpected Mahony error");
                None
            }
        };

        // Feedthrough prediction with IMU.
        estimator.predict(sample.accel_m_s2, sample.gyro_rad_s);

        // Drain VICON pose (non-blocking, ~100 Hz).
        while let Some(vicon) = pose_sub.try_next_message_pure() {
            estimator.correct_pose(vicon.position, vicon.orientation);
            mahony.set_orientation(vicon.orientation);
            if !vicon_received {
                vicon_received = true;
                defmt::info!("state estimator: first VICON pose received");
            }
        }

        // Choose attitude: VICON feedthrough when available, Mahony fallback otherwise.
        let attitude = if vicon_received {
            estimator.attitude
        } else if let Some(ori) = mahony_orientation {
            ori
        } else {
            continue;
        };

        let now = Instant::now();

        // Backward-compat: publish attitude for telemetry subscribers.
        att_pub.publish_immediate(msgs::VehicleAttitude {
            timestamp: now,
            orientation: attitude,
        });

        // Publish full vehicle state.
        odom_pub.publish_immediate(msgs::VehicleOdometry {
            timestamp: now,
            pose: msgs::Pose {
                position: estimator.position,
                orientation: attitude,
            },
            twist: msgs::Twist {
                linear: estimator.velocity,
                angular: estimator.angular_rate,
            },
        });
    }
}
