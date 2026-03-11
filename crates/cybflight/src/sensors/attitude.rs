use core::time::Duration;

use cybflight_core::mahony::{Mahony, MahonyError};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;

use super::{IMU_1, VEHICLE_ATTITUDE};
use cybflight_msgs as msgs;

#[embassy_executor::task]
pub async fn mahony_task() {
    let mut sub = IMU_1.subscriber().unwrap();
    let publisher = VEHICLE_ATTITUDE.immediate_publisher();
    let mut mahony = Mahony::<f32>::new();
    let mut prev_timestamp: Option<Instant> = None;

    loop {
        let sample = match sub.next_message().await {
            WaitResult::Message(m) => m,
            WaitResult::Lagged(n) => {
                defmt::warn!("Mahony: dropped {} IMU samples", n);
                // dt would be invalid after a gap; reset the timestamp reference.
                prev_timestamp = None;
                continue;
            }
        };

        let dt = match prev_timestamp {
            Some(prev) => {
                let emb_dt = sample.timestamp.duration_since(prev);
                emb_dt.into()
            }
            // Nominal 1 ms on the first sample (matches 8 kHz ODR order of magnitude).
            None => Duration::from_millis(1),
        };
        prev_timestamp = Some(sample.timestamp);

        // TODO: pass magnetometer reading once a MAG channel exists.
        match mahony.update(sample.gyro_rad_s, sample.accel_m_s2, None, dt) {
            Ok(orientation) => {
                publisher.publish_immediate(msgs::VehicleAttitude {
                    timestamp: Instant::now(),
                    orientation,
                });
            }
            Err(MahonyError::ZeroAcceleration) => {
                defmt::debug!("Mahony: accel norm below threshold, skipping update");
            }
            Err(_) => {
                defmt::warn!("Mahony: unexpected update error");
            }
        }
    }
}
