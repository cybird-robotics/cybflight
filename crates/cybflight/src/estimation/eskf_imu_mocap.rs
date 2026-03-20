//! Motion-capture/INS state estimation task.
//!
//! Fuses IMU and motion capture pose (~100–360 Hz) into a 15-state
//! Error-State Kalman Filter (ESKF) and publishes `VehicleOdometry`.
//!
//! The ESKF predict step runs at `PREDICT_RATE_HZ` (1 kHz) rather than the
//! full IMU rate (8 kHz) to keep CPU usage low enough for the attitude control
//! loop to run unimpeded on the single-threaded executor.
//!
//! # State machine
//! 1. **Init** — waits for first mocap pose, then calibrates IMU biases (500 ms).
//! 2. **Running** — processes sensor events via `select`:
//!    - IMU   → decimate to 1 kHz, `eskf.predict()`, publish odometry at ~100 Hz
//!    - Mocap → `eskf.update_pos()` + `eskf.update_att()` (direct pose)

use embassy_futures::select::{select, Either};
use embassy_sync::pubsub::WaitResult;
use embassy_time::{Duration, Instant};
use nalgebra::Vector3;

use cybflight_core::eskf::{Eskf, EskfConfig};

use crate::estimation::{EstimatorPhase, ESTIMATOR_STATUS};
use crate::sensors;
use cybflight_msgs as msgs;

/// ESKF predict rate after decimation.  8 kHz IMU / 8 = 1 kHz.
const PREDICT_DECIMATION: u32 = 8;

/// Odometry publish decimation relative to predict rate.  1 kHz / 10 = 100 Hz.
const ODOM_DECIMATION: u32 = 1;

/// Mocap position noise std-dev [m].
const MOCAP_POS_STD: f32 = 0.001;
/// Mocap attitude noise std-dev [rad].
const MOCAP_ATT_STD: f32 = 0.01;

#[embassy_executor::task]
pub async fn estimation_task() {
    let mut imu_sub = sensors::IMU_1.subscriber().unwrap();
    let mut mocap_sub = sensors::VICON_POSE.subscriber().unwrap();
    let odom_pub = sensors::VEHICLE_ODOMETRY.immediate_publisher();

    // --- Phase 1A: Wait for first mocap pose ---
    let first_pose = loop {
        match mocap_sub.next_message().await {
            WaitResult::Message(p) => break p,
            WaitResult::Lagged(_) => continue,
        }
    };
    ESTIMATOR_STATUS.lock(|c| c.set(EstimatorPhase::CalibImu));

    // --- Phase 1B: IMU timed averaging (500 ms) ---
    const IMU_CAL_DURATION: Duration = Duration::from_millis(500);
    let (_accel_avg, gyro_avg) = {
        let mut a_acc = Vector3::<f32>::zeros();
        let mut g_acc = Vector3::<f32>::zeros();
        let mut count = 0u32;
        let mut t0 = Instant::now();
        loop {
            match imu_sub.next_message().await {
                WaitResult::Message(m) => {
                    a_acc += m.accel_m_s2;
                    g_acc += m.gyro_rad_s;
                    count += 1;
                    if Instant::now().duration_since(t0) >= IMU_CAL_DURATION {
                        let s = 1.0 / count as f32;
                        break (a_acc * s, g_acc * s);
                    }
                }
                WaitResult::Lagged(_) => {
                    a_acc = Vector3::zeros();
                    g_acc = Vector3::zeros();
                    count = 0;
                    t0 = Instant::now();
                }
            }
        }
    };

    // --- Initialise ESKF from mocap pose + gyro bias ---
    let mut eskf = Eskf::new(EskfConfig::default());
    eskf.init(
        first_pose.position,
        first_pose.orientation,
        gyro_avg,         // gyro bias from stationary average
        Vector3::zeros(), // accel bias — let filter estimate
    );

    defmt::info!(
        "ESKF init: pos=[{},{},{}] gyro_bias=[{},{},{}]",
        first_pose.position.x,
        first_pose.position.y,
        first_pose.position.z,
        gyro_avg.x,
        gyro_avg.y,
        gyro_avg.z,
    );

    ESTIMATOR_STATUS.lock(|c| {
        c.set(EstimatorPhase::Running {
            roll_deg: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            pos: [
                first_pose.position.x,
                first_pose.position.y,
                first_pose.position.z,
            ],
            vel: [0.0; 3],
            gyro_bias: [gyro_avg.x, gyro_avg.y, gyro_avg.z],
            accel_bias: [0.0; 3],
        })
    });

    // --- Phase 2: Main loop ---
    // Decimation counters: `imu_skip` counts raw IMU samples between predicts,
    // `predict_count` counts predicts between odometry publishes.
    let mut imu_skip: u32 = 0;
    let mut predict_count: u32 = 0;
    let mut last_predict_ts = Instant::now();

    loop {
        match select(imu_sub.next_message(), mocap_sub.next_message()).await {
            Either::First(result) => {
                let sample = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(n) => {
                        defmt::warn!("estimation: dropped {} IMU samples", n);
                        continue;
                    }
                };

                // Decimate: only run predict every PREDICT_DECIMATION IMU samples.
                imu_skip += 1;
                if imu_skip < PREDICT_DECIMATION {
                    continue;
                }
                imu_skip = 0;

                let now = sample.timestamp;
                let dt = now.duration_since(last_predict_ts).as_micros() as f32 / 1_000_000.0;
                last_predict_ts = now;

                // Reject implausible dt (first sample after init, or huge gap)
                if dt <= 0.0 || dt > 0.05 {
                    continue;
                }

                eskf.predict(sample.accel_m_s2, sample.gyro_rad_s, dt);

                predict_count = predict_count.wrapping_add(1);
                if predict_count.is_multiple_of(ODOM_DECIMATION) {
                    let pos = eskf.position();
                    let vel = eskf.velocity();
                    let q = eskf.orientation();
                    let gb = eskf.gyro_bias();
                    let ab = eskf.accel_bias();
                    let (roll, pitch, yaw) = q.euler_angles();

                    ESTIMATOR_STATUS.lock(|c| {
                        c.set(EstimatorPhase::Running {
                            roll_deg: roll.to_degrees(),
                            pitch_deg: pitch.to_degrees(),
                            yaw_deg: yaw.to_degrees(),
                            pos: [pos.x, pos.y, pos.z],
                            vel: [vel.x, vel.y, vel.z],
                            gyro_bias: [gb.x, gb.y, gb.z],
                            accel_bias: [ab.x, ab.y, ab.z],
                        })
                    });
                    odom_pub.publish_immediate(msgs::VehicleOdometry {
                        timestamp: Instant::now(),
                        pose: msgs::Pose {
                            position: pos,
                            orientation: q,
                        },
                        twist: msgs::Twist {
                            linear: vel,
                            angular: sample.gyro_rad_s,
                        },
                    });
                }
            }

            Either::Second(result) => {
                let pose = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(_) => continue,
                };

                eskf.update_pos(pose.position, MOCAP_POS_STD);
                eskf.update_att(pose.orientation, MOCAP_ATT_STD);
            }
        }
    }
}
