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
//! 1. **Init** — waits for first mocap pose, initialises ESKF with zero biases.
//! 2. **Converging** — predict/update loop is active; gyro-bias covariance is
//!    still above the convergence threshold.  Arming is blocked.
//! 3. **Running** — covariance has converged; `ESTIMATOR_READY` is set and
//!    arming is permitted.

use embassy_futures::select::{select, Either};
use embassy_sync::pubsub::WaitResult;
use embassy_time::{Duration, Instant};
use nalgebra::Vector3;

use cybflight_core::eskf::{Eskf, EskfConfig};

use core::sync::atomic::Ordering;

use crate::estimation::{EstimatorPhase, ESTIMATOR_READY, ESTIMATOR_STATUS};
use crate::sensors;
use cybflight_msgs as msgs;

/// ESKF predict rate after decimation.  8 kHz IMU / 8 = 1 kHz.
const PREDICT_DECIMATION: u32 = 8;

/// Odometry publish decimation relative to predict rate.  1 kHz / 10 = 100 Hz.
const ODOM_DECIMATION: u32 = 1;

/// Mocap position noise std-dev [m].
const MOCAP_POS_STD: f32 = 0.01;
/// Mocap attitude noise std-dev [rad].
const MOCAP_ATT_STD: f32 = 0.03;

/// Gyro-bias covariance trace threshold for convergence.
/// Initial trace = 3 × 0.01 = 0.03; this requires roughly a 10× reduction.
const GYRO_BIAS_COV_TRACE_THRESH: f32 = 0.003;

/// Mocap staleness threshold. If no fresh ViconPose has been accepted
/// within this window, the estimator stops publishing `VEHICLE_ODOMETRY`
/// so downstream consumers (INDI → DShot → failsafe) detect silence and
/// disarm. Mocap typically arrives at 100–360 Hz, so 100 ms tolerates
/// several missed frames before declaring loss.
const MOCAP_STALE: Duration = Duration::from_millis(100);

/// Reject IMU samples with any non-finite component — they cascade straight
/// into the filter's predict step and produce NaN state in one call.
fn imu_is_valid(accel: &Vector3<f32>, gyro: &Vector3<f32>) -> bool {
    accel.iter().all(|v| v.is_finite()) && gyro.iter().all(|v| v.is_finite())
}

/// Reject mocap frames with any non-finite component. One bad frame from the
/// transport pipeline (ESP bridge / COBS decode) would otherwise be absorbed
/// directly into the filter via `update_pos`/`update_att`.
fn mocap_is_valid(pose: &msgs::ViconPose) -> bool {
    let q = pose.orientation.as_vector();
    pose.position.iter().all(|v| v.is_finite())
        && q.x.is_finite()
        && q.y.is_finite()
        && q.z.is_finite()
        && q.w.is_finite()
}

#[embassy_executor::task]
pub async fn estimation_task() {
    let mut imu_sub = sensors::IMU_1.subscriber().unwrap();
    let mut mocap_sub = sensors::VICON_POSE.subscriber().unwrap();
    let odom_pub = sensors::VEHICLE_ODOMETRY.immediate_publisher();
    // Attitude telemetry (formerly published by mahony_task). Downstream
    // consumers: CRSF telemetry, ESP bridge, USB streaming — all cosmetic.
    let att_pub = sensors::VEHICLE_ATTITUDE.immediate_publisher();

    // --- Wait for first mocap pose ---
    let first_pose = loop {
        match mocap_sub.next_message().await {
            WaitResult::Message(p) => break p,
            WaitResult::Lagged(_) => continue,
        }
    };

    // --- Initialise ESKF from mocap pose, zero biases ---
    let mut eskf = Eskf::new(EskfConfig::default());
    eskf.init(
        first_pose.position,
        first_pose.orientation,
        Vector3::zeros(), // gyro bias — let filter estimate
        Vector3::zeros(), // accel bias — let filter estimate
    );

    defmt::info!(
        "ESKF init: pos=[{},{},{}]",
        first_pose.position.x,
        first_pose.position.y,
        first_pose.position.z,
    );

    let state_fields = |eskf: &Eskf| {
        let pos = eskf.position();
        let vel = eskf.velocity();
        let q = eskf.orientation();
        let gb = eskf.gyro_bias();
        let ab = eskf.accel_bias();
        let (roll, pitch, yaw) = q.euler_angles();
        (
            roll.to_degrees(),
            pitch.to_degrees(),
            yaw.to_degrees(),
            [pos.x, pos.y, pos.z],
            [vel.x, vel.y, vel.z],
            [gb.x, gb.y, gb.z],
            [ab.x, ab.y, ab.z],
        )
    };

    ESTIMATOR_STATUS.lock(|c| {
        let (roll_deg, pitch_deg, yaw_deg, pos, vel, gyro_bias, accel_bias) = state_fields(&eskf);
        c.set(EstimatorPhase::Converging {
            roll_deg,
            pitch_deg,
            yaw_deg,
            pos,
            vel,
            gyro_bias,
            accel_bias,
        })
    });

    // --- Main loop ---
    let mut imu_skip: u32 = 0;
    let mut predict_count: u32 = 0;
    let mut last_predict_ts = Instant::now();
    let mut last_mocap_ts = Instant::now();
    let mut converged = false;

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

                // Reject non-finite IMU samples before they enter the filter.
                if !imu_is_valid(&sample.accel_m_s2, &sample.gyro_rad_s) {
                    defmt::warn!("estimation: non-finite IMU sample, skipping");
                    continue;
                }

                // If the filter reset itself (NaN guard tripped), wait for a
                // fresh mocap frame to re-seed instead of running predict.
                if !eskf.is_initialized() {
                    continue;
                }

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

                // predict() clears `initialized` if it produced NaN. Report
                // it and drop the convergence flag so downstream consumers
                // know the estimate is no longer trusted.
                if !eskf.is_initialized() {
                    defmt::error!(
                        "ESKF: non-finite state after predict — awaiting re-init from mocap"
                    );
                    converged = false;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                    continue;
                }

                // Publish IMU biases for INDI bias correction (every predict step)
                super::ESKF_GYRO_BIAS.signal(eskf.gyro_bias());
                super::ESKF_ACCEL_BIAS.signal(eskf.accel_bias());

                predict_count = predict_count.wrapping_add(1);
                if predict_count.is_multiple_of(ODOM_DECIMATION) {
                    // Producer-side staleness gate: if mocap has stopped
                    // arriving, go silent so the failure propagates per
                    // docs/safety_protocol.md (VICON dropout example), and
                    // drop ESTIMATOR_READY so re-arming is refused at gate 4.
                    if Instant::now().duration_since(last_mocap_ts) > MOCAP_STALE {
                        if converged {
                            converged = false;
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            defmt::warn!("ESKF: mocap stale — arming blocked");
                        }
                        continue;
                    }

                    let (roll_deg, pitch_deg, yaw_deg, pos, vel, gyro_bias, accel_bias) =
                        state_fields(&eskf);

                    // Check convergence transition
                    if !converged && eskf.gyro_bias_cov_trace() < GYRO_BIAS_COV_TRACE_THRESH {
                        converged = true;
                        ESTIMATOR_READY.store(true, Ordering::Release);
                        defmt::info!(
                            "ESKF converged: gyro_bias=[{},{},{}] trace={}",
                            gyro_bias[0],
                            gyro_bias[1],
                            gyro_bias[2],
                            eskf.gyro_bias_cov_trace(),
                        );
                    }

                    let phase = if converged {
                        EstimatorPhase::Running {
                            roll_deg,
                            pitch_deg,
                            yaw_deg,
                            pos,
                            vel,
                            gyro_bias,
                            accel_bias,
                        }
                    } else {
                        EstimatorPhase::Converging {
                            roll_deg,
                            pitch_deg,
                            yaw_deg,
                            pos,
                            vel,
                            gyro_bias,
                            accel_bias,
                        }
                    };
                    ESTIMATOR_STATUS.lock(|c| c.set(phase));

                    let q = eskf.orientation();
                    let gb = eskf.gyro_bias();
                    let now_publish = Instant::now();
                    odom_pub.publish_immediate(msgs::VehicleOdometry {
                        timestamp: now_publish,
                        pose: msgs::Pose {
                            position: eskf.position(),
                            orientation: q,
                        },
                        twist: msgs::Twist {
                            linear: eskf.velocity(),
                            angular: sample.gyro_rad_s - gb,
                        },
                    });
                    att_pub.publish_immediate(msgs::VehicleAttitude {
                        timestamp: now_publish,
                        orientation: q,
                    });
                }
            }

            Either::Second(result) => {
                let pose = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(_) => continue,
                };

                // Reject non-finite mocap frames before they enter the filter.
                if !mocap_is_valid(&pose) {
                    defmt::warn!("estimation: non-finite mocap frame, rejecting");
                    continue;
                }

                last_mocap_ts = Instant::now();

                // If the filter reset (NaN guard), re-seed from this pose.
                if !eskf.is_initialized() {
                    defmt::error!("ESKF: re-initializing from mocap after NaN reset");
                    eskf.init(
                        pose.position,
                        pose.orientation,
                        Vector3::zeros(),
                        Vector3::zeros(),
                    );
                    converged = false;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                    last_predict_ts = Instant::now();
                    continue;
                }

                eskf.update_pos(pose.position, MOCAP_POS_STD);
                eskf.update_att(pose.orientation, MOCAP_ATT_STD);

                // If an update produced NaN, drop convergence so the next
                // mocap frame re-seeds the filter.
                if !eskf.is_initialized() {
                    defmt::error!("ESKF: non-finite state after mocap update — awaiting re-init");
                    converged = false;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                }
            }
        }
    }
}
