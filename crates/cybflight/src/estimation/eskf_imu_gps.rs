//! GNSS/INS state estimation task.
//!
//! Fuses IMU (1 kHz), GPS (5 Hz), and barometer into a 15-state Error-State
//! Kalman Filter (ESKF) and publishes `VehicleOdometry`.
//!
//! # State machine
//! 1. **Init** — waits for a baro sample (baseline) then a GPS fix
//!    (fix_type≥3, num_sv≥6) to set the ENU origin.
//! 2. **Armed** — processes sensor events via `select3`:
//!    - IMU  → `eskf.predict()` + publish odometry every 10 samples (~100 Hz)
//!    - GPS  → `eskf.update_gnss_delayed()` (ENU pos + NED→ENU vel)
//!    - Baro → `eskf.update_altitude()` (relative, Pa/m at sea level)

use core::f64::consts::PI;

use cybflight_core::eskf::{Eskf, EskfConfig};
use embassy_futures::select::{Either3, select3};
use embassy_sync::pubsub::WaitResult;
use embassy_time::{Duration, Instant};
use nalgebra::{UnitQuaternion, Vector3};

use crate::estimation::{ESTIMATOR_STATUS, EstimatorPhase};
use crate::sensors;
use cybflight_msgs as msgs;

#[embassy_executor::task]
pub async fn estimation_task() {
    let mut imu_sub = sensors::IMU_1.subscriber().unwrap();
    let mut gps_sub = sensors::GPS_FIX.subscriber().unwrap();
    let mut baro_sub = sensors::BARO_1.subscriber().unwrap();
    let odom_pub = sensors::VEHICLE_ODOMETRY.immediate_publisher();

    // Declare Eskf before first await so it lives in the task's static Future
    // storage, not on a transient stack frame.
    let mut eskf = Eskf::new(EskfConfig::default());

    // --- Phase 1A: Init — get baro baseline ---
    let baro_ref_pa: f32 = loop {
        match baro_sub.next_message().await {
            WaitResult::Message(b) => break b.pressure_pa,
            WaitResult::Lagged(_) => continue,
        }
    };
    ESTIMATOR_STATUS.lock(|c| {
        c.set(EstimatorPhase::AwaitingGps {
            imu_ready: false,
            roll_deg: None,
            pitch_deg: None,
        })
    });

    // --- Phase 1B: Init — wait for good GPS fix ---
    let (lat0_deg, lon0_deg, alt0_m): (f64, f64, f64) = loop {
        match gps_sub.next_message().await {
            WaitResult::Message(fix) if fix.fix_type >= 3 && fix.num_sv >= 6 => {
                break (fix.lat_deg, fix.lon_deg, fix.alt_msl_mm as f64 / 1000.0);
            }
            _ => continue,
        }
    };
    ESTIMATOR_STATUS.lock(|c| c.set(EstimatorPhase::CalibImu));

    // Pre-compute longitude scaling factor for flat-earth ENU (valid < 5 km).
    let cos_lat0 = libm::cos(lat0_deg * PI / 180.0);

    // --- Phase 1C: IMU timed averaging (500 ms) ---
    const IMU_CAL_DURATION: Duration = Duration::from_millis(500);
    let (accel_avg, gyro_avg) = {
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

    // --- Phase 1E: Compute orientation and biases ---
    let roll = libm::atan2f(accel_avg.y, accel_avg.z);
    let pitch = libm::atan2f(
        -accel_avg.x,
        libm::sqrtf(accel_avg.y * accel_avg.y + accel_avg.z * accel_avg.z),
    );

    let yaw = 0.0_f32; // No magnetometer — GPS velocity will correct heading

    let q_init: nalgebra::Unit<nalgebra::Quaternion<f32>> =
        UnitQuaternion::from_euler_angles(roll, pitch, yaw);

    let gyro_bias_init = gyro_avg;
    let g_up = Vector3::new(0.0f32, 0.0, 9.81);
    let accel_bias_init = accel_avg - q_init.inverse() * g_up;

    eskf.init(Vector3::zeros(), q_init, gyro_bias_init, accel_bias_init);
    defmt::info!(
        "ESKF init: roll={}° pitch={}° yaw={}°  gyro_bias={} accel_bias={}  origin lat={} lon={} alt={}m baro_ref={}Pa",
        roll.to_degrees(),
        pitch.to_degrees(),
        yaw.to_degrees(),
        gyro_bias_init,
        accel_bias_init,
        lat0_deg as f32,
        lon0_deg as f32,
        alt0_m as f32,
        baro_ref_pa,
    );

    // Seed Running status immediately after init so a shell query never sees a stale phase.
    let gb = eskf.gyro_bias();
    let ab = eskf.accel_bias();
    ESTIMATOR_STATUS.lock(|c| {
        c.set(EstimatorPhase::Running {
            roll_deg: roll.to_degrees(),
            pitch_deg: pitch.to_degrees(),
            yaw_deg: yaw.to_degrees(),
            pos: [0.0; 3],
            vel: [0.0; 3],
            gyro_bias: [gb.x, gb.y, gb.z],
            accel_bias: [ab.x, ab.y, ab.z],
        })
    });

    // --- Phase 2: Main estimation loop ---
    let mut prev_imu_ts: Option<Instant> = None;
    let mut imu_count: u32 = 0;

    loop {
        match select3(
            imu_sub.next_message(),
            gps_sub.next_message(),
            baro_sub.next_message(),
        )
        .await
        {
            Either3::First(result) => {
                let sample = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(n) => {
                        defmt::warn!("ESKF: dropped {} IMU samples", n);
                        prev_imu_ts = None;
                        continue;
                    }
                };

                let dt = prev_imu_ts.map_or(0.001_f32, |prev| {
                    sample.timestamp.duration_since(prev).as_micros() as f32 / 1_000_000.0
                });
                prev_imu_ts = Some(sample.timestamp);

                let ts_us = sample.timestamp.as_micros();
                eskf.predict(sample.accel_m_s2, sample.gyro_rad_s, dt, ts_us);

                imu_count = imu_count.wrapping_add(1);
                if imu_count % 10 == 0 {
                    let pos = eskf.position();
                    let vel = eskf.velocity();
                    let gb = eskf.gyro_bias();
                    let ab = eskf.accel_bias();
                    let (roll_r, pitch_r, yaw_r) = eskf.orientation().euler_angles();
                    ESTIMATOR_STATUS.lock(|c| {
                        c.set(EstimatorPhase::Running {
                            roll_deg: roll_r.to_degrees(),
                            pitch_deg: pitch_r.to_degrees(),
                            yaw_deg: yaw_r.to_degrees(),
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
                            orientation: eskf.orientation(),
                        },
                        twist: msgs::Twist {
                            linear: vel,
                            angular: sample.gyro_rad_s - gb,
                        },
                    });
                }
            }

            Either3::Second(result) => {
                let fix = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(_) => continue,
                };

                // Require at least a 2D fix with reasonable accuracy.
                if fix.fix_type < 2 || fix.h_acc_mm > 5000 {
                    continue;
                }

                // LLA → ENU (flat-earth, valid < ~5 km from origin).
                let east = ((fix.lon_deg - lon0_deg) * 111_320.0 * cos_lat0) as f32;
                let north = ((fix.lat_deg - lat0_deg) * 111_320.0) as f32;
                let up = (fix.alt_msl_mm as f64 / 1000.0 - alt0_m) as f32;
                let pos = Vector3::new(east, north, up);

                // Velocity is already in ENU [m/s] — converted in sensors/gps.rs.
                let vel = fix.vel_enu_m_s;

                let pos_std = fix.h_acc_mm as f32 / 1000.0;
                let vel_std = if fix.s_acc_m_s > 0.0 {
                    fix.s_acc_m_s
                } else {
                    EskfConfig::default().vel_noise_std
                };

                let ts_us = fix.timestamp.as_micros();
                eskf.update_gnss_delayed(pos, vel, pos_std, vel_std, ts_us);
            }

            Either3::Third(result) => {
                let baro = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(_) => continue,
                };
                // Simple linear Pa → m conversion (≈1% error below 500 m AGL).
                let rel_alt = (baro_ref_pa - baro.pressure_pa) / 12.01;
                eskf.update_altitude(rel_alt);
            }
        }
    }
}
