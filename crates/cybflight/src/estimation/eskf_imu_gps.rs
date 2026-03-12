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

    // --- Phase 1E: Compute roll/pitch from gravity for initial status ---
    let roll_init = libm::atan2f(accel_avg.y, accel_avg.z);
    let pitch_init = libm::atan2f(
        -accel_avg.x,
        libm::sqrtf(accel_avg.y * accel_avg.y + accel_avg.z * accel_avg.z),
    );

    defmt::info!(
        "Init: roll={}° pitch={}°  origin lat={} lon={} alt={}m baro_ref={}Pa",
        roll_init.to_degrees(),
        pitch_init.to_degrees(),
        lat0_deg as f32,
        lon0_deg as f32,
        alt0_m as f32,
        baro_ref_pa,
    );

    ESTIMATOR_STATUS.lock(|c| {
        c.set(EstimatorPhase::Running {
            roll_deg: roll_init.to_degrees(),
            pitch_deg: pitch_init.to_degrees(),
            yaw_deg: 0.0,
            pos: [0.0; 3],
            vel: [0.0; 3],
            gyro_bias: [0.0; 3],
            accel_bias: [0.0; 3],
        })
    });

    // --- Phase 2: Main loop — raw GPS pos + IMU tilt, no ESKF ---
    let mut imu_count: u32 = 0;
    let mut roll_deg = roll_init.to_degrees();
    let mut pitch_deg = pitch_init.to_degrees();
    let mut gps_pos = [0.0f32; 3];
    let mut gps_vel = [0.0f32; 3];

    // Welford online variance for GPS ENU position (E, N, U).
    let mut gps_n: u32 = 0;
    let mut gps_mean = [0.0f32; 3];
    let mut gps_m2 = [0.0f32; 3]; // sum of squared deviations

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
                        defmt::warn!("estimation: dropped {} IMU samples", n);
                        continue;
                    }
                };

                let a = sample.accel_m_s2;
                roll_deg = libm::atan2f(a.y, a.z).to_degrees();
                pitch_deg = libm::atan2f(
                    -a.x,
                    libm::sqrtf(a.y * a.y + a.z * a.z),
                )
                .to_degrees();

                imu_count = imu_count.wrapping_add(1);
                if imu_count % 10 == 0 {
                    ESTIMATOR_STATUS.lock(|c| {
                        c.set(EstimatorPhase::Running {
                            roll_deg,
                            pitch_deg,
                            yaw_deg: 0.0,
                            pos: gps_pos,
                            vel: gps_vel,
                            gyro_bias: [0.0; 3],
                            accel_bias: [0.0; 3],
                        })
                    });
                    let q = UnitQuaternion::from_euler_angles(
                        roll_deg.to_radians(),
                        pitch_deg.to_radians(),
                        0.0,
                    );
                    odom_pub.publish_immediate(msgs::VehicleOdometry {
                        timestamp: Instant::now(),
                        pose: msgs::Pose {
                            position: Vector3::new(gps_pos[0], gps_pos[1], gps_pos[2]),
                            orientation: q,
                        },
                        twist: msgs::Twist {
                            linear: Vector3::new(gps_vel[0], gps_vel[1], gps_vel[2]),
                            angular: sample.gyro_rad_s,
                        },
                    });
                }
            }

            Either3::Second(result) => {
                let fix = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(_) => continue,
                };

                if fix.fix_type < 2 || fix.h_acc_mm > 5000 {
                    defmt::warn!(
                        "GPS rejected: fix_type={} h_acc={}mm",
                        fix.fix_type, fix.h_acc_mm,
                    );
                    continue;
                }

                // LLA → ENU (flat-earth, valid < ~5 km from origin).
                let east = ((fix.lon_deg - lon0_deg) * 111_320.0 * cos_lat0) as f32;
                let north = ((fix.lat_deg - lat0_deg) * 111_320.0) as f32;
                let up = (fix.alt_msl_mm as f64 / 1000.0 - alt0_m) as f32;
                let vel = fix.vel_enu_m_s;
                gps_pos = [east, north, up];
                gps_vel = [vel.x, vel.y, vel.z];

                // Log raw ENU position + quality indicators on every accepted fix.
                // v_acc is intentionally separate — it is typically 3–5× worse than
                // h_acc and explains large U errors during receiver warm-up.
                // pdop < 2.0 is excellent, < 5.0 acceptable.
                defmt::info!(
                    "GPS fix#{} type={} sv={} | E={}mm N={}mm U={}mm | h_acc={}mm v_acc={}mm pdop={}",
                    gps_n + 1,
                    fix.fix_type,
                    fix.num_sv,
                    (east * 1000.0) as i32,
                    (north * 1000.0) as i32,
                    (up * 1000.0) as i32,
                    fix.h_acc_mm,
                    fix.v_acc_mm,
                    fix.pdop,
                );

                // Welford update.
                gps_n += 1;
                for i in 0..3 {
                    let delta = gps_pos[i] - gps_mean[i];
                    gps_mean[i] += delta / gps_n as f32;
                    let delta2 = gps_pos[i] - gps_mean[i];
                    gps_m2[i] += delta * delta2;
                }

                if gps_n >= 2 {
                    let std_mm = [
                        (libm::sqrtf(gps_m2[0] / (gps_n - 1) as f32) * 1000.0) as i32,
                        (libm::sqrtf(gps_m2[1] / (gps_n - 1) as f32) * 1000.0) as i32,
                        (libm::sqrtf(gps_m2[2] / (gps_n - 1) as f32) * 1000.0) as i32,
                    ];
                    defmt::info!(
                        "GPS pos std (n={}): E={}mm N={}mm U={}mm",
                        gps_n, std_mm[0], std_mm[1], std_mm[2],
                    );
                }
            }

            Either3::Third(result) => {
                // Baro received — no-op in passthrough mode.
                let _ = result;
            }
        }
    }
}
