//! GNSS/INS state estimation task — mirror of `eskf_imu_mocap` with GPS
//! replacing motion capture.
//!
//! The ENU origin is anchored at the first **RTK-fixed** PVT
//! (NAV-PVT `carr_soln == 2`) rather than the first 3D fix, so the world
//! frame is centimeter-accurate from the start. Float-RTK (carr_soln=1)
//! and stand-alone fixes are not used as the anchor — they have meter-
//! scale absolute bias that would translate into a permanently miscalibrated
//! ENU frame. Subsequent fixes (any `pvt_is_usable`, including post-anchor
//! drops to float or stand-alone) feed `update_pos` / `update_vel` with σ
//! derived from u-blox's h_acc / v_acc / s_acc estimates, floored so the
//! filter cannot get over-confident.
//!
//! GPS-only builds have no attitude measurement — the ESKF leans on IMU
//! gravity-aided tilt + gyro integration alone, so yaw is observable only
//! through accel-coupled motion or (future work) a magnetometer. Expect
//! yaw drift in still hover; tracking degrades on turning legs but stays
//! bounded. See docs/HACKING.md GPS section (to be added) for the gap.
//!
//! # State machine
//! 1. **Init** — waits for first **RTK-fixed** PVT, anchors `LlhOrigin`,
//!    initialises ESKF with zero position / identity attitude / zero
//!    biases.
//! 2. **Converging** — predict/update loop active; gyro-bias covariance
//!    is above threshold.  Arming blocked.
//! 3. **Running** — covariance converged; `ESTIMATOR_READY` set.

use embassy_futures::select::{select, Either};
use embassy_sync::pubsub::WaitResult;
use embassy_time::{Duration, Instant};
use nalgebra::{UnitQuaternion, Vector3};

use cybflight_core::eskf::{Eskf, EskfConfig, UpdateOutcome};
use cybflight_core::geodetic::{ned_to_enu, LlhOrigin};

use core::f64::consts::PI;
use core::sync::atomic::Ordering;

use crate::estimation::{EstimatorPhase, ESTIMATOR_READY, ESTIMATOR_STATUS};
use crate::sensors;
use crate::sensors::gps::GpsNavPvt;
use cybflight_msgs as msgs;

/// ESKF predict rate after decimation.  8 kHz IMU / 8 = 1 kHz.
const PREDICT_DECIMATION: u32 = 8;

/// Odometry publish decimation relative to predict rate. 1 kHz / 10 = 100 Hz.
const ODOM_DECIMATION: u32 = 1;

/// Minimum SV count for a fix to initialise the filter or drive an update.
const GPS_MIN_SV: u8 = 6;

/// Drop fixes with horizontal accuracy estimates above this — they are
/// noise masquerading as data. 50 m covers degraded-but-useful conditions
/// without letting obvious garbage through.
const GPS_H_ACC_MAX_MM: u32 = 50_000;

/// Measurement σ floors so the filter cannot run with unrealistic
/// confidence when u-blox reports a < cm accuracy (which happens on a
/// stationary receiver with good geometry but doesn't reflect real
/// in-flight error).
const GPS_POS_SIGMA_FLOOR_M: f32 = 0.05;
#[allow(dead_code)] // Reserved for when update_vel is wired up.
const GPS_VEL_SIGMA_FLOOR_M_S: f32 = 0.10;

/// Consecutive rejected GPS frames before declaring the estimator
/// untrusted. NAV-PVT arrives at 5–10 Hz, so 5 frames ≈ 0.5–1 s of
/// pure dead-reckoning before the failsafe propagates. Tighter than
/// the 2 s `GPS_STALE` window so the system fails through this gate
/// (which has more diagnostic context) before staleness fires.
const MAX_CONSECUTIVE_REJECTS: u32 = 5;

/// Consecutive pose-jump rejections before forcing a filter re-init.
/// Tighter than `MAX_CONSECUTIVE_REJECTS` — and tighter than the mocap
/// equivalent — because a GPS position jump indicates catastrophic
/// carrier-phase ambiguity loss or a multipath wraparound, neither of
/// which heals within a frame or two. Two NAV-PVTs ≈ 200–400 ms.
const MAX_CONSECUTIVE_JUMPS: u32 = 2;

/// GPS staleness threshold. If no usable PVT has been accepted within
/// this window, the estimator stops publishing `VEHICLE_ODOMETRY` so
/// downstream consumers (INDI → DShot → failsafe) detect silence and
/// disarm. NAV-PVT arrives at 5–10 Hz; 2 s tolerates ~10–20 missed
/// frames before declaring loss.
const GPS_STALE: Duration = Duration::from_millis(2000);

/// Gyro-bias covariance trace threshold for convergence.
/// Initial trace = 3 × 0.01 = 0.03; this requires roughly a 10× reduction.
const GYRO_BIAS_COV_TRACE_THRESH: f32 = 0.003;

fn imu_is_valid(accel: &Vector3<f32>, gyro: &Vector3<f32>) -> bool {
    accel.iter().all(|v| v.is_finite()) && gyro.iter().all(|v| v.is_finite())
}

fn pvt_is_usable(pvt: &GpsNavPvt) -> bool {
    pvt.fix_type >= 3
        && pvt.num_sv >= GPS_MIN_SV
        && pvt.h_acc_mm <= GPS_H_ACC_MAX_MM
        && pvt.lat_deg.is_finite()
        && pvt.lon_deg.is_finite()
}

/// PVT acceptable as the *origin anchor*. Stricter than `pvt_is_usable`:
/// only an RTK-fixed solution (`carr_soln == 2`) is allowed, so the ENU
/// frame is centimeter-accurate from the start. Float-RTK and stand-alone
/// fixes carry meter-scale absolute bias that would bake a permanent
/// offset into every subsequent setpoint.
fn pvt_is_origin_anchor(pvt: &GpsNavPvt) -> bool {
    pvt_is_usable(pvt) && pvt.carr_soln >= 2
}

fn pvt_enu(pvt: &GpsNavPvt, origin: &LlhOrigin) -> (Vector3<f32>, Vector3<f32>) {
    let pos = origin.llh_to_enu(
        pvt.lat_deg * PI / 180.0,
        pvt.lon_deg * PI / 180.0,
        pvt.alt_msl_mm as f32 * 1e-3,
    );
    let vel_ned = Vector3::new(
        pvt.vel_north_mm_s as f32 * 1e-3,
        pvt.vel_east_mm_s as f32 * 1e-3,
        pvt.vel_down_mm_s as f32 * 1e-3,
    );
    (pos, ned_to_enu(vel_ned))
}

#[embassy_executor::task]
pub async fn estimation_task() {
    let mut imu_sub = sensors::IMU_1.subscriber().unwrap();
    let gps_signal = &sensors::GPS_NAV_PVT;
    let odom_pub = sensors::VEHICLE_ODOMETRY.immediate_publisher();
    let att_pub = sensors::VEHICLE_ATTITUDE.immediate_publisher();

    // --- Wait for first RTK-fixed PVT to anchor the ENU origin ---
    let origin = loop {
        let pvt = gps_signal.wait().await;
        if pvt_is_origin_anchor(&pvt) {
            defmt::info!(
                "GPS origin (RTK fixed): lat={} lon={} alt_msl_mm={} num_sv={} h_acc_mm={}",
                pvt.lat_deg,
                pvt.lon_deg,
                pvt.alt_msl_mm,
                pvt.num_sv,
                pvt.h_acc_mm,
            );
            break LlhOrigin::new(
                pvt.lat_deg * PI / 180.0,
                pvt.lon_deg * PI / 180.0,
                pvt.alt_msl_mm as f32 * 1e-3,
            );
        } else if pvt_is_usable(&pvt) {
            defmt::info!(
                "GPS waiting for RTK fixed: fix_type={} num_sv={} h_acc_mm={} carr_soln={} (need 2)",
                pvt.fix_type,
                pvt.num_sv,
                pvt.h_acc_mm,
                pvt.carr_soln,
            );
        } else {
            defmt::info!(
                "GPS waiting: fix_type={} num_sv={} h_acc_mm={}",
                pvt.fix_type,
                pvt.num_sv,
                pvt.h_acc_mm,
            );
        }
    };

    // --- Initialise ESKF at origin, identity orientation, zero biases ---
    let mut eskf = Eskf::new(EskfConfig::default());
    eskf.init(
        Vector3::zeros(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
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
            pos_reject_total: 0,
            att_reject_total: 0,
            pos_inflated_total: 0,
            att_inflated_total: 0,
            jump_total: 0,
        })
    });

    // --- Main loop ---
    let mut imu_skip: u32 = 0;
    let mut predict_count: u32 = 0;
    let mut last_predict_ts = Instant::now();
    // Seed `last_gps_ts` from the origin-anchor PVT timestamp so the
    // staleness gate doesn't fire spuriously on the first IMU sample.
    let mut last_gps_ts = Instant::now();
    let mut converged = false;
    let mut consecutive_rejects: u32 = 0;
    let mut consecutive_jumps: u32 = 0;
    let mut pos_reject_total: u32 = 0;
    let mut pos_inflated_total: u32 = 0;
    let mut jump_total: u32 = 0;

    loop {
        match select(imu_sub.next_message(), gps_signal.wait()).await {
            Either::First(result) => {
                let sample = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(n) => {
                        defmt::warn!("estimation: dropped {} IMU samples", n);
                        continue;
                    }
                };

                if !imu_is_valid(&sample.accel_m_s2, &sample.gyro_rad_s) {
                    defmt::warn!("estimation: non-finite IMU sample, skipping");
                    continue;
                }

                if !eskf.is_initialized() {
                    continue;
                }

                imu_skip += 1;
                if imu_skip < PREDICT_DECIMATION {
                    continue;
                }
                imu_skip = 0;

                let now = sample.timestamp;
                let dt = now.duration_since(last_predict_ts).as_micros() as f32 / 1_000_000.0;
                last_predict_ts = now;

                if dt <= 0.0 || dt > 0.05 {
                    continue;
                }

                eskf.predict(sample.accel_m_s2, sample.gyro_rad_s, dt);

                if !eskf.is_initialized() {
                    defmt::error!(
                        "ESKF: non-finite state after predict — awaiting re-init from GPS"
                    );
                    converged = false;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                    continue;
                }

                super::ESKF_GYRO_BIAS.signal(eskf.gyro_bias());
                super::ESKF_ACCEL_BIAS.signal(eskf.accel_bias());

                predict_count = predict_count.wrapping_add(1);
                if predict_count.is_multiple_of(ODOM_DECIMATION) {
                    // Producer-side staleness gate: if GPS has stopped
                    // arriving, go silent so the failure propagates per
                    // docs/safety_protocol.md (sensor-loss example), and
                    // drop ESTIMATOR_READY so re-arming is refused at gate 4.
                    if Instant::now().duration_since(last_gps_ts) > GPS_STALE {
                        if converged {
                            converged = false;
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            defmt::warn!("ESKF: GPS stale — arming blocked");
                        }
                        continue;
                    }

                    let (roll_deg, pitch_deg, yaw_deg, pos, vel, gyro_bias, accel_bias) =
                        state_fields(&eskf);

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

                    // GPS path has no attitude update, so att_*_total stay 0.
                    let phase = if converged {
                        EstimatorPhase::Running {
                            roll_deg,
                            pitch_deg,
                            yaw_deg,
                            pos,
                            vel,
                            gyro_bias,
                            accel_bias,
                            pos_reject_total,
                            att_reject_total: 0,
                            pos_inflated_total,
                            att_inflated_total: 0,
                            jump_total,
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
                            pos_reject_total,
                            att_reject_total: 0,
                            pos_inflated_total,
                            att_inflated_total: 0,
                            jump_total,
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

            Either::Second(pvt) => {
                if !pvt_is_usable(&pvt) {
                    defmt::warn!(
                        "GPS fix dropped: fix_type={} num_sv={} h_acc_mm={}",
                        pvt.fix_type,
                        pvt.num_sv,
                        pvt.h_acc_mm,
                    );
                    continue;
                }

                if !eskf.is_initialized() {
                    defmt::error!("ESKF: re-initializing at origin after NaN reset");
                    eskf.init(
                        Vector3::zeros(),
                        UnitQuaternion::identity(),
                        Vector3::zeros(),
                        Vector3::zeros(),
                    );
                    converged = false;
                    consecutive_rejects = 0;
                    consecutive_jumps = 0;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                    last_predict_ts = Instant::now();
                    last_gps_ts = Instant::now();
                    continue;
                }

                let (enu_pos, _enu_vel) = pvt_enu(&pvt, &origin);

                // Derive σ from u-blox's reported accuracy with a floor:
                // a stationary receiver under good geometry can report
                // sub-cm accuracy that doesn't reflect real in-flight
                // error, so the floor prevents the filter from running
                // with unrealistic confidence.
                let sigma_pos =
                    (pvt.h_acc_mm.max(pvt.v_acc_mm) as f32 * 1e-3).max(GPS_POS_SIGMA_FLOOR_M);

                let pos_outcome = eskf.update_pos_sparse(enu_pos, sigma_pos);

                match pos_outcome {
                    UpdateOutcome::Accepted { inflated: true } => {
                        pos_inflated_total = pos_inflated_total.wrapping_add(1);
                    }
                    UpdateOutcome::Accepted { inflated: false } => {}
                    UpdateOutcome::JumpRejected => {
                        // Counted in the jump branch below; don't count
                        // as a normal rejection too.
                    }
                    other => {
                        pos_reject_total = pos_reject_total.wrapping_add(1);
                        defmt::warn!(
                            "ESKF: GPS pos update rejected ({})",
                            super::outcome_tag(other)
                        );
                    }
                }

                // Jump rejection: a position jump is a catastrophic GPS
                // fault (carrier-ambiguity loss / multipath wraparound).
                // The filter's IMU-integrated estimate is still
                // trustworthy — only the incoming PVT is bad — so we
                // drop the frame and ride out on dead-reckoning until
                // clean fixes resume. After `MAX_CONSECUTIVE_JUMPS` we
                // drop `ESTIMATOR_READY` so arming is refused;
                // `last_gps_ts` is not refreshed either, so `GPS_STALE`
                // will eventually fire as the hard failsafe.
                if pos_outcome.is_jump() {
                    jump_total = jump_total.wrapping_add(1);
                    consecutive_jumps = consecutive_jumps.saturating_add(1);
                    defmt::error!(
                        "ESKF: GPS pos jump ({}/{})",
                        consecutive_jumps,
                        MAX_CONSECUTIVE_JUMPS,
                    );
                    if converged && consecutive_jumps >= MAX_CONSECUTIVE_JUMPS {
                        converged = false;
                        ESTIMATOR_READY.store(false, Ordering::Release);
                        defmt::error!(
                            "ESKF: {} consecutive GPS jumps — arming blocked",
                            consecutive_jumps,
                        );
                    }
                } else {
                    consecutive_jumps = 0;
                }

                // Fail-stop: only refresh staleness when the update was
                // absorbed (inflated or not). A frame that arrives but
                // is rejected by the gate doesn't count as a healthy
                // measurement. After `MAX_CONSECUTIVE_REJECTS` we drop
                // `ESTIMATOR_READY` — same gate as mocap, with a tighter
                // count to fit GPS's lower frame rate.
                if pos_outcome.is_accepted() {
                    last_gps_ts = Instant::now();
                    consecutive_rejects = 0;
                } else {
                    consecutive_rejects = consecutive_rejects.saturating_add(1);
                    if converged && consecutive_rejects >= MAX_CONSECUTIVE_REJECTS {
                        converged = false;
                        ESTIMATOR_READY.store(false, Ordering::Release);
                        defmt::error!(
                            "ESKF: {} consecutive GPS rejections — arming blocked",
                            consecutive_rejects,
                        );
                    }
                }

                if !eskf.is_initialized() {
                    defmt::error!("ESKF: non-finite state after GPS update — awaiting re-init");
                    converged = false;
                    consecutive_rejects = 0;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                }
            }
        }
    }
}
