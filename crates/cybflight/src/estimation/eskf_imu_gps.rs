//! GNSS/INS state estimation task — the firmware-side wrapper around
//! `cybflight_core::eskf::EskfGpsGuard`.
//!
//! The ENU origin is anchored at the first **RTK-fixed** PVT
//! (NAV-PVT `carr_soln == 2`) rather than the first 3D fix, so the
//! world frame is centimeter-accurate from the start. Float-RTK and
//! stand-alone fixes are not used as the anchor — they have meter-scale
//! absolute bias that would translate into a permanently miscalibrated
//! ENU frame. Subsequent fixes (any usable PVT, including post-anchor
//! drops to float or stand-alone) feed the guard's position update.
//!
//! GPS-only builds have no attitude measurement — the ESKF leans on IMU
//! gravity-aided tilt + gyro integration alone, so yaw is observable
//! only through accel-coupled motion or (future work) a magnetometer.
//! Expect yaw drift in still hover.
//!
//! All GPS-side failsafe state (RTK debounce, jump/reject cascades,
//! staleness, NaN re-init, convergence, ready flag) lives in the
//! host-testable `EskfGpsGuard`. This task is the Embassy-side I/O
//! shell: subscribes to IMU, waits on the GPS Signal, runs predict on
//! the IMU branch, defers all measurement-related decisions to the
//! guard, translates `GpsGuardOutcome` → defmt logs, and publishes
//! odometry / attitude / `ESTIMATOR_READY`.

use embassy_futures::select::{Either, select};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::Vector3;

use cybflight_core::eskf::{
    Eskf, EskfConfig, EskfGpsGuard, GpsFix, GpsGuardConfig, GpsGuardOutcome, ReinitCause,
    UsabilityReason, is_pvt_origin_anchor,
};
use cybflight_core::geodetic::{LlhOrigin, ned_to_enu};

use core::f64::consts::PI;
use core::sync::atomic::Ordering;

use crate::estimation::{
    attitude_health_bits, evaluate_faults, ATTITUDE_HEALTH, ESKF_DEGRADED, ESKF_FAULTS,
    ESKF_HEALTH, ESKF_LAST_ATT_UPDATE, ESKF_LAST_JUMP_CASCADE, ESKF_LAST_NAN_RESET,
    ESKF_LAST_POS_UPDATE, ESKF_LAST_REJECT_CASCADE, ESKF_LAST_VEL_UPDATE, ESKF_SEVERE_FAULT,
    ESTIMATOR_READY, ESTIMATOR_STATUS, EstimatorPhase, FaultEvalInputs,
};
use crate::motors::IS_ARMED;
use crate::sensors;
use crate::sensors::gps::GpsNavPvt;
use cybflight_msgs as msgs;

/// ESKF predict rate after decimation. 8 kHz IMU / 8 = 1 kHz.
const PREDICT_DECIMATION: u32 = 8;

/// Odometry publish decimation relative to predict rate. 1 kHz / 1 = 1 kHz.
const ODOM_DECIMATION: u32 = 1;

/// Bias telemetry decimation relative to predict rate. 1 kHz / 100 = 10 Hz.
/// Mirrors the eskf_imu_mocap constant; biases drift slowly so the
/// channel rate is set well below the 1 kHz predict rate.
const BIAS_TELEM_DECIMATION: u32 = 100;

/// Position-jump gate threshold [m] for the GPS path. Looser than
/// `EskfConfig::default()` (1 m, sized for mocap) because at 5–10 Hz
/// NAV-PVT the legitimate inter-frame residual at 10 m/s flight speed
/// is ~1 m by itself; a 1 m gate would hard-reject healthy fast-flight
/// frames. 3 m still catches RTK carrier-ambiguity loss / multipath
/// wraparound, which jump tens of metres.
const GPS_MAX_POS_JUMP_M: f32 = 3.0;

/// Translate a NAV-PVT plus the LLH origin into the guard's ENU `GpsFix`.
fn pvt_to_fix(pvt: &GpsNavPvt, origin: &LlhOrigin) -> GpsFix {
    let enu_pos = origin.llh_to_enu(
        pvt.lat_deg * PI / 180.0,
        pvt.lon_deg * PI / 180.0,
        pvt.alt_msl_mm as f32 * 1e-3,
    );
    let vel_ned = Vector3::new(
        pvt.vel_north_mm_s as f32 * 1e-3,
        pvt.vel_east_mm_s as f32 * 1e-3,
        pvt.vel_down_mm_s as f32 * 1e-3,
    );
    GpsFix {
        enu_pos,
        enu_vel: ned_to_enu(vel_ned),
        h_acc_mm: pvt.h_acc_mm,
        v_acc_mm: pvt.v_acc_mm,
        s_acc_mm_s: pvt.s_acc_mm_s,
        num_sv: pvt.num_sv,
        fix_type: pvt.fix_type,
        carr_soln: pvt.carr_soln,
        timestamp_ms: pvt.timestamp.as_millis(),
    }
}

fn imu_is_valid(accel: &Vector3<f32>, gyro: &Vector3<f32>) -> bool {
    accel.iter().all(|v| v.is_finite()) && gyro.iter().all(|v| v.is_finite())
}

fn usability_tag(reason: UsabilityReason) -> &'static str {
    match reason {
        UsabilityReason::LowFixType => "fix_type<3",
        UsabilityReason::InsufficientSv => "num_sv<min",
        UsabilityReason::HorizontalAccuracyTooLoose => "h_acc>max",
        UsabilityReason::NonFinitePosition => "non-finite",
    }
}

#[embassy_executor::task]
pub async fn estimation_task() {
    let mut imu_sub = crate::subscribe_or_park!(sensors::IMU_1, "IMU_1");
    let gps_signal = &sensors::GPS_NAV_PVT;
    let odom_pub = sensors::VEHICLE_ODOMETRY.immediate_publisher();
    let att_pub = sensors::VEHICLE_ATTITUDE.immediate_publisher();
    let bias_pub = super::ESTIMATOR_BIAS_TELEM
        .publisher()
        .expect("eskf_imu_gps: ESTIMATOR_BIAS_TELEM publisher");

    let cfg = GpsGuardConfig::default();

    // --- Wait for first RTK-fixed PVT to anchor the ENU origin ---
    let (origin, anchor_ms) = loop {
        let pvt = gps_signal.wait().await;
        // The guard's anchor predicate matches the firmware's stricter
        // criteria (carr_soln >= 2). Convert to the guard's GpsFix shape
        // first so we can reuse the predicate without duplicating logic.
        // We don't yet have an LLH origin, so synthesise enu_pos = 0; the
        // predicate doesn't read enu_pos beyond a finiteness check.
        let probe = GpsFix {
            enu_pos: Vector3::zeros(),
            enu_vel: Vector3::zeros(),
            h_acc_mm: pvt.h_acc_mm,
            v_acc_mm: pvt.v_acc_mm,
            s_acc_mm_s: pvt.s_acc_mm_s,
            num_sv: pvt.num_sv,
            fix_type: pvt.fix_type,
            carr_soln: pvt.carr_soln,
            timestamp_ms: pvt.timestamp.as_millis(),
        };
        if is_pvt_origin_anchor(&probe, cfg.gps_min_sv, cfg.gps_h_acc_max_mm) {
            defmt::info!(
                "GPS origin (RTK fixed): lat={} lon={} alt_msl_mm={} num_sv={} h_acc_mm={}",
                pvt.lat_deg,
                pvt.lon_deg,
                pvt.alt_msl_mm,
                pvt.num_sv,
                pvt.h_acc_mm,
            );
            let origin = LlhOrigin::new(
                pvt.lat_deg * PI / 180.0,
                pvt.lon_deg * PI / 180.0,
                pvt.alt_msl_mm as f32 * 1e-3,
            );
            break (origin, pvt.timestamp.as_millis());
        } else if pvt.fix_type >= 3 && pvt.num_sv >= cfg.gps_min_sv && pvt.h_acc_mm <= cfg.gps_h_acc_max_mm {
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
    // EskfConfig defaults are mocap-tuned; widen the position jump gate so
    // healthy GPS frames at 10 m/s × 100 ms aren't hard-rejected.
    let mut eskf = Eskf::new(EskfConfig {
        max_pos_jump_m: GPS_MAX_POS_JUMP_M,
        ..EskfConfig::default()
    });
    eskf.init_with_cov(
        Vector3::zeros(),
        nalgebra::UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
        Vector3::new(0.1, 0.1, cfg.init_yaw_cov),
    );

    let mut guard = EskfGpsGuard::new(cfg, anchor_ms);

    let publish_status = |eskf: &Eskf, guard: &EskfGpsGuard| {
        let pos = eskf.position();
        let vel = eskf.velocity();
        let q = eskf.orientation();
        let gb = eskf.gyro_bias();
        let ab = eskf.accel_bias();
        let (roll, pitch, yaw) = q.euler_angles();
        let s = guard.snapshot();
        let phase = if s.converged {
            EstimatorPhase::Running {
                roll_deg: roll.to_degrees(),
                pitch_deg: pitch.to_degrees(),
                yaw_deg: yaw.to_degrees(),
                pos: [pos.x, pos.y, pos.z],
                vel: [vel.x, vel.y, vel.z],
                gyro_bias: [gb.x, gb.y, gb.z],
                accel_bias: [ab.x, ab.y, ab.z],
                pos_reject_total: s.pos_reject_total,
                att_reject_total: 0,
                pos_inflated_total: s.pos_inflated_total,
                att_inflated_total: 0,
                jump_total: s.jump_total,
                carr_soln: s.last_carr_soln,
                num_sv: s.last_num_sv,
                h_acc_mm: s.last_h_acc_mm,
            }
        } else {
            EstimatorPhase::Converging {
                roll_deg: roll.to_degrees(),
                pitch_deg: pitch.to_degrees(),
                yaw_deg: yaw.to_degrees(),
                pos: [pos.x, pos.y, pos.z],
                vel: [vel.x, vel.y, vel.z],
                gyro_bias: [gb.x, gb.y, gb.z],
                accel_bias: [ab.x, ab.y, ab.z],
                pos_reject_total: s.pos_reject_total,
                att_reject_total: 0,
                pos_inflated_total: s.pos_inflated_total,
                att_inflated_total: 0,
                jump_total: s.jump_total,
                carr_soln: s.last_carr_soln,
                num_sv: s.last_num_sv,
                h_acc_mm: s.last_h_acc_mm,
            }
        };
        ESTIMATOR_STATUS.lock(|c| c.set(phase));
    };

    publish_status(&eskf, &guard);

    // --- Main loop ---
    let mut imu_skip: u32 = 0;
    let mut predict_count: u32 = 0;
    let mut last_predict_ts = Instant::now();
    let mut prev_ready = false;
    let mut prev_stale = false;
    let mut prev_converged = false;
    let mut prev_rtk_quality_ok = false;

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

                // Per-sample attitude observability check — see
                // mocap wrapper for rationale; same machinery.
                let last_nan_reset = ESKF_LAST_NAN_RESET.lock(|c| c.get());
                ATTITUDE_HEALTH.store(
                    attitude_health_bits(
                        &sample.accel_m_s2,
                        &sample.gyro_rad_s,
                        last_nan_reset,
                        sample.timestamp,
                    ),
                    Ordering::Relaxed,
                );

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
                    ESTIMATOR_READY.store(false, Ordering::Release);
                    prev_ready = false;
                    continue;
                }

                let gb = eskf.gyro_bias();
                let ab = eskf.accel_bias();
                super::ESKF_GYRO_BIAS.signal(gb);
                super::ESKF_ACCEL_BIAS.signal(ab);

                predict_count = predict_count.wrapping_add(1);

                // Decimated mirror onto the bias telemetry channel
                // (~10 Hz). Same predict_count counter — no extra
                // state, no race with the Signals above.
                if predict_count.is_multiple_of(BIAS_TELEM_DECIMATION) {
                    bias_pub.publish_immediate(msgs::EstimatorBias {
                        timestamp: Instant::now(),
                        gyro_bias_rad_s: gb,
                        accel_bias_m_s2: ab,
                    });
                }

                if !predict_count.is_multiple_of(ODOM_DECIMATION) {
                    continue;
                }

                let tick = guard.on_predict_tick(&eskf, Instant::now().as_millis());

                // Convergence-flip log: emitted on the first tick where
                // `converged` flipped to true. Mirrors the original
                // L382-389 message.
                let cur_converged = guard.snapshot().converged;
                if !prev_converged && cur_converged {
                    let gb = eskf.gyro_bias();
                    defmt::info!(
                        "ESKF converged: gyro_bias=[{},{},{}] trace_xy={}",
                        gb.x,
                        gb.y,
                        gb.z,
                        eskf.gyro_bias_cov_trace_xy(),
                    );
                }
                prev_converged = cur_converged;

                // Health snapshot + fault evaluation must run before
                // the staleness gate — see eskf_imu_mocap.rs for the
                // rationale (POS_STALE / VEL_STALE flags would freeze
                // at zero if we skipped this path on stale ticks).
                // GPS-only build: last_att_update stays None forever
                // (no attitude measurement source), so ATT_STALE
                // never fires. Same shape as the mocap wrapper
                // otherwise.
                ESKF_HEALTH.lock(|c| c.set(eskf.health()));
                let now_ms = Instant::now();
                let armed = IS_ARMED.load(Ordering::Relaxed);
                let inputs = FaultEvalInputs {
                    now: now_ms,
                    last_pos_update: ESKF_LAST_POS_UPDATE.lock(|c| c.get()),
                    last_vel_update: ESKF_LAST_VEL_UPDATE.lock(|c| c.get()),
                    last_att_update: ESKF_LAST_ATT_UPDATE.lock(|c| c.get()),
                    pos_cov_trace: eskf.pos_cov_trace(),
                    last_nan_reset: ESKF_LAST_NAN_RESET.lock(|c| c.get()),
                    last_jump_cascade: ESKF_LAST_JUMP_CASCADE.lock(|c| c.get()),
                    last_reject_cascade: ESKF_LAST_REJECT_CASCADE.lock(|c| c.get()),
                    armed,
                };
                let (flags, severe) = evaluate_faults(&inputs);
                ESKF_FAULTS.store(flags, Ordering::Relaxed);
                ESKF_DEGRADED.store(flags != 0 && !severe, Ordering::Relaxed);
                ESKF_SEVERE_FAULT.store(severe, Ordering::Relaxed);

                if tick.is_stale {
                    if !prev_stale {
                        defmt::warn!("ESKF: GPS stale — arming blocked");
                    }
                    prev_stale = true;
                    if prev_ready {
                        ESTIMATOR_READY.store(false, Ordering::Release);
                        prev_ready = false;
                    }
                    publish_status(&eskf, &guard);
                    continue;
                }
                prev_stale = false;

                if tick.is_ready != prev_ready {
                    ESTIMATOR_READY.store(tick.is_ready, Ordering::Release);
                    prev_ready = tick.is_ready;
                }

                publish_status(&eskf, &guard);

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

            Either::Second(pvt) => {
                let fix = pvt_to_fix(&pvt, &origin);
                let outcome = guard.on_pvt(&mut eskf, &fix);

                match outcome {
                    GpsGuardOutcome::PvtAccepted {
                        inflated,
                        sigma_pos_m,
                    } => {
                        // PVT accept refreshes pos and vel last-update
                        // clocks (the guard runs vel update in the same
                        // call). last_att_update stays None — GPS
                        // doesn't observe attitude.
                        let now = pvt.timestamp;
                        ESKF_LAST_POS_UPDATE.lock(|c| c.set(Some(now)));
                        ESKF_LAST_VEL_UPDATE.lock(|c| c.set(Some(now)));
                        if inflated {
                            defmt::debug!(
                                "ESKF: GPS pos inflated update sigma_pos={}m",
                                sigma_pos_m,
                            );
                        }
                    }
                    GpsGuardOutcome::PvtRejectedUsability { reason } => {
                        defmt::warn!(
                            "GPS fix dropped ({}): fix_type={} num_sv={} h_acc_mm={}",
                            usability_tag(reason),
                            pvt.fix_type,
                            pvt.num_sv,
                            pvt.h_acc_mm,
                        );
                    }
                    GpsGuardOutcome::PvtRejectedJump { consecutive } => {
                        defmt::error!(
                            "ESKF: GPS pos jump ({}/{})",
                            consecutive,
                            guard.config().max_consecutive_jumps,
                        );
                    }
                    GpsGuardOutcome::PvtRejectedFilter { reason } => {
                        defmt::warn!(
                            "ESKF: GPS pos update rejected ({})",
                            filter_tag(reason),
                        );
                    }
                    GpsGuardOutcome::Reinitialised { at_enu, cause } => {
                        // NaN re-init is a hard fault (covariance went
                        // singular); jump-cascade re-init is a planned
                        // recovery (filter & GPS disagreed too long,
                        // and we trusted GPS). Different log levels
                        // so an operator scanning defmt sees the
                        // severity at a glance.
                        match cause {
                            ReinitCause::NanState => {
                                ESKF_LAST_NAN_RESET.lock(|c| c.set(Some(Instant::now())));
                                defmt::error!(
                                    "ESKF: re-initializing at GPS pos [{},{},{}] (nan-state)",
                                    at_enu.x,
                                    at_enu.y,
                                    at_enu.z,
                                );
                            }
                            ReinitCause::JumpCascade => {
                                defmt::info!(
                                    "ESKF: re-anchoring at GPS pos [{},{},{}] (jump-cascade — \
                                     IMU and GPS diverged past tolerance, snapped to GPS)",
                                    at_enu.x,
                                    at_enu.y,
                                    at_enu.z,
                                );
                            }
                        }
                        // Re-anchor predict clock so the next IMU sample
                        // doesn't generate a huge dt against a stale
                        // last_predict_ts.
                        last_predict_ts = Instant::now();
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                    GpsGuardOutcome::DisarmedJumpCascade { consecutive } => {
                        // Surface the cascade event to the fault
                        // bitfield via a hold-down (see
                        // estimation::CASCADE_HOLD_S). Without this
                        // the live `health` view shows ready=no
                        // faults=0x0000 — the converged drop is
                        // invisible to operator + blackbox.
                        ESKF_LAST_JUMP_CASCADE.lock(|c| c.set(Some(Instant::now())));
                        defmt::error!(
                            "ESKF: {} consecutive GPS jumps — arming blocked",
                            consecutive,
                        );
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                    GpsGuardOutcome::DisarmedRejectCascade { consecutive } => {
                        ESKF_LAST_REJECT_CASCADE.lock(|c| c.set(Some(Instant::now())));
                        defmt::error!(
                            "ESKF: {} consecutive GPS rejections — arming blocked",
                            consecutive,
                        );
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                    GpsGuardOutcome::DisarmedNanState => {
                        defmt::error!("ESKF: non-finite state after GPS update — awaiting re-init");
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                }

                // RTK quality flip logs — diff the snapshot against the
                // wrapper's prev_rtk_quality_ok cache.
                let cur_rtk = guard.snapshot().rtk_quality_ok;
                if cur_rtk != prev_rtk_quality_ok {
                    if cur_rtk {
                        defmt::info!(
                            "ESKF: RTK quality re-acquired (sustained carr_soln=2 for {}ms)",
                            guard.config().rtk_fix_debounce_ms,
                        );
                    } else {
                        defmt::warn!(
                            "ESKF: RTK quality lost (carr_soln={}) — arming blocked",
                            pvt.carr_soln,
                        );
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                    prev_rtk_quality_ok = cur_rtk;
                }
            }
        }
    }
}

fn filter_tag(reason: cybflight_core::eskf::FilterReason) -> &'static str {
    use cybflight_core::eskf::FilterReason as R;
    match reason {
        R::InverseFailed => "inv-fail",
        R::InflationCapExceeded => "cap-exceeded",
        R::NaNAfterUpdate => "nan",
        R::NotInitialized => "not-init",
        R::Other => "other",
    }
}
