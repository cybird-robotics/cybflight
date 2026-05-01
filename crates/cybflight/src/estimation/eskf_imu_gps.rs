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

/// Position σ floor at carr_soln=2 (RTK fixed) [m]. u-blox can report
/// sub-cm accuracy on a stationary receiver with good geometry; the
/// floor prevents the filter from running with unrealistic confidence
/// in flight where multipath / ionospheric residuals dominate.
const GPS_POS_SIGMA_FLOOR_FIX_M: f32 = 0.05;

/// Position σ floor at carr_soln=1 (RTK float) [m]. Float-RTK biases
/// are dm-scale, an order of magnitude looser than fixed; the floor
/// must follow or the gate over-weights biased measurements.
const GPS_POS_SIGMA_FLOOR_FLOAT_M: f32 = 0.30;

/// Position σ floor at carr_soln=0 (stand-alone) [m]. Sub-meter is
/// optimistic for a stand-alone u-blox in flight; this floor keeps
/// the filter from snapping to a m-scale-biased fix as if it were
/// truth, while still letting healthy degraded fixes pull the estimate
/// gently toward the GPS-frame solution.
const GPS_POS_SIGMA_FLOOR_NONE_M: f32 = 2.0;

#[allow(dead_code)] // Reserved for when update_vel is wired up.
const GPS_VEL_SIGMA_FLOOR_M_S: f32 = 0.10;

/// Carrier-solution loss debounce. A single bad PVT (transient
/// reflection / SV geometry blip) shouldn't drop ESTIMATOR_READY; we
/// require carr_soln<2 to persist this long before declaring RTK lost.
/// Sized to be comfortably longer than a single NAV-PVT period (100–200
/// ms) but shorter than the GPS_STALE failsafe.
const RTK_LOSS_DEBOUNCE: Duration = Duration::from_millis(1000);

/// Carrier-solution acquisition debounce. After RTK loss, require
/// sustained carr_soln=2 for this long before re-arming. Catches the
/// fix-flicker pattern where a marginal geometry oscillates between
/// float and fixed every few frames — we don't want to bounce
/// ESTIMATOR_READY in lockstep.
const RTK_FIX_DEBOUNCE: Duration = Duration::from_millis(2000);

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

/// Gyro-bias x+y covariance threshold for convergence. Sums only the two
/// observable axes — yaw bias is unobservable in a GPS-only build (no
/// attitude or course-of-motion update wired in), so its variance never
/// decreases and the 3-axis trace would never cross a sensible threshold.
/// Initial 2-axis sum = 2 × 0.01 = 0.02; this requires ~10× reduction.
const GYRO_BIAS_COV_TRACE_XY_THRESH: f32 = 0.002;

/// Position-jump gate threshold [m] for the GPS path. Looser than the
/// `EskfConfig` default (1 m, sized for mocap) because at 5–10 Hz NAV-PVT
/// the legitimate inter-frame residual at 10 m/s flight speed is ~1 m by
/// itself; a 1 m gate would hard-reject healthy fast-flight measurements.
/// 3 m still catches RTK carrier-ambiguity loss / multipath wraparound,
/// which jump tens of metres.
const GPS_MAX_POS_JUMP_M: f32 = 3.0;

/// Initial yaw covariance [rad²]. GPS-only init has no yaw measurement,
/// so the orientation diagonal must reflect the genuine ignorance about
/// heading. ~π² ≈ 10 covers the full ±π wrap; the filter relies on
/// gravity-aided tilt for roll/pitch (those stay at 0.1) and waits for a
/// future yaw observable (course-of-motion / mag) before claiming
/// convergence on the heading channel.
const INIT_YAW_COV: f32 = 10.0;

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

/// σ floor for a position update, scaled by carrier-solution quality.
/// At fix the floor is cm-scale; at float and stand-alone it widens by
/// an order of magnitude each, so the filter doesn't over-weight a
/// known-biased measurement just because u-blox's internal h_acc number
/// is small.
fn pos_sigma_floor(pvt: &GpsNavPvt) -> f32 {
    match pvt.carr_soln {
        2 => GPS_POS_SIGMA_FLOOR_FIX_M,
        1 => GPS_POS_SIGMA_FLOOR_FLOAT_M,
        _ => GPS_POS_SIGMA_FLOOR_NONE_M,
    }
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
    // EskfConfig defaults are mocap-tuned; widen the position jump gate so
    // healthy GPS frames at 10 m/s × 100 ms aren't hard-rejected, and seed
    // the orientation covariance with high yaw uncertainty since GPS gives
    // us no heading measurement.
    let mut eskf = Eskf::new(EskfConfig {
        max_pos_jump_m: GPS_MAX_POS_JUMP_M,
        ..EskfConfig::default()
    });
    eskf.init_with_cov(
        Vector3::zeros(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
        Vector3::new(0.1, 0.1, INIT_YAW_COV),
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

    // The origin-anchor PVT was carr_soln=2 by definition (`pvt_is_origin_anchor`).
    // Track it as the last-known RTK quality state.
    let mut last_carr_soln: u8 = 2;
    let mut last_num_sv: u8 = 0;
    let mut last_h_acc_mm: u32 = 0;

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
            carr_soln: last_carr_soln,
            num_sv: last_num_sv,
            h_acc_mm: last_h_acc_mm,
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

    // RTK quality gate state. We just anchored on a carr_soln=2 frame,
    // but require RTK_FIX_DEBOUNCE of sustained good fixes before flipping
    // `rtk_quality_ok` to true — the anchor PVT on its own isn't proof of
    // a stable carrier solution. Until then, ESTIMATOR_READY stays low
    // even if `converged` flips early.
    let mut rtk_quality_ok = false;
    let mut rtk_fix_streak: Option<Instant> = Some(Instant::now());
    let mut rtk_loss_streak: Option<Instant> = None;

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

                    if !converged
                        && eskf.gyro_bias_cov_trace_xy() < GYRO_BIAS_COV_TRACE_XY_THRESH
                    {
                        converged = true;
                        defmt::info!(
                            "ESKF converged: gyro_bias=[{},{},{}] trace_xy={}",
                            gyro_bias[0],
                            gyro_bias[1],
                            gyro_bias[2],
                            eskf.gyro_bias_cov_trace_xy(),
                        );
                    }
                    // ESTIMATOR_READY = converged AND rtk_quality_ok. The
                    // RTK gate is updated in the GPS branch; here we just
                    // re-publish the AND every odom tick so the flag tracks
                    // whichever side flipped most recently.
                    ESTIMATOR_READY
                        .store(converged && rtk_quality_ok, Ordering::Release);

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
                            carr_soln: last_carr_soln,
                            num_sv: last_num_sv,
                            h_acc_mm: last_h_acc_mm,
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
                            carr_soln: last_carr_soln,
                            num_sv: last_num_sv,
                            h_acc_mm: last_h_acc_mm,
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

                // Snapshot RTK-quality fields for telemetry (always reflects
                // the most recent usable PVT — including degraded fixes, so
                // operators can see the system *is* tracking even if it has
                // dropped out of RTK).
                last_carr_soln = pvt.carr_soln;
                last_num_sv = pvt.num_sv;
                last_h_acc_mm = pvt.h_acc_mm;

                // RTK-quality streak tracking. Incoming PVT either extends
                // the current streak or starts the opposite one. The
                // debounced quality flag is checked further down so the
                // ESTIMATOR_READY transition has a single re-evaluation
                // point this iteration.
                if pvt.carr_soln >= 2 {
                    rtk_loss_streak = None;
                    if rtk_fix_streak.is_none() {
                        rtk_fix_streak = Some(Instant::now());
                    }
                } else {
                    rtk_fix_streak = None;
                    if rtk_loss_streak.is_none() {
                        rtk_loss_streak = Some(Instant::now());
                    }
                }

                let (enu_pos, _enu_vel) = pvt_enu(&pvt, &origin);

                if !eskf.is_initialized() {
                    // Mid-flight NaN re-init: seed at the *current* GPS-derived
                    // ENU position, not at origin. The LLH origin was anchored
                    // at boot — re-anchoring at origin here would teleport the
                    // estimate back to launch site and the controller would
                    // fly the vehicle home through a wall.
                    defmt::error!(
                        "ESKF: re-initializing at current GPS pos [{},{},{}] after NaN reset",
                        enu_pos.x,
                        enu_pos.y,
                        enu_pos.z,
                    );
                    eskf.init_with_cov(
                        enu_pos,
                        UnitQuaternion::identity(),
                        Vector3::zeros(),
                        Vector3::zeros(),
                        Vector3::new(0.1, 0.1, INIT_YAW_COV),
                    );
                    converged = false;
                    consecutive_rejects = 0;
                    consecutive_jumps = 0;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                    last_predict_ts = Instant::now();
                    last_gps_ts = Instant::now();
                    continue;
                }

                // Derive σ from u-blox's reported accuracy with a
                // carr_soln-scaled floor. The floor widens by ~10× per
                // step down (fix → float → stand-alone) so a degraded
                // fix doesn't get the cm-scale weighting that only
                // RTK-fixed deserves.
                let sigma_pos = (pvt.h_acc_mm.max(pvt.v_acc_mm) as f32 * 1e-3)
                    .max(pos_sigma_floor(&pvt));

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
                    consecutive_jumps = 0;
                    ESTIMATOR_READY.store(false, Ordering::Release);
                }

                // Evaluate the debounced RTK quality gate now that the
                // streak timestamps are up to date. Two-direction transition:
                //   ok=false → ok=true on RTK_FIX_DEBOUNCE of carr_soln>=2.
                //   ok=true  → ok=false on RTK_LOSS_DEBOUNCE of carr_soln<2.
                // ESTIMATOR_READY is the AND of `converged` and `rtk_quality_ok`,
                // so flipping either is enough to drop the flag; flipping both
                // back is required to re-arm.
                let now = Instant::now();
                if rtk_quality_ok {
                    if let Some(t) = rtk_loss_streak
                        && now.duration_since(t) > RTK_LOSS_DEBOUNCE
                    {
                        rtk_quality_ok = false;
                        ESTIMATOR_READY.store(false, Ordering::Release);
                        defmt::warn!(
                            "ESKF: RTK quality lost (carr_soln={}) — arming blocked",
                            pvt.carr_soln,
                        );
                    }
                } else if let Some(t) = rtk_fix_streak
                    && now.duration_since(t) > RTK_FIX_DEBOUNCE
                {
                    rtk_quality_ok = true;
                    ESTIMATOR_READY
                        .store(converged && rtk_quality_ok, Ordering::Release);
                    defmt::info!(
                        "ESKF: RTK quality re-acquired (sustained carr_soln=2 for {}ms)",
                        RTK_FIX_DEBOUNCE.as_millis(),
                    );
                }
            }
        }
    }
}
