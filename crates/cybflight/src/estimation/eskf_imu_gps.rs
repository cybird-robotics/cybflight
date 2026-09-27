//! GNSS/INS state estimation task — the firmware-side wrapper around
//! `cybflight_core::eskf::EskfGpsGuard`.
//!
//! ## ENU frame anchoring
//!
//! Two modes, selected by the vehicle YAML's optional `origin:` section:
//!
//! - **Fixed origin** (`origin:` declared): the frame is that geodetic
//!   point, identically on every flight. Use this when ENU coordinates
//!   must mean the same physical place each time — mission waypoints are
//!   expressed in ENU, so a per-flight origin flies the same mission in a
//!   different spot. The first RTK-fixed PVT then only seeds *where the
//!   vehicle is* inside that frame.
//! - **Runtime anchor** (no `origin:`): the first **RTK-fixed** PVT
//!   (NAV-PVT `carr_soln == 2`) defines the frame, so it is
//!   centimeter-accurate from the start. Float-RTK and stand-alone fixes
//!   are not used as the anchor — their meter-scale absolute bias would
//!   bake a permanent offset into every subsequent setpoint.
//!
//! Either way the task waits for an RTK-fixed PVT before initialising:
//! with a fixed origin that fix sets the seed position, and seeding from
//! a biased stand-alone fix would start the estimate metres off and hand
//! the absolute jump gate a large innovation once RTK converges.
//! Subsequent fixes (any usable PVT, including post-anchor drops to
//! float or stand-alone) feed the guard's position update.
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

use embassy_futures::select::{Either3, select3};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::Vector3;

use cybflight_core::eskf::{
    Eskf, EskfGpsGuard, GpsFix, GpsGuardOutcome, ReinitCause, UsabilityReason,
    is_pvt_origin_anchor,
};
use cybflight_core::geodetic::{LlhOrigin, ned_to_enu};

use core::f64::consts::PI;
use core::sync::atomic::Ordering;

use crate::estimation::{
    attitude_health_bits, evaluate_faults, AttitudeHealthParams, ATTITUDE_HEALTH, ESKF_DEGRADED, ESKF_FAULTS,
    ESKF_HEALTH, ESKF_LAST_ATT_UPDATE, ESKF_LAST_JUMP_CASCADE, ESKF_LAST_NAN_RESET,
    ESKF_LAST_POS_UPDATE, ESKF_LAST_REJECT_CASCADE, ESKF_LAST_VEL_UPDATE, ESKF_SEVERE_FAULT,
    ESTIMATOR_READY, ESTIMATOR_STATUS, EstimatorPhase, FaultEvalInputs,
};
use crate::motors::IS_ARMED;
use crate::sensors;
use crate::sensors::gps::GpsNavPvt;
use cybflight_msgs as msgs;

/// Whether this build's GPS driver can produce a dual-antenna heading at
/// all: the `gps_unicore` feature selects the UM982 driver, and the ublox
/// driver never signals `GPS_HEADING`, so on those builds the fusion
/// sites are unreachable regardless.
pub const GPS_HAS_HEADING: bool = cfg!(feature = "gps_unicore");

/// Whether ANT2 is actually populated on this airframe (vehicle YAML
/// `build: gps_dual_antenna`). Distinct from [`GPS_HAS_HEADING`]: the
/// receiver may be heading-*capable* while the install has one antenna.
///
/// Both are compile-time because both are immutable facts about the
/// hardware, and the vehicle-yaml loader rejects the inconsistent pairing
/// (`yes` without `gps_model: unicore`) at build time — so the `&&` below
/// can no longer be false for a reason worth logging at boot.
pub const GPS_DUAL_ANTENNA: bool = cfg!(feature = "gps_dual_antenna");

/// ESKF predict rate after decimation — IMU rate → ~1 kHz predict in every
/// build (8 kHz → 8, 1 kHz `imu_1khz` → 1). Predict dt is measured from
/// sample timestamps, so the divisor only sets the rate, not the model.
const PREDICT_DECIMATION: u32 = {
    let d = (crate::rates::IMU_ODR_HZ / 1000.0) as u32;
    if d == 0 { 1 } else { d }
};

/// Odometry publish decimation relative to predict rate. 1 kHz / 1 = 1 kHz.
const ODOM_DECIMATION: u32 = 1;

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

/// World-frame (ENU) unit direction of the ANT1→ANT2 baseline, reconstructed
/// from a dual-antenna heading sample (`heading_rad` = azimuth CW from True
/// North, `pitch_rad` = baseline elevation). Components are (E, N, U). Shared
/// by the runtime heading update and the init yaw-seed so they cannot drift.
fn heading_to_world_baseline(h: &sensors::gps::GpsHeading) -> Vector3<f32> {
    let (sh, ch) = libm::sincosf(h.heading_rad);
    let (sp, cp) = libm::sincosf(h.pitch_rad);
    Vector3::new(cp * sh, cp * ch, sp)
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
    let heading_signal = &sensors::GPS_HEADING;
    let odom_pub = sensors::VEHICLE_ODOMETRY.immediate_publisher();
    // Decimated mirror for the blackbox (`sensors::BLACKBOX_ODOMETRY`):
    // thinned here, at the publisher, so the recorder's channel buffers
    // records instead of samples it is about to discard. See
    // `rates::BLACKBOX_ODOM_DECIM`.
    let bb_odom_pub = sensors::BLACKBOX_ODOMETRY.immediate_publisher();
    let mut bb_odom_ctr: u32 = 0;
    let att_pub = sensors::VEHICLE_ATTITUDE.immediate_publisher();
    let bias_pub = super::ESTIMATOR_BIAS_TELEM
        .publisher()
        .expect("eskf_imu_gps: ESTIMATOR_BIAS_TELEM publisher");

    // GPS velocity fusion is estimator policy, so it stays a runtime param
    // (reboot-flagged — read once at task start, before the origin-anchor
    // wait so the guard config is consistent for the whole task lifetime).
    // Heading fusion is decided entirely at compile time by two hardware
    // facts the vehicle YAML declares under `build:`.
    let fusion_params = crate::params::get().sensors;
    let fuse_gps_heading = GPS_DUAL_ANTENNA && GPS_HAS_HEADING;

    // Guard tuning is fully parameterised (`eskf.gps_guard`, reboot-
    // flagged): thresholds, σ floors, debounce windows and the failsafe
    // cascade limits all come from the vehicle YAML / flash overrides.
    // `fuse_velocity` is channel selection, so it stays in `sensors`.
    let guard_params = crate::params::get().eskf.gps_guard;
    let cfg = guard_params.to_gps_guard_config(fusion_params.gps_fuse_vel);

    // Hot-path param snapshots — see the mocap task for the rationale.
    // Both source groups are reboot-flagged.
    let att_health_params = AttitudeHealthParams::snapshot();
    let fault_params = super::fault_params();

    // A vehicle YAML may declare a fixed geodetic anchor. When present it
    // *defines* the ENU frame and the first usable RTK-fixed PVT only
    // seeds the vehicle's position within that frame; when absent the
    // anchor PVT defines the frame, as before.
    let baked_origin = crate::vehicle::BAKED_ORIGIN.map(|o| {
        defmt::info!(
            "GPS origin (fixed, from vehicle YAML): lat={} lon={} alt_msl_m={}",
            o.lat_deg,
            o.lon_deg,
            o.alt_msl_m,
        );
        LlhOrigin::new(o.lat_deg * PI / 180.0, o.lon_deg * PI / 180.0, o.alt_msl_m)
    });

    // --- Wait for the first RTK-fixed PVT ---
    //
    // Required either way. Without a baked origin it *defines* the ENU
    // frame, so a float or stand-alone fix would bake metre-scale bias
    // into every subsequent setpoint. With a baked origin the frame is
    // already fixed, but this PVT still sets where the filter believes
    // it starts inside that frame — seeding from a stand-alone fix would
    // start the estimate metres off and hand the jump gate a large
    // innovation as RTK later converges.
    let (origin, anchor_ms, seed_pvt) = loop {
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
            let origin = match baked_origin {
                Some(o) => {
                    defmt::info!(
                        "GPS seed fix (RTK fixed): lat={} lon={} alt_msl_mm={} num_sv={} \
                         h_acc_mm={} — anchoring to the YAML origin",
                        pvt.lat_deg,
                        pvt.lon_deg,
                        pvt.alt_msl_mm,
                        pvt.num_sv,
                        pvt.h_acc_mm,
                    );
                    o
                }
                None => {
                    defmt::info!(
                        "GPS origin (RTK fixed): lat={} lon={} alt_msl_mm={} num_sv={} h_acc_mm={}",
                        pvt.lat_deg,
                        pvt.lon_deg,
                        pvt.alt_msl_mm,
                        pvt.num_sv,
                        pvt.h_acc_mm,
                    );
                    LlhOrigin::new(
                        pvt.lat_deg * PI / 180.0,
                        pvt.lon_deg * PI / 180.0,
                        pvt.alt_msl_mm as f32 * 1e-3,
                    )
                }
            };
            break (origin, pvt.timestamp.as_millis(), pvt);
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


    // Seed position: the vehicle's ENU location inside the anchored frame.
    //
    // With a runtime anchor this is exactly zero by construction (the
    // origin *is* this fix). With a YAML origin the vehicle is wherever
    // it happens to be relative to that fixed point, so initialising at
    // zero would place the estimate at the origin and hand the very
    // first PVT update an innovation equal to the true offset — which
    // the absolute jump gate (`eskf_max_pos_jump_m`) would reject,
    // cascading straight to a disarm.
    let seed_enu = origin.llh_to_enu(
        seed_pvt.lat_deg * PI / 180.0,
        seed_pvt.lon_deg * PI / 180.0,
        seed_pvt.alt_msl_mm as f32 * 1e-3,
    );
    if baked_origin.is_some() {
        defmt::info!(
            "ESKF seed position in fixed frame: [{},{},{}] m from origin",
            seed_enu.x,
            seed_enu.y,
            seed_enu.z,
        );
    }

    // --- Initialise ESKF at the seed position, identity orientation, zero biases ---
    // Filter tuning in full from `eskf.filter` (vehicle YAML / flash
    // overrides) — noise densities *and* the structural gates. Note
    // `eskf_max_pos_jump_m` is position-source coupled: the schema
    // default (1 m) is mocap-sized, so a GPS vehicle must pin ~3 m in
    // its YAML or healthy fast-flight frames get hard-rejected. The
    // bake warns when it is left unpinned (HW_COUPLED_KEYS).
    let filter_params = crate::params::get().eskf.filter;
    let mut eskf = Eskf::new(filter_params.to_eskf_config());
    // Read once: the predict path runs at ~1 kHz and `params::get()`
    // deep-clones the config inside a critical section.
    let max_predict_dt_s = super::max_predict_dt_s(&crate::params::get());
    // Roll/pitch initial variance is source-independent (both observable
    // from gravity at bootstrap) and lives with the filter; yaw's is
    // GPS-specific — a GPS-only build cannot observe it at all — so it
    // comes from the guard.
    eskf.init_with_cov(
        seed_enu,
        nalgebra::UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
        Vector3::new(
            filter_params.init_att_var_rp,
            filter_params.init_att_var_rp,
            cfg.init_yaw_cov,
        ),
    );

    // Antenna geometry (body/FLU frame, IMU at the origin): ANT1 (position
    // antenna) lever arm and the ANT1->ANT2 baseline direction. Per-install
    // extrinsics from `airframe.install` (vehicle YAML / flash overrides).
    let sp = crate::params::get().airframe.install;
    let r_b = Vector3::from(sp.gps_ant1_offset_m);
    let b_body = Vector3::from(sp.gps_baseline_body);
    // Floor on the dual-antenna heading 1-σ [rad]: an 8 cm baseline can't
    // honestly do better than the default ~1.4°, so don't over-trust an
    // optimistic reported σ. Baseline-length coupled → parameterised.
    let heading_sigma_floor: f32 = guard_params.heading_sigma_floor_rad;

    // Seed initial yaw from the dual-antenna heading if one is already locked at
    // anchor time. `init_with_cov` left the orientation at identity with a large
    // yaw cov (`init_yaw_cov`); one baseline update against that near-flat prior
    // snaps yaw to the measured heading — same model as the runtime fusion, so
    // the filter (and any heading-dependent control) starts on the true
    // heading instead of yaw=0. Best-effort: with no usable, fresh heading the
    // identity prior stands and yaw converges later through the loop's updates.
    if fuse_gps_heading {
        if let Some(h) = sensors::gps::LATEST_GPS_HEADING.lock(|c| c.get()) {
            let fresh =
                Instant::now().duration_since(h.timestamp).as_millis() < cfg.gps_stale_ms;
            if h.carr_soln >= 1 && fresh {
                let b_world = heading_to_world_baseline(&h);
                let sigma_h = h.heading_sigma_rad.max(heading_sigma_floor);
                let sigma_p = h.pitch_sigma_rad.max(heading_sigma_floor);
                if eskf.update_baseline(b_world, b_body, sigma_h, sigma_p).is_accepted() {
                    defmt::info!(
                        "ESKF yaw seeded from dual-antenna heading: {} deg (carr_soln={})",
                        eskf.orientation().euler_angles().2.to_degrees(),
                        h.carr_soln,
                    );
                } else {
                    defmt::warn!("ESKF yaw seed rejected; starting at identity yaw");
                }
            }
        }
    }

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
        match select3(imu_sub.next_message(), gps_signal.wait(), heading_signal.wait()).await {
            Either3::First(result) => {
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
                        &att_health_params,
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
                // `saturating_duration_since`, NOT `duration_since`: the
                // IMU_1 backlog queued while this task waited for its first
                // fix is older than the `Instant::now()` that seeded
                // `last_predict_ts`, and `duration_since` panics on that
                // underflow. Same boot-loop bug as the mocap task (see the
                // comment there); at 1 kHz PREDICT_DECIMATION=1 feeds the
                // first stale sample straight into this subtraction.
                // Saturated, stale samples yield dt = 0 and the guard
                // below skips them.
                let dt = now.saturating_duration_since(last_predict_ts).as_micros() as f32
                    / 1_000_000.0;
                last_predict_ts = now;

                if dt <= 0.0 || dt > max_predict_dt_s {
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
                if predict_count.is_multiple_of(super::BIAS_TELEM_DECIMATION) {
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
                    fault_params,
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
                let odom_msg = msgs::VehicleOdometry {
                    // The epoch the state is valid at: the predict propagated
                    // it to this IMU sample. `Instant::now()` here would add
                    // this task's scheduling latency to the label — measured
                    // sigma 5.0 ms against the stamper's 0.01 ms (flight_0009).
                    // `twist.angular` below is already taken from `sample`.
                    timestamp: sample.timestamp,
                    pose: msgs::Pose {
                        position: eskf.position(),
                        orientation: q,
                    },
                    twist: msgs::Twist {
                        linear: eskf.velocity(),
                        angular: sample.gyro_rad_s - gb,
                    },
                };
                odom_pub.publish_immediate(odom_msg.clone());
                bb_odom_ctr += 1;
                if bb_odom_ctr >= crate::rates::BLACKBOX_ODOM_DECIM {
                    bb_odom_ctr = 0;
                    bb_odom_pub.publish_immediate(odom_msg);
                }
                att_pub.publish_immediate(msgs::VehicleAttitude {
                    timestamp: now_publish,
                    orientation: q,
                });
            }

            Either3::Second(pvt) => {
                let mut fix = pvt_to_fix(&pvt, &origin);
                // Antenna lever arm: the receiver solves the ANT1 phase centre,
                // but the ESKF state is the IMU. Correct position to the IMU
                // using the current attitude. (No velocity ω×r term — velocity
                // is not fused; see the tombstone in cybflight_core eskf.rs.)
                let q_now = eskf.orientation();
                fix.enu_pos -= q_now * r_b;
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

            Either3::Third(h) => {
                // Dual-antenna heading -> yaw aiding. Fuse whenever the moving
                // baseline has a solution: float (carr_soln 1) OR fixed (2); skip
                // only carr_soln 0 (none). The 8 cm baseline rarely integer-fixes
                // in practice, but a float heading is still very usable — and we
                // keep the SAME covariance (the receiver's reported float sigma is
                // already larger, so the 2-axis R down-weights it on its own).
                // Vector/direction update: valid at all attitudes, no yaw singularity.
                if fuse_gps_heading && h.carr_soln >= 1 {
                    let b_world = heading_to_world_baseline(&h); // ENU (E,N,U)
                    let sigma_h = h.heading_sigma_rad.max(heading_sigma_floor);
                    let sigma_p = h.pitch_sigma_rad.max(heading_sigma_floor);
                    // Opportunistic yaw aid. We deliberately do NOT set
                    // ESKF_LAST_ATT_UPDATE: the moving-baseline heading is
                    // *expected* to drop out (carr_soln 0) through high-rate acro,
                    // and feeding that into the ATT_STALE fault would false-trip on
                    // every flip. Yaw coasts on the gyro between solutions.
                    let _ = eskf.update_baseline(b_world, b_body, sigma_h, sigma_p);
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
