//! Motion-capture / INS state estimation task — the firmware-side
//! wrapper around `cybflight_core::eskf::EskfMocapGuard`.
//!
//! Fuses IMU and motion capture pose (~100–360 Hz) into a 15-state
//! Error-State Kalman Filter (ESKF) and publishes `VehicleOdometry`.
//!
//! Cross-cutting failsafe state (jump cascade, reject cascade,
//! NaN-after-update, staleness, gyro-bias-cov convergence) lives in
//! `EskfMocapGuard` — the same shape as `EskfGpsGuard`, both
//! composing the shared `EskfFailsafe`. This task is the
//! Embassy-side I/O shell: it owns the `Eskf`, subscribes to IMU and
//! mocap, runs the predict step on the IMU branch, hands incoming
//! poses to the guard, translates `MocapGuardOutcome` into defmt
//! logs, and publishes odometry / attitude / `ESTIMATOR_READY`.
//!
//! # State machine
//! 1. **Init** — waits for first mocap pose, initialises ESKF.
//! 2. **Converging** — predict/update loop is active; gyro-bias
//!    covariance is still above the convergence threshold. Arming
//!    is blocked.
//! 3. **Running** — covariance has converged; `ESTIMATOR_READY` is
//!    set and arming is permitted.

use embassy_futures::select::{Either, select};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::Vector3;

use cybflight_core::eskf::{
    Eskf, EskfMocapGuard, MocapGuardOutcome, MocapPose,
    ReinitCause,
};

use core::sync::atomic::Ordering;

use crate::estimation::{
    attitude_health_bits, evaluate_faults, AttitudeHealthParams, ATTITUDE_HEALTH, ESKF_DEGRADED, ESKF_FAULTS,
    ESKF_HEALTH, ESKF_LAST_ATT_UPDATE, ESKF_LAST_JUMP_CASCADE, ESKF_LAST_NAN_RESET,
    ESKF_LAST_POS_UPDATE, ESKF_LAST_REJECT_CASCADE, ESKF_LAST_VEL_UPDATE, ESKF_SEVERE_FAULT,
    ESTIMATOR_READY, ESTIMATOR_STATUS, EstimatorPhase, FaultEvalInputs,
};
use crate::motors::IS_ARMED;
use crate::sensors;
use cybflight_msgs as msgs;

/// ESKF predict rate after decimation — IMU rate → ~1 kHz predict in every
/// build (8 kHz → 8, 1 kHz `imu_1khz` → 1). Predict dt is measured from
/// sample timestamps, so the divisor only sets the rate, not the model.
const PREDICT_DECIMATION: u32 = {
    let d = (crate::rates::IMU_ODR_HZ / 1000.0) as u32;
    if d == 0 { 1 } else { d }
};

/// Odometry publish decimation relative to predict rate. 1 kHz / 1 = 1 kHz.
const ODOM_DECIMATION: u32 = 1;

fn imu_is_valid(accel: &Vector3<f32>, gyro: &Vector3<f32>) -> bool {
    accel.iter().all(|v| v.is_finite()) && gyro.iter().all(|v| v.is_finite())
}

fn pose_to_mocap(pose: &msgs::ViconPose) -> MocapPose {
    MocapPose {
        position: pose.position,
        orientation: pose.orientation,
        timestamp_ms: pose.timestamp.as_millis(),
    }
}

fn outcome_tag(reason: cybflight_core::eskf::FilterReason) -> &'static str {
    use cybflight_core::eskf::FilterReason as R;
    match reason {
        R::InverseFailed => "inv-fail",
        R::InflationCapExceeded => "cap-exceeded",
        R::NaNAfterUpdate => "nan",
        R::NotInitialized => "not-init",
        R::Other => "other",
    }
}

#[embassy_executor::task]
pub async fn estimation_task() {
    let mut imu_sub = crate::subscribe_or_park!(sensors::IMU_1, "IMU_1");
    let mut mocap_sub = crate::subscribe_or_park!(sensors::VICON_POSE, "VICON_POSE");
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
        .expect("eskf_imu_mocap: ESTIMATOR_BIAS_TELEM publisher");

    // Guard tuning is fully parameterised (`eskf.mocap_guard`, reboot-
    // flagged): the pose measurement σ, staleness window, convergence
    // threshold and the failsafe cascade limits all come from the
    // vehicle YAML / flash overrides rather than code defaults.
    let cfg = crate::params::get().eskf.mocap_guard.to_mocap_guard_config();

    // Hot-path param snapshots. Both source groups are reboot-flagged,
    // and both of these feed code that runs at the IMU / predict rate —
    // reading them inline would clone the whole FirmwareConfig inside a
    // critical section thousands of times a second.
    let att_health_params = AttitudeHealthParams::snapshot();
    let fault_params = super::fault_params();

    // --- Wait for first mocap pose ---
    let first_pose = loop {
        match mocap_sub.next_message().await {
            WaitResult::Message(p) => break p,
            WaitResult::Lagged(_) => continue,
        }
    };

    // --- Initialise ESKF from mocap pose, zero biases ---
    // Filter tuning in full from `eskf.filter` (vehicle YAML / flash
    // overrides) — noise densities *and* the structural jump gates.
    let mut eskf = Eskf::new(crate::params::get().eskf.filter.to_eskf_config());
    // Read once: the predict path runs at ~1 kHz and `params::get()`
    // deep-clones the config inside a critical section.
    let max_predict_dt_s = super::max_predict_dt_s(&crate::params::get());
    eskf.init(
        first_pose.position,
        first_pose.orientation,
        Vector3::zeros(),
        Vector3::zeros(),
    );

    defmt::info!(
        "ESKF init: pos=[{},{},{}]",
        first_pose.position.x,
        first_pose.position.y,
        first_pose.position.z,
    );

    let anchor_ms = first_pose.timestamp.as_millis();
    let mut guard = EskfMocapGuard::new(cfg, anchor_ms);

    let publish_status = |eskf: &Eskf, guard: &EskfMocapGuard| {
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
                att_reject_total: s.att_reject_total,
                pos_inflated_total: s.pos_inflated_total,
                att_inflated_total: s.att_inflated_total,
                jump_total: s.jump_total,
                // Mocap path has no GNSS — fields stay at zero;
                // consumers should interpret carr_soln=0 here as
                // "not applicable".
                carr_soln: 0,
                num_sv: 0,
                h_acc_mm: 0,
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
                att_reject_total: s.att_reject_total,
                pos_inflated_total: s.pos_inflated_total,
                att_inflated_total: s.att_inflated_total,
                jump_total: s.jump_total,
                carr_soln: 0,
                num_sv: 0,
                h_acc_mm: 0,
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

                if !imu_is_valid(&sample.accel_m_s2, &sample.gyro_rad_s) {
                    defmt::warn!("estimation: non-finite IMU sample, skipping");
                    continue;
                }

                // Per-sample attitude observability check. Drives the
                // arm-gate's att_health requirement and the blackbox
                // ATTITUDE_HEALTH stream. Cheap (norm + bit ops) and
                // independent of ESKF state, so we run it before the
                // initialised-skip below.
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
                // sample can legitimately be OLDER than `last_predict_ts`.
                // While this task waits for the first pose, the IMU_1
                // subscriber accumulates a full CAP-deep backlog of stale
                // samples; `last_predict_ts` is seeded with `Instant::now()`
                // at init, so the first queued sample sits up to CAP
                // sample-periods in its past. `duration_since` panics on
                // that underflow (`checked_sub(...).unwrap()`), which
                // parked the core and IWDG-rebooted the board on every
                // vicon-stream start — but only on `imu_1khz` builds:
                // at 8 kHz PREDICT_DECIMATION=8 consumed the ≤4-deep
                // backlog in the `imu_skip` path before any timestamp
                // reached this subtraction, masking the bug. Saturated,
                // a stale sample yields dt = 0 and the `dt <= 0.0` guard
                // below skips it; the clock still advances to `now`, so
                // the first fresh sample computes a sane dt.
                let dt = now.saturating_duration_since(last_predict_ts).as_micros() as f32
                    / 1_000_000.0;
                last_predict_ts = now;

                if dt <= 0.0 || dt > max_predict_dt_s {
                    continue;
                }

                eskf.predict(sample.accel_m_s2, sample.gyro_rad_s, dt);

                if !eskf.is_initialized() {
                    defmt::error!(
                        "ESKF: non-finite state after predict — awaiting re-init from mocap"
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

                let cur_converged = guard.snapshot().converged;
                if !prev_converged && cur_converged {
                    let gb = eskf.gyro_bias();
                    defmt::info!(
                        "ESKF converged: gyro_bias=[{},{},{}] trace={}",
                        gb.x,
                        gb.y,
                        gb.z,
                        eskf.gyro_bias_cov_trace(),
                    );
                }
                prev_converged = cur_converged;

                // Health snapshot + fault evaluation must run **before**
                // the staleness gate — staleness is exactly what
                // evaluate_faults turns into POS_STALE / ATT_STALE flags,
                // and skipping the call when tick.is_stale=true would
                // freeze ESKF_FAULTS at its last pre-stale value (zero
                // for a clean takeoff). Counter snapshots run too: gate
                // rejections can accumulate during the stale window
                // (filter still ticks predict on IMU; corruption can
                // still trigger NaN-after-update), and we want them
                // visible to the live `health` shell verb and to
                // /health BB records throughout.
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

                // Staleness gates *arming and annunciation*, not the
                // state stream. `EskfMocapGuard`'s stated policy is to
                // ride out the gap on IMU dead-reckoning (mocap is
                // 100–360 Hz, so the window is O(100 ms) — well inside
                // the IMU's drift budget). Publishing has to follow the
                // same policy, because withholding odometry cascades:
                // >50 ms of silence starves the outer loop's
                // ODOM_STALE_TIMEOUT, which stops RATE_COMMAND, which
                // trips INDI's CMD_STALE_TIMEOUT, which lets the DShot
                // MOTOR_CMD_STALE watchdog force idle throttle. That put
                // the motors at idle ~110 ms into a routine marker
                // occlusion, then disarmed ~500 ms later — half a second
                // of armed free-fall from a dropout the filter was
                // designed to absorb.
                //
                // A *sustained* outage is still caught, but by the fault
                // path (POS_STALE → eskf_fault_pos_timeout_s → severe →
                // failsafe disarm), which is a deliberate, annunciated
                // action rather than a silent thrust cut.
                if tick.is_stale != prev_stale {
                    if tick.is_stale {
                        defmt::warn!(
                            "ESKF: mocap stale — arming blocked, dead-reckoning on IMU"
                        );
                    } else {
                        defmt::info!("ESKF: mocap stream recovered");
                    }
                    prev_stale = tick.is_stale;
                }

                // `is_ready` already collapses to false while stale —
                // `EskfMocapGuard::on_predict_tick` clears `converged`
                // on the stale edge — so the arming gate closes here
                // without a separate branch.
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

            Either::Second(result) => {
                let pose = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(n) => {
                        // Same annunciation as the IMU branch: dropping
                        // pose frames silently hides the one condition
                        // that matters most on a mocap build.
                        defmt::warn!("estimation: dropped {} mocap poses", n);
                        continue;
                    }
                };

                let mocap_pose = pose_to_mocap(&pose);
                // `armed` gates the guard's re-anchor escape hatch —
                // in flight the ride-out-IMU policy stays absolute.
                let outcome = guard.on_pose(
                    &mut eskf,
                    &mocap_pose,
                    IS_ARMED.load(Ordering::Relaxed),
                );

                match outcome {
                    MocapGuardOutcome::Accepted { inflated } => {
                        if inflated {
                            defmt::debug!("ESKF: mocap pose update inflated");
                        }
                        // Pose accept refreshes the pos, att *and* vel
                        // last-update clocks. Mocap is a joint pos+att
                        // measurement, and velocity is observable
                        // through the pose sequence, so all three are
                        // genuinely fresh. The fault evaluator reads
                        // these to decide POS_STALE / ATT_STALE /
                        // VEL_STALE — note this means VEL_STALE can
                        // never assert on a mocap build, by design.
                        let now = pose.timestamp;
                        ESKF_LAST_POS_UPDATE.lock(|c| c.set(Some(now)));
                        ESKF_LAST_ATT_UPDATE.lock(|c| c.set(Some(now)));
                        ESKF_LAST_VEL_UPDATE.lock(|c| c.set(Some(now)));
                    }
                    MocapGuardOutcome::NonFinitePose => {
                        defmt::warn!("estimation: non-finite mocap frame, rejecting");
                    }
                    MocapGuardOutcome::RejectedJump { consecutive } => {
                        defmt::error!(
                            "ESKF: pose jump ({}/{})",
                            consecutive,
                            guard.config().max_consecutive_jumps,
                        );
                    }
                    MocapGuardOutcome::RejectedFilter { reason } => {
                        defmt::warn!("ESKF: pose update rejected ({})", outcome_tag(reason));
                    }
                    MocapGuardOutcome::Reinitialised { at_pos, cause, .. } => {
                        // Both re-init causes start a hold-down window
                        // for the ATTITUDE_HEALTH NO_RECENT_NAN bit.
                        ESKF_LAST_NAN_RESET.lock(|c| c.set(Some(Instant::now())));
                        match cause {
                            ReinitCause::NanState => {
                                defmt::error!(
                                    "ESKF: re-initializing from mocap after NaN reset \
                                     pos=[{},{},{}]",
                                    at_pos.x,
                                    at_pos.y,
                                    at_pos.z,
                                );
                            }
                            ReinitCause::JumpCascade => {
                                // Disarmed-only escape hatch: the pose
                                // stream agreed with itself for
                                // REANCHOR_FRAMES while the filter was
                                // wedged outside the jump gate. Logged
                                // at error level because it means the
                                // filter had genuinely lost the
                                // airframe, not merely dropped frames.
                                defmt::error!(
                                    "ESKF: re-anchoring from mocap (jump-cascade, disarmed) \
                                     pos=[{},{},{}]",
                                    at_pos.x,
                                    at_pos.y,
                                    at_pos.z,
                                );
                            }
                        }
                        last_predict_ts = Instant::now();
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                    MocapGuardOutcome::DisarmedJumpCascade { consecutive } => {
                        // Record the cascade event so evaluate_faults
                        // surfaces GUARD_JUMP_CASCADE for CASCADE_HOLD_S.
                        // Without this the live `health` view shows
                        // ready=no faults=0x0000 — the converged drop
                        // is invisible.
                        ESKF_LAST_JUMP_CASCADE.lock(|c| c.set(Some(Instant::now())));
                        defmt::error!(
                            "ESKF: {} consecutive pose jumps — arming blocked",
                            consecutive,
                        );
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                    MocapGuardOutcome::DisarmedRejectCascade { consecutive } => {
                        ESKF_LAST_REJECT_CASCADE.lock(|c| c.set(Some(Instant::now())));
                        defmt::error!(
                            "ESKF: {} consecutive mocap rejections — arming blocked",
                            consecutive,
                        );
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                    MocapGuardOutcome::DisarmedNanState => {
                        defmt::error!(
                            "ESKF: non-finite state after mocap update — awaiting re-init"
                        );
                        if prev_ready {
                            ESTIMATOR_READY.store(false, Ordering::Release);
                            prev_ready = false;
                        }
                    }
                }
            }
        }
    }
}
