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
    Eskf, EskfConfig, EskfMocapGuard, MocapGuardConfig, MocapGuardOutcome, MocapPose,
    ReinitCause,
};

use core::sync::atomic::Ordering;

use crate::estimation::{
    attitude_health_bits, evaluate_faults, ATTITUDE_HEALTH, ESKF_DEGRADED, ESKF_FAULTS,
    ESKF_HEALTH, ESKF_LAST_ATT_UPDATE, ESKF_LAST_JUMP_CASCADE, ESKF_LAST_NAN_RESET,
    ESKF_LAST_POS_UPDATE, ESKF_LAST_REJECT_CASCADE, ESKF_LAST_VEL_UPDATE, ESKF_SEVERE_FAULT,
    ESTIMATOR_READY, ESTIMATOR_STATUS, EstimatorPhase, FaultEvalInputs,
};
use crate::motors::IS_ARMED;
use crate::sensors;
use cybflight_msgs as msgs;

/// ESKF predict rate after decimation. 8 kHz IMU / 8 = 1 kHz.
const PREDICT_DECIMATION: u32 = 8;

/// Odometry publish decimation relative to predict rate. 1 kHz / 1 = 1 kHz.
const ODOM_DECIMATION: u32 = 1;

/// Bias telemetry decimation relative to predict rate. 1 kHz / 100 = 10 Hz.
/// Biases drift at seconds-scale; a slower channel keeps blackbox
/// bandwidth bounded without losing meaningful information.
const BIAS_TELEM_DECIMATION: u32 = 100;

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
    let att_pub = sensors::VEHICLE_ATTITUDE.immediate_publisher();
    let bias_pub = super::ESTIMATOR_BIAS_TELEM
        .publisher()
        .expect("eskf_imu_mocap: ESTIMATOR_BIAS_TELEM publisher");

    let cfg = MocapGuardConfig::default();

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
                        defmt::warn!("ESKF: mocap stale — arming blocked");
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

            Either::Second(result) => {
                let pose = match result {
                    WaitResult::Message(m) => m,
                    WaitResult::Lagged(_) => continue,
                };

                let mocap_pose = pose_to_mocap(&pose);
                let outcome = guard.on_pose(&mut eskf, &mocap_pose);

                match outcome {
                    MocapGuardOutcome::Accepted { inflated } => {
                        if inflated {
                            defmt::debug!("ESKF: mocap pose update inflated");
                        }
                        // Pose accept refreshes both pos and att
                        // last-update clocks (mocap is a joint pos+att
                        // measurement). The fault evaluator reads these
                        // to decide POS_STALE / ATT_STALE.
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
                        // NaN re-init starts a hold-down window for the
                        // ATTITUDE_HEALTH NO_RECENT_NAN bit. JumpCascade
                        // re-init isn't surfaced by mocap (policy: ride
                        // out IMU), but if a future variant emits it the
                        // same hold-down applies.
                        ESKF_LAST_NAN_RESET.lock(|c| c.set(Some(Instant::now())));
                        // Mocap re-init only happens on NaN-state
                        // entry (no jump-cascade re-init policy on
                        // this path — see EskfMocapGuard rationale).
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
                                // Mocap shouldn't surface this; if a
                                // future variant adds a jump-cascade
                                // re-init policy, log it here.
                                defmt::error!(
                                    "ESKF: re-anchoring from mocap (jump-cascade) \
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
