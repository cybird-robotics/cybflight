//! RC interpreter: maps raw RC channels to a control setpoint.
//!
//! Which setpoint is published depends on the active control mode feature:
//!
//! | Feature            | Publishes to                | Semantics                      |
//! |--------------------|-----------------------------|--------------------------------|
//! | `outer_rate`       | `RATE_COMMAND`              | Stick → body rate + thrust     |
//! | `outer_geometric`  | `ACTIVE_POSITION_SETPOINT`  | Stick → ENU position (RMW)     |
//! | `outer_mpc`        | `ACTIVE_POSITION_SETPOINT`  | Stick → ENU position (RMW)     |

use embassy_time::Instant;

use crate::sensors::RC_INPUT;

// ── RATE MODE ────────────────────────────────────────────────────────────────

#[cfg(feature = "outer_rate")]
use cybflight_core::rc::rc_mapping::ChannelCalibration as RateChannelCalibration;

#[cfg(feature = "outer_rate")]
#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    use crate::vehicle::QUADROTOR_BODY;

    let mut rc_sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");

    let pitch_cal = RateChannelCalibration::centered(1);
    let roll_cal = RateChannelCalibration::centered(0);
    let throttle_cal = RateChannelCalibration::throttle(2);
    let yaw_cal = RateChannelCalibration::centered(3);

    let min_channels: u8 = {
        let mut m = pitch_cal
            .index
            .max(roll_cal.index)
            .max(throttle_cal.index)
            .max(yaw_cal.index);
        m += 1;
        m as u8
    };

    /// Max body rate for roll/pitch [rad/s] (~460 deg/s).
    const MAX_RATE_RP: f32 = 8.0;
    /// Max body rate for yaw [rad/s] (~230 deg/s).
    const MAX_RATE_YAW: f32 = 4.0;
    /// Stick deadband for rate axes (normalized).
    const RATE_DEADBAND: f32 = 0.05;
    /// Stick deadband for throttle (normalized).
    const THROTTLE_DEADBAND: f32 = 0.05;

    const LEARN_TOGGLE_CHANNEL: usize = 6;
    const LEARNER_PREARM_CHANNEL: usize = 7;
    const SWITCH_THRESHOLD: u16 = 1500;

    let hover_thrust_n = QUADROTOR_BODY.mass_kg * 9.81;

    defmt::info!("RC interpreter: rate mode started");

    loop {
        let mut rc = rc_sub.next_message_pure().await;
        while let Some(newer) = rc_sub.try_next_message_pure() {
            rc = newer;
        }

        if rc.channel_count < min_channels {
            continue;
        }

        // Learner switches
        let learn_on = rc.channel_count > LEARN_TOGGLE_CHANNEL as u8
            && rc.channels[LEARN_TOGGLE_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNING_ENABLED.store(learn_on, core::sync::atomic::Ordering::Release);

        let prearm_on = rc.channel_count > LEARNER_PREARM_CHANNEL as u8
            && rc.channels[LEARNER_PREARM_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNER_PREARM.store(prearm_on, core::sync::atomic::Ordering::Release);

        // Map sticks to body rates
        let roll_norm = roll_cal.normalize(rc.channels[roll_cal.index] as i16);
        let pitch_norm = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16);
        let yaw_norm = yaw_cal.normalize(rc.channels[yaw_cal.index] as i16);
        let throttle_norm = throttle_cal.normalize(rc.channels[throttle_cal.index] as i16);

        // Apply deadband
        let roll_cmd = apply_deadband(roll_norm, RATE_DEADBAND);
        let pitch_cmd = apply_deadband(pitch_norm, RATE_DEADBAND);
        let yaw_cmd = apply_deadband(yaw_norm, RATE_DEADBAND);
        let throttle_cmd = if throttle_norm < THROTTLE_DEADBAND {
            0.0
        } else {
            throttle_norm
        };

        let rate_ref = nalgebra::Vector3::new(
            roll_cmd * MAX_RATE_RP,
            pitch_cmd * MAX_RATE_RP,
            yaw_cmd * MAX_RATE_YAW,
        );
        // Throttle 0→1 maps to 0→2×hover thrust (mid-stick ≈ hover).
        let collective_thrust_n = throttle_cmd * 2.0 * hover_thrust_n;

        super::RATE_COMMAND.signal(cybflight_msgs::AttitudeControlSetpoint {
            timestamp: Instant::now(),
            collective_thrust_n,
            attitude_quaternion: nalgebra::UnitQuaternion::identity(),
            body_rate_rad_s: rate_ref,
            torque_n_m: nalgebra::Vector3::zeros(),
        });
    }
}

#[cfg(feature = "outer_rate")]
#[inline]
fn apply_deadband(s: f32, deadband: f32) -> f32 {
    let a = s.abs();
    if a <= deadband {
        0.0
    } else {
        let scaled = (a - deadband) / (1.0 - deadband);
        let scaled = if scaled > 1.0 { 1.0 } else { scaled };
        if s >= 0.0 { scaled } else { -scaled }
    }
}

// ── POSITION MODE ────────────────────────────────────────────────────────────

#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
use crate::sensors::VEHICLE_ODOMETRY;
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
use cybflight_core::rc::rc_mapping::ChannelCalibration;

// ── Incremental stick tuning ─────────────────────────────────────────────
//
// Rationale: an **absolute** stick-to-setpoint mapping (stick → origin + offset)
// fights any controller that wants to hold a setpoint elsewhere — e.g. the
// mission abort path that captures the drone's current pose as the hover
// target. An absolute mapping would immediately override that capture with
// `origin + current_stick_offset` on the next RC frame, snapping the drone
// back to stick-dictated coordinates.
//
// Instead, sticks command **rates**: a deflected XY stick moves the target
// position at a bounded rate; a centered stick holds the target still.
// This makes the setpoint a first-class piece of flight state that the
// mission planner, abort path, and RC pilot can all safely write.
/// Max rate at which XY sticks drift the target position [m/s].
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const XY_RATE_M_PER_S: f32 = 1.0;
/// Max rate at which the throttle stick drifts the target altitude [m/s].
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const Z_RATE_M_PER_S: f32 = 0.5;
/// Dead-band around XY stick center below which we treat input as zero.
/// Normalized in [0, 1]. Kept tight because pilots actively command
/// horizontal motion and forgive small residual offsets.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const XY_DEADBAND: f32 = 0.05;
/// Dead-band around throttle mid-stick. Wider than [`XY_DEADBAND`]
/// because altitude *hold* — not motion — is the default intent at
/// center, and typical transmitter spring slop / trim drift pushes
/// the centered throttle ±0.02..0.08 off true mid. A 0.12 band
/// (~±60 µs around 1500) comfortably swallows that slop so the pilot
/// can release the stick and the altitude target stops integrating.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const THROTTLE_DEADBAND: f32 = 0.12;
/// Throttle µs threshold: below this, descend toward the ground at
/// [`LAND_RATE_M_PER_S`] regardless of integrator state. With the
/// default 988/1500/2012 calibration, 1100 sits comfortably above the
/// stick's absolute minimum (988) but well below hover (1500).
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const THROTTLE_LAND_US: u16 = 1100;
/// Constant descent rate while `is_landing` holds [m/s]. Replaces the
/// previous `pos.z = 0.0` step, which handed the MPC a 2 m reference
/// jump and triggered near-free-fall descent (min-thrust = 10 % hover
/// ⇒ `a_down ≈ 8.8 m/s²`) with 200 : 1 MPC pos/vel weights — an
/// effective hard-landing. A rate-limited integrator caps descent at a
/// known, survivable velocity regardless of starting altitude.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const LAND_RATE_M_PER_S: f32 = 0.4;
/// Upper bound on the per-frame integration step. Guards against RC frame
/// gaps (e.g. transient link hiccups) producing huge single-step drifts.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const MAX_FRAME_DT_S: f32 = 0.1;

/// Indoor (mocap) safety envelope: `|target.x| ≤ X_ENVELOPE_M`,
/// `|target.y| ≤ Y_ENVELOPE_M`, and `0 ≤ target.z ≤ Z_CEILING_M` in the
/// world (ENU) frame. Sized for the lab arena — asymmetric because the
/// arena is longer along Y than X. Only compiled in `est_pos_mocap`
/// builds; outdoor (`est_pos_gps`) flight defines its envelope through
/// waypoint planning, not a fixed box.
#[cfg(all(
    feature = "est_pos_mocap",
    any(feature = "outer_geometric", feature = "outer_mpc")
))]
const X_ENVELOPE_M: f32 = 2.5;
#[cfg(all(
    feature = "est_pos_mocap",
    any(feature = "outer_geometric", feature = "outer_mpc")
))]
const Y_ENVELOPE_M: f32 = 3.5;
#[cfg(all(
    feature = "est_pos_mocap",
    any(feature = "outer_geometric", feature = "outer_mpc")
))]
const Z_CEILING_M: f32 = 2.0;

/// Clamp `pos` to the indoor envelope. No-op for non-mocap builds.
///
/// Defense-in-depth against stick drift, mocap glitches that survive the
/// estimator's rejection gate, or a drone that was physically moved before
/// re-arm. Without this, an Idle-state stick drift integrates without bound
/// and can push the setpoint outside the mocap volume → loss of pose
/// updates → estimator open-loop on IMU → crash.
#[cfg(all(
    feature = "est_pos_mocap",
    any(feature = "outer_geometric", feature = "outer_mpc")
))]
#[inline]
fn clamp_indoor_envelope(pos: &mut nalgebra::Vector3<f32>) {
    pos.x = pos.x.clamp(-X_ENVELOPE_M, X_ENVELOPE_M);
    pos.y = pos.y.clamp(-Y_ENVELOPE_M, Y_ENVELOPE_M);
    pos.z = pos.z.clamp(0.0, Z_CEILING_M);
}

#[cfg(all(
    not(feature = "est_pos_mocap"),
    any(feature = "outer_geometric", feature = "outer_mpc")
))]
#[inline]
fn clamp_indoor_envelope(_pos: &mut nalgebra::Vector3<f32>) {}

/// Apply a symmetric dead-band to a centered ±1 stick reading and rescale
/// so the output still saturates at ±1 at full stick deflection.
///
/// Without this, sensor noise near stick center would continuously drift
/// the incremental target position. With a naive zero-if-below-band
/// mapping you also get a "step" at the band edge; rescaling smooths it.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
#[inline]
fn apply_deadband_centered(s: f32, deadband: f32) -> f32 {
    let a = s.abs();
    if a <= deadband {
        0.0
    } else {
        let scaled = (a - deadband) / (1.0 - deadband);
        let scaled = if scaled > 1.0 { 1.0 } else { scaled };
        if s >= 0.0 { scaled } else { -scaled }
    }
}

#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    let mut rc_sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");
    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("rc_interpreter: VEHICLE_ODOMETRY subscriber");

    let pitch_cal = ChannelCalibration::centered(1);
    let roll_cal = ChannelCalibration::centered(0);
    let throttle_cal = ChannelCalibration::throttle(2);

    // Minimum `channel_count` we require before we are willing to trust the
    // primary axes — guards against degenerate/truncated RC frames whose
    // remaining positions would otherwise read as `u16::default() = 0`,
    // producing `normalize(0) ≈ -1` on the centered sticks (i.e. full
    // negative XY drift with the pilot's sticks centered!).
    let min_channels: u8 = {
        let mut m = pitch_cal.index.max(roll_cal.index).max(throttle_cal.index);
        m += 1;
        m as u8
    };

    // Phase 1: Wait for ESKF convergence, then capture origin from converged
    // odometry. `ESTIMATOR_READY` signals *covariance* convergence but makes
    // no guarantee about finiteness of any specific sample — we've seen
    // pathological ESKF initializations emit one NaN component on the first
    // post-ready tick. Loop until we see a fully finite position; if ESKF
    // never stabilizes, the failsafe controller watchdog will eventually
    // disarm the vehicle on its own.
    while !crate::estimation::ESTIMATOR_READY.load(core::sync::atomic::Ordering::Acquire) {
        embassy_time::Timer::after_millis(100).await;
    }
    let origin = loop {
        let odom = odom_sub.next_message_pure().await;
        let mut p = odom.pose.position;
        if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
            // Indoor builds: clamp to the lab envelope so the initial
            // setpoint is in-bounds even if the drone was placed near the
            // arena edge. No-op outdoors.
            clamp_indoor_envelope(&mut p);
            break p;
        }
        defmt::warn!("rc_interpreter: discarding non-finite odometry during origin capture");
    };

    // Seed the shared setpoint cell. This is the one and only init write —
    // consumers (outer_loop, indi_task, cascade_task, mission_planner) are
    // waiting on `ACTIVE_SETPOINT_READY` and will wake as soon as we
    // signal it immediately below.
    super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
        cell.set(Some(super::ActiveSetpoint {
            timestamp: Instant::now(),
            position: origin,
            yaw_rad: 0.0,
        }));
    });
    super::ACTIVE_SETPOINT_READY.signal(());

    // Phase 2: Incremental stick → position target.
    //
    // Every Idle-state RC frame becomes a read-modify-write on
    // `ACTIVE_POSITION_SETPOINT`: read the current setpoint, add the
    // stick Δ for this frame, indoor-clamp to the lab envelope, write
    // back. The write happens **every** tick (not just on motion
    // exceeding a threshold) so the cell's `timestamp` field is a real
    // liveness proof for downstream consumers — centered sticks produce
    // zero Δ after deadband, making the write idempotent apart from the
    // stamp. The envelope clamp is a no-op on outdoor (`est_pos_gps`)
    // builds.
    let mut last_frame_time = Instant::now();
    let mut was_armed = false;
    let mut origin = origin;

    /// RC channel index for the learning toggle switch (0-indexed).
    /// Channel 6 (7th channel). >1500 µs = learning data collection active.
    const LEARN_TOGGLE_CHANNEL: usize = 6;
    /// RC channel index for the learner prearm switch (0-indexed).
    /// Channel 7 (8th channel). >1500 µs = learner prearm active.
    const LEARNER_PREARM_CHANNEL: usize = 7;
    /// RC channel index for the mission trigger (0-indexed).
    /// Channel 4 = AUX1 (5th transmitter channel, typical for switches).
    /// Rising edge → request a mission plan (only honored while Idle).
    /// Falling edge while a mission is Planning/Executing → abort.
    #[cfg(feature = "outer_mpc")]
    const MISSION_TRIGGER_CHANNEL: usize = 4;
    const SWITCH_THRESHOLD: u16 = 1500;

    // Mission-trigger Schmitt trigger. The 400 µs band between LOW and
    // HIGH is wider than any physical switch's noise floor, so hysteresis
    // alone rejects spurious flips. No frame-count debounce: the earlier
    // 3-frame counter counted task *observations* (the backlog-collapse
    // loop means this task may see fewer frames than the RC link sends),
    // making trigger timing depend on executor scheduling rather than
    // switch physics.
    #[cfg(feature = "outer_mpc")]
    const MISSION_SWITCH_HIGH: u16 = 1700;
    #[cfg(feature = "outer_mpc")]
    const MISSION_SWITCH_LOW: u16 = 1300;

    // `None` = not yet synced (first RC frame seeds the confirmed level,
    // so a switch-high at boot does NOT register as a rising edge and
    // spuriously fire a plan). After sync, holds the current hysteretic level.
    #[cfg(feature = "outer_mpc")]
    let mut mission_level_confirmed: Option<bool> = None;

    loop {
        // Wait for any RC frame, then drain the subscriber to its latest
        // value. If a backlog accumulated (executor stall, RC-loss
        // recovery flushing a buffer), integrating each queued frame in
        // sequence would cause the integrator to apply stale stick
        // readings at real-time dt — effectively fast-forwarding the
        // target. We only care about the pilot's *current* intent.
        let mut rc = rc_sub.next_message_pure().await;
        while let Some(newer) = rc_sub.try_next_message_pure() {
            rc = newer;
        }

        // Guard: degenerate frames with fewer than the primary-axis count
        // get dropped. Their stick slots would read as u16::default() = 0,
        // which `normalize` maps to ≈ -1 on centered sticks → runaway.
        // Failsafe still sees frames arriving (no watchdog trip), so we
        // don't mask the protocol layer either way.
        if rc.channel_count < min_channels {
            defmt::warn!(
                "rc_interpreter: short RC frame ({} < {} channels), dropping",
                rc.channel_count,
                min_channels
            );
            continue;
        }

        // ── Arm transition: re-capture origin from current odometry ────
        //
        // After a disarm→arm cycle the drone may have been physically
        // moved. Reset the position setpoint to the current pose so the
        // outer loop doesn't snap to the stale pre-disarm target. On
        // indoor builds the captured origin is also clamped to the lab
        // envelope so the first setpoint is always in-bounds.
        let armed = crate::motors::IS_ARMED.load(core::sync::atomic::Ordering::Acquire);
        if armed && !was_armed {
            // Drain to latest valid odometry for the new origin.
            let mut new_origin = None;
            while let Some(o) = odom_sub.try_next_message_pure() {
                let p = o.pose.position;
                if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
                    new_origin = Some(p);
                }
            }
            if let Some(mut pos) = new_origin {
                clamp_indoor_envelope(&mut pos);
                origin = pos;
                super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
                    cell.set(Some(super::ActiveSetpoint {
                        timestamp: Instant::now(),
                        position: origin,
                        yaw_rad: 0.0,
                    }));
                });
                defmt::info!(
                    "rc_interpreter: arm transition — origin reset to [{},{},{}]",
                    origin.x,
                    origin.y,
                    origin.z,
                );
            } else {
                defmt::warn!(
                    "rc_interpreter: arm transition — no fresh odometry, keeping previous origin"
                );
            }
        }
        was_armed = armed;

        // Learning toggle: channel 6 > 1500 → enable RLS data collection
        let learn_on = rc.channel_count > LEARN_TOGGLE_CHANNEL as u8
            && rc.channels[LEARN_TOGGLE_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNING_ENABLED.store(learn_on, core::sync::atomic::Ordering::Release);

        // Learner prearm: channel 7 > 1500 → configure next arm for learning
        let prearm_on = rc.channel_count > LEARNER_PREARM_CHANNEL as u8
            && rc.channels[LEARNER_PREARM_CHANNEL] > SWITCH_THRESHOLD;
        super::LEARNER_PREARM.store(prearm_on, core::sync::atomic::Ordering::Release);

        // ── Mission trigger & stick-gate ──────────────────────────────
        //
        // The mission planner runs only under `outer_mpc + est_eskf`.
        // We derive the current mission state here from the shared atomic
        // so that, when a mission is active, stick updates do NOT write
        // `ACTIVE_POSITION_SETPOINT` — `outer_loop` is the authorized
        // writer during Executing (trajectory sample at τ₀ each tick).
        #[cfg(feature = "outer_mpc")]
        let mission_active = {
            use core::sync::atomic::Ordering;

            // Per-frame raw level with Schmitt-trigger hysteresis against
            // the last confirmed level. If the AUX channel is absent from
            // this frame (short frame), hold the confirmed level rather
            // than treating the missing channel as LOW.
            let ch_present = rc.channel_count > MISSION_TRIGGER_CHANNEL as u8;
            let raw_level: Option<bool> = if !ch_present {
                mission_level_confirmed
            } else {
                let v = rc.channels[MISSION_TRIGGER_CHANNEL];
                Some(match mission_level_confirmed {
                    Some(true) => v > MISSION_SWITCH_LOW,
                    Some(false) => v > MISSION_SWITCH_HIGH,
                    None => v > MISSION_SWITCH_HIGH,
                })
            };

            let state = super::MissionState::from_u8(super::MISSION_STATE.load(Ordering::Acquire));

            // Schmitt-hysteretic edge detection. First-frame sync seeds
            // the confirmed level without emitting an edge; any subsequent
            // disagreement with the hysteretic raw level fires immediately.
            let (rising, falling) = match (mission_level_confirmed, raw_level) {
                (None, Some(l)) => {
                    mission_level_confirmed = Some(l);
                    (false, false)
                }
                (Some(c), Some(r)) if c == r => (false, false),
                (Some(c), Some(r)) => {
                    mission_level_confirmed = Some(r);
                    (r && !c, !r && c)
                }
                (_, None) => (false, false),
            };

            // Rising edge → only honored when Idle AND armed. Requiring
            // IS_ARMED prevents a disarmed-on-the-bench plan that would
            // snap into action the instant the pilot arms.
            if rising {
                let armed = crate::motors::IS_ARMED.load(Ordering::Acquire);
                if state == super::MissionState::Idle && armed {
                    defmt::info!(
                        "RC: mission trigger (ch{} rising, armed) — requesting plan",
                        MISSION_TRIGGER_CHANNEL
                    );
                    super::PLAN_REQUEST.signal(());
                } else {
                    defmt::warn!(
                        "RC: mission trigger ignored (state={}, armed={})",
                        state as u8,
                        armed
                    );
                }
            }

            // Falling edge while a mission is non-Idle → request graceful abort.
            //
            // Both Planning and Executing honor the abort:
            //   - Executing: outer_loop consumes the flag, captures the
            //     current trajectory ref as the hover point, clears slot,
            //     flips state to Idle.
            //   - Planning: mission_planner observes the flag at its
            //     post-solve check (or inside its publish lock) and
            //     discards the result, flipping state to Idle.
            //
            // We DO NOT directly clear state or slot here: that is the
            // consumer's responsibility so it can capture the drone's
            // current pose as the hover fallback point (safety: prevents
            // snap-back to stale pre-mission `pos_setpoint`).
            if falling && state != super::MissionState::Idle {
                defmt::warn!(
                    "RC: mission abort requested (ch{} falling, state={})",
                    MISSION_TRIGGER_CHANNEL,
                    state as u8
                );
                super::MISSION_ABORT_REQUESTED.store(true, Ordering::Release);
                // No last_target poisoning needed: because sticks are now
                // incremental, centered sticks after the abort leave the
                // target wherever the outer_loop captured it (current pose).
            }

            state != super::MissionState::Idle
        };
        #[cfg(not(feature = "outer_mpc"))]
        let mission_active = false;

        // ── dt integration step ───────────────────────────────────────
        // Bound dt to avoid one-shot giant drifts on the first frame after
        // a long pause (e.g. RC-loss recovery).
        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).as_micros() as f32 * 1e-6;
        let dt = dt.clamp(0.0, MAX_FRAME_DT_S);
        last_frame_time = now;

        // ── Mission gate ──────────────────────────────────────────────
        //
        // While a mission owns the controller (state ≠ Idle), `outer_loop`
        // is the authorized writer of `ACTIVE_POSITION_SETPOINT` — it
        // writes the trajectory sample at τ₀ on every tick. We **skip**
        // any write here so there is never more than one writer at a time.
        //
        // The smooth-pickup invariant ("stick control resumes from the
        // trajectory endpoint after the mission") is now a structural
        // guarantee: on the natural/abort transition, outer_loop writes
        // the final trajectory reference to `ACTIVE_POSITION_SETPOINT`
        // **before** storing `MissionState::Idle`. Our next Idle tick
        // therefore reads that value and integrates from it — no local
        // `target` mirror needed.
        if mission_active {
            continue;
        }

        // ── Stick integration (Idle, pilot in control) ────────────────
        //
        // Read current setpoint → apply stick Δ → indoor envelope-clamp
        // → write back, all under one mutex acquisition. Deadbanded
        // sticks → zero Δ → idempotent write apart from the timestamp
        // refresh (which is the liveness proof consumers rely on). The
        // envelope clamp is a no-op on outdoor builds.
        //
        // `normalize` returns ±1 for centered channels and [0,1] for
        // throttle. Conventional stick convention:
        //     pitch forward → drone forward (world +X)
        //     roll right    → drone right   (world +Y)   (assumes yaw≈0)
        let sx_norm = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16);
        let sy_norm = roll_cal.normalize(rc.channels[roll_cal.index] as i16);
        let sx = apply_deadband_centered(sx_norm, XY_DEADBAND);
        let sy = apply_deadband_centered(sy_norm, XY_DEADBAND);

        let thr_us = rc.channels[throttle_cal.index];
        let is_landing = thr_us < THROTTLE_LAND_US;
        let sz = if is_landing {
            // Handled below as an unconditional z = 0 override.
            0.0
        } else {
            let thr_norm = throttle_cal.normalize(thr_us as i16);
            let thr_cmd = (thr_norm - 0.5) * 2.0; // [-1, 1]
            apply_deadband_centered(thr_cmd, THROTTLE_DEADBAND)
        };

        super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
            // Seed guaranteed by Phase 1's init write above; if somehow
            // absent, restart from origin rather than panic.
            let cur = cell.get().unwrap_or(super::ActiveSetpoint {
                timestamp: now,
                position: origin,
                yaw_rad: 0.0,
            });
            let mut pos = cur.position;
            pos.x += sx * XY_RATE_M_PER_S * dt;
            pos.y -= sy * XY_RATE_M_PER_S * dt;
            if is_landing {
                // Rate-limited descent toward the ground. The reference
                // decreases by at most `LAND_RATE_M_PER_S · dt` per RC
                // frame, so the MPC always sees a feasible target within
                // its 1 s horizon and never has to track a 2 m step. The
                // `.max(0.0)` anchor stops integration at ground level.
                pos.z = (pos.z - LAND_RATE_M_PER_S * dt).max(0.0);
            } else {
                pos.z += sz * Z_RATE_M_PER_S * dt;
            }

            // Indoor builds: clamp the integrated setpoint into the lab
            // envelope so a sustained stick deflection cannot drift the
            // target outside the mocap volume. No-op outdoors.
            clamp_indoor_envelope(&mut pos);

            cell.set(Some(super::ActiveSetpoint {
                timestamp: now,
                position: pos,
                yaw_rad: cur.yaw_rad,
            }));
        });
    }
}
