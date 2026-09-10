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
// Only the position-mode task builds a Duration (the landing leash's
// odometry trust horizon).
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
use embassy_time::Duration;

use crate::sensors::RC_INPUT;

// ── RATE MODE ────────────────────────────────────────────────────────────────

#[cfg(feature = "outer_rate")]
use cybflight_core::rc::rc_mapping::ChannelCalibration as RateChannelCalibration;
#[cfg(feature = "outer_rate")]
use cybflight_core::rc::rc_mapping::StickEndpoints as RateStickEndpoints;

#[cfg(feature = "outer_rate")]
#[embassy_executor::task]
pub async fn rc_interpreter_task() {

    let mut rc_sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");

    // Mirror of `RATE_COMMAND` for the blackbox recorder. INDI consumes
    // the Signal via `try_take()`; we build the setpoint once and
    // dispatch to both sinks so the logger doesn't race the inner loop.
    let ctrl_sp_pub = super::CONTROL_SETPOINT_TELEM
        .publisher()
        .expect("rc_interpreter: CONTROL_SETPOINT_TELEM publisher");

    // Stick travel comes from the `rc` group (`rc_min_us` / `rc_mid_us` /
    // `rc_max_us`), not from the constructor defaults: the endpoints are
    // a property of the transmitter's servo travel, and a radio that
    // does not use the standard 988/1500/2012 would otherwise mis-scale
    // every axis no matter what the vehicle YAML pinned.
    let endpoints = {
        let p = crate::params::get();
        RateStickEndpoints {
            min_us: p.rc.min_us as i16,
            mid_us: p.rc.mid_us as i16,
            max_us: p.rc.max_us as i16,
        }
    };
    let pitch_cal = RateChannelCalibration::centered_with(1, endpoints);
    let roll_cal = RateChannelCalibration::centered_with(0, endpoints);
    let throttle_cal = RateChannelCalibration::throttle_with(2, endpoints);
    let yaw_cal = RateChannelCalibration::centered_with(3, endpoints);

    let min_channels: u8 = {
        let mut m = pitch_cal
            .index
            .max(roll_cal.index)
            .max(throttle_cal.index)
            .max(yaw_cal.index);
        m += 1;
        m as u8
    };

    // Stick scaling and deadbands from the `rc` group (reboot-flagged,
    // so read once here). In rate mode the throttle is a direct thrust
    // axis and shares the stick deadband with the rate axes; the wider
    // `rc_throttle_deadband` applies only to position mode, where
    // throttle is a mid-stick-centred *velocity* command and has to
    // absorb the TX's spring slop.
    let params_snapshot = crate::params::get();
    let max_rate_rp = params_snapshot.rc.max_rate_rp_rad_s;
    let max_rate_yaw = params_snapshot.rc.max_rate_yaw_rad_s;
    let rate_deadband = params_snapshot.rc.rate_deadband;
    let throttle_deadband = rate_deadband;

    let hover_thrust_n =
        params_snapshot.airframe.body.mass_kg * params_snapshot.site.gravity_m_s2;

    defmt::info!("RC interpreter: rate mode started");

    loop {
        let mut rc = rc_sub.next_message_pure().await;
        while let Some(newer) = rc_sub.try_next_message_pure() {
            rc = newer;
        }

        if rc.channel_count < min_channels {
            continue;
        }

        // Map sticks to body rates
        let roll_norm = roll_cal.normalize(rc.channels[roll_cal.index] as i16);
        let pitch_norm = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16);
        let yaw_norm = yaw_cal.normalize(rc.channels[yaw_cal.index] as i16);
        let throttle_norm = throttle_cal.normalize(rc.channels[throttle_cal.index] as i16);

        // Apply deadband
        let roll_cmd = apply_deadband(roll_norm, rate_deadband);
        let pitch_cmd = apply_deadband(pitch_norm, rate_deadband);
        let yaw_cmd = apply_deadband(yaw_norm, rate_deadband);
        let throttle_cmd = if throttle_norm < throttle_deadband {
            0.0
        } else {
            throttle_norm
        };

        let rate_ref = nalgebra::Vector3::new(
            roll_cmd * max_rate_rp,
            pitch_cmd * max_rate_rp,
            yaw_cmd * max_rate_yaw,
        );
        // Throttle 0→1 maps to 0→2×hover thrust (mid-stick ≈ hover).
        let collective_thrust_n = throttle_cmd * 2.0 * hover_thrust_n;

        let setpoint = cybflight_msgs::AttitudeControlSetpoint {
            timestamp: Instant::now(),
            collective_thrust_n,
            attitude_quaternion: nalgebra::UnitQuaternion::identity(),
            body_rate_rad_s: rate_ref,
            torque_n_m: nalgebra::Vector3::zeros(),
        };
        super::RATE_COMMAND.signal(setpoint.clone());
        ctrl_sp_pub.publish_immediate(setpoint);
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
use cybflight_core::rc::rc_mapping::{ChannelCalibration, StickEndpoints};

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
/// Stick-integrator tuning and the position envelope, snapshotted from
/// the `rc` and `site` param groups.
///
/// Cached in a static rather than read per-frame: `params::get()` clones
/// the whole `FirmwareConfig` under a critical section. Both groups are
/// reboot-flagged, so one snapshot at task start is the whole story —
/// and the free-function envelope clamp needs access without threading a
/// config argument through every call site.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
#[derive(Clone, Copy)]
pub(crate) struct StickConfig {
    pub xy_rate_m_s: f32,
    pub z_rate_m_s: f32,
    pub land_rate_m_s: f32,
    pub land_lead_m: f32,
    /// Oldest odometry sample the landing leash will anchor on [s].
    ///
    /// Shared with the MPC's own gate (`mpc_odom_stale_s`) because it
    /// answers the same question — how long a pose stays trustworthy —
    /// and one horizon is easier to reason about than two. Unlike the
    /// rest of this struct's sources it is *not* reboot-flagged, so a
    /// `param set` between flights needs a reboot to reach this
    /// snapshot; that is acceptable for a trust horizon and keeps the
    /// per-frame path free of `params::get()`.
    pub odom_trust_s: f32,
    pub yaw_rate_rad_s: f32,
    pub xy_deadband: f32,
    pub throttle_deadband: f32,
    pub throttle_land_us: u16,
    pub mission_channel: usize,
    pub mission_high_us: u16,
    pub mission_low_us: u16,
    pub launch_us: u16,
    pub launch_confirm_frames: u8,
    pub fence_enable: bool,
    pub fence_x_m: f32,
    pub fence_y_m: f32,
    pub fence_z_max_m: f32,
    pub fence_z_min_m: f32,
}

#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
impl StickConfig {
    /// Pre-param values. `fence_enable` is false here and the envelope
    /// is switched on per vehicle, which reproduces the old behaviour
    /// exactly: the clamp used to be `#[cfg(est_pos_mocap)]`, so GPS
    /// builds had none and indoor builds had one.
    pub(crate) const FALLBACK: Self = Self {
        xy_rate_m_s: 1.0,
        z_rate_m_s: 0.5,
        land_rate_m_s: 0.4,
        land_lead_m: 1.0,
        odom_trust_s: 0.2,
        yaw_rate_rad_s: 4.0,
        xy_deadband: 0.05,
        throttle_deadband: 0.12,
        throttle_land_us: 1100,
        mission_channel: 4,
        mission_high_us: 1700,
        mission_low_us: 1300,
        launch_us: 1600,
        launch_confirm_frames: 5,
        fence_enable: false,
        fence_x_m: 2.5,
        fence_y_m: 3.5,
        fence_z_max_m: 2.0,
        fence_z_min_m: 0.0,
    };
}

#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
static STICK_CONFIG: embassy_sync::blocking_mutex::Mutex<
    embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex,
    core::cell::Cell<StickConfig>,
> = embassy_sync::blocking_mutex::Mutex::new(core::cell::Cell::new(StickConfig::FALLBACK));

/// Snapshot the stick/envelope params. Call once, after `params::init`.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
pub fn init_stick_config() {
    let p = crate::params::get();
    STICK_CONFIG.lock(|c| {
        c.set(StickConfig {
            xy_rate_m_s: p.rc.xy_rate_m_s,
            z_rate_m_s: p.rc.z_rate_m_s,
            land_rate_m_s: p.rc.land_rate_m_s,
            land_lead_m: p.rc.land_lead_m,
            odom_trust_s: p.mpc.odom_stale_s,
            yaw_rate_rad_s: p.rc.max_rate_yaw_rad_s,
            xy_deadband: p.rc.xy_deadband,
            throttle_deadband: p.rc.throttle_deadband,
            throttle_land_us: p.rc.throttle_land_us,
            mission_channel: p.rc.mission_channel as usize,
            mission_high_us: p.rc.mission_high_us,
            mission_low_us: p.rc.mission_low_us,
            launch_us: p.rc.launch_us,
            launch_confirm_frames: p.rc.launch_confirm_frames,
            fence_enable: p.site.fence_enable,
            fence_x_m: p.site.fence_x_m,
            fence_y_m: p.site.fence_y_m,
            fence_z_max_m: p.site.fence_z_max_m,
            fence_z_min_m: p.site.fence_z_min_m,
        })
    });
}

#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
#[inline]
pub(crate) fn stick_config() -> StickConfig {
    STICK_CONFIG.lock(|c| c.get())
}
/// Upper bound on the per-frame integration step. Guards against RC frame
/// gaps (e.g. transient link hiccups) producing huge single-step drifts.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const MAX_FRAME_DT_S: f32 = 0.1;

/// Extra rate, beyond the vehicle's own measured vertical speed, at which
/// the landing leash's altitude anchor may track a new measurement [m/s].
///
/// The anchor is the only ground reference the landing branch has, so a
/// single bad sample must not be able to move it far: latching one
/// metres-high reading yanks the descent reference up by that much, and
/// the reference can only walk back down at `land_rate_m_s`. Bounding the
/// anchor's slew by the *measured* vertical speed plus this margin
/// separates the two cases without needing to classify samples. In a real
/// climb or descent the estimator reports the speed that justifies the
/// altitude change, so the anchor keeps up exactly; in a localization
/// glitch (mocap re-association, ESKF re-init, GPS jump) the reported
/// speed does not change, so the anchor moves by at most a centimetre per
/// frame and the next good sample pulls it straight back.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
const LEASH_ANCHOR_TRACK_MARGIN_M_S: f32 = 0.5;

/// Clamp `pos` to the configured position envelope.
///
/// Defense-in-depth against stick drift, pose glitches that survive the
/// estimator's rejection gate, or a drone that was physically moved
/// before re-arm. Without it, an Idle-state stick drift integrates
/// without bound and can push the setpoint outside the tracking volume →
/// loss of pose updates → estimator open-loop on IMU → crash.
///
/// Previously `#[cfg(est_pos_mocap)]` with the lab arena's dimensions
/// baked in. Now driven by the `site` group: `fence_enable` defaults off
/// (reproducing the old GPS behaviour of no envelope), indoor vehicles
/// pin it on with their own arena size, and an outdoor vehicle can opt
/// in rather than the choice being welded to the position source.
#[cfg(any(feature = "outer_geometric", feature = "outer_mpc"))]
#[inline]
fn clamp_indoor_envelope(pos: &mut nalgebra::Vector3<f32>) {
    let c = stick_config();
    if !c.fence_enable {
        return;
    }
    pos.x = pos.x.clamp(-c.fence_x_m, c.fence_x_m);
    pos.y = pos.y.clamp(-c.fence_y_m, c.fence_y_m);
    pos.z = pos.z.clamp(c.fence_z_min_m, c.fence_z_max_m);
}

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
    // Stick tuning, trigger mapping and the position envelope. Both
    // source groups are reboot-flagged, so one snapshot covers the
    // task's lifetime.
    init_stick_config();
    let sc = stick_config();

    let mut rc_sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");
    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("rc_interpreter: VEHICLE_ODOMETRY subscriber");

    // Stick travel from the `rc` group, for the same reason as the rate
    // path above: `rc_min_us` / `rc_mid_us` / `rc_max_us` describe the
    // transmitter, and every vehicle YAML already pins them.
    let endpoints = {
        let p = crate::params::get();
        StickEndpoints {
            min_us: p.rc.min_us as i16,
            mid_us: p.rc.mid_us as i16,
            max_us: p.rc.max_us as i16,
        }
    };
    let pitch_cal = ChannelCalibration::centered_with(1, endpoints);
    let roll_cal = ChannelCalibration::centered_with(0, endpoints);
    let throttle_cal = ChannelCalibration::throttle_with(2, endpoints);
    // Inverted to match the AETR convention in `rc_mapping`: stick right
    // must yaw the drone right, i.e. *clockwise* seen from above, which
    // is **negative** about world +Z in ENU.
    let yaw_cal = ChannelCalibration::centered_inverted_with(3, endpoints);

    // Minimum `channel_count` we require before we are willing to trust the
    // primary axes — guards against degenerate/truncated RC frames whose
    // remaining positions would otherwise read as `u16::default() = 0`,
    // producing `normalize(0) ≈ -1` on the centered sticks (i.e. full
    // negative XY drift with the pilot's sticks centered!).
    let min_channels: u8 = {
        let mut m = pitch_cal
            .index
            .max(roll_cal.index)
            .max(throttle_cal.index)
            .max(yaw_cal.index);
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
    // Keep the *unclamped* capture too: it seeds the landing leash, which
    // is measured against where the vehicle actually is. Seeding that from
    // the clamped copy would bake the fence floor into it, and on a
    // fence-on vehicle resting below `fence_z_min_m` the leash would start
    // above the airframe.
    let raw_origin = loop {
        let odom = odom_sub.next_message_pure().await;
        let p = odom.pose.position;
        if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
            break p;
        }
        defmt::warn!("rc_interpreter: discarding non-finite odometry during origin capture");
    };
    let origin = {
        // Indoor builds: clamp to the lab envelope so the initial
        // setpoint is in-bounds even if the drone was placed near the
        // arena edge. No-op outdoors.
        let mut p = raw_origin;
        clamp_indoor_envelope(&mut p);
        p
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
    // Frame-count debounce for [`super::LAUNCHED`]. Counts consecutive
    // frames with throttle > [`sc.launch_us`]; latches LAUNCHED after
    // [`sc.launch_confirm_frames`]. Resets on any below-threshold
    // frame and on the disarm edge.
    let mut launch_above_count: u8 = 0;

    // Sticky last measured altitude, refreshed by the single odometry
    // drain at the top of the loop. Sticky rather than per-frame optional
    // on purpose: if odometry stops, the landing leash freezes at the
    // last known altitude instead of vanishing, so the reference cannot
    // keep ratcheting downward while the task is blind.
    let mut last_measured_z: f32 = raw_origin.z;
    // Timebase for the anchor's slew limit. Separate from
    // `last_frame_time` because the anchor is updated at the drain, above
    // every `continue` in the loop body, while `last_frame_time` advances
    // only on frames that reach the stick integrator.
    let mut last_anchor_time = Instant::now();
    let odom_trust = Duration::from_micros(
        if sc.odom_trust_s.is_finite() && sc.odom_trust_s > 0.0 {
            (sc.odom_trust_s * 1.0e6) as u64
        } else {
            defmt::warn!("rc_interpreter: mpc_odom_stale_s not usable — using 0.2 s");
            200_000
        },
    );

    // Mission trigger channel, thresholds and the launch latch all come
    // from `sc` (the `rc` param group): rising edge → request a mission
    // plan (only honored while Idle); falling edge while Planning or
    // Executing → abort.
    // Mission-trigger Schmitt trigger. The band between
    // `rc_mission_low_us` and `rc_mission_high_us` is wider than any
    // physical switch's noise floor, so hysteresis alone rejects
    // spurious flips. No frame-count debounce: the earlier
    // 3-frame counter counted task *observations* (the backlog-collapse
    // loop means this task may see fewer frames than the RC link sends),
    // making trigger timing depend on executor scheduling rather than
    // switch physics.

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

        // ── Odometry drain (exactly once per wakeup) ──────────────────
        //
        // The one place this subscriber is read. It used to be drained
        // only inside the arm-edge and pre-launch blocks, which meant it
        // went completely undrained once LAUNCHED latched — the task had
        // no measured position at all in steady flight, and the
        // subscriber lagged silently because the `_pure` accessors
        // swallow the lag notification.
        //
        // Deliberately above the short-frame guard below: that guard
        // rejects a malformed *RC* frame, which says nothing about the
        // validity of odometry, and draining unconditionally makes "one
        // drain per wakeup" an invariant rather than a path-dependent
        // property.
        //
        // Validity is the same test every other odometry consumer applies
        // (outer_loop step 3, cascade_task): finite components AND a
        // timestamp that is neither future-dated (clock skew or
        // corruption) nor older than the trust horizon (an estimator that
        // hung while still republishing). Finiteness alone is not enough
        // here, because the sample becomes the landing leash's ground
        // reference below.
        let drain_time = Instant::now();
        let mut fresh_pos: Option<nalgebra::Vector3<f32>> = None;
        let mut fresh_vz: f32 = 0.0;
        while let Some(o) = odom_sub.try_next_message_pure() {
            let p = o.pose.position;
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                continue;
            }
            if o.timestamp > drain_time
                || drain_time.saturating_duration_since(o.timestamp) > odom_trust
            {
                continue;
            }
            fresh_pos = Some(p);
            fresh_vz = o.twist.linear.z;
        }
        if let Some(p) = fresh_pos {
            // Slew-limited tracking, not a latch — see
            // `LEASH_ANCHOR_TRACK_MARGIN_M_S`. Updated here, above every
            // `continue` below, so the anchor is a property of the drain
            // rather than of the path a given frame happens to take: a
            // mission or a pre-launch hold must not leave it frozen at a
            // pre-takeoff altitude for the landing that follows.
            let anchor_dt = drain_time
                .saturating_duration_since(last_anchor_time)
                .as_micros() as f32
                * 1e-6;
            let anchor_dt = anchor_dt.clamp(0.0, MAX_FRAME_DT_S);
            let speed = if fresh_vz.is_finite() { fresh_vz.abs() } else { 0.0 };
            let step = (speed + LEASH_ANCHOR_TRACK_MARGIN_M_S) * anchor_dt;
            last_measured_z += (p.z - last_measured_z).clamp(-step, step);
        }
        last_anchor_time = drain_time;

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
            // Consume this wakeup's odometry sample (drained once at the
            // top of the loop). `take()` rather than a copy: the first
            // armed frame satisfies both this block and the pre-launch
            // pin below, and the pin must not also write on that frame —
            // which is exactly what happened before, when this block
            // emptied the queue and left the pin with nothing.
            if let Some(mut pos) = fresh_pos.take() {
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
        // Disarm edge: clear LAUNCHED + counter so the next arm cycle
        // re-enters pre-launch idle. The latch is one-way per arm
        // session — only the disarm edge resets it.
        if !armed && was_armed {
            super::LAUNCHED.store(false, core::sync::atomic::Ordering::Release);
            launch_above_count = 0;
            defmt::info!("rc_interpreter: disarm transition — LAUNCHED cleared");
        }
        was_armed = armed;

        // ── LAUNCHED debounce ──────────────────────────────────────────
        //
        // While armed and not yet launched, count consecutive RC frames
        // with throttle stick above [`sc.launch_us`]. After
        // [`sc.launch_confirm_frames`] consecutive frames, latch
        // LAUNCHED so [`indi_task`]'s pre-launch idle bypass releases
        // and closed-loop control engages. Any below-threshold frame
        // restarts the count — partial credit is the wrong UX here.
        if armed && !super::LAUNCHED.load(core::sync::atomic::Ordering::Acquire) {
            if rc.channels[throttle_cal.index] > sc.launch_us {
                launch_above_count = launch_above_count.saturating_add(1);
                if launch_above_count >= sc.launch_confirm_frames {
                    super::LAUNCHED
                        .store(true, core::sync::atomic::Ordering::Release);
                    defmt::info!("rc_interpreter: LAUNCHED latched");
                    // No special re-seed of ACTIVE_POSITION_SETPOINT
                    // here: the pre-launch pin block below already
                    // wrote the live odometry pose this frame, and the
                    // post-launch tick runs the existing
                    // stick-integration block starting from that fresh
                    // value.
                }
            } else {
                launch_above_count = 0;
            }
        }

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
            let ch_present = rc.channel_count > sc.mission_channel as u8;
            let raw_level: Option<bool> = if !ch_present {
                mission_level_confirmed
            } else {
                let v = rc.channels[sc.mission_channel];
                Some(match mission_level_confirmed {
                    Some(true) => v > sc.mission_low_us,
                    Some(false) => v > sc.mission_high_us,
                    None => v > sc.mission_high_us,
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
                        sc.mission_channel
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
                    sc.mission_channel,
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

        // ── Pre-launch setpoint pin (armed && !LAUNCHED) ──────────────
        //
        // While the pilot is armed but hasn't crossed the launch
        // threshold, INDI is bypassed (motors at fixed idle) and any
        // stick fidget would drift the position setpoint without
        // visible effect — but it would corrupt the launch handover.
        // Pin ACTIVE_POSITION_SETPOINT to the live odometry pose every
        // frame so the moment LAUNCHED latches, the closed-loop
        // controller starts tracking the drone's *actual* current
        // position, not a stale or drifted target.
        //
        // Uses this wakeup's odometry sample from the drain at the top
        // of the loop. If none arrived this frame we hold the previous
        // value rather than emit a stale-timestamp write that confuses
        // downstream liveness consumers. On the first armed frame the
        // arm-edge block above has already taken the sample, so this
        // holds — matching the previous behaviour, where that block
        // drained the queue and left this one nothing to write.
        if armed && !super::LAUNCHED.load(core::sync::atomic::Ordering::Acquire) {
            if let Some(mut p) = fresh_pos {
                clamp_indoor_envelope(&mut p);
                super::ACTIVE_POSITION_SETPOINT.lock(|cell| {
                    cell.set(Some(super::ActiveSetpoint {
                        timestamp: now,
                        position: p,
                        yaw_rad: 0.0,
                    }));
                });
            }
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
        //     yaw right     → heading clockwise (world −Z rotation)
        let sx_norm = pitch_cal.normalize(rc.channels[pitch_cal.index] as i16);
        let sy_norm = roll_cal.normalize(rc.channels[roll_cal.index] as i16);
        let sx = apply_deadband_centered(sx_norm, sc.xy_deadband);
        let sy = apply_deadband_centered(sy_norm, sc.xy_deadband);

        // Yaw is a *heading rate* stick, exactly like XY: a deflected
        // stick slews the yaw reference, a centered stick holds it.
        // Shares the centered-stick deadband with XY.
        let syaw_norm = yaw_cal.normalize(rc.channels[yaw_cal.index] as i16);
        let syaw = apply_deadband_centered(syaw_norm, sc.xy_deadband);

        let thr_us = rc.channels[throttle_cal.index];
        let is_landing = thr_us < sc.throttle_land_us;
        let sz = if is_landing {
            // Handled below as an unconditional z = 0 override.
            0.0
        } else {
            let thr_norm = throttle_cal.normalize(thr_us as i16);
            let thr_cmd = (thr_norm - 0.5) * 2.0; // [-1, 1]
            apply_deadband_centered(thr_cmd, sc.throttle_deadband)
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
            pos.x += sx * sc.xy_rate_m_s * dt;
            pos.y -= sy * sc.xy_rate_m_s * dt;
            if is_landing {
                // Rate-limited descent toward the ground. The reference
                // decreases by at most `sc.land_rate_m_s · dt` per RC
                // frame, so the MPC always sees a feasible target within
                // its 1 s horizon and never has to track a 2 m step.
                //
                // The bound is *relative to the vehicle*, not absolute.
                // z = 0 is the ENU origin — the mocap anchor, or wherever
                // the first RTK fix landed — and not the ground, so an
                // absolute floor stops the descent in mid-air over any
                // terrain below the takeoff point. The vehicle's own
                // measured altitude is the only ground reference this
                // task has that does not depend on where the origin
                // happened to land. `last_measured_z` is validated and
                // slew-limited at the drain rather than latched from
                // whatever arrived last, because a leash is only as good
                // as the altitude it is tied to.
                //
                // It is a leash rather than a floor: `.max` also pulls
                // the reference back *up* when the vehicle stops
                // descending, so an airframe held by ground effect, a
                // net, or its own landing gear cannot wind the reference
                // metres below itself and then dump that error the
                // instant it breaks free. Some bound is required either
                // way, because holding the land command on the ground
                // would otherwise integrate downward forever.
                //
                // `clamp_indoor_envelope` still runs below, so with the
                // fence on the effective floor is the tighter of this
                // leash and `fence_z_min_m`. The two are different
                // things now — an anti-windup lead limit and an absolute
                // envelope — rather than two floors competing.
                pos.z = (pos.z - sc.land_rate_m_s * dt)
                    .max(last_measured_z - sc.land_lead_m);
            } else {
                pos.z += sz * sc.z_rate_m_s * dt;
            }

            // Indoor builds: clamp the integrated setpoint into the lab
            // envelope so a sustained stick deflection cannot drift the
            // target outside the mocap volume. No-op outdoors.
            clamp_indoor_envelope(&mut pos);

            // Integrate the yaw stick and wrap to (-π, π]. The per-frame
            // step is bounded by `MAX_FRAME_DT_S · yaw_rate_rad_s`, far
            // under π, so a single additive wrap is sufficient — no
            // `rem_euclid` (and no libm call) needed on the hot path.
            let mut yaw = cur.yaw_rad + syaw * sc.yaw_rate_rad_s * dt;
            if yaw > core::f32::consts::PI {
                yaw -= 2.0 * core::f32::consts::PI;
            } else if yaw <= -core::f32::consts::PI {
                yaw += 2.0 * core::f32::consts::PI;
            }

            cell.set(Some(super::ActiveSetpoint {
                timestamp: now,
                position: pos,
                yaw_rad: yaw,
            }));
        });
    }
}
