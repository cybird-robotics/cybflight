//! Stage 6: record-triggered flight recorder.
//!
//! While the "should-record" predicate is false (default state): idle.
//! On rising edge: open `flight_<NNNN>.mcap`, emit envelope + schemas
//! + channels + ARM event, then run the multi-topic capture loop
//! **until the predicate falls**. On falling edge: emit DISARM +
//! LOG_END events, close envelope, flush and unmount.
//!
//! ## What "should-record" means
//!
//! [`super::should_record`] = `IS_ARMED || RECORDER_HOLD`:
//! - `IS_ARMED` is the production trigger — actual flight arm.
//! - `RECORDER_HOLD` is the bench-test trigger — `blackbox record
//!   on` sets it without touching `ARM_STATE` (no risk of spinning
//!   DShot output).
//!
//! Either path lands here: same arm-edge state machine, same MCAP
//! envelope, same `/events` records.
//!
//! Wire format: an MCAP envelope (`Magic / Header / Schemas /
//! Channels`) opens, then the multi-topic capture loop emits
//! `Message` records, then `/events` records bracket arm/disarm
//! and edge-detected health transitions. The capture loop checks
//! `should_record()` on every wake-up (timer or message) so the
//! falling-edge is observed within at most ~50 ms even when the
//! topics are silent.
//!
//! ## Bench-test recipe
//!
//! ```text
//! > blackbox record on
//! [ defmt: blackbox: record-edge → session 1 ]
//! [ samples accumulate, file grows ]
//! > blackbox record off
//! [ defmt: blackbox: closed flight_0001.mcap ]
//! ```
//!
//! Pop the card, run:
//! ```sh
//! mcap doctor flight_0001.mcap
//! mcap info   flight_0001.mcap        # channel count per tier; ids must be distinct
//! python read_mcap.py flight_0001.mcap | grep events
//! ```
//! → expect `kind=1` (ARM) at the start, `kind=2` (DISARM) and
//! `kind=16` (LOG_END) at the end.

use core::fmt::Write as _;
use core::future::pending;

use embassy_futures::select::{Either6, select6};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pubsub::{Subscriber, WaitResult};
use embassy_time::{Duration, Instant, Timer};
use embedded_io_async::Write;
use heapless::String;

use core::sync::atomic::Ordering;

use super::fat::{self, FileBody, OpError};
use super::mcap;
use super::cbor as cbor_result;
use super::record_set::RecordSet;
use super::sdmmc_block::SdmmcBlockStore;
use super::should_record;
#[cfg(feature = "est_eskf")]
use super::topics::estimator_state;
use super::topics::{
    attitude, control_setpoint, events, gps_health, health, imu, imu_raw, motor_state, motors, mpc,
    mpc_cost, odometry, power, rc, tracking_error,
};
use crate::sensors::POWER_TELEM;
use crate::sensors::power::PowerTelemetry;
use crate::control::TrackingError as TrackErrMsg;
use crate::control::failsafe::{FAILSAFE_ACTIVE, FAILSAFE_REASON, RC_LINK_HEALTHY};
use crate::control::indi_task::{
    INNER_SILENT_CAUSE, VOLTAGE_STALE, VOLTAGE_STALE_EPISODES, VOLTAGE_STALE_LAST_MS,
};
use crate::control::{
    ACTUATOR_MOTORS_TELEM, CONTROL_SETPOINT_TELEM, MPC_COST_ADAPT, OCP_SOLVER_OUTPUT,
    PROCESSED_MOTOR_STATE, TRACKING_ERROR,
};
#[cfg(feature = "outer_mpc")]
use crate::control::{MISSION_STATE, MissionState};
#[cfg(feature = "est_eskf")]
use crate::estimation::ESTIMATOR_BIAS_TELEM;
use crate::estimation::ESTIMATOR_READY;
use crate::msgs;
use super::record_set;
use crate::sensors::{BLACKBOX_ODOMETRY, IMU_1, IMU_1_RAW, MAHONY_ATTITUDE, RC_INPUT};

const MCAP_LIBRARY: &str = concat!("cybflight v", env!("CARGO_PKG_VERSION"));

/// Cadence of the IS_ARMED disarm-edge poll inside the capture loop.
/// 50 ms means worst-case latency to observe disarm = 50 ms (plus the
/// in-flight write_message that's currently blocked on FAT). The
/// recorder doesn't try harder than this — disarm-by-RC already has
/// its own debounce upstream.
const DISARM_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Floor on the gap between `KIND_RECORDER_OVERRUN` records.
///
/// This is the one event whose emission rate correlates with the
/// failure it reports: it fires when the recorder is already behind on
/// writes, and every record it emits is more bytes down the same
/// backed-up path. At 1 Hz it costs ~60 B/s against the ~140 KiB/s the
/// recorder sustains (`record_set::SUSTAINED_GOODPUT_B_S`) — 0.04 %,
/// which cannot meaningfully deepen the hole it is reporting. The
/// first edge out of zero is exempt so the onset is timestamped
/// precisely.
const DROP_EVENT_INTERVAL: Duration = Duration::from_secs(1);

/// Cadence of the `/health` emit. 20 Hz matches what the old `/odom`
/// piggyback delivered while the estimator was healthy, but is now
/// driven by the recorder's own loop tick rather than `/odom`. That
/// matters because the most interesting failsafe transitions happen
/// *after* the estimator drops — and `/odom` stops publishing then,
/// which used to take `/health` down with it. Decoupled, `/health`
/// keeps emitting through the failsafe progression and disarm.
const HEALTH_EMIT_INTERVAL: Duration = Duration::from_millis(50);

/// Longest gap between `/gps_health` records when nothing about the
/// GPS state has changed.
///
/// `/gps_health` is polled at [`HEALTH_EMIT_INTERVAL`] but only
/// *emitted* when the snapshot differs from the last one — plus this
/// heartbeat, so a reader can always distinguish "state unchanged"
/// from "recorder stopped" and every file still opens with a baseline
/// record.
///
/// On the six of eight fleet vehicles that are `pos_source: mocap`,
/// `GPS_HEALTH` sits at `NotConfigured` for the whole flight and every
/// field of the snapshot is constant, so the old fixed 20 Hz cadence
/// spent ~5.3 KB/s re-encoding an identical 264-byte record — 1.1 % of
/// Mid on an `imu_1khz` build. At 1 Hz that becomes ~0.26 KB/s while
/// keeping the explicit "GPS not present" marker the topic exists for.
/// On a GPS build `nav_pvt_arrival_us` advances every NAV-PVT, so
/// change detection alone reproduces the previous cadence and this
/// constant never binds.
const GPS_HEALTH_HEARTBEAT: Duration = Duration::from_secs(1);

/// Per-message encode scratch. Every topic's worst-case record must
/// fit: the hot topics are bounded by the size tests in
/// `cybflight_core::blackbox_wire`; the map topics are bounded by
/// their fixed key sets (largest today is `/health` at ~370 B).
/// `note_encode` counts and skips anything that doesn't fit, so an
/// overshoot degrades to a visible sequence gap, not a corrupt file.
const SCRATCH_LEN: usize = 512;

/// How often the capture loop commits the FAT directory entry.
///
/// `embedded_fatfs` only writes the entry — the file's *size* — in
/// `File::flush`. Data clusters and FAT chain updates stream out as
/// the file grows, but until an entry flush lands the on-disk size
/// is whatever `truncate()` set at session open: zero. So without a
/// periodic flush, a brownout, a crash-landing, or a watchdog reset
/// at minute 7 of an 8-minute flight leaves `flight_NNNN.mcap` at
/// 0 bytes with every cluster orphaned — the log of the flight you
/// most wanted to read is the one guaranteed to be lost.
///
/// 1 s bounds the loss to the last second of flight. The cost is one
/// directory-sector write plus a `BufStream` write-back of the
/// current partial block, i.e. two 512 B writes per second against a
/// stream running at hundreds of KB/s — under 0.5 % write
/// amplification, and it does not touch the control loop at all
/// (this task lives on the thread executor, below both interrupt
/// executors).
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// How long the capture loop continues after `should_record()` flips
/// false (i.e. after disarm). Exists so post-disarm state transitions
/// land in the log — the most common case is the 2 s pos-staleness
/// timeout firing after a measurement source vanishes right before /
/// during disarm. Without the grace window, `LOG_END` lands ~50 ms
/// after disarm and the STALE bits assert into a closed file.
///
/// 3 s = `POS_TIMEOUT_S` (2 s) + headroom for the loop's coarsest
/// poll interval (50 ms) and for whatever brief overrun the SD-write
/// path introduces. The DISARM event is still emitted at the actual
/// disarm-edge moment; only LOG_END (the file-close marker) shifts.
const POST_DISARM_GRACE: Duration = Duration::from_secs(3);

/// Per-topic drain budget per outer iteration for **small/mid-tier**
/// topics (`/rc`, `/odometry`, `/mpc`, `/motors`). Must be at least
/// every channel's `CAP` so a fully-buffered channel can be cleared in
/// one pass, while still being small enough that we re-check
/// `should_record()` and rotate across topics on a fast cadence.
///
/// The max of [`crate::rates::INDI_TELEM_PUBSUB_CAP`] (48) and
/// [`crate::rates::BLACKBOX_ODOM_PUBSUB_CAP`] (25) — the two deepest
/// channels in this tier (next is 16, shared by
/// `/motors`, `/motor_state` and `/tracking_error`) — the same
/// derive-from-the-channel treatment [`DRAIN_BUDGET_DEPRIO`] gets from
/// `IMU_PUBSUB_CAP`, so widening a channel can never silently leave the
/// budget below it. The const assert below keeps the other caps covered.
///
/// **Why a floor is needed.** Below `CAP` the failure is the one
/// [`DRAIN_BUDGET_DEPRIO`] describes: a channel that fills during one
/// stall can never be emptied, and a transient stall becomes permanent
/// loss. A short budget also caps the topic outright — one pass moves at
/// most `budget` messages, so the recorder can never log more than
/// `min(CAP, budget) × loop_rate` of it.
///
/// **Why a bound is needed.** With an unbounded `try_next_message`
/// drain and an 8 kHz topic like `/imu1`, the producer fires faster
/// than each `emit_msg` SD write can complete. The subscriber stays
/// 1–2 messages behind the producer, `try_next_message` never
/// returns `None`, and the drain never exits. The outer loop then
/// never re-checks `should_record()` — user disarm goes unobserved,
/// the file is never closed, and on power-down the FAT directory
/// entry is left at the size set by `truncate()` (i.e. zero bytes).
const DRAIN_BUDGET_NORMAL: usize = if crate::rates::INDI_TELEM_PUBSUB_CAP
    > crate::rates::BLACKBOX_ODOM_PUBSUB_CAP
{
    crate::rates::INDI_TELEM_PUBSUB_CAP
} else {
    crate::rates::BLACKBOX_ODOM_PUBSUB_CAP
};

// Every other small/mid-tier channel must also clear in one pass. The
// deepest not covered by the max() above is `/tracking_error` at 16;
// raise this floor alongside any channel that outgrows it.
const _: () = assert!(
    DRAIN_BUDGET_NORMAL >= 16,
    "drain budget below a small/mid-tier channel CAP — see DRAIN_BUDGET_NORMAL",
);

/// Drain budget for **large-tier** topics — `/imu1` and `/imu1_raw`.
///
/// Tied to `IMU_PUBSUB_CAP` so a fully-backed-up channel still clears
/// in a single pass: if the budget were below CAP, a channel that
/// filled during one stall could never be emptied and every
/// subsequent iteration would report `Lagged`, turning a transient
/// stall into permanent loss. Deriving it from the same const as the
/// channel keeps that invariant across both IMU rates instead of
/// hardcoding the 8 kHz-era value of 4.
///
/// Note the drain **order** — not this number — is what
/// deprioritises IMU: the large tier drains last, after every
/// small/mid topic has had its turn, so a shortfall in a given
/// iteration lands on IMU. See [`super::record_set`] for the full
/// drop-priority rationale and its rate caveat.
const DRAIN_BUDGET_DEPRIO: usize = crate::rates::IMU_PUBSUB_CAP;

type ImuSub =
    Subscriber<'static, CriticalSectionRawMutex, msgs::Imu, { crate::rates::IMU_PUBSUB_CAP }, { crate::sensors::IMU_PUBSUB_SUBS }, 1>;
/// Raw (pre-biquad-LP) IMU subscriber over `IMU_1_RAW`. Same shape
/// as `ImuSub`. Drains under `DRAIN_BUDGET_DEPRIO` like filtered
/// IMU — both run at the full IMU rate (1 kHz or 8 kHz depending on
/// the `imu_1khz` build knob).
type ImuRawSub =
    Subscriber<'static, CriticalSectionRawMutex, msgs::Imu, { crate::rates::IMU_PUBSUB_CAP }, { crate::sensors::IMU_PUBSUB_SUBS }, 1>;
type RcSub = Subscriber<'static, CriticalSectionRawMutex, msgs::RcInput, 4, 6, 1>;
/// Odometry subscriber over `BLACKBOX_ODOMETRY` — the decimated mirror,
/// NOT `VEHICLE_ODOMETRY`. CAP tracks `rates::BLACKBOX_ODOM_PUBSUB_CAP`
/// (25 slots = 200 ms at 125 Hz); because the channel carries records,
/// the whole buffer reaches the file rather than one sample in N of it.
type OdomSub = Subscriber<
    'static,
    CriticalSectionRawMutex,
    msgs::VehicleOdometry,
    { crate::rates::BLACKBOX_ODOM_PUBSUB_CAP },
    2,
    1,
>;
/// Attitude subscriber over `MAHONY_ATTITUDE` — the ~100 Hz IMU-only
/// Mahony estimate, NOT the ESKF attitude (that rides `/odometry`).
/// CAP=4/SUBS=4 matches the channel (40 ms drain-stall tolerance).
type AttSub = Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleAttitude, 4, 4, 1>;
type McpSub = Subscriber<'static, CriticalSectionRawMutex, msgs::OcpSolverOutput, 4, 4, 1>;
/// Learned-cost trace subscriber over `MPC_COST_ADAPT`. CAP=4/SUBS=2
/// matches the channel (only the outer loop publishes, ≤ 200 Hz).
type McpCostSub = Subscriber<'static, CriticalSectionRawMutex, msgs::MpcCostAdapt, 4, 2, 1>;
/// Motors subscriber over `ACTUATOR_MOTORS_TELEM`. CAP=16 matches the
/// channel (32 ms drain-stall tolerance at the sysid tier's 500 Hz);
/// SUBS=4 leaves headroom alongside esp_bridge + this recorder.
type MotorSub = Subscriber<
    'static,
    CriticalSectionRawMutex,
    msgs::ActuatorMotors,
    { crate::rates::INDI_TELEM_PUBSUB_CAP },
    4,
    1,
>;
/// Motor-state subscriber over `PROCESSED_MOTOR_STATE`. The
/// `commanded vs achieved` companion to `MotorSub` — same shape,
/// CAP=16/SUBS=4.
type MotorStateSub = Subscriber<
    'static,
    CriticalSectionRawMutex,
    msgs::MotorStateTelemetry,
    { crate::rates::INDI_TELEM_PUBSUB_CAP },
    4,
    1,
>;
/// Power subscriber over `POWER_TELEM`. CAP=8 matches the channel
/// (80 ms drain-stall tolerance at 100 Hz); SUBS=2.
type PowerSub = Subscriber<'static, CriticalSectionRawMutex, PowerTelemetry, 8, 2, 1>;
/// Tracking-error subscriber over `TRACKING_ERROR`. CAP=16/SUBS=3/PUBS=3
/// — see [`crate::control::TRACKING_ERROR`] for the rationale.
type TrackErrSub = Subscriber<'static, CriticalSectionRawMutex, TrackErrMsg, 16, 3, 3>;
/// Control-setpoint subscriber over `CONTROL_SETPOINT_TELEM`. The
/// outer-loop counterpart to `MotorSub` — same shape, CAP=4/SUBS=4,
/// PUBS=3 because all three outer-loop tasks declare a publisher.
type CtrlSpSub =
    Subscriber<'static, CriticalSectionRawMutex, msgs::AttitudeControlSetpoint, 4, 4, 3>;
/// Estimator-bias subscriber over `ESTIMATOR_BIAS_TELEM`. Decimated
/// ~10 Hz mirror of the per-predict-tick `ESKF_*_BIAS` Signals.
/// CAP=2 (200 ms tolerance at 10 Hz is plenty); SUBS=4; PUBS=2 =
/// both ESKF source tasks declare a publisher.
#[cfg(feature = "est_eskf")]
type EstStateSub = Subscriber<'static, CriticalSectionRawMutex, msgs::EstimatorBias, 2, 4, 2>;

/// Per-session snapshot of the Stage-7 shaping params
/// (`blackbox_rate_div`, `blackbox_mute_mask`), taken once at session
/// start — mid-session `param set` edits apply to the *next* session,
/// which is also why the shell refuses them while recording.
#[derive(Clone, Copy)]
pub struct SessionKnobs {
    /// The session's `blackbox_rate_div`, recorded in the file's
    /// `rate_div` metadata so a reader knows what `/imu1_raw`'s cadence
    /// means. The recorder no longer divides anything itself: `/imu1_raw`
    /// is thinned by `sensors::imu` at the publisher, `/odometry` rides
    /// the already-decimated `BLACKBOX_ODOMETRY` mirror, and `/imu1` is
    /// logged at its published rate.
    pub rate_div: u32,
    /// Muted topics by MCAP channel id (bit N = id N). Muted topics
    /// get no Schema/Channel records and no subscription. `/events`
    /// is never maskable — its bit is cleared at snapshot time.
    ///
    /// `u32`, not `u16`: channel ids now run past 15 (`/mpc_cost` is
    /// 16), and a mask narrower than the id space makes those topics
    /// silently unmutable rather than rejecting the setting.
    pub mute_mask: u32,
}

impl SessionKnobs {
    fn snapshot() -> Self {
        let p = crate::params::get();
        Self {
            rate_div: (p.system.blackbox_rate_div as u32).max(1),
            mute_mask: p.system.blackbox_mute_mask & !(1 << events::CHANNEL_ID),
        }
    }

    /// Channel ids at or above the mask's width cannot be addressed at
    /// all, so the bound is the mask's own bit count rather than a
    /// literal that has to be remembered when a topic is added.
    #[inline]
    fn muted(&self, channel_id: u16) -> bool {
        u32::from(channel_id) < u32::BITS && self.mute_mask & (1 << channel_id) != 0
    }
}

pub struct FlightRecorder {
    record_set: RecordSet,
    knobs: SessionKnobs,
    /// `Some` iff the active record set includes this topic.
    /// Subscribers are taken at session start and dropped on close
    /// — so a tier that excludes a topic doesn't even occupy a
    /// pubsub subscriber slot.
    imu_sub: Option<ImuSub>,
    imu_raw_sub: Option<ImuRawSub>,
    rc_sub: Option<RcSub>,
    att_sub: Option<AttSub>,
    odom_sub: Option<OdomSub>,
    mpc_sub: Option<McpSub>,
    mpc_cost_sub: Option<McpCostSub>,
    motor_sub: Option<MotorSub>,
    motor_state_sub: Option<MotorStateSub>,
    power_sub: Option<PowerSub>,
    track_err_sub: Option<TrackErrSub>,
    ctrl_sp_sub: Option<CtrlSpSub>,
    #[cfg(feature = "est_eskf")]
    est_state_sub: Option<EstStateSub>,
    imu_seq: u32,
    imu_raw_seq: u32,
    rc_seq: u32,
    att_seq: u32,
    odom_seq: u32,
    mpc_seq: u32,
    mpc_cost_seq: u32,
    motor_seq: u32,
    motor_state_seq: u32,
    power_seq: u32,
    track_err_seq: u32,
    ctrl_sp_seq: u32,
    #[cfg(feature = "est_eskf")]
    est_state_seq: u32,
    health_seq: u32,
    /// Latest IMU-1 die temperature seen on the `/imu1` stream, for
    /// the 20 Hz `/health` record (`Imu.v2` dropped per-sample
    /// `temp_c`). NaN until the first IMU message; stays NaN for the
    /// whole session on tiers that exclude IMU.
    last_imu_temp_c: f32,
    /// Wall-clock of the last `/health` emit. The capture loop calls
    /// `emit_health` every iteration; `emit_health` no-ops until at
    /// least `HEALTH_EMIT_INTERVAL` has passed. Initialised to a value
    /// far in the past so the first iteration emits a baseline record.
    last_health_emit: Instant,
    gps_health_seq: u32,
    /// Wall-clock of the last `/gps_health` *poll*. Bounds how often
    /// `gps_health::snapshot` takes its critical-section lock, which
    /// is why it is throttled separately from the capture loop's own
    /// iteration rate.
    last_gps_health_poll: Instant,
    /// Wall-clock of the last `/gps_health` record actually written.
    /// Drives [`GPS_HEALTH_HEARTBEAT`].
    last_gps_health_emit: Instant,
    /// Last snapshot written, for change detection. `None` until the
    /// first poll, so every session opens with a baseline record.
    last_gps_health: Option<gps_health::Flat>,
    ev_seq: u32,
    /// Last-seen `FAILSAFE_ACTIVE` value. Edge-detected each
    /// iteration so we emit `KIND_FAILSAFE` / `KIND_FAILSAFE_CLEAR`
    /// only on transitions, not every poll.
    prev_failsafe: bool,
    /// Last-seen `ESTIMATOR_READY` value. Same edge-detection for
    /// `KIND_ESTIMATOR_DOWN` / `KIND_ESTIMATOR_UP`.
    prev_est_ready: bool,
    /// Last-seen `RC_LINK_HEALTHY` value. Edge-detected for
    /// `KIND_RC_LOSS` / `KIND_RC_RECOVERED`. Distinct from
    /// `prev_failsafe` — the RC link can flip multiple times
    /// inside a single failsafe lifecycle (e.g. drop → recover-
    /// during-guard → drop-again → guard-expires → failsafe).
    prev_rc_link_healthy: bool,
    /// Last-seen `MISSION_STATE` packed `repr(u8)` value. Edge-
    /// detected for `KIND_MISSION_PLANNING` / `KIND_MISSION_EXECUTING`
    /// / `KIND_MISSION_IDLE`. Stored as the raw `u8` (rather than the
    /// `MissionState` enum) so the field exists on builds without
    /// `outer_mpc` — the edge-detect block is feature-gated, but
    /// keeping the field unconditionally avoids a second `cfg` site.
    #[cfg(feature = "outer_mpc")]
    prev_mission_state: u8,
    /// Last-seen `indi_task::VOLTAGE_STALE`. Edge-detected for
    /// `KIND_POWER_STALE` / `KIND_POWER_OK`.
    prev_voltage_stale: bool,
    /// Last-seen `indi_task::INNER_SILENT_CAUSE`. Snapshotted at
    /// session open so a trip latched by an earlier session in the same
    /// boot doesn't re-emit; edge-detected for `KIND_INNER_SILENT`.
    prev_inner_silent: u8,
    /// `drops` as of the last `KIND_RECORDER_OVERRUN`. The event is
    /// emitted only when the count has actually moved since then, so a
    /// session that drops once and then recovers reports once.
    last_drops_reported: u32,
    /// Wall-clock of the last `KIND_RECORDER_OVERRUN`. Initialised to
    /// the epoch so the first edge is emitted without waiting out
    /// [`DROP_EVENT_INTERVAL`].
    last_drop_event: Instant,
    pub messages: u32,
    pub drops: u32,
    /// Messages skipped because a topic encoder returned `OutOfSpace`
    /// — the record didn't fit the 512 B scratch. Should be zero
    /// forever: every current topic's worst case is bounded well
    /// below scratch (the hot topics provably so, via the size tests
    /// in `cybflight_core::blackbox_wire`). Nonzero means a topic
    /// grew past its budget; the affected messages are *skipped*, not
    /// emitted empty, so the channel's `sequence` shows a gap where
    /// each one would have been.
    pub encode_overflows: u32,
}

/// Yields the next message on `sub`; pends forever if `sub` is None.
/// Lets the per-tier subscriber pattern compose into a fixed-arity
/// `select4` regardless of which topics are active — the `pending()`
/// branch never resolves so the corresponding `Either4` arm
/// effectively becomes inert.
async fn next_or_pend<M: Clone, const C: usize, const S: usize, const P: usize>(
    sub: &mut Option<Subscriber<'static, CriticalSectionRawMutex, M, C, S, P>>,
) -> WaitResult<M> {
    match sub {
        Some(s) => s.next_message().await,
        None => pending().await,
    }
}

impl FileBody for FlightRecorder {
    async fn write<W: Write>(&mut self, w: &mut W) -> Result<u32, W::Error> {
        let mut total: u32 = 0;

        // ── envelope ────────────────────────────────────────────
        mcap::write_magic(w).await?;
        total += 8;
        mcap::write_header(w, "", MCAP_LIBRARY).await?;
        total += (1 + 8 + 4 + 0 + 4 + MCAP_LIBRARY.len()) as u32;

        // ── acquisition-provenance metadata ─────────────────────
        // One Metadata record naming the constants a reader needs to
        // interpret the streams: without the ODR and the *effective*
        // LP cutoffs on disk, reconstructing /imu1 from /imu1_raw (or
        // judging filter delay at all) requires out-of-band knowledge
        // of this flight's tune. Values are formatted into small
        // stack strings; the record costs ~200 bytes once per file.
        {
            use core::sync::atomic::Ordering;
            let accel_lpf = f32::from_bits(
                crate::sensors::imu::IMU1_EFFECTIVE_ACCEL_LPF_HZ_BITS.load(Ordering::Acquire),
            );
            let gyro_lpf = f32::from_bits(
                crate::sensors::imu::IMU1_EFFECTIVE_GYRO_LPF_HZ_BITS.load(Ordering::Acquire),
            );
            let mut odr: String<16> = String::new();
            let _ = write!(&mut odr, "{}", crate::rates::IMU_ODR_HZ as u32);
            let mut accel_s: String<16> = String::new();
            let _ = write!(&mut accel_s, "{}", accel_lpf);
            let mut gyro_s: String<16> = String::new();
            let _ = write!(&mut gyro_s, "{}", gyro_lpf);
            let mut div_s: String<8> = String::new();
            let _ = write!(&mut div_s, "{}", self.knobs.rate_div);
            let mut mute_s: String<12> = String::new();
            let _ = write!(&mut mute_s, "{:#010x}", self.knobs.mute_mask);
            let pairs: [(&str, &str); 8] = [
                ("imu_odr_hz", odr.as_str()),
                // Post-clamp values from the constructed filters, not
                // the params as requested — only these describe the
                // data actually on the card.
                ("imu_accel_lpf_hz", accel_s.as_str()),
                ("imu_gyro_lpf_hz", gyro_s.as_str()),
                ("imu_filter", "biquad_lp_df2"),
                ("board", crate::bsp::BOARD_NAME),
                ("record_set", self.record_set.name()),
                // Session shaping: readers must not assume nominal
                // rates when rate_div > 1, nor a full tier topic set
                // when mute_mask != 0.
                ("rate_div", div_s.as_str()),
                ("mute_mask", mute_s.as_str()),
            ];
            mcap::write_metadata(w, "cybflight", &pairs).await?;
            total += mcap::metadata_record_len("cybflight", &pairs);
        }

        // ── schemas + channels (data-driven from the active record-set) ──
        // `def.channel_id` is stable per topic, so a Mid-tier file
        // and a Large-tier file both call `/imu1` channel 1.
        // Muted topics (`blackbox_mute_mask`) are absent from the file
        // entirely — no Schema/Channel record, so readers never see a
        // channel that carries zero messages by configuration.
        let topic_set = self.record_set.topic_set();
        for def in topic_set.iter().filter(|d| !self.knobs.muted(d.channel_id)) {
            mcap::write_schema(
                w,
                def.channel_id,
                def.schema_name,
                "jsonschema",
                def.schema_data,
            )
            .await?;
            total += (1
                + 8
                + 2
                + 4
                + def.schema_name.len()
                + 4
                + "jsonschema".len()
                + 4
                + def.schema_data.len()) as u32;
        }
        for def in topic_set.iter().filter(|d| !self.knobs.muted(d.channel_id)) {
            mcap::write_channel(w, def.channel_id, def.channel_id, def.topic, "cbor").await?;
            total += (1 + 8 + 2 + 2 + 4 + def.topic.len() + 4 + "cbor".len() + 4) as u32;
        }

        // ── ARM event ───────────────────────────────────────────
        let mut scratch = [0u8; SCRATCH_LEN];
        if self.record_set.includes_events() {
            total += self
                .emit_event(w, &mut scratch, events::KIND_ARM, 0)
                .await?;

            // ── Prior-boot post-mortem mirror ───────────────────
            //
            // If the previous boot left a valid post-mortem record,
            // emit it inline as `/events` records bracketed by
            // KIND_BOOT_POSTMORTEM start/end sentinels. The opening
            // bracket's `data` field packs `(reset_cause << 8 |
            // fatal_kind)` so a post-flight reader can summarize the
            // prior crash from a single record without parsing the
            // full BKPSRAM struct.
            //
            // `take_pending` is single-shot: only the *first* SD
            // session opened after a crashed boot mirrors the
            // record. Subsequent sessions on the same boot don't
            // re-emit it.
            #[cfg(feature = "postmortem")]
            if let Some(prior) = crate::postmortem::recovery::take_pending() {
                let summary_data = ((prior.header.reset_cause & 0x00FF_FFFF) << 8)
                    | (prior.fatal.kind as u32 & 0xFF);
                total += self
                    .emit_event(w, &mut scratch, events::KIND_BOOT_POSTMORTEM, summary_data)
                    .await?;
                // If the fatal slot is populated, emit a synthetic
                // event with the fatal kind so post-flight tools
                // see a single explicit "this is what killed it"
                // marker without having to inspect the bracket
                // payload.
                let fatal_kind = prior.fatal.kind;
                let synth_kind = match crate::postmortem::record::FatalKind::from_u8(fatal_kind) {
                    crate::postmortem::record::FatalKind::Panic => Some(events::KIND_PANIC),
                    crate::postmortem::record::FatalKind::HardFault => Some(events::KIND_HARDFAULT),
                    crate::postmortem::record::FatalKind::Brownout => Some(events::KIND_BROWNOUT),
                    crate::postmortem::record::FatalKind::IwdgReset => {
                        Some(events::KIND_IWDG_RESET)
                    }
                    crate::postmortem::record::FatalKind::None => None,
                };
                if let Some(k) = synth_kind {
                    // Encode CFSR low 16 bits (or panic_msg_idx for
                    // panic) in `data` — same convention as the
                    // event-kind docs.
                    let data = match k {
                        events::KIND_HARDFAULT => prior.fatal.cfsr & 0xFFFF,
                        events::KIND_PANIC => prior.fatal.panic_msg_idx as u32,
                        _ => 0,
                    };
                    total += self.emit_event(w, &mut scratch, k, data).await?;
                }
                // Replay the prior-boot event ring. Each entry's
                // `kind` is in the same KIND_* code space — readers
                // see the prior session's RC/FAILSAFE/MISSION events
                // inline, all between the BOOT_POSTMORTEM brackets.
                for ev in prior.events_in_order() {
                    total += self.emit_event(w, &mut scratch, ev.kind, ev.data).await?;
                }
                total += self
                    .emit_event(w, &mut scratch, events::KIND_BOOT_POSTMORTEM, 0)
                    .await?;
            }
        }

        // ── capture loop ────────────────────────────────────────
        // Two-phase per iteration:
        //
        // 1. **Wait** on `select6` for any sub to fire (or the
        //    DISARM_POLL deadline). Whichever arm wins consumed one
        //    message — process it.
        //
        // 2. **Fairness drain.** `select6` polls left-to-right and
        //    returns at the first-ready arm. With IMU at 8 kHz the
        //    IMU arm is essentially always ready, so the slower
        //    arms (`/rc`, `/odometry`, `/mpc`, `/motors`) get
        //    woken but never reached. After the wait fires, drain
        //    every remaining ready message on every sub
        //    synchronously via `try_next_message`. That guarantees
        //    forward progress on all topics each iteration
        //    regardless of select bias.
        //
        // The fairness drain was added after a flight where a
        // `Large` tier session yielded 1037 IMU messages and zero
        // of everything else — including `/odometry`, which the
        // ESKF publishes at 1 kHz in every build (its predict is
        // decimated to ~1 kHz regardless of the IMU rate, see
        // `estimation::eskf_imu_mocap::PREDICT_DECIMATION`).
        //
        // Loop exit: when `should_record()` flips false we don't
        // break immediately. Instead emit the DISARM event at that
        // moment and continue running for `POST_DISARM_GRACE` so
        // post-disarm transitions (notably the 2 s POS_TIMEOUT_S
        // staleness assertion) land in the log. Re-arm during the
        // grace window doesn't cancel — we're committed to closing.
        let mut grace_deadline: Option<Instant> = None;
        let mut last_flush = Instant::now();
        loop {
            if !should_record() && grace_deadline.is_none() {
                if self.record_set.includes_events() {
                    // Why we're disarming, not just that we are. The
                    // failsafe path stores FAILSAFE_REASON before
                    // FAILSAFE_ACTIVE (both Release), so an Acquire load
                    // of the flag orders the reason load behind it —
                    // same handshake `emit_status_edges` relies on. No
                    // failsafe ⇒ the disarm was commanded.
                    let cause = if FAILSAFE_ACTIVE.load(Ordering::Acquire) {
                        FAILSAFE_REASON.load(Ordering::Acquire) as u32
                    } else {
                        events::DISARM_CAUSE_COMMANDED
                    };
                    total += self
                        .emit_event(w, &mut scratch, events::KIND_DISARM, cause)
                        .await?;
                }
                grace_deadline = Some(Instant::now() + POST_DISARM_GRACE);
            }
            if let Some(deadline) = grace_deadline
                && Instant::now() >= deadline
            {
                break;
            }
            let timer = Timer::after(DISARM_POLL_INTERVAL);
            // Select-arm order is **deliberate**. `select6` polls
            // left-to-right and ties go to the leftmost ready arm,
            // so high-level / low-rate topics (rc, att, odom, mpc)
            // come before raw IMU. With IMU at 8 kHz and the others
            // at 50–200 Hz, the previous ordering (IMU first) meant
            // every wake almost always handled an IMU sample, even
            // when ESKF / RC / MPC frames were also waiting; the
            // fairness drain then caught the rest. Putting IMU last
            // makes the *first* emit-per-iteration a high-level
            // sample whenever one is ready, which keeps the
            // high-level stream tight under SD backpressure.
            //
            // `/motors` now holds the arm `/attitude` used to
            // occupy. It was previously excluded only because
            // `select6` is the maximum named arity; dropping the
            // redundant `/attitude` topic freed a slot, and `/motors`
            // is the natural occupant — a 100 Hz high-level stream,
            // so it belongs on the same side of the ordering as rc /
            // odom / mpc. The fairness drain still covers every
            // topic; this just means a waiting motor sample can be
            // the first emit of an iteration rather than always
            // waiting for the drain.
            match select6(
                timer,
                next_or_pend(&mut self.rc_sub),
                next_or_pend(&mut self.odom_sub),
                next_or_pend(&mut self.mpc_sub),
                next_or_pend(&mut self.motor_sub),
                next_or_pend(&mut self.imu_sub),
            )
            .await
            {
                Either6::First(()) => {
                    // poll-wake — re-check IS_ARMED at top of loop
                }
                Either6::Second(wr) => total += self.emit_rc(w, &mut scratch, wr).await?,
                Either6::Third(wr) => total += self.emit_odom(w, &mut scratch, wr).await?,
                Either6::Fourth(wr) => total += self.emit_mpc(w, &mut scratch, wr).await?,
                Either6::Fifth(wr) => total += self.emit_motors(w, &mut scratch, wr).await?,
                Either6::Sixth(wr) => total += self.emit_imu(w, &mut scratch, wr).await?,
            }

            // Fairness drain — see comment above. Drains run in
            // **tier order**: small (rc) → mid (attitude, odometry,
            // mpc, motors) → large (imu). Order is what does the
            // prioritising: IMU drains last, so when the SD
            // pipeline can't keep up the shortfall naturally lands
            // on IMU instead of the smaller / more-critical topics.
            // When healthy, every drain hits `try_next == None`
            // before its budget so order/budget are immaterial.

            // ── small tier ──────────────────────────────────────
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.rc_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_rc(w, &mut scratch, wr).await?;
            }
            // ── mid tier ────────────────────────────────────────
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.att_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_att(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.odom_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_odom(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.mpc_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_mpc(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.mpc_cost_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_mpc_cost(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.motor_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_motors(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self
                    .motor_state_sub
                    .as_mut()
                    .and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_motor_state(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.power_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_power(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self
                    .track_err_sub
                    .as_mut()
                    .and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_track_err(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self.ctrl_sp_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_ctrl_sp(w, &mut scratch, wr).await?;
            }
            #[cfg(feature = "est_eskf")]
            for _ in 0..DRAIN_BUDGET_NORMAL {
                let next = self
                    .est_state_sub
                    .as_mut()
                    .and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_est_state(w, &mut scratch, wr).await?;
            }
            // ── large tier (deprioritised) ──────────────────────
            // Raw IMU drains **before** filtered IMU. Large tier
            // exists exclusively to capture the pre-LP stream for
            // sysid / RPM-notch fits — filtered IMU is already in
            // Mid, so anyone who's opted into Large is asking
            // specifically for raw. Information density also favours
            // raw: filtered IMU is a deterministic function of raw
            // + biquad coefficients, so preserving raw preserves
            // both views; preserving filtered loses raw forever.
            // Under SD backpressure the shared deprio budget
            // therefore biases drops onto filtered IMU instead.
            for _ in 0..DRAIN_BUDGET_DEPRIO {
                let next = self
                    .imu_raw_sub
                    .as_mut()
                    .and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_imu_raw(w, &mut scratch, wr).await?;
            }
            for _ in 0..DRAIN_BUDGET_DEPRIO {
                let next = self.imu_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_imu(w, &mut scratch, wr).await?;
            }

            // ── /health + /gps_health periodic emits ────────────
            // Both detached from any data-topic emit so failsafe and
            // GPS-loss progressions stay observable when their
            // companion topics stop publishing. Each function self-
            // throttles to `HEALTH_EMIT_INTERVAL`.
            total += self.emit_health(w, &mut scratch).await?;
            total += self.emit_gps_health(w, &mut scratch).await?;

            // ── extraordinary-event edge poll ───────────────────
            // Poll the failsafe + estimator atomics once per outer
            // iteration and emit a `/events` record on each
            // transition. Read-only access to atomics maintained by
            // `control::failsafe` and `estimation::*`; no
            // producer-side cooperation needed. Latency to detect
            // an edge equals the outer-loop period (≤ a few tens
            // of ms in steady state, ≤ 50 ms when topics are
            // silent because of `DISARM_POLL_INTERVAL`).
            if self.record_set.includes_events() {
                total += self.emit_status_edges(w, &mut scratch).await?;
                total += self.emit_drop_edges(w, &mut scratch).await?;
            }

            // ── periodic durability flush ───────────────────────
            // Commits the FAT directory entry so the file has a
            // real size on disk, bounded by `FLUSH_INTERVAL`. Runs
            // last in the iteration so a flush never sits between
            // a message and the record that explains it.
            let now = Instant::now();
            if now.saturating_duration_since(last_flush) >= FLUSH_INTERVAL {
                last_flush = now;
                w.flush().await?;
            }
        }

        // ── LOG_END event ───────────────────────────────────────
        // DISARM is emitted at the disarm-edge inside the capture
        // loop above; LOG_END marks the actual file-close moment,
        // ~POST_DISARM_GRACE later.
        if self.record_set.includes_events() {
            total += self
                .emit_event(w, &mut scratch, events::KIND_LOG_END, 0)
                .await?;
        }

        // ── close envelope ──────────────────────────────────────
        mcap::write_data_end(w, 0).await?;
        total += 1 + 8 + 4;
        mcap::write_footer(w, 0, 0, 0).await?;
        total += 1 + 8 + 20;
        mcap::write_magic(w).await?;
        total += 8;

        Ok(total)
    }
}

impl FlightRecorder {
    /// Per-topic emit helpers. Each takes a `WaitResult` (from either
    /// the `select6` await or `try_next_message` in the drain loop),
    /// runs `handle_wr` to extract `Some(m)` or count drops, encodes
    /// to CBOR, and appends an MCAP `Message` record. Factored so
    /// the wait-arm and the drain-arm share the same emit
    /// implementation — saves both source duplication and codegen
    /// (one monomorphisation per topic instead of two).
    async fn emit_imu<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::Imu>,
    ) -> Result<u32, W::Error> {
        // No divider: `/imu1` is logged at the rate it is published.
        // The one topic `blackbox_rate_div` thins is `/imu1_raw`, and
        // `sensors::imu` does that at the publisher.
        let Some(m) = handle_wr(wr, &mut self.imu_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.last_imu_temp_c = m.temp_c;
        self.imu_seq = self.imu_seq.wrapping_add(1);
        let Some(n) = self.note_encode(imu::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(w, imu::CHANNEL_ID, self.imu_seq, m.timestamp, &scratch[..n]).await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_imu_raw<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::Imu>,
    ) -> Result<u32, W::Error> {
        // No divider here: `sensors::imu` already thinned this channel
        // at the publisher (`record_set::rate_div`), so what arrives is
        // the record stream itself. A `Lagged(n)` is therefore n lost
        // *records* and plain `handle_wr` sizes the seq hole correctly —
        // and, the point of the exercise, the channel's CAP now buffers
        // `div ×` the wall-clock it used to instead of filling with
        // samples the recorder was about to discard.
        let Some(m) = handle_wr(wr, &mut self.imu_raw_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.imu_raw_seq = self.imu_raw_seq.wrapping_add(1);
        let Some(n) = self.note_encode(imu_raw::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            imu_raw::CHANNEL_ID,
            self.imu_raw_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_rc<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::RcInput>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.rc_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.rc_seq = self.rc_seq.wrapping_add(1);
        let Some(n) = self.note_encode(rc::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(w, rc::CHANNEL_ID, self.rc_seq, m.timestamp, &scratch[..n]).await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_att<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::VehicleAttitude>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.att_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.att_seq = self.att_seq.wrapping_add(1);
        let Some(n) = self.note_encode(attitude::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            attitude::CHANNEL_ID,
            self.att_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_odom<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::VehicleOdometry>,
    ) -> Result<u32, W::Error> {
        // No divider: `BLACKBOX_ODOMETRY` is already the decimated
        // stream (`rates::BLACKBOX_ODOM_DECIM`), so a `Lagged(n)` here is
        // n lost *records* and plain `handle_wr` sizes the seq hole.
        let Some(m) = handle_wr(wr, &mut self.odom_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.odom_seq = self.odom_seq.wrapping_add(1);
        let Some(n) = self.note_encode(odometry::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            odometry::CHANNEL_ID,
            self.odom_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    /// Emit one `/health` record if at least `HEALTH_EMIT_INTERVAL`
    /// has passed since the last one. Called every loop iteration.
    /// No-op when the active record set excludes health.
    ///
    /// Decoupled from `/odom` so the failsafe progression — which is
    /// when `/odom` typically stops — stays observable. Worst-case
    /// cadence is bounded by `DISARM_POLL_INTERVAL`: even if every
    /// subscriber stalls, the loop still wakes every 50 ms and emits
    /// one health record.
    async fn emit_health<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
    ) -> Result<u32, W::Error> {
        if !self.record_set.includes_health() || self.knobs.muted(health::CHANNEL_ID) {
            return Ok(0);
        }
        let now = Instant::now();
        if now.saturating_duration_since(self.last_health_emit) < HEALTH_EMIT_INTERVAL {
            return Ok(0);
        }
        self.last_health_emit = now;
        self.health_seq = self.health_seq.wrapping_add(1);
        let Some(n) = self.note_encode(health::encode(scratch, now, self.last_imu_temp_c)) else {
            return Ok(0);
        };
        let bytes = emit_msg(w, health::CHANNEL_ID, self.health_seq, now, &scratch[..n]).await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    /// Poll `/gps_health` at `HEALTH_EMIT_INTERVAL` and emit a record
    /// when the snapshot changed, or when [`GPS_HEALTH_HEARTBEAT`] has
    /// elapsed since the last one. No-op when the active record set
    /// excludes gps_health.
    ///
    /// On builds without `est_pos_gps` the source enum stays
    /// `NotConfigured` for the whole session, so the file still carries
    /// an explicit "GPS not present" marker — it just carries it once
    /// per second instead of twenty times, because the record is
    /// byte-identical every time. Transitions are still caught within
    /// one poll interval, which is what the topic is actually for.
    async fn emit_gps_health<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
    ) -> Result<u32, W::Error> {
        if !self.record_set.includes_gps_health() || self.knobs.muted(gps_health::CHANNEL_ID) {
            return Ok(0);
        }
        let now = Instant::now();
        if now.saturating_duration_since(self.last_gps_health_poll) < HEALTH_EMIT_INTERVAL {
            return Ok(0);
        }
        self.last_gps_health_poll = now;

        let f = gps_health::snapshot();
        let changed = self.last_gps_health != Some(f);
        let heartbeat_due =
            now.saturating_duration_since(self.last_gps_health_emit) >= GPS_HEALTH_HEARTBEAT;
        if !changed && !heartbeat_due {
            return Ok(0);
        }
        self.last_gps_health = Some(f);
        self.last_gps_health_emit = now;
        self.gps_health_seq = self.gps_health_seq.wrapping_add(1);
        let Some(n) = self.note_encode(gps_health::encode(scratch, now, &f)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            gps_health::CHANNEL_ID,
            self.gps_health_seq,
            now,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_mpc<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::OcpSolverOutput>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.mpc_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.mpc_seq = self.mpc_seq.wrapping_add(1);
        let Some(n) = self.note_encode(mpc::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(w, mpc::CHANNEL_ID, self.mpc_seq, m.timestamp, &scratch[..n]).await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_mpc_cost<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::MpcCostAdapt>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.mpc_cost_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.mpc_cost_seq = self.mpc_cost_seq.wrapping_add(1);
        let Some(n) = self.note_encode(mpc_cost::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(w, mpc_cost::CHANNEL_ID, self.mpc_cost_seq, m.timestamp, &scratch[..n])
            .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_motors<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::ActuatorMotors>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.motor_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.motor_seq = self.motor_seq.wrapping_add(1);
        let Some(n) = self.note_encode(motors::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            motors::CHANNEL_ID,
            self.motor_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_motor_state<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::MotorStateTelemetry>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.motor_state_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.motor_state_seq = self.motor_state_seq.wrapping_add(1);
        let Some(n) = self.note_encode(motor_state::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            motor_state::CHANNEL_ID,
            self.motor_state_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_power<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<PowerTelemetry>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.power_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.power_seq = self.power_seq.wrapping_add(1);
        let Some(n) = self.note_encode(power::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(w, power::CHANNEL_ID, self.power_seq, m.timestamp, &scratch[..n]).await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_track_err<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<TrackErrMsg>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.track_err_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.track_err_seq = self.track_err_seq.wrapping_add(1);
        let Some(n) = self.note_encode(tracking_error::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            tracking_error::CHANNEL_ID,
            self.track_err_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_ctrl_sp<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::AttitudeControlSetpoint>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.ctrl_sp_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.ctrl_sp_seq = self.ctrl_sp_seq.wrapping_add(1);
        let Some(n) = self.note_encode(control_setpoint::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            control_setpoint::CHANNEL_ID,
            self.ctrl_sp_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    #[cfg(feature = "est_eskf")]
    async fn emit_est_state<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::EstimatorBias>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.est_state_seq, &mut self.drops) else {
            return Ok(0);
        };
        self.est_state_seq = self.est_state_seq.wrapping_add(1);
        let Some(n) = self.note_encode(estimator_state::encode(scratch, &m)) else {
            return Ok(0);
        };
        let bytes = emit_msg(
            w,
            estimator_state::CHANNEL_ID,
            self.est_state_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    /// Encode + emit one Event record on `/events`. Returns the byte
    /// count. Inlined here rather than going through TOPIC_SET because
    /// events have a tiny per-emit cost and the kind/data payload is
    /// dynamic.
    async fn emit_event<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        kind: u8,
        data: u32,
    ) -> Result<u32, W::Error> {
        let now = Instant::now();
        let Some(n) = self.note_encode(events::encode(scratch, now, kind, data)) else {
            return Ok(0);
        };
        self.ev_seq = self.ev_seq.wrapping_add(1);
        emit_msg(w, events::CHANNEL_ID, self.ev_seq, now, &scratch[..n]).await
    }

    /// Edge-detect every health atomic the firmware exposes
    /// (`FAILSAFE_ACTIVE`, `ESTIMATOR_READY`, `RC_LINK_HEALTHY`)
    /// and emit a `/events` record on each transition. Called
    /// once per outer-loop iteration so latency to detect a
    /// transition is bounded by the loop period.
    ///
    /// **Failsafe entry carries a reason code** in the `data`
    /// field. The failsafe task stores `FAILSAFE_REASON` *before*
    /// `FAILSAFE_ACTIVE` (program order, both `Release`); we load
    /// `FAILSAFE_ACTIVE` first with `Acquire` and then load the
    /// reason. The Release/Acquire pair ensures we see whichever
    /// reason store happened-before the rising edge.
    ///
    /// **The three atomics are independent.** A Phase::GuardPeriod
    /// → Phase::Idle transition (RC blip recovered before guard
    /// expired) flips `RC_LINK_HEALTHY` twice without ever flipping
    /// `FAILSAFE_ACTIVE`, so a tier-Small log captures marginal
    /// RC-link conditions even when no hard failsafe occurs.
    async fn emit_status_edges<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
    ) -> Result<u32, W::Error> {
        let mut total = 0u32;

        let now_failsafe = FAILSAFE_ACTIVE.load(Ordering::Acquire);
        if now_failsafe != self.prev_failsafe {
            if now_failsafe {
                // Rising edge — read REASON now (after the Acquire
                // load above ordered everything happens-before the
                // FAILSAFE_ACTIVE store).
                let reason = FAILSAFE_REASON.load(Ordering::Acquire);
                total += self
                    .emit_event(w, scratch, events::KIND_FAILSAFE, reason as u32)
                    .await?;
            } else {
                total += self
                    .emit_event(w, scratch, events::KIND_FAILSAFE_CLEAR, 0)
                    .await?;
            }
            self.prev_failsafe = now_failsafe;
        }

        let now_est = ESTIMATOR_READY.load(Ordering::Acquire);
        if now_est != self.prev_est_ready {
            let kind = if now_est {
                events::KIND_ESTIMATOR_UP
            } else {
                events::KIND_ESTIMATOR_DOWN
            };
            total += self.emit_event(w, scratch, kind, 0).await?;
            self.prev_est_ready = now_est;
        }

        let now_rc = RC_LINK_HEALTHY.load(Ordering::Acquire);
        if now_rc != self.prev_rc_link_healthy {
            let kind = if now_rc {
                events::KIND_RC_RECOVERED
            } else {
                events::KIND_RC_LOSS
            };
            total += self.emit_event(w, scratch, kind, 0).await?;
            self.prev_rc_link_healthy = now_rc;
        }

        // ── trajectory tracking status (outer_mpc only) ─────────
        // Edge-detect `MISSION_STATE` transitions. The state machine
        // is Idle → Planning → Executing → Idle, but the recorder
        // doesn't assume a specific transition order — we just emit
        // an event whenever the value changes. For Idle entries we
        // pack the *previous* state into `data` so post-flight
        // analysis can distinguish "rejected mid-plan"
        // (data=Planning) from "trajectory finished or failsafe-
        // aborted" (data=Executing) without joining against
        // `MissionStatus.solve.reject_reason`.
        #[cfg(feature = "outer_mpc")]
        {
            let now_mission_raw = MISSION_STATE.load(Ordering::Acquire);
            if now_mission_raw != self.prev_mission_state {
                let prev = self.prev_mission_state;
                let kind = match MissionState::from_u8(now_mission_raw) {
                    MissionState::Planning => events::KIND_MISSION_PLANNING,
                    MissionState::Executing => events::KIND_MISSION_EXECUTING,
                    MissionState::Idle => events::KIND_MISSION_IDLE,
                };
                total += self.emit_event(w, scratch, kind, prev as u32).await?;
                self.prev_mission_state = now_mission_raw;
            }
        }

        // ── inner-loop power / trip status ──────────────────────
        // Soft battery-telemetry staleness. Episodes are >= 500 ms by
        // construction (`VOLTAGE_STALE_TIMEOUT`), comfortably longer
        // than this poll's worst observed period, so a single episode
        // is not aliased away — but a *burst* of them can be, which is
        // what the episode index in `data` exists to expose.
        let now_v_stale = VOLTAGE_STALE.load(Ordering::Acquire);
        if now_v_stale != self.prev_voltage_stale {
            let (kind, data) = if now_v_stale {
                (
                    events::KIND_POWER_STALE,
                    VOLTAGE_STALE_EPISODES.load(Ordering::Acquire),
                )
            } else {
                (
                    events::KIND_POWER_OK,
                    VOLTAGE_STALE_LAST_MS.load(Ordering::Acquire),
                )
            };
            total += self.emit_event(w, scratch, kind, data).await?;
            self.prev_voltage_stale = now_v_stale;
        }

        // The inner loop deciding to go silent. Latched for the boot,
        // so this fires at most once — and it must be logged eagerly:
        // the disarm it causes arrives ~`fs_ctrl_timeout_s` later as a
        // ControllerTimeout failsafe that names the symptom instead of
        // the cause.
        let now_silent = INNER_SILENT_CAUSE.load(Ordering::Acquire);
        if now_silent != self.prev_inner_silent {
            total += self
                .emit_event(w, scratch, events::KIND_INNER_SILENT, now_silent as u32)
                .await?;
            self.prev_inner_silent = now_silent;
        }

        Ok(total)
    }

    /// Emit `KIND_RECORDER_OVERRUN` when the session's drop count has
    /// moved, at most once per [`DROP_EVENT_INTERVAL`].
    ///
    /// Reports the *cumulative* count rather than the delta so a
    /// reader that loses one of these records still knows the true
    /// total from the next one — the topic that reports loss must
    /// itself be robust to being lost.
    async fn emit_drop_edges<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
    ) -> Result<u32, W::Error> {
        if self.drops == self.last_drops_reported {
            return Ok(0);
        }
        let now = Instant::now();
        if now.saturating_duration_since(self.last_drop_event) < DROP_EVENT_INTERVAL {
            return Ok(0);
        }
        let drops = self.drops;
        let n = self
            .emit_event(w, scratch, events::KIND_RECORDER_OVERRUN, drops)
            .await?;
        // Latch the count that was actually written, not `self.drops`
        // as it stands now — an emit that awaited a slow FAT write can
        // return with more drops already accumulated, and those belong
        // to the next record.
        self.last_drops_reported = drops;
        self.last_drop_event = now;
        Ok(n)
    }
}

/// Unwrap one `WaitResult`, folding a `Lagged` report into both the
/// session drop counter and the topic's MCAP sequence number.
///
/// **Why `seq` advances on a drop.** MCAP's per-channel `sequence`
/// exists precisely so a reader can tell that messages went missing.
/// Bumping it only on a successful emit produces a dense 1,2,3,…
/// run whatever happens, so a file that lost 40 % of `/imu1` is
/// byte-for-byte indistinguishable from a clean one — the drop total
/// survives only as a defmt line at session close, which is not on
/// the card. Skipping `n` here makes the gap show up in the file:
/// a reader sees `…, 3, 9, …` and knows five samples were lost
/// between them, and *where*.
///
/// This matters most for the `sakura_bench_hunter_1khz` /
/// `_8khz` pair, which exists to compare the two IMU rates: the
/// 8 kHz session runs at several times the byte rate, so it drops
/// where the 1 kHz session does not, and without this any spectral
/// comparison between the two logs is silently invalid.
impl FlightRecorder {
    /// Unwrap one encoder result, converting `OutOfSpace` into a
    /// counted, logged skip.
    ///
    /// On overflow the caller returns without emitting anything — an
    /// empty-payload message would decode as an error with no
    /// explanation on the reader side, and since the per-topic `seq`
    /// was already incremented, skipping leaves the same sequence gap
    /// a dropped message does. Warn once per session (the first
    /// overflow implies the rest; one line per lost message at up to
    /// 8 kHz would bury the defmt stream).
    fn note_encode(&mut self, r: cbor_result::Result<usize>) -> Option<usize> {
        match r {
            Ok(n) => Some(n),
            Err(_) => {
                self.encode_overflows = self.encode_overflows.wrapping_add(1);
                if self.encode_overflows == 1 {
                    defmt::warn!(
                        "recorder: encoder overflow — a topic outgrew the {} B scratch; \
                         skipping (counted in session summary)",
                        SCRATCH_LEN,
                    );
                }
                None
            }
        }
    }
}

fn handle_wr<M>(wr: WaitResult<M>, seq: &mut u32, drops: &mut u32) -> Option<M> {
    match wr {
        WaitResult::Message(m) => Some(m),
        WaitResult::Lagged(n) => {
            *drops = drops.saturating_add(n as u32);
            // Leave a hole exactly `n` wide; the caller's own
            // `+= 1` on the next successful emit closes it.
            *seq = seq.wrapping_add(n as u32);
            None
        }
    }
}


async fn emit_msg<W: Write>(
    w: &mut W,
    channel_id: u16,
    sequence: u32,
    timestamp: Instant,
    data: &[u8],
) -> Result<u32, W::Error> {
    let log_time_ns = timestamp.as_micros().saturating_mul(1_000);
    mcap::write_message(w, channel_id, sequence, log_time_ns, log_time_ns, data).await?;
    Ok((1 + 8 + 2 + 4 + 8 + 8 + data.len()) as u32)
}

pub type FileNameBuf = String<32>;

fn build_file_name(seq: u32) -> FileNameBuf {
    let mut name = FileNameBuf::new();
    let _ = write!(&mut name, "flight_{:04}.mcap", seq);
    name
}

/// Result of one armed session.
pub struct SessionSummary {
    pub bytes: u32,
    pub messages: u32,
    pub drops: u32,
    /// See [`FlightRecorder::encode_overflows`]; expected 0.
    pub encode_overflows: u32,
    pub file_name: FileNameBuf,
}

/// Run one arm-to-disarm session. Returns when `IS_ARMED` goes
/// false, the file is flushed, and the FS is unmounted.
///
/// `record_set` selects which topics get logged. Subscribers are
/// taken only for topics in the active tier, so e.g. `Mid`-tier
/// sessions don't even occupy `IMU_1`'s subscriber slot.
///
/// The caller is responsible for not invoking `run_session` with
/// [`RecordSet::None`] — that's a "muted" tier and the file should
/// not be opened at all. The blackbox task gates on `enabled()`
/// before calling.
pub async fn run_session(
    store: &mut SdmmcBlockStore,
    sequence: u32,
    record_set: RecordSet,
) -> Result<SessionSummary, OpError> {
    let knobs = SessionKnobs::snapshot();
    if knobs.mute_mask != 0 || knobs.rate_div > 1 {
        defmt::info!(
            "recorder: session shaping: rate_div={} mute_mask={:#010x}",
            knobs.rate_div,
            knobs.mute_mask,
        );
    }
    // Say up front whether this tier fits. Nothing used to: `Sysid` asked
    // for ~2x the recorder's goodput for weeks, every topic lost ~half its
    // messages, and it was only visible by reconstructing sequence holes
    // from the file afterwards. See `record_set::estimated_bytes_per_s`.
    {
        let want = record_set.estimated_bytes_per_s(knobs.rate_div, knobs.mute_mask);
        let have = record_set::SUSTAINED_GOODPUT_B_S;
        if want > have {
            defmt::warn!(
                "recorder: tier {} wants ~{} KiB/s but only ~{} KiB/s is sustainable \
                 — expect ~{} % of every topic to be dropped",
                record_set.name(),
                want / 1024,
                have / 1024,
                100u32.saturating_sub(have.saturating_mul(100) / want.max(1)),
            );
        } else {
            defmt::info!(
                "recorder: tier {} wants ~{} KiB/s of ~{} KiB/s sustainable",
                record_set.name(),
                want / 1024,
                have / 1024,
            );
        }
    }
    let imu_sub = if record_set.includes_imu() && !knobs.muted(imu::CHANNEL_ID) {
        Some(IMU_1.subscriber().map_err(|_| {
            defmt::warn!("recorder: IMU_1 SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let imu_raw_sub = if record_set.includes_imu_raw() && !knobs.muted(imu_raw::CHANNEL_ID) {
        Some(IMU_1_RAW.subscriber().map_err(|_| {
            defmt::warn!("recorder: IMU_1_RAW SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let rc_sub = if record_set.includes_rc() && !knobs.muted(rc::CHANNEL_ID) {
        Some(RC_INPUT.subscriber().map_err(|_| {
            defmt::warn!("recorder: RC_INPUT SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let att_sub = if record_set.includes_attitude() && !knobs.muted(attitude::CHANNEL_ID) {
        Some(MAHONY_ATTITUDE.subscriber().map_err(|_| {
            defmt::warn!("recorder: MAHONY_ATTITUDE SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let odom_sub = if record_set.includes_odometry() && !knobs.muted(odometry::CHANNEL_ID) {
        Some(BLACKBOX_ODOMETRY.subscriber().map_err(|_| {
            defmt::warn!("recorder: BLACKBOX_ODOMETRY SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let mpc_sub = if record_set.includes_mpc() && !knobs.muted(mpc::CHANNEL_ID) {
        Some(OCP_SOLVER_OUTPUT.subscriber().map_err(|_| {
            defmt::warn!("recorder: OCP_SOLVER_OUTPUT SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let mpc_cost_sub = if record_set.includes_mpc_cost() && !knobs.muted(mpc_cost::CHANNEL_ID) {
        Some(MPC_COST_ADAPT.subscriber().map_err(|_| {
            defmt::warn!("recorder: MPC_COST_ADAPT SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let motor_sub = if record_set.includes_motors() && !knobs.muted(motors::CHANNEL_ID) {
        Some(ACTUATOR_MOTORS_TELEM.subscriber().map_err(|_| {
            defmt::warn!("recorder: ACTUATOR_MOTORS_TELEM SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let motor_state_sub = if record_set.includes_motor_state() && !knobs.muted(motor_state::CHANNEL_ID) {
        Some(PROCESSED_MOTOR_STATE.subscriber().map_err(|_| {
            defmt::warn!("recorder: PROCESSED_MOTOR_STATE SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let power_sub = if record_set.includes_power() && !knobs.muted(power::CHANNEL_ID) {
        Some(POWER_TELEM.subscriber().map_err(|_| {
            defmt::warn!("recorder: POWER_TELEM SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let track_err_sub = if record_set.includes_tracking_error() && !knobs.muted(tracking_error::CHANNEL_ID) {
        Some(TRACKING_ERROR.subscriber().map_err(|_| {
            defmt::warn!("recorder: TRACKING_ERROR SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let ctrl_sp_sub = if record_set.includes_control_setpoint() && !knobs.muted(control_setpoint::CHANNEL_ID) {
        Some(CONTROL_SETPOINT_TELEM.subscriber().map_err(|_| {
            defmt::warn!("recorder: CONTROL_SETPOINT_TELEM SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    #[cfg(feature = "est_eskf")]
    let est_state_sub = if record_set.includes_estimator_state() && !knobs.muted(estimator_state::CHANNEL_ID) {
        Some(ESTIMATOR_BIAS_TELEM.subscriber().map_err(|_| {
            defmt::warn!("recorder: ESTIMATOR_BIAS_TELEM SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };

    // Snapshot the failsafe / estimator / mission state at session
    // open. The capture loop's edge poll only emits `/events`
    // records on transitions *from* these snapshots, so a session
    // that opens with the estimator already converged
    // (`ESTIMATOR_READY=true`, the normal arming-permitted state)
    // does **not** emit a spurious `KIND_ESTIMATOR_UP` immediately.
    // For the mission state machine the same rule applies: a
    // session armed during `Idle` (the typical case — pilot must
    // arm before triggering a mission) snapshots Idle and only
    // emits an event when the planner or controller advances the
    // state.
    let mut body = FlightRecorder {
        record_set,
        knobs,
        imu_sub,
        imu_raw_sub,
        rc_sub,
        att_sub,
        odom_sub,
        mpc_sub,
        mpc_cost_sub,
        motor_sub,
        motor_state_sub,
        power_sub,
        track_err_sub,
        ctrl_sp_sub,
        #[cfg(feature = "est_eskf")]
        est_state_sub,
        imu_seq: 0,
        imu_raw_seq: 0,
        rc_seq: 0,
        att_seq: 0,
        odom_seq: 0,
        mpc_seq: 0,
        mpc_cost_seq: 0,
        motor_seq: 0,
        motor_state_seq: 0,
        power_seq: 0,
        track_err_seq: 0,
        ctrl_sp_seq: 0,
        #[cfg(feature = "est_eskf")]
        est_state_seq: 0,
        health_seq: 0,
        last_imu_temp_c: f32::NAN,
        last_health_emit: Instant::from_ticks(0),
        gps_health_seq: 0,
        last_gps_health_poll: Instant::from_ticks(0),
        last_gps_health_emit: Instant::from_ticks(0),
        last_gps_health: None,
        ev_seq: 0,
        prev_failsafe: FAILSAFE_ACTIVE.load(Ordering::Acquire),
        prev_est_ready: ESTIMATOR_READY.load(Ordering::Acquire),
        prev_rc_link_healthy: RC_LINK_HEALTHY.load(Ordering::Acquire),
        #[cfg(feature = "outer_mpc")]
        prev_mission_state: MISSION_STATE.load(Ordering::Acquire),
        prev_voltage_stale: VOLTAGE_STALE.load(Ordering::Acquire),
        prev_inner_silent: INNER_SILENT_CAUSE.load(Ordering::Acquire),
        last_drops_reported: 0,
        last_drop_event: Instant::from_ticks(0),
        messages: 0,
        drops: 0,
        encode_overflows: 0,
    };

    let name = build_file_name(sequence);
    defmt::info!(
        "recorder: opening /{} (record_set={})",
        name.as_str(),
        record_set.name(),
    );
    let bytes = fat::write_file(store, name.as_str(), &mut body).await?;

    Ok(SessionSummary {
        bytes,
        messages: body.messages,
        drops: body.drops,
        encode_overflows: body.encode_overflows,
        file_name: name,
    })
}
