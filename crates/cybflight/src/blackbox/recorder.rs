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
//! mcap info   flight_0001.mcap        # 4 channels (imu, attitude, rc, events)
//! python read_mcap.py flight_0001.mcap | grep events
//! ```
//! → expect `kind=1` (ARM) at the start, `kind=2` (DISARM) and
//! `kind=16` (LOG_END) at the end.

use core::fmt::Write as _;
use core::future::pending;

use embassy_futures::select::{select6, Either6};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pubsub::{Subscriber, WaitResult};
use embassy_time::{Duration, Instant, Timer};
use embedded_io_async::Write;
use heapless::String;

use core::sync::atomic::Ordering;

use super::fat::{self, FileBody, OpError};
use super::mcap;
use super::record_set::RecordSet;
use super::sdmmc_block::SdmmcBlockStore;
use super::should_record;
use super::topics::{
    attitude, events, imu, motor_state, motors, mpc, odometry, rc, tracking_error,
};
use crate::control::failsafe::{FAILSAFE_ACTIVE, FAILSAFE_REASON, RC_LINK_HEALTHY};
use crate::control::{
    ACTUATOR_MOTORS_TELEM, OCP_SOLVER_OUTPUT, PROCESSED_MOTOR_STATE, TRACKING_ERROR,
};
use crate::control::TrackingError as TrackErrMsg;
#[cfg(feature = "outer_mpc")]
use crate::control::{MissionState, MISSION_STATE};
use crate::estimation::ESTIMATOR_READY;
use crate::msgs;
use crate::sensors::{IMU_1, RC_INPUT, VEHICLE_ATTITUDE, VEHICLE_ODOMETRY};

const MCAP_LIBRARY: &str = concat!("cybflight v", env!("CARGO_PKG_VERSION"));

/// Cadence of the IS_ARMED disarm-edge poll inside the capture loop.
/// 50 ms means worst-case latency to observe disarm = 50 ms (plus the
/// in-flight write_message that's currently blocked on FAT). The
/// recorder doesn't try harder than this — disarm-by-RC already has
/// its own debounce upstream.
const DISARM_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Per-topic drain budget per outer iteration for **small/mid-tier**
/// topics (`/rc`, `/attitude`, `/odometry`, `/mpc`). Chosen larger
/// than every channel's `CAP` (max is 8 for `VEHICLE_ODOMETRY`) so a
/// fully-buffered channel can be cleared in one pass, while still
/// being small enough that we re-check `should_record()` and rotate
/// across topics on a fast cadence.
///
/// **Why a bound is needed.** With an unbounded `try_next_message`
/// drain and an 8 kHz topic like `/imu1`, the producer fires faster
/// than each `emit_msg` SD write can complete. The subscriber stays
/// 1–2 messages behind the producer, `try_next_message` never
/// returns `None`, and the drain never exits. The outer loop then
/// never re-checks `should_record()` — user disarm goes unobserved,
/// the file is never closed, and on power-down the FAT directory
/// entry is left at the size set by `truncate()` (i.e. zero bytes).
const DRAIN_BUDGET_NORMAL: usize = 16;

/// Drain budget for **large-tier** topics — currently just `/imu1`.
/// Deliberately smaller than `DRAIN_BUDGET_NORMAL` so that under SD
/// backpressure (publisher faster than consumer) the drops fall on
/// IMU instead of on the smaller, more-critical topics.
///
/// Equal to `IMU_1`'s `CAP`. When the SD pipeline is keeping up,
/// every IMU sample fires at most once before we drain — the budget
/// is never reached and no IMU drops occur. Only when emit_msg is
/// slower than the 125 µs IMU period does the drain hit the budget;
/// at that point any further queued IMU messages get reported as
/// `Lagged` next iteration, which is exactly the behaviour we want
/// (IMU is the high-bandwidth debug stream — losing some samples is
/// preferable to starving `/attitude` / `/rc` / `/odometry`/ `/mpc`).
const DRAIN_BUDGET_DEPRIO: usize = 4;

type ImuSub = Subscriber<'static, CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>;
type AttSub = Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleAttitude, 4, 6, 1>;
type RcSub = Subscriber<'static, CriticalSectionRawMutex, msgs::RcInput, 4, 6, 1>;
type OdomSub = Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleOdometry, 8, 6, 1>;
type McpSub = Subscriber<'static, CriticalSectionRawMutex, msgs::OcpSolverOutput, 4, 4, 1>;
/// Motors subscriber over `ACTUATOR_MOTORS_TELEM`. CAP=2 matches the
/// channel; SUBS=3 leaves one slot for a future shell stream consumer
/// alongside esp_bridge + this recorder.
type MotorSub = Subscriber<'static, CriticalSectionRawMutex, msgs::ActuatorMotors, 2, 3, 1>;
/// Motor-state subscriber over `PROCESSED_MOTOR_STATE`. The
/// `commanded vs achieved` companion to `MotorSub` — same shape,
/// CAP=2/SUBS=3.
type MotorStateSub =
    Subscriber<'static, CriticalSectionRawMutex, msgs::MotorStateTelemetry, 2, 3, 1>;
/// Tracking-error subscriber over `TRACKING_ERROR`. CAP=4/SUBS=3/PUBS=3
/// — see [`crate::control::TRACKING_ERROR`] for the rationale.
type TrackErrSub = Subscriber<'static, CriticalSectionRawMutex, TrackErrMsg, 4, 3, 3>;

pub struct FlightRecorder {
    record_set: RecordSet,
    /// `Some` iff the active record set includes this topic.
    /// Subscribers are taken at session start and dropped on close
    /// — so a tier that excludes a topic doesn't even occupy a
    /// pubsub subscriber slot.
    imu_sub: Option<ImuSub>,
    att_sub: Option<AttSub>,
    rc_sub: Option<RcSub>,
    odom_sub: Option<OdomSub>,
    mpc_sub: Option<McpSub>,
    motor_sub: Option<MotorSub>,
    motor_state_sub: Option<MotorStateSub>,
    track_err_sub: Option<TrackErrSub>,
    imu_seq: u32,
    att_seq: u32,
    rc_seq: u32,
    odom_seq: u32,
    mpc_seq: u32,
    motor_seq: u32,
    motor_state_seq: u32,
    track_err_seq: u32,
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
    pub messages: u32,
    pub drops: u32,
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

        // ── schemas + channels (data-driven from the active record-set) ──
        // `def.channel_id` is stable per topic, so a Mid-tier file
        // and a Large-tier file both call `/imu1` channel 1.
        let topic_set = self.record_set.topic_set();
        for def in topic_set.iter() {
            mcap::write_schema(w, def.channel_id, def.schema_name, "jsonschema", def.schema_data)
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
        for def in topic_set.iter() {
            mcap::write_channel(w, def.channel_id, def.channel_id, def.topic, "cbor").await?;
            total += (1 + 8 + 2 + 2 + 4 + def.topic.len() + 4 + "cbor".len() + 4) as u32;
        }

        // ── ARM event ───────────────────────────────────────────
        let mut scratch = [0u8; 192];
        if self.record_set.includes_events() {
            total += self
                .emit_event(w, &mut scratch, events::KIND_ARM, 0)
                .await?;
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
        //    arms (`/attitude`, `/rc`, `/odometry`, `/mpc`) get
        //    woken but never reached. After the wait fires, drain
        //    every remaining ready message on every sub
        //    synchronously via `try_next_message`. That guarantees
        //    forward progress on all topics each iteration
        //    regardless of select bias.
        //
        // The fairness drain was added after a flight where a
        // `Large` tier session yielded 1037 IMU messages and zero
        // of everything else — including 8 kHz `/attitude`.
        loop {
            if !should_record() {
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
            // `motors` (`/motors`, INDI inner-loop output) is
            // intentionally NOT in the select arms — `select6` is
            // the maximum named arity and the fairness drain below
            // covers it. With IMU at 8 kHz waking the loop every
            // 125 µs (and the 50 ms timer floor when nothing is
            // publishing), the 100 Hz motors stream is drained well
            // within its CAP=2 channel before lag accrues.
            match select6(
                timer,
                next_or_pend(&mut self.rc_sub),
                next_or_pend(&mut self.att_sub),
                next_or_pend(&mut self.odom_sub),
                next_or_pend(&mut self.mpc_sub),
                next_or_pend(&mut self.imu_sub),
            )
            .await
            {
                Either6::First(()) => {
                    // poll-wake — re-check IS_ARMED at top of loop
                }
                Either6::Second(wr) => total += self.emit_rc(w, &mut scratch, wr).await?,
                Either6::Third(wr) => total += self.emit_att(w, &mut scratch, wr).await?,
                Either6::Fourth(wr) => total += self.emit_odom(w, &mut scratch, wr).await?,
                Either6::Fifth(wr) => total += self.emit_mpc(w, &mut scratch, wr).await?,
                Either6::Sixth(wr) => total += self.emit_imu(w, &mut scratch, wr).await?,
            }

            // Fairness drain — see comment above. Drains run in
            // **tier order**: small (rc) → mid (attitude, odometry,
            // mpc, motors) → large (imu). The large-tier drain uses
            // `DRAIN_BUDGET_DEPRIO` (smaller), so when the SD
            // pipeline can't keep up the drops naturally land on
            // IMU instead of the smaller / more-critical topics.
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
                let next = self
                    .track_err_sub
                    .as_mut()
                    .and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_track_err(w, &mut scratch, wr).await?;
            }
            // ── large tier (deprioritised) ──────────────────────
            for _ in 0..DRAIN_BUDGET_DEPRIO {
                let next = self.imu_sub.as_mut().and_then(|s| s.try_next_message());
                let Some(wr) = next else { break };
                total += self.emit_imu(w, &mut scratch, wr).await?;
            }

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
            }
        }

        // ── DISARM + LOG_END events ─────────────────────────────
        if self.record_set.includes_events() {
            total += self
                .emit_event(w, &mut scratch, events::KIND_DISARM, 0)
                .await?;
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
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.imu_seq = self.imu_seq.wrapping_add(1);
        let n = imu::encode(scratch, &m).unwrap_or(0);
        let bytes = emit_msg(
            w,
            imu::CHANNEL_ID,
            self.imu_seq,
            m.timestamp,
            &scratch[..n],
        )
        .await?;
        self.messages = self.messages.wrapping_add(1);
        Ok(bytes)
    }

    async fn emit_att<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::VehicleAttitude>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.att_seq = self.att_seq.wrapping_add(1);
        let n = attitude::encode(scratch, &m).unwrap_or(0);
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

    async fn emit_rc<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::RcInput>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.rc_seq = self.rc_seq.wrapping_add(1);
        let n = rc::encode(scratch, &m).unwrap_or(0);
        let bytes = emit_msg(
            w,
            rc::CHANNEL_ID,
            self.rc_seq,
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
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.odom_seq = self.odom_seq.wrapping_add(1);
        let n = odometry::encode(scratch, &m).unwrap_or(0);
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

    async fn emit_mpc<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<msgs::OcpSolverOutput>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.mpc_seq = self.mpc_seq.wrapping_add(1);
        let n = mpc::encode(scratch, &m).unwrap_or(0);
        let bytes = emit_msg(
            w,
            mpc::CHANNEL_ID,
            self.mpc_seq,
            m.timestamp,
            &scratch[..n],
        )
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
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.motor_seq = self.motor_seq.wrapping_add(1);
        let n = motors::encode(scratch, &m).unwrap_or(0);
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
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.motor_state_seq = self.motor_state_seq.wrapping_add(1);
        let n = motor_state::encode(scratch, &m).unwrap_or(0);
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

    async fn emit_track_err<W: Write>(
        &mut self,
        w: &mut W,
        scratch: &mut [u8],
        wr: WaitResult<TrackErrMsg>,
    ) -> Result<u32, W::Error> {
        let Some(m) = handle_wr(wr, &mut self.drops) else {
            return Ok(0);
        };
        self.track_err_seq = self.track_err_seq.wrapping_add(1);
        let n = tracking_error::encode(scratch, &m).unwrap_or(0);
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
        let n = events::encode(scratch, now, kind, data).unwrap_or(0);
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

        Ok(total)
    }
}

fn handle_wr<M>(wr: WaitResult<M>, drops: &mut u32) -> Option<M> {
    match wr {
        WaitResult::Message(m) => Some(m),
        WaitResult::Lagged(n) => {
            *drops = drops.saturating_add(n as u32);
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
    let imu_sub = if record_set.includes_imu() {
        Some(IMU_1.subscriber().map_err(|_| {
            defmt::warn!("recorder: IMU_1 SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let att_sub = if record_set.includes_attitude() {
        Some(VEHICLE_ATTITUDE.subscriber().map_err(|_| {
            defmt::warn!("recorder: VEHICLE_ATTITUDE SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let rc_sub = if record_set.includes_rc() {
        Some(RC_INPUT.subscriber().map_err(|_| {
            defmt::warn!("recorder: RC_INPUT SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let odom_sub = if record_set.includes_odometry() {
        Some(VEHICLE_ODOMETRY.subscriber().map_err(|_| {
            defmt::warn!("recorder: VEHICLE_ODOMETRY SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let mpc_sub = if record_set.includes_mpc() {
        Some(OCP_SOLVER_OUTPUT.subscriber().map_err(|_| {
            defmt::warn!("recorder: OCP_SOLVER_OUTPUT SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let motor_sub = if record_set.includes_motors() {
        Some(ACTUATOR_MOTORS_TELEM.subscriber().map_err(|_| {
            defmt::warn!("recorder: ACTUATOR_MOTORS_TELEM SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let motor_state_sub = if record_set.includes_motor_state() {
        Some(PROCESSED_MOTOR_STATE.subscriber().map_err(|_| {
            defmt::warn!("recorder: PROCESSED_MOTOR_STATE SUBS exhausted");
            OpError::NoSubscriberSlot
        })?)
    } else {
        None
    };
    let track_err_sub = if record_set.includes_tracking_error() {
        Some(TRACKING_ERROR.subscriber().map_err(|_| {
            defmt::warn!("recorder: TRACKING_ERROR SUBS exhausted");
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
        imu_sub,
        att_sub,
        rc_sub,
        odom_sub,
        mpc_sub,
        motor_sub,
        motor_state_sub,
        track_err_sub,
        imu_seq: 0,
        att_seq: 0,
        rc_seq: 0,
        odom_seq: 0,
        mpc_seq: 0,
        motor_seq: 0,
        motor_state_seq: 0,
        track_err_seq: 0,
        ev_seq: 0,
        prev_failsafe: FAILSAFE_ACTIVE.load(Ordering::Acquire),
        prev_est_ready: ESTIMATOR_READY.load(Ordering::Acquire),
        prev_rc_link_healthy: RC_LINK_HEALTHY.load(Ordering::Acquire),
        #[cfg(feature = "outer_mpc")]
        prev_mission_state: MISSION_STATE.load(Ordering::Acquire),
        messages: 0,
        drops: 0,
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
        file_name: name,
    })
}
