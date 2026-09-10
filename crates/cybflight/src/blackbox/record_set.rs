//! Record-set system: a four-tier classification (`none`, `small`,
//! `mid`, `large`) selecting which topics a session logs.
//!
//! ## Design
//!
//! The cybflight blackbox has a growing catalogue of topics. Rather
//! than letting each capture op hard-code its own topic list, the
//! recorder reads [`BLACKBOX_RECORD_SET`] once at session start and
//! emits exactly the topics in that tier's
//! [`topic_set`](RecordSet::topic_set).
//!
//! `none` → `small` → `mid` → `large` are **monotonic**: each is a
//! strict superset of the previous. One axis, one knob, "turn it up".
//! `sysid` sits beside that axis rather than on top of it — a
//! purpose-built set for `analysis/sysid_mcap.py` (2026-08-24).
//!
//! ## Tier definitions
//!
//! | Tier  | Topics |
//! |-------|--------|
//! | none  | (nothing — recording disabled) |
//! | small | events + rc |
//! | mid   | small + imu1 + attitude + motors + motor_state + health + gps_health + mpc_cost |
//! | large | mid + odometry + mpc + tracking_error + control_setpoint + estimator_state |
//! | sysid | events + imu1_raw + motors + motor_state + odometry + health + power, INDI telemetry at ≥500 Hz instead of 100 Hz |
//!
//! Mid is "what the vehicle did": the IMU, the attitude it believed,
//! the motors it commanded and achieved, and health. Large adds "why":
//! the estimator state and the outer-loop command chain. Sysid is
//! exactly what the identification script consumes — the pre-filter
//! IMU instead of the filtered one, the commanded/achieved motor pair
//! at frame rate, and odometry for the drag fit — and nothing else,
//! because at a true 8 kHz IMU the old all-topics set (~1.2 MB/s
//! measured) was twice what the card path sustains (~600 KB/s) and the
//! recorder lost half of every topic, including the ones the fits
//! need.
//!
//! ## Bandwidth, and why it has a rate axis
//!
//! Only `/imu1` and `/imu1_raw` scale with [`crate::rates::IMU_ODR_HZ`].
//! Everything else is rate-invariant: the ESKF decimates its predict to
//! ~1 kHz in every build, so `/odometry` is 1 kHz at
//! both IMU rates, and the INDI telemetry mirrors publish at a fixed
//! 100 Hz (`indi_task`'s `outer_decimation = loop_rate_hz / 100`, where
//! `loop_rate_hz = IMU_ODR_HZ / indi_ctrl_div` is the control rate).
//!
//! So the same tier name means very different things on an `imu_1khz`
//! build than on an 8 kHz one:
//!
//! | Tier  | 1 kHz build | 8 kHz build |
//! |-------|-------------|-------------|
//! | none  | 0 | 0 |
//! | small | ~19 KB/s (~1 MB/min) | ~19 KB/s (~1 MB/min) |
//! | mid   | ~110 KB/s (~7 MB/min) | ~540 KB/s (~32 MB/min) |
//! | large | ~230 KB/s (~14 MB/min) | ~660 KB/s (~40 MB/min) |
//! | sysid | ~129 KiB/s (~8 MB/min) | ~129 KiB/s (~8 MB/min) |
//!
//! `sysid` no longer scales with the IMU rate: since 2026-09-09 every
//! one of its topics is decimated at its publisher to what
//! `analysis/sysid_mcap.py` actually needs (`/imu1_raw` by
//! `blackbox_rate_div`, `/odometry` and `/power` by their own consts),
//! so the figure above holds on both builds. [`RecordSet::estimated_bytes_per_s`]
//! computes it at session start and warns when a tier does not fit —
//! prefer it over this hand-maintained table.
//!
//! Estimates (2026-08-24 restructure): on-disk message size measured
//! from a bench log (62 B for the IMU topics incl. the 22 B MCAP
//! record header, 81/112 B motors/motor_state, 160 B tracking_error)
//! × nominal publish rate, no drops. `/odometry` (~106 KB/s) counts
//! only when an estimator runs. The card path sustained ~600 KB/s on
//! the bench, so `large` and `sysid` at 8 kHz sit at the ceiling —
//! `blackbox_rate_div 2` for margin. Message sizes are exact (CBOR is deterministic here); the
//! rates assume MPC at its 50 Hz default, RC at 150 Hz, and
//! `/gps_health` on its 1 Hz heartbeat (the `pos_source: mocap` case —
//! a GPS build emits it on every NAV-PVT instead, adding a few KB/s).
//! Those sub-100 Hz topics (now including the ~100 Hz Mahony
//! `/attitude` at ~5.5 KB/s) total ~90 KB/s in every build, which is
//! noise next to the IMU stream.
//!
//! Two consequences worth keeping in mind:
//!
//! - **`/imu1_raw` costs exactly what `/imu1` costs** — same encoder,
//!   same rate. `large` is not a small increment on `mid`; it is `mid`
//!   plus a second full-rate IMU stream.
//! - **At 8 kHz the dominant cost is `/imu1` (~568 KB/s in the
//!   `Imu.v2` array format); at 1 kHz it is `/odometry` (~106 KB/s,
//!   against 71 KB/s for `/imu1`).** Both hot topics use the compact
//!   positional-array wire format from
//!   [`cybflight_core::blackbox_wire`]; the low-rate topics keep
//!   their readable string-keyed maps, which cost ~80 KB/s combined.
//!
//! ## `/attitude` — dropped once, revived with new semantics
//!
//! The topic was originally dropped because the ESKF published
//! `/attitude` and `/odometry` from the same statement with the same
//! `eskf.orientation()` — byte-identical quaternions at 1 kHz, 86 KB/s
//! for zero information.
//!
//! It is back as a **different signal**: the always-on Mahony IMU-only
//! filter (`estimation::mahony_task`, ~100 Hz, ~5.5 KB/s). That makes
//! it the only attitude record on builds with no mocap and no GPS
//! (`outer_rate`, where no ESKF runs and `/odometry` is silent), and an
//! independent cross-check against `/odometry.pose.orientation` on
//! ESKF builds — a mocap/GPS-corruption postmortem can tell "external
//! reference went bad" from "IMU went bad". Channel id 2 was reserved
//! across the gap precisely so this revival costs no id churn.
//!
//! "Small" is the *cheap* tier — arm/disarm brackets and pilot
//! intent only. "Mid" is the **default flight-debug tier**: IMU-rate
//! post-LP IMU, the Mahony attitude, the commanded-vs-achieved motor
//! pair (`/motors`, `/motor_state`) and health. "Large" adds the
//! reasoning behind it: 1 kHz ESKF fused odometry (whose
//! `pose.orientation` is the fused attitude — see below), outer-loop
//! MPC commands, the rate-setpoint mirror (`/control_setpoint`), the
//! controller tracking error and the slow estimator-bias snapshot
//! (`/estimator_state`) — the full
//! `RC → MPC command → ESKF state → attitude estimate → motor` chain.
//! "Sysid" swaps the filtered IMU for the pre-biquad-LP raw stream
//! (filter tuning, analyse.py-style RPM-notch fits, and the
//! identification script itself) and drops the debug chain.
//!
//! ## Tier-aware drop priority
//!
//! Tiers are *also* a drop-priority ordering. Each iteration of the
//! recorder loop drains topics in tier **order** — small, then mid,
//! then large (raw IMU before filtered IMU within the deprio
//! block). That ordering is what does the prioritising: when an
//! iteration runs short, the shortfall lands on whatever drains
//! last. `/rc`, `/odometry` and the controller stream
//! have already had their turn, so IMU is what shows up as `Lagged`
//! (counted in the session summary, and — since the drop-aware
//! sequence fix — visible as a gap in the affected channel's MCAP
//! `sequence`). The motivating intuition: IMU is the
//! high-bandwidth debug stream we're most willing to lose under
//! contention; smaller, lower-rate flight-state topics are the ones
//! we can't afford to drop.
//!
//! The per-iteration budgets are a forward-progress bound, not the
//! prioritisation mechanism. `DRAIN_BUDGET_DEPRIO` tracks
//! [`crate::rates::IMU_PUBSUB_CAP`] so a channel that filled during
//! a stall can still be emptied in one pass; `DRAIN_BUDGET_NORMAL`
//! tracks [`crate::rates::BLACKBOX_ODOM_PUBSUB_CAP`] (25, the deepest
//! small/mid channel) for the same reason.
//!
//! **Rate caveat.** This ordering was calibrated when `/imu1` was
//! unambiguously the fat stream, which is true at 8 kHz and *not* at
//! 1 kHz — see the bandwidth table above, where `/odometry` alone
//! outweighs `/imu1` on an `imu_1khz` build. On those builds the
//! drain order sheds the sysid signal first while faithfully
//! recording 1 kHz odometry that would lose little decimated to
//! 200 Hz. Revisit if 1 kHz builds start reporting drops.
//!
//! `none` exists as an explicit "off" so the recorder can be muted
//! at the BSP level without touching `RECORDER_HOLD` or arm-edge
//! logic. With `none` selected, an arm rising edge does **not**
//! open a file; the task observes the tier and skips the session.
//!
//! ## Channel ids
//!
//! Each topic carries a stable [`channel_id`](super::topics::TopicDef::channel_id)
//! independent of its position in any per-tier slice. So `/imu1` is
//! channel 1 in every file regardless of tier, `/rc` is
//! channel 3, etc. Ids are assigned from `topics::ALL` and are
//! never reissued, so id 2 stays reserved for `/attitude` even
//! though no tier currently logs it. Foxglove / `mcap cat` consumers can rely on this
//! across files captured at different tiers.

use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use super::topics::{self, TopicDef};

// ── Per-tier topic-set arrays (static slices) ──────────────────────────────
//
// These arrays are the single source of truth for "which topics are
// in tier X". The recorder asks the `RecordSet` enum
// for the slice and emit schemas/channels by iterating it.
//
// Order within an array doesn't affect channel ids (those live in
// each TopicDef), but it does affect the file-write order of the
// schema/channel records, which is purely cosmetic for MCAP
// consumers.

const TOPICS_SMALL: &[TopicDef] = &[topics::events::DEF, topics::rc::DEF];

const TOPICS_MID: &[TopicDef] = &[
    topics::events::DEF,
    topics::rc::DEF,
    topics::imu::DEF,
    topics::attitude::DEF,
    topics::motors::DEF,
    topics::motor_state::DEF,
    topics::health::DEF,
    topics::gps_health::DEF,
    topics::mpc_cost::DEF,
];

const TOPICS_LARGE: &[TopicDef] = &[
    topics::events::DEF,
    topics::rc::DEF,
    topics::imu::DEF,
    topics::attitude::DEF,
    topics::motors::DEF,
    topics::motor_state::DEF,
    topics::health::DEF,
    topics::gps_health::DEF,
    topics::odometry::DEF,
    topics::mpc::DEF,
    topics::mpc_cost::DEF,
    topics::tracking_error::DEF,
    topics::control_setpoint::DEF,
    #[cfg(feature = "est_eskf")]
    topics::estimator_state::DEF,
];

/// Exactly what `analysis/sysid_mcap.py` reads, plus the Small
/// bracket topics. Not a superset of Large: `/imu1_raw` replaces
/// `/imu1`, and the flight-debug stream is left out so an 8 kHz
/// session fits the card (see module docs).
const TOPICS_SYSID: &[TopicDef] = &[
    topics::events::DEF,
    topics::imu_raw::DEF,
    topics::motors::DEF,
    topics::motor_state::DEF,
    topics::odometry::DEF,
    // Estimator/failsafe state (ESKF gate counters, fault bitfield),
    // INDI loop timing and RC link statistics. A fixed 20 Hz — NOT
    // change-driven (that is `/gps_health`), so every field is paid 20×
    // per second; the positional `.v2` form keeps that to ~100 B × 20 Hz
    // ≈ 2 KiB/s. The only way to read an ESTIMATOR_DOWN after the fact —
    // see `includes_health`.
    topics::health::DEF,
    // Pack voltage raw + filtered, current — decimated to 10 Hz at the
    // publisher (`rates::BLACKBOX_POWER_DECIM`) because the thrust table
    // is linearized at the *filtered* voltage, which `batt_lpf_hz` holds
    // to 2 Hz. ~117 B × 10 Hz ≈ 1.1 KiB/s. Resolving the raw transient
    // sag under a punch would need the full 100 Hz — see that const.
    topics::power::DEF,
];

// ── Channel-id uniqueness (compile-time) ───────────────────────────────────
//
// MCAP identifies both Channel and Schema records by a u16 id, and the
// recorder emits one of each per topic using `TopicDef::channel_id` for
// both. A file containing two non-identical records under one id is
// malformed: readers either reject it or keep whichever was written
// last, in which case the other topic's messages are silently
// mislabelled and decoded against the wrong schema.
//
// This went wrong once already — ids 10 and 11 were each issued twice,
// so every Mid file mislabelled `/control_setpoint` as `/gps_health`,
// and every Large file additionally mislabelled `/imu1_raw` (the entire
// reason that tier exists) as `/health`. Nothing caught it because
// nothing checked. These assertions are that check.

/// True if any two entries of `set` share a `channel_id`.
///
/// O(n²) over a ≤16-element list, evaluated once at compile time.
const fn has_duplicate_channel_id(set: &[TopicDef]) -> bool {
    let mut i = 0;
    while i < set.len() {
        let mut j = i + 1;
        while j < set.len() {
            if set[i].channel_id == set[j].channel_id {
                return true;
            }
            j += 1;
        }
        i += 1;
    }
    false
}

const _: () = {
    // The real invariant: ids are globally unique, so a topic keeps its
    // id whichever tiers it appears in and no future tier combination
    // can collide.
    assert!(
        !has_duplicate_channel_id(topics::ALL),
        "two blackbox topics share a channel_id — see topics::ALL",
    );
    // Per-tier too: `ALL` being unique does not stop a tier array from
    // listing the same topic twice, which would emit duplicate records
    // just the same.
    assert!(
        !has_duplicate_channel_id(TOPICS_SMALL),
        "TOPICS_SMALL emits a duplicate channel_id",
    );
    assert!(
        !has_duplicate_channel_id(TOPICS_MID),
        "TOPICS_MID emits a duplicate channel_id",
    );
    assert!(
        !has_duplicate_channel_id(TOPICS_LARGE),
        "TOPICS_LARGE emits a duplicate channel_id",
    );
    assert!(
        !has_duplicate_channel_id(TOPICS_SYSID),
        "TOPICS_SYSID emits a duplicate channel_id",
    );
};

/// Recording tier. See module docs for byte-rate estimates per tier
/// and the monotonic-superset property.
#[derive(Clone, Copy, PartialEq, Eq, defmt::Format)]
#[repr(u8)]
pub enum RecordSet {
    /// Recording is muted. Arm-edge does not open a file; the
    /// session is skipped. Use this when a board ships with storage
    /// but no SD card is fitted, or when tests don't want files
    /// piling up.
    None = 0,
    /// Events + RC. Cheapest tier — preserves arm/disarm brackets
    /// and pilot intent without paying for high-rate state.
    Small = 1,
    /// Small + what the vehicle did: post-LP IMU (`/imu1`), Mahony
    /// attitude, the commanded-vs-achieved motor pair (`/motors`,
    /// `/motor_state`), health, gps_health.
    /// **The default flight-debug tier.**
    Mid = 2,
    /// Mid + why it did it: ESKF odometry (pose+twist, incl. the fused
    /// attitude quaternion), mpc, tracking_error, control_setpoint,
    /// estimator_state — the full outer-loop / estimator chain.
    Large = 3,
    /// Events + `/imu1_raw` + `/motors` + `/motor_state` + `/odometry` + `/health` + `/power`,
    /// with the INDI inner-loop telemetry mirrors published at
    /// ≥500 Hz instead of 100 Hz (actuated inside `indi_task` via
    /// [`RecordSet::fast_indi_telem`]). Exactly the topics
    /// `analysis/sysid_mcap.py` reads plus `/mpc` (the outer-loop
    /// command and solver telemetry, ≤ 100 Hz); not a superset of Large.
    ///
    /// Exists because 100 Hz is below the 10–30 ms motor time
    /// constants `/motor_state` measures: for actuator sysid the
    /// commanded increment and the measured response must be sampled
    /// on the same timescale as the dynamics, which is the whole
    /// reason indiflight logs its `u`/`omega`/`omega_dot` at frame
    /// rate.
    ///
    /// The trimmed topic set is what makes it recordable: at a true
    /// 8 kHz IMU the old all-topics set demanded ~1.2 MB/s against the
    /// ~600 KB/s the card path sustains (bench, 2026-08-24) and the
    /// recorder lost half of every topic; this set is ~600 KB/s
    /// (raw 491 + motors 47 + motor_state 62 + rc/events) — at the
    /// ceiling; `blackbox_rate_div 2` gives margin.
    Sysid = 4,
}

impl RecordSet {
    /// Compile-time default selected at boot.
    ///
    /// `Mid` covers the routine flight-debug case — ~29 MB/min on an
    /// `imu_1khz` build, ~79 MB/min at 8 kHz. Power users opt up to
    /// `Large` when they specifically want the raw IMU stream; nobody
    /// opts up to `Large` "just in case" and discovers a 1 GB file
    /// from an 8-minute flight.
    pub const DEFAULT: Self = Self::Mid;

    /// True if this tier should record at all (anything except `None`).
    #[inline]
    pub const fn enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    /// True for tiers where `indi_task` publishes its telemetry
    /// mirrors (`/motors`, `/motor_state`, `/tracking_error`) at the
    /// sysid rate (≥500 Hz) instead of the 100 Hz default. Read by
    /// the publisher once per telemetry wrap, so a shell-side tier
    /// change takes effect within ~2 ms without re-arming.
    #[inline]
    pub const fn fast_indi_telem(self) -> bool {
        matches!(self, Self::Sysid)
    }

    #[inline]
    pub const fn includes_imu(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_attitude(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_odometry(self) -> bool {
        matches!(self, Self::Large | Self::Sysid)
    }

    #[inline]
    pub const fn includes_mpc(self) -> bool {
        // Not Sysid: `sysid_mcap.py` does not read `/mpc` either, and
        // the tier is bandwidth-bound (see the module docs' budget).
        matches!(self, Self::Large)
    }

    /// `/mpc_cost` rides in Mid: the learned-cost ramp-in workflow
    /// (docs/learned_mpc_cost_deploy.md) reads `z` / effective-weight
    /// traces from ordinary flights, and at ≤ 200 Hz × ~150 B it is
    /// noise next to the Mid IMU stream. The publisher only emits while
    /// `mpc_learned_cost` is enabled, so vehicles without a baked
    /// policy pay nothing.
    #[inline]
    pub const fn includes_mpc_cost(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_motors(self) -> bool {
        matches!(self, Self::Mid | Self::Large | Self::Sysid)
    }

    #[inline]
    pub const fn includes_motor_state(self) -> bool {
        matches!(self, Self::Mid | Self::Large | Self::Sysid)
    }

    /// True for tiers that include the `/power` topic (raw +
    /// filtered pack voltage, current, at the 100 Hz `power_task`
    /// tick). Sysid only: it exists to fit the voltage filter and
    /// the thrust table's sag dependence, not for flight debugging.
    #[inline]
    pub const fn includes_power(self) -> bool {
        matches!(self, Self::Sysid)
    }

    /// True for tiers that include the `/health` estimator-fault
    /// snapshot (Mid + Large). Self-throttled to ~20 Hz inside the
    /// recorder loop.
    #[inline]
    pub const fn includes_health(self) -> bool {
        // Sysid included since 2026-09-09: `eskf_faults`,
        // `gate_rejects_*` and `last_pos_update_ns` are the only record
        // that distinguishes "mocap stopped delivering" from "the guard
        // is rejecting fresh poses". Without it an ESTIMATOR_DOWN in the
        // log is uninterpretable, which is exactly where the flight_0008
        // postmortem ran out of evidence. ~370 B at 20 Hz.
        matches!(self, Self::Mid | Self::Large | Self::Sysid)
    }

    /// True for tiers that include the `/gps_health` flat-snapshot
    /// of `crate::sensors::gps::GPS_HEALTH` + `LATEST_NAV_PVT`.
    /// Self-throttled to ~20 Hz; stays at `NotConfigured` on
    /// `est_pos_mocap` builds (still emits, just with the inert
    /// payload, so a file always carries an explicit "GPS not
    /// present" marker).
    #[inline]
    pub const fn includes_gps_health(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_tracking_error(self) -> bool {
        matches!(self, Self::Large)
    }

    /// True for tiers that include the `/control_setpoint` topic
    /// (mirror of `RATE_COMMAND` — outer-loop output to INDI).
    /// Mid + Large: ~5 KB/min, completes the
    /// `command → tracking_error → motor` triangle the rest of Mid
    /// already carries.
    #[inline]
    pub const fn includes_control_setpoint(self) -> bool {
        matches!(self, Self::Large)
    }

    /// True for tiers that include the `/estimator_state` topic
    /// (decimated ESKF gyro/accel bias estimates, ~10 Hz). Mid +
    /// Large and `est_eskf`-only on the recorder side — pure-Mahony
    /// builds have no biases to publish, so the topic is silently
    /// absent from the file.
    #[inline]
    pub const fn includes_estimator_state(self) -> bool {
        matches!(self, Self::Large)
    }

    /// True for tiers that include the `/imu1_raw` topic
    /// (pre-biquad-LP IMU samples at the same IMU-rate cadence as
    /// `/imu1`). **Sysid only** — same encoder and rate as `/imu1`,
    /// which it replaces there; for filter-tuning / RPM-notch fit
    /// work and the identification script.
    #[inline]
    pub const fn includes_imu_raw(self) -> bool {
        matches!(self, Self::Sysid)
    }

    #[inline]
    pub const fn includes_rc(self) -> bool {
        // Not Sysid: `sysid_mcap.py` never reads `/rc`, and at 244 Hz ×
        // 118 B it was 28 KiB/s — a fifth of the recorder's whole
        // delivered budget — spent on a topic no fit consumes. Flight
        // status for the fits comes from `/events`, which is unmutable.
        matches!(self, Self::Small | Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_events(self) -> bool {
        matches!(self, Self::Small | Self::Mid | Self::Large | Self::Sysid)
    }

    /// Static slice of every topic this tier emits. Used by capture
    /// ops to emit the schema/channel records before the data loop.
    /// Empty for [`RecordSet::None`].
    #[inline]
    pub const fn topic_set(self) -> &'static [TopicDef] {
        match self {
            Self::None => &[],
            Self::Small => TOPICS_SMALL,
            Self::Mid => TOPICS_MID,
            // Sysid logs the same topics as Large; the tiers differ
            // only in the INDI telemetry publish rate.
            Self::Large => TOPICS_LARGE,
            Self::Sysid => TOPICS_SYSID,
        }
    }

    /// Lowercase name used by shell + persistence. Stable on disk.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Small => "small",
            Self::Mid => "mid",
            Self::Large => "large",
            Self::Sysid => "sysid",
        }
    }

    /// Inverse of [`name`](Self::name). Returns `None` on
    /// unrecognised input so the shell can echo a usage line.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "small" => Some(Self::Small),
            "mid" => Some(Self::Mid),
            "large" => Some(Self::Large),
            "sysid" => Some(Self::Sysid),
            _ => None,
        }
    }

    /// Decode the `repr(u8)` value back to `RecordSet`. Used to
    /// read [`BLACKBOX_RECORD_SET`] without an unsafe transmute.
    /// Out-of-range bytes (only possible from a corrupted store
    /// or a downgraded firmware reading a flash slot last written
    /// by a build that defined more variants) fall back to
    /// [`Self::DEFAULT`].
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::None,
            1 => Self::Small,
            2 => Self::Mid,
            3 => Self::Large,
            4 => Self::Sysid,
            _ => Self::DEFAULT,
        }
    }
}

/// The currently-selected record set. Read at every session start;
/// written by the `blackbox set <tier>` shell command (and, in the
/// future, by a flash-persisted param). `AtomicU8` so the shell can
/// update it without a critical section.
pub static BLACKBOX_RECORD_SET: AtomicU8 = AtomicU8::new(RecordSet::DEFAULT as u8);

/// Snapshot the current record set.
#[inline]
pub fn current() -> RecordSet {
    RecordSet::from_u8(BLACKBOX_RECORD_SET.load(Ordering::Acquire))
}

/// Replace the current record set.
#[inline]
pub fn set(rs: RecordSet) {
    BLACKBOX_RECORD_SET.store(rs as u8, Ordering::Release);
}

/// Blackbox rate divider (`blackbox_rate_div`), mirrored out of the
/// param store so a **publisher** can honour it.
///
/// **Why a publisher reads this.** Decimating inside the recorder —
/// pulling a message off the channel and throwing it away — pays the
/// full price of the sample and gets nothing: the message still
/// occupied a PubSub slot, and it still burned one of the drain
/// budget's iterations. So a post-buffer divider does not free
/// bandwidth, it just lowers the records the recorder can emit per
/// pass, one for one. Measured 2026-09-09 on
/// `analysis/datasets/indoor_exp_timeopt_learned` at `rate_div 2`:
/// `/imu1_raw` drain passes went from a ceiling of 24 records to
/// exactly 12, and the logged sample rate *fell* 489 → 268 Hz.
///
/// Ahead of the buffer the same divider is unambiguously a win: the
/// channel then buffers emitted **records**, so CAP covers
/// `div ×` the wall-clock it used to, and the bytes really do halve.
/// That is only sound for a channel no control task reads — see
/// [`rate_div`] for which topics qualify.
///
/// `AtomicU32` so the sensor task can load it without a critical
/// section; 0 is normalised to 1 by [`rate_div`].
pub static BLACKBOX_RATE_DIV: AtomicU32 = AtomicU32::new(1);

/// Snapshot the blackbox rate divider, normalised to `>= 1`.
///
/// Honoured **at the publisher** for `sensors::IMU_1_RAW`, whose only
/// subscriber is the recorder (`board_init` wires it, nothing else
/// reads it), so dividing it ahead of the channel cannot change what
/// any control task sees.
///
/// Deliberately NOT applied to `IMU_1` or `VEHICLE_ODOMETRY`: both
/// feed the inner loop, the ESKF and the outer loop, so decimating
/// them at the publisher would hand control staler samples to save
/// log bytes. Dividing them behind the buffer is the trade this knob
/// exists to avoid, so the recorder leaves them at full rate and the
/// tier (`blackbox set`) is the lever for their bandwidth.
#[inline]
pub fn rate_div() -> u32 {
    BLACKBOX_RATE_DIV.load(Ordering::Acquire).max(1)
}

/// Replace the blackbox rate divider.
#[inline]
pub fn set_rate_div(div: u32) {
    BLACKBOX_RATE_DIV.store(div.max(1), Ordering::Release);
}

/// Recorder goodput this firmware can actually sustain, in bytes/s.
///
/// NOT the card's rate (~600 KB/s on the bench). The recorder shares
/// the thread executor with the outer loop, so what it gets is the
/// card rate times its share of the CPU: measured 2026-09-09 on
/// `sakura_bench_hunter_indoor`, 181 KiB/s with `mpc_learned_cost 0`
/// and 140 KiB/s with the policy on (a ~7.3 ms non-yielding solve
/// every 10 ms at `mpc_rate_hz 100`). The lower figure is the one to
/// budget against, because that is the configuration that flies.
pub const SUSTAINED_GOODPUT_B_S: u32 = 140 * 1024;

impl RecordSet {
    /// Rough on-disk bytes/s this tier asks for, given the session's rate
    /// divider and mute mask.
    ///
    /// Per-record sizes are measured medians from the 2026-09 indoor logs
    /// (payload + the 22 B MCAP record header); rates come from
    /// `crate::rates` and [`Self::fast_indi_telem`]. Approximate by
    /// construction — it exists to catch an over-subscribed tier at the
    /// order-of-magnitude level, which is the failure that actually
    /// happened: `Sysid` asked for ~301 KiB/s against ~140 available and
    /// nothing in the firmware said so. Every topic then lost ~50 % and
    /// the losses were only visible by forensics on the file afterwards.
    pub fn estimated_bytes_per_s(self, rate_div: u32, mute_mask: u32) -> u32 {
        if !self.enabled() {
            return 0;
        }
        let div = rate_div.max(1) as f32;
        let imu_hz = crate::rates::IMU_ODR_HZ;
        let indi_hz = if self.fast_indi_telem() { 500.0 } else { 100.0 };
        // (included?, channel id, bytes/record, records/s)
        let rows: [(bool, u16, f32, f32); 9] = [
            (self.includes_imu(), topics::imu::CHANNEL_ID, 62.0, imu_hz / div),
            (self.includes_imu_raw(), topics::imu_raw::CHANNEL_ID, 62.0, imu_hz / div),
            (self.includes_odometry(), topics::odometry::CHANNEL_ID, 97.0,
             crate::rates::BLACKBOX_ODOM_HZ),
            (self.includes_motors(), topics::motors::CHANNEL_ID, 81.0, indi_hz),
            (self.includes_motor_state(), topics::motor_state::CHANNEL_ID, 125.0, indi_hz),
            (self.includes_rc(), topics::rc::CHANNEL_ID, 118.0, 244.0),
            (self.includes_power(), topics::power::CHANNEL_ID, 117.0,
             crate::rates::BLACKBOX_POWER_HZ),
            (self.includes_mpc(), topics::mpc::CHANNEL_ID, 132.0, 100.0),
            // Computed, not measured: `blackbox_wire::steady_state_sizes`
            // pins the 78 B `.v2` payload; + 22 B header. Was 370.0 as a map.
            (self.includes_health(), topics::health::CHANNEL_ID, 100.0, 20.0),
        ];
        let mut total = 0.0;
        let mut i = 0;
        while i < rows.len() {
            let (included, id, bytes, hz) = rows[i];
            if included && (mute_mask & (1u32 << id)) == 0 {
                total += bytes * hz;
            }
            i += 1;
        }
        total as u32
    }
}

