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
//! Tiers are **monotonic**: each level is a strict superset of the
//! previous one. One axis, one knob, "turn it up". The rule for
//! adding a new topic is "the cheapest tier whose bandwidth budget
//! the topic fits into" — cheap (< 100 KB/min) topics go in Mid;
//! the only thing in Large today is the high-bandwidth sysid extra
//! (`/imu1_raw`, ~30 MB/min on top of Mid's ~60 MB/min IMU stream).
//!
//! ## Tier definitions
//!
//! | Tier  | Topics added on top of previous                                                                                                                  | Approx bytes / minute |
//! |-------|---------------------------------------------------------------------------------------------------------------------------------------------------|---|
//! | none  | (nothing — recording disabled)                                                                                                                    | 0       |
//! | small | events + rc                                                                                                                                       | ~250 KB |
//! | mid   | + attitude + odometry + mpc + motors + motor_state + tracking_error + imu + control_setpoint + estimator_state + health + gps_health             | ~60 MB  |
//! | large | + imu1_raw                                                                                                                                        | ~90 MB  |
//!
//! "Small" is the *cheap* tier — arm/disarm brackets and pilot
//! intent only. "Mid" is the **default flight-debug tier**: the
//! 8 kHz attitude estimate, 1 kHz ESKF fused odometry, 8 kHz
//! post-LP IMU, outer-loop MPC commands, the
//! commanded-vs-achieved motor pair (`/motors`, `/motor_state`),
//! the rate-setpoint mirror (`/control_setpoint`) that completes
//! the controller-input picture, and the slow estimator-bias
//! snapshot (`/estimator_state`) for offline accel correction. That
//! covers the typical flight-debug flow:
//! `RC → MPC command → ESKF state → attitude estimate → motor`,
//! plus the sysid signals on the cheap. "Large" adds the only
//! topic with a real bandwidth cost — pre-biquad-LP raw IMU — for
//! filter-tuning and analyse.py-style RPM-notch fits.
//!
//! ## Tier-aware drop priority
//!
//! Tiers are *also* a drop-priority ordering. Each iteration of the
//! recorder loop drains topics in tier order — small, then mid,
//! then large — and the large-tier drains (`/imu1`, `/imu1_raw`)
//! use a smaller per-iteration budget (`DRAIN_BUDGET_DEPRIO`,
//! currently 4) than the small/mid drains (`DRAIN_BUDGET_NORMAL`,
//! currently 16). When the SD pipeline keeps up with combined
//! publisher rates, every drain finishes before its budget so the
//! budgets are immaterial; every topic is emitted in full. When
//! the SD pipeline can't keep up, the smaller IMU budget pushes
//! the overflow onto IMU first (raw IMU before filtered IMU within
//! the deprio block), so `/rc`, `/attitude`, `/odometry`, and the
//! controller stream stay intact and IMU drops show up as `Lagged`
//! samples (counted in the session summary). The motivating
//! intuition: IMU is the high-bandwidth debug stream we're most
//! willing to lose under contention; smaller, lower-rate
//! flight-state topics are the ones we can't afford to drop.
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
//! channel 1 in every file regardless of tier, `/attitude` is
//! channel 2, etc. Foxglove / `mcap cat` consumers can rely on this
//! across files captured at different tiers.

use core::sync::atomic::{AtomicU8, Ordering};

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
    topics::attitude::DEF,
    topics::odometry::DEF,
    topics::mpc::DEF,
    topics::motors::DEF,
    topics::motor_state::DEF,
    topics::tracking_error::DEF,
    topics::imu::DEF,
    topics::control_setpoint::DEF,
    #[cfg(feature = "est_eskf")]
    topics::estimator_state::DEF,
    topics::health::DEF,
    topics::gps_health::DEF,
];

const TOPICS_LARGE: &[TopicDef] = &[
    topics::events::DEF,
    topics::rc::DEF,
    topics::attitude::DEF,
    topics::odometry::DEF,
    topics::mpc::DEF,
    topics::motors::DEF,
    topics::motor_state::DEF,
    topics::tracking_error::DEF,
    topics::imu::DEF,
    topics::imu_raw::DEF,
    topics::control_setpoint::DEF,
    #[cfg(feature = "est_eskf")]
    topics::estimator_state::DEF,
    topics::health::DEF,
    topics::gps_health::DEF,
];

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
    /// Small + the full instrumented controller stream — attitude,
    /// odometry, mpc, motors, motor_state, tracking_error, post-LP
    /// IMU, control_setpoint, estimator_state, health, gps_health.
    /// **The default flight-debug tier**: enough to reconstruct any
    /// recent firmware decision, including INDI sysid signals on
    /// the cheap, without paying for the raw IMU stream.
    Mid = 2,
    /// Mid + pre-biquad-LP raw IMU (`/imu1_raw`). The only topic
    /// that doubles the file size; opt-in for filter-tuning and
    /// RPM-notch fit work.
    Large = 3,
}

impl RecordSet {
    /// Compile-time default selected at boot.
    ///
    /// `Mid` covers the routine flight-debug case at ~60 MB/min.
    /// Power users opt up to `Large` when they specifically want
    /// the raw IMU stream; nobody opts up to `Large` "just in case"
    /// and discovers a 1 GB file from an 8-minute flight.
    pub const DEFAULT: Self = Self::Mid;

    /// True if this tier should record at all (anything except `None`).
    #[inline]
    pub const fn enabled(self) -> bool {
        !matches!(self, Self::None)
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
        matches!(self, Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_mpc(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_motors(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_motor_state(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    /// True for tiers that include the `/health` estimator-fault
    /// snapshot (Mid + Large). Self-throttled to ~20 Hz inside the
    /// recorder loop.
    #[inline]
    pub const fn includes_health(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
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
        matches!(self, Self::Mid | Self::Large)
    }

    /// True for tiers that include the `/control_setpoint` topic
    /// (mirror of `RATE_COMMAND` — outer-loop output to INDI).
    /// Mid + Large: ~5 KB/min, completes the
    /// `command → tracking_error → motor` triangle the rest of Mid
    /// already carries.
    #[inline]
    pub const fn includes_control_setpoint(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    /// True for tiers that include the `/estimator_state` topic
    /// (decimated ESKF gyro/accel bias estimates, ~10 Hz). Mid +
    /// Large and `est_eskf`-only on the recorder side — pure-Mahony
    /// builds have no biases to publish, so the topic is silently
    /// absent from the file.
    #[inline]
    pub const fn includes_estimator_state(self) -> bool {
        matches!(self, Self::Mid | Self::Large)
    }

    /// True for tiers that include the `/imu1_raw` topic
    /// (pre-biquad-LP IMU samples at the same 8 kHz cadence as
    /// `/imu1`). **Large only** — the only topic with a meaningful
    /// bandwidth cost (~doubles the IMU bytes); opt-in for
    /// filter-tuning / RPM-notch fit work.
    #[inline]
    pub const fn includes_imu_raw(self) -> bool {
        matches!(self, Self::Large)
    }

    #[inline]
    pub const fn includes_rc(self) -> bool {
        matches!(self, Self::Small | Self::Mid | Self::Large)
    }

    #[inline]
    pub const fn includes_events(self) -> bool {
        matches!(self, Self::Small | Self::Mid | Self::Large)
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
            Self::Large => TOPICS_LARGE,
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
