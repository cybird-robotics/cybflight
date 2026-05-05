//! Record-set system: a four-tier classification (`none`, `small`,
//! `mid`, `large`) selecting which topics a session logs.
//!
//! ## Design
//!
//! The cybflight blackbox has a fixed catalogue of topics today
//! (`/imu1`, `/attitude`, `/rc`, `/events`); future profiles will
//! add more. Rather than letting each capture op hard-code its own
//! topic list (the pre-Stage-7 status quo), the recorder reads
//! [`BLACKBOX_RECORD_SET`] once at session start and emits exactly
//! the topics in that tier's [`topic_set`](RecordSet::topic_set).
//!
//! Tiers are **monotonic**: each level is a strict superset of the
//! previous one. That keeps mental model simple ("turn it up
//! one") and matches Betaflight's `blackbox_mode = NORMAL/ALWAYS`
//! convention.
//!
//! ## Tier definitions
//!
//! | Tier  | Topics                                           | Approx bytes / minute |
//! |-------|--------------------------------------------------|---|
//! | none  | (nothing — recording disabled)                   | 0     |
//! | small | events + rc                                      | ~250 KB |
//! | mid   | + attitude + odometry + mpc + motors + motor_state | ~22 MB |
//! | large | + raw IMU1                                       | ~60 MB  |
//!
//! "Small" is the *cheap* tier: low byte rate, just enough to
//! reconstruct what the pilot commanded and which arm/disarm
//! brackets bound the session. "Mid" adds the full inner-control
//! picture — the 8 kHz attitude estimate, the 1 kHz ESKF fused
//! odometry (`/odometry`), and the outer-loop MPC commands
//! (`/mpc`). That covers the typical flight-debug flow:
//! `RC → MPC command → ESKF state → attitude estimate`. "Large"
//! adds raw 6 DoF IMU on top — the high-bandwidth debug option for
//! filter / sensor work.
//!
//! ## Tier-aware drop priority
//!
//! Tiers are *also* a drop-priority ordering. Each iteration of the
//! recorder loop drains topics in tier order — small, then mid,
//! then large — and the large-tier drain (`/imu1`) uses a smaller
//! per-iteration budget (`DRAIN_BUDGET_DEPRIO`, currently 4) than
//! the small/mid drains (`DRAIN_BUDGET_NORMAL`, currently 16).
//! When the SD pipeline keeps up with combined publisher rates,
//! every drain finishes before its budget so the budgets are
//! immaterial; every topic is emitted in full. When the SD
//! pipeline can't keep up, the smaller IMU budget pushes the
//! overflow onto IMU first, so `/rc`, `/attitude`, `/odometry`,
//! and `/mpc` stay intact and IMU drops show up as `Lagged`
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
//! channel 1 in every file regardless of profile, `/attitude` is
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
];

const TOPICS_LARGE: &[TopicDef] = &[
    topics::events::DEF,
    topics::rc::DEF,
    topics::attitude::DEF,
    topics::odometry::DEF,
    topics::mpc::DEF,
    topics::motors::DEF,
    topics::motor_state::DEF,
    topics::imu::DEF,
];

/// Recording tier. See module docs for byte-rate estimates per tier.
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
    /// Events + RC + attitude. Adds the 8 kHz quaternion estimate;
    /// useful for reconstructing what the airframe actually did.
    Mid = 2,
    /// Events + RC + attitude + raw IMU1. Full debug fidelity for
    /// filter / sensor work. ~50 MB / minute.
    Large = 3,
}

impl RecordSet {
    /// Compile-time default selected at boot. Conservative — preserves
    /// existing behavior (all four topics) so this refactor is a
    /// no-behavior-change refactor unless the user explicitly
    /// changes tiers.
    pub const DEFAULT: Self = Self::Large;

    /// True if this tier should record at all (anything except `None`).
    #[inline]
    pub const fn enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    #[inline]
    pub const fn includes_imu(self) -> bool {
        matches!(self, Self::Large)
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
    /// Out-of-range bytes (only possible from a corrupted store)
    /// fall back to [`Self::DEFAULT`].
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
