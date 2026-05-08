//! `/gps_health` topic --post-flight mirror of the live `gpshealth`
//! shell command.
//!
//! Snapshots the same atomics the live report reads:
//!
//! - [`crate::sensors::gps::GPS_HEALTH`] --the operator-facing
//!   variant enum (NotConfigured / Initializing / UartInitFailed /
//!   InitTimedOut / InitFailed / Waiting / Locked / ReadError),
//!   flattened to `(state, err_kind, fix_type, num_sv, h_acc_mm,
//!   diff_soln, carr_soln)`. Variant fields the active state doesn't
//!   define are zero.
//! - [`crate::sensors::gps::LATEST_NAV_PVT`] --the latest NAV-PVT
//!   accuracy + carrier-solution fields the `Locked` enum doesn't
//!   carry: `v_acc_mm`, `s_acc_mm_s`, `gnss_fix_ok`, plus
//!   `nav_pvt_arrival_us` --the raw `Instant` of when the GPS task
//!   parsed the most recent NAV-PVT.
//!
//! ## Inter-PVT interval / GPS jitter
//!
//! Previous schemas exposed `nav_pvt_age_ms` and `last_fix_age_ms`
//! --snapshots of `<emit time> - <arrival time>`. Those misled
//! readers into reading the recorder's sampling jitter as GPS-side
//! jitter. Removed.
//!
//! Instead `nav_pvt_arrival_us` carries the raw arrival `Instant`.
//! The post-flight tool computes whatever metric it actually wants:
//!
//! - **Liveness / pipeline-stuck**: `record.timestamp_ns/1000 -
//!   record.nav_pvt_arrival_us`. Climbs into the seconds = GPS task
//!   is stuck.
//! - **Inter-PVT interval (the GPS-jitter signal)**: dedupe the
//!   `nav_pvt_arrival_us` values across records (each PVT shows up
//!   in multiple `/gps_health` records because the recorder emits
//!   faster than NAV-PVT arrives), then diff successive unique
//!   values. At 5 Hz the spread should be tight around 200 000 µs;
//!   deviations are real receiver-side variability (sensor
//!   scheduling, UART/parser delay, dropped frames).
//!
//! Cadence: same self-throttled 20 Hz as `/health` (see
//! `HEALTH_EMIT_INTERVAL` in `blackbox::recorder`). The receiver only
//! produces NAV-PVT at ~5 Hz, so the on-disk record is intentionally
//! oversampled --it lets a post-flight reader join `/gps_health`
//! against `/health` and `/odometry` at matching wall-clock instants
//! without a resample step.
//!
//! Cost: ~120 B CBOR × 20 Hz = ~2.4 KB/s, dwarfed by `/imu1` and
//! `/odometry`. Lives in the **Mid + Large** tiers alongside
//! `/health`; never in **Small** (which is reserved for the
//! event-bracket-only profile).
//!
//! ## On builds without `est_pos_gps`
//!
//! The `GPS_HEALTH` static stays at `NotConfigured` for the entire
//! session, and `LATEST_NAV_PVT` stays `None`. The topic still emits
//! --empty by design --so a tier check is never feature-gated and
//! the recorder doesn't need a second cfg surface. Cost on a
//! mocap-only build: ~50 B/record × 20 Hz = 1 KB/s of "GPS not
//! present" markers, which doubles as a build-config breadcrumb in
//! the file itself.

use embassy_time::Instant;

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::sensors::gps::{GPS_HEALTH, GpsErrKind, GpsHealth, LATEST_NAV_PVT};

/// MCAP channel id for `/gps_health`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 11;
pub const TOPIC: &str = "/gps_health";
pub const SCHEMA_NAME: &str = "GpsHealth";
pub const SCHEMA: &[u8] = br#"{
  "title": "GpsHealth",
  "description": "Post-flight mirror of the live `gpshealth` shell report (crate::sensors::gps::GPS_HEALTH + LATEST_NAV_PVT). Flattens the GpsHealth enum to a fixed-shape record so post-flight tooling can index and plot.",
  "type": "object",
  "properties": {
    "timestamp_ns":      { "type": "integer", "description": "Sample wall-clock time, ns since boot." },
    "state":             { "type": "integer",
                           "description": "GpsHealth variant: 0=NotConfigured, 1=Initializing, 2=UartInitFailed, 3=InitTimedOut, 4=InitFailed, 5=Waiting, 6=Locked, 7=ReadError." },
    "err_kind":          { "type": "integer",
                           "description": "Set when state is InitFailed or ReadError, zero otherwise. 0=none, 1=Io, 2=BadChecksum, 3=Nak, 4=Timeout." },
    "fix_type":          { "type": "integer",
                           "description": "u-blox NAV-PVT fix_type. 0=no fix, 2=2D, 3=3D, 4=GNSS+dead-reckoning, 5=time-only. Populated for Waiting/Locked." },
    "num_sv":            { "type": "integer", "description": "Satellites used in the navigation solution. Populated for Waiting/Locked." },
    "h_acc_mm":          { "type": "integer", "description": "Horizontal accuracy estimate, mm. From the GpsHealth enum (Waiting/Locked) --matches NAV-PVT." },
    "v_acc_mm":          { "type": "integer", "description": "Vertical accuracy estimate, mm. From LATEST_NAV_PVT (not in the enum). Zero when no NAV-PVT seen yet." },
    "s_acc_mm_s":        { "type": "integer", "description": "Speed accuracy estimate, mm/s. From LATEST_NAV_PVT." },
    "diff_soln":         { "type": "boolean", "description": "Differential corrections (e.g. RTCM3) applied. NAV-PVT flags bit 1." },
    "carr_soln":         { "type": "integer", "description": "RTK carrier-phase solution status. 0=none, 1=float, 2=fixed. NAV-PVT flags bits 6-7." },
    "gnss_fix_ok":       { "type": "boolean", "description": "Receiver believes the fix is valid. NAV-PVT flags bit 0. False when no NAV-PVT seen yet." },
    "nav_pvt_arrival_us":         { "type": "integer",
                                    "description": "`Instant` (microseconds since boot) when the GPS task parsed the most recent NAV-PVT, i.e. the source-of-truth arrival timestamp. 0 when no NAV-PVT has ever been received. Two derived metrics: liveness = (timestamp_ns/1000 - nav_pvt_arrival_us); inter-PVT interval (the GPS-jitter signal) = diff of unique nav_pvt_arrival_us values across records. The recorder oversamples relative to NAV-PVT, so successive records typically share an arrival timestamp -- dedupe before differencing." }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

// ── State / error-kind enum encoding ───────────────────────────────────────
//
// Stable on disk --never renumber. Add new GpsHealth variants only
// with values that don't conflict; readers must keep working when
// they see an unknown code.

const STATE_NOT_CONFIGURED: u8 = 0;
const STATE_INITIALIZING: u8 = 1;
const STATE_UART_INIT_FAILED: u8 = 2;
const STATE_INIT_TIMED_OUT: u8 = 3;
const STATE_INIT_FAILED: u8 = 4;
const STATE_WAITING: u8 = 5;
const STATE_LOCKED: u8 = 6;
const STATE_READ_ERROR: u8 = 7;

const ERR_NONE: u8 = 0;
const ERR_IO: u8 = 1;
const ERR_BAD_CHECKSUM: u8 = 2;
const ERR_NAK: u8 = 3;
const ERR_TIMEOUT: u8 = 4;

fn encode_err(k: GpsErrKind) -> u8 {
    match k {
        GpsErrKind::Io => ERR_IO,
        GpsErrKind::BadChecksum => ERR_BAD_CHECKSUM,
        GpsErrKind::Nak => ERR_NAK,
        GpsErrKind::Timeout => ERR_TIMEOUT,
    }
}

/// Flattened snapshot --populated by [`snapshot`] from the
/// [`GpsHealth`] enum + the latest NAV-PVT cell. All zero-or-sentinel
/// initial values; per-variant logic fills the slots that apply.
struct Flat {
    state: u8,
    err_kind: u8,
    fix_type: u8,
    num_sv: u8,
    h_acc_mm: u32,
    v_acc_mm: u32,
    s_acc_mm_s: u32,
    diff_soln: bool,
    carr_soln: u8,
    gnss_fix_ok: bool,
    /// Source-of-truth arrival timestamp of the most recent NAV-PVT,
    /// in microseconds since boot. 0 when no NAV-PVT has ever been
    /// received. Post-flight derives liveness AND inter-PVT interval
    /// from this single field — see schema doc.
    nav_pvt_arrival_us: u64,
}

fn snapshot(_now: Instant) -> Flat {
    let mut f = Flat {
        state: STATE_NOT_CONFIGURED,
        err_kind: ERR_NONE,
        fix_type: 0,
        num_sv: 0,
        h_acc_mm: 0,
        v_acc_mm: 0,
        s_acc_mm_s: 0,
        diff_soln: false,
        carr_soln: 0,
        gnss_fix_ok: false,
        nav_pvt_arrival_us: 0,
    };

    // Enum → state + variant fields. The Mutex is contended only by
    // the GPS task's once-per-frame write, so the hold here is
    // effectively zero-cost.
    let h = GPS_HEALTH.lock(|c| c.get());
    match h {
        GpsHealth::NotConfigured => f.state = STATE_NOT_CONFIGURED,
        GpsHealth::Initializing => f.state = STATE_INITIALIZING,
        GpsHealth::UartInitFailed => f.state = STATE_UART_INIT_FAILED,
        GpsHealth::InitTimedOut => f.state = STATE_INIT_TIMED_OUT,
        GpsHealth::InitFailed(k) => {
            f.state = STATE_INIT_FAILED;
            f.err_kind = encode_err(k);
        }
        GpsHealth::Waiting {
            fix_type,
            num_sv,
            h_acc_mm,
        } => {
            f.state = STATE_WAITING;
            f.fix_type = fix_type;
            f.num_sv = num_sv;
            f.h_acc_mm = h_acc_mm;
        }
        GpsHealth::Locked {
            fix_type,
            num_sv,
            h_acc_mm,
            diff_soln,
            carr_soln,
            last_fix_at: _,
        } => {
            f.state = STATE_LOCKED;
            f.fix_type = fix_type;
            f.num_sv = num_sv;
            f.h_acc_mm = h_acc_mm;
            f.diff_soln = diff_soln;
            f.carr_soln = carr_soln;
        }
        GpsHealth::ReadError(k) => {
            f.state = STATE_READ_ERROR;
            f.err_kind = encode_err(k);
        }
    }

    // NAV-PVT supplies the accuracy fields the enum doesn't carry,
    // plus the raw arrival timestamp. Both liveness and inter-PVT
    // jitter are derived from this single value in post.
    if let Some(p) = LATEST_NAV_PVT.lock(|c| c.get()) {
        f.v_acc_mm = p.v_acc_mm;
        f.s_acc_mm_s = p.s_acc_mm_s;
        f.gnss_fix_ok = p.gnss_fix_ok;
        f.nav_pvt_arrival_us = p.timestamp.as_micros();
    }

    f
}

/// Encode one `/gps_health` record at `now`. Mirrors the layout of
/// [`super::health::encode`] --same throttle path in
/// `blackbox::recorder::emit_gps_health`.
pub fn encode(scratch: &mut [u8], now: Instant) -> cbor::Result<usize> {
    let f = snapshot(now);

    let mut w = CborWriter::new(scratch);
    w.map(12)?;
    w.str("timestamp_ns")?;
    w.u64(now.as_micros().saturating_mul(1_000))?;
    w.str("state")?;
    w.u64(f.state as u64)?;
    w.str("err_kind")?;
    w.u64(f.err_kind as u64)?;
    w.str("fix_type")?;
    w.u64(f.fix_type as u64)?;
    w.str("num_sv")?;
    w.u64(f.num_sv as u64)?;
    w.str("h_acc_mm")?;
    w.u64(f.h_acc_mm as u64)?;
    w.str("v_acc_mm")?;
    w.u64(f.v_acc_mm as u64)?;
    w.str("s_acc_mm_s")?;
    w.u64(f.s_acc_mm_s as u64)?;
    w.str("diff_soln")?;
    w.bool(f.diff_soln)?;
    w.str("carr_soln")?;
    w.u64(f.carr_soln as u64)?;
    w.str("gnss_fix_ok")?;
    w.bool(f.gnss_fix_ok)?;
    w.str("nav_pvt_arrival_us")?;
    w.u64(f.nav_pvt_arrival_us)?;
    Ok(w.pos())
}
