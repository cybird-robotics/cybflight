//! `/health` topic -- post-flight mirror of the live `health` shell
//! command.
//!
//! Captures the same atomics the live General Health Report
//! (`crate::health::SystemHealth::snapshot`) reads -- failsafe state,
//! estimator readiness, fault flags, attitude observability -- plus
//! the cumulative `EskfHealth` counters that the live report omits
//! (gate rejections / NaN resets / latest NIS), which are useful for
//! reconstruction but too verbose for a one-shot shell line.
//!
//! Each record is what `health` would print at this exact moment, in
//! a form post-flight tooling can index over time. The two views stay
//! consistent by reading the **same** sources of truth -- there's no
//! BB-specific shadow state.
//!
//! Cadence: 20 Hz, driven by the recorder's own loop tick (see
//! `HEALTH_EMIT_INTERVAL` in `blackbox::recorder`). Decoupled from
//! `/odom` so the failsafe progression — when `/odom` typically
//! stops publishing — stays observable through to disarm. High
//! enough to catch sub-second transient faults, low enough that the
//! 100 B record (positional `.v2` array, `cybflight_core::blackbox_wire::
//! encode_health`) adds ~2 KB/s. The keyed-map form it replaced was
//! 370 B — ~80 % of it the 18 key strings, repeated every record.
//!
//! Beyond the shell report's content, `.v2` carries the two things no
//! other topic records: the INDI loop's worst iteration and worst
//! inter-iteration gap per window (`indi_task::step_stats`, previously
//! shell-only via `indistat`), and the RC link statistics the arming
//! gates test (`arm_min_link_quality`) but no topic logged — `/rc`
//! carries channels only and is absent from `Sysid` entirely.

use core::sync::atomic::Ordering;

use super::TopicDef;
use crate::blackbox::cbor;
use crate::control::indi_task::step_stats;
use crate::sensors::rc::LATEST_RC_LINK_STATUS;
use cybflight_core::blackbox_wire::{self, HealthRecord, RcLinkSample};
use crate::control::failsafe::FAILSAFE_ACTIVE;
use crate::estimation::{
    ATTITUDE_HEALTH, ESKF_DEGRADED, ESKF_FAULTS, ESKF_HEALTH, ESKF_LAST_ATT_UPDATE,
    ESKF_LAST_POS_UPDATE, ESKF_LAST_VEL_UPDATE, ESKF_SEVERE_FAULT, ESTIMATOR_READY,
};
use embassy_time::Instant;

pub const CHANNEL_ID: u16 = 10;
pub const TOPIC: &str = "/health";
/// `.v2` = the positional-array layout (plus the INDI timing and RC
/// link fields). Readers key their decoder on this name; the keyed-map
/// layout was plain `SystemHealth`.
pub const SCHEMA_NAME: &str = "SystemHealth.v2";
pub const SCHEMA: &[u8] = br#"{
  "title": "SystemHealth.v2",
  "description": "Flat positional array; see prefixItems for element order. Post-flight mirror of the live `health` shell report (crate::health::SystemHealth) plus cumulative ESKF gate counters, INDI loop timing and RC link statistics. Source-of-truth atomics live in crate::estimation, crate::control::failsafe, crate::control::indi_task::step_stats and crate::sensors::rc.",
  "type": "array",
  "minItems": 23, "maxItems": 23,
  "prefixItems": [
    { "title": "timestamp_ns",       "type": "integer", "description": "Sample wall-clock time, ns since boot." },
    { "title": "imu1_temp_c",        "type": "number",  "description": "Latest IMU-1 die temperature seen on /imu1; NaN before the first sample." },
    { "title": "failsafe_active",    "type": "boolean", "description": "control::failsafe::FAILSAFE_ACTIVE -- top-level disarm gate." },
    { "title": "estimator_ready",    "type": "boolean", "description": "estimation::ESTIMATOR_READY -- converged AND quality-good (RTK debounce on GPS, plain converged on mocap)." },
    { "title": "eskf_degraded",      "type": "boolean", "description": "estimation::ESKF_DEGRADED -- non-severe fault present." },
    { "title": "eskf_severe_fault",  "type": "boolean", "description": "estimation::ESKF_SEVERE_FAULT -- failsafe-watchdog signal. Only set while armed." },
    { "title": "eskf_faults",        "type": "integer", "description": "u32 bitfield. NAN_RESET=1, POS_STALE=2, VEL_STALE=4, ATT_STALE=8, COV_TRACE_BLOWUP=16, GUARD_JUMP_CASCADE=32, GUARD_REJECT_CASCADE=64. See estimation::fault." },
    { "title": "attitude_health",    "type": "integer", "description": "u8 bitfield. ACCEL_OK=1, GYRO_OK=2, NO_RECENT_NAN=4. See estimation::att_health." },
    { "title": "nan_resets",         "type": "integer", "description": "Cumulative ESKF NaN-state recoveries since boot." },
    { "title": "gate_rejects_pos",   "type": "integer", "description": "Cumulative pos-update gate rejections (Mahalanobis + jump)." },
    { "title": "gate_rejects_vel",   "type": "integer", "description": "Cumulative vel-update gate rejections." },
    { "title": "gate_rejects_att",   "type": "integer", "description": "Cumulative att-update gate rejections." },
    { "title": "last_nis_pos",       "type": "number",  "description": "Most recent pos normalised innovation^2, per-DoF." },
    { "title": "last_nis_vel",       "type": "number",  "description": "Most recent vel normalised innovation^2." },
    { "title": "last_nis_att",       "type": "number",  "description": "Most recent att normalised innovation^2." },
    { "title": "last_pos_update_ns", "type": "integer", "description": "Wall-clock time of last accepted pos update, ns since boot. 0 = never. A frozen value means upstream stopped delivering; a recent one means the guard is rejecting fresh measurements." },
    { "title": "last_vel_update_ns", "type": "integer", "description": "Wall-clock time of last accepted vel update, ns since boot. 0 = never." },
    { "title": "last_att_update_ns", "type": "integer", "description": "Wall-clock time of last accepted att update, ns since boot. 0 = never." },
    { "title": "indi_step_max_us",   "type": "integer", "description": "Longest INDI iteration (compute, decimation gate to motor publish) since the previous /health record, us. 0 = no iteration ran in the window." },
    { "title": "indi_period_max_us", "type": "integer", "description": "Longest gap between INDI iteration starts since the previous /health record, us. Compare against the nominal control period (IMU ODR / indi_ctrl_div); an excess is a missed deadline." },
    { "title": "rc_link_quality",    "type": ["integer", "null"], "description": "Latest CRSF/GHST link quality, percent [0..100]. null = no link-statistics frame since boot." },
    { "title": "rc_rssi_dbm",        "type": ["integer", "null"], "description": "Latest uplink RSSI, dBm (negative). null = never heard." },
    { "title": "rc_link_age_ms",     "type": ["integer", "null"], "description": "Age of that link-statistics frame at record time, ms. Growing while quality holds steady = frames stopped arriving. null = never heard." }
  ]
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

/// Snapshot the global health state and emit it to `scratch`.
/// `now` is the timestamp the record will carry -- supply the same
/// `Instant` the surrounding `emit_*` call uses so blackbox readers
/// can correlate health with the message it piggybacks on.
///
/// Consistency: the boolean / bitfield fields are read from the same
/// atomics that `crate::health::SystemHealth::snapshot` reads, so a
/// `/health` MCAP record and a contemporaneous `health` shell line
/// agree by construction.
/// `imu1_temp_c` is the newest IMU-1 temperature the recorder has
/// seen on the `/imu1` stream (NaN before the first sample). It rides
/// here at 20 Hz because the compact `Imu.v2` wire format dropped the
/// per-sample `temp_c` — a ~1 Hz signal has no business costing
/// 12 bytes at up to 8 kHz.
pub fn encode(scratch: &mut [u8], now: Instant, imu1_temp_c: f32) -> cbor::Result<usize> {
    let h = ESKF_HEALTH.lock(|c| c.get());
    // Last-update timestamps in nanoseconds since boot. 0 encodes
    // "never accepted" (matches `None` on the live shell as
    // "pos=never"). Post-flight tools subtract these from
    // timestamp_ns to get measurement age, the same disambiguation
    // the live `data freshness` line provides.
    let inst_ns = |t: Option<Instant>| t.map_or(0u64, |t| t.as_micros().saturating_mul(1_000));

    // Window maxima since the previous record, from the recorder-owned
    // mirrors — the shell's `indistat` keeps its own, so neither reader
    // blinds the other.
    let (step_max_cyc, period_max_cyc) = step_stats::take_log_max();
    let cyc_to_us =
        |c: u32| (c as u64 * 1_000_000 / step_stats::SYSCLK_HZ as u64) as u32;

    let rc_link = LATEST_RC_LINK_STATUS
        .lock(|c| c.borrow().clone())
        .map(|l| RcLinkSample {
            quality: l.link_quality,
            rssi_dbm: l.rssi_dbm,
            // Saturating: an embassy `duration_since` that underflows
            // panics, and a panic here is an IWDG boot loop.
            age_ms: now.saturating_duration_since(l.timestamp).as_millis() as u32,
        });

    let r = HealthRecord {
        t_ns: now.as_micros().saturating_mul(1_000),
        imu1_temp_c,
        failsafe_active: FAILSAFE_ACTIVE.load(Ordering::Acquire),
        estimator_ready: ESTIMATOR_READY.load(Ordering::Acquire),
        eskf_degraded: ESKF_DEGRADED.load(Ordering::Acquire),
        eskf_severe_fault: ESKF_SEVERE_FAULT.load(Ordering::Acquire),
        eskf_faults: ESKF_FAULTS.load(Ordering::Relaxed),
        attitude_health: ATTITUDE_HEALTH.load(Ordering::Relaxed),
        nan_resets: h.nan_resets,
        gate_rejects_pos: h.gate_rejects_pos,
        gate_rejects_vel: h.gate_rejects_vel,
        gate_rejects_att: h.gate_rejects_att,
        last_nis_pos: h.last_nis_pos,
        last_nis_vel: h.last_nis_vel,
        last_nis_att: h.last_nis_att,
        last_pos_update_ns: inst_ns(ESKF_LAST_POS_UPDATE.lock(|c| c.get())),
        last_vel_update_ns: inst_ns(ESKF_LAST_VEL_UPDATE.lock(|c| c.get())),
        last_att_update_ns: inst_ns(ESKF_LAST_ATT_UPDATE.lock(|c| c.get())),
        indi_step_max_us: cyc_to_us(step_max_cyc),
        indi_period_max_us: cyc_to_us(period_max_cyc),
        rc_link,
    };
    blackbox_wire::encode_health(scratch, &r)
}
