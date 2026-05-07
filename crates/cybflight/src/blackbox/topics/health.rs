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
//! Cadence: piggybacks on the `/odom` emit path (ODOM_DECIMATION-tied
//! ~100 Hz). High enough to catch sub-second transient faults, low
//! enough that the ~100-byte CBOR map adds <15 KB/s to file growth.

use core::sync::atomic::Ordering;

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::control::failsafe::FAILSAFE_ACTIVE;
use crate::estimation::{
    ATTITUDE_HEALTH, ESKF_DEGRADED, ESKF_FAULTS, ESKF_HEALTH, ESKF_LAST_ATT_UPDATE,
    ESKF_LAST_POS_UPDATE, ESKF_LAST_VEL_UPDATE, ESKF_SEVERE_FAULT, ESTIMATOR_READY,
};
use embassy_time::Instant;

pub const CHANNEL_ID: u16 = 10;
pub const TOPIC: &str = "/health";
pub const SCHEMA_NAME: &str = "SystemHealth";
pub const SCHEMA: &[u8] = br#"{
  "title": "SystemHealth",
  "description": "Post-flight mirror of the live `health` shell report (crate::health::SystemHealth). Bitfield-aggregated estimator/attitude state plus cumulative ESKF gate counters. Source-of-truth atomics live in crate::estimation and crate::control::failsafe.",
  "type": "object",
  "properties": {
    "timestamp_ns":      { "type": "integer", "description": "Sample wall-clock time, ns since boot." },
    "failsafe_active":   { "type": "boolean", "description": "control::failsafe::FAILSAFE_ACTIVE -- top-level disarm gate." },
    "estimator_ready":   { "type": "boolean", "description": "estimation::ESTIMATOR_READY -- converged AND quality-good (RTK debounce on GPS, plain converged on mocap)." },
    "eskf_degraded":     { "type": "boolean", "description": "estimation::ESKF_DEGRADED -- non-severe fault present (set when faults != 0 AND !severe)." },
    "eskf_severe_fault": { "type": "boolean", "description": "estimation::ESKF_SEVERE_FAULT -- failsafe-watchdog signal. Only set while armed." },
    "eskf_faults":       { "type": "integer",
                           "description": "u32 bitfield. NAN_RESET=1, POS_STALE=2, VEL_STALE=4, ATT_STALE=8, COV_TRACE_BLOWUP=16, GUARD_JUMP_CASCADE=32 (held 3 s after the EskfFailsafe jump-cascade gate fires), GUARD_REJECT_CASCADE=64 (same hold for reject-cascade). See estimation::fault." },
    "attitude_health":   { "type": "integer",
                           "description": "u8 bitfield. ACCEL_OK=1, GYRO_OK=2, NO_RECENT_NAN=4. See estimation::att_health." },
    "nan_resets":        { "type": "integer", "description": "Cumulative ESKF NaN-state recoveries since boot." },
    "gate_rejects_pos":  { "type": "integer", "description": "Cumulative pos-update gate rejections (Mahalanobis + jump)." },
    "gate_rejects_vel":  { "type": "integer", "description": "Cumulative vel-update gate rejections." },
    "gate_rejects_att":  { "type": "integer", "description": "Cumulative att-update gate rejections." },
    "last_nis_pos":      { "type": "number",  "description": "Most recent pos normalised innovation^2, per-DoF." },
    "last_nis_vel":      { "type": "number",  "description": "Most recent vel normalised innovation^2." },
    "last_nis_att":      { "type": "number",  "description": "Most recent att normalised innovation^2." },
    "last_pos_update_ns":{ "type": "integer", "description": "Wall-clock time of last accepted pos update, ns since boot. 0 = never. Disambiguates POS_STALE: a frozen value means upstream stopped delivering; a recent one means the guard is rejecting fresh measurements." },
    "last_vel_update_ns":{ "type": "integer", "description": "Wall-clock time of last accepted vel update, ns since boot. 0 = never." },
    "last_att_update_ns":{ "type": "integer", "description": "Wall-clock time of last accepted att update, ns since boot. 0 = never." }
  }
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
pub fn encode(scratch: &mut [u8], now: Instant) -> cbor::Result<usize> {
    let failsafe_active = FAILSAFE_ACTIVE.load(Ordering::Acquire);
    let estimator_ready = ESTIMATOR_READY.load(Ordering::Acquire);
    let eskf_degraded = ESKF_DEGRADED.load(Ordering::Acquire);
    let severe = ESKF_SEVERE_FAULT.load(Ordering::Acquire);
    let faults = ESKF_FAULTS.load(Ordering::Relaxed);
    let att_health = ATTITUDE_HEALTH.load(Ordering::Relaxed);
    let h = ESKF_HEALTH.lock(|c| c.get());
    let last_pos = ESKF_LAST_POS_UPDATE.lock(|c| c.get());
    let last_vel = ESKF_LAST_VEL_UPDATE.lock(|c| c.get());
    let last_att = ESKF_LAST_ATT_UPDATE.lock(|c| c.get());

    let mut w = CborWriter::new(scratch);
    w.map(17)?;
    w.str("timestamp_ns")?;
    w.u64(now.as_micros().saturating_mul(1_000))?;
    w.str("failsafe_active")?;
    w.bool(failsafe_active)?;
    w.str("estimator_ready")?;
    w.bool(estimator_ready)?;
    w.str("eskf_degraded")?;
    w.bool(eskf_degraded)?;
    w.str("eskf_severe_fault")?;
    w.bool(severe)?;
    w.str("eskf_faults")?;
    w.u64(faults as u64)?;
    w.str("attitude_health")?;
    w.u64(att_health as u64)?;
    w.str("nan_resets")?;
    w.u64(h.nan_resets as u64)?;
    w.str("gate_rejects_pos")?;
    w.u64(h.gate_rejects_pos as u64)?;
    w.str("gate_rejects_vel")?;
    w.u64(h.gate_rejects_vel as u64)?;
    w.str("gate_rejects_att")?;
    w.u64(h.gate_rejects_att as u64)?;
    w.str("last_nis_pos")?;
    w.f32(h.last_nis_pos)?;
    w.str("last_nis_vel")?;
    w.f32(h.last_nis_vel)?;
    w.str("last_nis_att")?;
    w.f32(h.last_nis_att)?;
    // Last-update timestamps in nanoseconds since boot. 0 encodes
    // "never accepted" (matches `None` on the live shell as
    // "pos=never"). Post-flight tools subtract these from
    // timestamp_ns to get measurement age, the same disambiguation
    // the live `data freshness` line provides.
    let inst_ns = |t: Option<Instant>| t.map_or(0u64, |t| t.as_micros().saturating_mul(1_000));
    w.str("last_pos_update_ns")?;
    w.u64(inst_ns(last_pos))?;
    w.str("last_vel_update_ns")?;
    w.u64(inst_ns(last_vel))?;
    w.str("last_att_update_ns")?;
    w.u64(inst_ns(last_att))?;
    Ok(w.pos())
}
