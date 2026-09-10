//! `/power` topic — pack voltage / current at the `power_task` tick.
//!
//! Mirrors [`crate::sensors::POWER_TELEM`] (100 Hz, ADC3). Carries
//! both the **unfiltered** pack voltage and the `batt_lpf_hz`-filtered
//! one every other consumer sees (INDI thrust-table linearization
//! included), plus the unfiltered current and the running mAh
//! integral. The raw/filtered pair exists so a sysid flight can
//! measure how much the PT1 lags a throttle-punch IR drop — and what
//! the ripple floor actually is — instead of picking the cutoff by
//! intuition. Correlate against `/motors` (commanded) and
//! `/motor_state` (achieved), which share the tier.
//!
//! ~55 B per record → ~5.5 KB/s at 100 Hz, negligible against the
//! IMU stream. Not decimated: the point is the fast edges.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::sensors::power::PowerTelemetry;

/// MCAP channel id for `/power`. Stable across all record-set
/// profiles; next free id after `/control_setpoint` (14).
pub const CHANNEL_ID: u16 = 15;
pub const TOPIC: &str = "/power";
pub const SCHEMA_NAME: &str = "PowerTelemetry";
pub const SCHEMA: &[u8] = br#"{
  "title": "PowerTelemetry",
  "description": "One power_task ADC tick (100 Hz). voltage_v is the batt_lpf_hz-filtered pack voltage every consumer (INDI thrust table, shell, GCS) sees; voltage_raw_v is the same tick's hardware-oversampled reading with no software filter, ripple included.",
  "type": "object",
  "properties": {
    "timestamp_ns":  { "type": "integer", "description": "Sample wall-clock time, ns since boot." },
    "voltage_v":     { "type": "number",  "description": "Filtered pack voltage (V), PT1 at batt_lpf_hz." },
    "voltage_raw_v": { "type": "number",  "description": "Unfiltered pack voltage (V) from the same ADC conversion." },
    "current_a":     { "type": "number",  "description": "Unfiltered pack current (A); negative = charging / sensor offset." },
    "mah_drawn":     { "type": "integer", "description": "Integrated charge drawn since power-on (mAh)." },
    "cell_count":    { "type": "integer", "description": "Auto-detected series cell count; 0 = no battery." }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &PowerTelemetry) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(6)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("voltage_v")?;
    w.f32(m.voltage_cv as f32 * 0.01)?;
    w.str("voltage_raw_v")?;
    w.f32(m.voltage_raw_cv as f32 * 0.01)?;
    w.str("current_a")?;
    w.f32(m.current_ca as f32 * 0.01)?;
    w.str("mah_drawn")?;
    w.u64(m.mah_drawn as u64)?;
    w.str("cell_count")?;
    w.u64(m.cell_count as u64)?;
    Ok(w.pos())
}
