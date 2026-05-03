//! `/imu1` topic definition + CBOR encoder.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/imu1`. Stable across all record-set
/// profiles — the value the topic carries on disk.
pub const CHANNEL_ID: u16 = 1;
pub const TOPIC: &str = "/imu1";
pub const SCHEMA_NAME: &str = "Imu";
pub const SCHEMA: &[u8] = br#"{
  "title": "Imu",
  "type": "object",
  "properties": {
    "timestamp_ns": { "type": "integer" },
    "accel_m_s2":   { "type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"} },
    "gyro_rad_s":   { "type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"} },
    "temp_c":       { "type": "number" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], sample: &msgs::Imu) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(4)?;
    w.str("timestamp_ns")?;
    w.u64(sample.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("accel_m_s2")?;
    w.array(3)?;
    for v in sample.accel_m_s2.iter() {
        w.f32(*v)?;
    }
    w.str("gyro_rad_s")?;
    w.array(3)?;
    for v in sample.gyro_rad_s.iter() {
        w.f32(*v)?;
    }
    w.str("temp_c")?;
    w.f32(sample.temp_c)?;
    Ok(w.pos())
}
