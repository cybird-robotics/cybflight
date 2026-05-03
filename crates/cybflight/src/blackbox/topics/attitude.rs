//! `/attitude` topic definition + CBOR encoder.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/attitude`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 2;
pub const TOPIC: &str = "/attitude";
pub const SCHEMA_NAME: &str = "VehicleAttitude";
pub const SCHEMA: &[u8] = br#"{
  "title": "VehicleAttitude",
  "type": "object",
  "properties": {
    "timestamp_ns": { "type": "integer" },
    "quaternion":   { "type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"},
                      "description": "[w, i, j, k]" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::VehicleAttitude) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(2)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("quaternion")?;
    w.array(4)?;
    let q = m.orientation.as_ref();
    w.f32(q.w)?;
    w.f32(q.i)?;
    w.f32(q.j)?;
    w.f32(q.k)?;
    Ok(w.pos())
}
