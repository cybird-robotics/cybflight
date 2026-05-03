//! `/rc` topic definition + CBOR encoder.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/rc`. Stable across all record-set profiles.
pub const CHANNEL_ID: u16 = 3;
pub const TOPIC: &str = "/rc";
pub const SCHEMA_NAME: &str = "RcInput";
pub const SCHEMA: &[u8] = br#"{
  "title": "RcInput",
  "type": "object",
  "properties": {
    "timestamp_ns":  { "type": "integer" },
    "channels":      { "type": "array", "minItems": 0, "maxItems": 16,
                       "items": {"type": "integer"},
                       "description": "PWM microseconds [988..2012]; length == channel_count" },
    "channel_count": { "type": "integer" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::RcInput) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    let n = (m.channel_count as usize).min(m.channels.len());
    w.map(3)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("channels")?;
    w.array(n as u64)?;
    for v in &m.channels[..n] {
        w.u64(*v as u64)?;
    }
    w.str("channel_count")?;
    w.u64(m.channel_count as u64)?;
    Ok(w.pos())
}
