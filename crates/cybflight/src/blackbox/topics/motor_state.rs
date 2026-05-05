//! `/motor_state` topic — measured per-motor dynamics.
//!
//! Carries the KF-fused motor state estimates (`omega`, `omega_dot`,
//! and the raw bidir-DShot eRPM-derived measurement) the
//! [`crate::control::PROCESSED_MOTOR_STATE`] PubSub publishes at
//! 100 Hz.
//!
//! Pairs with `/motors` (the INDI **commanded** input) to give the
//! commanded-vs-achieved view that's needed to spot motor saturation,
//! ESC desync, prop damage, or single-motor failure. `/motors` says
//! "what the controller asked for"; `/motor_state` says "what
//! actually happened".
//!
//! `raw` is `Option<f32>` upstream; on disk we encode missing
//! samples as the JSON `null` value (CBOR major 7, simple value 22)
//! rather than substituting a sentinel — readers can distinguish a
//! genuine zero from "no telemetry frame this cycle".

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/motor_state`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 8;
pub const TOPIC: &str = "/motor_state";
pub const SCHEMA_NAME: &str = "MotorStateTelemetry";
pub const SCHEMA: &[u8] = br#"{
  "title": "MotorStateTelemetry",
  "type": "object",
  "properties": {
    "timestamp_ns": { "type": "integer" },
    "omega":        { "type": "array", "minItems": 4, "maxItems": 4,
                      "items": {"type": "number"},
                      "description": "Filtered motor angular velocity (rad/s), mixer order" },
    "omega_dot":    { "type": "array", "minItems": 4, "maxItems": 4,
                      "items": {"type": "number"},
                      "description": "Filtered motor angular acceleration (rad/s^2)" },
    "raw":          { "type": "array", "minItems": 4, "maxItems": 4,
                      "items": {"type": ["number", "null"]},
                      "description": "Raw bidir-DShot eRPM-derived measurement; null when telemetry frame missing" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::MotorStateTelemetry) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(4)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("omega")?;
    w.array(4)?;
    for d in m.motors.iter() {
        w.f32(d.omega)?;
    }
    w.str("omega_dot")?;
    w.array(4)?;
    for d in m.motors.iter() {
        w.f32(d.omega_dot)?;
    }
    w.str("raw")?;
    w.array(4)?;
    for d in m.motors.iter() {
        match d.raw {
            Some(v) => w.f32(v)?,
            None => w.null()?,
        }
    }
    Ok(w.pos())
}
