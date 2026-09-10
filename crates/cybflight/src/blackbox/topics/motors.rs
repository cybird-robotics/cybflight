//! `/motors` topic — INDI inner-loop output (per-motor normalized
//! commands).
//!
//! Carries the four normalized motor commands the INDI loop publishes
//! to the DShot driver. Sourced from
//! [`crate::control::ACTUATOR_MOTORS_TELEM`] (a 100 Hz decimated
//! mirror of the IMU-rate `motors::ACTUATOR_MOTORS` Signal — going to
//! the decimated PubSub keeps the recorder out of the hot inner-loop path).
//!
//! Useful for spotting motor saturation, mixer asymmetry, or ESC
//! desync after the fact: pair with `/odometry` (state) and `/mpc`
//! (high-level command) to see the full inner-to-outer-loop
//! response.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/motors`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 7;
pub const TOPIC: &str = "/motors";
pub const SCHEMA_NAME: &str = "ActuatorMotors";
pub const SCHEMA: &[u8] = br#"{
  "title": "ActuatorMotors",
  "type": "object",
  "properties": {
    "timestamp_ns":   { "type": "integer" },
    "motor_commands": { "type": "array", "minItems": 4, "maxItems": 4,
                        "items": {"type": "number"},
                        "description": "Normalized [0..1] per-motor command, mixer order" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::ActuatorMotors) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(2)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("motor_commands")?;
    w.array(4)?;
    for c in m.motor_commands.iter() {
        w.f32(c.value())?;
    }
    Ok(w.pos())
}
