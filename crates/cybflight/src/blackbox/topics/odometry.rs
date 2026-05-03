//! `/odometry` topic — fused-state output of the ESKF.
//!
//! Carries the vehicle's pose (position + orientation) and twist
//! (linear + angular velocity). Published at ~1 kHz by the
//! estimator; this is what the controllers and shell snapshots
//! consume as ground truth for "where the airframe is + how fast".

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/odometry`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 5;
pub const TOPIC: &str = "/odometry";
pub const SCHEMA_NAME: &str = "VehicleOdometry";
pub const SCHEMA: &[u8] = br#"{
  "title": "VehicleOdometry",
  "type": "object",
  "properties": {
    "timestamp_ns":    { "type": "integer" },
    "position":        { "type": "array", "minItems": 3, "maxItems": 3,
                         "items": {"type": "number"},
                         "description": "[x, y, z] in body/world frame as published by ESKF" },
    "orientation":     { "type": "array", "minItems": 4, "maxItems": 4,
                         "items": {"type": "number"},
                         "description": "[w, i, j, k] (UnitQuaternion)" },
    "linear_velocity": { "type": "array", "minItems": 3, "maxItems": 3,
                         "items": {"type": "number"} },
    "angular_velocity":{ "type": "array", "minItems": 3, "maxItems": 3,
                         "items": {"type": "number"} }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::VehicleOdometry) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(5)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("position")?;
    w.array(3)?;
    w.f32(m.pose.position.x)?;
    w.f32(m.pose.position.y)?;
    w.f32(m.pose.position.z)?;
    w.str("orientation")?;
    w.array(4)?;
    let q = m.pose.orientation.as_ref();
    w.f32(q.w)?;
    w.f32(q.i)?;
    w.f32(q.j)?;
    w.f32(q.k)?;
    w.str("linear_velocity")?;
    w.array(3)?;
    w.f32(m.twist.linear.x)?;
    w.f32(m.twist.linear.y)?;
    w.f32(m.twist.linear.z)?;
    w.str("angular_velocity")?;
    w.array(3)?;
    w.f32(m.twist.angular.x)?;
    w.f32(m.twist.angular.y)?;
    w.f32(m.twist.angular.z)?;
    Ok(w.pos())
}
