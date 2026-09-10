//! `/odometry` topic — ESKF fused pose + twist, compact array form.
//!
//! The 1 kHz `/odometry` stream is the single largest byte consumer
//! on `imu_1khz` builds, so it uses the positional-array wire format
//! from [`cybflight_core::blackbox_wire`] (see the layout + golden
//! tests there). Its `pose.orientation` doubles as the attitude
//! estimate — the reason the old `/attitude` topic could be dropped.

use super::TopicDef;
use crate::blackbox::cbor;
use crate::msgs;
use cybflight_core::blackbox_wire;

/// MCAP channel id for `/odometry`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 5;
pub const TOPIC: &str = "/odometry";
/// `.v2` = the positional-array layout (old map layout was
/// `VehicleOdometry`).
pub const SCHEMA_NAME: &str = "VehicleOdometry.v2";
pub const SCHEMA: &[u8] = br#"{
  "title": "VehicleOdometry.v2",
  "description": "ESKF fused pose+twist. Flat positional array; see prefixItems for element order. Quaternion is [w,i,j,k]. position m, velocity m/s, angular velocity rad/s.",
  "type": "array",
  "minItems": 14, "maxItems": 14,
  "prefixItems": [
    { "title": "timestamp_ns", "type": "integer" },
    { "title": "pos_x_m", "type": "number" },
    { "title": "pos_y_m", "type": "number" },
    { "title": "pos_z_m", "type": "number" },
    { "title": "q_w", "type": "number" },
    { "title": "q_i", "type": "number" },
    { "title": "q_j", "type": "number" },
    { "title": "q_k", "type": "number" },
    { "title": "vel_x_m_s", "type": "number" },
    { "title": "vel_y_m_s", "type": "number" },
    { "title": "vel_z_m_s", "type": "number" },
    { "title": "omega_x_rad_s", "type": "number" },
    { "title": "omega_y_rad_s", "type": "number" },
    { "title": "omega_z_rad_s", "type": "number" }
  ]
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::VehicleOdometry) -> cbor::Result<usize> {
    let q = m.pose.orientation.as_ref();
    blackbox_wire::encode_odometry(
        scratch,
        m.timestamp.as_micros().saturating_mul(1_000),
        &m.pose.position.into(),
        &[q.w, q.i, q.j, q.k],
        &m.twist.linear.into(),
        &m.twist.angular.into(),
    )
}
