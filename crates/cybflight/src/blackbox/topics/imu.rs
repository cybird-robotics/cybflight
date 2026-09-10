//! `/imu1` topic definition — compact positional-array wire format.
//!
//! Encoded by [`cybflight_core::blackbox_wire::encode_imu`]; see that
//! module for the layout rationale (string keys dominated the old map
//! form at IMU rate) and the golden tests that pin the element order.
//!
//! `temp_c` is not in this record: it changes at ~1 Hz and cost
//! 12 B/sample in the map form. The recorder mirrors the latest IMU
//! temperature into the 20 Hz `/health` record instead.

use super::TopicDef;
use crate::blackbox::cbor;
use crate::msgs;
use cybflight_core::blackbox_wire;

/// MCAP channel id for `/imu1`. Stable across all record-set
/// profiles — the value the topic carries on disk.
pub const CHANNEL_ID: u16 = 1;
pub const TOPIC: &str = "/imu1";
/// `.v2` = the positional-array layout. Readers key their decoder on
/// this name; the old map layout was plain `Imu`.
pub const SCHEMA_NAME: &str = "Imu.v2";
pub const SCHEMA: &[u8] = br#"{
  "title": "Imu.v2",
  "description": "Flat positional array; see prefixItems for element order. accel m/s^2, gyro rad/s. temp_c moved to /health.imu1_temp_c.",
  "type": "array",
  "minItems": 7, "maxItems": 7,
  "prefixItems": [
    { "title": "timestamp_ns", "type": "integer" },
    { "title": "accel_x_m_s2", "type": "number" },
    { "title": "accel_y_m_s2", "type": "number" },
    { "title": "accel_z_m_s2", "type": "number" },
    { "title": "gyro_x_rad_s", "type": "number" },
    { "title": "gyro_y_rad_s", "type": "number" },
    { "title": "gyro_z_rad_s", "type": "number" }
  ]
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], sample: &msgs::Imu) -> cbor::Result<usize> {
    blackbox_wire::encode_imu(
        scratch,
        sample.timestamp.as_micros().saturating_mul(1_000),
        &sample.accel_m_s2.into(),
        &sample.gyro_rad_s.into(),
    )
}
