//! `/imu1_raw` topic — pre-biquad-LP IMU mirror, compact array form.
//!
//! Same wire layout as `/imu1` ([`cybflight_core::blackbox_wire`]),
//! same units and element order — only the tap point differs (before
//! the accel/gyro biquad LPs instead of after). Large tier only.

use super::TopicDef;
use crate::blackbox::cbor;
use crate::msgs;
use cybflight_core::blackbox_wire;

/// MCAP channel id for `/imu1_raw`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 13;
pub const TOPIC: &str = "/imu1_raw";
/// `.v2` = the positional-array layout (old map layout was `ImuRaw`).
pub const SCHEMA_NAME: &str = "ImuRaw.v2";
pub const SCHEMA: &[u8] = br#"{
  "title": "ImuRaw.v2",
  "description": "Pre-biquad-LP IMU sample. Flat positional array, same element order as Imu.v2. accel m/s^2, gyro rad/s.",
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
