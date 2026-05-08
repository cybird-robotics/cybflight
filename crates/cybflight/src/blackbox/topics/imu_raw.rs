//! `/imu1_raw` topic — pre-biquad-LP IMU sample mirror of `/imu1`.
//!
//! Sourced from [`crate::sensors::IMU_1_RAW`], a parallel
//! 8 kHz channel published by the IMU reader task **before** the
//! accel/gyro biquads apply. Pairs with `/imu1` (post-LP) so
//! analyse.py-style RPM-notch and filter-tuning fits have access to
//! the raw spectrum.
//!
//! Schema body matches `Imu` exactly. The schema *name* is distinct
//! ("ImuRaw") so MCAP consumers can filter by schema even when both
//! topics are present in the same file.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/imu1_raw`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 10;
pub const TOPIC: &str = "/imu1_raw";
pub const SCHEMA_NAME: &str = "ImuRaw";
pub const SCHEMA: &[u8] = br#"{
  "title": "ImuRaw",
  "type": "object",
  "properties": {
    "timestamp_ns": { "type": "integer" },
    "accel_m_s2":   { "type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"},
                      "description": "Pre-biquad-LP accelerometer (m/s^2)" },
    "gyro_rad_s":   { "type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"},
                      "description": "Pre-biquad-LP gyroscope (rad/s)" },
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
