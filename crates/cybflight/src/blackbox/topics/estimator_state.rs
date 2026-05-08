//! `/estimator_state` topic — ESKF-estimated IMU biases.
//!
//! Mirror of [`crate::estimation::ESTIMATOR_BIAS_TELEM`] (decimated to
//! ~10 Hz inside the ESKF tasks). Pairs with `/imu1` for offline
//! bias-corrected accel reconstruction in INDI sysid fits.
//!
//! `est_eskf` builds only — feature-gated in the recorder. On
//! Mahony-only builds the topic is absent from the file even when
//! the active record-set tier nominally includes it.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/estimator_state`. Stable across all
/// record-set profiles.
pub const CHANNEL_ID: u16 = 12;
pub const TOPIC: &str = "/estimator_state";
pub const SCHEMA_NAME: &str = "EstimatorBias";
pub const SCHEMA: &[u8] = br#"{
  "title": "EstimatorBias",
  "type": "object",
  "properties": {
    "timestamp_ns":     { "type": "integer" },
    "gyro_bias_rad_s":  { "type": "array", "minItems": 3, "maxItems": 3,
                          "items": {"type": "number"},
                          "description": "ESKF gyro bias estimate (rad/s)" },
    "accel_bias_m_s2":  { "type": "array", "minItems": 3, "maxItems": 3,
                          "items": {"type": "number"},
                          "description": "ESKF accel bias estimate (m/s^2)" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::EstimatorBias) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(3)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("gyro_bias_rad_s")?;
    w.array(3)?;
    w.f32(m.gyro_bias_rad_s.x)?;
    w.f32(m.gyro_bias_rad_s.y)?;
    w.f32(m.gyro_bias_rad_s.z)?;
    w.str("accel_bias_m_s2")?;
    w.array(3)?;
    w.f32(m.accel_bias_m_s2.x)?;
    w.f32(m.accel_bias_m_s2.y)?;
    w.f32(m.accel_bias_m_s2.z)?;
    Ok(w.pos())
}
