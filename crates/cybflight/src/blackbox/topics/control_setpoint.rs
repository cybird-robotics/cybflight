//! `/control_setpoint` topic — outer-loop output to INDI.
//!
//! Mirrors `crate::control::RATE_COMMAND` (a `Signal` consumed by INDI
//! at 8 kHz via `try_take()`). The active outer-loop task publishes
//! the same `AttitudeControlSetpoint` to
//! [`crate::control::CONTROL_SETPOINT_TELEM`] at the loop rate
//! (50–100 Hz, depending on outer mode), keeping INDI's hot path
//! untouched.
//!
//! Pairs with `/tracking_error` (body_rate_err from INDI) and `/imu1`
//! (gyro_rad_s) to give the full `setpoint → measurement → error`
//! triangle that controller-tuning and INDI sysid both want.
//!
//! `torque_n_m` is omitted from the schema — every current publish
//! site sets it to `Vector3::zeros()`. Re-add when a controller
//! starts populating it.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/control_setpoint`. Stable across all
/// record-set profiles.
pub const CHANNEL_ID: u16 = 11;
pub const TOPIC: &str = "/control_setpoint";
pub const SCHEMA_NAME: &str = "AttitudeControlSetpoint";
pub const SCHEMA: &[u8] = br#"{
  "title": "AttitudeControlSetpoint",
  "type": "object",
  "properties": {
    "timestamp_ns":         { "type": "integer" },
    "collective_thrust_n":  { "type": "number",
                              "description": "Newtons of collective thrust commanded by the outer loop" },
    "body_rate_rad_s":      { "type": "array", "minItems": 3, "maxItems": 3,
                              "items": {"type": "number"},
                              "description": "Body-frame rate setpoint [rad/s]; consumed by INDI" },
    "attitude_quaternion":  { "type": "array", "minItems": 4, "maxItems": 4,
                              "items": {"type": "number"},
                              "description": "[w, i, j, k] reference attitude" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::AttitudeControlSetpoint) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(4)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("collective_thrust_n")?;
    w.f32(m.collective_thrust_n)?;
    w.str("body_rate_rad_s")?;
    w.array(3)?;
    w.f32(m.body_rate_rad_s.x)?;
    w.f32(m.body_rate_rad_s.y)?;
    w.f32(m.body_rate_rad_s.z)?;
    w.str("attitude_quaternion")?;
    w.array(4)?;
    let q = m.attitude_quaternion.as_ref();
    w.f32(q.w)?;
    w.f32(q.i)?;
    w.f32(q.j)?;
    w.f32(q.k)?;
    Ok(w.pos())
}
