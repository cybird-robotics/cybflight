//! `/tracking_error` topic — controller-reported `reference - actual`
//! per tick.
//!
//! Sourced from [`crate::control::TRACKING_ERROR`], a multi-publisher
//! channel fed by:
//!
//! - the **MPC outer loop** (50 Hz) — pos / vel / attitude error
//!   against the τ₀ reference, in the same parameterization the SQP
//!   cost uses (`model_utils::attitude_error`);
//! - the **cascade outer loop** (100 Hz) — pos / vel error against
//!   the live position setpoint, raw (unclamped);
//! - the **INDI inner loop** (100 Hz, decimated from 8 kHz) —
//!   body-rate error `rate_ref - gyro_corrected`.
//!
//! Each publish carries a `source` byte
//! ([`msgs::TRACKING_ERROR_SOURCE_*`]) so consumers know which
//! subset of fields is meaningful — fields the publishing source
//! doesn't compute are zero.
//!
//! Why log this rather than reconstruct on the host? Three reasons,
//! ordered by importance:
//!
//! 1. **Same-instant.** Recorded `reference` and `actual` channels
//!    can lag each other by ~10–20 ms; the in-controller error is
//!    the exact value the controller acted on.
//! 2. **MPC's tilt-prio attitude error** is non-trivial to
//!    re-derive on the host (and would have to track firmware
//!    changes to `model_utils::attitude_error`).
//! 3. **INDI's `gyro_corrected`** is post-RPM-notch; reconstructing
//!    it from raw IMU + motor freqs requires re-running the notch
//!    bank, which is more state than the host wants to carry.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::control::TrackingError;

/// MCAP channel id for `/tracking_error`. Stable across all
/// record-set profiles.
pub const CHANNEL_ID: u16 = 9;
pub const TOPIC: &str = "/tracking_error";
pub const SCHEMA_NAME: &str = "TrackingError";
pub const SCHEMA: &[u8] = br#"{
  "title": "TrackingError",
  "type": "object",
  "properties": {
    "timestamp_ns":  { "type": "integer" },
    "pos_err":       { "type": "array", "minItems": 3, "maxItems": 3,
                       "items": {"type": "number"},
                       "description": "World-frame reference - actual position [m]; raw (unclamped)" },
    "vel_err":       { "type": "array", "minItems": 3, "maxItems": 3,
                       "items": {"type": "number"},
                       "description": "World-frame reference - actual velocity [m/s]; raw" },
    "attitude_err":  { "type": "array", "minItems": 3, "maxItems": 3,
                       "items": {"type": "number"},
                       "description": "Body-frame tilt-prio 3-vec from model_utils::attitude_error (MPC source only)" },
    "body_rate_err": { "type": "array", "minItems": 3, "maxItems": 3,
                       "items": {"type": "number"},
                       "description": "Body-frame rate_ref - gyro_corrected [rad/s] (INDI source only)" },
    "source":        { "type": "integer",
                       "description": "0=cascade, 1=mpc, 2=indi. Tells the consumer which fields are meaningful; others are zero." }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &TrackingError) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(6)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("pos_err")?;
    w.array(3)?;
    w.f32(m.pos_err.x)?;
    w.f32(m.pos_err.y)?;
    w.f32(m.pos_err.z)?;
    w.str("vel_err")?;
    w.array(3)?;
    w.f32(m.vel_err.x)?;
    w.f32(m.vel_err.y)?;
    w.f32(m.vel_err.z)?;
    w.str("attitude_err")?;
    w.array(3)?;
    w.f32(m.attitude_err.x)?;
    w.f32(m.attitude_err.y)?;
    w.f32(m.attitude_err.z)?;
    w.str("body_rate_err")?;
    w.array(3)?;
    w.f32(m.body_rate_err.x)?;
    w.f32(m.body_rate_err.y)?;
    w.f32(m.body_rate_err.z)?;
    w.str("source")?;
    w.u64(m.source as u64)?;
    Ok(w.pos())
}
