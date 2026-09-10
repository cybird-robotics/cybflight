//! `/mpc_cost` topic — learned cost adaptation trace.
//!
//! One record per commanding outer-loop tick (docs/learned_mpc_cost_deploy.md):
//! the residual `z` that was in force for that tick's solve, the
//! effective contour/lag/rate weights it produced, the ramp-in gain and
//! the policy's wall time. This is what lets a flight's weight
//! modulation be compared against the sim probes
//! (`target/cost_policy_*/probe.json`).
//!
//! Published whether or not `mpc_learned_cost` is on, so every solve in
//! the log has the cost it actually ran under sitting next to it rather
//! than an inferred one. With the policy absent or disabled the record
//! carries `z = 0`, `gain = 0` and the vehicle's nominal weights. With it
//! enabled the policy runs on every tick — including off-trajectory,
//! where the record carries its live (out-of-distribution) `z` at
//! `gain = 0`: the weights are still nominal, the number is logged but
//! was not flown. `policy_time_us > 0` is the "the policy ran" signal.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/mpc_cost`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 16;
pub const TOPIC: &str = "/mpc_cost";
pub const SCHEMA_NAME: &str = "MpcCostAdapt";
pub const SCHEMA: &[u8] = br#"{
  "title": "MpcCostAdapt",
  "type": "object",
  "properties": {
    "timestamp_ns":   { "type": "integer" },
    "z":              { "type": "array", "items": {"type": "number"},
                        "description": "Policy output for the accompanying solve, clamped to [-1,1], BEFORE the ramp-in gain (multiply by `gain` for the applied residual): stage [contour, lag, vel x3, att x3], terminal [same 8], rates [x, y, z]. All-zero = nominal weights" },
    "w_contour":      { "type": "number" },
    "w_lag":          { "type": "number" },
    "w_rate":         { "type": "array", "items": {"type": "number"} },
    "gain":           { "type": "number",
                        "description": "Effective ramp-in gain for this solve: `mpc_learned_gain` while tracking a trajectory with the mission Executing, 0 otherwise (the policy is not trained off-trajectory, so its output is withheld)" },
    "policy_time_us": { "type": "integer" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::MpcCostAdapt) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(7)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("z")?;
    w.array(m.z.len() as u64)?;
    for v in m.z.iter() {
        w.f32(*v)?;
    }
    w.str("w_contour")?;
    w.f32(m.w_contour)?;
    w.str("w_lag")?;
    w.f32(m.w_lag)?;
    w.str("w_rate")?;
    w.array(3)?;
    for v in m.w_rate.iter() {
        w.f32(*v)?;
    }
    w.str("gain")?;
    w.f32(m.gain)?;
    w.str("policy_time_us")?;
    w.u64(m.policy_time_us as u64)?;
    Ok(w.pos())
}
