//! `/mpc` topic — output of the MPC / OCP solver.
//!
//! Carries the high-level control command emitted by the outer
//! loop's NMPC each tick, plus solver telemetry (iterations,
//! convergence, wall-clock solve time). Useful for understanding
//! whether the controller saturated, timed out, or commanded
//! something the inner loop couldn't track.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use crate::msgs;

/// MCAP channel id for `/mpc`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 6;
pub const TOPIC: &str = "/mpc";
pub const SCHEMA_NAME: &str = "OcpSolverOutput";
pub const SCHEMA: &[u8] = br#"{
  "title": "OcpSolverOutput",
  "type": "object",
  "properties": {
    "timestamp_ns":  { "type": "integer" },
    "command":       { "type": "array", "items": {"type": "number"},
                       "description": "Solver-output command vector (size depends on OCP problem)" },
    "iterations":    { "type": "integer" },
    "converged":     { "type": "boolean" },
    "solve_time_us": { "type": "integer" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

pub fn encode(scratch: &mut [u8], m: &msgs::OcpSolverOutput) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(5)?;
    w.str("timestamp_ns")?;
    w.u64(m.timestamp.as_micros().saturating_mul(1_000))?;
    w.str("command")?;
    let cmd = m.command.as_slice();
    w.array(cmd.len() as u64)?;
    for v in cmd.iter() {
        w.f32(*v)?;
    }
    w.str("iterations")?;
    w.i64(m.iterations as i64)?;
    w.str("converged")?;
    w.bool(m.converged)?;
    w.str("solve_time_us")?;
    w.u64(m.solve_time_us)?;
    Ok(w.pos())
}
