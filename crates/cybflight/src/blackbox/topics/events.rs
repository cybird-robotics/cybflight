//! `/events` topic — small CBOR records that bracket and annotate
//! a flight session.
//!
//! Used by the recorder to emit ARM at session start, DISARM /
//! LOG_END at session close, plus optional intra-flight events
//! (failsafe, NaN, stale-cmd) added by future stages.

use super::TopicDef;
use crate::blackbox::cbor::{self, CborWriter};
use embassy_time::Instant;

/// MCAP channel id for `/events`. Stable across all record-set
/// profiles.
pub const CHANNEL_ID: u16 = 4;
pub const TOPIC: &str = "/events";
pub const SCHEMA_NAME: &str = "Event";
pub const SCHEMA: &[u8] = br#"{
  "title": "Event",
  "type": "object",
  "properties": {
    "timestamp_ns": { "type": "integer" },
    "kind":         { "type": "integer",
                      "description": "Kind enum: 1=ARM, 2=DISARM, 3=FAILSAFE, 4=FAILSAFE_CLEAR, 5=ESTIMATOR_DOWN, 6=ESTIMATOR_UP, 7=RC_LOSS, 8=RC_RECOVERED, 16=LOG_END" },
    "data":         { "type": "integer", "description": "kind-specific: for KIND_FAILSAFE, encodes FailsafeReason (1=ControllerTimeout, 2=RcLoss); 0 otherwise" }
  }
}"#;

pub const DEF: TopicDef = TopicDef {
    channel_id: CHANNEL_ID,
    topic: TOPIC,
    schema_name: SCHEMA_NAME,
    schema_data: SCHEMA,
};

// Event kind codes. Stable on disk — never renumber. Add new
// codes only with values that don't conflict; existing tooling
// must keep working unchanged when it sees an unknown code.
pub const KIND_ARM: u8 = 0x01;
pub const KIND_DISARM: u8 = 0x02;
/// Failsafe ENTERED — controller silent >500 ms or RC loss
/// >1500 ms. The recorder watches `control::failsafe::FAILSAFE_ACTIVE`
/// inside the capture loop and emits this on the false→true edge.
pub const KIND_FAILSAFE: u8 = 0x03;
/// Failsafe cleared — RC link recovered while disarmed (the
/// failsafe state machine only re-arms via the normal arming
/// path, but the flag itself can clear during the same session
/// if the recorder is held via `RECORDER_HOLD`).
pub const KIND_FAILSAFE_CLEAR: u8 = 0x04;
/// ESKF lost convergence — `estimation::ESTIMATOR_READY` flipped
/// true→false. Indicates the state estimate the inner controllers
/// rely on can no longer be trusted; the failsafe path normally
/// disarms shortly after, but the event tags the *moment* of
/// estimator loss for post-flight diagnostics.
pub const KIND_ESTIMATOR_DOWN: u8 = 0x05;
/// ESKF (re)converged — false→true edge on `ESTIMATOR_READY`.
/// Emitted at session start once if the estimator was already
/// converged when the recorder opened, plus on every later
/// recovery.
pub const KIND_ESTIMATOR_UP: u8 = 0x06;
/// RC frames stopped arriving within `RXLOSS_TRIGGER` (150 ms).
/// Edge-detected from `control::failsafe::RC_LINK_HEALTHY`. Fires
/// **before** the failsafe state machine commits — a brief drop
/// that recovers within the guard period (`GUARD_PERIOD = 1500 ms`)
/// produces a `KIND_RC_LOSS` / `KIND_RC_RECOVERED` pair without
/// any `KIND_FAILSAFE`. Sustained RC loss produces
/// `KIND_RC_LOSS` then, ~1.5 s later, `KIND_FAILSAFE`
/// (with `data = FailsafeReason::RcLoss = 2`).
pub const KIND_RC_LOSS: u8 = 0x07;
/// RC frames resumed (Acquire-edge on `RC_LINK_HEALTHY`). Emitted
/// regardless of whether failsafe committed — captures both
/// "blip recovered before guard expired" and "post-failsafe
/// RC arriving but recovery_period not yet satisfied".
pub const KIND_RC_RECOVERED: u8 = 0x08;
pub const KIND_LOG_END: u8 = 0x10;

pub fn encode(scratch: &mut [u8], timestamp: Instant, kind: u8, data: u32) -> cbor::Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.map(3)?;
    w.str("timestamp_ns")?;
    w.u64(timestamp.as_micros().saturating_mul(1_000))?;
    w.str("kind")?;
    w.u64(kind as u64)?;
    w.str("data")?;
    w.u64(data as u64)?;
    Ok(w.pos())
}
