//! Per-topic schemas + CBOR encoders.
//!
//! Each module under `topics::` defines **everything Stage 5+ needs
//! to log one topic**:
//!
//! - `TOPIC: &'static str`        — MCAP topic name (`/imu1`, ...)
//! - `SCHEMA_NAME: &'static str`  — MCAP schema record `name` field
//! - `SCHEMA: &'static [u8]`      — JSON Schema document (encoding `"jsonschema"`)
//! - `pub fn encode(scratch: &mut [u8], msg: &MsgType) -> cbor::Result<usize>`
//! - `DEF: TopicDef`              — struct bundling the three constants for
//!                                   data-driven schema/channel emission
//!
//! [`crate::blackbox::recorder`] imports these and references them.
//! The schema/channel emission loop is data-driven via [`TopicDef`];
//! the per-message encode arm stays hand-written because each
//! topic's message type is different.
//!
//! ## Adding a new topic
//!
//! 1. Add a new `topics/<name>.rs` exporting the constants + `encode` fn above.
//! 2. Re-export the module here.
//! 3. In whichever capture op should include it, extend the `&[TopicDef]`
//!    array and add one `select` arm calling its `encode`.
//!
//! ## Why not a TOML config + `build.rs`?
//!
//! Considered. The encoders are necessarily typed Rust fns
//! (different `msgs::*` types per topic), so a config file would
//! reduce to a list of module names and add a build dependency for
//! marginal benefit. When staff define real log profiles in Stage 7,
//! the right knob is **cargo features** — `profile_flight`,
//! `profile_bench`, etc — selecting which `TopicDef` arrays the
//! capture ops include. That's compile-time config, zero runtime
//! cost.

pub mod attitude;
pub mod control_setpoint;
#[cfg(feature = "est_eskf")]
pub mod estimator_state;
pub mod events;
pub mod gps_health;
pub mod health;
pub mod imu;
pub mod imu_raw;
pub mod motor_state;
pub mod motors;
pub mod mpc;
pub mod odometry;
pub mod rc;
pub mod tracking_error;

/// Static description of one topic: everything the schema + channel
/// records need. Encoders are NOT here — they're per-topic typed fns
/// kept in each topic module.
///
/// `channel_id` is **stable per topic**, not per array index. The
/// record-set system mixes & matches topics into different
/// per-profile arrays (`crate::blackbox::record_set`), and
/// downstream MCAP tooling treats channel ids as opaque keys — so
/// we want `/imu1` to be channel id 1 whether the profile contains
/// just `[imu]` or `[events, rc, attitude, imu]`. Each topic module
/// hard-codes a `pub const CHANNEL_ID: u16` and embeds it in `DEF`.
pub struct TopicDef {
    pub channel_id: u16,
    pub topic: &'static str,
    pub schema_name: &'static str,
    pub schema_data: &'static [u8],
}
