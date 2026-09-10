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
pub mod mpc_cost;
pub mod odometry;
pub mod power;
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
///
/// **Ids must be globally unique across every topic**, not merely
/// within one tier: MCAP identifies Channel *and* Schema records by
/// id, and a file carrying two non-identical records under one id is
/// malformed — readers either reject it or silently keep whichever
/// was written last, so the other topic's messages are mislabelled
/// and decoded against the wrong schema.
///
/// [`ALL`] is the registry that makes that checkable;
/// `record_set`'s const assertions enforce it at compile time. When
/// adding a topic, take the next free id from [`ALL`] and add the
/// module's `DEF` to it — do not infer "the next free id" from the
/// docs or from one tier's array.
pub struct TopicDef {
    pub channel_id: u16,
    pub topic: &'static str,
    pub schema_name: &'static str,
    pub schema_data: &'static [u8],
}

/// Every topic the firmware can log, regardless of tier.
///
/// Exists so channel-id uniqueness is checkable at compile time
/// (see `crate::blackbox::record_set`'s const assertions) rather
/// than being an invariant maintained by hand across 14 files. A
/// topic missing from this list is not a build error, but it does
/// escape the uniqueness check — so add new topics here first.
///
/// Historical note: ids 10 and 11 were each issued twice, because
/// `/health` and `/gps_health` were documented as "channel id
/// reserved" without a concrete value and the next author read ids
/// 1–9 as the high-water mark. `/imu1_raw` and `/control_setpoint`
/// were moved to 13/14 to resolve it; 12 stays with
/// `/estimator_state`, which was never in conflict.
pub const ALL: &[TopicDef] = &[
    imu::DEF,              // 1
    attitude::DEF,         // 2
    rc::DEF,               // 3
    events::DEF,           // 4
    odometry::DEF,         // 5
    mpc::DEF,              // 6
    motors::DEF,           // 7
    motor_state::DEF,      // 8
    tracking_error::DEF,   // 9
    health::DEF,           // 10
    gps_health::DEF,       // 11
    #[cfg(feature = "est_eskf")]
    estimator_state::DEF,  // 12
    imu_raw::DEF,          // 13
    control_setpoint::DEF, // 14
    power::DEF,            // 15
];
