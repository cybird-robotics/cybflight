//! Neural-network flight policies.
//!
//! [`mlp`] is generic feed-forward inference; [`race_policy`] wraps it with
//! the observation assembly, gate state machine and action mapping that turn
//! a trained checkpoint into a controller producing normalized motor
//! commands.
//!
//! The policy is a **static, memoryless map** — no recurrence, no internal
//! filter state, no delta-action. It may therefore be evaluated at any rate
//! without changing its meaning; the rate it was trained at sets the
//! zero-order-hold delay it was tuned against, not a structural constraint.
//! Running faster reduces that delay and so increases phase margin. What it
//! does *not* do is filter sensor noise: policies trained on noiseless
//! simulator states see every bit of measurement noise the caller passes in,
//! which is a property of the observation, not of the rate.

pub mod mlp;
pub mod race_policy;

pub use mlp::{Activation, LayerShape, Mlp, MlpError, MAX_WIDTH};
pub use race_policy::{
    raw_state_in_frame, Gate, GateEvent, PolicyConfig, PolicyError, PolicyFrame, RacePolicy,
    Track, VehicleState, MAX_GATES, MAX_OBS, NUM_MOTORS, OBS_BASE, OBS_PER_GATE,
};
