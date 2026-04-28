//! Reference samplers for the MPC outer loop.
//!
//! A `Sampler` fills a per-horizon-node buffer of flat-output references
//! (position, velocity, acceleration, plus a `past_end` flag) from a single
//! `PiecewisePolynomial` mission trajectory. Quaternion construction stays
//! at the call site, where the desired yaw is owned.
//!
//! Two implementations exist:
//!
//! - [`TimeSampler`] — fills the horizon at `τ = tau0_s + k·dt`, clamped
//!   to the trajectory's `[0, total_duration_s]` interval. This matches
//!   the in-line behaviour the outer loop carried before this module was
//!   introduced.
//! - [`PositionSampler`] — closest-point search on the trajectory in a
//!   weighted (position, time-anchor) cost, then horizon fill from the
//!   resolved `τ`. Selected at compile time via the `position_sampler`
//!   feature on the firmware crate.
//!
//! The samplers are pure: they read inputs and write into a caller-owned
//! `&mut [SamplerNode]`. They do not touch mission state, the active
//! setpoint cell, or any other shared firmware globals — that ownership
//! stays with the outer loop.

use super::piecewise_polynomial::PiecewisePolynomial;
use super::types::Vec3;

mod position;
mod time;

pub use position::{PositionSampler, PositionSamplerParams};
pub use time::TimeSampler;

/// Per-horizon-node flat-output reference, filled by a sampler.
///
/// `past_end == true` means this node sits at or beyond the end of the
/// trajectory. Callers should treat it as a terminal-hover node:
/// `(pos = end, vel = 0, acc = 0)`, and typically substitute an
/// identity-tilt attitude reference rather than calling the flatness map
/// on a zero acceleration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerNode {
    pub pos: Vec3,
    pub vel: Vec3,
    pub acc: Vec3,
    pub past_end: bool,
}

impl Default for SamplerNode {
    fn default() -> Self {
        Self {
            pos: Vec3::zeros(),
            vel: Vec3::zeros(),
            acc: Vec3::zeros(),
            past_end: false,
        }
    }
}

/// Inputs to a single `sample` call.
///
/// `tau0_s` is the trajectory time the caller wants node 0 to track — already
/// future-dated-start-clamped (i.e. `≥ 0`). The caller owns the conversion
/// from monotonic clock to seconds because doing it inside the sampler
/// would force two independent `f32` conversions and lose microsecond
/// precision on the subtraction.
pub struct SamplerInputs<'a> {
    pub traj: &'a PiecewisePolynomial,
    /// `traj.total_duration()` cached by the caller — sampler never recomputes.
    pub total_duration_s: f32,
    /// Trajectory time at node 0. `0.0` means "wait at the start"; values
    /// `≥ total_duration_s` flag past-end and trigger `mission_done`.
    pub tau0_s: f32,
    /// Current vehicle position (used by `PositionSampler`; ignored by `TimeSampler`).
    pub state_pos: Vec3,
    /// Spacing between successive horizon nodes (the MPC discretisation step).
    pub horizon_dt: f32,
}

/// Result returned alongside the filled buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleResult {
    /// `τ` at node 0 — the trajectory time the controller is currently
    /// tracking. Used by the outer loop for telemetry and to refresh the
    /// active-setpoint cell.
    pub tau0_s: f32,
    /// Node 0 has reached or passed the end of the trajectory. The outer
    /// loop owns the resulting state-machine transition (Executing → Idle).
    pub mission_done: bool,
}

/// Compile-time dispatch over the available samplers. `outer_loop` holds
/// one of these as a task-local. New variants drop in as additional
/// `Sampler::*` cases without changing the call-site contract.
pub enum Sampler {
    Time(TimeSampler),
    Position(PositionSampler),
}

impl Sampler {
    /// Per-mission lifecycle hook. The outer loop calls this on the
    /// `Idle → Executing` transition so a stateful sampler (e.g. the
    /// position sampler's `prev_query_tau`) starts each mission fresh.
    /// `TimeSampler` is stateless, so this is a no-op for it.
    #[inline]
    pub fn reset(&mut self) {
        match self {
            Sampler::Time(s) => s.reset(),
            Sampler::Position(s) => s.reset(),
        }
    }

    /// Fill `out` with one `SamplerNode` per horizon node. `out.len()` is
    /// the number of nodes (typically `MPC_N + 1`). The sampler asserts
    /// `out.len() >= 1` and `inputs.horizon_dt > 0`.
    #[inline]
    pub fn sample(&mut self, inputs: &SamplerInputs<'_>, out: &mut [SamplerNode]) -> SampleResult {
        match self {
            Sampler::Time(s) => s.sample(inputs, out),
            Sampler::Position(s) => s.sample(inputs, out),
        }
    }
}
