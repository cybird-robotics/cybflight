//! Position-based reference sampler — pure axis-weighted closest-point
//! search on the trajectory. Single-trajectory port of the field-tested
//! `MyPositionSampler` from `agilib`.
//!
//! Each tick the sampler walks `τ` forward from where it left off last
//! tick, picking the largest `τ` for which the axis-weighted distance
//!
//! ```text
//!   d(τ) = ‖axis_w ⊙ (traj.get_pos(τ) − state_pos)‖
//! ```
//!
//! does not increase by more than `search_tol`. The slack mirrors
//! `MyPositionSampler`'s "prioritize future setpoints" tolerance and biases
//! the search toward forward progress on plateaus and through
//! tracking-error noise.
//!
//! The horizon is then filled from the converged `τ` using the same
//! kernel as `TimeSampler` (`τ_k = (τ_curr + k · dt).min(end)`).
//!
//! Behaviour notes:
//!
//! - **No time-penalty term.** Earlier ports tried a time-anchored cost
//!   (`(1 + tw·|τ − τ_anchor|)`); on curvy paths that loop near themselves
//!   the anchor either stalled the forward search or pulled τ to a stale
//!   earlier root, producing abrupt setpoint jumps. The simpler
//!   pure-position search has no such failure mode and matches the
//!   original `agilib` implementation that worked in flight.
//! - **Forward-bias slack.** `next_dist ≤ curr_dist + search_tol` advances
//!   even when the next candidate is marginally worse, so flat regions
//!   and sensor noise don't lock the search prematurely. `search_tol` has
//!   units of metres (axis-weighted).
//! - **`prev_query_tau` is monotone within a mission.** The search is
//!   forward-only; only `reset()` (called by the outer loop on
//!   Idle→Executing) ever rewinds it.
//! - **`SamplerInputs.tau0_s` is ignored.** The field is part of the
//!   shared `SamplerInputs` contract because `TimeSampler` consumes it
//!   as the time-based τ to track. `PositionSampler`'s cost depends only
//!   on `state_pos` and the trajectory geometry, so any value (including
//!   NaN/Inf) passed in `tau0_s` produces identical output here.

use super::super::types::Vec3;
use super::{SampleResult, SamplerInputs, SamplerNode};

#[derive(Clone, Copy, Debug)]
pub struct PositionSamplerParams {
    /// Per-axis sqrt-weights on the position-error vector inside the
    /// distance term. `Vec3::new(1, 1, 1)` weights all axes equally;
    /// lowering Z effectively says "match XY tightly, Z loosely".
    pub axis_weights_sqrt: Vec3,
    /// Step size for the forward search. Smaller = finer resolution but
    /// more polynomial evaluations per tick (capped by `max_search_steps`).
    pub search_dt: f32,
    /// Forward-progress slack (metres, axis-weighted). The search advances
    /// from `curr` to `next` whenever `d(next) ≤ d(curr) + search_tol`.
    /// Set to a value just above the controller's expected positional
    /// noise floor so legitimate forward progress isn't blocked by
    /// jitter, but real divergence still halts the search.
    pub search_tol: f32,
    /// Hard cap on per-tick `traj.get_pos` calls during search. Bounds
    /// the worst-case CPU on every outer-loop tick. With `search_dt = 10 ms`
    /// and `max_search_steps = 50` the search sweeps at most 0.5 s of
    /// trajectory per tick, which is plenty when the controller is
    /// tracking with bounded position error.
    pub max_search_steps: u16,
    /// Position tolerance for declaring the mission done. When
    /// `‖state_pos − traj.get_pos(end)‖ < radius_of_acceptance`, the
    /// sampler reports `mission_done = true` even if `τ_curr` is still
    /// short of the terminal time.
    pub radius_of_acceptance: f32,
}

impl PositionSamplerParams {
    pub const fn defaults() -> Self {
        Self {
            axis_weights_sqrt: Vec3::new(1.0, 1.0, 1.0),
            search_dt: 0.01,
            search_tol: 1e-3,
            max_search_steps: 100,
            radius_of_acceptance: 0.15,
        }
    }
}

impl Default for PositionSamplerParams {
    fn default() -> Self {
        Self::defaults()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PositionSampler {
    pub params: PositionSamplerParams,
    /// Last tick's converged `τ`. `None` immediately after construction
    /// or `reset()`; the next `sample()` then starts from `τ = 0`.
    prev_query_tau: Option<f32>,
}

impl PositionSampler {
    pub const fn new(params: PositionSamplerParams) -> Self {
        Self {
            params,
            prev_query_tau: None,
        }
    }

    /// Per-mission lifecycle hook. Called by the outer loop on the
    /// `Idle → Executing` transition so the next mission's search starts
    /// from `τ = 0` instead of inheriting the previous mission's final τ.
    #[inline]
    pub fn reset(&mut self) {
        self.prev_query_tau = None;
    }

    /// For tests: peek at the stored `prev_query_tau`. Not used by the
    /// outer loop — the firmware only ever interacts with the sampler
    /// through `sample()` and `reset()`.
    #[doc(hidden)]
    pub fn prev_query_tau(&self) -> Option<f32> {
        self.prev_query_tau
    }

    pub fn sample(&mut self, inp: &SamplerInputs<'_>, out: &mut [SamplerNode]) -> SampleResult {
        debug_assert!(
            !out.is_empty(),
            "PositionSampler::sample: empty output buffer"
        );
        debug_assert!(
            inp.horizon_dt > 0.0,
            "PositionSampler::sample: horizon_dt must be positive"
        );
        debug_assert!(
            self.params.search_dt > 0.0,
            "PositionSampler::sample: search_dt must be positive"
        );

        // Non-finite-input guard. A NaN/Inf in `state_pos` poisons every
        // distance, makes the advance check always false (NaN
        // comparisons), and locks `prev_query_tau` at its previous value
        // forever — the controller would then track a stale τ-sample for
        // the rest of the mission and `mission_done` could never fire.
        // The outer loop's odom validity gate normally catches
        // `state_pos`, but we defend in depth here.
        //
        // On non-finite input we leave `prev_query_tau` untouched, fill
        // the output buffer with a deterministic hover-at-current-τ
        // pattern (so the caller never reads uninitialised data), and
        // report `mission_done = false`.
        let inputs_finite = inp.state_pos.x.is_finite()
            && inp.state_pos.y.is_finite()
            && inp.state_pos.z.is_finite()
            && inp.total_duration_s.is_finite();
        if !inputs_finite {
            let tau_safe = self.prev_query_tau.unwrap_or(0.0);
            let pos_safe = if inp.total_duration_s.is_finite() {
                inp.traj.get_pos(tau_safe.clamp(0.0, inp.total_duration_s))
            } else {
                Vec3::zeros()
            };
            for node in out.iter_mut() {
                *node = SamplerNode {
                    pos: pos_safe,
                    vel: Vec3::zeros(),
                    acc: Vec3::zeros(),
                    past_end: false,
                };
            }
            return SampleResult {
                tau0_s: tau_safe,
                mission_done: false,
            };
        }

        let end = inp.total_duration_s;
        let axis_w = self.params.axis_weights_sqrt;
        let tol = self.params.search_tol;

        let mut tau_curr = self.prev_query_tau.unwrap_or(0.0).clamp(0.0, end);
        let mut dist_curr = weighted_distance(inp.traj.get_pos(tau_curr), inp.state_pos, axis_w);

        // Forward closest-point search. Cap on iterations bounds CPU.
        for _ in 0..self.params.max_search_steps {
            let tau_next = tau_curr + self.params.search_dt;
            if tau_next >= end {
                // Try one final step clamped to the endpoint, under the
                // same tolerance check.
                let dist_end = weighted_distance(inp.traj.get_pos(end), inp.state_pos, axis_w);
                if dist_end <= dist_curr + tol {
                    tau_curr = end;
                }
                break;
            }
            let dist_next = weighted_distance(inp.traj.get_pos(tau_next), inp.state_pos, axis_w);
            if dist_next <= dist_curr + tol {
                tau_curr = tau_next;
                dist_curr = dist_next;
            } else {
                break;
            }
        }

        self.prev_query_tau = Some(tau_curr);

        // Horizon fill — same kernel as TimeSampler, anchored at tau_curr.
        for (k, node) in out.iter_mut().enumerate() {
            let t_k = (tau_curr + k as f32 * inp.horizon_dt).min(end);
            let past_end = t_k >= end;
            let (pos, vel, acc) = if past_end {
                (inp.traj.get_pos(end), Vec3::zeros(), Vec3::zeros())
            } else {
                (
                    inp.traj.get_pos(t_k),
                    inp.traj.get_vel(t_k),
                    inp.traj.get_acc(t_k),
                )
            };
            *node = SamplerNode {
                pos,
                vel,
                acc,
                past_end,
            };
        }

        // Mission done if the time index has reached the endpoint OR if
        // the state is already inside the acceptance radius of the
        // terminal pose.
        //
        // The `>= end - search_dt` threshold (rather than `>= end`)
        // tolerates f32 accumulation across many `+search_dt` steps:
        // 199 additions of `0.01` lands at `1.9999985`, not `2.0`. The
        // search-step granularity is the natural floor anyway — being
        // within one step of the endpoint is "at the endpoint" for any
        // downstream purpose.
        let mission_done_by_time = tau_curr >= end - self.params.search_dt;
        let mission_done_by_radius = !mission_done_by_time && {
            let end_pos = inp.traj.get_pos(end);
            let r = self.params.radius_of_acceptance;
            let d = inp.state_pos - end_pos;
            d.dot(&d) <= r * r
        };

        SampleResult {
            tau0_s: tau_curr,
            mission_done: mission_done_by_time || mission_done_by_radius,
        }
    }
}

#[inline]
fn weighted_distance(traj_pos: Vec3, state_pos: Vec3, axis_w: Vec3) -> f32 {
    let diff = traj_pos - state_pos;
    let weighted = Vec3::new(
        axis_w[0] * diff[0],
        axis_w[1] * diff[1],
        axis_w[2] * diff[2],
    );
    libm::sqrtf(weighted.dot(&weighted))
}
