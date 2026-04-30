//! Time-based reference sampler — pure extraction of the inline loop the
//! outer MPC task carried before the `Sampler` abstraction landed.
//!
//! Per node `k = 0..=N`:
//!
//! ```text
//!   τ_k      = (τ₀ + k · dt).min(total_duration_s)
//!   past_end = τ_k >= total_duration_s
//!   (p,v,a)  = past_end ? (traj(end), 0, 0) : (traj.pos|vel|acc(τ_k))
//! ```
//!
//! `τ₀` arrives pre-clamped from the caller (`SamplerInputs::tau0_s`).
//! A future-dated start corresponds to `τ₀ = 0` so the controller idles
//! at the trajectory's initial point. `mission_done` flips when node 0
//! itself has reached the end.

use super::super::types::Vec3;
use super::{SampleResult, SamplerInputs, SamplerNode};

/// Stateless time-based sampler. Constructable with `TimeSampler`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimeSampler;

impl TimeSampler {
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    pub fn reset(&mut self) {
        // Stateless; nothing to reset.
    }

    pub fn sample(&mut self, inp: &SamplerInputs<'_>, out: &mut [SamplerNode]) -> SampleResult {
        debug_assert!(!out.is_empty(), "TimeSampler::sample: empty output buffer");
        debug_assert!(
            inp.horizon_dt > 0.0,
            "TimeSampler::sample: horizon_dt must be positive"
        );

        let tau0 = inp.tau0_s;
        let end = inp.total_duration_s;
        for (k, node) in out.iter_mut().enumerate() {
            let t_k = (tau0 + k as f32 * inp.horizon_dt).min(end);
            let past_end = t_k >= end;
            let (pos, vel, acc, jerk) = if past_end {
                (
                    inp.traj.get_pos(end),
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Vec3::zeros(),
                )
            } else {
                (
                    inp.traj.get_pos(t_k),
                    inp.traj.get_vel(t_k),
                    inp.traj.get_acc(t_k),
                    inp.traj.get_jerk(t_k),
                )
            };
            *node = SamplerNode {
                pos,
                vel,
                acc,
                jerk,
                past_end,
            };
        }

        SampleResult {
            tau0_s: tau0,
            mission_done: tau0 >= end,
        }
    }
}
