//! Position-based reference sampler — forward-constrained polynomial
//! minimizer.
//!
//! The sampler is solving a 1-D minimization problem:
//!
//! ```text
//!   minimize   g(τ) = ‖axis_w ⊙ (p(τ) − s)‖²
//!   subject to prev_τ ≤ τ ≤ min(prev_τ + lookahead, end)
//! ```
//!
//! where `p(τ)` is the piecewise polynomial trajectory, `s` is the
//! current state position, and `axis_w` are the per-axis weights. Because
//! `p` is polynomial, `g` is polynomial of degree `2·deg` and `g'` is
//! polynomial of degree `2·deg − 1`; the minimum on each piece is a
//! closed-form critical point of `g'` or a piece/window boundary.
//!
//! ## Algorithm
//!
//! Per call, three steps are run inside the forward window
//! `[prev_τ, prev_τ + lookahead]`:
//!
//! 1. **Coarse grid running-min on `g`.** Sample `g(τ)` at a fixed grid
//!    spacing `search_dt`, tracking the τ with the smallest `g`. This
//!    bracket localises the basin to within `±search_dt`. The earlier
//!    "break on first rise" heuristic is dropped — on sharp curves a
//!    transient rise (drone tracking lag against a high-curvature
//!    segment) does not abort the search.
//! 2. **Bounded Newton refinement on `g'/g''`.** Two or three Newton
//!    iterations from the grid argmin recover the analytic critical
//!    point of `g` to f32 precision. Each iteration steps by
//!    `−g'(τ)/g''(τ)`, clamped to `±search_dt` so a poorly-conditioned
//!    quadratic approximation cannot eject Newton out of its basin.
//!    Newton uses the polynomial trajectory's `get_pos`/`get_vel`/
//!    `get_acc` directly — Horner is already O(deg), so explicit
//!    coefficient expansion buys nothing.
//! 3. **Time floor.** When the geometric optimum sits *behind* the drone
//!    (drone has overshot a sharp turn; the closest point on the curve
//!    really is the corner), no purely geometric criterion advances τ —
//!    the controller would be commanded backward toward the corner.
//!    `tau0_s` (wall-clock elapsed since trajectory start, supplied by
//!    the outer loop) provides a forward floor and a forward ceiling:
//!
//! ```text
//!   τ_floor = clamp(tau0_s − max_lag_s,  prev_τ,  end)
//!   τ_ceil  = clamp(tau0_s + max_lead_s, prev_τ,  end)
//!   τ_curr  = clamp( τ_geom, τ_floor, τ_ceil )
//! ```
//!
//! The clamps are one-sided / monotone: the floor only nudges τ forward
//! when geometry stalls, and the ceiling only clips τ when it would race
//! ahead of wall-clock progress. Together they form a "trust window"
//! `[tau0_s − max_lag_s, tau0_s + max_lead_s]` around real time inside
//! which the geometric optimum is trusted.
//!
//! The ceiling matters when the trajectory passes spatially close to
//! itself (loops, splits, return-to-home). The geometric cost has
//! multiple basins inside the lookahead window in that case, and Newton
//! can lock onto a basin that is far ahead in `τ` but spatially closer
//! to the drone — committing the controller to a "phantom" setpoint
//! that skips over a whole segment of trajectory. Position alone cannot
//! disambiguate; only progress (time) can.
//!
//! This is distinct from the rejected time-weight cost
//! (`(1 + tw·|τ − τ_anchor|)`), which biased the minimum itself and so
//! could refuse to advance past `τ_anchor` — a clamp is monotone.
//!
//! ## Invariants
//!
//! - **Forward-only.** The search window starts at `prev_τ`, so no step
//!   in the algorithm can produce τ < `prev_τ`. Within a mission `τ` is
//!   monotone non-decreasing; only `reset()` (called by the outer loop
//!   on Idle→Executing) ever rewinds it.
//! - **Bounded compute.** Worst-case per-tick cost is
//!   `max_search_steps + 3` polynomial evaluations.

use super::super::piecewise_polynomial::PiecewisePolynomial;
use super::super::types::Vec3;
use super::{SampleResult, SamplerInputs, SamplerNode};

#[derive(Clone, Copy, Debug)]
pub struct PositionSamplerParams {
    /// Per-axis sqrt-weights on the position-error vector inside the
    /// distance term. `Vec3::new(1, 1, 1)` weights all axes equally;
    /// lowering Z effectively says "match XY tightly, Z loosely".
    pub axis_weights_sqrt: Vec3,
    /// Grid spacing for the running-min bracket and the cap on Newton's
    /// step size. Smaller = finer initial bracket but more polynomial
    /// evaluations per tick (capped by `max_search_steps`).
    pub search_dt: f32,
    /// Hard cap on per-tick `traj.get_pos` calls during the grid scan.
    /// Together with `search_dt`, this defines the lookahead window:
    /// `lookahead = search_dt · max_search_steps`. With `search_dt = 10 ms`
    /// and `max_search_steps = 100` the search sweeps at most 1.0 s of
    /// trajectory per tick.
    pub max_search_steps: u16,
    /// Position tolerance for declaring the mission done. When
    /// `‖state_pos − traj.get_pos(end)‖ < radius_of_acceptance` AND the
    /// sampler is in the trajectory's terminal phase
    /// (`τ_curr ≥ end − max_lag_s`), `mission_done = true` is reported
    /// even if `τ_curr` has not yet reached the terminal time. The
    /// terminal-phase gate prevents a trajectory whose route passes
    /// within `radius_of_acceptance` of its own endpoint from triggering
    /// a false-positive `mission_done` at mid-flight nearest pass.
    pub radius_of_acceptance: f32,
    /// Maximum tolerated lag of `τ_curr` behind `tau0_s` (the wall-clock
    /// trajectory time supplied by the outer loop), in seconds. The
    /// time floor `τ ≥ tau0_s − max_lag_s` activates only when geometric
    /// minimization stalls with τ behind real time — overshoot at a
    /// sharp turn, where the closest-point on the curve is permanently
    /// behind the drone. A non-finite or non-positive value disables
    /// the floor and reverts to pure geometric minimization.
    pub max_lag_s: f32,
    /// Maximum tolerated lead of `τ_curr` ahead of `tau0_s`, in seconds.
    /// The time ceiling `τ ≤ tau0_s + max_lead_s` clips runaway matches
    /// that could otherwise occur when the trajectory passes spatially
    /// close to itself (loops, splits, return-to-home): the geometric
    /// cost has multiple basins inside the lookahead window and Newton
    /// can lock onto a far-ahead basin that is spatially closer to the
    /// drone. The ceiling restricts the search to a "trust window"
    /// around wall-clock progress so disambiguation falls back to
    /// trajectory time. A non-finite value disables the ceiling.
    pub max_lead_s: f32,
}

impl PositionSamplerParams {
    pub const fn defaults() -> Self {
        Self {
            axis_weights_sqrt: Vec3::new(1.0, 1.0, 1.0),
            search_dt: 0.01,
            max_search_steps: 100,
            radius_of_acceptance: 0.15,
            max_lag_s: 0.1,
            max_lead_s: 0.1,
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
                    jerk: Vec3::zeros(),
                    past_end: false,
                };
            }
            return SampleResult {
                tau0_s: tau_safe,
                mission_done: false,
            };
        }

        let end = inp.total_duration_s;
        let w_sqrt = self.params.axis_weights_sqrt;
        let dt = self.params.search_dt;
        let max_steps = self.params.max_search_steps as usize;

        let prev_tau = self.prev_query_tau.unwrap_or(0.0).clamp(0.0, end);
        // Forward ceiling derived from wall-clock progress. Disabled
        // when `max_lead_s` is non-finite (NaN/±Inf) — the search then
        // falls back to the pure `prev_tau + dt·max_steps` lookahead.
        let tau_lead_raw = inp.tau0_s + self.params.max_lead_s;
        let tau_lead = if tau_lead_raw.is_finite() {
            tau_lead_raw.clamp(prev_tau, end)
        } else {
            end
        };
        let tau_hi = (prev_tau + dt * max_steps as f32).min(end).min(tau_lead);

        // Step 1: coarse grid running-min on g(τ) over [prev_tau, tau_hi].
        // No early break on rise — sharp curves can produce a transient
        // dist increase before a deeper basin within the window.
        let mut tau_best = prev_tau;
        let mut g_best = weighted_dist2(inp.traj.get_pos(prev_tau), inp.state_pos, w_sqrt);
        let mut tau = prev_tau;
        for _ in 0..max_steps {
            tau += dt;
            if tau > tau_hi {
                tau = tau_hi;
            }
            let g = weighted_dist2(inp.traj.get_pos(tau), inp.state_pos, w_sqrt);
            if g < g_best {
                g_best = g;
                tau_best = tau;
            }
            if tau >= tau_hi {
                break;
            }
        }

        // Step 2: bounded Newton refinement of `tau_best` on `g'/g''`.
        // Step is clamped to ±search_dt so a noisy quadratic
        // approximation cannot eject Newton from its basin. Two-three
        // iterations are enough for f32 precision on a smooth polynomial.
        let mut tau_geom = tau_best;
        for _ in 0..3 {
            let (gp_half, gpp_half) =
                weighted_grad_half(inp.traj, tau_geom, inp.state_pos, w_sqrt);
            if gpp_half <= 1e-12 {
                break;
            }
            let raw_delta = -gp_half / gpp_half;
            let delta = raw_delta.clamp(-dt, dt);
            let tau_new = (tau_geom + delta).clamp(prev_tau, tau_hi);
            if (tau_new - tau_geom).abs() < 1e-7 {
                tau_geom = tau_new;
                break;
            }
            tau_geom = tau_new;
        }

        // Step 3: time floor and ceiling (one-sided clamps in `tau0_s`
        // space). Newton already clamped `tau_geom` to `[prev_tau, tau_hi]`
        // and `tau_hi ≤ tau_lead`, so the explicit `.min(tau_lead)` here
        // is defensive — it matters only if `tau_floor > tau_lead`, which
        // a sane (max_lag_s, max_lead_s) configuration cannot produce
        // but a misconfiguration could.
        let tau_floor_raw = inp.tau0_s - self.params.max_lag_s;
        let tau_floor = if tau_floor_raw.is_finite() {
            tau_floor_raw.clamp(prev_tau, end)
        } else {
            prev_tau
        };
        let tau_curr = tau_geom.max(tau_floor).min(tau_lead);

        self.prev_query_tau = Some(tau_curr);

        // Horizon fill — same kernel as TimeSampler, anchored at tau_curr.
        for (k, node) in out.iter_mut().enumerate() {
            let t_k = (tau_curr + k as f32 * inp.horizon_dt).min(end);
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
        // The radius shortcut is gated by τ being in the trajectory's
        // terminal phase (`τ_curr ≥ end − max_lag_s`). Without this gate,
        // any trajectory whose route passes within `radius_of_acceptance`
        // of its own endpoint (return-to-home loops, figure-8s with the
        // start near the goal, waypoint missions where one leg dips near
        // the destination) would trigger a false-positive `mission_done`
        // at the moment of nearest pass — terminating mid-flight. Reusing
        // `max_lag_s` couples the radius shortcut to the time floor's
        // grace period: the shortcut can only fire inside the same
        // window during which the time floor would itself force the
        // sampler to declare done.
        let in_terminal_phase = end - tau_curr <= self.params.max_lag_s;
        let mission_done_by_radius = !mission_done_by_time
            && in_terminal_phase
            && {
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

/// `g(τ) = Σᵢ wᵢ² (pᵢ(τ) − sᵢ)²`. Axis weights enter squared (the public
/// parameter is a sqrt-weight so a weight of 0.5 down-weights an axis
/// 4× in the distance, matching the physical "Z loosely" intuition).
#[inline]
fn weighted_dist2(traj_pos: Vec3, state_pos: Vec3, w_sqrt: Vec3) -> f32 {
    let dx = (traj_pos[0] - state_pos[0]) * w_sqrt[0];
    let dy = (traj_pos[1] - state_pos[1]) * w_sqrt[1];
    let dz = (traj_pos[2] - state_pos[2]) * w_sqrt[2];
    dx * dx + dy * dy + dz * dz
}

/// Returns `(g'(τ)/2, g''(τ)/2)` at `τ`. The factor of 2 cancels in
/// Newton's `−g'/g''` so it never needs to be reintroduced.
///
/// Identities:
///
/// ```text
///   g'(τ)/2  = Σᵢ wᵢ² (pᵢ − sᵢ) · pᵢ'
///   g''(τ)/2 = Σᵢ wᵢ² ( pᵢ'² + (pᵢ − sᵢ) · pᵢ'' )
/// ```
#[inline]
fn weighted_grad_half(
    traj: &PiecewisePolynomial,
    tau: f32,
    state_pos: Vec3,
    w_sqrt: Vec3,
) -> (f32, f32) {
    let p = traj.get_pos(tau);
    let v = traj.get_vel(tau);
    let a = traj.get_acc(tau);
    let wx2 = w_sqrt[0] * w_sqrt[0];
    let wy2 = w_sqrt[1] * w_sqrt[1];
    let wz2 = w_sqrt[2] * w_sqrt[2];
    let dx = p[0] - state_pos[0];
    let dy = p[1] - state_pos[1];
    let dz = p[2] - state_pos[2];
    let gp_half = wx2 * dx * v[0] + wy2 * dy * v[1] + wz2 * dz * v[2];
    let gpp_half = wx2 * (v[0] * v[0] + dx * a[0])
        + wy2 * (v[1] * v[1] + dy * a[1])
        + wz2 * (v[2] * v[2] + dz * a[2]);
    (gp_half, gpp_half)
}
