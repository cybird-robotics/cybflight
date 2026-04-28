//! Behaviour tests for `PositionSampler`. Drives the sampler against a
//! synthetic straight-line trajectory (so the closest-point τ is exactly
//! computable) and asserts:
//!   1. The forward search converges to the predicted τ.
//!   2. `prev_query_tau` is monotonically non-decreasing across calls.
//!   3. `reset()` clears `prev_query_tau`.
//!   4. `mission_done` flips by the time-index when τ reaches the end.
//!   5. `mission_done` flips by `radius_of_acceptance` when the state is
//!      already inside the terminal-pose ball.
//!   6. The switch-tolerance keeps the search from oscillating against
//!      a tied cost.

use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::polynomial::Polynomial;
use cybflight_core::trajectory_planning::sampler::{
    PositionSampler, PositionSamplerParams, Sampler, SamplerInputs, SamplerNode,
};
use cybflight_core::trajectory_planning::types::Vec3;

const HORIZON_N1: usize = 11;
const HORIZON_DT: f32 = 0.05;

/// 2 m straight line along +X over 2 s — `traj.get_pos(τ) = (τ, 0, 0)`.
/// Trivial closest-point: for state `(s, 0, 0)`, the optimum τ is `s`.
fn line_trajectory() -> (PiecewisePolynomial, f32) {
    let pieces = [Polynomial::new(
        1,
        2.0,
        &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
    )];
    let pp = PiecewisePolynomial::from_pieces(&pieces);
    let total = pp.total_duration();
    (pp, total)
}

fn default_params() -> PositionSamplerParams {
    PositionSamplerParams {
        axis_weights_sqrt: Vec3::new(1.0, 1.0, 1.0),
        search_dt: 0.01,
        // Tight slack keeps the line-trajectory tests deterministic; the
        // weighted distance changes by 0.01 per step so a 1e-5 tolerance
        // does not let the search overshoot the predicted τ.
        search_tol: 1e-5,
        max_search_steps: 500,
        radius_of_acceptance: 0.05,
    }
}

#[test]
fn closest_point_converges_on_line() {
    let (traj, total) = line_trajectory();
    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // State at (1.3, 0, 0) — closest τ on the line is 1.3.
    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(1.3, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let result = s.sample(&inputs, &mut buf);

    // search_dt = 0.01, so the sampler should land within one step of 1.3.
    assert!(
        (result.tau0_s - 1.3).abs() < 0.02,
        "tau0_s {} not close to expected 1.3",
        result.tau0_s
    );
    // Node 0 sits at the converged τ — its position should match.
    assert!((buf[0].pos[0] - result.tau0_s).abs() < 1e-6);
    assert!(!result.mission_done);
}

#[test]
fn prev_query_tau_is_monotone_across_calls() {
    let (traj, total) = line_trajectory();
    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // Sweep state forward in 0.1 m increments along the line. Each call
    // should produce a τ that is greater than or equal to the previous —
    // forward search never rewinds.
    let mut prev = -1.0;
    for k in 0..15 {
        let s_x = 0.1 * k as f32;
        let inputs = SamplerInputs {
            traj: &traj,
            total_duration_s: total,
            tau0_s: 0.0,
            state_pos: Vec3::new(s_x, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        };
        let r = s.sample(&inputs, &mut buf);
        assert!(
            r.tau0_s >= prev - 1e-6,
            "step {k}: tau {} regressed from {}",
            r.tau0_s,
            prev
        );
        prev = r.tau0_s;
    }
}

#[test]
fn reset_clears_prev_query_tau() {
    let (traj, total) = line_trajectory();
    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // Drive τ forward.
    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(1.5, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    s.sample(&inputs, &mut buf);
    assert!(s.prev_query_tau().unwrap_or(0.0) > 1.0);

    s.reset();
    assert!(s.prev_query_tau().is_none());

    // After reset, the next call with state at the START of the line
    // should resolve τ ≈ 0 — proving the sampler restarted from 0
    // rather than inheriting the previous τ.
    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(0.05, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r = s.sample(&inputs, &mut buf);
    assert!(r.tau0_s < 0.1, "after reset, tau {} not near 0", r.tau0_s);
}

#[test]
fn mission_done_by_time_index() {
    let (traj, total) = line_trajectory();
    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // State far past the trajectory's end in X (3 m, line ends at 2 m) —
    // pure position search will walk τ up to total_duration_s and stop.
    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: total,
        state_pos: Vec3::new(3.0, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r = s.sample(&inputs, &mut buf);
    // Default search_dt = 0.01, so the mission_done_by_time threshold
    // is `end - 0.01 = 1.99`. The search should walk τ to within one
    // search step of `end`.
    assert!(
        r.tau0_s >= total - 0.011,
        "tau {} did not reach end {}",
        r.tau0_s,
        total
    );
    assert!(r.mission_done, "mission_done should fire at end-of-time");
    // `past_end` is a strict `>=` check on the per-node t_k; with
    // tau_curr at f32-accuracy below `end` it stays false. That's the
    // correct semantics — the polynomial is still evaluable at t_k.
    // We don't assert it here.
}

#[test]
fn mission_done_by_radius_of_acceptance() {
    let (traj, total) = line_trajectory();
    // Wide acceptance radius so a state near (but not past) the endpoint
    // triggers `mission_done` even though τ is short of total.
    let mut params = default_params();
    params.radius_of_acceptance = 0.5;
    // Disable any forward search past τ_curr = 0 by slamming the
    // tau-anchor pull on. Easier: just pick a short-ish state position
    // and verify the radius branch fires.
    let mut s = PositionSampler::new(params);
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // State at (1.7, 0, 0) — within 0.5 m of the endpoint (2.0, 0, 0)
    // but the closest τ on the line is 1.7, well short of the end.
    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(1.7, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r = s.sample(&inputs, &mut buf);
    assert!(
        r.tau0_s < total - 0.1,
        "tau should not reach end (got {})",
        r.tau0_s
    );
    assert!(
        r.mission_done,
        "mission_done should fire by radius_of_acceptance"
    );
}

#[test]
fn no_oscillation_at_stationary_state() {
    // With state held fixed and no traffic on the input, repeated calls
    // must produce strictly non-decreasing τ. The relative-tolerance
    // gate inside `can_advance` is what guarantees this in the presence
    // of cost ties.
    let (traj, total) = line_trajectory();
    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(1.0, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };

    let mut prev = -1.0;
    for _ in 0..50 {
        let r = s.sample(&inputs, &mut buf);
        assert!(
            r.tau0_s >= prev - 1e-6,
            "τ regressed: {} → {}",
            prev,
            r.tau0_s
        );
        prev = r.tau0_s;
    }
    // After convergence τ should pin near 1.0 (the line's closest point
    // to the state) and stop drifting forward.
    assert!(
        (prev - 1.0).abs() < 0.03,
        "τ {} drifted away from expected 1.0",
        prev
    );
}

#[test]
fn dispatched_sampler_position_variant_works() {
    // The Sampler enum's Position variant must dispatch correctly and
    // reset() must clear PositionSampler's state.
    let (traj, total) = line_trajectory();
    let mut s = Sampler::Position(PositionSampler::new(default_params()));
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(1.2, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r1 = s.sample(&inputs, &mut buf);
    s.reset();
    // After reset, the same call from state (0.05, 0, 0) must converge
    // back near τ = 0 — proving reset cleared prev_query_tau.
    let inputs2 = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(0.05, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r2 = s.sample(&inputs2, &mut buf);
    assert!(r1.tau0_s > 1.0);
    assert!(r2.tau0_s < 0.1, "after reset, tau {} should be near 0", r2.tau0_s);
}

#[test]
fn non_finite_state_pos_does_not_advance_tau() {
    // A NaN in `state_pos` poisons every cost; pre-guard the sampler
    // froze τ forever AND `mission_done` never fired (NaN ≤ r² is false).
    // Post-guard: τ unchanged, mission_done=false, output buffer filled
    // deterministically so callers never see uninitialised data.
    let (traj, total) = line_trajectory();
    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // First, drive τ forward to a known non-zero value with valid input.
    let inputs_ok = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(1.2, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r_ok = s.sample(&inputs_ok, &mut buf);
    let tau_before = s.prev_query_tau().unwrap_or(0.0);
    assert!(tau_before > 1.0);
    assert!(!r_ok.mission_done);

    // Now feed a NaN state_pos. τ must NOT advance, mission_done must
    // be false, and the output buffer must contain only finite values.
    let inputs_bad = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(f32::NAN, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r_bad = s.sample(&inputs_bad, &mut buf);
    assert_eq!(s.prev_query_tau().unwrap_or(0.0), tau_before, "τ advanced on NaN input");
    assert!(!r_bad.mission_done);
    assert_eq!(r_bad.tau0_s, tau_before);
    for node in &buf {
        assert!(node.pos.x.is_finite() && node.pos.y.is_finite() && node.pos.z.is_finite());
        assert!(node.vel.x.is_finite() && node.vel.y.is_finite() && node.vel.z.is_finite());
        assert!(node.acc.x.is_finite() && node.acc.y.is_finite() && node.acc.z.is_finite());
    }

    // After the bad input clears, the sampler must resume normal
    // operation — the guard is a *skip*, not a permanent latch.
    let r_recovered = s.sample(&inputs_ok, &mut buf);
    assert!(s.prev_query_tau().unwrap_or(0.0) >= tau_before);
    assert!(!r_recovered.mission_done);
}

#[test]
fn forward_only_on_curve_revisit() {
    // Trajectory revisits the SAME spatial point at two different τs:
    //   piece 0: x = τ over [0, 1]            → end at (1, 0, 0)
    //   piece 1: x = 1 − (τ − 1) over [1, 2]  → returns to (0, 0, 0)
    // The drone has been tracking forward and has already passed τ ≈ 1.0
    // (apex). When the controller sits at (0.5, 0, 0), there are two
    // candidate τs at the same spatial distance: τ=0.5 (earlier) and
    // τ=1.5 (later). Pure-position search starting from τ_prev > 1.0 must
    // monotonically pick the later root and never jump back to the
    // earlier one. This is the regression case for the curvy-path
    // backward-jump bug from the time-anchored variant.
    let pieces = [
        Polynomial::new(
            1,
            1.0,
            &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
        ),
        Polynomial::new(
            1,
            1.0,
            &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)],
        ),
    ];
    let pp = PiecewisePolynomial::from_pieces(&pieces);
    let total = pp.total_duration();

    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // Drive τ forward across the apex by feeding state positions that
    // walk along the trajectory: 0.0 → 0.7 → 1.0 (apex) → 0.7 (after
    // apex on piece 1) → 0.5 (on piece 1, mirror of the τ=0.5 candidate).
    for x in [0.0_f32, 0.7, 1.0, 0.9, 0.7, 0.5] {
        let inputs = SamplerInputs {
            traj: &pp,
            total_duration_s: total,
            tau0_s: 0.0, // ignored by the sampler, kept for the input contract
            state_pos: Vec3::new(x, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        };
        s.sample(&inputs, &mut buf);
    }
    let tau = s.prev_query_tau().unwrap_or(0.0);
    assert!(
        tau > 1.0,
        "τ {} should remain on the post-apex (later) root, not jump back to τ≈0.5",
        tau
    );
    assert!((tau - 1.5).abs() < 0.05, "τ {} not near 1.5", tau);
}
