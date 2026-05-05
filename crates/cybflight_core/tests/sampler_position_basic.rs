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
        max_search_steps: 500,
        radius_of_acceptance: 0.05,
        // Time floor and ceiling both disabled by default in tests —
        // existing tests exercise pure geometric matching. Tests that
        // need the floor (`time_floor_unsticks_corner_overshoot`) or
        // the ceiling override these fields explicitly.
        max_lag_s: f32::INFINITY,
        max_lead_s: f32::INFINITY,
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

#[test]
fn principled_search_finds_global_min_past_local_max() {
    // 3-piece path where the closest-point distance to state=(0,0,0) is
    // non-monotone in τ:
    //   piece 0 (τ ∈ [0.0, 0.3]): x walks 1.0 → 0.5  (dist falls)
    //   piece 1 (τ ∈ [0.3, 0.5]): x walks 0.5 → 0.6  (dist rises)
    //   piece 2 (τ ∈ [0.5, 0.8]): x walks 0.6 → 0.0  (dist falls to 0)
    //
    // Local minima of dist² in τ:
    //   τ = 0.3 (end of piece 0, dist = 0.5)
    //   τ = 0.8 (end of piece 2, dist = 0)        ← global min
    //
    // The earlier break-on-rise heuristic stopped the moment piece 1's
    // slight rise was detected, leaving τ ≈ 0.3 — which on a real flight
    // would put the controller's setpoint 0.5 m behind the drone, i.e.
    // commanding it to fly *back*. This test pins the principled
    // forward-window minimizer: scan the whole window, take the smallest
    // dist², so τ converges to the global min at 0.8 even after a
    // transient rise.
    let pieces = [
        Polynomial::new(
            1,
            0.3,
            &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(-5.0 / 3.0, 0.0, 0.0)],
        ),
        Polynomial::new(
            1,
            0.2,
            &[Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)],
        ),
        Polynomial::new(
            1,
            0.3,
            &[Vec3::new(0.6, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)],
        ),
    ];
    let pp = PiecewisePolynomial::from_pieces(&pieces);
    let total = pp.total_duration();

    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];
    let inputs = SamplerInputs {
        traj: &pp,
        total_duration_s: total,
        tau0_s: 0.0,
        state_pos: Vec3::new(0.0, 0.0, 0.0),
        horizon_dt: HORIZON_DT,
    };
    let r = s.sample(&inputs, &mut buf);
    assert!(
        (r.tau0_s - 0.8).abs() < 0.011,
        "τ {} not at global min 0.8 — sampler stalled at the τ=0.3 local min",
        r.tau0_s
    );
}

#[test]
fn time_floor_unsticks_corner_overshoot() {
    // Trajectory with a sharp 90° corner: 1 m east, then 1 m north.
    //   piece 0 (τ ∈ [0, 1]): p = (τ,   0,   0)
    //   piece 1 (τ ∈ [1, 2]): p = (1,   τ−1, 0)
    //
    // Drone overshoots the corner: state at (1.1, 0, 0). The geometric
    // closest point on the trajectory is exactly the corner (τ = 1.0):
    // every τ > 1 has y > 0, which is *farther* from the drone (y = 0).
    // Pure geometric minimization correctly identifies τ = 1.0 — and a
    // controller tracking (1.0, 0, 0) would command the drone *backward*
    // from (1.1, 0, 0) toward the corner.
    //
    // The principled escape is the time floor: `tau0_s` (wall-clock
    // elapsed) advances independently of geometry; once the geometric
    // search lags by more than `max_lag_s`, the floor pulls τ forward,
    // and the controller's setpoint moves into piece 1 (north), so the
    // drone turns instead of flying backward.
    let pieces = [
        Polynomial::new(
            1,
            1.0,
            &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
        ),
        Polynomial::new(
            1,
            1.0,
            &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        ),
    ];
    let pp = PiecewisePolynomial::from_pieces(&pieces);
    let total = pp.total_duration();

    let mut params = default_params();
    params.max_lag_s = 0.3;
    let mut s = PositionSampler::new(params);
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // Walk the sampler to the corner first using on-trajectory states
    // (and tau0_s synchronised with the state's piece-0 progress) so τ
    // converges to 1.0 before the overshoot tick.
    for x in [0.0_f32, 0.5, 1.0] {
        let inputs = SamplerInputs {
            traj: &pp,
            total_duration_s: total,
            tau0_s: x,
            state_pos: Vec3::new(x, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        };
        s.sample(&inputs, &mut buf);
    }
    let tau_before = s.prev_query_tau().unwrap_or(0.0);
    assert!(
        (tau_before - 1.0).abs() < 0.02,
        "setup: τ should be at the corner (1.0), got {}",
        tau_before
    );

    // Overshoot tick: drone past the corner (x = 1.1) but still at y = 0.
    // Real time has continued: tau0_s = 1.5 → floor = 1.5 − 0.3 = 1.2.
    // Without the floor, geometric search would pin τ at 1.0 forever.
    let r = s.sample(
        &SamplerInputs {
            traj: &pp,
            total_duration_s: total,
            tau0_s: 1.5,
            state_pos: Vec3::new(1.1, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        },
        &mut buf,
    );
    assert!(
        r.tau0_s >= 1.2 - 0.02,
        "time floor failed to unstick: τ = {} (expected ≥ 1.2)",
        r.tau0_s
    );
    // Floor is one-sided — geometry doesn't pull τ past tau0_s.
    assert!(
        r.tau0_s <= 1.5 + 0.02,
        "τ = {} ran past tau0_s = 1.5 — floor should not push past wall-clock time",
        r.tau0_s
    );
}

#[test]
fn time_floor_disabled_when_max_lag_non_finite() {
    // With `max_lag_s = INF` the floor expression `tau0_s − max_lag_s`
    // is non-finite; the sampler must fall back to pure geometric
    // minimization. Reproduces the corner-overshoot setup above and
    // asserts τ stays pinned at the geometric minimum (the corner).
    let pieces = [
        Polynomial::new(
            1,
            1.0,
            &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
        ),
        Polynomial::new(
            1,
            1.0,
            &[Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        ),
    ];
    let pp = PiecewisePolynomial::from_pieces(&pieces);
    let total = pp.total_duration();

    // default_params() already disables the floor (max_lag_s = INF).
    let mut s = PositionSampler::new(default_params());
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];
    for x in [0.0_f32, 0.5, 1.0] {
        s.sample(
            &SamplerInputs {
                traj: &pp,
                total_duration_s: total,
                tau0_s: x,
                state_pos: Vec3::new(x, 0.0, 0.0),
                horizon_dt: HORIZON_DT,
            },
            &mut buf,
        );
    }
    let r = s.sample(
        &SamplerInputs {
            traj: &pp,
            total_duration_s: total,
            tau0_s: 1.5,
            state_pos: Vec3::new(1.1, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        },
        &mut buf,
    );
    assert!(
        (r.tau0_s - 1.0).abs() < 0.02,
        "with the floor disabled, τ should stay at the geometric corner (1.0); got {}",
        r.tau0_s
    );
}

#[test]
fn radius_of_acceptance_does_not_fire_on_mid_flight_near_pass() {
    // Out-and-back path A → B → A:
    //   piece 0 (τ ∈ [0, 1]): p = (τ,         0, 0) — out
    //   piece 1 (τ ∈ [1, 2]): p = (1 − (τ−1), 0, 0) — back
    // Endpoint = (0, 0, 0); start = (0, 0, 0). The trajectory passes
    // exactly through the endpoint at τ = 0 (start) and again at τ = 2
    // (end). Mid-flight near-passes happen near both ends.
    //
    // Without a terminal-phase gate, the radius shortcut fires on the
    // very first tick — drone at start, state inside the ball around
    // the endpoint — and the mission never runs. With the gate
    // (max_lag_s = 0.3), the shortcut is blocked outside the last
    // 0.3 s of trajectory time.
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

    let mut params = default_params();
    params.radius_of_acceptance = 0.15;
    params.max_lag_s = 0.3;
    let mut s = PositionSampler::new(params);
    let mut buf = vec![SamplerNode::default(); HORIZON_N1];

    // Tick 0: drone at start (= endpoint). State exactly inside the
    // ball. Without the gate, mission_done would fire immediately.
    let r0 = s.sample(
        &SamplerInputs {
            traj: &pp,
            total_duration_s: total,
            tau0_s: 0.0,
            state_pos: Vec3::new(0.0, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        },
        &mut buf,
    );
    assert!(
        !r0.mission_done,
        "mission_done fired on tick 0 — the radius shortcut is not gated by terminal phase"
    );

    // Mid-mission: drone happens to be near (0,0,0) again — could be
    // a real near-pass or just the start of the return leg. tau0_s is
    // well below `end − max_lag_s = 1.7`, so the gate must keep the
    // shortcut closed.
    let r_mid = s.sample(
        &SamplerInputs {
            traj: &pp,
            total_duration_s: total,
            tau0_s: 0.5,
            state_pos: Vec3::new(0.05, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        },
        &mut buf,
    );
    assert!(
        !r_mid.mission_done,
        "mission_done fired mid-flight (τ < end − max_lag_s) on a state inside the terminal ball"
    );

    // Terminal phase: tau0_s = 1.95, time floor pulls τ_curr to
    // ≥ end − max_lag_s = 1.7. State at (0.05, 0, 0) is well inside
    // the ball. The gate opens and the shortcut fires.
    let r_end = s.sample(
        &SamplerInputs {
            traj: &pp,
            total_duration_s: total,
            tau0_s: 1.95,
            state_pos: Vec3::new(0.05, 0.0, 0.0),
            horizon_dt: HORIZON_DT,
        },
        &mut buf,
    );
    assert!(
        r_end.mission_done,
        "mission_done failed to fire in terminal phase with state inside the ball — gate is too strict"
    );
}
