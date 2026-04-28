//! Pins `TimeSampler` to the exact behaviour of the inline reference-fill
//! loop that lived inside `crates/cybflight/src/control/outer_loop.rs`
//! (the `if mission_state == Executing` block at the time of extraction).
//!
//! The reference kernel below is a verbatim transcription of that block,
//! reduced to the per-node arithmetic — no embassy, no x_refs layout, no
//! quaternion construction (the sampler intentionally returns `acc` and
//! lets the outer loop assemble the quaternion from `yaw_setpoint_rad`).
//!
//! The test sweeps four representative `τ₀` regimes (the caller is
//! responsible for the `now − t_start` clamp; `τ₀ = 0` represents both
//! "trajectory at its start" and "future-dated start, controller idle"):
//!   1. τ₀ = 0                                          → idle / start
//!   2. mid-mission (τ₀ inside the trajectory)          → no clamping
//!   3. exactly at the end (τ₀ == total_duration)       → all-past_end
//!   4. past-end (τ₀ > total_duration)                  → all-past_end
//!
//! The kernel and the sampler must produce bitwise-identical
//! `(pos, vel, acc, past_end, tau0_s, mission_done)` for every scenario,
//! every node. Running with `--release` would not be enough — the test
//! enforces equality with `==` on f32 (no tolerance) so any floating
//! reordering would surface.

use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::polynomial::Polynomial;
use cybflight_core::trajectory_planning::sampler::{
    SampleResult, Sampler, SamplerInputs, SamplerNode, TimeSampler,
};
use cybflight_core::trajectory_planning::types::Vec3;

const HORIZON_N1: usize = 21; // matches MPC_N + 1 for the firmware's typical horizon
const HORIZON_DT: f32 = 0.05; // 20 Hz horizon spacing — typical MPC dt

/// Build a 3-piece, degree-3 polynomial trajectory with non-trivial,
/// distinct coefficients per piece so every `get_*` call produces a value
/// that can't be confused with constant or zero.
fn build_test_trajectory() -> (PiecewisePolynomial, f32) {
    let pieces = [
        Polynomial::new(
            3,
            0.7,
            &[
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, -0.5, 0.0),
                Vec3::new(0.2, 0.7, -0.3),
                Vec3::new(-0.1, 0.05, 0.04),
            ],
        ),
        Polynomial::new(
            3,
            1.1,
            &[
                Vec3::new(0.6, -0.3, 1.05),
                Vec3::new(0.7, 0.2, 0.1),
                Vec3::new(-0.4, 0.5, -0.2),
                Vec3::new(0.05, -0.07, 0.02),
            ],
        ),
        Polynomial::new(
            3,
            0.5,
            &[
                Vec3::new(1.2, 0.4, 1.2),
                Vec3::new(-0.2, 0.1, -0.05),
                Vec3::new(0.3, -0.2, 0.07),
                Vec3::new(-0.06, 0.04, -0.01),
            ],
        ),
    ];
    let pp = PiecewisePolynomial::from_pieces(&pieces);
    let total = pp.total_duration();
    (pp, total)
}

/// Verbatim transcription of the per-node fill that used to live inline
/// in `outer_loop.rs::control_loop_task` (the body of the
/// `MISSION_TRAJECTORY_SLOT.lock` closure during Executing). Returns the
/// (pos, vel, acc, past_end) for one node plus the (tau0, mission_done)
/// derived from the same inputs.
///
/// Keeping this function in the test (rather than parameterising over the
/// sampler) is the whole point of the equivalence test — it's the
/// independent oracle the new code must reproduce.
fn reference_fill(
    traj: &PiecewisePolynomial,
    total_duration_s: f32,
    tau0_s: f32,
    horizon_dt: f32,
    nodes: usize,
) -> (Vec<SamplerNode>, f32, bool) {
    let tau0 = tau0_s;
    let mut out = Vec::with_capacity(nodes);
    for k in 0..nodes {
        let t_k = (tau0 + k as f32 * horizon_dt).min(total_duration_s);
        let past_end = t_k >= total_duration_s;
        let (p, v, a) = if past_end {
            (
                traj.get_pos(total_duration_s),
                Vec3::zeros(),
                Vec3::zeros(),
            )
        } else {
            (traj.get_pos(t_k), traj.get_vel(t_k), traj.get_acc(t_k))
        };
        out.push(SamplerNode {
            pos: p,
            vel: v,
            acc: a,
            past_end,
        });
    }
    let mission_done = tau0 >= total_duration_s;
    (out, tau0, mission_done)
}

fn run_case(label: &str, tau0_s: f32) {
    let (traj, total) = build_test_trajectory();
    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s,
        // TimeSampler ignores state_pos; pass something non-zero anyway
        // so that a future bug accidentally reading it would be loud.
        state_pos: Vec3::new(7.0, -3.0, 1.5),
        horizon_dt: HORIZON_DT,
    };

    let (ref_nodes, ref_tau0, ref_done) =
        reference_fill(&traj, total, tau0_s, HORIZON_DT, HORIZON_N1);

    // Drive both the bare TimeSampler and the dispatching enum: they must
    // agree, since the enum is just a match wrapper.
    let mut bare = TimeSampler::new();
    let mut buf_bare = vec![SamplerNode::default(); HORIZON_N1];
    let res_bare = bare.sample(&inputs, &mut buf_bare);

    let mut dispatched = Sampler::Time(TimeSampler::new());
    let mut buf_disp = vec![SamplerNode::default(); HORIZON_N1];
    let res_disp = dispatched.sample(&inputs, &mut buf_disp);

    // Equality on f32 with no tolerance: any reordering of the arithmetic
    // would surface here. The kernel above and TimeSampler::sample do the
    // same operations in the same order, so this must hold.
    for k in 0..HORIZON_N1 {
        assert_eq!(
            buf_bare[k], ref_nodes[k],
            "case {label}: node {k} (bare) diverged from reference kernel"
        );
        assert_eq!(
            buf_disp[k], ref_nodes[k],
            "case {label}: node {k} (dispatched) diverged from reference kernel"
        );
    }
    assert_eq!(
        res_bare,
        SampleResult {
            tau0_s: ref_tau0,
            mission_done: ref_done,
        },
        "case {label}: bare SampleResult diverged"
    );
    assert_eq!(
        res_disp, res_bare,
        "case {label}: dispatched SampleResult diverged from bare"
    );
}

#[test]
fn case_idle_or_future_start() {
    // τ₀ = 0 represents either "trajectory at its very start" or
    // "future-dated start, controller waiting". Every node samples
    // [0, k·dt] and (assuming the horizon fits inside the trajectory)
    // none should be past-end.
    run_case("idle_or_future_start", 0.0);
}

#[test]
fn case_mid_mission() {
    // Trajectory total ≈ 2.3 s; pick τ₀ ≈ 0.4 s into a 3-piece poly so
    // node 0 lands inside piece 0 and the horizon spans into piece 1.
    let (_, total) = build_test_trajectory();
    assert!(total > 1.5, "test trajectory shorter than expected: {total}");
    run_case("mid_mission", 0.4);
}

#[test]
fn case_exactly_at_end() {
    let (_, total) = build_test_trajectory();
    // tau0 == end → every node past_end, mission_done=true, all
    // (pos, vel, acc) take the past-end branch.
    run_case("exactly_at_end", total);
}

#[test]
fn case_past_end() {
    let (_, total) = build_test_trajectory();
    // tau0 well past the end. Same expected behaviour as exactly_at_end.
    run_case("past_end", total + 5.0);
}

#[test]
fn dispatched_reset_is_noop_for_time_sampler() {
    // The Sampler enum's reset() must not panic and must not change the
    // sampler's behaviour on the next call (TimeSampler is stateless).
    let (traj, total) = build_test_trajectory();
    let inputs = SamplerInputs {
        traj: &traj,
        total_duration_s: total,
        tau0_s: 0.6,
        state_pos: Vec3::zeros(),
        horizon_dt: HORIZON_DT,
    };

    let mut s = Sampler::Time(TimeSampler::new());
    let mut buf_a = vec![SamplerNode::default(); HORIZON_N1];
    let r_a = s.sample(&inputs, &mut buf_a);
    s.reset();
    let mut buf_b = vec![SamplerNode::default(); HORIZON_N1];
    let r_b = s.sample(&inputs, &mut buf_b);
    assert_eq!(buf_a, buf_b);
    assert_eq!(r_a, r_b);
}
