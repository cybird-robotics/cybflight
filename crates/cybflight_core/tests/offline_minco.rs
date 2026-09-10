//! Integration test for the offline-precomputed MINCO trajectory pipeline
//! used by `cybflight::control::mission_planner::plan_offline`.
//!
//! Mirrors the firmware's offline-planning workflow on the host:
//!   1. Recover per-segment durations from the absolute-time YAML schedule.
//!   2. Feed the n−1 intermediate waypoints + tail = wp[n−1] into a
//!      `MincoSnap` solver of size n with zero-PVAJ boundaries — the same
//!      solver + boundary conditions `plan_offline` uses (`OFFLINE_MINCO`).
//!      Min-jerk (s=3) is NOT equivalent here: it leaves the head/tail
//!      jerk free, and on dense schedules the solution front-loads jerk
//!      at t=0, spiking the flatness body rate ∝ ‖j‖/‖α‖ at the very
//!      first sample.
//!   3. Sample the resulting trajectory at 10 ms and compute peak collective
//!      thrust + peak body rates via `flatness_to_thrust_omega` — the same
//!      pole-safe map the outer loop uses for the `u_refs` feedforward.
//!   4. Assert the peaks stay within the planner's published limits
//!      (small overshoots tolerated; gross violations fail the test).
//!
//! Two fixtures are exercised — `trajectory_slow.yaml` and
//! `trajectory_time_optimal.yaml` — with one test per fixture. The
//! continuity / waypoint-hit check is also done per fixture so the
//! MINCO solver itself is verified end-to-end on each schedule.
//!
//! Run: `cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --test offline_minco`

use cybflight_core::trajectory_planning::flatness::flatness_to_thrust_omega;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};

// ─── physical constants matching the firmware's planner config ─────────
//
// Mass and gravity match `test_vehicle_params()` in `planner_convergence.rs`
// (mass = 0.55 kg, g = 9.81 m/s²) so the thrust check is consistent with
// what the live firmware will see.

const MASS_KG: f32 = 0.55;
const GRAVITY_M_S2: f32 = 9.8066;

// User-specified envelope for the offline race trajectory.
const MAX_THRUST_N: f32 = 27.2;
const MAX_BODY_RATE_XY_RAD_S: f32 = 10.0;
const MAX_BODY_RATE_Z_RAD_S: f32 = 6.0;

/// Sampling period for the constraint sweep [s].
const SAMPLE_DT_S: f32 = 0.010;

/// Slack tolerance: "small violation allowed, large is not". 30 % is the
/// upper bound on what we'll accept — the offline planner is supposed to
/// have produced a trajectory that the on-board controller can track, so
/// gross overshoots indicate a broken pipeline (bad timestamps, MINCO
/// solver bug, or unit mismatch) rather than honest tracking slop.
const CONSTRAINT_SLACK: f32 = 1.30;

/// Expected continuity tolerance at internal piece boundaries.
/// MINCO-snap degree-7 polynomials are C⁴ by construction (continuity
/// through snap across boundaries); we double-check pos/vel/acc
/// empirically.
const CONTINUITY_TOL_POS: f32 = 1e-3;
const CONTINUITY_TOL_VEL: f32 = 1e-2;
const CONTINUITY_TOL_ACC: f32 = 1e-1;

/// One YAML schedule (waypoints + absolute-time stamps + recorded start).
struct Fixture {
    name: &'static str,
    start_pos: [f32; 3],
    waypoints: &'static [[f32; 3]],
    timestamps: &'static [f32],
}

// ─── trajectory_slow.yaml ──────────────────────────────────────────────

const SLOW_START_POS: [f32; 3] = [-2.5, -3.5, 1.0];

static SLOW_WAYPOINTS: [[f32; 3]; 60] = [
    [-2.31, -3.483, 1.032],
    [-1.187, -3.222, 1.269],
    [-0.3347, -2.231, 1.6],
    [-0.6571, -1.146, 1.657],
    [-1.37, 0.2179, 1.514],
    [-1.837, 1.643, 1.199],
    [-1.14, 3.186, 0.7041],
    [0.7796, 3.265, 0.6496],
    [2.285, 1.635, 1.203],
    [2.585, 0.2585, 1.59],
    [2.604, -1.092, 1.824],
    [2.553, -2.107, 1.795],
    [2.555, -2.529, 1.581],
    [2.581, -2.515, 1.281],
    [2.543, -2.106, 1.007],
    [2.22, -1.198, 0.8225],
    [1.458, -0.2428, 0.8563],
    [0.3081, 0.3508, 0.9944],
    [-1.327, 0.2032, 1.102],
    [-2.4, -0.8644, 1.054],
    [-2.39, -2.211, 1.003],
    [-1.53, -3.067, 1.113],
    [-0.5896, -3.087, 1.369],
    [-0.3345, -2.232, 1.599],
    [-0.7456, -1.084, 1.634],
    [-1.399, 0.2807, 1.49],
    [-1.837, 1.643, 1.2],
    [-1.213, 3.315, 0.6677],
    [0.7386, 3.371, 0.6137],
    [2.285, 1.635, 1.203],
    [2.581, 0.246, 1.592],
    [2.602, -1.1, 1.825],
    [2.553, -2.107, 1.795],
    [2.554, -2.529, 1.581],
    [2.58, -2.515, 1.281],
    [2.543, -2.106, 1.007],
    [2.221, -1.197, 0.822],
    [1.459, -0.2409, 0.856],
    [0.3082, 0.3508, 0.9944],
    [-1.327, 0.1998, 1.102],
    [-2.4, -0.8675, 1.053],
    [-2.39, -2.211, 1.003],
    [-1.532, -3.063, 1.114],
    [-0.5915, -3.083, 1.369],
    [-0.3345, -2.232, 1.599],
    [-0.7479, -1.081, 1.635],
    [-1.405, 0.2845, 1.49],
    [-1.837, 1.643, 1.2],
    [-1.18, 3.299, 0.6677],
    [0.7865, 3.342, 0.6125],
    [2.285, 1.635, 1.202],
    [2.56, 0.1874, 1.626],
    [2.566, -1.17, 1.866],
    [2.554, -2.107, 1.797],
    [2.588, -2.404, 1.577],
    [2.61, -2.389, 1.285],
    [2.54, -2.108, 1.003],
    [1.593, -0.7033, 0.7181],
    [0.4993, 0.2339, 0.9388],
    [0.3099, 0.3554, 1.0],
];

static SLOW_TIMESTAMPS: [f32; 60] = [
    0.7289, 1.458, 2.188, 2.633, 3.079, 3.524, 4.176, 4.828, 5.48, 5.848, 6.216, 6.584, 6.888,
    7.193, 7.498, 7.893, 8.289, 8.684, 9.197, 9.709, 10.22, 10.71, 11.2, 11.69, 12.08, 12.47,
    12.86, 13.52, 14.19, 14.85, 15.22, 15.58, 15.94, 16.25, 16.55, 16.86, 17.25, 17.65, 18.04,
    18.56, 19.07, 19.58, 20.07, 20.56, 21.05, 21.44, 21.84, 22.23, 22.9, 23.57, 24.24, 24.63,
    25.02, 25.42, 25.7, 25.98, 26.25, 27.02, 27.78, 28.55,
];

const SLOW_FIXTURE: Fixture = Fixture {
    name: "trajectory_slow.yaml",
    start_pos: SLOW_START_POS,
    waypoints: &SLOW_WAYPOINTS,
    timestamps: &SLOW_TIMESTAMPS,
};

// ─── trajectory_time_optimal.yaml ──────────────────────────────────────

const TO_START_POS: [f32; 3] = [-2.5, -3.5, 1.0];

static TO_WAYPOINTS: [[f32; 3]; 60] = [
    [-2.426, -3.525, 1.064],
    [-1.528, -3.399, 1.301],
    [-0.3353, -2.231, 1.6],
    [-0.321, -0.9448, 1.347],
    [-1.1, 0.4391, 1.051],
    [-1.837, 1.643, 1.199],
    [-1.178, 3.047, 1.336],
    [0.6448, 3.121, 1.188],
    [2.284, 1.635, 1.203],
    [2.759, 0.1793, 1.328],
    [2.736, -1.214, 1.601],
    [2.554, -2.107, 1.795],
    [2.504, -2.406, 1.655],
    [2.531, -2.412, 1.335],
    [2.542, -2.105, 1.007],
    [2.303, -1.225, 0.7595],
    [1.537, -0.2245, 0.8175],
    [0.308, 0.3498, 0.9945],
    [-1.318, 0.1108, 0.9669],
    [-2.366, -0.9641, 0.9091],
    [-2.389, -2.211, 1.003],
    [-1.709, -2.851, 1.215],
    [-0.853, -2.861, 1.511],
    [-0.335, -2.232, 1.599],
    [-0.5085, -1.169, 1.154],
    [-1.341, 0.2274, 0.909],
    [-1.837, 1.643, 1.2],
    [-1.114, 3.195, 1.357],
    [0.6489, 3.297, 1.281],
    [2.284, 1.635, 1.203],
    [2.723, 0.1554, 1.333],
    [2.713, -1.227, 1.627],
    [2.554, -2.107, 1.795],
    [2.505, -2.411, 1.629],
    [2.53, -2.413, 1.314],
    [2.542, -2.105, 1.007],
    [2.306, -1.219, 0.7755],
    [1.538, -0.2223, 0.8314],
    [0.308, 0.3498, 0.9945],
    [-1.335, 0.1166, 0.9754],
    [-2.378, -0.9572, 0.909],
    [-2.389, -2.211, 1.003],
    [-1.704, -2.84, 1.202],
    [-0.8434, -2.849, 1.489],
    [-0.335, -2.232, 1.599],
    [-0.5374, -1.191, 1.202],
    [-1.355, 0.1862, 0.9638],
    [-1.837, 1.643, 1.2],
    [-1.108, 3.317, 1.336],
    [0.6783, 3.386, 1.244],
    [2.284, 1.635, 1.203],
    [2.703, 0.1248, 1.349],
    [2.691, -1.254, 1.651],
    [2.554, -2.107, 1.796],
    [2.515, -2.444, 1.605],
    [2.549, -2.431, 1.282],
    [2.54, -2.107, 1.004],
    [1.619, -0.4639, 0.8026],
    [0.4887, 0.2996, 1.041],
    [0.3099, 0.3554, 1.0],
];

static TO_TIMESTAMPS: [f32; 60] = [
    0.1358, 0.3138, 0.514, 0.6521, 0.8031, 0.9633, 1.177, 1.367, 1.553, 1.668, 1.786, 1.907, 1.99,
    2.072, 2.154, 2.279, 2.402, 2.524, 2.674, 2.825, 2.983, 3.107, 3.229, 3.36, 3.502, 3.647,
    3.786, 3.986, 4.18, 4.376, 4.488, 4.604, 4.721, 4.804, 4.886, 4.967, 5.091, 5.213, 5.334,
    5.484, 5.636, 5.794, 5.918, 6.039, 6.17, 6.313, 6.457, 6.592, 6.796, 7.001, 7.197, 7.309,
    7.425, 7.537, 7.625, 7.711, 7.796, 8.02, 8.23, 8.425,
];

const TO_FIXTURE: Fixture = Fixture {
    name: "trajectory_time_optimal.yaml",
    start_pos: TO_START_POS,
    waypoints: &TO_WAYPOINTS,
    timestamps: &TO_TIMESTAMPS,
};

// ─── helpers ───────────────────────────────────────────────────────────

/// Body rates from the pole-safe flatness map at ψ=0 — the same
/// `flatness_to_thrust_omega` the outer loop feeds `u_refs` with.
/// Returns `(omega_xy_norm, |omega_z|)` in rad/s; zeros for the
/// near-free-fall samples the map refuses (should not occur on these
/// schedules).
///
/// Note on `omega_z`: the map returns the *min-norm* body rate
/// `ω_world = z_b × dz_b + ψ̇·ẑ`, whose component along `z_b` is
/// identically zero when `ψ̇ = 0`. These fixtures carry no yaw
/// schedule, so `omega_z ≡ 0` and its envelope assertion below is a
/// finiteness/regression guard rather than a live constraint — it
/// becomes load-bearing the moment a fixture gains a yaw schedule
/// (`headings` / `lookahead`). The previous hand-rolled body rate
/// reported a nonzero `ω_z` here, but that term is the spin required
/// to hold the *intrinsic tilt-yaw Euler angle* constant, not a
/// physical requirement of flying the path.
fn body_rates_flatness(traj: &PiecewisePolynomial, t: f32) -> (f32, f32) {
    let a = traj.get_acc(t);
    let j = traj.get_jerk(t);
    match flatness_to_thrust_omega(a, j, 0.0, 0.0, GRAVITY_M_S2) {
        Ok((_tpm, _q, omega)) => {
            let omega_xy = (omega[0] * omega[0] + omega[1] * omega[1]).sqrt();
            (omega_xy, omega[2].abs())
        }
        Err(_) => (0.0, 0.0),
    }
}

/// Build the offline trajectory exactly the way `plan_offline` does
/// (modulo defmt logging and the abort/state plumbing).
///
/// For n-piece schedules: MINCO needs n−1 intermediate waypoints + 1
/// tail. The YAML provides exactly n waypoints, so wp[0..n-1] are the
/// intermediates and wp[n-1] is the tail — the offline schedule maps
/// 1:1 onto the MINCO solver, no degenerate hover at the end.
fn build_trajectory(fix: &Fixture) -> (PiecewisePolynomial, Vec<f32>) {
    let n_pieces = fix.timestamps.len();
    assert_eq!(
        fix.waypoints.len(),
        n_pieces,
        "{}: expected #waypoints == #timestamps (= n_pieces)",
        fix.name
    );

    // Recover per-segment durations from the absolute timestamp schedule.
    let mut durations = vec![0.0f32; n_pieces];
    let mut prev_t = 0.0f32;
    for i in 0..n_pieces {
        let ts = fix.timestamps[i];
        let d = ts - prev_t;
        assert!(
            d.is_finite() && d > 0.0,
            "{}: invalid timestamps at i={i} (d={d}) — non-monotonic schedule",
            fix.name
        );
        durations[i] = d;
        prev_t = ts;
    }

    let n_intermediate = n_pieces - 1;
    let mut intermediate = vec![ZERO3; n_intermediate];
    for i in 0..n_intermediate {
        intermediate[i] = Vec3::from(fix.waypoints[i]);
    }
    let tail_pos = Vec3::from(fix.waypoints[n_pieces - 1]);
    let head_pos = Vec3::from(fix.start_pos);

    // PVAJ boundaries with zero v/a/j — identical to `plan_offline`.
    let head: [Vec3; 4] = [head_pos, ZERO3, ZERO3, ZERO3];
    let tail: [Vec3; 4] = [tail_pos, ZERO3, ZERO3, ZERO3];

    // Host test: the ~42 KB MincoSnap lives on the (8 MB) test stack;
    // only firmware callers need the StaticCell treatment.
    let mut minco = MincoSnap::new(&head, &tail, n_pieces);
    minco.set_boundary(&head, &tail);
    minco.solve(&intermediate, &durations);

    (minco.get_trajectory(), durations)
}

/// Continuity + waypoint-hit checks. Panics on first violation.
fn check_continuity(traj: &PiecewisePolynomial, fix: &Fixture) {
    let n_pieces = fix.timestamps.len();
    let dur = traj.total_duration();
    let expected_dur = fix.timestamps[n_pieces - 1];
    assert!(
        (dur - expected_dur).abs() < 1e-3,
        "{}: total duration mismatch: minco={dur:.6}, yaml={expected_dur:.6}",
        fix.name
    );

    // Boundary: head pose and zero head velocity.
    let p0 = traj.get_pos(0.0);
    let v0 = traj.get_vel(0.0);
    let head_expected = Vec3::from(fix.start_pos);
    assert!(
        (p0 - head_expected).norm() < 1e-3,
        "{}: head position mismatch: {p0:?} vs {head_expected:?}",
        fix.name
    );
    assert!(
        v0.norm() < 1e-3,
        "{}: head velocity not zero: {v0:?}",
        fix.name
    );

    // Boundary: tail pose (= last waypoint) and zero tail velocity.
    let pf = traj.get_pos(dur);
    let vf = traj.get_vel(dur);
    let tail_expected = Vec3::from(fix.waypoints[n_pieces - 1]);
    assert!(
        (pf - tail_expected).norm() < 1e-3,
        "{}: tail position mismatch: {pf:?} vs {tail_expected:?}",
        fix.name
    );
    assert!(
        vf.norm() < 1e-3,
        "{}: tail velocity not zero: {vf:?}",
        fix.name
    );

    // Internal C² continuity: evaluate piece i at its own duration vs.
    // piece i+1 at parameter 0. MINCO degree-5 is C² by construction.
    for i in 0..(n_pieces - 1) {
        let left = traj.piece(i);
        let right = traj.piece(i + 1);
        let t_left = left.duration;

        let dp = (right.get_pos(0.0) - left.get_pos(t_left)).norm();
        let dv = (right.get_vel(0.0) - left.get_vel(t_left)).norm();
        let da = (right.get_acc(0.0) - left.get_acc(t_left)).norm();
        assert!(
            dp < CONTINUITY_TOL_POS,
            "{}: position jump at boundary {i}: {dp:.6}",
            fix.name
        );
        assert!(
            dv < CONTINUITY_TOL_VEL,
            "{}: velocity jump at boundary {i}: {dv:.6}",
            fix.name
        );
        assert!(
            da < CONTINUITY_TOL_ACC,
            "{}: acceleration jump at boundary {i}: {da:.6}",
            fix.name
        );
    }

    // Each YAML waypoint must be hit at the correct timestamp.
    for i in 0..n_pieces {
        let t = fix.timestamps[i];
        let p_at = traj.get_pos(t);
        let wp = Vec3::from(fix.waypoints[i]);
        assert!(
            (p_at - wp).norm() < 1e-3,
            "{}: waypoint {i} not hit at t={t:.4}: {p_at:?} vs {wp:?}",
            fix.name
        );
    }
}

/// Sweep the trajectory at 10 ms and assert peak thrust + body rates
/// stay within `CONSTRAINT_SLACK × limit`.
fn check_envelope(traj: &PiecewisePolynomial, fix: &Fixture) {
    let dur = traj.total_duration();
    let n_steps = (dur / SAMPLE_DT_S).ceil() as usize;

    let mut peak_thrust_n = 0.0f32;
    let mut peak_thrust_t = 0.0f32;
    let mut peak_omega_xy = 0.0f32;
    let mut peak_omega_xy_t = 0.0f32;
    let mut peak_omega_z = 0.0f32;
    let mut peak_omega_z_t = 0.0f32;

    for k in 0..=n_steps {
        let t = (k as f32 * SAMPLE_DT_S).min(dur);

        let a = traj.get_acc(t);
        let alpha = a + Vec3::new(0.0, 0.0, GRAVITY_M_S2);
        let f = MASS_KG * alpha.norm();
        if f > peak_thrust_n {
            peak_thrust_n = f;
            peak_thrust_t = t;
        }

        let (omega_xy, omega_z) = body_rates_flatness(traj, t);
        if omega_xy > peak_omega_xy {
            peak_omega_xy = omega_xy;
            peak_omega_xy_t = t;
        }
        if omega_z > peak_omega_z {
            peak_omega_z = omega_z;
            peak_omega_z_t = t;
        }
    }

    let thrust_pct = 100.0 * (peak_thrust_n - MAX_THRUST_N) / MAX_THRUST_N;
    let xy_pct = 100.0 * (peak_omega_xy - MAX_BODY_RATE_XY_RAD_S) / MAX_BODY_RATE_XY_RAD_S;
    let z_pct = 100.0 * (peak_omega_z - MAX_BODY_RATE_Z_RAD_S) / MAX_BODY_RATE_Z_RAD_S;

    println!(
        "\n  [{}] peaks (sampled at {} ms over {:.2} s):\n\
           thrust    : peak = {:.3} N    at t = {:.3} s   (limit {:.3} N, over {:+.2}%)\n\
           omega_xy  : peak = {:.3} rad/s at t = {:.3} s   (limit {:.3}, over {:+.2}%)\n\
           omega_z   : peak = {:.3} rad/s at t = {:.3} s   (limit {:.3}, over {:+.2}%)\n",
        fix.name,
        (SAMPLE_DT_S * 1000.0) as i32,
        dur,
        peak_thrust_n,
        peak_thrust_t,
        MAX_THRUST_N,
        thrust_pct,
        peak_omega_xy,
        peak_omega_xy_t,
        MAX_BODY_RATE_XY_RAD_S,
        xy_pct,
        peak_omega_z,
        peak_omega_z_t,
        MAX_BODY_RATE_Z_RAD_S,
        z_pct,
    );

    assert!(
        peak_thrust_n.is_finite(),
        "{}: non-finite peak thrust",
        fix.name
    );
    assert!(
        peak_omega_xy.is_finite() && peak_omega_z.is_finite(),
        "{}: non-finite peak body rate",
        fix.name
    );

    assert!(
        peak_thrust_n <= MAX_THRUST_N * CONSTRAINT_SLACK,
        "{}: collective thrust LARGE VIOLATION: peak {:.3} N at t={:.3}s exceeds {:.3} N \
         (limit {:.3} × slack {:.2})",
        fix.name,
        peak_thrust_n,
        peak_thrust_t,
        MAX_THRUST_N * CONSTRAINT_SLACK,
        MAX_THRUST_N,
        CONSTRAINT_SLACK,
    );
    assert!(
        peak_omega_xy <= MAX_BODY_RATE_XY_RAD_S * CONSTRAINT_SLACK,
        "{}: body rate xy LARGE VIOLATION: peak {:.3} rad/s at t={:.3}s exceeds {:.3} rad/s \
         (limit {:.3} × slack {:.2})",
        fix.name,
        peak_omega_xy,
        peak_omega_xy_t,
        MAX_BODY_RATE_XY_RAD_S * CONSTRAINT_SLACK,
        MAX_BODY_RATE_XY_RAD_S,
        CONSTRAINT_SLACK,
    );
    assert!(
        peak_omega_z <= MAX_BODY_RATE_Z_RAD_S * CONSTRAINT_SLACK,
        "{}: body rate z LARGE VIOLATION: peak {:.3} rad/s at t={:.3}s exceeds {:.3} rad/s \
         (limit {:.3} × slack {:.2})",
        fix.name,
        peak_omega_z,
        peak_omega_z_t,
        MAX_BODY_RATE_Z_RAD_S * CONSTRAINT_SLACK,
        MAX_BODY_RATE_Z_RAD_S,
        CONSTRAINT_SLACK,
    );
}

// ─── tests ─────────────────────────────────────────────────────────────

#[test]
fn trajectory_slow_yaml_passes_envelope() {
    let (traj, _) = build_trajectory(&SLOW_FIXTURE);
    check_continuity(&traj, &SLOW_FIXTURE);
    check_envelope(&traj, &SLOW_FIXTURE);
}

#[test]
fn trajectory_time_optimal_yaml_passes_envelope() {
    let (traj, _) = build_trajectory(&TO_FIXTURE);
    check_continuity(&traj, &TO_FIXTURE);
    check_envelope(&traj, &TO_FIXTURE);
}
