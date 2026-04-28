//! Offline-trajectory test for `MincoSnap` (degree-7 min-snap solver),
//! the higher-order sibling of the `MincoJerk` test in
//! `tests/offline_minco.rs`.
//!
//! For each of the two YAML fixtures (slow, time-optimal):
//!   1. Recover per-segment durations from the absolute-time schedule.
//!   2. Feed the n−1 intermediate waypoints + tail = wp[n−1] into a
//!      `MincoSnap` solver of size n with PVAJ boundary conditions
//!      (head/tail PVA = 0, head/tail jerk = 0).
//!   3. Evaluate continuity at internal piece boundaries.
//!   4. Walk the trajectory at 10 ms, push each (p, v, a, j, s) sample
//!      through `flatness_to_state_tilt_yaw` (yaw triple = 0), and
//!      record peak collective thrust + peak body rates.
//!   5. Assert the peaks stay within `CONSTRAINT_SLACK × limit`.
//!
//! Run: `cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --test minco_snap_offline`

use cybflight_core::rotation::quaternion_to_euler_angles_rpy;
use cybflight_core::trajectory_planning::minco_snap::{
    flatness_to_state_tilt_yaw, MincoSnap,
};
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};

// ─── physical constants matching the firmware's planner config ─────────

// Match the C++ vehicle params used to produce the reference statistics:
//   mass=0.55 kg, gravity=9.8066 m/s², inertia=[0.0021, 0.0018, 0.0030],
//   arm offsets ±0.075 m (x) and ±0.10 m (y) on a +-config.
const MASS_KG: f32 = 0.55;
const GRAVITY_M_S2: f32 = 9.8066;
const INERTIA: [f32; 3] = [0.0021, 0.0018, 0.0030];
// Motor positions matching the YAML: tbm_fr/bl/br/fl. C++ T_mb = M⁻¹
// where M maps motor thrusts → (F_z, τ_x, τ_y, τ_z) on the body. We
// rebuild it from these four positions and the standard ±torque
// alternating pattern so per-motor thrusts align with the C++ output.
// fr=0, bl=1, br=2, fl=3 → arm vectors:
//   fr: ( 0.075, -0.10)  spin: cw  (-z torque)
//   bl: (-0.075,  0.10)  spin: cw  (-z torque)
//   br: (-0.075, -0.10)  spin: ccw (+z torque)
//   fl: ( 0.075,  0.10)  spin: ccw (+z torque)
const ARM_X: [f32; 4] = [0.075, -0.075, -0.075, 0.075];
const ARM_Y: [f32; 4] = [-0.10, 0.10, -0.10, 0.10];
const SPIN_DIR: [f32; 4] = [-1.0, -1.0, 1.0, 1.0]; // ±torque coefficient sign
const TORQUE_COEFF_M: f32 = 0.022;

const MAX_THRUST_N: f32 = 27.2;
const MAX_BODY_RATE_XY_RAD_S: f32 = 10.0;
const MAX_BODY_RATE_Z_RAD_S: f32 = 6.0;

const SAMPLE_DT_S: f32 = 0.010;
/// 30 % slack — same convention as the MincoJerk test.
const CONSTRAINT_SLACK: f32 = 1.30;

const CONTINUITY_TOL_POS: f32 = 1e-3;
const CONTINUITY_TOL_VEL: f32 = 1e-2;
const CONTINUITY_TOL_ACC: f32 = 1e-1;
const CONTINUITY_TOL_JERK: f32 = 1.0;

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

fn build_trajectory(fix: &Fixture) -> PiecewisePolynomial {
    let n_pieces = fix.timestamps.len();
    assert_eq!(
        fix.waypoints.len(),
        n_pieces,
        "{}: expected #waypoints == #timestamps (= n_pieces)",
        fix.name
    );

    let mut durations = vec![0.0f32; n_pieces];
    let mut prev_t = 0.0f32;
    for i in 0..n_pieces {
        let ts = fix.timestamps[i];
        let d = ts - prev_t;
        assert!(
            d.is_finite() && d > 0.0,
            "{}: invalid timestamps at i={i} (d={d})",
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

    // PVAJ boundaries: zero v, a, j at both ends.
    let head = [head_pos, ZERO3, ZERO3, ZERO3];
    let tail = [tail_pos, ZERO3, ZERO3, ZERO3];

    let mut minco = MincoSnap::new(&head, &tail, n_pieces);
    minco.set_boundary(&head, &tail);
    minco.solve(&intermediate, &durations);
    minco.get_trajectory()
}

fn check_continuity(traj: &PiecewisePolynomial, fix: &Fixture) {
    let n_pieces = fix.timestamps.len();
    let dur = traj.total_duration();
    let expected_dur = fix.timestamps[n_pieces - 1];
    assert!(
        (dur - expected_dur).abs() < 1e-3,
        "{}: total duration mismatch: minco={dur:.6}, yaml={expected_dur:.6}",
        fix.name
    );

    let p0 = traj.get_pos(0.0);
    let v0 = traj.get_vel(0.0);
    let a0 = traj.get_acc(0.0);
    let j0 = traj.get_jerk(0.0);
    let head_expected = Vec3::from(fix.start_pos);
    assert!((p0 - head_expected).norm() < 1e-3, "{}: head pos", fix.name);
    assert!(v0.norm() < 1e-3, "{}: head v not zero: {v0:?}", fix.name);
    assert!(a0.norm() < 1e-2, "{}: head a not zero: {a0:?}", fix.name);
    assert!(j0.norm() < 1e-1, "{}: head j not zero: {j0:?}", fix.name);

    let pf = traj.get_pos(dur);
    let vf = traj.get_vel(dur);
    let af = traj.get_acc(dur);
    let jf = traj.get_jerk(dur);
    let tail_expected = Vec3::from(fix.waypoints[n_pieces - 1]);
    assert!((pf - tail_expected).norm() < 1e-3, "{}: tail pos", fix.name);
    assert!(vf.norm() < 1e-3, "{}: tail v not zero: {vf:?}", fix.name);
    assert!(af.norm() < 1e-2, "{}: tail a not zero: {af:?}", fix.name);
    assert!(jf.norm() < 1e-1, "{}: tail j not zero: {jf:?}", fix.name);

    // C^5 continuity at internal boundaries: position, velocity,
    // acceleration, jerk all agree across each junction.
    for i in 0..(n_pieces - 1) {
        let left = traj.piece(i);
        let right = traj.piece(i + 1);
        let t_left = left.duration;

        let dp = (right.get_pos(0.0) - left.get_pos(t_left)).norm();
        let dv = (right.get_vel(0.0) - left.get_vel(t_left)).norm();
        let da = (right.get_acc(0.0) - left.get_acc(t_left)).norm();
        let dj = (right.get_jerk(0.0) - left.get_jerk(t_left)).norm();
        assert!(dp < CONTINUITY_TOL_POS, "{}: pos jump i={i}: {dp}", fix.name);
        assert!(dv < CONTINUITY_TOL_VEL, "{}: vel jump i={i}: {dv}", fix.name);
        assert!(da < CONTINUITY_TOL_ACC, "{}: acc jump i={i}: {da}", fix.name);
        assert!(dj < CONTINUITY_TOL_JERK, "{}: jerk jump i={i}: {dj}", fix.name);
    }

    for i in 0..n_pieces {
        let t = fix.timestamps[i];
        let p_at = traj.get_pos(t);
        let wp = Vec3::from(fix.waypoints[i]);
        assert!(
            (p_at - wp).norm() < 1e-3,
            "{}: waypoint {i} not hit at t={t}: {p_at:?} vs {wp:?}",
            fix.name
        );
    }
}

/// Solve the static thrust mix `F = M · u` where
///   F = (F_z, τ_x, τ_y, τ_z) = (collective, body torques)
///   u = (u₀, u₁, u₂, u₃)     = per-motor thrusts [N]
/// Mirrors the C++ `quad_params_.T_mb` (which is M⁻¹) — we form M
/// from the YAML arm positions and torque-coefficient signs and solve
/// the 4x4 system in-place. This isolates the discrepancy between
/// "scalar collective thrust" (zB·α scaled by mass) and "per-motor
/// thrust force" (after distributing by τ).
fn motor_thrusts(collective_n: f32, tau: Vec3) -> [f32; 4] {
    // M[0, k] = 1                 (each motor adds to F_z)
    // M[1, k] = -arm_y[k]         (τ_x = sum -y · u   →  rolling moment)
    // M[2, k] =  arm_x[k]         (τ_y = sum  x · u)
    // M[3, k] =  spin[k] · κ      (τ_z = sum spin · κ · u)
    // (Right-handed body frame: x-fwd, y-left, z-up.)
    let mut m = [[0.0f32; 4]; 4];
    for k in 0..4 {
        m[0][k] = 1.0;
        m[1][k] = -ARM_Y[k];
        m[2][k] = ARM_X[k];
        m[3][k] = SPIN_DIR[k] * TORQUE_COEFF_M;
    }
    let b = [collective_n, tau[0], tau[1], tau[2]];
    solve_4x4(&m, &b)
}

/// Tiny dense 4x4 solver via Gaussian elimination with partial
/// pivoting. Only used for the per-motor thrust diagnostic.
fn solve_4x4(a: &[[f32; 4]; 4], b: &[f32; 4]) -> [f32; 4] {
    let mut m = [[0.0f32; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j];
        }
        m[i][4] = b[i];
    }
    for k in 0..4 {
        let mut pivot = k;
        for r in (k + 1)..4 {
            if m[r][k].abs() > m[pivot][k].abs() {
                pivot = r;
            }
        }
        m.swap(k, pivot);
        let piv = m[k][k];
        for r in (k + 1)..4 {
            let factor = m[r][k] / piv;
            for c in k..5 {
                m[r][c] -= factor * m[k][c];
            }
        }
    }
    let mut x = [0.0f32; 4];
    for i in (0..4).rev() {
        let mut s = m[i][4];
        for j in (i + 1)..4 {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    x
}

/// Dump the same statistics the C++ `MincoSnapTrajectory` printer
/// produces (see `tmp/planner/include/drolib/system/minco_snap_trajectory.hpp:46`),
/// computed by sampling at `SAMPLE_DT_S` and pushing through
/// `flatness_to_state_tilt_yaw` with yaw triple [0, 0, 0]. Units match
/// the C++ printout — including the misleading `[N]` label on
/// `min/maxCollectivethrust`, which is in fact per-mass acceleration
/// (m/s²) since the C++ assigns `setpoint.input.collective_thrust =
/// zB·α` directly.
fn dump_extremum_stats(traj: &PiecewisePolynomial, fix: &Fixture) {
    let dur = traj.total_duration();
    let n_steps = (dur / SAMPLE_DT_S).ceil() as usize;

    let mut min_pos = [f32::INFINITY; 3];
    let mut max_pos = [f32::NEG_INFINITY; 3];
    let mut max_vel = 0.0f32;
    let mut max_acc = 0.0f32;
    let mut max_tilt_rad = 0.0f32;
    let mut min_omg = [f32::INFINITY; 3];
    let mut max_omg = [f32::NEG_INFINITY; 3];
    let mut min_rpy_deg = [f32::INFINITY; 3];
    let mut max_rpy_deg = [f32::NEG_INFINITY; 3];
    let mut min_thrusts = [f32::INFINITY; 4];
    let mut max_thrusts = [f32::NEG_INFINITY; 4];
    let mut min_coll_per_mass = f32::INFINITY;
    let mut max_coll_per_mass = f32::NEG_INFINITY;

    for k in 0..=n_steps {
        let t = (k as f32 * SAMPLE_DT_S).min(dur);
        let p = traj.get_pos(t);
        let v = traj.get_vel(t);
        let a = traj.get_acc(t);
        let j = traj.get_jerk(t);
        let s = traj.get_snap(t);

        for d in 0..3 {
            min_pos[d] = min_pos[d].min(p[d]);
            max_pos[d] = max_pos[d].max(p[d]);
        }
        max_vel = max_vel.max(v.norm());
        max_acc = max_acc.max(a.norm());

        let st = flatness_to_state_tilt_yaw(a, j, s, [0.0; 3], GRAVITY_M_S2)
            .expect("offline trajectory should not hit a flatness singularity");

        // Tilt = acos(zB.z). zB = α / ‖α‖ → zB.z = (a.z + g) / ‖a + g·ẑ‖.
        let alpha = Vec3::new(a[0], a[1], a[2] + GRAVITY_M_S2);
        let alpha_norm = alpha.norm().max(1e-8);
        let zb_z = (alpha[2] / alpha_norm).clamp(-1.0, 1.0);
        let tilt = libm::acosf(zb_z);
        max_tilt_rad = max_tilt_rad.max(tilt);

        for d in 0..3 {
            min_omg[d] = min_omg[d].min(st.omega[d]);
            max_omg[d] = max_omg[d].max(st.omega[d]);
        }

        let rpy = quaternion_to_euler_angles_rpy(&st.attitude);
        for d in 0..3 {
            let deg = rpy[d].to_degrees();
            min_rpy_deg[d] = min_rpy_deg[d].min(deg);
            max_rpy_deg[d] = max_rpy_deg[d].max(deg);
        }

        // Collective thrust per-unit-mass (the C++ `[N]` printout).
        min_coll_per_mass = min_coll_per_mass.min(st.thrust_per_mass);
        max_coll_per_mass = max_coll_per_mass.max(st.thrust_per_mass);

        // Per-motor thrust: collective in Newtons + body torque
        // τ = I·ω̇ + ω × (I·ω) (Euler's equations, full form).
        let omg = st.omega;
        let omd = st.omega_dot;
        let i_omg = Vec3::new(INERTIA[0] * omg[0], INERTIA[1] * omg[1], INERTIA[2] * omg[2]);
        let gyro = omg.cross(&i_omg);
        let tau = Vec3::new(
            INERTIA[0] * omd[0] + gyro[0],
            INERTIA[1] * omd[1] + gyro[1],
            INERTIA[2] * omd[2] + gyro[2],
        );
        let collective_n = MASS_KG * st.thrust_per_mass;
        let u = motor_thrusts(collective_n, tau);
        for k in 0..4 {
            min_thrusts[k] = min_thrusts[k].min(u[k]);
            max_thrusts[k] = max_thrusts[k].max(u[k]);
        }
    }

    println!(
        "\n  ===== [{}] MincoSnap statistics (vs C++ reference) =====\n\
           minPos:     {:.4} {:.4} {:.4} [m]\n\
           maxPos:     {:.4} {:.4} {:.4} [m]\n\
           maxVel:     {:.4} [m/s]\n\
           maxAcc:     {:.4} [m^2/s]\n\
           maxTilt:    {:.4} [deg]\n\
           minOmg:     {:.4} {:.4} {:.4} [rad/s]\n\
           maxOmg:     {:.4} {:.4} {:.4} [rad/s]\n\
           minEuler:   {:.4} {:.4} {:.4} [deg]\n\
           maxEuler:   {:.4} {:.4} {:.4} [deg]\n\
           minThrusts: {:.4} {:.4} {:.4} {:.4} [N]\n\
           maxThrusts: {:.4} {:.4} {:.4} {:.4} [N]\n\
           minCollectivethrust: {:.4} [N]    (note: per-mass, m/s²)\n\
           maxCollectivethrust: {:.4} [N]    (note: per-mass, m/s²)\n\
           Length:     (n/a — not computed)\n\
           Duration:   {:.4} [s]\n",
        fix.name,
        min_pos[0], min_pos[1], min_pos[2],
        max_pos[0], max_pos[1], max_pos[2],
        max_vel,
        max_acc,
        max_tilt_rad.to_degrees(),
        min_omg[0], min_omg[1], min_omg[2],
        max_omg[0], max_omg[1], max_omg[2],
        min_rpy_deg[0], min_rpy_deg[1], min_rpy_deg[2],
        max_rpy_deg[0], max_rpy_deg[1], max_rpy_deg[2],
        min_thrusts[0], min_thrusts[1], min_thrusts[2], min_thrusts[3],
        max_thrusts[0], max_thrusts[1], max_thrusts[2], max_thrusts[3],
        min_coll_per_mass,
        max_coll_per_mass,
        dur,
    );
}

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
        let j = traj.get_jerk(t);
        let s = traj.get_snap(t);

        let st = flatness_to_state_tilt_yaw(a, j, s, [0.0; 3], 9.81)
            .expect("offline trajectory should not hit a flatness singularity");
        let f = MASS_KG * st.thrust_per_mass;
        if f > peak_thrust_n {
            peak_thrust_n = f;
            peak_thrust_t = t;
        }
        let omg_xy =
            (st.omega[0] * st.omega[0] + st.omega[1] * st.omega[1]).sqrt();
        let omg_z = st.omega[2].abs();
        if omg_xy > peak_omega_xy {
            peak_omega_xy = omg_xy;
            peak_omega_xy_t = t;
        }
        if omg_z > peak_omega_z {
            peak_omega_z = omg_z;
            peak_omega_z_t = t;
        }
    }

    let thrust_pct = 100.0 * (peak_thrust_n - MAX_THRUST_N) / MAX_THRUST_N;
    let xy_pct = 100.0 * (peak_omega_xy - MAX_BODY_RATE_XY_RAD_S) / MAX_BODY_RATE_XY_RAD_S;
    let z_pct = 100.0 * (peak_omega_z - MAX_BODY_RATE_Z_RAD_S) / MAX_BODY_RATE_Z_RAD_S;

    println!(
        "\n  [{}] MincoSnap peaks (sampled at {} ms over {:.2} s):\n\
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
        peak_thrust_n.is_finite() && peak_omega_xy.is_finite() && peak_omega_z.is_finite(),
        "{}: non-finite peak — degenerate trajectory",
        fix.name
    );

    assert!(
        peak_thrust_n <= MAX_THRUST_N * CONSTRAINT_SLACK,
        "{}: thrust LARGE VIOLATION: peak {:.3} N > {:.3} N",
        fix.name,
        peak_thrust_n,
        MAX_THRUST_N * CONSTRAINT_SLACK,
    );
    assert!(
        peak_omega_xy <= MAX_BODY_RATE_XY_RAD_S * CONSTRAINT_SLACK,
        "{}: omega_xy LARGE VIOLATION: peak {:.3} rad/s > {:.3} rad/s",
        fix.name,
        peak_omega_xy,
        MAX_BODY_RATE_XY_RAD_S * CONSTRAINT_SLACK,
    );
    assert!(
        peak_omega_z <= MAX_BODY_RATE_Z_RAD_S * CONSTRAINT_SLACK,
        "{}: omega_z LARGE VIOLATION: peak {:.3} rad/s > {:.3} rad/s",
        fix.name,
        peak_omega_z,
        MAX_BODY_RATE_Z_RAD_S * CONSTRAINT_SLACK,
    );
}

// ─── tests ─────────────────────────────────────────────────────────────

#[test]
fn snap_trajectory_slow_yaml_passes_envelope() {
    let traj = build_trajectory(&SLOW_FIXTURE);
    check_continuity(&traj, &SLOW_FIXTURE);
    dump_extremum_stats(&traj, &SLOW_FIXTURE);
    check_envelope(&traj, &SLOW_FIXTURE);
}

#[test]
fn snap_trajectory_time_optimal_yaml_passes_envelope() {
    let traj = build_trajectory(&TO_FIXTURE);
    check_continuity(&traj, &TO_FIXTURE);
    dump_extremum_stats(&traj, &TO_FIXTURE);
    check_envelope(&traj, &TO_FIXTURE);
}
