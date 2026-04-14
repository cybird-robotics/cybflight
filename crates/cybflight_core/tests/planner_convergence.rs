//! Integration tests for the trajectory planner.
//!
//! Verifies:
//! 1. Solver converges on typical goto / waypoint problems.
//! 2. Boundary conditions (head/tail pos, vel, acc) are respected.
//! 3. Dynamic constraints (velocity, tilt, body rate, collective thrust) are
//!    respected along the optimized trajectory within a small slack tolerance
//!    (soft constraints — smoothed-L1 penalties allow small excursions at the
//!    boundary by design).
//!
//! Run: `cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --test planner_convergence`

use cybflight_core::params::{PlannerParams, VehicleParams};
use cybflight_core::trajectory_planning::planner::{plan, PlannerInput, PlannerResult, SolverStatus};
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_core::trajectory_planning::types::{norm_sq3, sub3, Vec3, ZERO3};
use std::fs;
use std::path::PathBuf;

// ─── tolerances ─────────────────────────────────────────────────────────────

/// Soft-constraint slack: smoothed-L1 penalties allow small excursions by
/// design, so we verify constraints up to a small multiple of the bound.
const CONSTRAINT_SLACK: f32 = 1.15; // 15% over-shoot allowed

/// Tolerance on boundary position / velocity matching.
const BOUNDARY_POS_TOL: f32 = 1e-3;
const BOUNDARY_VEL_TOL: f32 = 1e-3;

// ─── helpers ────────────────────────────────────────────────────────────────

fn vec_norm(v: Vec3) -> f32 {
    norm_sq3(v).sqrt()
}

fn assert_converged(status: SolverStatus, final_cost: f32) {
    assert!(
        matches!(
            status,
            SolverStatus::Convergence | SolverStatus::Stop | SolverStatus::MaxIterations
        ),
        "solver failed: {status:?}"
    );
    assert!(final_cost.is_finite(), "final cost is not finite");
    assert!(final_cost >= 0.0, "final cost is negative: {final_cost}");
}

/// Sample the trajectory densely and return the max of `f(t)` across samples,
/// plus the time at which the max occurred.
fn max_along<F: Fn(f32) -> f32>(total_dur: f32, n_samples: usize, f: F) -> (f32, f32) {
    let mut best = f32::NEG_INFINITY;
    let mut best_t = 0.0;
    for i in 0..=n_samples {
        let t = total_dur * i as f32 / n_samples as f32;
        let v = f(t);
        if v > best {
            best = v;
            best_t = t;
        }
    }
    (best, best_t)
}

/// Quadrotor config tuned for small-vehicle planning tests.
fn test_config() -> QuadPlanningConfig {
    // Start from a sensible VehicleParams and override planner bounds for tests.
    let mut vp = test_vehicle_params();
    vp.planner.max_vel_m_s = 4.0;
    vp.planner.max_tilt_rad = 60_f32.to_radians();
    // Use strong constraint weights to make penalties effective.
    vp.planner.weight_vel = 50.0;
    vp.planner.weight_tilt = 50.0;
    vp.planner.weight_body_rate = 50.0;
    vp.planner.weight_thrust = 10.0;
    vp.planner.weight_energy = 0.1;
    vp.planner.weight_time = 1.0;
    vp.planner.smoothing_eps = 0.01;
    vp.planner.num_check_per_piece = 8;
    vp.planner.bfgs_trust.max_iterations = 200;
    QuadPlanningConfig::from_vehicle_params(&vp)
}

/// Build a config specifically for CONSTRAINT-ACTIVATION tests.
///
/// Uses `weight_energy = 0.0` to avoid the "waypoint collapse" pathology:
/// with any smoothness regularizer, the optimizer drifts the free waypoint
/// positions toward each other (the Rust port lacks the C++ corridor/ball
/// machinery that anchors D variables). At we=0 the waypoints stay close
/// to user-specified values, so the trajectory actually traverses the
/// intended path and constraints can genuinely become binding.
fn test_config_for_activation(max_vel_m_s: f32, max_tilt_deg: f32) -> QuadPlanningConfig {
    let mut vp = test_vehicle_params();
    vp.planner.max_vel_m_s = max_vel_m_s;
    vp.planner.max_tilt_rad = max_tilt_deg.to_radians();
    // Keep all four dynamic penalties active.
    vp.planner.weight_vel = 50.0;
    vp.planner.weight_tilt = 50.0;
    vp.planner.weight_body_rate = 50.0;
    vp.planner.weight_thrust = 10.0;
    // Zero energy → no waypoint collapse.
    vp.planner.weight_energy = 0.0;
    vp.planner.weight_time = 1.0;
    vp.planner.smoothing_eps = 0.01;
    vp.planner.num_check_per_piece = 8;
    vp.planner.bfgs_trust.max_iterations = 500;
    QuadPlanningConfig::from_vehicle_params(&vp)
}

/// Same as `test_config` but with `weight_time` and `weight_energy` overridden.
/// Keeps velocity / tilt / body-rate / thrust constraints active so the
/// resulting trajectory is still physically admissible.
fn test_config_with_time_energy(weight_time: f32, weight_energy: f32) -> QuadPlanningConfig {
    let mut vp = test_vehicle_params();
    vp.planner.max_vel_m_s = 4.0;
    vp.planner.max_tilt_rad = 60_f32.to_radians();
    vp.planner.weight_vel = 50.0;
    vp.planner.weight_tilt = 50.0;
    vp.planner.weight_body_rate = 50.0;
    vp.planner.weight_thrust = 10.0;
    vp.planner.weight_time = weight_time;
    vp.planner.weight_energy = weight_energy;
    vp.planner.smoothing_eps = 0.01;
    vp.planner.num_check_per_piece = 8;
    vp.planner.bfgs_trust.max_iterations = 200;
    QuadPlanningConfig::from_vehicle_params(&vp)
}

fn test_vehicle_params() -> VehicleParams {
    use cybflight_core::mixer::{MotorParams, RigidBodyParams, SpinDir};
    use cybflight_core::params::{
        ControlGains, IndiControllerParams, IndiEffectivenessParams, LearnerParams, MpcParams,
    };

    VehicleParams {
        body: RigidBodyParams {
            mass_kg: 0.55,
            inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
            max_rate_rad_s: [10.0, 10.0, 6.0],
        },
        motors: [
            MotorParams {
                position_m: [-0.075, -0.1],
                spin_dir: SpinDir::Cw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
            MotorParams {
                position_m: [0.075, -0.1],
                spin_dir: SpinDir::Ccw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
            MotorParams {
                position_m: [-0.075, 0.1],
                spin_dir: SpinDir::Ccw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
            MotorParams {
                position_m: [0.075, 0.1],
                spin_dir: SpinDir::Cw,
                max_thrust_n: 8.5,
                torque_coeff_m: 0.022,
            },
        ],
        control: ControlGains {
            pos_kp: [4.0, 4.0, 5.0],
            pos_kd: [4.0, 4.0, 4.0],
            att_k_rate: [3.0, 3.0, 1.0],
            rate_kp: [0.1, 0.08, 0.05],
            rate_ki: [0.0, 0.0, 0.0],
            rate_kd: [0.0, 0.0, 0.0],
        },
        indi_effectiveness: IndiEffectivenessParams::default(),
        indi_controller: IndiControllerParams::default(),
        learner: LearnerParams::default(),
        mpc: MpcParams::default(),
        planner: PlannerParams::default(),
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[test]
fn goto_converges_and_matches_boundary() {
    let config = test_config();
    let start = [0.0, 0.0, 1.0];
    let target = [3.0, 0.0, 1.0];
    let input = PlannerInput::goto(start, ZERO3, target);

    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);
    assert_eq!(result.num_pieces, 1);

    let dur = result.trajectory.total_duration();
    assert!(dur > 0.0 && dur.is_finite(), "bad duration: {dur}");

    // Boundary: start position and zero velocity.
    let p0 = result.trajectory.get_pos(0.0);
    let v0 = result.trajectory.get_vel(0.0);
    assert!(
        vec_norm(sub3(p0, start)) < BOUNDARY_POS_TOL,
        "start pos mismatch: {p0:?} vs {start:?}"
    );
    assert!(
        vec_norm(v0) < BOUNDARY_VEL_TOL,
        "start vel not zero: {v0:?}"
    );

    // Boundary: target position and zero velocity.
    let pf = result.trajectory.get_pos(dur);
    let vf = result.trajectory.get_vel(dur);
    assert!(
        vec_norm(sub3(pf, target)) < BOUNDARY_POS_TOL,
        "end pos mismatch: {pf:?} vs {target:?}"
    );
    assert!(
        vec_norm(vf) < BOUNDARY_VEL_TOL,
        "end vel not zero: {vf:?}"
    );
}

#[test]
fn waypoints_converges_and_passes_through() {
    let config = test_config();
    let start = [0.0, 0.0, 1.0];
    let targets: [Vec3; 3] = [[2.0, 1.0, 1.0], [4.0, -1.0, 1.0], [6.0, 0.0, 1.0]];
    let input = PlannerInput::waypoints(start, ZERO3, &targets);

    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);
    assert_eq!(result.num_pieces, 3);

    // Boundary: start and end.
    let dur = result.trajectory.total_duration();
    let p0 = result.trajectory.get_pos(0.0);
    let pf = result.trajectory.get_pos(dur);
    assert!(vec_norm(sub3(p0, start)) < BOUNDARY_POS_TOL);
    assert!(vec_norm(sub3(pf, targets[2])) < BOUNDARY_POS_TOL);

    // Intermediate waypoints: the trajectory should pass through the
    // optimized waypoints (which may differ from the initial waypoints)
    // exactly at the segment boundaries.
    let mut t_boundary = 0.0;
    for i in 0..(result.num_pieces - 1) {
        t_boundary += result.optimized_times[i];
        let p_at = result.trajectory.get_pos(t_boundary);
        let wp = result.optimized_waypoints[i];
        assert!(
            vec_norm(sub3(p_at, wp)) < BOUNDARY_POS_TOL,
            "waypoint {i} not hit: {p_at:?} vs {wp:?}"
        );
    }
}

#[test]
fn velocity_constraint_respected() {
    // Aggressive target relative to max_vel to stress the constraint.
    let config = test_config();
    let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [5.0, 0.0, 1.0]);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let max_vel = config.planner.max_vel_m_s;
    let dur = result.trajectory.total_duration();
    let (v_max, t_at) = max_along(dur, 200, |t| vec_norm(result.trajectory.get_vel(t)));

    assert!(
        v_max <= max_vel * CONSTRAINT_SLACK,
        "velocity {v_max:.3} m/s at t={t_at:.3} exceeds {:.3} m/s (max {:.3} * slack {:.2})",
        max_vel * CONSTRAINT_SLACK,
        max_vel,
        CONSTRAINT_SLACK
    );
}

// ─── CONSTRAINT-ACTIVATION TESTS ────────────────────────────────────────────
//
// These tests verify not just that a constraint is *not violated* but that it
// is *actively binding* — i.e. the trajectory peak approaches the limit, so
// we know the penalty is really steering the optimizer. Both use
// weight_energy = 0 to avoid the waypoint-collapse pathology.
//
// Tolerances follow the smoothed-L1 penalty's equilibrium behavior:
// - "active" means peak ≥ 0.90 × limit (the constraint is actually biting)
// - "not significantly violated" means peak ≤ 1.10 × limit

const ACTIVATION_LOWER: f32 = 0.90; // peak must reach at least 90% of limit
const ACTIVATION_UPPER: f32 = 1.10; // peak must not exceed 110% of limit

/// Peak tilt angle [rad] along a trajectory (from body-z vs world-z).
fn peak_tilt_rad(result: &PlannerResult, config: &QuadPlanningConfig) -> (f32, f32) {
    let dur = result.trajectory.total_duration();
    let g = config.grav;
    max_along(dur, 500, |t| {
        let a = result.trajectory.get_acc(t);
        let alpha = [a[0], a[1], a[2] + g];
        let na = vec_norm(alpha).max(1e-8);
        (alpha[2] / na).clamp(-1.0, 1.0).acos()
    })
}

#[test]
fn velocity_constraint_actively_binding() {
    // Tight max_vel: well below what the 5m goto would naturally reach.
    // Tilt is loose (85°) so only velocity drives the trajectory shape.
    let config = test_config_for_activation(2.0, 85.0);
    let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [5.0, 0.0, 1.0]);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let max_vel = config.planner.max_vel_m_s;
    let dur = result.trajectory.total_duration();
    let (v_max, t_at) = max_along(dur, 500, |t| vec_norm(result.trajectory.get_vel(t)));

    println!(
        "  velocity: max_vel = {:.3} m/s, peak = {:.3} m/s at t={:.3}s, dur={:.3}s",
        max_vel, v_max, t_at, dur
    );

    assert!(
        v_max >= max_vel * ACTIVATION_LOWER,
        "velocity constraint NOT ACTIVE: peak {v_max:.3} < {:.3} (90% of {:.3})",
        max_vel * ACTIVATION_LOWER,
        max_vel,
    );
    assert!(
        v_max <= max_vel * ACTIVATION_UPPER,
        "velocity constraint VIOLATED: peak {v_max:.3} > {:.3} (110% of {:.3})",
        max_vel * ACTIVATION_UPPER,
        max_vel,
    );
}

#[test]
fn tilt_constraint_actively_binding() {
    // Tight max_tilt = 25° with a LOOSE velocity limit (15 m/s, unreachable
    // for this goto). This forces tilt to be the binding constraint:
    // without a velocity cap, the optimizer wants high acceleration (and
    // hence high tilt) to finish quickly, but is capped by the tilt penalty.
    let config = test_config_for_activation(15.0, 25.0);
    let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [5.0, 0.0, 1.0]);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let max_tilt_rad = config.planner.max_tilt_rad;
    let (tilt_max, t_at) = peak_tilt_rad(&result, &config);

    println!(
        "  tilt: max_tilt = {:.2}°, peak = {:.2}° at t={:.3}s, dur={:.3}s",
        max_tilt_rad.to_degrees(),
        tilt_max.to_degrees(),
        t_at,
        result.trajectory.total_duration(),
    );

    assert!(
        tilt_max >= max_tilt_rad * ACTIVATION_LOWER,
        "tilt constraint NOT ACTIVE: peak {:.2}° < {:.2}° (90% of {:.2}°)",
        tilt_max.to_degrees(),
        (max_tilt_rad * ACTIVATION_LOWER).to_degrees(),
        max_tilt_rad.to_degrees(),
    );
    assert!(
        tilt_max <= max_tilt_rad * ACTIVATION_UPPER,
        "tilt constraint VIOLATED: peak {:.2}° > {:.2}° (110% of {:.2}°)",
        tilt_max.to_degrees(),
        (max_tilt_rad * ACTIVATION_UPPER).to_degrees(),
        max_tilt_rad.to_degrees(),
    );
}

#[test]
fn tilt_limit_sweep_shows_activation() {
    // Sweep multiple max_tilt values with a LOOSE velocity cap so tilt is
    // the binding constraint. Expect peak tilt to saturate at the limit as
    // it tightens.
    println!("\n  max_tilt    → peak_tilt   dur      status");
    for &max_tilt_deg in &[60.0_f32, 45.0, 30.0, 25.0, 20.0, 15.0] {
        let config = test_config_for_activation(15.0, max_tilt_deg);
        let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [5.0, 0.0, 1.0]);
        let result = plan(&input, &config);
        let (tilt_max, _) = peak_tilt_rad(&result, &config);
        let over_pct = 100.0 * (tilt_max - config.planner.max_tilt_rad) / config.planner.max_tilt_rad;
        println!(
            "  {:.1}°     →  {:.2}°   {:.3}s   {:?}   over: {:+.2}%",
            max_tilt_deg,
            tilt_max.to_degrees(),
            result.trajectory.total_duration(),
            result.status,
            over_pct
        );
    }
}

#[test]
fn tilt_constraint_respected() {
    let config = test_config();
    // A lateral translation forces significant tilt.
    let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [6.0, 0.0, 1.0]);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let max_tilt = config.planner.max_tilt_rad;
    // Tilt constraint: angle between body-z and world-z at most max_tilt.
    // With gravity compensation, cos(tilt) = (acc_z + g) / ‖α‖.
    let g = config.grav;
    let dur = result.trajectory.total_duration();
    let (tilt_max, t_at) = max_along(dur, 200, |t| {
        let a = result.trajectory.get_acc(t);
        let alpha = [a[0], a[1], a[2] + g];
        let n = vec_norm(alpha).max(1e-8);
        let cos_tilt = alpha[2] / n;
        cos_tilt.clamp(-1.0, 1.0).acos()
    });

    assert!(
        tilt_max <= max_tilt * CONSTRAINT_SLACK,
        "tilt {:.3} rad ({:.1}°) at t={t_at:.3} exceeds {:.3} rad ({:.1}°)",
        tilt_max,
        tilt_max.to_degrees(),
        max_tilt * CONSTRAINT_SLACK,
        (max_tilt * CONSTRAINT_SLACK).to_degrees(),
    );
}

#[test]
fn collective_thrust_constraint_respected() {
    let config = test_config();
    let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [4.0, 0.0, 2.0]);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let g = config.grav;
    let mass = config.mass;
    let min_n = config.min_collective_thrust_n;
    let max_n = config.max_collective_thrust_n;
    // Planner penalizes (F - F_mean)² > F_radius², i.e. F outside [min, max].
    let slack = (max_n - min_n) * (CONSTRAINT_SLACK - 1.0) * 0.5;
    let max_bound = max_n + slack;
    let min_bound = (min_n - slack).max(0.0);

    let dur = result.trajectory.total_duration();
    for i in 0..=200 {
        let t = dur * i as f32 / 200.0;
        let a = result.trajectory.get_acc(t);
        let alpha = [a[0], a[1], a[2] + g];
        let f = mass * vec_norm(alpha);
        assert!(
            f >= min_bound && f <= max_bound,
            "collective thrust {f:.3} N at t={t:.3} outside [{min_bound:.3}, {max_bound:.3}]"
        );
    }
}

#[test]
fn body_rate_constraint_respected() {
    let config = test_config();
    // A fast direction change stresses body rates.
    let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [3.0, 3.0, 1.0]);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let max_rate = config.max_rate_rad_s;
    let g = config.grav;
    let dur = result.trajectory.total_duration();

    // Body rates from differential flatness at ψ=0.
    // Planner bounds ‖ω_xy‖² ≤ max_rate_rad_s[0]² (pitch/roll scalar bound),
    // matching the C++ `maxOmgXY` semantics.
    let max_rate_xy = max_rate[0];
    let max_rate_z = max_rate[2];

    for i in 0..=200 {
        let t = dur * i as f32 / 200.0;
        let a = result.trajectory.get_acc(t);
        let j = result.trajectory.get_jerk(t);
        let alpha = [a[0], a[1], a[2] + g];
        let na = vec_norm(alpha).max(1e-8);
        let zb = [alpha[0] / na, alpha[1] / na, alpha[2] / na];
        // Skip near-inverted samples (model undefined).
        if zb[2] <= -0.9 {
            continue;
        }
        let dot_zj = zb[0] * j[0] + zb[1] * j[1] + zb[2] * j[2];
        let dzb = [
            (j[0] - zb[0] * dot_zj) / na,
            (j[1] - zb[1] * dot_zj) / na,
            (j[2] - zb[2] * dot_zj) / na,
        ];
        let s_inv = 1.0 / (1.0 + zb[2]).max(0.01);
        let omega = [
            -dzb[1] + s_inv * zb[1] * dzb[2],
            dzb[0] - s_inv * zb[0] * dzb[2],
            s_inv * (zb[1] * dzb[0] - zb[0] * dzb[1]),
        ];
        let omega_xy = (omega[0] * omega[0] + omega[1] * omega[1]).sqrt();
        let omega_z = omega[2].abs();

        assert!(
            omega_xy <= max_rate_xy * CONSTRAINT_SLACK,
            "body rate xy {omega_xy:.3} rad/s at t={t:.3} exceeds {:.3}",
            max_rate_xy * CONSTRAINT_SLACK
        );
        assert!(
            omega_z <= max_rate_z * CONSTRAINT_SLACK,
            "body rate z {omega_z:.3} rad/s at t={t:.3} exceeds {:.3}",
            max_rate_z * CONSTRAINT_SLACK
        );
    }
}

// ─── trajectory CSV export ──────────────────────────────────────────────────

/// Write a sampled trajectory to CSV in the same schema as `tof_x15.csv`:
/// t, p_x,p_y,p_z, q_x,q_y,q_z,q_w, v_x,v_y,v_z, w_x,w_y,w_z, a_lin_x,a_lin_y,a_lin_z,
/// a_rot_x,a_rot_y,a_rot_z, u_1,u_2,u_3,u_4, jerk_x,jerk_y,jerk_z,
/// snap_x,snap_y,snap_z, thrust
///
/// Quaternion and body rates are derived from differential flatness at ψ=0.
/// Angular acceleration is computed by finite-differencing ω.
/// Per-motor thrusts `u_i` are reported as the equal-split (collective/4),
/// since the planner uses a collective-thrust constraint (not per-motor).
fn write_trajectory_csv(
    result: &PlannerResult,
    config: &QuadPlanningConfig,
    path: &PathBuf,
    dt: f32,
) {
    let dur = result.trajectory.total_duration();
    let g = config.grav;
    let mass = config.mass;

    // Uniform sampling at dt intervals: t = 0, dt, 2·dt, ...
    // Clamp the final sample to exactly `dur` so the trajectory endpoint is included.
    let n_steps = (dur / dt).ceil() as usize;

    // First pass: compute per-sample state (omega needed for finite-difference a_rot).
    let mut rows: Vec<(f32, Vec3, Vec3, Vec3, Vec3, Vec3, [f32; 4], Vec3, f32)> =
        Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        let t = (i as f32 * dt).min(dur);
        let p = result.trajectory.get_pos(t);
        let v = result.trajectory.get_vel(t);
        let a = result.trajectory.get_acc(t);
        let j = result.trajectory.get_jerk(t);
        let s = result.trajectory.get_snap(t);

        // Thrust vector and body z-axis.
        let alpha = [a[0], a[1], a[2] + g];
        let na = (alpha[0] * alpha[0] + alpha[1] * alpha[1] + alpha[2] * alpha[2])
            .sqrt()
            .max(1e-8);
        let inv_na = 1.0 / na;
        let zb = [alpha[0] * inv_na, alpha[1] * inv_na, alpha[2] * inv_na];

        // Quaternion (tilt only, ψ=0):
        //   q_w = √(2(1+zb_z))/2,  q_x = -zb_y/√(2(1+zb_z)),
        //   q_y =  zb_x/√(2(1+zb_z)), q_z = 0.
        let zb2_1 = (1.0 + zb[2]).max(1e-6);
        let tilt_den = (2.0 * zb2_1).sqrt();
        let qw = 0.5 * tilt_den;
        let qx = -zb[1] / tilt_den;
        let qy = zb[0] / tilt_den;
        let qz = 0.0;
        let quat = [qx, qy, qz, qw];

        // Body rates from flatness at ψ=0.
        let omega = if zb[2] > -0.9 {
            let dot_zj = zb[0] * j[0] + zb[1] * j[1] + zb[2] * j[2];
            let dzb = [
                (j[0] - zb[0] * dot_zj) * inv_na,
                (j[1] - zb[1] * dot_zj) * inv_na,
                (j[2] - zb[2] * dot_zj) * inv_na,
            ];
            let s_inv = 1.0 / (1.0 + zb[2]).max(0.01);
            [
                -dzb[1] + s_inv * zb[1] * dzb[2],
                dzb[0] - s_inv * zb[0] * dzb[2],
                s_inv * (zb[1] * dzb[0] - zb[0] * dzb[1]),
            ]
        } else {
            [0.0; 3]
        };

        let thrust = mass * na;
        rows.push((t, p, v, a, j, s, quat, omega, thrust));
    }

    // Second pass: finite-difference angular acceleration and emit CSV.
    let mut csv = String::new();
    csv.push_str(
        "t,p_x,p_y,p_z,q_x,q_y,q_z,q_w,v_x,v_y,v_z,w_x,w_y,w_z,\
         a_lin_x,a_lin_y,a_lin_z,a_rot_x,a_rot_y,a_rot_z,\
         u_1,u_2,u_3,u_4,jerk_x,jerk_y,jerk_z,snap_x,snap_y,snap_z,thrust\n",
    );
    for k in 0..rows.len() {
        let (t, p, v, a, j, s, q, omega, thrust) = rows[k];
        let a_rot = if k == 0 {
            [0.0f32; 3]
        } else {
            let (t_prev, _, _, _, _, _, _, om_prev, _) = rows[k - 1];
            let dt = (t - t_prev).max(1e-9);
            [
                (omega[0] - om_prev[0]) / dt,
                (omega[1] - om_prev[1]) / dt,
                (omega[2] - om_prev[2]) / dt,
            ]
        };
        let u = thrust / 4.0;
        csv.push_str(&format!(
            "{t:.6},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},\
             {:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},\
             {:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},\
             {:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},\
             {:.6e},{:.6e},{:.6e},{:.6e}\n",
            p[0], p[1], p[2],
            q[0], q[1], q[2], q[3],
            v[0], v[1], v[2],
            omega[0], omega[1], omega[2],
            a[0], a[1], a[2],
            a_rot[0], a_rot[1], a_rot[2],
            u, u, u, u,
            j[0], j[1], j[2],
            s[0], s[1], s[2],
            thrust,
        ));
    }

    fs::write(path, csv).expect("failed to write trajectory CSV");
}

/// Return `<CARGO_TARGET_TMPDIR>/<name>` (created on demand).
fn output_path(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&dir).ok();
    dir.join(name)
}

#[test]
fn save_goto_trajectory_csv() {
    let config = test_config();
    let input = PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [3.0, 0.0, 1.0]);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let path = output_path("planner_goto.csv");
    write_trajectory_csv(&result, &config, &path, 0.01);
    println!("wrote goto trajectory: {}", path.display());
    assert!(path.exists());
}

#[test]
fn save_waypoints_trajectory_csv() {
    let config = test_config();
    let start = [0.0, 0.0, 1.0];
    let targets: [Vec3; 3] = [[2.0, 1.0, 1.5], [4.0, -1.0, 2.0], [6.0, 0.0, 1.0]];
    let input = PlannerInput::waypoints(start, ZERO3, &targets);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let path = output_path("planner_waypoints.csv");
    write_trajectory_csv(&result, &config, &path, 0.01);
    println!("wrote waypoint trajectory: {}", path.display());
    assert!(path.exists());
}

/// Closed-loop stress path: start = end = [-2, -2, 1], visit a 4-point
/// radius-1.8 circle three times (12 intermediate waypoints → 13 pieces).
fn circular_input() -> PlannerInput {
    let start = [-2.0, -2.0, 1.0];
    let a = [-1.8, 0.0, 1.0];
    let b = [0.0, 1.8, 1.0];
    let c = [1.8, 0.0, 1.0];
    let d = [0.0, -1.8, 1.0];
    // 13 targets: 12 intermediate waypoints + return-to-start as the tail.
    let targets: [Vec3; 13] = [a, b, c, d, a, b, c, d, a, b, c, d, start];
    PlannerInput::waypoints(start, ZERO3, &targets)
}

#[test]
fn save_circular_waypoints_trajectory_csv() {
    // Baseline circular trajectory with default weight_time / weight_energy.
    let config = test_config();
    let input = circular_input();
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);
    assert_eq!(result.num_pieces, 13);

    let path = output_path("planner_circular.csv");
    write_trajectory_csv(&result, &config, &path, 0.01);
    println!("wrote circular trajectory: {}", path.display());
    assert!(path.exists());
}

#[test]
fn save_circular_time_heavy_trajectory_csv() {
    // weight_time = 1.0, weight_energy = 0.0:
    // Fully time-minimizing (no smoothness regularizer). The optimizer pushes
    // segment durations down as far as the dynamic constraints allow.
    // Expect: shortest total duration, largest jerks at constraint boundaries.
    let config = test_config_with_time_energy(1.0, 0.0);
    let input = circular_input();
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);
    assert_eq!(result.num_pieces, 13);

    let path = output_path("planner_circular_time_heavy.csv");
    write_trajectory_csv(&result, &config, &path, 0.01);
    println!(
        "wrote time-heavy (wt=1, we=0) trajectory: dur={:.3}s  cost={:.3}  path={}",
        result.trajectory.total_duration(),
        result.final_cost,
        path.display()
    );
    assert!(path.exists());
}

#[test]
fn save_circular_energy_heavy_trajectory_csv() {
    // weight_time = 0.1, weight_energy = 1.0:
    // Smoothness-dominated. The optimizer accepts longer durations to keep
    // ∫‖jerk‖² small. Expect: longer total duration, much smoother profile.
    let config = test_config_with_time_energy(0.1, 1.0);
    let input = circular_input();
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);
    assert_eq!(result.num_pieces, 13);

    let path = output_path("planner_circular_energy_heavy.csv");
    write_trajectory_csv(&result, &config, &path, 0.01);
    println!(
        "wrote energy-heavy (wt=0.1, we=1) trajectory: dur={:.3}s  cost={:.3}  path={}",
        result.trajectory.total_duration(),
        result.final_cost,
        path.display()
    );
    assert!(path.exists());
}

#[test]
fn trajectory_is_finite_everywhere() {
    // Defensive: ensure no NaN/Inf anywhere in pos/vel/acc/jerk for a
    // non-trivial multi-waypoint path.
    let config = test_config();
    let start = [0.0, 0.0, 1.0];
    let targets: [Vec3; 4] = [
        [1.0, 2.0, 1.5],
        [3.0, -1.0, 2.0],
        [5.0, 1.0, 1.5],
        [6.0, 0.0, 1.0],
    ];
    let input = PlannerInput::waypoints(start, ZERO3, &targets);
    let result = plan(&input, &config);
    assert_converged(result.status, result.final_cost);

    let dur = result.trajectory.total_duration();
    for i in 0..=500 {
        let t = dur * i as f32 / 500.0;
        let p = result.trajectory.get_pos(t);
        let v = result.trajectory.get_vel(t);
        let a = result.trajectory.get_acc(t);
        let j = result.trajectory.get_jerk(t);
        for d in 0..3 {
            assert!(p[d].is_finite(), "pos NaN at t={t}");
            assert!(v[d].is_finite(), "vel NaN at t={t}");
            assert!(a[d].is_finite(), "acc NaN at t={t}");
            assert!(j[d].is_finite(), "jerk NaN at t={t}");
        }
    }
}
