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
#[ignore] // benchmark — run with `--release --ignored`
fn bench_runtime_across_tasks() {
    // Comprehensive runtime evaluation across task complexity, using the
    // NEW default parameterization (ball-shape radius 0.01 m, we=0.01).
    //
    //   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
    //     --release --test planner_convergence bench_runtime_across_tasks \
    //     -- --nocapture --ignored
    use std::time::Instant;

    fn make_bench_config() -> QuadPlanningConfig {
        // Use PlannerParams::default() to verify the new defaults work.
        let mut vp = test_vehicle_params();
        vp.planner.max_vel_m_s = 4.0;
        vp.planner.max_tilt_rad = 60_f32.to_radians();
        vp.planner.weight_vel = 50.0;
        vp.planner.weight_tilt = 50.0;
        vp.planner.weight_body_rate = 50.0;
        vp.planner.weight_thrust = 10.0;
        // weight_energy = 0.01 and BFGS defaults come from PlannerParams::default()
        // via VehicleParams default. Only override max_iterations for headroom.
        vp.planner.bfgs_trust.max_iterations = 2000;
        QuadPlanningConfig::from_vehicle_params(&vp)
    }

    fn time_task<F: Fn() -> PlannerInput>(
        label: &str,
        n_pieces: usize,
        make_input: F,
    ) -> (f64, f64, f64, f64, usize, f32) {
        let config = make_bench_config();
        let _ = plan(&make_input(), &config); // warm-up

        let n_samples = 50;
        let mut durations_ms = Vec::with_capacity(n_samples);
        let mut last_iters = 0;
        let mut last_traj_dur = 0.0;
        for _ in 0..n_samples {
            let input = make_input();
            let t0 = Instant::now();
            let result = plan(&input, &config);
            durations_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_iters = result.iterations;
            last_traj_dur = result.trajectory.total_duration();
        }
        durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p10 = durations_ms[n_samples / 10];
        let p50 = durations_ms[n_samples / 2];
        let p90 = durations_ms[(n_samples * 9) / 10];
        let p99 = durations_ms[(n_samples * 99) / 100];
        let _ = label;
        let _ = n_pieces;
        (p10, p50, p90, p99, last_iters, last_traj_dur)
    }

    println!(
        "\n=== Planner runtime evaluation (defaults: ball_radius=0.01m, we=0.01) ==="
    );
    println!("Weights: w_time=1, w_energy=0.01, w_vel=50, w_tilt=50, w_rate=50, w_thr=10");
    println!("Limits:  max_vel=4.0 m/s, max_tilt=60°");
    println!("Desktop x86 release build, 50 samples per task.\n");
    println!(
        "  {:<28} {:>3}  {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}",
        "task", "pc", "iters", "p10", "p50", "p90", "p99", "us/it", "traj(s)"
    );
    println!(
        "  {:<28} {:>3}  {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}",
        "", "", "", "(ms)", "(ms)", "(ms)", "(ms)", "(us)", ""
    );

    struct Task {
        label: &'static str,
        n_pieces: usize,
        p50_ms: f64,
    }
    let mut tasks: Vec<Task> = Vec::new();

    macro_rules! run {
        ($label:expr, $n_pieces:expr, $input:expr) => {{
            let (p10, p50, p90, p99, iters, traj) = time_task($label, $n_pieces, $input);
            let us_it = if iters > 0 { p50 * 1000.0 / iters as f64 } else { 0.0 };
            println!(
                "  {:<28} {:>3}  {:>5}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.1}  {:>8.3}",
                $label, $n_pieces, iters, p10, p50, p90, p99, us_it, traj
            );
            tasks.push(Task { label: $label, n_pieces: $n_pieces, p50_ms: p50 });
        }};
    }

    // 1 piece: simple goto (no waypoints)
    run!("goto 3m", 1, || {
        PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [3.0, 0.0, 1.0])
    });

    // 1 piece: slightly more demanding goto
    run!("goto 5m", 1, || {
        PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [5.0, 0.0, 1.0])
    });

    // 2 pieces: single intermediate waypoint
    run!("waypoints (1 wp)", 2, || {
        PlannerInput::waypoints(
            [0.0, 0.0, 1.0], ZERO3,
            &[[2.0, 1.0, 1.0], [4.0, 0.0, 1.0]])
    });

    // 3 pieces: zig-zag
    run!("waypoints (2 wp zig-zag)", 3, || {
        PlannerInput::waypoints(
            [0.0, 0.0, 1.0], ZERO3,
            &[[2.0, 1.0, 1.0], [4.0, -1.0, 1.0], [6.0, 0.0, 1.0]])
    });

    // 5 pieces
    run!("waypoints (4 wp)", 5, || {
        let pts: [Vec3; 5] = [
            [1.0, 1.0, 1.0], [2.0, 0.0, 1.0],
            [3.0, -1.0, 1.0], [4.0, 0.0, 1.0], [5.0, 1.0, 1.0],
        ];
        PlannerInput::waypoints([0.0, 0.0, 1.0], ZERO3, &pts)
    });

    // 8 pieces: medium
    run!("waypoints (7 wp)", 8, || {
        let pts: [Vec3; 7] = [
            [1.0, 1.0, 1.0], [2.0, -1.0, 1.2], [3.0, 1.0, 1.0],
            [4.0, -1.0, 0.8], [5.0, 1.0, 1.0], [6.0, -1.0, 1.2], [7.0, 0.0, 1.0],
        ];
        PlannerInput::waypoints([0.0, 0.0, 1.0], ZERO3, &pts)
    });

    // 13 pieces: circular closed loop
    run!("circular (12 wp)", 13, || circular_input());

    // STM32H7 scaled estimates.
    const STM32_FACTOR: f64 = 35.0;
    println!("\nSTM32H7 @ 480 MHz estimates (×{:.0} typical slowdown):", STM32_FACTOR);
    println!(
        "  {:<28} {:>3}  {:>10}  {:>10}  {:>12}",
        "task", "pc", "p50 (ms)", "p99 (ms)", "max rate (Hz)"
    );
    for t in &tasks {
        let stm = t.p50_ms * STM32_FACTOR;
        let max_rate_hz = 1000.0 / stm;
        println!(
            "  {:<28} {:>3}  {:>10.2}  {:>10.2}  {:>12.1}",
            t.label, t.n_pieces, stm, stm * 1.3, max_rate_hz
        );
    }
}

#[test]
#[ignore] // benchmark — run with `--release --ignored`
fn bench_runtime_vs_weight_energy() {
    // Sweep weight_energy ∈ [0, 1.0] (with weight_time fixed at 1.0) and
    // measure planner runtime on three problem sizes.
    //
    //   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
    //     --release --test planner_convergence bench_runtime_vs_weight_energy \
    //     -- --nocapture --ignored
    use std::time::Instant;

    fn make_bench_config(weight_energy: f32) -> QuadPlanningConfig {
        let mut vp = test_vehicle_params();
        vp.planner.max_vel_m_s = 4.0;
        vp.planner.max_tilt_rad = 60_f32.to_radians();
        vp.planner.weight_vel = 50.0;
        vp.planner.weight_tilt = 50.0;
        vp.planner.weight_body_rate = 50.0;
        vp.planner.weight_thrust = 10.0;
        vp.planner.weight_energy = weight_energy;
        vp.planner.weight_time = 1.0;
        vp.planner.smoothing_eps = 0.01;
        vp.planner.num_check_per_piece = 8;
        // Large budget so every case can fully converge (or hit the ceiling).
        vp.planner.bfgs_trust.max_iterations = 2000;
        QuadPlanningConfig::from_vehicle_params(&vp)
    }

    struct Row {
        we: f32,
        p50_ms: f64,
        mean_ms: f64,
        iters: usize,
        us_per_iter: f64,
        traj_dur: f32,
        final_cost: f32,
    }

    fn time_case<F: Fn() -> PlannerInput>(
        label: &str,
        make_input: F,
        weight_energies: &[f32],
    ) {
        let n_samples = 30;
        let mut rows: Vec<Row> = Vec::new();
        for &we in weight_energies {
            let config = make_bench_config(we);
            let _ = plan(&make_input(), &config); // warm-up

            let mut durations_ms = Vec::with_capacity(n_samples);
            let mut last_iters = 0;
            let mut last_dur = 0.0;
            let mut last_cost = 0.0;
            for _ in 0..n_samples {
                let input = make_input();
                let t0 = Instant::now();
                let result = plan(&input, &config);
                durations_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                last_iters = result.iterations;
                last_dur = result.trajectory.total_duration();
                last_cost = result.final_cost;
            }
            durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = durations_ms[n_samples / 2];
            let mean: f64 = durations_ms.iter().sum::<f64>() / n_samples as f64;
            let us_per_iter = (p50 / last_iters.max(1) as f64) * 1000.0;
            rows.push(Row {
                we,
                p50_ms: p50,
                mean_ms: mean,
                iters: last_iters,
                us_per_iter,
                traj_dur: last_dur,
                final_cost: last_cost,
            });
        }

        println!("\n=== {label} — weight_time=1.0, sweep weight_energy ===");
        println!(
            "  {:<8}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}",
            "we", "iters", "p50(ms)", "mean(ms)", "us/iter", "traj(s)", "cost"
        );
        for r in &rows {
            println!(
                "  {:<8.4}  {:>5}  {:>8.3}  {:>8.3}  {:>8.2}  {:>8.3}  {:>9.3}",
                r.we, r.iters, r.p50_ms, r.mean_ms, r.us_per_iter, r.traj_dur, r.final_cost
            );
        }
    }

    let we_values: &[f32] = &[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0];

    // 1 piece: goto
    time_case(
        "GOTO 3m (1 piece)",
        || PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [3.0, 0.0, 1.0]),
        we_values,
    );

    // 5 pieces: 4-waypoint path
    time_case(
        "WAYPOINTS (5 pieces)",
        || {
            let pts: [Vec3; 5] = [
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 1.0],
                [3.0, -1.0, 1.0],
                [4.0, 0.0, 1.0],
                [5.0, 1.0, 1.0],
            ];
            PlannerInput::waypoints([0.0, 0.0, 1.0], ZERO3, &pts)
        },
        we_values,
    );

    // 13 pieces: circular closed loop
    time_case(
        "CIRCULAR (13 pieces)",
        || circular_input(),
        we_values,
    );

    println!(
        "\nNote: with we > 0 the waypoint D-variables drift (the Rust port has no \
         corridor/ball anchoring), so the \"traj(s)\" column for we > 0 often \
         reflects a collapsed path rather than the intended traversal. This \
         benchmark measures runtime only — constraint satisfaction at we > 0 \
         should not be inferred from the trajectory duration."
    );
    println!(
        "STM32H7 @ 480 MHz estimate: multiply desktop p50 by ~35 for typical workload."
    );
}

#[test]
#[ignore] // benchmark — run with `--release --ignored`
fn bench_planner_scaling_by_piece_count() {
    // Measure planner runtime vs. number of trajectory pieces.
    // Runs each problem to FULL convergence (max_iter=2000) so we capture
    // the actual workload, not a fixed-budget cutoff.
    //
    //   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
    //     --release --test planner_convergence bench_planner_scaling \
    //     -- --nocapture --ignored
    use std::time::Instant;

    fn make_bench_config() -> QuadPlanningConfig {
        let mut vp = test_vehicle_params();
        vp.planner.max_vel_m_s = 4.0;
        vp.planner.max_tilt_rad = 60_f32.to_radians();
        vp.planner.weight_vel = 50.0;
        vp.planner.weight_tilt = 50.0;
        vp.planner.weight_body_rate = 50.0;
        vp.planner.weight_thrust = 10.0;
        vp.planner.weight_energy = 0.1;
        vp.planner.weight_time = 1.0;
        vp.planner.smoothing_eps = 0.01;
        vp.planner.num_check_per_piece = 8;
        vp.planner.bfgs_trust.max_iterations = 2000;
        QuadPlanningConfig::from_vehicle_params(&vp)
    }

    fn time_case<F: Fn() -> PlannerInput>(
        label: &str,
        n_pieces: usize,
        make_input: F,
    ) {
        let config = make_bench_config();
        let _ = plan(&make_input(), &config); // warm-up
        let n_samples = 50;
        let mut durations_ms = Vec::with_capacity(n_samples);
        let mut iter_counts = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let input = make_input();
            let t0 = Instant::now();
            let result = plan(&input, &config);
            durations_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            iter_counts.push(result.iterations);
        }
        durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = durations_ms[n_samples / 2];
        let p99 = durations_ms[(n_samples * 99) / 100];
        let mean: f64 = durations_ms.iter().sum::<f64>() / n_samples as f64;
        let mean_iters: f64 =
            iter_counts.iter().map(|&x| x as f64).sum::<f64>() / n_samples as f64;
        let us_per_iter = (p50 / mean_iters.max(1.0)) * 1000.0;
        println!(
            "  {:<18} {:>2} pc   iters={:>5.0}   mean={:>6.2}ms  p50={:>6.2}ms  p99={:>6.2}ms  {:>5.1} us/iter",
            label, n_pieces, mean_iters, mean, p50, p99, us_per_iter
        );
    }

    println!("\n=== Planner scaling: pieces vs. wall-clock ===");
    println!(" (desktop x86 release build, full convergence)\n");
    println!("  problem           n  pieces   iters        mean       p50       p99    per-iter");

    // 1 piece: simple goto
    time_case("goto 3m", 1, || {
        PlannerInput::goto([0.0, 0.0, 1.0], ZERO3, [3.0, 0.0, 1.0])
    });

    // 2 pieces: 1 intermediate waypoint
    time_case("waypoints (1 wp)", 2, || {
        PlannerInput::waypoints(
            [0.0, 0.0, 1.0],
            ZERO3,
            &[[2.0, 1.0, 1.0], [4.0, 0.0, 1.0]],
        )
    });

    // 3 pieces: 2 intermediate waypoints
    time_case("waypoints (2 wp)", 3, || {
        PlannerInput::waypoints(
            [0.0, 0.0, 1.0],
            ZERO3,
            &[[2.0, 1.0, 1.0], [4.0, -1.0, 1.0], [6.0, 0.0, 1.0]],
        )
    });

    // 5 pieces: 4 intermediate waypoints
    time_case("waypoints (4 wp)", 5, || {
        let pts: [Vec3; 5] = [
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, -1.0, 1.0],
            [4.0, 0.0, 1.0],
            [5.0, 1.0, 1.0],
        ];
        PlannerInput::waypoints([0.0, 0.0, 1.0], ZERO3, &pts)
    });

    // 13 pieces: the full circular closed loop
    time_case("circular (12 wp)", 13, || circular_input());

    println!();
    println!("STM32H7 @ 480 MHz scaled estimates (using 35× typical slowdown):");
    println!("  (multiply p50 above by 35 — per-iter costs by 35× too)");
}

#[test]
#[ignore] // benchmark — run with `--release --ignored`
fn bench_circular_planner_runtime() {
    // Measure planner runtime on the 13-piece closed-loop circular path,
    // across multiple iteration budgets to characterize per-iteration cost.
    //
    // Run with:
    //   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
    //     --release --test planner_convergence bench_circular \
    //     -- --nocapture --ignored
    use std::time::Instant;

    fn time_one_budget(max_iters: usize, n_samples: usize) -> (f64, f64, f64, usize) {
        // Build a config with the given iteration cap.
        let mut vp = test_vehicle_params();
        vp.planner.max_vel_m_s = 4.0;
        vp.planner.max_tilt_rad = 60_f32.to_radians();
        vp.planner.weight_vel = 50.0;
        vp.planner.weight_tilt = 50.0;
        vp.planner.weight_body_rate = 50.0;
        vp.planner.weight_thrust = 10.0;
        vp.planner.weight_energy = 0.1;
        vp.planner.weight_time = 1.0;
        vp.planner.smoothing_eps = 0.01;
        vp.planner.num_check_per_piece = 8;
        vp.planner.bfgs_trust.max_iterations = max_iters;
        let config = QuadPlanningConfig::from_vehicle_params(&vp);

        // Warm-up.
        let _ = plan(&circular_input(), &config);

        let mut durations_ms = Vec::with_capacity(n_samples);
        let mut iter_counts = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let input = circular_input();
            let t0 = Instant::now();
            let result = plan(&input, &config);
            durations_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            iter_counts.push(result.iterations);
        }
        durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f64 = durations_ms.iter().sum::<f64>() / n_samples as f64;
        let p50 = durations_ms[n_samples / 2];
        let p99 = durations_ms[(n_samples * 99) / 100];
        let mean_iters = iter_counts.iter().sum::<usize>() / iter_counts.len();
        (mean, p50, p99, mean_iters)
    }

    println!("\n=== Planner runtime: 13-piece circular waypoint path ===");
    println!("desktop x86 (release build, f32, no AVX explicit):");
    println!(" max_iter  actual_iters  mean (ms)  p50 (ms)  p99 (ms)  us/iter");
    let budgets = [50_usize, 100, 200, 500, 1000];
    let mut per_iter_ms_samples = Vec::new();
    for &budget in &budgets {
        let (mean, p50, p99, mean_iters) = time_one_budget(budget, 30);
        let us_per_iter = (p50 / mean_iters.max(1) as f64) * 1000.0;
        println!(
            "  {:>5}    {:>5}        {:>7.2}   {:>7.2}   {:>7.2}   {:>5.1}",
            budget, mean_iters, mean, p50, p99, us_per_iter
        );
        // Only sample per-iter from budgets that actually hit the cap
        // (so the total time accurately reflects the cap).
        if mean_iters == budget {
            per_iter_ms_samples.push(p50 / mean_iters as f64);
        }
    }

    // Derive per-iteration cost from the capped runs.
    let per_iter_ms: f64 = per_iter_ms_samples.iter().sum::<f64>()
        / per_iter_ms_samples.len() as f64;

    println!("\nderived per-BFGS-iteration cost: {:.3} ms ({:.1} μs)",
        per_iter_ms, per_iter_ms * 1000.0);

    // STM32H7 scaling estimate.
    //
    // Cortex-M7 @ 480 MHz: single-lane scalar FPU, f32 mul/add ~1/cycle,
    // f32 div ~14 cycles, f32 sqrt ~14 cycles. No SIMD, no FMA.
    // Branch predictor exists but small; caches are 16 KB L1I / 16 KB L1D.
    //
    // Desktop x86 with opt-level=3 (no explicit AVX here): ~3-4 GHz, scalar
    // f32 ops through SSE2 (~1 FLOP/cycle effective), but with aggressive
    // out-of-order execution and much larger caches.
    //
    // Observed slowdowns for float-heavy no_std Rust on STM32H7:
    //   - simple hot loops (cache-resident, straight-line):  15-25×
    //   - mixed arithmetic with branches & divisions:        25-50×
    //   - memory-bound (banded system >4KB):                 40-80×
    //
    // The planner has:
    //   - A 78×78 banded LU factorize + solve per BFGS iter (MincoJerk)
    //   - 13 × 9 = 117 sample points with flatness chain per iter
    //   - Several f32 divisions and sqrts (inv_norm_alpha, s_inv, etc.)
    //   - BFGS also does a 52×52 Cholesky per iter (dim_k + dim_d = 13+36)
    // So the workload is mixed — estimate ~30× typical, with caveat that the
    // BFGS Hessian (52×52 = 10KB of f32) likely spills out of M7 L1D.
    println!("\nSTM32H7 @ 480 MHz scaled estimate (f32 scalar FPU):");
    for &(label, factor) in &[
        ("optimistic (×20)", 20.0_f64),
        ("typical    (×35)", 35.0),
        ("worst      (×60)", 60.0),
    ] {
        let per_iter_stm = per_iter_ms * factor;
        println!(
            "  {}:  {:.2} ms/iter  →  200 iters = {:.0} ms, 500 iters = {:.0} ms",
            label,
            per_iter_stm,
            per_iter_stm * 200.0,
            per_iter_stm * 500.0
        );
    }
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

// ─── PRODUCTION-REGIME TORTURE TESTS ────────────────────────────────────────
//
// Reproduce the live `PlannerParams` used by `crates/cybflight/src/control/
// mission_planner.rs` — no velocity / tilt / position penalties, only
// body-rate + thrust soft penalties against `weight_time = 1.0`, and the
// actual 15-waypoint circuit the firmware issues.
//
// This exercises the regime where flight tests show the planner publishing
// degenerate trajectories: MaxIterations status with over-compressed early
// segment times → MPC thrashes the Z-axis at the start of execution.

/// Mirror of `PlannerParams::default()` in `crates/cybflight_core/src/params.rs`.
/// Keep in sync with the production defaults.
fn production_regime_config() -> QuadPlanningConfig {
    let mut vp = test_vehicle_params();
    vp.planner.max_vel_m_s = 5.0;
    vp.planner.max_tilt_rad = core::f32::consts::FRAC_PI_3;
    vp.planner.weight_time = 1.0;
    vp.planner.weight_energy = 0.0;
    vp.planner.weight_pos = 0.0;
    vp.planner.weight_vel = 0.0;
    vp.planner.weight_tilt = 0.0;
    vp.planner.weight_body_rate = 10.0;
    vp.planner.weight_thrust = 10.0;
    vp.planner.smoothing_eps = 0.01;
    vp.planner.num_check_per_piece = 8;
    vp.planner.bfgs_trust.max_iterations = 500;
    QuadPlanningConfig::from_vehicle_params(&vp)
}

/// The exact target list hardcoded in `mission_planner.rs` (~line 263).
const PRODUCTION_WAYPOINTS: [Vec3; 15] = [
    [-0.3267, -2.231, 1.6],
    [-1.845, 1.942, 1.0],
    [2.292, 1.637, 1.0],
    [2.547, -2.108, 1.8],
    [2.547, -2.108, 0.8],
    [0.3099, 0.3554, 1.0],
    [-2.396, -2.214, 1.0],
    [-0.3267, -2.231, 1.6],
    [-1.845, 1.942, 1.0],
    [2.292, 1.637, 1.0],
    [2.547, -2.108, 1.8],
    [2.547, -2.108, 0.8],
    [0.3099, 0.3554, 1.0],
    [-2.396, -2.214, 1.0],
    [-0.3267, -2.231, 1.6],
];

/// Nominal hover seed used when the firmware issues a plan request.
const PRODUCTION_START: Vec3 = [0.0, 0.0, 1.0];

/// Summary of kinematic peaks sampled densely across a trajectory.
struct KinematicPeaks {
    v_max: f32,
    a_max: f32,
    omega_xy_max: f32,
    omega_z_max: f32,
}

fn sample_kinematic_peaks(
    result: &PlannerResult,
    config: &QuadPlanningConfig,
) -> KinematicPeaks {
    let dur = result.trajectory.total_duration();
    let g = config.grav;
    let n = 1000;
    let mut v_max = 0.0f32;
    let mut a_max = 0.0f32;
    let mut omega_xy_max = 0.0f32;
    let mut omega_z_max = 0.0f32;
    for i in 0..=n {
        let t = dur * i as f32 / n as f32;
        let v = result.trajectory.get_vel(t);
        let a = result.trajectory.get_acc(t);
        let j = result.trajectory.get_jerk(t);
        v_max = v_max.max(vec_norm(v));
        a_max = a_max.max(vec_norm(a));
        let alpha = [a[0], a[1], a[2] + g];
        let na = vec_norm(alpha).max(1e-8);
        let zb = [alpha[0] / na, alpha[1] / na, alpha[2] / na];
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
        omega_xy_max = omega_xy_max.max(omega_xy);
        omega_z_max = omega_z_max.max(omega[2].abs());
    }
    KinematicPeaks {
        v_max,
        a_max,
        omega_xy_max,
        omega_z_max,
    }
}

fn print_piece_diagnostics(label: &str, result: &PlannerResult, config: &QuadPlanningConfig) {
    let n = result.num_pieces;
    let times = &result.optimized_times[..n];
    let min_t = times.iter().copied().fold(f32::INFINITY, f32::min);
    let max_t = times.iter().copied().fold(0.0f32, f32::max);
    let peaks = sample_kinematic_peaks(result, config);
    println!(
        "  [{label}] status={:?} iters={} pieces={} dur={:.3}s cost={:.3}",
        result.status, result.iterations, n, result.trajectory.total_duration(),
        result.final_cost,
    );
    println!(
        "    per-piece T: min={:.3}s max={:.3}s  peaks: v={:.2}m/s a={:.2}m/s² ωxy={:.2}rad/s ωz={:.2}rad/s",
        min_t, max_t, peaks.v_max, peaks.a_max, peaks.omega_xy_max, peaks.omega_z_max,
    );
    print!("    times=[");
    for (i, t) in times.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.2}", t);
    }
    println!("]");
}

/// Baseline: the firmware's exact plan request (15 targets → 15 pieces)
/// must converge cleanly. If BFGS returns `MaxIterations`, the published
/// trajectory carries unresolved body-rate/thrust penalties and sends the
/// drone into the failure mode observed in flight.
#[test]
fn production_regime_full_circuit_converges() {
    let config = production_regime_config();
    let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
    let result = plan(&input, &config);
    print_piece_diagnostics("full 15wp", &result, &config);
    assert!(result.final_cost.is_finite(), "non-finite cost");
    assert!(
        matches!(result.status, SolverStatus::Convergence | SolverStatus::Stop),
        "under-converged: {:?} at iter {}",
        result.status, result.iterations,
    );
}

/// If the min segment time collapses below 0.3 s, the piece's polynomial
/// commands accelerations and jerks that the MPC cannot realize at 100 Hz.
/// This is the direct signature of "jumps up and down at start" failures.
#[test]
fn production_regime_min_segment_time_feasible() {
    let config = production_regime_config();
    let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
    let result = plan(&input, &config);
    print_piece_diagnostics("min-T gate", &result, &config);
    let n = result.num_pieces;
    let min_t = result.optimized_times[..n]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(
        min_t >= 0.3,
        "min segment time {min_t:.3}s < 0.3s — trajectory over-compressed",
    );
}

/// The trajectory must be physically realizable: velocity under the 5 m/s
/// planner bound with a 15% slack, body rates under the vehicle limits.
/// These are the soft-constraint knees the solver is supposed to honor.
#[test]
fn production_regime_kinematics_feasible() {
    let config = production_regime_config();
    let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
    let result = plan(&input, &config);
    print_piece_diagnostics("kinematics", &result, &config);
    let peaks = sample_kinematic_peaks(&result, &config);
    let max_vel = config.planner.max_vel_m_s * CONSTRAINT_SLACK;
    let max_omega_xy = config.max_rate_rad_s[0] * CONSTRAINT_SLACK;
    let max_omega_z = config.max_rate_rad_s[2] * CONSTRAINT_SLACK;
    assert!(
        peaks.v_max <= max_vel,
        "peak velocity {:.2} > {:.2} (max_vel {} × {})",
        peaks.v_max, max_vel, config.planner.max_vel_m_s, CONSTRAINT_SLACK,
    );
    assert!(
        peaks.omega_xy_max <= max_omega_xy,
        "peak ω_xy {:.2} > {:.2} (limit {} × {})",
        peaks.omega_xy_max, max_omega_xy, config.max_rate_rad_s[0], CONSTRAINT_SLACK,
    );
    assert!(
        peaks.omega_z_max <= max_omega_z,
        "peak ω_z {:.2} > {:.2} (limit {} × {})",
        peaks.omega_z_max, max_omega_z, config.max_rate_rad_s[2], CONSTRAINT_SLACK,
    );
}

/// Walk the prefix of PRODUCTION_WAYPOINTS and report how the solver
/// behaves as the piece count grows. Surfaces the exact count at which
/// convergence / segment-time feasibility break under production weights.
#[test]
#[ignore] // diagnostic — run with `--release --ignored -- --nocapture`
fn production_regime_piece_count_sweep() {
    let config = production_regime_config();
    println!(
        "\n=== Production-regime sweep: n targets → n pieces (wv=0, wtilt=0, we=0) ==="
    );
    for n_targets in 1..=PRODUCTION_WAYPOINTS.len() {
        let targets = &PRODUCTION_WAYPOINTS[..n_targets];
        let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, targets);
        let result = plan(&input, &config);
        print_piece_diagnostics(&format!("{n_targets} tgts"), &result, &config);
    }
}

/// Repeat-seed stability test: re-plan the same circuit many times. Any
/// dependence of the outcome on float accumulation order (e.g. a stochastic
/// trust-region trigger) would show up as differing iterations/status.
/// In the production regime at high piece count, "converged" runs
/// intermixed with "MaxIterations" runs from identical inputs is the
/// hallmark of sitting on a convergence knife-edge.
#[test]
#[ignore] // diagnostic — run with `--release --ignored -- --nocapture`
fn production_regime_repeatability() {
    let config = production_regime_config();
    println!("\n=== Production-regime repeatability (20 runs, same inputs) ===");
    let mut statuses = [0usize; 5]; // Convergence, Stop, MaxIterations, InvalidValue, TimeExceeded
    for run in 0..20 {
        let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
        let result = plan(&input, &config);
        let idx = match result.status {
            SolverStatus::Convergence => 0,
            SolverStatus::Stop => 1,
            SolverStatus::MaxIterations => 2,
            SolverStatus::InvalidValue => 3,
            SolverStatus::TimeExceeded => 4,
        };
        statuses[idx] += 1;
        let n = result.num_pieces;
        let min_t = result.optimized_times[..n]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        println!(
            "  run {:02}: status={:?} iters={:3} min_T={:.3}s dur={:.3}s",
            run, result.status, result.iterations, min_t,
            result.trajectory.total_duration(),
        );
    }
    println!(
        "  tallies: Convergence={} Stop={} MaxIter={} Invalid={} Timeout={}",
        statuses[0], statuses[1], statuses[2], statuses[3], statuses[4],
    );
}

/// Equispaced waypoints around a 2 m circle at z=1 m — a clean geometric
/// family parameterized only by n so that the only variable between runs
/// is the piece count. No near-duplicate waypoints, no sharp reversals.
fn circle_targets(n_targets: usize, buf: &mut [Vec3; MAX_PIECES_TEST]) -> &[Vec3] {
    use core::f32::consts::TAU;
    assert!(n_targets >= 1 && n_targets <= MAX_PIECES_TEST);
    let radius = 2.0f32;
    for i in 0..n_targets {
        let theta = TAU * (i + 1) as f32 / n_targets as f32;
        buf[i] = [radius * libm::cosf(theta), radius * libm::sinf(theta), 1.0];
    }
    &buf[..n_targets]
}

/// Scratch buffer size for `circuit_targets`. Sized to the larger of the
/// two MAX_PIECES values we've been running with so the same test code
/// works whether the planner is compiled at 16 or 20.
const MAX_PIECES_TEST: usize = 24;

/// Walk piece counts from 8 up to the compiled `MAX_PIECES` on the same
/// repeating circuit geometry. Surfaces any sharp transition in solver
/// behavior — status flip, iteration explosion, compression-ratio
/// collapse — that would explain the observed "some counts converge,
/// some don't" coupling on device.
#[test]
#[ignore] // diagnostic — run with `--release --ignored -- --nocapture`
fn production_regime_piece_count_sweep_to_max() {
    use cybflight_core::trajectory_planning::MAX_PIECES;
    let config = production_regime_config();
    println!(
        "\n=== Production-regime sweep: n ∈ [8, {MAX_PIECES}] (repeating 7-point circuit) ==="
    );
    let mut buf = [[0.0f32; 3]; MAX_PIECES_TEST];
    for n_targets in 8..=MAX_PIECES {
        let targets = circle_targets(n_targets, &mut buf);
        let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, targets);
        let init_dur: f32 = input.init_times[..n_targets].iter().sum();
        let result = plan(&input, &config);
        let final_dur = result.trajectory.total_duration();
        let ratio = if init_dur > 0.0 {
            final_dur / init_dur
        } else {
            0.0
        };
        let peaks = sample_kinematic_peaks(&result, &config);
        println!(
            "  n={:2}  status={:?}  iters={:3}  init={:.2}s  final={:.2}s  ratio={:.2}  peak_v={:.2}m/s  cost={:.2}",
            n_targets, result.status, result.iterations,
            init_dur, final_dur, ratio, peaks.v_max, result.final_cost,
        );
    }
}

/// Does turning on `weight_vel` (and only that — no other change to the
/// production regime) rescue convergence on the full 15-waypoint circuit?
/// If yes, the fix for the live failure is a one-line param bump.
#[test]
#[ignore] // diagnostic — run with `--release --ignored -- --nocapture`
fn production_regime_with_weight_vel_sweep() {
    println!("\n=== Production-regime + weight_vel sweep (full 15 wp) ===");
    for &wv in &[0.0_f32, 1.0, 10.0, 50.0, 100.0] {
        let mut vp = test_vehicle_params();
        vp.planner.max_vel_m_s = 5.0;
        vp.planner.max_tilt_rad = core::f32::consts::FRAC_PI_3;
        vp.planner.weight_time = 1.0;
        vp.planner.weight_energy = 0.0;
        vp.planner.weight_vel = wv;
        vp.planner.weight_tilt = 0.0;
        vp.planner.weight_body_rate = 10.0;
        vp.planner.weight_thrust = 10.0;
        vp.planner.smoothing_eps = 0.01;
        vp.planner.num_check_per_piece = 8;
        vp.planner.bfgs_trust.max_iterations = 500;
        let config = QuadPlanningConfig::from_vehicle_params(&vp);
        let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
        let result = plan(&input, &config);
        print_piece_diagnostics(&format!("wv={wv:.1}"), &result, &config);
    }
}

/// Vary the BFGS iteration cap on the exact production regime — simulates
/// what an STM32 solve sees if it hits `max_iterations` before converging.
/// Correlates the in-flight trajectory duration with the iteration count
/// at which the solver was actually cut off.
#[test]
#[ignore] // diagnostic — run with `--release --ignored -- --nocapture`
fn production_regime_max_iterations_sweep() {
    println!(
        "\n=== Production-regime + max_iterations sweep (wv=0, wtilt=0, we=0) ==="
    );
    for &max_iters in &[0_usize, 1, 2, 5, 10, 25, 50, 100, 150, 172, 500, 2000] {
        let mut vp = test_vehicle_params();
        vp.planner.max_vel_m_s = 5.0;
        vp.planner.max_tilt_rad = core::f32::consts::FRAC_PI_3;
        vp.planner.weight_time = 1.0;
        vp.planner.weight_energy = 0.0;
        vp.planner.weight_vel = 0.0;
        vp.planner.weight_tilt = 0.0;
        vp.planner.weight_body_rate = 10.0;
        vp.planner.weight_thrust = 10.0;
        vp.planner.smoothing_eps = 0.01;
        vp.planner.num_check_per_piece = 8;
        vp.planner.bfgs_trust.max_iterations = max_iters;
        let config = QuadPlanningConfig::from_vehicle_params(&vp);
        let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
        let result = plan(&input, &config);
        print_piece_diagnostics(&format!("max_it={max_iters}"), &result, &config);
    }
}

/// Sweep `weight_energy` while keeping everything else at the production
/// regime (in particular `weight_vel = 0`, `weight_tilt = 0`). Answers the
/// question "in this configuration, does raising we change peak speed or
/// trajectory duration at all?". If peak v and duration are flat across
/// we ∈ [0, 0.1], the we knob is doing nothing useful in production.
#[test]
#[ignore] // diagnostic — run with `--release --ignored -- --nocapture`
fn production_regime_with_weight_energy_sweep() {
    println!("\n=== Production-regime + weight_energy sweep (wv=0, wtilt=0) ===");
    for &we in &[0.0_f32, 0.001, 0.003, 0.01, 0.03, 0.1] {
        let mut vp = test_vehicle_params();
        vp.planner.max_vel_m_s = 5.0;
        vp.planner.max_tilt_rad = core::f32::consts::FRAC_PI_3;
        vp.planner.weight_time = 1.0;
        vp.planner.weight_energy = we;
        vp.planner.weight_vel = 0.0;
        vp.planner.weight_tilt = 0.0;
        vp.planner.weight_body_rate = 10.0;
        vp.planner.weight_thrust = 10.0;
        vp.planner.smoothing_eps = 0.01;
        vp.planner.num_check_per_piece = 8;
        vp.planner.bfgs_trust.max_iterations = 500;
        let config = QuadPlanningConfig::from_vehicle_params(&vp);
        let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
        let result = plan(&input, &config);
        print_piece_diagnostics(&format!("we={we:.4}"), &result, &config);
    }
}

/// Dump the production-regime trajectory to CSV for offline inspection
/// (plot the Z axis over time — if the first 1–2 seconds show Z exceeding
/// the hover setpoint by more than a few tens of cm, that's the reference
/// the MPC is being told to chase).
#[test]
fn save_production_regime_trajectory_csv() {
    let config = production_regime_config();
    let input = PlannerInput::waypoints(PRODUCTION_START, ZERO3, &PRODUCTION_WAYPOINTS);
    let result = plan(&input, &config);
    let path = output_path("planner_production_regime.csv");
    write_trajectory_csv(&result, &config, &path, 0.01);
    println!(
        "wrote production-regime trajectory: status={:?} iters={} dur={:.3}s path={}",
        result.status,
        result.iterations,
        result.trajectory.total_duration(),
        path.display(),
    );
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
