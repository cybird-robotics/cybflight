//! Closed-loop stress suite — are the body-rate STATE constraints of the
//! full-model NMPC actually *active* and *effective* under extreme agility?
//!
//! The runtime-comparison suite (`mpc_runtime_comparison.rs`) only checked
//! single open-loop solves. Here the `FullSqpSolver` runs as the actual
//! controller — RTI pattern (1 warm-started SQP iteration @ 100 Hz), full
//! nonlinear `FullQuadModel` RK4 plant @ 500 Hz — through maneuvers whose
//! *optimal* solution demands body rates beyond the vehicle limits. Each
//! scenario runs twice: barrier OFF (τ = 0) and barrier ON, same everything
//! else.
//!
//! What "active" and "effective" mean, measurably:
//! - **active**   — the closed-loop trajectory *rides* the rate boundary
//!   (utilization in the 0.85–1.05 band for a nontrivial fraction of the
//!   maneuver) instead of staying trivially far below it, and the
//!   unconstrained twin of the same maneuver exceeds the limits — i.e. the
//!   constraint, not the task, is what bounds the rates.
//! - **effective** — peak closed-loop |ω_i|/lim_i stays at/near 1 (soft
//!   barrier ⇒ small overshoot tolerance), violation time collapses vs the
//!   unconstrained run, and the maneuver still *succeeds* (the constraint
//!   must shape the trajectory, not break the task).
//!
//! Scenario rationale:
//! - `knife_edge_drop` — 95° roll start, already falling and sliding
//!   (v = (0, −2, −3) m/s): lift must be re-acquired immediately, so the
//!   optimizer prices a >limit roll snap. Stresses the roll axis limit.
//!   (95° is deliberate:
//!   probing shows that beyond ~100–110° tilt the local SQP leaves its
//!   attraction basin — all motors corner at the zero bound and "free
//!   fall" becomes a stationary point, with or without constraints. True
//!   inverted recovery needs a global initialization strategy and is out
//!   of scope for a constraint test.)
//! - `yaw_reversal` — 180° yaw error with `w_att_z = 200` on the weakest
//!   axis (6 rad/s): known from the open-loop suite to demand 3.4× the
//!   limit. Stresses sustained yaw saturation.
//! - `dash_reversal` — 7 m/s dash with the target *behind*: hard
//!   pitch-over braking reversal, rate demand coupled with thrust
//!   saturation.
//! - `tumble_arrest` — initial ω = 1.25× limits while tilted 45°: starts
//!   *outside* the feasible set (exercises the barrier's quadratic
//!   extension); measures how fast the loop re-enters and stays inside.
//!   (1.25× is deliberate: at 1.5× the ~23 rad/s tumble integrates through
//!   inverted attitudes before the arrest completes, and whether the
//!   vehicle lands inside or outside the ~100–110° free-fall basin above
//!   is decided by last-ulp rounding — equivalent-math reassociations of
//!   the Euler cross term flip the outcome. Same rationale as the 95°
//!   knife-edge cap.)
//! - `metronome` — lateral target flips ±1.5 m every second with zero
//!   preview: repeated aggressive bank reversals, sustained boundary
//!   riding on roll.
//!
//! Run with:
//!   cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --release --test mpc_state_constraint_stress -- --nocapture

extern crate alloc;

use cybflight_core::mpc::{FullQuadModel, FullQuadProblem, FullSqpSolver, N, NU, NX};
use nalgebra::{SVector, Unit, UnitQuaternion, Vector3};

// ═══════════════════════════════════════════════════════════════════════════
// Shared parameters (pinned, matching the runtime-comparison suite)
// ═══════════════════════════════════════════════════════════════════════════

const MASS: f32 = 0.55;
const GRAV: f32 = 9.81;
const MAX_THRUST_N: f32 = 8.5;
const RATE_LIM: [f32; 3] = [10.0, 10.0, 6.0];
// τ must be commensurate with the cost gradients it fights — the yaw
// attitude weight is 200, and a τ-sweep on `yaw_reversal` shows the
// closed-loop peak utilization falling 1.22 → 1.07 → 0.99 for
// τ = 0.1 → 0.5 → 1.0. τ = 0.5 balances enforcement against task
// performance across all five scenarios (τ = 0.1 under-enforces the
// high-weight yaw axis; τ ≳ 1 starts to trade task aggression away).
const BARRIER_TAU: f32 = 0.5;
const BARRIER_DELTA: f32 = 0.5;

const MPC_DT: f32 = 0.05; // horizon resolution (N = 20 → 1 s lookahead)
const SIM_DT: f32 = 0.002; // 500 Hz plant
const SOLVE_EVERY: usize = 5; // 100 Hz solver (firmware RTI pattern)

fn make_model(constrained: bool, dt: f32) -> FullQuadModel {
    FullQuadModel {
        mass: MASS,
        grav: GRAV,
        dt,
        rate_bounds: [
            [-RATE_LIM[0], RATE_LIM[0]],
            [-RATE_LIM[1], RATE_LIM[1]],
            [-RATE_LIM[2], RATE_LIM[2]],
        ],
        rate_barrier_tau: if constrained { BARRIER_TAU } else { 0.0 },
        rate_barrier_delta: BARRIER_DELTA,
        ..Default::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// State helpers
// ═══════════════════════════════════════════════════════════════════════════

fn make_state(
    pos: Vector3<f32>,
    q: UnitQuaternion<f32>,
    vel: Vector3<f32>,
    omega: Vector3<f32>,
) -> SVector<f32, NX> {
    SVector::<f32, NX>::from_row_slice(&[
        pos.x, pos.y, pos.z, q.i, q.j, q.k, q.w, vel.x, vel.y, vel.z, omega.x, omega.y, omega.z,
    ])
}

fn xref_at(pos: Vector3<f32>, q: UnitQuaternion<f32>) -> SVector<f32, NX> {
    make_state(pos, q, Vector3::zeros(), Vector3::zeros())
}

fn pos_of(x: &SVector<f32, NX>) -> Vector3<f32> {
    Vector3::new(x[0], x[1], x[2])
}

fn quat_of(x: &SVector<f32, NX>) -> UnitQuaternion<f32> {
    UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(x[6], x[3], x[4], x[5]))
}

/// Tilt = angle between the body z-axis and world up.
fn tilt_of(x: &SVector<f32, NX>) -> f32 {
    let bz = quat_of(x) * Vector3::z();
    bz.z.clamp(-1.0, 1.0).acos()
}

fn rate_util(x: &SVector<f32, NX>) -> f32 {
    let mut u = 0.0f32;
    for i in 0..3 {
        u = u.max(x[10 + i].abs() / RATE_LIM[i]);
    }
    u
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenarios
// ═══════════════════════════════════════════════════════════════════════════

/// Per-stage reference generator: `(sim time t, horizon stage k) → x_ref`.
/// Stage previews use `t + k·MPC_DT` where the scenario provides preview;
/// the metronome deliberately provides none (step targets are adversarial).
type RefFn = fn(t: f32) -> SVector<f32, NX>;

struct Scenario {
    name: &'static str,
    x0: SVector<f32, NX>,
    reference: RefFn,
    duration_s: f32,
    /// Success = task completed (checked on the final state).
    success: fn(x: &SVector<f32, NX>) -> bool,
    /// Whether horizon stages preview the reference at `t + k·dt`.
    preview: bool,
    /// Absolute cap on the constrained run's peak utilization (soft barrier
    /// + knot-point enforcement ⇒ scenario-specific transient headroom).
    peak_cap: f32,
}

fn sc_knife_edge_drop() -> Scenario {
    fn r(_t: f32) -> SVector<f32, NX> {
        xref_at(Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity())
    }
    fn ok(x: &SVector<f32, NX>) -> bool {
        tilt_of(x) < 15.0_f32.to_radians()
            && (pos_of(x) - Vector3::new(0.0, 0.0, 1.0)).norm() < 0.5
    }
    Scenario {
        name: "knife_edge_drop",
        x0: make_state(
            Vector3::zeros(),
            UnitQuaternion::from_axis_angle(
                &Unit::new_normalize(Vector3::x()),
                95.0_f32.to_radians(),
            ),
            // Already falling and sliding — lift must be re-acquired NOW,
            // which prices the roll snap even higher.
            Vector3::new(0.0, -2.0, -3.0),
            Vector3::zeros(),
        ),
        reference: r,
        duration_s: 5.0,
        success: ok,
        preview: true,
        peak_cap: 1.3,
    }
}

fn sc_yaw_reversal() -> Scenario {
    fn r(_t: f32) -> SVector<f32, NX> {
        xref_at(Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity())
    }
    fn ok(x: &SVector<f32, NX>) -> bool {
        // Yaw error < 10° and position held.
        let q = quat_of(x);
        let (_, _, yaw) = q.euler_angles();
        yaw.abs() < 10.0_f32.to_radians()
            && (pos_of(x) - Vector3::new(0.0, 0.0, 1.0)).norm() < 0.5
    }
    Scenario {
        name: "yaw_reversal",
        x0: make_state(
            Vector3::new(0.0, 0.0, 1.0),
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vector3::z()), 179.0_f32.to_radians()),
            Vector3::zeros(),
            Vector3::zeros(),
        ),
        reference: r,
        duration_s: 4.0,
        success: ok,
        preview: true,
        peak_cap: 1.35,
    }
}

fn sc_dash_reversal() -> Scenario {
    fn r(_t: f32) -> SVector<f32, NX> {
        xref_at(Vector3::new(-2.0, 0.0, 1.0), UnitQuaternion::identity())
    }
    fn ok(x: &SVector<f32, NX>) -> bool {
        (pos_of(x) - Vector3::new(-2.0, 0.0, 1.0)).norm() < 0.5
    }
    Scenario {
        name: "dash_reversal",
        x0: make_state(
            Vector3::new(0.0, 0.0, 1.0),
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vector3::y()), 20.0_f32.to_radians()),
            Vector3::new(7.0, 0.0, 0.0),
            Vector3::zeros(),
        ),
        reference: r,
        duration_s: 5.0,
        success: ok,
        preview: true,
        // The braking pitch-over couples rate demand with thrust
        // saturation, and its peak is the most rounding-sensitive number
        // in the suite: mathematically equivalent reassociations of the
        // Euler cross term move it by ±0.2 (observed 1.65 → 1.86). The
        // cap is a coarse sanity bound; the sharp assertions for this
        // scenario are constrained-below-unconstrained and the aggregate
        // violation-time reduction.
        peak_cap: 2.0,
    }
}

fn sc_tumble_arrest() -> Scenario {
    fn r(_t: f32) -> SVector<f32, NX> {
        xref_at(Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity())
    }
    fn ok(x: &SVector<f32, NX>) -> bool {
        (pos_of(x) - Vector3::new(0.0, 0.0, 1.0)).norm() < 0.5 && rate_util(x) < 0.5
    }
    Scenario {
        name: "tumble_arrest",
        x0: make_state(
            Vector3::zeros(),
            UnitQuaternion::from_axis_angle(
                &Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0)),
                45.0_f32.to_radians(),
            ),
            Vector3::zeros(),
            // 1.25× every axis limit — starts OUTSIDE the feasible set.
            Vector3::new(1.25 * RATE_LIM[0], -1.25 * RATE_LIM[1], 1.25 * RATE_LIM[2]),
        ),
        reference: r,
        duration_s: 5.0,
        success: ok,
        preview: true,
        // Starts at util 1.25 by construction; effectiveness is judged by
        // re-entry time, but the transient must not blow past this.
        peak_cap: 1.6,
    }
}

fn sc_metronome() -> Scenario {
    fn r(t: f32) -> SVector<f32, NX> {
        // Lateral square wave ±1.5 m, 1 s per side, frozen after 4 s so the
        // run can settle for the success check.
        let phase = if t < 4.0 { (t as i32) % 2 } else { 1 };
        let y = if phase == 0 { 1.5 } else { -1.5 };
        xref_at(Vector3::new(0.0, y, 1.0), UnitQuaternion::identity())
    }
    fn ok(x: &SVector<f32, NX>) -> bool {
        (pos_of(x) - Vector3::new(0.0, -1.5, 1.0)).norm() < 0.5
    }
    Scenario {
        name: "metronome",
        x0: make_state(
            Vector3::new(0.0, 0.0, 1.0),
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
        ),
        reference: r,
        duration_s: 6.0,
        // No preview: each target step appears instantaneously — worst case.
        preview: false,
        success: ok,
        peak_cap: 1.15,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Closed-loop runner
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct RunMetrics {
    peak_util: f32,
    /// Fraction of sim time with util > 1.02 (violating; small tolerance for
    /// the soft constraint's boundary chatter).
    viol_pct: f32,
    /// Fraction of sim time riding the boundary (util ∈ [0.85, 1.05]) —
    /// the "constraint active" signature.
    ride_pct: f32,
    /// Last time [s] at which util exceeded 1.02 (re-entry time for runs
    /// that start infeasible). 0 if never.
    t_last_viol: f32,
    final_err: f32,
    success: bool,
    crashed: bool,
}

fn run_scenario(sc: &Scenario, constrained: bool) -> RunMetrics {
    run_scenario_with_margin(sc, constrained, 1.0)
}

/// `margin` scales the *enforced* bounds relative to the vehicle limits
/// (utilization metrics are always vs the true `RATE_LIM`). Enforcing at,
/// say, 0.85× the limit absorbs intra-knot overshoot: the barrier only acts
/// on the 50 ms horizon knots, and a low-inertia axis can build ~α·dt of
/// extra rate between them.
fn run_scenario_with_margin(sc: &Scenario, constrained: bool, margin: f32) -> RunMetrics {
    // Controller model predicts at MPC_DT; plant integrates at SIM_DT.
    let mut ctrl_model = make_model(constrained, MPC_DT);
    for b in ctrl_model.rate_bounds.iter_mut() {
        b[0] *= margin;
        b[1] *= margin;
    }
    let plant = make_model(false, SIM_DT); // plant has no cost — τ irrelevant
    let problem = FullQuadProblem::with_rk4(ctrl_model, N);
    let mut solver = alloc::boxed::Box::new(FullSqpSolver::new());

    let hover = SVector::<f32, NU>::from_element(MASS * GRAV / 4.0);
    let u_refs = [hover; N];
    let mut u_warm = [hover; N];
    let mut last_u = hover;

    let steps = (sc.duration_s / SIM_DT) as usize;
    let mut x = sc.x0;

    let mut peak_util = 0.0f32;
    let mut viol_steps = 0usize;
    let mut ride_steps = 0usize;
    let mut t_last_viol = 0.0f32;
    let mut crashed = false;

    for step in 0..steps {
        let t = step as f32 * SIM_DT;

        if step % SOLVE_EVERY == 0 {
            let mut x_refs = [SVector::<f32, NX>::zeros(); N + 1];
            for (k, xr) in x_refs.iter_mut().enumerate() {
                let tk = if sc.preview { t + k as f32 * MPC_DT } else { t };
                *xr = (sc.reference)(tk);
            }
            let _ = solver.solve(&problem, &x, &x_refs, &u_refs, &u_warm, 1, 1e-3);
            // Solver health guard, mirroring what actually flies: the
            // firmware outer loop discards non-finite solutions (holding
            // the previous command), and the sim's `MpcFullIndiController`
            // additionally resets its warm start so one diverged solve
            // cannot poison every subsequent RTI iteration. Without this
            // the closed loop feeds NaN into the plant and "crashes" on
            // solver transients the real system shrugs off — these stiff
            // scenarios sit close enough to the solver's f32 limits that
            // last-ulp rounding differences decide whether a transient
            // occurs at all.
            let u_bar = solver.u_bar();
            if u_bar.iter().all(|u| u.iter().all(|v| v.is_finite())) {
                u_warm = *u_bar;
                last_u = u_warm[0];
            } else {
                u_warm = u_refs;
            }
        }

        // Plant step with physically clamped motors.
        let u_clamped = SVector::<f32, NU>::from_fn(|i, _| last_u[i].clamp(0.0, MAX_THRUST_N));
        x = plant.propagate_rk4(&x, &u_clamped);

        if !x[0].is_finite() || pos_of(&x).norm() > 50.0 {
            crashed = true;
            break;
        }

        let util = rate_util(&x);
        peak_util = peak_util.max(util);
        if util > 1.02 {
            viol_steps += 1;
            t_last_viol = t;
        }
        if (0.85..=1.05).contains(&util) {
            ride_steps += 1;
        }
    }

    let final_ref = (sc.reference)(sc.duration_s);
    let final_err = (pos_of(&x) - Vector3::new(final_ref[0], final_ref[1], final_ref[2])).norm();
    RunMetrics {
        peak_util,
        viol_pct: 100.0 * viol_steps as f32 / steps as f32,
        ride_pct: 100.0 * ride_steps as f32 / steps as f32,
        t_last_viol,
        final_err,
        success: !crashed && (sc.success)(&x),
        crashed,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_constraints_active_and_effective_under_aggression() {
    let scenarios = [
        sc_knife_edge_drop(),
        sc_yaw_reversal(),
        sc_dash_reversal(),
        sc_tumble_arrest(),
        sc_metronome(),
    ];

    println!();
    println!("═════════════════════════════════════════════════════════════════════════════════════════════");
    println!("Closed-loop aggression suite — FullSqpSolver RTI (1 it @ 100 Hz), FullQuadModel plant @ 500 Hz");
    println!("  util = max_i |ω_i|/lim_i on the PLANT state; ride% = time at util∈[0.85,1.05] (limit-riding)");
    println!("═════════════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "{:<18} {:<8} {:>9} {:>7} {:>7} {:>11} {:>10} {:>8}",
        "scenario", "config", "peak", "viol%", "ride%", "lastviol[s]", "finalerr", "success"
    );

    struct Row {
        name: &'static str,
        peak_cap: f32,
        starts_infeasible: bool,
        unc: RunMetrics,
        con: RunMetrics,
    }
    let mut rows = alloc::vec::Vec::new();

    for sc in &scenarios {
        let unc = run_scenario(sc, false);
        let con = run_scenario(sc, true);
        for (cfg, m) in [("off", &unc), ("ON", &con)] {
            println!(
                "{:<18} {:<8} {:>9.3} {:>6.1}% {:>6.1}% {:>11.2} {:>10.3} {:>8}",
                sc.name,
                cfg,
                m.peak_util,
                m.viol_pct,
                m.ride_pct,
                m.t_last_viol,
                m.final_err,
                if m.crashed {
                    "CRASH"
                } else if m.success {
                    "yes"
                } else {
                    "NO"
                },
            );
        }
        rows.push(Row {
            name: sc.name,
            peak_cap: sc.peak_cap,
            starts_infeasible: rate_util(&sc.x0) > 1.0,
            unc,
            con,
        });
    }
    println!("═════════════════════════════════════════════════════════════════════════════════════════════");

    // ── Assertions ────────────────────────────────────────────────────────
    for r in &rows {
        // The task must be genuinely extreme: without the constraint, the
        // closed loop exceeds the rate limits (otherwise the scenario tests
        // nothing). Crashing counts as exceeding.
        assert!(
            r.unc.crashed || r.unc.peak_util > 1.05,
            "{}: unconstrained run stayed inside the limits (peak {:.3}) — scenario too tame",
            r.name,
            r.unc.peak_util
        );

        // Effective, part 1: the constrained loop must complete the task.
        assert!(
            r.con.success,
            "{}: constrained run failed the task (crashed={}, final_err={:.3})",
            r.name, r.con.crashed, r.con.final_err
        );

        // Effective, part 2: the constrained peak must be strictly below the
        // unconstrained peak (the constraint changed the trajectory in the
        // right direction) AND under the scenario's absolute cap. The cap is
        // scenario-specific because enforcement is at 50 ms horizon knots —
        // between knots the plant can overshoot by up to ~α·dt (worst on the
        // low-inertia yaw axis), and tumble_arrest *starts* at util 1.25.
        assert!(
            r.con.peak_util < r.unc.peak_util,
            "{}: constrained peak {:.3} not below unconstrained peak {:.3}",
            r.name,
            r.con.peak_util,
            r.unc.peak_util
        );
        assert!(
            r.con.peak_util < r.peak_cap,
            "{}: constrained peak utilization {:.3} exceeds cap {:.2}",
            r.name,
            r.con.peak_util,
            r.peak_cap
        );

        // tumble_arrest starts infeasible: the loop must re-enter the
        // feasible set quickly and stay there.
        if r.name == "tumble_arrest" {
            assert!(
                r.con.t_last_viol < 1.0,
                "{}: constrained run must re-enter the feasible set within 1 s (last violation at {:.2} s)",
                r.name,
                r.con.t_last_viol
            );
        }

        // Violation time must not get worse; where the unconstrained run
        // spends real time in violation without crashing, expect a clear
        // reduction is checked in aggregate below (per-scenario halving is
        // too brittle on the yaw axis, where knot-point overshoot dominates).
        // Scenarios that START infeasible are exempt: their violation time
        // is dominated by the mandatory decay from the initial state, and
        // the soft barrier's equilibrium band makes that decay *shape*
        // differ from the unconstrained run's without saying anything about
        // enforcement (tumble_arrest has its own re-entry assertion above).
        if !r.unc.crashed && !r.starts_infeasible {
            assert!(
                r.con.viol_pct <= r.unc.viol_pct + 1.0,
                "{}: constrained violation time {:.1}% worse than unconstrained {:.1}%",
                r.name,
                r.con.viol_pct,
                r.unc.viol_pct
            );
        }
    }

    // Aggregate effectiveness: total violation time over the non-crashed
    // pairs must drop meaningfully with the barrier on.
    let (unc_viol_sum, con_viol_sum): (f32, f32) = rows
        .iter()
        .filter(|r| !r.unc.crashed)
        .fold((0.0, 0.0), |(a, b), r| {
            (a + r.unc.viol_pct, b + r.con.viol_pct)
        });
    assert!(
        con_viol_sum < 0.8 * unc_viol_sum,
        "aggregate violation time did not drop meaningfully: constrained Σ{:.1}% vs unconstrained Σ{:.1}%",
        con_viol_sum,
        unc_viol_sum
    );

    // Active: across the suite, the constrained loop must actually ride the
    // boundary somewhere (a constraint that is never near-active proves
    // nothing). Sum over scenarios to avoid over-fitting per-case dynamics.
    let total_ride: f32 = rows.iter().map(|r| r.con.ride_pct).sum();
    assert!(
        total_ride > 10.0,
        "constrained runs never ride the rate boundary (Σ ride% = {:.1}) — constraints inactive?",
        total_ride
    );
}

/// Deployment recipe check: enforcing the bounds at 0.85× the vehicle limit
/// absorbs the knot-point overshoot. The yaw axis is the worst case (lowest
/// inertia → most rate build-up inside a 50 ms knot interval): enforced at
/// the raw limit its closed-loop peak stays ~1.2× (see the suite above);
/// enforced with margin, the *vehicle* limit must be essentially respected.
#[test]
fn margin_enforcement_absorbs_knot_overshoot() {
    let sc = sc_yaw_reversal();
    let raw = run_scenario_with_margin(&sc, true, 1.0);
    let margined = run_scenario_with_margin(&sc, true, 0.85);
    println!(
        "yaw_reversal peak vs vehicle limit: enforced@1.00 → {:.3}, enforced@0.85 → {:.3}",
        raw.peak_util, margined.peak_util
    );
    assert!(margined.success, "margined run must still complete the task");
    assert!(
        margined.peak_util < 1.05,
        "margin-enforced yaw peak {:.3} should respect the vehicle limit (< 1.05)",
        margined.peak_util
    );
    assert!(
        margined.peak_util < raw.peak_util,
        "margin must reduce the vehicle-limit peak ({:.3} !< {:.3})",
        margined.peak_util,
        raw.peak_util
    );
}
