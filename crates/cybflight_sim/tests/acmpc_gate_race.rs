//! End-to-end validation of the ACMPC gate-racing policy against the
//! environment it was trained in.
//!
//! The policy is the 3.03 M-step ACMPC checkpoint from `cybflight-isaac`
//! (`train_out/cmp_acmpc_final.zip`) — a differentiable MPC used as the
//! actor of PPO (Romero et al., IEEE T-RO 2025). Fixtures under
//! `tests/fixtures/acmpc_race/` are exported from that project by
//! `tools/export_acmpc_policy.py` and carry, for the same deterministic
//! rollout: the cost network's weights, 150 paired (world state,
//! observation, cost logits, MPC first control, action) samples, and the
//! full 1200-step reference trajectory.
//!
//! The tests build up in dependency order, so a failure localizes:
//!
//! 1. `weights_*`             — the blob matches the declared topology
//! 2. `cost_network_*`        — the GELU tower == PyTorch, given an observation
//! 3. `observation_*`         — ENU→policy-frame adapter == the trainer's own
//!                              observation (all 30 entries), given a state
//! 4. `mpc_*`                 — the box-DDP solve == the trainer's, given an
//!                              observation: the actual method under test
//! 5. `reference_plant_*`     — closed loop reproduces the Python trajectory
//! 6. `policy_completes_*`    — the task succeeds, on the reference plant's
//!                              identified parameters and on cybflight's own
//!                              vehicle
//!
//! Steps 1–3 mirror `nn_gate_race`, deliberately: the two methods share
//! the observation prefix and the course, so the comparison in step 6 is
//! between control methods and not between two different problems.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test acmpc_gate_race -- --nocapture

mod common;

use common::{fixture, RaceOutcome};
use cybflight_core::acmpc::{
    ddp::LineSearch, AcmpcConfig, AcmpcPolicy, CtbrLimits, COST_OUT, HORIZON,
};
use cybflight_core::mpc::NU;
use cybflight_core::nn::mlp::{Activation, LayerShape, Mlp};
use cybflight_core::nn::race_policy::{PolicyConfig, PolicyFrame, VehicleState};
use cybflight_sim::controller::CtbrRateLoop;
use cybflight_sim::plant::QuadPlant;
use cybflight_sim::rl_reference::{
    rl_track, world_state_ned_to_enu, RateLoopConfig, RlParams, RlReferencePlant, RL_DT_S,
    RL_GATE_SIZE_M, RL_OMEGA_NORM_MAX, RL_START_POS_ENU, RL_START_YAW_ENU,
};
use nalgebra::{SVector, UnitQuaternion, Vector3};

// ── Fixture topology (pinned; see fixtures/acmpc_race/meta.json) ─────────
const SHAPES: [LayerShape; 4] = [
    LayerShape::new(20, 512),
    LayerShape::new(512, 512),
    LayerShape::new(512, 512),
    LayerShape::new(512, COST_OUT as u16),
];
/// Gate-relative prefix, then the `[p, q_wxyz, v]` raw-state suffix.
const OBS_PREFIX: usize = 20;
const OBS_LEN: usize = 30;
const SAMPLE_COLS: usize = 16 + OBS_LEN + COST_OUT + NU + NU + 1;
const TRAJ_COLS: usize = 16 + NU + 1;
/// Gate passes the Python reference achieved in 1200 steps.
const PYTHON_GATE_PASSES: u32 = 27;
/// `4·k_w·ω_max²` of the identified quad — the collective-thrust scale of
/// the action space (`cybflight.ctbr.A_MAX`).
const A_MAX: f32 = 89.64;

fn weights() -> Vec<f32> {
    fixture("acmpc_race", "cost_net.bin")
}

fn samples() -> Vec<f32> {
    fixture("acmpc_race", "samples.bin")
}

/// `AcmpcConfig` for the trained checkpoint. Every field is a property of
/// the weights, not a tunable.
fn upstream_config() -> AcmpcConfig {
    AcmpcConfig {
        policy: PolicyConfig {
            frame: PolicyFrame::LegacyNed,
            gates_ahead: 1,
            gate_size_m: RL_GATE_SIZE_M,
            omega_norm_min: 0.0,
            omega_norm_max: RL_OMEGA_NORM_MAX,
            // Unused on this path: the MPC commands CTBR, not motors. The
            // map still orders the rotor-speed observation entries.
            motor_limit: 1.0,
            motor_map: [0, 1, 2, 3],
        },
        ctbr: CtbrLimits {
            max_specific_thrust: A_MAX,
            max_body_rate: Vector3::new(10.0, 10.0, 4.0),
        },
        mpc_dt: 0.02,
        range_q: 1e5,
        range_p: 1e5,
        line_search: LineSearch { decay: 0.2, max_iter: 5 },
    }
}

fn policy(w: &[f32]) -> AcmpcPolicy<'_> {
    AcmpcPolicy::new(w, &SHAPES, &rl_track(), upstream_config()).expect("policy")
}

fn start_state_enu() -> (Vector3<f32>, f32) {
    (Vector3::from(RL_START_POS_ENU), RL_START_YAW_ENU)
}

// ─────────────────────────────────────────────────────────────────────────
// 1. Weight blob
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn weights_match_declared_topology() {
    let w = weights();
    let expected: usize = SHAPES.iter().map(|s| s.weight_count()).sum();
    assert_eq!(
        w.len(),
        expected,
        "cost_net.bin has {} f32 but the declared topology needs {expected}; \
         the fixture and SHAPES have diverged",
        w.len()
    );
    assert!(w.iter().all(|v| v.is_finite()), "non-finite weight");
    // The cost head emits a diagonal Q and a linear p per prediction step;
    // its width is what pins the horizon the checkpoint was trained at.
    assert_eq!(COST_OUT, 2 * HORIZON * 14);
    policy(&w);
}

// ─────────────────────────────────────────────────────────────────────────
// 2. The cost network
// ─────────────────────────────────────────────────────────────────────────

/// Given the trainer's own observation, the Rust GELU tower must
/// reproduce the trainer's raw cost logits. Comparing *before* the sigmoid
/// head is deliberate: the sigmoid saturates, and would compress a real
/// divergence into a passing number.
#[test]
fn cost_network_matches_pytorch() {
    let w = weights();
    let net = Mlp::<512>::new(&w, &SHAPES, Activation::Gelu).unwrap();
    let s = samples();
    let n = s.len() / SAMPLE_COLS;
    assert_eq!(s.len() % SAMPLE_COLS, 0);
    assert!(n >= 100, "expected a meaningful sample count, got {n}");

    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for r in 0..n {
        let row = &s[r * SAMPLE_COLS..(r + 1) * SAMPLE_COLS];
        let want = &row[16 + OBS_LEN..16 + OBS_LEN + COST_OUT];
        let mut got = [0.0f32; COST_OUT];
        net.forward(&row[16..16 + OBS_PREFIX], &mut got).unwrap();
        for i in 0..COST_OUT {
            worst = worst.max((got[i] - want[i]).abs());
            scale = scale.max(want[i].abs());
        }
    }
    println!(
        "cost network: worst |Δlogit| over {n} samples = {worst:.3e} \
         (logit magnitude up to {scale:.2})"
    );
    // ~170x the observed worst (5.7e-6), and still far under what swapping
    // the exact GELU for its tanh approximation would cost (~1e-2).
    assert!(
        worst < 1e-3,
        "Rust inference diverges from PyTorch by {worst:.3e} — a GELU or a \
         weight-layout mismatch, not accumulated rounding"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 3. Observation
// ─────────────────────────────────────────────────────────────────────────

/// Index of the raw-state quaternion's `w` entry within the observation.
const RAW_QW: usize = OBS_PREFIX + 3;

/// Given a vehicle state in ENU, the observation this crate builds must
/// equal the observation the trainer built from the equivalent NED state,
/// including the 10-entry raw-state suffix.
///
/// The quaternion is compared **up to sign**. That is not a weakened
/// assertion, it is the strongest one available: the trainer's plant
/// integrates a ZYX Euler triple and never wraps its yaw, so the
/// quaternion it derives carries the winding number of that integration
/// (yaw reaches 17.7 rad over this rollout) and flips sign every half
/// turn. A vehicle state does not contain that number.
/// `action_is_insensitive_to_the_quaternion_sign` bounds what the
/// ambiguity costs.
#[test]
fn observation_matches_python_ground_truth() {
    let w = weights();
    let mut pol = policy(&w);
    let s = samples();
    let n = s.len() / SAMPLE_COLS;

    let mut worst = [0.0f32; OBS_LEN];
    let mut sign_flips = 0usize;
    for r in 0..n {
        let row = &s[r * SAMPLE_COLS..(r + 1) * SAMPLE_COLS];
        let state = world_state_ned_to_enu(&row[0..16]);
        let want = &row[16..16 + OBS_LEN];
        pol.track.set_target_gate(row[SAMPLE_COLS - 1] as usize);

        let mut got = [0.0f32; OBS_LEN];
        pol.observe(&state, &mut got);

        let dot: f32 = (RAW_QW..RAW_QW + 4).map(|i| got[i] * want[i]).sum();
        let q_sign = if dot < 0.0 {
            sign_flips += 1;
            -1.0
        } else {
            1.0
        };
        for i in 0..OBS_LEN {
            let mut d = (got[i] - want[i]).abs();
            if i == 8 {
                // a wrapped angle: ±2π apart is identical
                let raw = got[i] - want[i];
                d = (raw - core::f32::consts::TAU * (raw / core::f32::consts::TAU).round()).abs();
            } else if (RAW_QW..RAW_QW + 4).contains(&i) {
                d = (q_sign * got[i] - want[i]).abs();
            }
            worst[i] = worst[i].max(d);
        }
    }
    println!("  raw-state quaternion sign differed on {sign_flips}/{n} samples");
    let labels = [
        "pos_g.x", "pos_g.y", "pos_g.z", "vel_g.x", "vel_g.y", "vel_g.z", "roll", "pitch",
        "yaw_rel", "p", "q", "r", "w0", "w1", "w2", "w3", "ahead.x", "ahead.y", "ahead.z",
        "ahead.yaw", "raw.px", "raw.py", "raw.pz", "raw.qw", "raw.qx", "raw.qy", "raw.qz",
        "raw.vx", "raw.vy", "raw.vz",
    ];
    println!("observation: worst |Δ| per entry over {n} samples");
    for i in 0..OBS_LEN {
        println!("  {:>9}  {:.3e}", labels[i], worst[i]);
    }
    let overall = worst.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        overall < 1e-4,
        "ENU→policy observation adapter diverges from the trainer by {overall:.3e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 4. The differentiable-MPC actor
// ─────────────────────────────────────────────────────────────────────────

/// The method itself: given the trainer's own observation, the ported
/// box-DDP must return the trainer's action.
///
/// The tolerance is looser than the network tests' and cannot be otherwise.
/// The solver's projected-Newton QP takes discrete decisions — which
/// controls are on their bounds, how far the Armijo and the outer line
/// search backtrack — and the learned cost is scaled to `1e5`, so a
/// single-precision difference of one ULP in a Riccati product can flip an
/// active-set membership. What must hold is that the *commands* agree, and
/// they are compared in the normalized `[-1, 1]` action space where their
/// magnitude is bounded by construction.
#[test]
fn mpc_action_matches_python_ground_truth() {
    let w = weights();
    let pol = policy(&w);
    let s = samples();
    let n = s.len() / SAMPLE_COLS;

    let (mut worst_u0, mut worst_action, mut worst_row) = (0.0f32, 0.0f32, 0usize);
    for r in 0..n {
        let row = &s[r * SAMPLE_COLS..(r + 1) * SAMPLE_COLS];
        let obs = &row[16..16 + OBS_LEN];
        let want_u0 = &row[16 + OBS_LEN + COST_OUT..16 + OBS_LEN + COST_OUT + NU];
        let want_action = &row[SAMPLE_COLS - 1 - NU..SAMPLE_COLS - 1];

        let got = pol.action(obs).expect("solve");
        // The trainer's `u0` is pre-normalization; undo the map to compare
        // in the same units the solver works in.
        let cfg = upstream_config().ctbr;
        let got_u0 = [
            (got[0] + 1.0) * 0.5 * cfg.max_specific_thrust,
            got[1] * cfg.max_body_rate.x,
            got[2] * cfg.max_body_rate.y,
            got[3] * cfg.max_body_rate.z,
        ];
        for i in 0..NU {
            worst_u0 = worst_u0.max((got_u0[i] - want_u0[i]).abs());
            let d = (got[i] - want_action[i]).abs();
            if d > worst_action {
                worst_action = d;
                worst_row = r;
            }
        }
    }
    println!(
        "box-DDP over {n} samples: worst |Δu0| = {worst_u0:.3e} (physical), \
         worst |Δaction| = {worst_action:.3e} (normalized, at sample {worst_row})"
    );
    // ~7x the observed worst (3.0e-4). Headroom for an active-set flip on
    // one sample, not for a wrong Riccati recursion.
    assert!(
        worst_action < 2e-3,
        "ported box-DDP diverges from the trainer's solve by {worst_action:.3e} \
         in normalized action units"
    );
}

/// What the unrecoverable quaternion sign is worth, measured rather than
/// assumed.
///
/// `q` and `−q` are the same attitude, so a *quadratic* cost cannot tell
/// them apart — but the learned linear term can, and it is scaled to
/// `1e5`. Since the port cannot reproduce the trainer's choice of sign
/// (see `observation_matches_python_ground_truth`), the question is
/// whether the trained cost network is insensitive to it in practice. It
/// is: the policy saw both signs throughout training, because its own
/// plant flipped them every half turn of yaw.
///
/// The closed-loop tests corroborate this from the other end — flown from
/// canonicalized quaternions, the policy passes exactly the same number of
/// gates as the Python rollout.
#[test]
fn action_is_insensitive_to_the_quaternion_sign() {
    let w = weights();
    let pol = policy(&w);
    let s = samples();
    let n = s.len() / SAMPLE_COLS;

    let mut worst = 0.0f32;
    for r in 0..n {
        let obs = &s[r * SAMPLE_COLS + 16..r * SAMPLE_COLS + 16 + OBS_LEN];
        let mut flipped: [f32; OBS_LEN] = obs.try_into().unwrap();
        for v in &mut flipped[RAW_QW..RAW_QW + 4] {
            *v = -*v;
        }
        let (a, b) = (pol.action(obs).unwrap(), pol.action(&flipped).unwrap());
        for i in 0..NU {
            worst = worst.max((a[i] - b[i]).abs());
        }
    }
    println!(
        "quaternion sign flip over {n} samples: worst |Δaction| = {worst:.3e} \
         (normalized units, action space spans 2.0)"
    );
    assert!(
        worst < 0.1,
        "the trained cost network is sensitive to the quaternion sign \
         ({worst:.3e} of a 2.0-wide action space) — the canonicalization in \
         `raw_state_in_frame` would then be a behavioural change, not a \
         representation choice"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 5-6. Closed loop
// ─────────────────────────────────────────────────────────────────────────

/// Fly the identified 5-inch quad, tracked by the training environment's
/// own CTBR rate loop and allocation — the plant and interface the policy
/// was optimized against.
fn fly_reference(steps: usize) -> RaceOutcome {
    let w = weights();
    let mut pol = policy(&w);
    let (pos, yaw) = start_state_enu();
    let mut plant = RlReferencePlant::hovering_at(RlParams::five_inch(), pos, yaw);
    let rate_loop = RateLoopConfig::default();

    let mut out = RaceOutcome::new(steps);
    for _ in 0..steps {
        let state = plant.vehicle_state();
        out.sample(&state);
        let u = plant.track_ctbr(&pol.step(&state).expect("policy step"), &rate_loop);
        let prev = plant.position_m;
        plant.step(&u, RL_DT_S);
        if !out.advance(&mut pol.track, prev, plant.position_m) {
            break;
        }
    }
    out
}

/// Closed loop from the identical initial condition must track the Python
/// rollout. Divergence is expected eventually — a chaotic nonlinear loop
/// whose actor makes discrete active-set decisions, integrated as a
/// quaternion here and as ZYX Euler there — so the assertion is
/// short-horizon.
#[test]
fn reference_plant_tracks_python_trajectory() {
    const HORIZON_STEPS: usize = 200; // 2 s
    let traj = fixture("acmpc_race", "traj.bin");
    let out = fly_reference(HORIZON_STEPS);

    let err = |k: usize| {
        let ws = &traj[k * TRAJ_COLS..(k + 1) * TRAJ_COLS];
        (out.positions[k] - Vector3::new(ws[1], ws[0], -ws[2])).norm() // NED → ENU
    };

    println!("reference plant vs Python, position error:");
    for &k in &[1usize, 10, 25, 50, 100, 150, 199] {
        println!("  step {k:>3} ({:>4.2} s)  {:.3e} m", k as f32 * RL_DT_S, err(k));
    }

    // A sign or axis error anywhere in the chain — observation, solver,
    // rate loop, allocation — shows up immediately, before any integration
    // difference can accumulate.
    assert!(
        err(1) < 1e-3,
        "diverges immediately ({:.3e} m after one step) — the CTBR command \
         or its allocation disagrees with the trainer",
        err(1)
    );
    assert!(
        err(25) < 1e-2,
        "diverges within 0.25 s ({:.3e} m) — faster than integrator \
         difference alone explains",
        err(25)
    );
    assert!(
        err(HORIZON_STEPS - 1) < 1.0,
        "diverges by {:.3} m over {:.1} s — beyond integrator difference",
        err(HORIZON_STEPS - 1),
        HORIZON_STEPS as f32 * RL_DT_S
    );
}

/// The task itself, on the identified parameters the policy was trained
/// against.
#[test]
fn policy_completes_laps_on_reference_plant() {
    let out = fly_reference(1200);
    out.report("reference plant", RL_DT_S, 8);
    assert_eq!(out.clips, 0, "clipped a gate");
    assert_eq!(out.steps, 1200, "flight terminated early: {}", out.ended);
    assert!(
        out.passes >= PYTHON_GATE_PASSES - 2,
        "only {} gate passes, Python reference achieves {PYTHON_GATE_PASSES}",
        out.passes
    );
}

/// The deliverable: the same policy, unmodified, flying cybflight's own
/// plant and vehicle — RK4 at 8 kHz, rigid-body `ω × Iω` coupling,
/// gyroscopic rotor precession and the physical quadratic yaw law — with
/// cybflight's own rate loop and allocator closing the CTBR command.
///
/// This is where ACMPC and the model-free policy in `nn_gate_race` part
/// company. That policy commands motors directly, so a plant whose yaw
/// authority differs from the identified model's is an open-loop
/// mismatch it cannot see. ACMPC commands *rates*, so the same mismatch
/// is inside a feedback loop.
#[test]
fn policy_completes_laps_on_cybflight_plant() {
    const VEHICLE_YAML: &str = include_str!("../../../vehicles/rl_racer.yaml");
    let v = vehicle_yaml::load("rl_racer", VEHICLE_YAML).expect("rl_racer.yaml invalid");

    let dt_plant = 1.0 / 8000.0;
    let substeps = (RL_DT_S / dt_plant).round() as usize;
    assert_eq!(substeps, 80);

    let w = weights();
    let mut pol = policy(&w);
    let (pos, yaw) = start_state_enu();
    // Same proportional gains the training environment's rate loop used,
    // so the inner-loop bandwidth the policy was tuned against carries
    // over; only the vehicle underneath it changes.
    let rate_loop = CtbrRateLoop::new(&v.params, &v.sim, Vector3::new(15.0, 15.0, 8.0));
    let mut plant = QuadPlant::new(v.params.clone(), &v.sim, dt_plant);
    plant.reset(
        pos,
        Vector3::zeros(),
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
    );

    let mut out = RaceOutcome::new(1200);
    let mut sat_ticks = 0u32;
    for _ in 0..1200 {
        let x = plant.control_state();
        let state = VehicleState {
            position_m: Vector3::new(x[0], x[1], x[2]),
            velocity_m_s: Vector3::new(x[7], x[8], x[9]),
            attitude: plant.attitude(),
            body_rate_rad_s: Vector3::new(x[10], x[11], x[12]),
            rotor_omega_rad_s: core::array::from_fn(|i| plant.rotor_omega()[i]),
        };
        out.sample(&state);
        let cmd = pol.step(&state).expect("policy step");

        // The rate loop runs at the plant's rate, as the firmware's inner
        // loop does; only the ACMPC solve is held over the control period.
        let prev = plant.position();
        for _ in 0..substeps {
            let rate = plant.control_state().fixed_rows::<3>(10).into_owned();
            let d = rate_loop.command(&cmd, rate);
            if d.iter().any(|v| *v >= 0.999) {
                sat_ticks += 1;
            }
            plant.step(&SVector::<f32, NU>::from(d));
        }
        if !out.advance(&mut pol.track, prev, plant.position()) {
            break;
        }
    }

    out.report("cybflight plant", RL_DT_S, 8);
    println!(
        "  motor-saturated on {:.1}% of inner-loop ticks",
        100.0 * sat_ticks as f32 / (out.steps * substeps) as f32
    );

    assert_eq!(out.clips, 0, "clipped a gate on the cybflight plant");
    assert_eq!(out.steps, 1200, "flight terminated early: {}", out.ended);
    assert!(
        out.passes >= 8,
        "completed only {} gate passes — not a full lap of the 8-gate track",
        out.passes
    );
}
