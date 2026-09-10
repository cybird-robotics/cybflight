//! End-to-end validation of the neural gate-racing policy against the
//! upstream RL environment it was trained in.
//!
//! The policy is a PPO checkpoint from `optimal_quad_control_RL`
//! (`models/my_session/run0_5inch_10_percent/98000000.zip`, 20→64→64→4
//! ReLU). Fixtures under `tests/fixtures/rl_race/` are exported from that
//! project by `export_fixtures.py` and carry, for the same deterministic
//! rollout: the flat weights, 150 paired (world state, observation, raw
//! action) samples, and the full 1200-step reference trajectory.
//!
//! The tests build up in dependency order, so a failure localizes:
//!
//! 1. `weights_*`             — the blob matches the declared topology
//! 2. `forward_pass_*`        — Rust inference == PyTorch, given an observation
//! 3. `observation_*`         — ENU→policy-frame adapter == the trainer's own
//!                              observation, given a state
//! 4. `reference_plant_*`     — closed loop reproduces the Python trajectory
//! 5. `policy_completes_*`    — the task actually succeeds, on both the
//!                              reference plant and cybflight's own plant
//!
//! Everything cybflight-side is ENU world / FLU body. The NED convention
//! appears only where fixtures are decoded, because that is the frame the
//! *weights* were trained in — see `PolicyFrame`.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test nn_gate_race -- --nocapture

mod common;

use common::{fixture, RaceOutcome};
use cybflight_core::mpc::NU;
use cybflight_core::nn::mlp::{Activation, LayerShape, Mlp};
use cybflight_core::nn::race_policy::{
    PolicyConfig, PolicyFrame, RacePolicy, VehicleState, NUM_MOTORS,
};
use cybflight_sim::plant::QuadPlant;
use cybflight_sim::rl_reference::{
    rl_track, world_state_ned_to_enu, RlParams, RlReferencePlant, RL_DT_S, RL_GATE_SIZE_M,
    RL_OMEGA_NORM_MAX, RL_START_POS_ENU, RL_START_YAW_ENU, RL_TRACK_ENU,
};
use nalgebra::{SVector, UnitQuaternion, Vector3};

// ── Fixture topology (pinned; see fixtures/rl_race/meta.json) ────────────
const SHAPES: [LayerShape; 3] = [
    LayerShape::new(20, 64),
    LayerShape::new(64, 64),
    LayerShape::new(64, 4),
];
const OBS_LEN: usize = 20;
const SAMPLE_COLS: usize = 16 + OBS_LEN + 4 + 1;
const TRAJ_COLS: usize = 16 + 4 + 1;
/// Gate passes the Python reference achieved in 1200 steps.
const PYTHON_GATE_PASSES: u32 = 58;

fn weights() -> Vec<f32> {
    fixture("rl_race", "policy.bin")
}

/// `PolicyConfig` for the upstream checkpoint. Every field here is a
/// property of the weights, not a tunable.
fn upstream_config() -> PolicyConfig {
    PolicyConfig {
        frame: PolicyFrame::LegacyNed,
        gates_ahead: 1,
        gate_size_m: RL_GATE_SIZE_M,
        omega_norm_min: 0.0,
        omega_norm_max: RL_OMEGA_NORM_MAX,
        motor_limit: 1.0,
        motor_map: [0, 1, 2, 3],
    }
}

fn policy(w: &[f32]) -> RacePolicy<'_> {
    RacePolicy::new(w, &SHAPES, &rl_track(), upstream_config()).expect("policy")
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
        "policy.bin has {} f32 but the declared topology needs {expected}; \
         the fixture and SHAPES have diverged",
        w.len()
    );
    assert!(w.iter().all(|v| v.is_finite()), "non-finite weight");
    let mlp = Mlp::<64>::new(&w, &SHAPES, Activation::Relu).expect("topology");
    assert_eq!(mlp.input_dim(), OBS_LEN);
    assert_eq!(mlp.output_dim(), NUM_MOTORS);
}

/// The ENU track must be exactly the upstream NED table re-expressed.
#[test]
fn enu_track_matches_upstream_ned_table() {
    // Verbatim from optimal_quad_control_RL/train.py (r = 1.5).
    const R: f32 = 1.5;
    let gate_pos_ned = [
        [R, -R, -1.5],
        [0.0, 0.0, -1.5],
        [-R, R, -1.5],
        [0.0, 2.0 * R, -1.5],
        [R, R, -1.5],
        [0.0, 0.0, -1.5],
        [-R, -R, -1.5],
        [0.0, -2.0 * R, -1.5],
    ];
    let gate_yaw_ned: [f32; 8] = [1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0]
        .map(|k: f32| k * core::f32::consts::FRAC_PI_2);

    for i in 0..8 {
        let (p_enu, yaw_enu) = RL_TRACK_ENU[i];
        // ENU → NED
        let back = [p_enu[1], p_enu[0], -p_enu[2]];
        for k in 0..3 {
            assert!(
                (back[k] - gate_pos_ned[i][k]).abs() < 1e-6,
                "gate {i} axis {k}: {} vs {}",
                back[k],
                gate_pos_ned[i][k]
            );
        }
        // Headings compare as directions, not raw angles (±2π aliases).
        let yaw_back = core::f32::consts::FRAC_PI_2 - yaw_enu;
        let d = yaw_back - gate_yaw_ned[i];
        let wrapped = d - core::f32::consts::TAU * (d / core::f32::consts::TAU).round();
        assert!(wrapped.abs() < 1e-6, "gate {i} yaw: {yaw_back} vs {}", gate_yaw_ned[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 2. Inference
// ─────────────────────────────────────────────────────────────────────────

/// Given the trainer's own observation, Rust inference must reproduce the
/// trainer's raw action. This isolates weights + forward pass from the
/// observation pipeline: it is the test that catches a transposed weight
/// matrix, a reordered layer, or a wrong endianness.
#[test]
fn forward_pass_matches_pytorch() {
    let w = weights();
    let mlp = Mlp::<64>::new(&w, &SHAPES, Activation::Relu).unwrap();
    let samples = fixture("rl_race", "samples.bin");
    let n = samples.len() / SAMPLE_COLS;
    assert_eq!(samples.len() % SAMPLE_COLS, 0);
    assert!(n >= 100, "expected a meaningful sample count, got {n}");

    let mut worst = 0.0f32;
    for r in 0..n {
        let row = &samples[r * SAMPLE_COLS..(r + 1) * SAMPLE_COLS];
        let obs = &row[16..16 + OBS_LEN];
        let want = &row[16 + OBS_LEN..16 + OBS_LEN + 4];
        let mut got = [0.0f32; NUM_MOTORS];
        mlp.forward(obs, &mut got).unwrap();
        for i in 0..NUM_MOTORS {
            worst = worst.max((got[i] - want[i]).abs());
        }
    }
    println!("forward pass: worst |Δaction| over {n} samples = {worst:.3e}");
    assert!(
        worst < 1e-5,
        "Rust inference diverges from PyTorch by {worst:.3e}"
    );
}

/// Given a vehicle state in ENU, the observation this crate builds must
/// equal the observation the trainer built from the equivalent NED state.
/// This is the test that polices the frame adapter, the gate-relative
/// projection and the look-ahead features.
#[test]
fn observation_matches_python_ground_truth() {
    let w = weights();
    let mut pol = policy(&w);
    let samples = fixture("rl_race", "samples.bin");
    let n = samples.len() / SAMPLE_COLS;

    let mut worst = [0.0f32; OBS_LEN];
    for r in 0..n {
        let row = &samples[r * SAMPLE_COLS..(r + 1) * SAMPLE_COLS];
        let state = world_state_ned_to_enu(&row[0..16]);
        let want = &row[16..16 + OBS_LEN];
        pol.track.set_target_gate(row[SAMPLE_COLS - 1] as usize);

        let mut got = [0.0f32; OBS_LEN];
        pol.track.observe(&state, &mut got);
        for i in 0..OBS_LEN {
            let mut d = (got[i] - want[i]).abs();
            // entry 8 is a wrapped angle: ±2π apart is identical
            if i == 8 {
                let raw = got[i] - want[i];
                d = (raw - core::f32::consts::TAU * (raw / core::f32::consts::TAU).round()).abs();
            }
            worst[i] = worst[i].max(d);
        }
    }
    let labels = [
        "pos_g.x", "pos_g.y", "pos_g.z", "vel_g.x", "vel_g.y", "vel_g.z", "roll", "pitch",
        "yaw_rel", "p", "q", "r", "w0", "w1", "w2", "w3", "ahead.x", "ahead.y", "ahead.z",
        "ahead.yaw",
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
// 3. Closed loop on the reference plant
// ─────────────────────────────────────────────────────────────────────────

fn fly_reference(steps: usize) -> RaceOutcome {
    let w = weights();
    let mut pol = policy(&w);
    let (pos, yaw) = start_state_enu();
    let mut plant = RlReferencePlant::hovering_at(RlParams::five_inch(), pos, yaw);

    let mut out = RaceOutcome::new(steps);
    for _ in 0..steps {
        let state = plant.vehicle_state();
        out.sample(&state);
        let u = pol.step(&state).expect("policy step");
        let prev = plant.position_m;
        plant.step(&u, RL_DT_S);
        if !out.advance(&mut pol.track, prev, plant.position_m) {
            break;
        }
    }
    out
}

/// Closed loop from the identical initial condition must track the Python
/// rollout. Divergence is expected eventually — this is a chaotic nonlinear
/// loop, and the reference plant integrates attitude as a quaternion where
/// the trainer integrates ZYX Euler — so the assertion is short-horizon.
#[test]
fn reference_plant_tracks_python_trajectory() {
    const HORIZON: usize = 200; // 2 s
    let traj = fixture("rl_race", "traj.bin");
    let out = fly_reference(HORIZON);

    let err = |k: usize| {
        let ws = &traj[k * TRAJ_COLS..(k + 1) * TRAJ_COLS];
        let want = Vector3::new(ws[1], ws[0], -ws[2]); // NED → ENU
        (out.positions[k] - want).norm()
    };

    println!("reference plant vs Python, position error:");
    for &k in &[1usize, 10, 25, 50, 100, 150, 199] {
        println!("  step {k:>3} ({:>4.2} s)  {:.3e} m", k as f32 * RL_DT_S, err(k));
    }

    // A sign or axis error in the transcription shows up immediately, as a
    // large error within the first few steps, before any integration
    // difference can accumulate. Divergence thereafter is expected and not
    // evidence of a bug: this is a chaotic closed loop, and the reference
    // plant integrates attitude as a quaternion where the trainer
    // integrates ZYX Euler. What must hold over the long horizon is the
    // task outcome, which `policy_completes_laps_on_reference_plant`
    // asserts.
    assert!(
        err(1) < 1e-3,
        "diverges immediately ({:.3e} m after one step) — the ENU \
         transcription of the upstream ODE has a sign or axis error",
        err(1)
    );
    assert!(
        err(25) < 1e-2,
        "diverges within 0.25 s ({:.3e} m) — faster than integrator \
         difference alone explains",
        err(25)
    );
    assert!(
        err(HORIZON - 1) < 1.0,
        "diverges by {:.3} m over {:.1} s — beyond integrator difference",
        err(HORIZON - 1),
        HORIZON as f32 * RL_DT_S
    );
}

/// The task itself, on the plant the policy was trained against.
#[test]
fn policy_completes_laps_on_reference_plant() {
    let out = fly_reference(1200);
    out.report("reference plant", RL_DT_S, 8);
    assert_eq!(out.clips, 0, "clipped a gate");
    assert_eq!(out.steps, 1200, "flight terminated early: {}", out.ended);
    // The Python reference gets 58. Allow a small margin for the quaternion
    // vs Euler integration difference, but require the same lap rate.
    assert!(
        out.passes >= PYTHON_GATE_PASSES - 2,
        "only {} gate passes, Python reference achieves {PYTHON_GATE_PASSES}",
        out.passes
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 4. Closed loop on cybflight's own plant
// ─────────────────────────────────────────────────────────────────────────

/// The deliverable: the same policy, unmodified, flying cybflight's real
/// plant — RK4 at 8 kHz, rigid-body `ω × Iω` coupling, gyroscopic rotor
/// precession, and the physical quadratic yaw law — on the `rl_racer`
/// airframe derived from the upstream identification.
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

        let u = pol.step(&state).expect("policy step");
        if u.iter().any(|v| *v >= 0.999) {
            sat_ticks += 1;
        }
        let cmd = SVector::<f32, NU>::from_column_slice(&u);

        let prev = plant.position();
        for _ in 0..substeps {
            plant.step(&cmd);
        }
        if !out.advance(&mut pol.track, prev, plant.position()) {
            break;
        }
    }

    out.report("cybflight plant", RL_DT_S, 8);
    println!(
        "  motor-saturated on {:.0}% of ticks",
        100.0 * sat_ticks as f32 / out.steps as f32
    );

    assert_eq!(out.clips, 0, "clipped a gate on the cybflight plant");
    assert_eq!(out.steps, 1200, "flight terminated early: {}", out.ended);
    assert!(
        out.passes >= 8,
        "completed only {} gate passes — not a full lap of the 8-gate track",
        out.passes
    );
}
