//! Evaluation of the learned situation-conditioned cost policy
//! (docs/learned_mpc_cost.md, PLAN B) on the indoor missions.
//!
//! For every mission, the flight-configured SQP stack
//! (`figure8_tinympc_compare.rs`'s `mpc_indi`: tilt-yaw `q_ref`, flatness
//! feedforward, position sampler, N = 20, RTI) is flown three ways:
//!
//! - `nominal`  — the hand tune (`z = 0`), the reference number;
//! - `learned`  — with the trained policy modulating the cost every solve;
//! - `blind`    — with the constant-`z` policy trained with the observation
//!   zeroed (the "is it adaptivity or just a better tune?" control),
//!   when `target/cost_policy_blind/policy.bin` exists.
//!
//! `indoor_splits_timeopt` is the held-out test trajectory: the policy is
//! never trained on it (or on any mission file — training uses procedural
//! primitives only, see `cybflight_sim::primitives`).
//!
//! Reports both the sim's time-indexed `rms_pos_err_m` and the geometric
//! (closest-point) RMSE. Writes `target/sim-out/learned_cost/summary.json`
//! and per-run traces. Policy paths via `COST_POLICY` / `COST_POLICY_BLIND`
//! (default `target/cost_policy/policy.bin`, `target/cost_policy_blind/policy.bin`).
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test learned_cost_eval -- --nocapture

use std::path::PathBuf;

use cybflight_core::params::FirmwareConfig;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_sim::{
    controller::{Controller, MpcIndiController, OwnedCostPolicy},
    plant::QuadPlant,
    rl_env::closest_point_dist_global,
    runner::{MissionRunner, RunnerConfig},
    scenario::{PassCriteria, Scenario, default_sim_params},
    trajectory::MissionSetpoints,
};
use nalgebra::{UnitQuaternion, Vector3};
use vehicle_yaml::mission::{Mission, load_mission};

const MISSIONS: [&str; 13] = [
    "indoor_circle_slow",
    "indoor_circle_mid",
    "indoor_circle_timeopt",
    "indoor_figure8_slow",
    "indoor_figure8_mid",
    "indoor_figure8_timeopt",
    "indoor_slalom_slow",
    "indoor_slalom_mid",
    "indoor_slalom_timeopt",
    "indoor_splits_slow",
    "indoor_splits_mid",
    "indoor_splits_fast",
    "indoor_splits_timeopt",
];
const VEHICLE_YAML: &str = "sakura_bench_leader_1khz";
const INDI_RATE_HZ: f32 = 1000.0;
const SQP_RATE_HZ: f32 = 100.0;
const SQP_N: usize = 20;

/// `SAMPLER_MAX_LAG` overrides the position sampler's lag allowance [s]
/// (YAML: 0.1). Larger lets the MPC slow into corners and hold the line.
fn sampler_params(vp: &FirmwareConfig) -> cybflight_core::trajectory_planning::sampler::PositionSamplerParams {
    let mut p = vp.trajectory.sampler.to_position_sampler_params();
    if let Some(l) = std::env::var("SAMPLER_MAX_LAG").ok().and_then(|v| v.parse::<f32>().ok()) {
        p.max_lag_s = l;
    }
    p
}

/// `MPC_THRUST_FRAC` overrides the controller's collective-thrust ceiling
/// fraction (YAML: 0.75) — controller only; the plant keeps the airframe.
fn thrust_frac_override(vp: &mut FirmwareConfig) {
    if let Some(f) = std::env::var("MPC_THRUST_FRAC").ok().and_then(|v| v.parse::<f32>().ok()) {
        vp.mpc.thrust_frac = f;
    }
}

/// `MPC_DRAG=1` puts the identified rotor drag (`sim: aero_drag`, leader
/// motors 10 N / 4800 rad/s) into the prediction model. Controller-only;
/// the plant is unchanged.
fn apply_model_options(c: &mut MpcIndiController, vp: &FirmwareConfig) {
    if std::env::var("MPC_DRAG").as_deref() == Ok("1") {
        let sim = default_sim_params();
        let m = &vp.airframe.motors[0];
        let c_t = m.max_thrust_n / (m.max_omega_rad_s * m.max_omega_rad_s);
        c.set_rotor_drag(sim.aero_drag, c_t);
    }
    // `MPC_BODYDRAG=<kxy>` models quadratic body drag `[kxy, kxy, 2·kxy]`
    // (the plant's `body_drag` condition uses the same layout).
    if let Some(k) = std::env::var("MPC_BODYDRAG").ok().and_then(|v| v.parse::<f32>().ok()) {
        c.set_body_drag([k, k, 2.0 * k]);
    }
}

/// A sim-only trajectory given directly as degree-7 pieces (JSON written
/// by the CSV fit in docs/learned_mpc_cost.md, "ultra" test), for
/// references that are not MINCO-through-waypoints. `PIECES_JSON=<path>`
/// selects it in place of a mission; the mission name is the file's.
fn load_pieces_json(path: &str) -> (String, Vec3, PiecewisePolynomial) {
    use cybflight_core::trajectory_planning::polynomial::{Polynomial, MAX_COEFFS};
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).expect("pieces json")).unwrap();
    let name = v["name"].as_str().unwrap().to_string();
    let s = &v["start"];
    let start = Vec3::new(s[0].as_f64().unwrap() as f32, s[1].as_f64().unwrap() as f32, s[2].as_f64().unwrap() as f32);
    let pieces: Vec<Polynomial> = v["pieces"].as_array().unwrap().iter().map(|pc| {
        let mut coeffs = [ZERO3; MAX_COEFFS];
        for (i, c) in pc["coeffs"].as_array().unwrap().iter().enumerate() {
            coeffs[i] = Vec3::new(c[0].as_f64().unwrap() as f32, c[1].as_f64().unwrap() as f32, c[2].as_f64().unwrap() as f32);
        }
        Polynomial { degree: 7, duration: pc["duration"].as_f64().unwrap() as f32, coeffs }
    }).collect();
    (name, start, PiecewisePolynomial::from_pieces(&pieces))
}

fn rate_weight_nominal() -> f32 {
    std::env::var("MPC_W_RATE").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0)
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn out_dir() -> PathBuf {
    let dir = root().join("target/sim-out/learned_cost");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn load(name: &str) -> Mission {
    let path = root().join("missions").join(format!("{name}.yaml"));
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    load_mission(name, &yaml).expect("mission loads")
}

fn plan_fixed_time(m: &Mission) -> PiecewisePolynomial {
    let n = m.waypoints.len();
    let mut durations = Vec::with_capacity(n);
    let mut prev = 0.0f32;
    for &t in &m.timestamps {
        durations.push(t - prev);
        prev = t;
    }
    let intermediate: Vec<Vec3> = m.waypoints[..n - 1].iter().map(|w| Vec3::from(*w)).collect();
    let head = [Vec3::from(m.start), ZERO3, ZERO3, ZERO3];
    let tail = [Vec3::from(m.waypoints[n - 1]), ZERO3, ZERO3, ZERO3];
    let mut minco = Box::new(MincoSnap::new(&head, &tail, n));
    minco.solve(&intermediate, &durations);
    minco.get_trajectory()
}

fn vehicle() -> FirmwareConfig {
    let path = root().join("vehicles").join(format!("{VEHICLE_YAML}.yaml"));
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    vehicle_yaml::load(VEHICLE_YAML, &yaml).expect("vehicle loads").params
}

fn scenario_for(name: &str, m: &Mission, traj: PiecewisePolynomial, vp: FirmwareConfig) -> Scenario {
    // `SIM_ROTOR_DRAG_SCALE=<f>` scales the *plant's* rotor-drag
    // coefficients (the controllers keep their nominal model — a pure
    // plant/model mismatch study).
    let mut sim = default_sim_params();
    if let Some(f) = std::env::var("SIM_ROTOR_DRAG_SCALE").ok().and_then(|v| v.parse::<f32>().ok()) {
        for c in sim.aero_drag.iter_mut() {
            *c *= f;
        }
    }
    Scenario {
        name: name.to_string(),
        vehicle_params: vp,
        sim_params: sim,
        initial_position: Vector3::from(m.start),
        initial_velocity: Vector3::zeros(),
        initial_attitude: UnitQuaternion::identity(),
        setpoints: Box::new(MissionSetpoints::from_trajectory(traj)),
        imu_model: Box::new(cybflight_sim::sensors::PerfectImu),
        gps_model: None,
        rotor_model: Box::new(cybflight_sim::sensors::PerfectRotorTelemetry),
        pass_criteria: PassCriteria {
            // Wide enough for the outdoor missions and the sim-only "ultra"
            // reference (±190 m, 100 m altitude).
            geofence_min: Vector3::new(-300.0, -300.0, -1.0),
            geofence_max: Vector3::new(300.0, 300.0, 150.0),
            ..PassCriteria::default()
        },
        terminal_hold_s: 3.0,
    }
}

fn controller(vp: &FirmwareConfig, traj: &PiecewisePolynomial, policy: Option<&str>) -> MpcIndiController {
    let mut vpc = vp.clone();
    thrust_frac_override(&mut vpc);
    let vp = &vpc;
    let mut c = MpcIndiController::with_options(vp, vp.mpc.pos_cost_mode, SQP_RATE_HZ, INDI_RATE_HZ, SQP_N);
    c.use_tilt_map = true;
    c.flatness_feedforward = true;
    // Thrust weight from the YAML; nominal rate weight from `MPC_W_RATE`
    // (default 5, the v7 nominal per the sweep; 20 = the YAML's flown value).
    let r = rate_weight_nominal();
    c.set_input_weights([vp.mpc.thrust_weight, r, r, r]);
    c.attach_position_sampler(traj.clone(), sampler_params(vp));
    apply_model_options(&mut c, vp);
    if let Some(p) = policy {
        c.set_cost_policy(OwnedCostPolicy::from_file(p).expect("policy file"));
        c.log_cost = true;
        // `COST_POLICY_DELAY=1` mirrors the firmware's latency-first
        // ordering: z computed at solve k applies at solve k+1.
        c.cost_policy_delayed = std::env::var("COST_POLICY_DELAY").as_deref() == Ok("1");
    }
    c
}

fn run_case(name: &str, tag: &str, mut scenario: Scenario, traj: &PiecewisePolynomial, c: &mut MpcIndiController) -> serde_json::Value {
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
    let runner = MissionRunner::new(RunnerConfig { dt_sim: 1.0 / 8000.0, max_sim_time_s: 120.0, history_rate_hz: 100.0 });
    let out = runner.run(&mut scenario, &mut plant, c);
    let s = &out.summary;
    // Geometric RMSE over the mission (closest point within ±0.5 s of
    // the mission clock), on the same records the time-indexed RMSE uses.
    let dur = traj.total_duration();
    let (mut sum_sq, mut n, mut peak) = (0.0f64, 0usize, 0.0f32);
    for r in out.history.iter().filter(|r| r.in_mission && r.t <= dur) {
        let e = closest_point_dist_global(traj, r.position);
        sum_sq += (e * e) as f64;
        n += 1;
        peak = peak.max(e);
    }
    let rms_geom = if n > 0 { (sum_sq / n as f64).sqrt() as f32 } else { 0.0 };
    let completed = s.early_exit_reason.is_none() && s.terminal_pos_err_m < 0.15;
    println!(
        "{:<24} [{:<8}] rms_time={:.4}m rms_geom={:.4}m peak_geom={:.4}m peak_time={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {}",
        name, tag, s.rms_pos_err_m, rms_geom, peak, s.peak_pos_err_m, s.terminal_pos_err_m,
        s.peak_tilt_rad.to_degrees(), s.peak_motor_saturation_pct,
        if completed { "COMPLETED".to_string() } else { format!("INCOMPLETE ({:?})", s.early_exit_reason) }
    );
    let hist: Vec<serde_json::Value> = out
        .history
        .iter()
        .map(|r| {
            serde_json::json!({
                "t": r.t,
                "pos": [r.position.x, r.position.y, r.position.z],
                "sp_pos": [r.setpoint.position.x, r.setpoint.position.y, r.setpoint.position.z],
                "tilt_rad": r.tilt_rad,
                "in_mission": r.in_mission,
            })
        })
        .collect();
    let zlog: Vec<serde_json::Value> = c
        .cost_log
        .iter()
        .map(|(z, o)| serde_json::json!({"z": z.to_vec(), "obs": o.to_vec()}))
        .collect();
    std::fs::write(
        out_dir().join(format!("trace__{name}__{tag}.json")),
        serde_json::to_string(&serde_json::json!({"mission": name, "tag": tag, "history": hist, "cost_log": zlog})).unwrap(),
    )
    .unwrap();
    serde_json::json!({
        "mission": name, "tag": tag, "completed": completed,
        "rms_pos_err_m": s.rms_pos_err_m, "rms_geom_m": rms_geom, "peak_geom_m": peak,
        "peak_pos_err_m": s.peak_pos_err_m, "terminal_pos_err_m": s.terminal_pos_err_m,
        "peak_tilt_deg": s.peak_tilt_rad.to_degrees(),
        "peak_motor_saturation_pct": s.peak_motor_saturation_pct,
        "early_exit": s.early_exit_reason,
    })
}

#[test]
fn learned_cost_vs_nominal() {
    let vp = vehicle();
    let policy = std::env::var("COST_POLICY").unwrap_or_else(|_| root().join("target/cost_policy/policy.bin").to_string_lossy().into_owned());
    let blind = std::env::var("COST_POLICY_BLIND").unwrap_or_else(|_| root().join("target/cost_policy_blind/policy.bin").to_string_lossy().into_owned());
    let have_policy = std::path::Path::new(&policy).exists();
    let have_blind = std::path::Path::new(&blind).exists();
    if !have_policy {
        println!("no policy at {policy}: running nominal only");
    }
    let only: Option<String> = std::env::var("MISSION").ok();
    // `MISSION_SET=all` evaluates every file in `missions/` (indoor and
    // outdoor); default is the 13 indoor missions.
    let names: Vec<String> = if std::env::var("MISSION_SET").as_deref() == Ok("all") {
        let mut v: Vec<String> = std::fs::read_dir(root().join("missions"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
            .collect();
        v.sort();
        v
    } else {
        MISSIONS.iter().map(|s| s.to_string()).collect()
    };
    // `MPC_W_RATE_CURRENT=<w>` adds a "current" row: the nominal state
    // weights with that rate weight and no policy (the flown tune).
    let current: Option<f32> = std::env::var("MPC_W_RATE_CURRENT").ok().and_then(|v| v.parse().ok());
    let mut rows = Vec::new();
    if let Ok(pj) = std::env::var("PIECES_JSON") {
        let (name, start, traj) = load_pieces_json(&pj);
        let m = Mission { start: [start.x, start.y, start.z], ..load("indoor_circle_slow") };
        let name = name.as_str();
        if let Some(r) = current {
            let mut c = controller(&vp, &traj, None);
            c.set_input_weights([vp.mpc.thrust_weight, r, r, r]);
            rows.push(run_case(name, "current", scenario_for(name, &m, traj.clone(), vp.clone()), &traj, &mut c));
        }
        let mut c = controller(&vp, &traj, None);
        rows.push(run_case(name, "nominal", scenario_for(name, &m, traj.clone(), vp.clone()), &traj, &mut c));
        if have_policy {
            let mut c = controller(&vp, &traj, Some(&policy));
            rows.push(run_case(name, "learned", scenario_for(name, &m, traj.clone(), vp.clone()), &traj, &mut c));
        }
        std::fs::write(out_dir().join("summary.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        return;
    }
    for name in &names {
        let name = name.as_str();
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let m = load(name);
        let traj = plan_fixed_time(&m);
        if let Some(r) = current {
            let mut c = controller(&vp, &traj, None);
            c.set_input_weights([vp.mpc.thrust_weight, r, r, r]);
            rows.push(run_case(name, "current", scenario_for(name, &m, traj.clone(), vp.clone()), &traj, &mut c));
        }
        let mut c = controller(&vp, &traj, None);
        rows.push(run_case(name, "nominal", scenario_for(name, &m, traj.clone(), vp.clone()), &traj, &mut c));
        if have_policy {
            let mut c = controller(&vp, &traj, Some(&policy));
            rows.push(run_case(name, "learned", scenario_for(name, &m, traj.clone(), vp.clone()), &traj, &mut c));
        }
        if have_blind {
            let mut c = controller(&vp, &traj, Some(&blind));
            rows.push(run_case(name, "blind", scenario_for(name, &m, traj.clone(), vp.clone()), &traj, &mut c));
        }
    }
    std::fs::write(out_dir().join("summary.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
    // The nominal stack is expected to complete the held-out mission;
    // with a large sampler lag it can miss the terminal gate without any
    // tracking problem, so this is a warning, not a failure.
    for r in &rows {
        if r["mission"] == "indoor_splits_timeopt" && r["tag"] == "nominal" && only.is_none()
            && !r["completed"].as_bool().unwrap()
        {
            println!("warning: nominal stack did not complete the held-out mission (terminal gate)");
        }
    }
}
