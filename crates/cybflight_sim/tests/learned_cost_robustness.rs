//! Robustness of the learned cost policy (docs/learned_mpc_cost.md) to
//! localization error and plant/model mismatch, against the nominal tune.
//!
//! Neither perturbation was seen in training in this form: training used
//! ground-truth state and only a ±10 % plant mass jitter.
//!
//! * **Localization error**: the 13-state the outer loop consumes is
//!   corrupted every tick with white Gaussian noise on position, velocity
//!   and attitude plus a slowly drifting position bias (first-order random
//!   walk, 1 s correlation). INDI still runs on the (perfect) IMU — the
//!   corruption models the estimator, not the gyro.
//! * **Model mismatch**: the plant is built from a perturbed vehicle
//!   (mass, per-motor thrust ceiling, motor time constant, rotor drag) while
//!   the controller — MPC model, INDI effectiveness, feed-forward — keeps
//!   the nominal one.
//!
//! Writes `target/sim-out/learned_cost/robustness.json`. Policy via
//! `COST_POLICY` (default `target/cost_policy_v7/policy.bin`);
//! `MISSIONS=a,b` narrows the mission set.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test learned_cost_robustness -- --nocapture

use std::path::PathBuf;

use cybflight_core::mpc::{NU, NX};
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
    sensors::{ImuMeasurement, RotorTelemetry},
    trajectory::{MissionSetpoints, Setpoint},
};
use nalgebra::{Quaternion, SVector, UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use vehicle_yaml::mission::{Mission, load_mission};

const VEHICLE_YAML: &str = "simulation/research_1khz";
const INDI_RATE_HZ: f32 = 1000.0;
const SQP_RATE_HZ: f32 = 100.0;
const SQP_N: usize = 20;
const DEFAULT_MISSIONS: [&str; 4] = [
    "indoor_splits_timeopt",
    "indoor_figure8_timeopt",
    "indoor_slalom_mid",
    "indoor_circle_mid",
];

/// Localization-error model applied to the controller's state.
#[derive(Clone, Copy, Debug, Default)]
struct LocNoise {
    /// White gyro noise into INDI σ [rad/s] (accel σ = 10× in m/s²).
    gyro: f32,
    /// White position noise σ [m].
    sigma_p: f32,
    /// White velocity noise σ [m/s].
    sigma_v: f32,
    /// White attitude noise σ [rad] (small random rotation).
    sigma_th: f32,
    /// Stationary σ of the drifting position bias [m] (1 s correlation).
    bias_p: f32,
}

/// Plant-only perturbation (controller keeps the nominal vehicle).
#[derive(Clone, Copy, Debug)]
struct Mismatch {
    mass: f32,
    thrust: f32,
    motor_tau: f32,
    /// Scale on the baseline (linear, rotor-speed-proportional) drag.
    drag: f32,
    /// Absolute quadratic body drag `½ρC_dA` [N·s²/m²], `[xy, z]`
    /// (plant-only; the baseline has none).
    body_drag: [f32; 2],
}

impl Default for Mismatch {
    fn default() -> Self {
        Self { mass: 1.0, thrust: 1.0, motor_tau: 1.0, drag: 1.0, body_drag: [0.0, 0.0] }
    }
}

struct Condition {
    name: &'static str,
    noise: LocNoise,
    mismatch: Mismatch,
}

fn conditions() -> Vec<Condition> {
    let n = LocNoise::default;
    let m = Mismatch::default;
    vec![
        Condition { name: "clean", noise: n(), mismatch: m() },
        Condition { name: "gyro_0.03", noise: LocNoise { gyro: 0.03, ..n() }, mismatch: m() },
        Condition { name: "gyro_0.06", noise: LocNoise { gyro: 0.06, ..n() }, mismatch: m() },
        Condition { name: "loc_3cm", noise: LocNoise { gyro: 0.0, sigma_p: 0.03, sigma_v: 0.10, sigma_th: 0.017, bias_p: 0.0 }, mismatch: m() },
        Condition { name: "loc_8cm_drift10cm", noise: LocNoise { gyro: 0.0, sigma_p: 0.08, sigma_v: 0.25, sigma_th: 0.035, bias_p: 0.10 }, mismatch: m() },
        Condition { name: "mass+15%", noise: n(), mismatch: Mismatch { mass: 1.15, ..m() } },
        Condition { name: "mass-15%", noise: n(), mismatch: Mismatch { mass: 0.85, ..m() } },
        Condition { name: "thrust-15%", noise: n(), mismatch: Mismatch { thrust: 0.85, ..m() } },
        Condition { name: "motor_tau x1.5", noise: n(), mismatch: Mismatch { motor_tau: 1.5, ..m() } },
        Condition { name: "motor_tau x2", noise: n(), mismatch: Mismatch { motor_tau: 2.0, ..m() } },
        Condition { name: "drag x3", noise: n(), mismatch: Mismatch { drag: 3.0, ..m() } },
        // Quadratic body drag, ½ρC_dA: 0.006 / 0.012 / 0.024 N·s²/m² in xy
        // (C_dA ≈ 0.01 / 0.02 / 0.04 m²), twice that on z (larger planform).
        // At 12 m/s the middle level is ≈ 1.7 N ≈ 2.9 m/s² on this 0.6 kg frame.
        Condition { name: "bodydrag 0.006", noise: n(), mismatch: Mismatch { body_drag: [0.006, 0.012], ..m() } },
        Condition { name: "bodydrag 0.012", noise: n(), mismatch: Mismatch { body_drag: [0.012, 0.024], ..m() } },
        Condition { name: "bodydrag 0.024", noise: n(), mismatch: Mismatch { body_drag: [0.024, 0.048], ..m() } },
        Condition {
            name: "combined",
            noise: LocNoise { gyro: 0.03, sigma_p: 0.03, sigma_v: 0.10, sigma_th: 0.017, bias_p: 0.05 },
            mismatch: Mismatch { mass: 1.10, thrust: 0.90, motor_tau: 1.5, drag: 2.0, body_drag: [0.006, 0.012] },
        },
    ]
}

/// Wraps the MPC+INDI stack and corrupts the state it sees.
struct NoisyState {
    inner: MpcIndiController,
    noise: LocNoise,
    rng: ChaCha8Rng,
    bias: Vector3<f32>,
}

impl NoisyState {
    fn gauss(&mut self) -> f32 {
        StandardNormal.sample(&mut self.rng)
    }
}

impl Controller for NoisyState {
    fn name(&self) -> &'static str {
        "mpc_indi_noisy"
    }
    fn tick_rate_hz(&self) -> f32 {
        self.inner.tick_rate_hz()
    }
    fn horizon_samples(&self) -> usize {
        self.inner.horizon_samples()
    }
    fn horizon_stride_s(&self) -> f32 {
        self.inner.horizon_stride_s()
    }
    fn solve_stats(&self) -> Option<(u64, std::time::Duration)> {
        self.inner.solve_stats()
    }
    fn step(
        &mut self,
        x: &SVector<f32, NX>,
        imu: &ImuMeasurement,
        rotor: &RotorTelemetry,
        horizon: &[Setpoint],
    ) -> SVector<f32, NU> {
        let n = self.noise;
        let mut xn = *x;
        // Drifting bias: OU process with 1 s correlation at the tick rate.
        let dt = 1.0 / self.tick_rate_hz();
        let a = (-dt).exp();
        let s = n.bias_p * (1.0 - a * a).sqrt();
        for i in 0..3 {
            let g = self.gauss();
            self.bias[i] = a * self.bias[i] + s * g;
        }
        for i in 0..3 {
            let (gp, gv) = (self.gauss(), self.gauss());
            xn[i] += n.sigma_p * gp + self.bias[i];
            xn[7 + i] += n.sigma_v * gv;
        }
        if n.sigma_th > 0.0 {
            let dth = Vector3::new(self.gauss(), self.gauss(), self.gauss()) * n.sigma_th;
            let q = UnitQuaternion::from_quaternion(Quaternion::new(x[6], x[3], x[4], x[5]));
            let qn = q * UnitQuaternion::from_scaled_axis(dth);
            xn[3] = qn.i;
            xn[4] = qn.j;
            xn[5] = qn.k;
            xn[6] = qn.w;
        }
        self.inner.step(&xn, imu, rotor, horizon)
    }
}

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

fn load(name: &str) -> Mission {
    let path = root().join("missions").join(format!("{name}.yaml"));
    load_mission(name, &std::fs::read_to_string(&path).expect("mission")).expect("mission loads")
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
    let mut minco = Box::new(MincoSnap::new(
        &[Vec3::from(m.start), ZERO3, ZERO3, ZERO3],
        &[Vec3::from(m.waypoints[n - 1]), ZERO3, ZERO3, ZERO3],
        n,
    ));
    minco.solve(&intermediate, &durations);
    minco.get_trajectory()
}

fn vehicle() -> FirmwareConfig {
    let path = root().join("vehicles").join(format!("{VEHICLE_YAML}.yaml"));
    vehicle_yaml::load(VEHICLE_YAML, &std::fs::read_to_string(&path).expect("vehicle"))
        .expect("vehicle loads")
        .params
}

fn perturbed(vp: &FirmwareConfig, mm: &Mismatch) -> FirmwareConfig {
    let mut p = vp.clone();
    p.airframe.body.mass_kg *= mm.mass;
    for m in p.airframe.motors.iter_mut() {
        m.max_thrust_n *= mm.thrust;
        m.time_const_s *= mm.motor_tau;
    }
    p
}

struct Row {
    mission: String,
    condition: &'static str,
    tag: &'static str,
    rms_geom: f32,
    rms_time: f32,
    peak_geom: f32,
    completed: bool,
    early_exit: Option<String>,
}

fn mission_start(name: &str, traj: &PiecewisePolynomial) -> Vector3<f32> {
    if std::env::var("PIECES_JSON").is_ok() {
        let p = traj.get_pos(0.0);
        return Vector3::new(p.x, p.y, p.z);
    }
    Vector3::from(load(name).start)
}

fn run_case(name: &str, cond: &Condition, tag: &'static str, vp_nom: &FirmwareConfig, traj: &PiecewisePolynomial, policy: Option<&str>, seed: u64) -> Row {
    // `MPC_W_RATE_NOMINAL` lets the no-policy baseline fly a different
    // rate weight than the policy's nominal (current tune vs recommendation).
    let rate_nom_baseline: f32 = std::env::var("MPC_W_RATE_NOMINAL").ok().and_then(|v| v.parse().ok()).unwrap_or(rate_weight_nominal());
    let start = mission_start(name, traj);
    let mut sim = default_sim_params();
    sim.aero_drag = [sim.aero_drag[0] * cond.mismatch.drag, sim.aero_drag[1] * cond.mismatch.drag, sim.aero_drag[2] * cond.mismatch.drag];
    sim.body_drag = [cond.mismatch.body_drag[0], cond.mismatch.body_drag[0], cond.mismatch.body_drag[1]];
    let vp_plant = perturbed(vp_nom, &cond.mismatch);
    let mut scenario = Scenario {
        name: name.to_string(),
        vehicle_params: vp_plant,
        sim_params: sim,
        initial_position: start,
        initial_velocity: Vector3::zeros(),
        initial_attitude: UnitQuaternion::identity(),
        setpoints: Box::new(MissionSetpoints::from_trajectory(traj.clone())),
        imu_model: if cond.noise.gyro > 0.0 {
            Box::new(cybflight_sim::sensors::NoisyImu::isotropic(seed ^ 0x51, cond.noise.gyro, 10.0 * cond.noise.gyro))
        } else {
            Box::new(cybflight_sim::sensors::PerfectImu)
        },
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
    };
    let mut vpc = vp_nom.clone();
    thrust_frac_override(&mut vpc);
    let vp_nom = &vpc;
    let mut c = MpcIndiController::with_options(vp_nom, vp_nom.mpc.pos_cost_mode, SQP_RATE_HZ, INDI_RATE_HZ, SQP_N);
    c.use_tilt_map = true;
    c.flatness_feedforward = true;
    // Thrust weight from the YAML; nominal rate weight from `MPC_W_RATE`
    // (default 5, the v7 nominal per the sweep; 20 = the YAML's flown value).
    let r = if policy.is_some() { rate_weight_nominal() } else { rate_nom_baseline };
    c.set_input_weights([vp_nom.mpc.thrust_weight, r, r, r]);
    c.attach_position_sampler(traj.clone(), sampler_params(vp_nom));
    apply_model_options(&mut c, vp_nom);
    if let Some(p) = policy {
        c.set_cost_policy(OwnedCostPolicy::from_file(p).expect("policy file"));
        c.cost_policy_delayed = std::env::var("COST_POLICY_DELAY").as_deref() == Ok("1");
    }
    let mut ctrl = NoisyState { inner: c, noise: cond.noise, rng: ChaCha8Rng::seed_from_u64(seed), bias: Vector3::zeros() };
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
    let runner = MissionRunner::new(RunnerConfig { dt_sim: 1.0 / 8000.0, max_sim_time_s: 120.0, history_rate_hz: 100.0 });
    let out = runner.run(&mut scenario, &mut plant, &mut ctrl);
    let s = &out.summary;
    let dur = traj.total_duration();
    let (mut sum_sq, mut n, mut peak) = (0.0f64, 0usize, 0.0f32);
    for r in out.history.iter().filter(|r| r.in_mission && r.t <= dur) {
        let e = closest_point_dist_global(traj, r.position);
        sum_sq += (e * e) as f64;
        n += 1;
        peak = peak.max(e);
    }
    let rms_geom = if n > 0 { (sum_sq / n as f64).sqrt() as f32 } else { f32::NAN };
    let completed = s.early_exit_reason.is_none() && s.terminal_pos_err_m < 0.15;
    Row {
        mission: name.to_string(),
        condition: cond.name,
        tag,
        rms_geom,
        rms_time: s.rms_pos_err_m,
        peak_geom: peak,
        completed,
        early_exit: s.early_exit_reason.clone(),
    }
}

#[test]
fn robustness_matrix() {
    let vp = vehicle();
    let policy = std::env::var("COST_POLICY")
        .unwrap_or_else(|_| root().join("target/cost_policy_v7/policy.bin").to_string_lossy().into_owned());
    assert!(std::path::Path::new(&policy).exists(), "no policy at {policy}");
    let missions: Vec<String> = std::env::var("MISSIONS")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(|_| DEFAULT_MISSIONS.iter().map(|s| s.to_string()).collect());
    let mut rows = Vec::new();
    println!("{:<24} {:<18} {:>9} {:>9} {:>8} {:>9} {:>9} {:>8}   {}", "mission", "condition", "nom geom", "v2b geom", "Δgeom", "nom time", "v2b time", "peak v2b", "status");
    let pieces = std::env::var("PIECES_JSON").ok().map(|pj| load_pieces_json(&pj));
    let missions: Vec<String> = match &pieces { Some((n, _, _)) => vec![n.clone()], None => missions };
    for name in &missions {
        let traj = match &pieces { Some((_, _, t)) => t.clone(), None => plan_fixed_time(&load(name)) };
        for (ci, cond) in conditions().iter().enumerate() {
            let seed = 1000 + ci as u64;
            let a = run_case(name, cond, "nominal", &vp, &traj, None, seed);
            let b = run_case(name, cond, "v2b", &vp, &traj, Some(&policy), seed);
            let status = format!(
                "{}{}",
                if a.completed { "" } else { "nominal-INCOMPLETE " },
                if b.completed { "" } else { "v2b-INCOMPLETE" }
            );
            println!(
                "{:<24} {:<18} {:>9.4} {:>9.4} {:>+7.1}% {:>9.4} {:>9.4} {:>8.3}   {}",
                name, cond.name, a.rms_geom, b.rms_geom, 100.0 * (b.rms_geom / a.rms_geom - 1.0), a.rms_time, b.rms_time, b.peak_geom, status
            );
            for r in [a, b] {
                rows.push(serde_json::json!({
                    "mission": r.mission, "condition": r.condition, "tag": r.tag,
                    "rms_geom_m": r.rms_geom, "rms_pos_err_m": r.rms_time, "peak_geom_m": r.peak_geom,
                    "completed": r.completed, "early_exit": r.early_exit,
                }));
            }
        }
    }
    let dir = root().join("target/sim-out/learned_cost");
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("robustness.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
}
