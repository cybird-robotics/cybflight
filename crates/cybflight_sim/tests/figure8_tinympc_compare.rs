//! TinyMPC vs. SQP-NMPC tracking comparison on every indoor mission
//! (`missions/indoor_{circle,figure8,slalom,splits}_*.yaml`).
//!
//! Both stacks share the INDI inner loop, the reference chain and the
//! vehicle (`vehicles/sakura_bench_leader_1khz.yaml`: flight airframe and
//! its flown MPC tune, see [`vehicle`]). The only difference is the outer
//! solver:
//!
//! - `mpc_indi`     — `SimpleSqpSolver` over the nonlinear `QuadModel`
//! - `tinympc_indi` — `TinyMpc` (ADMM) over the hover-linearised model
//!
//! Prints one summary row per (mission, controller) and writes
//! `target/sim-out/figure8_tinympc/{summary.json, trace__*.json}` for
//! `analysis/plot_figure8_compare.py`.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test figure8_tinympc_compare -- --nocapture

use std::path::PathBuf;

use cybflight_core::params::FirmwareConfig;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_sim::{
    controller::{Controller, GeometricIndiController, MpcIndiController, TinyMpcIndiController},
    plant::QuadPlant,
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

fn out_dir() -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/sim-out/figure8_tinympc");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn load(name: &str) -> Mission {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../missions")
        .join(format!("{name}.yaml"));
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    load_mission(name, &yaml).expect("mission loads")
}

/// MINCO fit through the mission waypoints at the mission's own
/// timestamps — the same fixed-time plan the firmware bakes.
fn plan_fixed_time(m: &Mission) -> MissionSetpoints {
    let n = m.waypoints.len();
    let mut durations = Vec::with_capacity(n);
    let mut prev = 0.0f32;
    for &t in &m.timestamps {
        durations.push(t - prev);
        prev = t;
    }
    let intermediate: Vec<Vec3> = m.waypoints[..n - 1]
        .iter()
        .map(|w| Vec3::from(*w))
        .collect();
    let head = [Vec3::from(m.start), ZERO3, ZERO3, ZERO3];
    let tail = [Vec3::from(m.waypoints[n - 1]), ZERO3, ZERO3, ZERO3];
    let mut minco = Box::new(MincoSnap::new(&head, &tail, n));
    minco.solve(&intermediate, &durations);
    MissionSetpoints::from_trajectory(minco.get_trajectory())
}

fn scenario_for(name: &str, m: &Mission, vp: FirmwareConfig) -> Scenario {
    Scenario {
        name: name.to_string(),
        vehicle_params: vp,
        sim_params: default_sim_params(),
        initial_position: Vector3::from(m.start),
        initial_velocity: Vector3::zeros(),
        initial_attitude: UnitQuaternion::identity(),
        setpoints: Box::new(plan_fixed_time(m)),
        imu_model: Box::new(cybflight_sim::sensors::PerfectImu),
        gps_model: None,
        rotor_model: Box::new(cybflight_sim::sensors::PerfectRotorTelemetry),
        pass_criteria: PassCriteria {
            geofence_min: Vector3::new(-15.0, -15.0, -1.0),
            geofence_max: Vector3::new(15.0, 15.0, 15.0),
            ..PassCriteria::default()
        },
        terminal_hold_s: 3.0,
    }
}

/// Vehicle under test: `vehicles/sakura_bench_leader_1khz.yaml` — the
/// flight airframe (0.6 kg, 4 × 10 N, `max_rate [10, 10, 6]`) and its
/// flown MPC tune (contouring cost, `thrust_frac 0.75`, tilt fence at
/// 178° / τ 0.5 — effectively unconstrained, RTI). Used unchanged for the
/// plant, the SQP stack and the TinyMPC weights/bounds.
///
/// The leader YAML carries no `sim:` section (rotor drag, ESC idle
/// floor, throttle curvature are plant-only physics the firmware never
/// reads), so the plant takes those from `vehicles/sim_baseline.yaml`.
const VEHICLE_YAML: &str = "sakura_bench_leader_1khz";
/// Inner-loop (INDI) rate for both stacks.
const INDI_RATE_HZ: f32 = 1000.0;
/// SQP-NMPC outer solve rate (the leader's `mpc_rate_hz`; horizon spacing
/// stays `mpc_dt` = 50 ms). TinyMPC runs the paper's 500 Hz / N = 15.
const SQP_RATE_HZ: f32 = 100.0;
/// SQP horizon in stages (the firmware's 20).
const SQP_N: usize = 20;

fn vehicle() -> FirmwareConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vehicles")
        .join(format!("{VEHICLE_YAML}.yaml"));
    let yaml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    vehicle_yaml::load(VEHICLE_YAML, &yaml)
        .expect("vehicle loads")
        .params
}

/// `tag` names the trace file (`trace__<mission>__<tag>.json`); the main
/// comparison passes the controller name, sweeps pass their variant label
/// so they never clobber the traces the plot script reads.
fn run_case(
    mission_name: &str,
    tag: &str,
    mut scenario: Scenario,
    controller: &mut dyn Controller,
) -> serde_json::Value {
    let mut plant = QuadPlant::new(
        scenario.vehicle_params.clone(),
        &scenario.sim_params,
        1.0 / 8000.0,
    );
    let runner = MissionRunner::new(RunnerConfig {
        dt_sim: 1.0 / 8000.0,
        max_sim_time_s: 120.0,
        history_rate_hz: 100.0,
    });
    let out = runner.run(&mut scenario, &mut plant, controller);
    let s = &out.summary;
    let name = controller.name();
    let mean_solve_us = controller
        .solve_stats()
        .filter(|(n, _)| *n > 0)
        .map(|(n, t)| t.as_secs_f64() * 1e6 / n as f64);
    let completed = s.early_exit_reason.is_none() && s.terminal_pos_err_m < 0.15;
    println!(
        "{:<24} [{:<13}] rms={:.4}m peak={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {}",
        mission_name,
        name,
        s.rms_pos_err_m,
        s.peak_pos_err_m,
        s.terminal_pos_err_m,
        s.peak_tilt_rad.to_degrees(),
        s.peak_motor_saturation_pct,
        if completed {
            "COMPLETED".to_string()
        } else {
            format!("INCOMPLETE ({:?})", s.early_exit_reason)
        }
    );
    let hist: Vec<serde_json::Value> = out
        .history
        .iter()
        .map(|r| {
            serde_json::json!({
                "t": r.t,
                "pos": [r.position.x, r.position.y, r.position.z],
                "vel": [r.velocity.x, r.velocity.y, r.velocity.z],
                "tilt_rad": r.tilt_rad,
                "sp_pos": [r.setpoint.position.x, r.setpoint.position.y, r.setpoint.position.z],
                "sp_vel": [r.setpoint.velocity.x, r.setpoint.velocity.y, r.setpoint.velocity.z],
                "cmd": r.motor_commands,
                "in_mission": r.in_mission,
            })
        })
        .collect();
    std::fs::write(
        out_dir().join(format!("trace__{mission_name}__{tag}.json")),
        serde_json::to_string(
            &serde_json::json!({"mission": mission_name, "controller": name, "history": hist}),
        )
        .unwrap(),
    )
    .unwrap();
    serde_json::json!({
        "mission": mission_name,
        "controller": name,
        "completed": completed,
        "rms_pos_err_m": s.rms_pos_err_m,
        "peak_pos_err_m": s.peak_pos_err_m,
        "terminal_pos_err_m": s.terminal_pos_err_m,
        "peak_tilt_deg": s.peak_tilt_rad.to_degrees(),
        "peak_motor_saturation_pct": s.peak_motor_saturation_pct,
        "trajectory_duration_s": s.trajectory_duration_s,
        "mean_solve_us": mean_solve_us,
        "early_exit": s.early_exit_reason,
    })
}

#[test]
fn indoor_tinympc_vs_nmpc() {
    let vp = vehicle();
    let mut rows = Vec::new();
    for name in MISSIONS {
        let m = load(name);
        // Firmware-faithful SQP stack: tilt-yaw q_ref, flatness u_ref
        // feedforward and the position sampler — what the leader flies
        // (see `sqp_fidelity_ablation` and docs/tinympc_comparison.md).
        let mut sqp = MpcIndiController::with_options(
            &vp,
            vp.mpc.pos_cost_mode,
            SQP_RATE_HZ,
            INDI_RATE_HZ,
            SQP_N,
        );
        sqp.use_tilt_map = true;
        sqp.flatness_feedforward = true;
        sqp.attach_position_sampler(
            plan_fixed_time(&m).trajectory().clone(),
            vp.trajectory.sampler.to_position_sampler_params(),
        );
        rows.push(run_case(
            name,
            "mpc_indi",
            scenario_for(name, &m, vp.clone()),
            &mut sqp,
        ));
        let mut tiny = TinyMpcIndiController::with_settings(
            &vp,
            TinyMpcIndiController::DEFAULT_RHO,
            cybflight_core::mpc::TinySettings::default(),
            INDI_RATE_HZ,
        );
        rows.push(run_case(
            name,
            "tinympc_indi",
            scenario_for(name, &m, vp.clone()),
            &mut tiny,
        ));
        // TinyMPC with its own reference feedforward (`u_ref` = reference
        // thrust + body rates), the linear-MPC counterpart of the SQP's
        // flatness feedforward.
        let mut tiny_ff = TinyMpcIndiController::with_settings(
            &vp,
            TinyMpcIndiController::DEFAULT_RHO,
            cybflight_core::mpc::TinySettings::default(),
            INDI_RATE_HZ,
        );
        tiny_ff.reference_feedforward = true;
        rows.push(run_case(
            name,
            "tinympc_ff_indi",
            scenario_for(name, &m, vp.clone()),
            &mut tiny_ff,
        ));
        // Geometric tracking controller (RPG position-controller port),
        // gains from `sweep_geometric`.
        let mut geo = GeometricIndiController::with_params(
            &vp,
            GeometricIndiController::tuned_params(&vp),
            INDI_RATE_HZ,
        );
        rows.push(run_case(
            name,
            "geometric_indi",
            scenario_for(name, &m, vp.clone()),
            &mut geo,
        ));
    }
    std::fs::write(
        out_dir().join("summary.json"),
        serde_json::to_string_pretty(&rows).unwrap(),
    )
    .unwrap();
    // Regression guard on the one cell every configuration so far has
    // passed: both stacks complete `indoor_figure8_slow`. Every other
    // cell is reported, not asserted (see docs/tinympc_comparison.md).
    for r in &rows {
        if r["mission"] == "indoor_figure8_slow" {
            assert!(
                r["completed"].as_bool().unwrap(),
                "{} / {} did not complete: {}",
                r["mission"],
                r["controller"],
                r["early_exit"]
            );
        }
    }
}

/// Exploratory sweep (ignored by default): SQP variations around the
/// leader tune on the time-optimal profile.
#[test]
#[ignore]
fn sweep_sqp_timeopt() {
    use cybflight_core::mpc::quad_model::PosCostMode;
    let m = load("indoor_figure8_timeopt");
    let variants: [(&str, Box<dyn Fn(&mut FirmwareConfig)>); 6] = [
        ("leader", Box::new(|_| {})),
        ("iters5", Box::new(|vp| vp.mpc.max_iters = 5)),
        (
            "quadratic",
            Box::new(|vp| vp.mpc.pos_cost_mode = PosCostMode::Quadratic),
        ),
        ("fence_off", Box::new(|vp| vp.mpc.tilt_barrier_tau = 0.0)),
        ("thrust_frac1", Box::new(|vp| vp.mpc.thrust_frac = 1.0)),
        (
            "rate16",
            Box::new(|vp| vp.airframe.body.max_rate_rad_s = [16.0, 16.0, 8.0]),
        ),
    ];
    for (label, f) in variants {
        let mut vp = vehicle();
        f(&mut vp);
        let mut sqp = MpcIndiController::with_options(
            &vp,
            vp.mpc.pos_cost_mode,
            SQP_RATE_HZ,
            INDI_RATE_HZ,
            SQP_N,
        );
        print!("{label:<14} ");
        run_case(
            "indoor_figure8_timeopt",
            &format!("sweep_sqp_{label}"),
            scenario_for("indoor_figure8_timeopt", &m, vp),
            &mut sqp,
        );
    }
}

/// Exploratory (ignored): reference demand of each figure-8 profile.
#[test]
#[ignore]
fn demand_probe() {
    use cybflight_sim::trajectory::SetpointSource;
    let vp = vehicle();
    let mass = vp.airframe.body.mass_kg;
    let g = 9.81f32;
    let ceiling: f32 = vp.airframe.motors.iter().map(|m| m.max_thrust_n).sum();
    for name in MISSIONS {
        let m = load(name);
        let mut src = plan_fixed_time(&m);
        let dur = src.duration_s();
        let dt = 0.01f32;
        let (mut pk_t, mut pk_v, mut pk_tilt, mut pk_zmin) = (0.0f32, 0.0f32, 0.0f32, 10.0f32);
        let mut q_prev: Option<UnitQuaternion<f32>> = None;
        let mut pk_rate = 0.0f32;
        for i in 0..(dur / dt) as usize {
            let s = src.sample(i as f32 * dt);
            let acc = Vector3::new(s.acceleration.x, s.acceleration.y, s.acceleration.z + g);
            pk_t = pk_t.max(mass * acc.norm());
            pk_v = pk_v.max(s.velocity.norm());
            pk_tilt = pk_tilt.max((acc.z / acc.norm()).acos().to_degrees());
            pk_zmin = pk_zmin.min(s.position.z);
            let zb = acc / acc.norm();
            let q = cybflight_core::rotation::quaternion_from_zb_and_yaw(&zb, 0.0, true);
            if let Some(qp) = q_prev {
                pk_rate = pk_rate.max((qp.inverse() * q).scaled_axis().norm() / dt);
            }
            q_prev = Some(q);
        }
        println!(
            "{name:<24} dur={dur:.2}s peak thrust {pk_t:.1}N ({:.0}% of {ceiling:.0}N) v={pk_v:.1}m/s tilt={pk_tilt:.0}° rate={pk_rate:.1}rad/s zmin={pk_zmin:.2}",
            pk_t / ceiling * 100.0
        );
    }
}

/// Exploratory (ignored): TinyMPC ρ / feedforward sweep on slow + mid.
#[test]
#[ignore]
fn sweep_tinympc() {
    use cybflight_core::mpc::TinySettings;
    let vp = vehicle();
    for name in ["indoor_figure8_slow", "indoor_figure8_mid"] {
        let m = load(name);
        for ff in [false, true] {
            for rho in [1.0f32, 5.0, 20.0, 100.0] {
                let mut c = TinyMpcIndiController::with_settings(
                    &vp,
                    rho,
                    TinySettings::default(),
                    INDI_RATE_HZ,
                );
                c.reference_feedforward = ff;
                print!("ff={ff:<5} rho={rho:<5} ");
                run_case(
                    name,
                    &format!("sweep_tiny_ff{ff}_rho{rho}"),
                    scenario_for(name, &m, vp.clone()),
                    &mut c,
                );
            }
        }
    }
}

/// Exploratory (ignored): sim-vs-flight fidelity ablation for the SQP
/// stack. The real leader flies the time-optimal missions (blackbox in
/// `analysis/datasets/indoor_exp_timeopt`) with three reference-chain
/// mechanisms the frozen sim controller lacks — tilt-yaw `q_ref`,
/// flatness `u_ref` feedforward, position sampler — plus a thrust-curve
/// mismatch (leader YAML leaves `m*_nonlin` at the 0 sentinel → sim INDI
/// inverts k = 0.025 against the plant's k = 0.95). Switch them on one at
/// a time and together.
#[test]
#[ignore]
fn sqp_fidelity_ablation() {
    let configs: [(&str, bool, bool, bool, bool); 7] = [
        // label, tilt_map, feedforward, position_sampler, nonlin_fix
        ("base", false, false, false, false),
        ("tilt", true, false, false, false),
        ("ff", false, true, false, false),
        ("sampler", false, false, true, false),
        ("tilt+ff+sampler", true, true, true, false),
        ("nonlin", false, false, false, true),
        ("all", true, true, true, true),
    ];
    for name in MISSIONS {
        let m = load(name);
        for (label, tilt, ff, sampler, nonlin) in configs {
            let mut vp = vehicle();
            if nonlin {
                for mo in &mut vp.airframe.motors {
                    mo.nonlinearity = 0.8;
                }
            }
            let mut c = MpcIndiController::with_options(
                &vp,
                vp.mpc.pos_cost_mode,
                SQP_RATE_HZ,
                INDI_RATE_HZ,
                SQP_N,
            );
            c.use_tilt_map = tilt;
            c.flatness_feedforward = ff;
            if sampler {
                let sp = plan_fixed_time(&m);
                c.attach_position_sampler(
                    sp.trajectory().clone(),
                    vp.trajectory.sampler.to_position_sampler_params(),
                );
            }
            print!("{label:<16} ");
            run_case(
                name,
                &format!("ablation_{label}"),
                scenario_for(name, &m, vp),
                &mut c,
            );
        }
    }
}

/// Exploratory (ignored): gain sweep for the geometric tracking controller
/// on the leader vehicle. Grid over the reference implementation's
/// position/velocity/attitude gains and two error-saturation levels;
/// traces are tagged `geo_<label>` so `analysis/` scripts can rank by
/// geometric error.
#[test]
#[ignore]
fn sweep_geometric() {
    use cybflight_core::attitude_control::geometric_controller::GeometricTrackingParams;
    let vp = vehicle();
    let missions = [
        "indoor_figure8_slow",
        "indoor_figure8_mid",
        "indoor_figure8_timeopt",
        "indoor_splits_mid",
        "indoor_splits_fast",
    ];
    let loaded: Vec<_> = missions.iter().map(|m| (*m, load(m))).collect();
    for kp in [6.0f32, 10.0, 16.0] {
        for kd in [4.0f32, 6.0, 8.0] {
            for krp in [8.0f32, 12.0, 20.0] {
                for (sat, pxy, vxy, pz, vz) in [
                    ("tight", 0.6f32, 1.0f32, 0.3f32, 0.75f32),
                    ("loose", 2.0, 4.0, 1.0, 3.0),
                ] {
                    let params = GeometricTrackingParams {
                        kpxy: kp,
                        kdxy: kd,
                        kpz: kp * 1.5,
                        kdz: kd * 1.5,
                        krp,
                        kyaw: 5.0,
                        pxy_error_max: pxy,
                        vxy_error_max: vxy,
                        pz_error_max: pz,
                        vz_error_max: vz,
                        min_normalized_thrust: 1.0,
                        gravity: vp.site.gravity_m_s2,
                    };
                    let label = format!("kp{kp}_kd{kd}_krp{krp}_{sat}");
                    for (name, m) in &loaded {
                        let mut c = GeometricIndiController::with_params(&vp, params, INDI_RATE_HZ);
                        print!("{label:<26} ");
                        run_case(
                            name,
                            &format!("geo_{label}"),
                            scenario_for(name, m, vp.clone()),
                            &mut c,
                        );
                    }
                }
            }
        }
    }
}

/// Second, finer geometric sweep around the coarse optimum (kp 16 / kd 8 /
/// krp 20, loose saturation).
#[test]
#[ignore]
fn sweep_geometric_fine() {
    use cybflight_core::attitude_control::geometric_controller::GeometricTrackingParams;
    let vp = vehicle();
    let missions = [
        "indoor_figure8_slow",
        "indoor_figure8_mid",
        "indoor_figure8_timeopt",
        "indoor_splits_mid",
        "indoor_splits_fast",
    ];
    let loaded: Vec<_> = missions.iter().map(|m| (*m, load(m))).collect();
    for kp in [16.0f32, 24.0, 32.0] {
        for kd in [8.0f32, 10.0, 12.0] {
            for krp in [20.0f32, 30.0] {
                for (sat, pxy, vxy, pz, vz) in [
                    ("loose", 2.0f32, 4.0f32, 1.0f32, 3.0f32),
                    ("open", 5.0, 8.0, 3.0, 6.0),
                ] {
                    let params = GeometricTrackingParams {
                        kpxy: kp,
                        kdxy: kd,
                        kpz: kp * 1.5,
                        kdz: kd * 1.5,
                        krp,
                        kyaw: 5.0,
                        pxy_error_max: pxy,
                        vxy_error_max: vxy,
                        pz_error_max: pz,
                        vz_error_max: vz,
                        min_normalized_thrust: 1.0,
                        gravity: vp.site.gravity_m_s2,
                    };
                    let label = format!("f_kp{kp}_kd{kd}_krp{krp}_{sat}");
                    for (name, m) in &loaded {
                        let mut c = GeometricIndiController::with_params(&vp, params, INDI_RATE_HZ);
                        print!("{label:<28} ");
                        run_case(
                            name,
                            &format!("geo_{label}"),
                            scenario_for(name, m, vp.clone()),
                            &mut c,
                        );
                    }
                }
            }
        }
    }
}
