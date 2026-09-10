//! TEMPORARY diagnostic harness — why does mpc_full + INDI crash on the
//! hard missions? Runs the failing mission classes with per-solve solver
//! diagnostics (`FullSolveDiag`) and dumps full 100 Hz timelines to JSON
//! for offline analysis.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test mpc_full_failure_diag -- \
//!       --nocapture --test-threads=8

use std::path::PathBuf;

use cybflight_core::params::FirmwareConfig;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_sim::{
    controller::{Controller, MpcFullIndiController, MpcIndiController},
    plant::QuadPlant,
    runner::{MissionRunner, RunnerConfig},
    scenario::{default_sim_params, tweaked_vehicle, PassCriteria, Scenario},
    trajectory::MissionSetpoints,
};
use nalgebra::{UnitQuaternion, Vector3};
use vehicle_yaml::mission::{load_mission, Mission};

fn out_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mpc_full_diag");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn load(name: &str) -> Mission {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../missions")
        .join(format!("{name}.yaml"));
    let yaml = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    load_mission(name, &yaml).expect("mission loads")
}

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

fn geofence(m: &Mission) -> (Vector3<f32>, Vector3<f32>) {
    let mut lo = Vector3::from(m.start);
    let mut hi = lo;
    for w in &m.waypoints {
        for a in 0..3 {
            lo[a] = lo[a].min(w[a]);
            hi[a] = hi[a].max(w[a]);
        }
    }
    let margin = 10.0;
    (
        Vector3::new(lo.x - margin, lo.y - margin, -1.0),
        hi.add_scalar(margin),
    )
}

fn scenario_for(name: &str, m: &Mission, vp: FirmwareConfig) -> Scenario {
    let (gmin, gmax) = geofence(m);
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
            geofence_min: gmin,
            geofence_max: gmax,
            ..PassCriteria::default()
        },
        terminal_hold_s: 3.0,
    }
}

/// Tilt fence off (matches the notilt campaign grid), rate barrier for
/// the full model.
fn vehicle(full_model: bool) -> FirmwareConfig {
    tweaked_vehicle(|vp| {
        vp.mpc.tilt_max_deg = 180.0;
        vp.mpc.tilt_barrier_tau = 0.0;
        if full_model {
            vp.mpc.rate_barrier_tau = 0.5;
            vp.mpc.rate_barrier_delta = 0.5;
        }
    })
}

fn dump<C: Controller>(
    mission_name: &str,
    label: &str,
    mut scenario: Scenario,
    mut controller: C,
    solve_log: impl FnOnce(&C) -> serde_json::Value,
) {
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
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    let hist: Vec<serde_json::Value> = out
        .history
        .iter()
        .map(|r| {
            serde_json::json!({
                "t": r.t,
                "pos": [r.position.x, r.position.y, r.position.z],
                "vel": [r.velocity.x, r.velocity.y, r.velocity.z],
                "quat_wxyz": [r.attitude.w, r.attitude.i, r.attitude.j, r.attitude.k],
                "rate": [r.body_rate.x, r.body_rate.y, r.body_rate.z],
                "tilt_rad": r.tilt_rad,
                "sp_pos": [r.setpoint.position.x, r.setpoint.position.y, r.setpoint.position.z],
                "sp_vel": [r.setpoint.velocity.x, r.setpoint.velocity.y, r.setpoint.velocity.z],
                "sp_acc": [r.setpoint.acceleration.x, r.setpoint.acceleration.y, r.setpoint.acceleration.z],
                "cmd": r.motor_commands,
                "omega": r.rotor_omega,
            })
        })
        .collect();

    let blob = serde_json::json!({
        "mission": mission_name,
        "case": label,
        "summary": {
            "rms": out.summary.rms_pos_err_m,
            "terminal": out.summary.terminal_pos_err_m,
            "peak_tilt_deg": out.summary.peak_tilt_rad.to_degrees(),
            "sat_pct": out.summary.peak_motor_saturation_pct,
            "early_exit": out.summary.early_exit_reason,
            "reasons": out.failure_reasons,
        },
        "solves": solve_log(&controller),
        "history": hist,
    });
    let path = out_dir().join(format!("{mission_name}__{label}.json"));
    std::fs::write(&path, serde_json::to_string(&blob).unwrap()).unwrap();
    println!(
        "{mission_name} [{label}] rms={:.3} early_exit={:?} -> {}",
        out.summary.rms_pos_err_m,
        out.summary.early_exit_reason,
        path.display()
    );
}

fn diag_mission(name: &str) {
    let m = load(name);

    // mpc_full + 8 kHz INDI with solver logging.
    let vp = vehicle(true);
    let mut c = MpcFullIndiController::with_options_at_rate(&vp, 100.0, true, 8000.0);
    c.log_solves = true;
    dump(name, "mpc_full_8k", scenario_for(name, &m, vp), c, |c| {
        serde_json::to_value(&c.solve_log).unwrap()
    });

    // Reduced-stack contrast run (no solver log — history only).
    let vp = vehicle(false);
    let c = MpcIndiController::from_params_at_indi_rate(&vp, 8000.0);
    dump(name, "mpc_8k", scenario_for(name, &m, vp), c, |_| {
        serde_json::Value::Null
    });
}

/// Causal experiments on the collapse mechanism:
///   E1 — SQP depth: max_iters 1 (RTI, baseline) vs 5.
///   E2 — S1 reference completion (Ω_r + u_r feedforward) on/off.
#[test]
fn experiments() {
    for mission_name in [
        "outdoor_splits-large_mid",
        "indoor_splits_fast",
        "outdoor_splits_mid",
        "outdoor_drag-super_mid",
    ] {
        let m = load(mission_name);
        for (label, iters, complete) in [
            ("e_base", 1u8, false),
            ("e_iters5", 5, false),
            ("e_s1", 1, true),
            ("e_iters5_s1", 5, true),
        ] {
            let mut vp = vehicle(true);
            vp.mpc.max_iters = iters;
            let mut c = MpcFullIndiController::with_options_at_rate(&vp, 100.0, true, 8000.0);
            c.complete_references = complete;
            c.log_solves = true;
            dump(mission_name, label, scenario_for(mission_name, &m, vp), c, |c| {
                serde_json::to_value(&c.solve_log).unwrap()
            });
        }
    }
}

#[test]
fn diag_indoor_splits_fast() {
    diag_mission("indoor_splits_fast");
}

#[test]
fn diag_outdoor_splits_large_mid() {
    diag_mission("outdoor_splits-large_mid");
}

#[test]
fn diag_outdoor_splits_mid() {
    diag_mission("outdoor_splits_mid");
}

/// Healthy baseline for solver statistics.
#[test]
fn diag_outdoor_splits_slow_baseline() {
    diag_mission("outdoor_splits_slow");
}
