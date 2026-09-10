//! TEMPORARY campaign harness — tilt-constraint-free mission grid.
//!
//! Re-runs the docs/mission_campaign_report.md campaign with the MPC
//! tilt fence disabled (`mpc_tilt_max_deg = 180` + `tilt_barrier_tau = 0`,
//! the exact off switch) across four stacks:
//!   1. mpc      + INDI @ 8 kHz
//!   2. mpc      + INDI @ 1 kHz
//!   3. mpc_full + INDI @ 8 kHz
//!   4. mpc_full + INDI @ 1 kHz
//! mpc_full keeps the adopted body-rate barrier (τ=0.5/δ=0.5); the
//! reduced model keeps its input-space rate bounds. Everything else
//! matches the original campaign: frozen sim_baseline vehicle, fixed-time
//! MincoSnap over the mission YAML schedule (the firmware `plan_offline`
//! recipe), 100 Hz MPC replanning, cross-product reference map, yaw = 0.
//!
//! Non-asserting: every run prints one summary line and drops a JSON blob
//! under $CARGO_TARGET_TMPDIR/campaign_notilt/ for aggregation.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test mission_campaign_notilt -- \
//!       --nocapture --test-threads=14

use std::path::PathBuf;

use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_sim::{
    controller::{Controller, MpcFullIndiController, MpcIndiController},
    plant::QuadPlant,
    runner::{MissionRunner, RunnerConfig},
    scenario::{default_sim_params, tweaked_vehicle, PassCriteria, Scenario},
    trajectory::MissionSetpoints,
};
use cybflight_core::params::FirmwareConfig;
use nalgebra::{UnitQuaternion, Vector3};
use vehicle_yaml::mission::{load_mission, Mission};

fn out_dir() -> PathBuf {
    let sub = if tilt_fence_on() {
        "campaign_tilt60"
    } else {
        "campaign_notilt"
    };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(sub);
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

/// Fixed-time MincoSnap solve — the firmware `plan_offline` recipe:
/// durations from consecutive timestamp diffs, `wp[0..n−1]` intermediate,
/// `wp[n−1]` tail, zero-PVAJ boundaries, head = mission start.
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
    let head_pos = Vec3::from(m.start);
    let tail_pos = Vec3::from(m.waypoints[n - 1]);
    let head = [head_pos, ZERO3, ZERO3, ZERO3];
    let tail = [tail_pos, ZERO3, ZERO3, ZERO3];

    // ~100 KB of inline buffers — keep it off the test stack.
    let mut minco = Box::new(MincoSnap::new(&head, &tail, n));
    minco.solve(&intermediate, &durations);
    MissionSetpoints::from_trajectory(minco.get_trajectory())
}

/// Geofence from the mission's waypoint bounding box + 10 m margin
/// (z floor pinned at −1 m: below ground = crashed). The default ±15 m
/// fence is too small for the large/super outdoor missions.
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

/// `CAMPAIGN_TILT=on` runs the control grid with the adopted tilt fence
/// (60°, τ=0.5, δ=0.05) instead — same harness, so the fence effect is
/// isolated from any harness-reconstruction delta vs the original report.
fn tilt_fence_on() -> bool {
    std::env::var("CAMPAIGN_TILT").is_ok_and(|v| v == "on")
}

/// Tilt constraint OFF (default) or adopted fence (control); rate barrier
/// only for the full model (adopted).
fn campaign_vehicle(full_model: bool) -> FirmwareConfig {
    tweaked_vehicle(|vp| {
        if tilt_fence_on() {
            vp.mpc.tilt_max_deg = 60.0;
            vp.mpc.tilt_barrier_tau = 0.5;
            vp.mpc.tilt_barrier_delta = 0.05;
        } else {
            vp.mpc.tilt_max_deg = 180.0;
            vp.mpc.tilt_barrier_tau = 0.0;
        }
        if full_model {
            vp.mpc.rate_barrier_tau = 0.5;
            vp.mpc.rate_barrier_delta = 0.5;
        }
    })
}

fn run_case(mission_name: &str, case: &str, mut scenario: Scenario, controller: &mut dyn Controller) {
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
    println!(
        "{:<26} [{:<12}] rms={:.4}m peak={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {:?} {}",
        mission_name,
        case,
        s.rms_pos_err_m,
        s.peak_pos_err_m,
        s.terminal_pos_err_m,
        s.peak_tilt_rad.to_degrees(),
        s.peak_motor_saturation_pct,
        out.verdict,
        if out.failure_reasons.is_empty() {
            String::new()
        } else {
            format!("fail: {}", out.failure_reasons.join("; "))
        }
    );
    let blob = serde_json::json!({
        "mission": mission_name,
        "case": case,
        "rms_pos_err_m": s.rms_pos_err_m,
        "peak_pos_err_m": s.peak_pos_err_m,
        "terminal_pos_err_m": s.terminal_pos_err_m,
        "peak_tilt_deg": s.peak_tilt_rad.to_degrees(),
        "peak_motor_saturation_pct": s.peak_motor_saturation_pct,
        "geofence_violation": s.geofence_violation,
        "early_exit": s.early_exit_reason,
        "verdict": format!("{:?}", out.verdict),
        "failure_reasons": out.failure_reasons,
    });
    let path = out_dir().join(format!("{mission_name}__{case}.json"));
    std::fs::write(path, serde_json::to_string_pretty(&blob).unwrap()).unwrap();
}

fn run_mission(name: &str) {
    let m = load(name);

    let vp_a = campaign_vehicle(false);
    let mut c = MpcIndiController::from_params_at_indi_rate(&vp_a, 8000.0);
    run_case(name, "mpc_8k", scenario_for(name, &m, vp_a.clone()), &mut c);

    let mut c = MpcIndiController::from_params_at_indi_rate(&vp_a, 1000.0);
    run_case(name, "mpc_1k", scenario_for(name, &m, vp_a.clone()), &mut c);

    let vp_b = campaign_vehicle(true);
    let mut c = MpcFullIndiController::with_options_at_rate(&vp_b, 100.0, true, 8000.0);
    run_case(name, "mpc_full_8k", scenario_for(name, &m, vp_b.clone()), &mut c);

    let mut c = MpcFullIndiController::with_options_at_rate(&vp_b, 100.0, true, 1000.0);
    run_case(name, "mpc_full_1k", scenario_for(name, &m, vp_b), &mut c);
}

macro_rules! mission_tests {
    ($($fn_name:ident => $file:expr;)*) => {
        $(
            #[test]
            fn $fn_name() {
                run_mission($file);
            }
        )*
    };
}

mission_tests! {
    indoor_splits_slow => "indoor_splits_slow";
    indoor_splits_mid => "indoor_splits_mid";
    indoor_splits_fast => "indoor_splits_fast";
    outdoor_drag_slow => "outdoor_drag_slow";
    outdoor_drag_mid => "outdoor_drag_mid";
    outdoor_drag_large_mid => "outdoor_drag-large_mid";
    outdoor_drag_super_mid => "outdoor_drag-super_mid";
    outdoor_splits_slow => "outdoor_splits_slow";
    outdoor_splits_mid => "outdoor_splits_mid";
    outdoor_splits_large_slow => "outdoor_splits-large_slow";
    outdoor_splits_large_mid => "outdoor_splits-large_mid";
    outdoor_splits_large_fast => "outdoor_splits-large_fast";
    outdoor_splits_super_slow => "outdoor_splits-super_slow";
    outdoor_splits_super_fast => "outdoor_splits-super_fast";
}
