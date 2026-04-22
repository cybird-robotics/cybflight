//! Integration autotest: run hover, point-to-point, and mission scenarios
//! through the MPC+INDI stack and assert the pass criteria hold.
//! Diagnostic baselines for MpcDirect and cascade are emitted as non-
//! asserting tests so regression deltas can be read off the same reports.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --test autotest_mission --release -- --nocapture --test-threads=1

use std::path::PathBuf;

use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_sim::{
    controller::{CascadeController, Controller, MpcDirectController, MpcIndiController},
    plant::{QuadPlant, VEHICLE},
    report,
    runner::{MissionRunner, RunOutput},
    scenario::{Scenario, Verdict},
};
use nalgebra::Vector3;

fn out_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn run_with<C: Controller>(mut scenario: Scenario, mut controller: C) -> RunOutput {
    let vp = VEHICLE.build();
    let mut plant = QuadPlant::new(vp, 1.0 / 8000.0);
    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    let dir = out_dir(&format!("{}_{}", scenario.name, controller.name()));
    let json = report::write_json(&dir, &scenario, controller.name(), &out).expect("write json");
    println!(
        "{:<18} [{:<10}] rms={:.4}m peak={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {:?} ({})",
        scenario.name,
        controller.name(),
        out.summary.rms_pos_err_m,
        out.summary.peak_pos_err_m,
        out.summary.terminal_pos_err_m,
        out.summary.peak_tilt_rad.to_degrees(),
        out.summary.peak_motor_saturation_pct,
        out.verdict,
        json.display(),
    );
    for r in &out.failure_reasons {
        println!("  fail: {r}");
    }
    out
}

fn make_hover_level() -> Scenario {
    Scenario::hover("hover_level", Vector3::new(0.0, 0.0, 1.0), 0.0)
}

fn make_hover_tilt30() -> Scenario {
    Scenario::hover(
        "hover_tilt30",
        Vector3::new(0.0, 0.0, 1.0),
        30.0_f32.to_radians(),
    )
}

fn make_p2p() -> Scenario {
    let vp = VEHICLE.build();
    let cfg = QuadPlanningConfig::from_vehicle_params(&vp);
    Scenario::point_to_point(
        "p2p_x3",
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(3.0, 0.0, 1.0),
        &cfg,
    )
}

fn make_mission_square() -> Scenario {
    let vp = VEHICLE.build();
    let cfg = QuadPlanningConfig::from_vehicle_params(&vp);
    Scenario::mission(
        "mission_square",
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
        &cfg,
    )
}

fn mpc_indi() -> MpcIndiController {
    MpcIndiController::from_params(&VEHICLE.build())
}

fn mpc_direct() -> MpcDirectController {
    MpcDirectController::from_params(&VEHICLE.build())
}

fn cascade() -> CascadeController {
    CascadeController::from_params(&VEHICLE.build())
}

// ── Authoritative MPC+INDI tests (firmware-match topology) ──────────────────

#[test]
fn mpc_indi_hover_level_converges() {
    let out = run_with(make_hover_level(), mpc_indi());
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

#[test]
fn mpc_indi_hover_with_initial_tilt_recovers() {
    let out = run_with(make_hover_tilt30(), mpc_indi());
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

#[test]
fn mpc_indi_point_to_point_x3() {
    let out = run_with(make_p2p(), mpc_indi());
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

#[test]
fn mpc_indi_mission_square() {
    let out = run_with(make_mission_square(), mpc_indi());
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

// ── Diagnostic baselines — non-asserting ────────────────────────────────────

#[test]
fn mpc_direct_baseline_all_scenarios() {
    let _ = run_with(make_hover_level(), mpc_direct());
    let _ = run_with(make_hover_tilt30(), mpc_direct());
    let _ = run_with(make_p2p(), mpc_direct());
    let _ = run_with(make_mission_square(), mpc_direct());
}

#[test]
fn cascade_baseline_all_scenarios() {
    let _ = run_with(make_hover_level(), cascade());
    let _ = run_with(make_hover_tilt30(), cascade());
    let _ = run_with(make_p2p(), cascade());
    let _ = run_with(make_mission_square(), cascade());
}
