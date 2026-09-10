//! Integration autotest: run hover, point-to-point, and mission scenarios
//! through the MPC+INDI stack and assert the pass criteria hold.
//! Diagnostic baselines for MpcDirect and cascade are emitted as non-
//! asserting tests so regression deltas can be read off the same reports.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --test autotest_mission --release -- --nocapture --test-threads=1

use std::path::PathBuf;

use cybflight_sim::{
    controller::{CascadeController, Controller, MpcDirectController, MpcIndiController},
    plant::QuadPlant,
    report,
    runner::{MissionRunner, RunOutput},
    scenario::{tweaked_vehicle, Scenario, Verdict},
};
use nalgebra::Vector3;

fn out_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn run_with<C: Controller>(mut scenario: Scenario, mut controller: C) -> RunOutput {
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
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
    Scenario::point_to_point(
        "p2p_x3",
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(3.0, 0.0, 1.0),
    )
}

fn make_mission_square() -> Scenario {
    Scenario::mission(
        "mission_square",
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
    )
}

fn mpc_indi(scenario: &Scenario) -> MpcIndiController {
    MpcIndiController::from_params(&scenario.vehicle_params)
}

fn mpc_direct(scenario: &Scenario) -> MpcDirectController {
    MpcDirectController::from_params(&scenario.vehicle_params, &scenario.sim_params)
}

fn cascade(scenario: &Scenario) -> CascadeController {
    CascadeController::from_params(&scenario.vehicle_params, &scenario.sim_params)
}

// ── Authoritative MPC+INDI tests (firmware-match topology) ──────────────────

#[test]
fn mpc_indi_hover_level_converges() {
    let s = make_hover_level();
    let c = mpc_indi(&s);
    let out = run_with(s, c);
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

#[test]
fn mpc_indi_hover_with_initial_tilt_recovers() {
    let s = make_hover_tilt30();
    let c = mpc_indi(&s);
    let out = run_with(s, c);
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

#[test]
fn mpc_indi_point_to_point_x3() {
    let s = make_p2p();
    let c = mpc_indi(&s);
    let out = run_with(s, c);
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

#[test]
fn mpc_indi_mission_square() {
    let s = make_mission_square();
    let c = mpc_indi(&s);
    let out = run_with(s, c);
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

// ── Diagnostic baselines — non-asserting ────────────────────────────────────

fn run_baseline<F, C>(scenario: Scenario, make_ctrl: F)
where
    F: FnOnce(&Scenario) -> C,
    C: Controller,
{
    let c = make_ctrl(&scenario);
    let _ = run_with(scenario, c);
}

#[test]
fn mpc_direct_baseline_all_scenarios() {
    run_baseline(make_hover_level(), mpc_direct);
    run_baseline(make_hover_tilt30(), mpc_direct);
    run_baseline(make_p2p(), mpc_direct);
    run_baseline(make_mission_square(), mpc_direct);
}

#[test]
fn cascade_baseline_all_scenarios() {
    run_baseline(make_hover_level(), cascade);
    run_baseline(make_hover_tilt30(), cascade);
    run_baseline(make_p2p(), cascade);
    run_baseline(make_mission_square(), cascade);
}

// ── Tweaked-vehicle smoke test ──────────────────────────────────────────────
//
// Exercises the `tweaked_vehicle` + `_with_params` affordance: build a
// mission scenario against a 1.5× heavier airframe and confirm MpcIndi
// still tracks. The plant, MPC inner model, and the planner config all
// read from the same `VehicleParams` by construction, so this is the
// canonical "change one param, everything updates" check.

#[test]
fn mpc_indi_heavier_vehicle_still_tracks() {
    let vp = tweaked_vehicle(|p| {
        p.airframe.body.mass_kg *= 1.5;
    });
    let scenario = Scenario::mission_with_params(
        "mission_square_heavy",
        vp,
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
    );
    let c = mpc_indi(&scenario);
    let out = run_with(scenario, c);
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}
