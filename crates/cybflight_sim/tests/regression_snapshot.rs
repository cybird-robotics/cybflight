//! Regression snapshot: asserts the 12-row (scenario × controller) metric
//! table against a committed JSON file. Drift ⇒ test fails.
//!
//! Tolerances are tight (~1% relative + small absolute floor) so routine
//! nalgebra/rustc updates don't churn, but a real behavior change always
//! shows up.
//!
//! Run:
//!   just sim-check       # compare against the snapshot
//!   just sim-snapshot    # regenerate the snapshot (then `git diff` to
//!                        # review; commit both the code and the updated
//!                        # snapshot together)
//!
//! See docs/HACKING.md for the review workflow.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_sim::{
    controller::{CascadeController, Controller, MpcDirectController, MpcIndiController},
    plant::{QuadPlant, VEHICLE},
    runner::{MissionRunner, RunOutput},
    scenario::Scenario,
};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Row {
    rms_pos_err_m: f32,
    peak_pos_err_m: f32,
    terminal_pos_err_m: f32,
    peak_tilt_rad: f32,
    peak_motor_saturation_pct: f32,
}

impl From<&RunOutput> for Row {
    fn from(out: &RunOutput) -> Self {
        let s = &out.summary;
        Row {
            rms_pos_err_m: s.rms_pos_err_m,
            peak_pos_err_m: s.peak_pos_err_m,
            terminal_pos_err_m: s.terminal_pos_err_m,
            peak_tilt_rad: s.peak_tilt_rad,
            peak_motor_saturation_pct: s.peak_motor_saturation_pct,
        }
    }
}

const SCENARIOS: &[&str] = &["hover_level", "hover_tilt30", "p2p_x3", "mission_square"];
const CONTROLLERS: &[&str] = &["cascade", "mpc_direct", "mpc_indi"];

fn build_scenario(name: &str) -> Scenario {
    let vp = VEHICLE.build();
    let cfg = QuadPlanningConfig::from_vehicle_params(&vp);
    match name {
        "hover_level" => Scenario::hover(name, Vector3::new(0.0, 0.0, 1.0), 0.0),
        "hover_tilt30" => Scenario::hover(name, Vector3::new(0.0, 0.0, 1.0), 30.0_f32.to_radians()),
        "p2p_x3" => Scenario::point_to_point(
            name,
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(3.0, 0.0, 1.0),
            &cfg,
        ),
        "mission_square" => Scenario::mission(
            name,
            Vector3::new(0.0, 0.0, 1.0),
            &[
                Vector3::new(3.0, 0.0, 1.0),
                Vector3::new(3.0, 3.0, 1.0),
                Vector3::new(0.0, 3.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            &cfg,
        ),
        other => panic!("unknown scenario: {other}"),
    }
}

fn build_controller(name: &str) -> Box<dyn Controller> {
    let vp = VEHICLE.build();
    match name {
        "cascade" => Box::new(CascadeController::from_params(&vp)),
        "mpc_direct" => Box::new(MpcDirectController::from_params(&vp)),
        "mpc_indi" => Box::new(MpcIndiController::from_params(&vp)),
        other => panic!("unknown controller: {other}"),
    }
}

fn compute_current() -> BTreeMap<String, Row> {
    let mut out = BTreeMap::new();
    for &s in SCENARIOS {
        for &c in CONTROLLERS {
            let mut scenario = build_scenario(s);
            let mut controller = build_controller(c);
            let mut plant = QuadPlant::new(VEHICLE.build(), 1.0 / 8000.0);
            let runner = MissionRunner::new(Default::default());
            let run_out = runner.run(&mut scenario, &mut plant, &mut *controller);
            out.insert(format!("{s}/{c}"), (&run_out).into());
        }
    }
    out
}

// Tolerance: drift fails the test when the actual value moves more than
// max(tol_rel * |expected|, tol_abs) away from the snapshot. Relative
// component catches multiplicative drift; absolute component avoids
// spurious fails on tiny reference values (hover_level rms ≈ 0).
struct Tol {
    rel: f32,
    abs: f32,
}

const TOL_POS: Tol = Tol { rel: 0.01, abs: 1e-4 }; // 1 % or 0.1 mm
const TOL_TILT: Tol = Tol { rel: 0.01, abs: 1e-3 }; // 1 % or ~0.057°
const TOL_SAT: Tol = Tol { rel: 0.01, abs: 0.5 }; //  1 % or 0.5 %-point

fn drift(exp: f32, got: f32, tol: Tol) -> Option<f32> {
    let d = (got - exp).abs();
    let allowed = tol.abs.max(exp.abs() * tol.rel);
    (d > allowed).then_some(d)
}

fn diff(exp: &Row, got: &Row) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(d) = drift(exp.rms_pos_err_m, got.rms_pos_err_m, TOL_POS) {
        parts.push(format!(
            "rms {:.4}→{:.4} (Δ{:.5})",
            exp.rms_pos_err_m, got.rms_pos_err_m, d
        ));
    }
    if let Some(d) = drift(exp.peak_pos_err_m, got.peak_pos_err_m, TOL_POS) {
        parts.push(format!(
            "peak {:.4}→{:.4} (Δ{:.5})",
            exp.peak_pos_err_m, got.peak_pos_err_m, d
        ));
    }
    if let Some(d) = drift(exp.terminal_pos_err_m, got.terminal_pos_err_m, TOL_POS) {
        parts.push(format!(
            "term {:.4}→{:.4} (Δ{:.5})",
            exp.terminal_pos_err_m, got.terminal_pos_err_m, d
        ));
    }
    if let Some(d) = drift(exp.peak_tilt_rad, got.peak_tilt_rad, TOL_TILT) {
        parts.push(format!(
            "tilt {:.4}→{:.4} (Δ{:.5})",
            exp.peak_tilt_rad, got.peak_tilt_rad, d
        ));
    }
    if let Some(d) = drift(
        exp.peak_motor_saturation_pct,
        got.peak_motor_saturation_pct,
        TOL_SAT,
    ) {
        parts.push(format!(
            "sat {:.1}→{:.1} (Δ{:.3})",
            exp.peak_motor_saturation_pct, got.peak_motor_saturation_pct, d
        ));
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/regression_snapshot.json")
}

#[test]
fn regression_snapshot() {
    let current = compute_current();
    let path = snapshot_path();

    // Update mode: write the current numbers and exit. The `just sim-snapshot`
    // recipe sets this env var. Diff afterwards with `git diff`.
    if std::env::var("UPDATE_SNAPSHOTS").is_ok_and(|v| v == "1") {
        let json = serde_json::to_string_pretty(&current).expect("serialize snapshot");
        fs::write(&path, json + "\n").expect("write snapshot");
        println!("wrote snapshot: {}", path.display());
        return;
    }

    // Compare mode: load the committed snapshot and diff.
    let expected: BTreeMap<String, Row> = match fs::read_to_string(&path) {
        Ok(data) => serde_json::from_str(&data).expect("parse regression_snapshot.json"),
        Err(e) => panic!(
            "{}: {e}\n\
             snapshot does not exist yet — run `just sim-snapshot` to create it",
            path.display()
        ),
    };

    let mut failures = Vec::new();
    for (key, actual) in &current {
        match expected.get(key) {
            Some(exp) => {
                if let Some(msg) = diff(exp, actual) {
                    failures.push(format!("  {key}: {msg}"));
                }
            }
            None => failures.push(format!("  {key}: NEW row (not in snapshot)")),
        }
    }
    for key in expected.keys() {
        if !current.contains_key(key) {
            failures.push(format!("  {key}: REMOVED row (in snapshot but not produced)"));
        }
    }

    assert!(
        failures.is_empty(),
        "regression_snapshot drift — review the diff, then either fix the code \
         or re-run `just sim-snapshot` if the change is intentional:\n{}",
        failures.join("\n")
    );
}
