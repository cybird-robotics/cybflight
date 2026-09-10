//! Stability autotest for the `outer_loop: mpc_full` prototype —
//! full-model NMPC → (T_d, α_d) → INDI α inner loop (Sun et al., T-RO
//! 2022, Fig. 3; plan: `docs/mpc_full_indi_plan.md` Stage 5).
//!
//! Answers the design question empirically before hardware: **is a
//! 50–100 Hz NMPC enough to stabilize when the rate gains are gone and
//! the only inner loop is the α/torque loop at IMU rate?** The asserting
//! grid runs the flight-representative configuration (INDI active,
//! barrier τ = 0.5) at both solve rates, clean and with a noisy IMU. The
//! `mpc_full_noindi` ablation (static inversion, the paper's "NMPC w/o
//! INDI" baseline / firmware `build: indi: no` analogue) is emitted
//! non-asserting, like the other diagnostic baselines.
//!
//! Deliberately NOT part of the frozen regression snapshot — new
//! controller rows join the snapshot only once the topology is promoted
//! past prototype.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --test mpc_full_indi_stability --profile release-host -- \
//!       --nocapture --test-threads=1

use std::path::PathBuf;

use cybflight_sim::{
    controller::{Controller, MpcFullIndiController},
    plant::QuadPlant,
    report,
    runner::{MissionRunner, RunOutput},
    scenario::{tweaked_vehicle, Scenario, Verdict},
    sensors::NoisyImu,
};
use cybflight_core::params::FirmwareConfig;
use nalgebra::Vector3;

fn out_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Sim-baseline vehicle with the stress-suite-validated body-rate
/// barrier enabled — the flight-representative `mpc_full` configuration.
fn mpc_full_vehicle() -> FirmwareConfig {
    tweaked_vehicle(|vp| {
        vp.mpc.rate_barrier_tau = 0.5;
        vp.mpc.rate_barrier_delta = 0.5;
    })
}

fn run_with<C: Controller>(mut scenario: Scenario, mut controller: C) -> RunOutput {
    let mut plant =
        QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    let dir = out_dir(&format!("{}_{}", scenario.name, controller.name()));
    let json = report::write_json(&dir, &scenario, controller.name(), &out).expect("write json");
    println!(
        "{:<22} [{:<15}] rms={:.4}m peak={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {:?} ({})",
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

fn hover_level(vp: FirmwareConfig) -> Scenario {
    Scenario::hover_with_params("hover_level", vp, Vector3::new(0.0, 0.0, 1.0), 0.0)
}

fn hover_tilt30(vp: FirmwareConfig) -> Scenario {
    Scenario::hover_with_params(
        "hover_tilt30",
        vp,
        Vector3::new(0.0, 0.0, 1.0),
        30.0_f32.to_radians(),
    )
}

fn p2p(vp: FirmwareConfig) -> Scenario {
    let mut s = Scenario::point_to_point(
        "p2p_x3",
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(3.0, 0.0, 1.0),
    );
    s.vehicle_params = vp;
    s
}

// ── Asserting grid: flight-representative configuration ───────────────────

#[test]
fn mpc_full_indi_stabilizes_at_100hz() {
    for scenario in [
        hover_level(mpc_full_vehicle()),
        hover_tilt30(mpc_full_vehicle()),
        p2p(mpc_full_vehicle()),
    ] {
        let ctrl = MpcFullIndiController::with_options(&scenario.vehicle_params, 100.0, true);
        let out = run_with(scenario, ctrl);
        assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
    }
}

#[test]
fn mpc_full_indi_stabilizes_at_50hz() {
    for scenario in [
        hover_level(mpc_full_vehicle()),
        hover_tilt30(mpc_full_vehicle()),
        p2p(mpc_full_vehicle()),
    ] {
        let ctrl = MpcFullIndiController::with_options(&scenario.vehicle_params, 50.0, true);
        let out = run_with(scenario, ctrl);
        assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
    }
}

/// The configuration that matters for hardware: noisy IMU into the α
/// inner loop. INDI's incremental correction must absorb the noise the
/// same way it does for the rate-setpoint stack.
#[test]
fn mpc_full_indi_noisy_imu() {
    let scenario = hover_tilt30(mpc_full_vehicle())
        .with_imu(Box::new(NoisyImu::isotropic(0xC0FFEE, 0.03, 0.3)));
    let ctrl = MpcFullIndiController::with_options(&scenario.vehicle_params, 100.0, true);
    let out = run_with(scenario, ctrl);
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}

// ── Non-asserting ablation: static inversion (paper's "NMPC w/o INDI") ────

#[test]
fn baseline_mpc_full_noindi() {
    for scenario in [
        hover_level(mpc_full_vehicle()),
        hover_tilt30(mpc_full_vehicle()),
        p2p(mpc_full_vehicle()),
    ] {
        let ctrl = MpcFullIndiController::with_options(&scenario.vehicle_params, 100.0, false);
        // Diagnostic only — the ablation is expected to be worse (78 %
        // in the paper) and is not gated.
        let _ = run_with(scenario, ctrl);
    }
}
