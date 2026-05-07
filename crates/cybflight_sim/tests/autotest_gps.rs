//! ESKF-in-the-loop autotest: runs the square mission through
//! `MpcIndiController` with a `NoisyGps` feeding an in-sim ESKF.
//!
//! Unlike the clean / noisy-IMU autotests, this one puts the **real
//! firmware estimator** in the control path — the controllers see ESKF
//! output, not plant truth. It's the safeguard for the firmware GPS
//! integration (`est_pos_gps` feature): if the ESKF + GPS + controller
//! loop falls apart under realistic position noise here, it will fall
//! apart in flight.
//!
//! Only `MpcIndi` is exercised — it matches the firmware's outer_mpc
//! topology. Cascade / MpcDirect are ground-truth controllers and aren't
//! representative of the firmware GPS path.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test autotest_gps -- --nocapture

use std::path::PathBuf;

use cybflight_sim::{
    controller::MpcIndiController,
    plant::QuadPlant,
    report,
    runner::MissionRunner,
    scenario::{Scenario, Verdict},
    sensors::NoisyGps,
};
use nalgebra::Vector3;

fn out_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Noise profile: σ_pos=0.5 m, σ_vel=0.2 m/s at 5 Hz — roughly u-blox
/// M10 open-sky with SBAS aiding. Tighter than a nominal L1-only fix,
/// looser than RTK; a realistic outdoor-flight baseline.
fn gps_mpc_indi_mission() -> (Scenario, MpcIndiController) {
    let mut scenario = Scenario::mission(
        "mission_square_gps",
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
    )
    .with_gps(Box::new(NoisyGps::isotropic(0xDEADBEEF, 5.0, 0.5, 0.2)));
    // The previous override of indi_controller.rate_gains / sync_filter_hz
    // and mpc.thrust_weight is no longer needed: the sim baseline in
    // `plant.rs::VehicleParamsBuilder::build` now sets these fields
    // explicitly (schema-stability contract), so the values flow
    // through without depending on `Default` impls in cybflight-core.
    // Looser terminal / RMS gates than the clean-sensor scenarios: the
    // ESKF filters σ=0.5 m GPS noise at 5 Hz, which at 3 s terminal hold
    // averages ~15 samples → ~0.13 m residual floor, plus estimator lag.
    // This test is a don't-blow-up safeguard, not a pixel-perfect assert;
    // tight numerics live in the regression snapshot.
    scenario.pass_criteria.terminal_pos_err_m = 0.35;
    scenario.pass_criteria.rms_pos_err_m = 0.80;
    let controller = MpcIndiController::from_params(&scenario.vehicle_params);
    (scenario, controller)
}

#[test]
fn mpc_indi_tracks_through_noisy_gps() {
    let (mut scenario, mut controller) = gps_mpc_indi_mission();
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), 1.0 / 8000.0);
    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    let dir = out_dir("mission_square_gps_mpc_indi");
    let json = report::write_json(&dir, &scenario, "mpc_indi", &out).expect("write json");
    println!(
        "mission_square_gps [mpc_indi] rms={:.4}m peak={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {:?} ({})",
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
    // The pass criteria are coarse (rms < 0.5 m, peak_tilt < 60°, etc.).
    // Expect the ESKF to filter 5 Hz σ=0.5 m noise into something the
    // MPC can track within tens of cm on the straight legs — corners
    // will ring out harder, which is why we care about peak, not just
    // rms. Tighter bounds belong in the regression snapshot.
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}
