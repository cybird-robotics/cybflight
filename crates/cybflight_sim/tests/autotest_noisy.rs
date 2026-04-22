//! Sensors-in-the-loop autotest: runs the square mission through
//! `MpcIndiController` with a `NoisyImu` feeding INDI. Ground-truth
//! controllers (cascade, mpc_direct) are not exercised here — they ignore
//! IMU measurements, so noise wouldn't reach them. This test's value is
//! in showing how much tracking error the INDI rate loop absorbs when its
//! sensors are realistic.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test autotest_noisy -- --nocapture

use std::path::PathBuf;

use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_sim::{
    controller::MpcIndiController,
    plant::{QuadPlant, VEHICLE},
    report,
    runner::MissionRunner,
    scenario::{Scenario, Verdict},
    sensors::NoisyImu,
};
use nalgebra::Vector3;

fn out_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Representative consumer-grade MEMS-IMU noise: ~0.005 rad/s gyro ARW
/// sampled at 8 kHz gives ≈0.5 rad/s/√Hz · √(8000 Hz) * k — we pick 0.03
/// rad/s per-axis σ as a round "noticeable but not crippling" value.
/// Accel similar — ≈0.3 m/s² per-axis σ is consistent with a typical
/// ICM-42688P at flight vibration.
fn noisy_mpc_indi_mission() -> (Scenario, MpcIndiController) {
    let vp = VEHICLE.build();
    let cfg = QuadPlanningConfig::from_vehicle_params(&vp);
    let scenario = Scenario::mission(
        "mission_square_noisy",
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
        &cfg,
    )
    .with_imu(Box::new(NoisyImu::isotropic(0xC0FFEE, 0.03, 0.3)));
    let controller = MpcIndiController::from_params(&vp);
    (scenario, controller)
}

#[test]
fn mpc_indi_tracks_through_noisy_imu() {
    let (mut scenario, mut controller) = noisy_mpc_indi_mission();
    let vp = VEHICLE.build();
    let mut plant = QuadPlant::new(vp, 1.0 / 8000.0);
    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    let dir = out_dir("mission_square_noisy_mpc_indi");
    let json = report::write_json(&dir, &scenario, "mpc_indi", &out).expect("write json");
    println!(
        "mission_square_noisy [mpc_indi] rms={:.4}m peak={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {:?} ({})",
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
    // The pass criteria (rms < 0.5 m, peak_tilt < 60°, etc.) are coarse;
    // we expect INDI to ride through σ=0.03 rad/s gyro + σ=0.3 m/s² accel
    // without running away. Any tighter bound belongs in the regression
    // snapshot, not here.
    assert_eq!(out.verdict, Verdict::Pass, "{:?}", out.failure_reasons);
}
