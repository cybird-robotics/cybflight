//! A/B: the ω half of the MPC feedforward from the min-norm map
//! (`flatness_to_thrust_omega`, body-z ≡ 0 — what the firmware's
//! `outer_loop.rs` uses) vs the closed-form tilt-yaw map
//! (`flatness_to_state_tilt_yaw`, the angular velocity of the tilt `q_ref`
//! itself). Everything else identical and firmware-faithful: tilt `q_ref`,
//! feedforward on. The sim baseline has no tilt fence (like the flying
//! vehicles), so the high-tilt regime where the two disagree is exercised;
//! `OMEGA_AB_FENCE=on` adds a 60° fence for comparison. Non-asserting;
//! prints one line per run.
//!
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test omega_ref_ab -- --nocapture --test-threads=6
use std::path::PathBuf;

use cybflight_core::trajectory_planning::flatness::flatness_to_state_tilt_yaw;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_sim::{
    controller::{Controller, MpcIndiController},
    plant::QuadPlant,
    runner::{MissionRunner, RunnerConfig},
    scenario::{default_sim_params, tweaked_vehicle, PassCriteria, Scenario},
    trajectory::MissionSetpoints,
};
use nalgebra::{UnitQuaternion, Vector3};
use vehicle_yaml::mission::{load_mission, Mission};

fn load(name: &str) -> Mission {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../missions").join(format!("{name}.yaml"));
    load_mission(name, &std::fs::read_to_string(&path).unwrap()).expect("mission loads")
}

fn plan_fixed_time(m: &Mission) -> Box<MincoSnap> {
    let n = m.waypoints.len();
    let mut durations = Vec::with_capacity(n);
    let mut prev = 0.0f32;
    for &t in &m.timestamps { durations.push(t - prev); prev = t; }
    let intermediate: Vec<Vec3> = m.waypoints[..n - 1].iter().map(|w| Vec3::from(*w)).collect();
    let head = [Vec3::from(m.start), ZERO3, ZERO3, ZERO3];
    let tail = [Vec3::from(m.waypoints[n - 1]), ZERO3, ZERO3, ZERO3];
    let mut minco = Box::new(MincoSnap::new(&head, &tail, n));
    minco.solve(&intermediate, &durations);
    minco
}

fn geofence(m: &Mission) -> (Vector3<f32>, Vector3<f32>) {
    let mut lo = Vector3::from(m.start);
    let mut hi = lo;
    for w in &m.waypoints { for a in 0..3 { lo[a] = lo[a].min(w[a]); hi[a] = hi[a].max(w[a]); } }
    (Vector3::new(lo.x - 10.0, lo.y - 10.0, -1.0), hi.add_scalar(10.0))
}

fn run_arm(name: &str, m: &Mission, closed_form: bool) {
    // Default = the baseline (no fence, as flown). `OMEGA_AB_FENCE=on`
    // adds a 60° fence (τ=0.5, δ=0.05) as a comparison point.
    let fence_on = std::env::var("OMEGA_AB_FENCE").is_ok_and(|v| v == "on");
    let vp = tweaked_vehicle(|vp| {
        if fence_on {
            vp.mpc.tilt_max_deg = 60.0;
            vp.mpc.tilt_barrier_tau = 0.5;
            vp.mpc.tilt_barrier_delta = 0.05;
        } else {
            vp.mpc.tilt_max_deg = 180.0;
            vp.mpc.tilt_barrier_tau = 0.0;
        }
    });
    let (gmin, gmax) = geofence(m);
    let mut scenario = Scenario {
        name: name.to_string(),
        vehicle_params: vp.clone(),
        sim_params: default_sim_params(),
        initial_position: Vector3::from(m.start),
        initial_velocity: Vector3::zeros(),
        initial_attitude: UnitQuaternion::identity(),
        setpoints: Box::new(MissionSetpoints::from_trajectory(plan_fixed_time(m).get_trajectory())),
        imu_model: Box::new(cybflight_sim::sensors::PerfectImu),
        gps_model: None,
        rotor_model: Box::new(cybflight_sim::sensors::PerfectRotorTelemetry),
        pass_criteria: PassCriteria { geofence_min: gmin, geofence_max: gmax, ..PassCriteria::default() },
        terminal_hold_s: 3.0,
    };
    let mut c = MpcIndiController::from_params_at_indi_rate(&vp, 1000.0);
    c.use_tilt_map = true;
    c.flatness_feedforward = true;
    c.closed_form_omega_ref = closed_form;
    let mut plant = QuadPlant::new(vp, &scenario.sim_params, 1.0 / 8000.0);
    let runner = MissionRunner::new(RunnerConfig { dt_sim: 1.0 / 8000.0, max_sim_time_s: 120.0, history_rate_hz: 100.0 });
    let out = runner.run(&mut scenario, &mut plant, &mut c as &mut dyn Controller);
    let s = &out.summary;
    println!(
        "{:<26} [{:<11}][fence {}] rms={:.4}m peak={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {:?} {}",
        name,
        if closed_form { "closed-form" } else { "min-norm" },
        if fence_on { "on " } else { "off" },
        s.rms_pos_err_m, s.peak_pos_err_m, s.terminal_pos_err_m,
        s.peak_tilt_rad.to_degrees(), s.peak_motor_saturation_pct, out.verdict,
        if out.failure_reasons.is_empty() { String::new() } else { format!("fail: {}", out.failure_reasons.join("; ")) }
    );
}

fn ab(name: &str) {
    let m = load(name);
    run_arm(name, &m, false);
    run_arm(name, &m, true);
}

/// The closed-form ω must not depend on snap (only ω̇ does), otherwise the
/// sampler's snap-less nodes could not feed it.
#[test]
fn closed_form_omega_is_snap_independent() {
    let m = load("indoor_circle_timeopt");
    let traj = plan_fixed_time(&m).get_trajectory();
    let dur = traj.total_duration();
    let mut worst = 0.0f32;
    let mut t = 0.0f32;
    while t <= dur {
        let (a, j, sn) = (traj.get_acc(t), traj.get_jerk(t), traj.get_snap(t));
        if let (Ok(with), Ok(without)) = (
            flatness_to_state_tilt_yaw(a, j, sn, [0.0; 3], 9.81),
            flatness_to_state_tilt_yaw(a, j, Vector3::zeros(), [0.0; 3], 9.81),
        ) {
            worst = worst.max((with.omega - without.omega).norm());
        }
        t += 0.01;
    }
    println!("closed-form ω: max |ω(snap) − ω(0)| = {worst:.2e} rad/s");
    assert!(worst < 1e-5);
}

#[test] fn circle_mid() { ab("indoor_circle_mid"); }
#[test] fn circle_timeopt() { ab("indoor_circle_timeopt"); }
#[test] fn figure8_mid() { ab("indoor_figure8_mid"); }
#[test] fn splits_mid() { ab("indoor_splits_mid"); }
#[test] fn slalom_timeopt() { ab("indoor_slalom_timeopt"); }
#[test] fn splits_large_mid() { ab("outdoor_splits-large_mid"); }
