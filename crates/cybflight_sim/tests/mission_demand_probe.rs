//! Prints the peak flatness thrust / rate demand of every indoor mission
//! (calibration for `primitives`' budgets). Run with `--nocapture`.
use std::path::PathBuf;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_sim::primitives::{peak_demand, Envelope};
use vehicle_yaml::mission::load_mission;

#[test]
fn probe() {
    let env = Envelope { mass_kg: 0.6, grav: 9.81, thrust_max_n: 30.0, rate_max: Vec3::new(10.0, 10.0, 6.0) };
    for name in ["indoor_figure8_slow","indoor_figure8_mid","indoor_figure8_timeopt","indoor_splits_slow","indoor_splits_mid","indoor_splits_fast","indoor_splits_timeopt","indoor_slalom_timeopt","indoor_circle_timeopt"] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../missions").join(format!("{name}.yaml"));
        let m = load_mission(name, &std::fs::read_to_string(&path).unwrap()).unwrap();
        let n = m.waypoints.len();
        let mut durations = Vec::new(); let mut prev = 0.0f32;
        for &t in &m.timestamps { durations.push(t - prev); prev = t; }
        let inter: Vec<Vec3> = m.waypoints[..n-1].iter().map(|w| Vec3::from(*w)).collect();
        let mut minco = Box::new(MincoSnap::new(&[Vec3::from(m.start), ZERO3, ZERO3, ZERO3], &[Vec3::from(m.waypoints[n-1]), ZERO3, ZERO3, ZERO3], n));
        minco.solve(&inter, &durations);
        let traj = minco.get_trajectory();
        let (pt, pr) = peak_demand(&traj, &env);
        let dur = traj.total_duration();
        let mut vmax = 0.0f32; let mut t = 0.0;
        while t < dur { vmax = vmax.max(traj.get_vel(t).norm()); t += 0.01; }
        println!("{name:<24} dur={dur:.2} vmax={vmax:.1} peak_thrust_frac={pt:.2} peak_rate_frac={pr:.2}");
    }
}

/// How far the 63-waypoint MINCO fit of `outdoor_splits-super_timeopt`
/// strays from the 42 waypoints the decimation dropped (the full
/// 105-point export exceeds `MAX_PIECES`). Prints max / mean distance.
#[test]
fn super_timeopt_decimation_fidelity() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let m = load_mission("outdoor_splits-super_timeopt", &std::fs::read_to_string(root.join("missions/outdoor_splits-super_timeopt.yaml")).unwrap()).unwrap();
    let n = m.waypoints.len();
    let mut durations = Vec::new(); let mut prev = 0.0f32;
    for &t in &m.timestamps { durations.push(t - prev); prev = t; }
    let inter: Vec<Vec3> = m.waypoints[..n-1].iter().map(|w| Vec3::from(*w)).collect();
    let mut minco = Box::new(MincoSnap::new(&[Vec3::from(m.start), ZERO3, ZERO3, ZERO3], &[Vec3::from(m.waypoints[n-1]), ZERO3, ZERO3, ZERO3], n));
    minco.solve(&inter, &durations);
    let traj = minco.get_trajectory();
    // The full export.
    let src: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(root.join("tmp/tmp_missions/outdoor_missions/outdoor_splits-super_timeopt.yaml")).unwrap()).unwrap();
    let wps: Vec<Vec3> = src["waypoints"].as_sequence().unwrap().iter().map(|p| Vec3::new(p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32, p[2].as_f64().unwrap() as f32)).collect();
    let ts: Vec<f32> = src["timestamps"].as_sequence().unwrap().iter().map(|t| t.as_f64().unwrap() as f32).collect();
    let (mut worst, mut sum, mut cnt) = (0.0f32, 0.0f32, 0);
    for (p, &t) in wps.iter().zip(ts.iter()).skip(1) {
        let d = cybflight_sim::rl_env::closest_point_dist(&traj, nalgebra::Vector3::new(p.x, p.y, p.z), t, 0.6);
        worst = worst.max(d); sum += d; cnt += 1;
    }
    let env = Envelope { mass_kg: 0.6, grav: 9.81, thrust_max_n: 30.0, rate_max: Vec3::new(10.0, 10.0, 6.0) };
    let (pt, pr) = peak_demand(&traj, &env);
    println!("super_timeopt 63-pt fit vs all 105 export waypoints: max {worst:.3} m, mean {:.3} m; peak thrust frac {pt:.2}, rate frac {pr:.2}, dur {:.2}", sum / cnt as f32, traj.total_duration());
}
