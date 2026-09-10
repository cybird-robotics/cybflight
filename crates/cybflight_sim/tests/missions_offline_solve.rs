//! Every `missions/*.yaml` must survive the exact path the firmware's
//! `plan_offline` runs at the mission trigger: bake-time load, duration
//! recovery from timestamps, the offline MINCO-snap solve with zero-PVAJ
//! boundaries, the duration guards, and — for `headings` missions — the
//! yaw spline. Asserts solver invariants (waypoints hit, C³ junctions,
//! finite everywhere, rest boundaries) and prints the flatness demand so
//! an infeasible reference is visible even though it is not a solver bug.
use std::path::PathBuf;

use cybflight_core::trajectory_planning::minco_acc::{unwrap_nearest, MincoAcc};
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_core::params::PlannerParams;
use cybflight_core::trajectory_planning::MAX_PIECES;
use cybflight_sim::primitives::{peak_demand, Envelope};
use vehicle_yaml::mission::load_mission;


/// Shortest segment for which the f32 banded LU (no pivoting, rows mixing
/// 1 … T⁷) still hits waypoints to < 1 mm. Measured by
/// `waypoint_miss_vs_segment_duration`: the miss is ~1e-4 m down to
/// ~0.09 s and then climbs to 1 cm at 0.06 s and 3–8 cm at 0.03–0.045 s.
/// Schedules below this get a warning, not a failure — the firmware runs
/// the same f32 code, so the on-device trajectory carries the same miss.
const F32_ACCURATE_MIN_SEGMENT_S: f32 = 0.08;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Everything a firmware-equivalent solve is judged against, taken from
/// the vehicle's own YAML — the flight envelope for the demand report and
/// the planner's duration window for the acceptance guard.
///
/// The window is read rather than mirrored on purpose: the firmware
/// rejects on `plan_dur_min_s` / `plan_dur_max_s` (mission_planner's
/// `duration_bounds`), so a vehicle that pins either key would otherwise
/// reject a mission at the RC trigger while this test — whose whole
/// claim is "solves like the firmware" — stayed green against a literal.
struct Vehicle {
    env: Envelope,
    planner: PlannerParams,
}

fn vehicle(name: &str) -> Vehicle {
    let yaml = std::fs::read_to_string(root().join(format!("vehicles/{name}.yaml"))).unwrap();
    let params = vehicle_yaml::load(name, &yaml).expect("vehicle loads").params;
    let q = QuadPlanningConfig::from_vehicle_params(&params);
    let planner = params.trajectory.planner.clone();
    let env = Envelope {
        mass_kg: q.mass,
        grav: q.grav,
        thrust_max_n: q.max_collective_thrust_n,
        rate_max: Vec3::from(q.max_rate_rad_s),
    };
    Vehicle { env, planner }
}

struct Report {
    name: String,
    n: usize,
    dur: f32,
    wp_err_max: f32,
    junction_err_max: f32,
    vmax: f32,
    amax: f32,
    thrust_frac: f32,
    rate_frac: f32,
    yaw: &'static str,
    yaw_err_max: f32,
    min_dt: f32,
}

/// One firmware-equivalent solve; panics on any solver-invariant violation.
fn solve_like_firmware(name: &str, head_pos: Vec3, start_yaw: f32, veh: &Vehicle) -> Report {
    solve_scaled(name, head_pos, start_yaw, veh, 1.0)
}

/// Same as [`solve_like_firmware`] with every segment duration scaled by
/// `time_scale` (diagnostic: how the solve conditions with segment length).
fn solve_scaled(name: &str, head_pos: Vec3, start_yaw: f32, veh: &Vehicle, time_scale: f32) -> Report {
    let path = root().join("missions").join(format!("{name}.yaml"));
    let m = load_mission(name, &std::fs::read_to_string(&path).unwrap())
        .unwrap_or_else(|e| panic!("{name}: bake would fail: {e}"));

    // ---- shape guard (plan_offline / bake) ----
    let n = m.waypoints.len();
    assert!(n >= 1 && n <= MAX_PIECES, "{name}: {n} pieces outside [1, {MAX_PIECES}]");
    assert_eq!(n, m.timestamps.len(), "{name}: waypoint/timestamp length mismatch");

    // ---- durations from timestamps (plan_offline rejects on any bad one) ----
    let mut durations = Vec::with_capacity(n);
    let mut prev_t = 0.0f32;
    for (i, &ts) in m.timestamps.iter().enumerate() {
        let d = ts - prev_t;
        assert!(
            d.is_finite() && d > 0.0,
            "{name}: firmware would drop the request — timestamps non-monotonic at i={i} (d={d})"
        );
        durations.push(d * time_scale);
        prev_t = ts;
    }
    let init_duration_s = m.timestamps[n - 1] * time_scale;
    let timestamps: Vec<f32> = m.timestamps.iter().map(|t| t * time_scale).collect();
    let min_dt = durations.iter().cloned().fold(f32::INFINITY, f32::min);

    // ---- MINCO snap, zero-PVAJ boundaries, same instantiation as OFFLINE_MINCO ----
    let intermediate: Vec<Vec3> = m.waypoints[..n - 1].iter().map(|w| Vec3::from(*w)).collect();
    let tail_pos = Vec3::from(m.waypoints[n - 1]);
    let head = [head_pos, ZERO3, ZERO3, ZERO3];
    let tail = [tail_pos, ZERO3, ZERO3, ZERO3];
    let mut minco = Box::new(MincoSnap::new(&head, &tail, MAX_PIECES));
    minco.set_piece_count(n);
    minco.set_boundary(&head, &tail);
    minco.solve(&intermediate, &durations);
    let traj = minco.get_trajectory();
    let energy = minco.get_energy();

    // ---- the firmware's `basic_valid` guard ----
    let dur = traj.total_duration();
    assert!(
        dur.is_finite() && energy.is_finite(),
        "{name}: non-finite solve (dur={dur}, energy={energy})"
    );
    let (dur_min_s, dur_max_s) = (veh.planner.duration_min_s, veh.planner.duration_max_s);
    assert!(
        (dur_min_s..=dur_max_s).contains(&dur),
        "{name}: firmware would reject — duration {dur:.2}s outside [{dur_min_s}, {dur_max_s}]"
    );
    assert!((dur - init_duration_s).abs() < 1e-3, "{name}: duration {dur} != schedule {init_duration_s}");

    // ---- solver invariants ----
    for i in 0..n {
        for c in traj.piece(i).coeffs.iter() {
            assert!(c.iter().all(|v| v.is_finite()), "{name}: non-finite coefficient in piece {i}");
        }
    }
    // Every waypoint is hit at its timestamp.
    let mut wp_err_max = 0.0f32;
    for i in 0..n {
        let p = traj.get_pos(timestamps[i]);
        let e = (p - Vec3::from(m.waypoints[i])).norm();
        wp_err_max = wp_err_max.max(e);
    }
    // Junctions are continuous through jerk (snap continuity is what
    // MINCO-s4 enforces; jerk is the tightest thing a tracker feels).
    let mut junction_err_max = 0.0f32;
    for i in 0..n - 1 {
        let a = traj.piece(i);
        let b = traj.piece(i + 1);
        let t = a.duration;
        for (ea, eb) in [
            (a.get_pos(t), b.get_pos(0.0)),
            (a.get_vel(t), b.get_vel(0.0)),
            (a.get_acc(t), b.get_acc(0.0)),
            (a.get_jerk(t), b.get_jerk(0.0)),
        ] {
            let e = (ea - eb).norm() / ea.norm().max(1.0);
            junction_err_max = junction_err_max.max(e);
        }
    }
    // Rest boundaries.
    for (label, t) in [("head", 0.0), ("tail", dur)] {
        assert!(traj.get_vel(t).norm() < 1e-2, "{name}: {label} velocity {:?}", traj.get_vel(t));
        assert!(traj.get_acc(t).norm() < 1e-1, "{name}: {label} acceleration {:?}", traj.get_acc(t));
    }
    assert!((traj.get_pos(0.0) - head_pos).norm() < 1e-3, "{name}: head position");
    assert!((traj.get_pos(dur) - tail_pos).norm() < 1e-3, "{name}: tail position");
    // Finite everywhere on a 1 ms grid, plus peak kinematics.
    let (mut vmax, mut amax) = (0.0f32, 0.0f32);
    let mut t = 0.0f32;
    while t <= dur {
        let (p, v, a, j) = (traj.get_pos(t), traj.get_vel(t), traj.get_acc(t), traj.get_jerk(t));
        for (what, x) in [("pos", p), ("vel", v), ("acc", a), ("jerk", j)] {
            assert!(x.iter().all(|c| c.is_finite()), "{name}: non-finite {what} at t={t}");
        }
        vmax = vmax.max(v.norm());
        amax = amax.max(a.norm());
        t += 1e-3;
    }
    let (thrust_frac, rate_frac) = peak_demand(&traj, &veh.env);

    // ---- yaw path ----
    let (yaw, yaw_err_max) = if m.lookahead {
        ("lookahead", 0.0)
    } else if let Some(h) = &m.headings {
        assert_eq!(h.len(), n, "{name}: headings length");
        let mut unwrapped = Vec::with_capacity(n);
        let mut prev = start_yaw;
        for &psi in h {
            prev = unwrap_nearest(prev, psi);
            unwrapped.push(prev);
        }
        let mut yaw_solver = Box::new(MincoAcc::new(&[0.0, 0.0], &[0.0, 0.0], MAX_PIECES));
        yaw_solver.set_piece_count(n);
        yaw_solver.set_boundary(&[start_yaw, 0.0], &[unwrapped[n - 1], 0.0]);
        yaw_solver.solve(&unwrapped[..n - 1], &durations);
        let yt = yaw_solver.get_trajectory();
        let end = yt.sample(yt.total_duration());
        assert!(end[0].is_finite() && end[1].is_finite(), "{name}: yaw spline non-finite");
        let mut err = 0.0f32;
        for i in 0..n {
            let s = yt.sample(timestamps[i]);
            assert!(s.iter().all(|v| v.is_finite()), "{name}: yaw sample non-finite at wp {i}");
            err = err.max((s[0] - unwrapped[i]).abs());
        }
        assert!(err < 5e-3, "{name}: yaw waypoint miss {err:.4} rad");
        ("headings", err)
    } else {
        ("constant", 0.0)
    };

    Report {
        name: name.to_string(),
        n,
        dur,
        wp_err_max,
        junction_err_max,
        vmax,
        amax,
        thrust_frac,
        rate_frac,
        yaw,
        yaw_err_max,
        min_dt,
    }
}

#[test]
fn every_mission_yaml_solves_like_the_firmware() {
    let indoor = vehicle("sakura_bench_leader_1khz");
    let outdoor = vehicle("sakura_bench_racer_outdoor");

    let mut names: Vec<String> = std::fs::read_dir(root().join("missions"))
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "yaml").then(|| p.file_stem().unwrap().to_string_lossy().to_string())
        })
        .collect();
    names.sort();
    assert!(!names.is_empty());

    println!(
        "\n{:<30} {:>3} {:>7} {:>6} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6}  {:<9} {:>8}",
        "mission", "n", "dur[s]", "min_dt", "wp_err", "junc", "vmax", "amax", "T/Tmax", "w/wmax", "yaw", "yaw_err"
    );
    let mut infeasible = Vec::new();
    let mut inaccurate = Vec::new();
    let mut short_segment_warn = Vec::new();
    for name in &names {
        let veh = if name.starts_with("indoor") { &indoor } else { &outdoor };
        let m = load_mission(name, &std::fs::read_to_string(root().join(format!("missions/{name}.yaml"))).unwrap()).unwrap();
        let start = Vec3::from(m.start);
        // 1) YAML start (OFFLINE_USE_YAML_START = true).
        let r = solve_like_firmware(name, start, 0.0, veh);
        // 2) Live-setpoint head 0.3 m off the recorded start, entry yaw
        //    away from zero (the default firmware behaviour).
        let _ = solve_like_firmware(name, start + Vec3::new(0.2, -0.2, 0.1), 2.5, veh);

        let flag = if r.thrust_frac > 1.0 || r.rate_frac > 1.0 { " INFEASIBLE" } else { "" };
        if !flag.is_empty() {
            infeasible.push(format!("{} (T {:.2}×, ω {:.2}×)", r.name, r.thrust_frac, r.rate_frac));
        }
        if r.wp_err_max > 5e-3 || r.junction_err_max > 1e-2 {
            let entry = format!("{} (min_dt {:.3} s: wp {:.4} m, junction {:.1e})", r.name, r.min_dt, r.wp_err_max, r.junction_err_max);
            if r.min_dt >= F32_ACCURATE_MIN_SEGMENT_S {
                inaccurate.push(entry);
            } else {
                short_segment_warn.push(entry);
            }
        }
        println!(
            "{:<30} {:>3} {:>7.2} {:>6.3} {:>8.1e} {:>8.1e} {:>6.2} {:>6.2} {:>6.2} {:>6.2}  {:<9} {:>8.1e}{flag}",
            r.name, r.n, r.dur, r.min_dt, r.wp_err_max, r.junction_err_max, r.vmax, r.amax,
            r.thrust_frac, r.rate_frac, r.yaw, r.yaw_err_max
        );
    }
    println!(
        "\n{} missions solved; {} exceed the vehicle envelope (reference infeasible, not a solver defect): {}",
        names.len(),
        infeasible.len(),
        if infeasible.is_empty() { "none".to_string() } else { infeasible.join(", ") }
    );
    if !short_segment_warn.is_empty() {
        println!(
            "WARN f32 precision floor (segments < {F32_ACCURATE_MIN_SEGMENT_S} s): {}",
            short_segment_warn.join("; ")
        );
    }
    assert!(
        inaccurate.is_empty(),
        "solver accuracy: waypoint miss > 5 mm or junction > 1e-2 on: {}",
        inaccurate.join("; ")
    );
}

/// How the worst waypoint miss scales with segment length for the two
/// densest schedules — separates f32 conditioning of the 8N×8N banded LU
/// (entries span 1 … T⁷) from a genuine solver defect.
#[test]
fn waypoint_miss_vs_segment_duration() {
    let outdoor = vehicle("sakura_bench_racer_outdoor");
    for name in ["outdoor_splits-super_timeopt", "outdoor_splits-super_fast", "indoor_slalom_timeopt"] {
        let m = load_mission(name, &std::fs::read_to_string(root().join(format!("missions/{name}.yaml"))).unwrap()).unwrap();
        let start = Vec3::from(m.start);
        print!("{name:<30}");
        for scale in [0.5f32, 1.0, 2.0, 4.0] {
            let r = solve_scaled(name, start, 0.0, &outdoor, scale);
            print!("  x{scale}: min_dt={:.3} wp_err={:.1e} junc={:.1e}", r.min_dt, r.wp_err_max, r.junction_err_max);
        }
        println!();
    }
}

/// Solve a mission exactly as `plan_offline` does and hand back the trajectory.
fn firmware_trajectory(name: &str, head_pos: Vec3) -> cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial {
    let m = load_mission(name, &std::fs::read_to_string(root().join(format!("missions/{name}.yaml"))).unwrap()).unwrap();
    let n = m.waypoints.len();
    let mut durations = Vec::with_capacity(n);
    let mut prev = 0.0f32;
    for &ts in &m.timestamps {
        durations.push(ts - prev);
        prev = ts;
    }
    let intermediate: Vec<Vec3> = m.waypoints[..n - 1].iter().map(|w| Vec3::from(*w)).collect();
    let head = [head_pos, ZERO3, ZERO3, ZERO3];
    let tail = [Vec3::from(m.waypoints[n - 1]), ZERO3, ZERO3, ZERO3];
    let mut minco = Box::new(MincoSnap::new(&head, &tail, MAX_PIECES));
    minco.set_piece_count(n);
    minco.set_boundary(&head, &tail);
    minco.solve(&intermediate, &durations);
    minco.get_trajectory()
}

/// The baked `indoor_circle_timeopt` (39-waypoint decimation, re-solved by
/// the firmware's MINCO-snap) against the original planner's continuous
/// export `analysis/datasets/indoor_exp_timeopt/exp_circle_timeopt.csv`.
#[test]
fn indoor_circle_timeopt_matches_original_export() {
    let csv = std::fs::read_to_string(root().join("analysis/datasets/indoor_exp_timeopt/exp_circle_timeopt.csv")).unwrap();
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |n: &str| header.iter().position(|h| *h == n).unwrap_or_else(|| panic!("no column {n}"));
    let (ct, cp, cv, ca, cj) = (col("t"), col("p_x"), col("v_x"), col("a_lin_x"), col("jerk_x"));
    let rows: Vec<Vec<f32>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|x| x.parse::<f32>().unwrap()).collect())
        .collect();
    let v3 = |r: &Vec<f32>, c: usize| Vec3::new(r[c], r[c + 1], r[c + 2]);

    let m = load_mission("indoor_circle_timeopt", &std::fs::read_to_string(root().join("missions/indoor_circle_timeopt.yaml")).unwrap()).unwrap();
    let traj = firmware_trajectory("indoor_circle_timeopt", Vec3::from(m.start));
    let dur = traj.total_duration();
    let csv_end = rows.last().unwrap()[ct];

    struct Stat { max: f32, t_max: f32, sum_sq: f32, n: usize }
    impl Stat {
        fn new() -> Self { Stat { max: 0.0, t_max: 0.0, sum_sq: 0.0, n: 0 } }
        fn push(&mut self, e: f32, t: f32) { if e > self.max { self.max = e; self.t_max = t; } self.sum_sq += e * e; self.n += 1; }
        fn rms(&self) -> f32 { (self.sum_sq / self.n.max(1) as f32).sqrt() }
    }
    let (mut sp, mut sv, mut sa, mut sj) = (Stat::new(), Stat::new(), Stat::new(), Stat::new());
    let (mut ref_vmax, mut ref_amax, mut vmax, mut amax) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for r in &rows {
        let t = r[ct];
        if t > dur.min(csv_end) { break; }
        let (p, v, a, j) = (traj.get_pos(t), traj.get_vel(t), traj.get_acc(t), traj.get_jerk(t));
        let (rp, rv, ra, rj) = (v3(r, cp), v3(r, cv), v3(r, ca), v3(r, cj));
        sp.push((p - rp).norm(), t);
        sv.push((v - rv).norm(), t);
        sa.push((a - ra).norm(), t);
        sj.push((j - rj).norm(), t);
        ref_vmax = ref_vmax.max(rv.norm()); ref_amax = ref_amax.max(ra.norm());
        vmax = vmax.max(v.norm()); amax = amax.max(a.norm());
    }
    // Are the YAML waypoints on the original path at their timestamps?
    let mut wp_on_ref = 0.0f32;
    for (w, &ts) in m.waypoints.iter().zip(&m.timestamps) {
        let r = rows.iter().min_by(|a, b| (a[ct] - ts).abs().partial_cmp(&(b[ct] - ts).abs()).unwrap()).unwrap();
        wp_on_ref = wp_on_ref.max((Vec3::from(*w) - v3(r, cp)).norm());
    }

    println!("\nindoor_circle_timeopt vs exp_circle_timeopt.csv ({} samples @ 10 ms)", sp.n);
    println!("  duration      firmware {dur:.4} s   export {csv_end:.4} s   (Δ {:+.4} s)", dur - csv_end);
    println!("  YAML waypoints vs export path at their timestamps: max {wp_on_ref:.4} m");
    println!("  position      rms {:.4} m   max {:.4} m @ t={:.2}", sp.rms(), sp.max, sp.t_max);
    println!("  velocity      rms {:.3} m/s  max {:.3} m/s @ t={:.2}   (peak: firmware {vmax:.2}, export {ref_vmax:.2})", sv.rms(), sv.max, sv.t_max);
    println!("  acceleration  rms {:.2} m/s² max {:.2} m/s² @ t={:.2}   (peak: firmware {amax:.2}, export {ref_amax:.2})", sa.rms(), sa.max, sa.t_max);
    println!("  jerk          rms {:.1} m/s³ max {:.1} m/s³ @ t={:.2}", sj.rms(), sj.max, sj.t_max);

    // Flatness outputs: collective thrust per unit mass and body rates,
    // first through the min-norm map (body-z ≡ 0; kept as the contrast —
    // the outer loop now uses the closed form, compared further down).
    use cybflight_core::trajectory_planning::flatness::flatness_to_thrust_omega;
    let (cw, cth) = (col("w_x"), col("thrust"));
    let mut st = Stat::new();
    let mut sw = [Stat::new(), Stat::new(), Stat::new()];
    let (mut th_pk, mut th_pk_ref) = (0.0f32, 0.0f32);
    let (mut w_pk, mut w_pk_ref) = (Vec3::zeros(), Vec3::zeros());
    let mut faults = 0usize;
    for r in &rows {
        let t = r[ct];
        if t > dur.min(csv_end) { break; }
        let (a, j) = (traj.get_acc(t), traj.get_jerk(t));
        let Ok((alpha, _, w)) = flatness_to_thrust_omega(a, j, 0.0, 0.0, 9.81) else { faults += 1; continue };
        let (rth, rw) = (r[cth], v3(r, cw));
        st.push((alpha - rth).abs(), t);
        for k in 0..3 {
            sw[k].push((w[k] - rw[k]).abs(), t);
            w_pk[k] = w_pk[k].max(w[k].abs());
            w_pk_ref[k] = w_pk_ref[k].max(rw[k].abs());
        }
        th_pk = th_pk.max(alpha);
        th_pk_ref = th_pk_ref.max(rth);
    }
    println!("  thrust/mass   rms {:.3} m/s² max {:.3} m/s² @ t={:.2}   (peak: firmware {th_pk:.2}, export {th_pk_ref:.2})", st.rms(), st.max, st.t_max);
    for (k, ax) in ["x", "y", "z"].iter().enumerate() {
        println!("  rate w_{ax}      rms {:.3} rad/s max {:.3} rad/s @ t={:.2}   (peak: firmware {:.2}, export {:.2})",
            sw[k].rms(), sw[k].max, sw[k].t_max, w_pk[k], w_pk_ref[k]);
    }
    println!("  flatness faults: {faults}");
    assert_eq!(faults, 0);

    // Same comparison through the other yaw convention (Euler yaw ψ = 0).
    use cybflight_core::trajectory_planning::flatness::flatness_to_thrust_omega_true_yaw;
    let mut sw2 = [Stat::new(), Stat::new(), Stat::new()];
    let mut w2_pk = Vec3::zeros();
    let mut faults2 = 0usize;
    for r in &rows {
        let t = r[ct];
        if t > dur.min(csv_end) { break; }
        let (a, j) = (traj.get_acc(t), traj.get_jerk(t));
        let Ok((_, _, w)) = flatness_to_thrust_omega_true_yaw(a, j, 0.0, 0.0, 9.81) else { faults2 += 1; continue };
        let rw = v3(r, cw);
        for k in 0..3 { sw2[k].push((w[k] - rw[k]).abs(), t); w2_pk[k] = w2_pk[k].max(w[k].abs()); }
    }
    for (k, ax) in ["x", "y", "z"].iter().enumerate() {
        println!("  [true_yaw ψ=0] w_{ax}  rms {:.3} max {:.3} rad/s   (peak firmware {:.2}, export {:.2})", sw2[k].rms(), sw2[k].max, w2_pk[k], w_pk_ref[k]);
    }
    println!("  [true_yaw ψ=0] faults: {faults2}");

    // And through the closed-form tilt-yaw map (the C++ reference planner's
    // convention, which keeps the (zb1·ż0 − zb0·ż1)/(zb.z+1) body-z term).
    use cybflight_core::trajectory_planning::flatness::flatness_to_state_tilt_yaw;
    let mut sw3 = [Stat::new(), Stat::new(), Stat::new()];
    let mut w3_pk = Vec3::zeros();
    let mut faults3 = 0usize;
    for r in &rows {
        let t = r[ct];
        if t > dur.min(csv_end) { break; }
        let (a, j, sn) = (traj.get_acc(t), traj.get_jerk(t), traj.get_snap(t));
        let Ok(st) = flatness_to_state_tilt_yaw(a, j, sn, [0.0, 0.0, 0.0], 9.81) else { faults3 += 1; continue };
        let rw = v3(r, cw);
        for k in 0..3 { sw3[k].push((st.omega[k] - rw[k]).abs(), t); w3_pk[k] = w3_pk[k].max(st.omega[k].abs()); }
    }
    for (k, ax) in ["x", "y", "z"].iter().enumerate() {
        println!("  [tilt_yaw closed form ψ=0] w_{ax}  rms {:.3} max {:.3} rad/s   (peak firmware {:.2}, export {:.2})", sw3[k].rms(), sw3[k].max, w3_pk[k], w_pk_ref[k]);
    }
    println!("  [tilt_yaw closed form ψ=0] faults: {faults3}");
    assert!((dur - csv_end).abs() < 0.05, "duration mismatch");
    assert!(sp.max.is_finite() && sv.max.is_finite() && sa.max.is_finite());
}
