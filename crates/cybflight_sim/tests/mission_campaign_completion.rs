//! TEMPORARY campaign harness — mission-completion configuration.
//!
//! Goal: every controller completes every `missions/*.yaml` without
//! crash/failure, with the tilt constraint disabled. Applies the fix
//! stack identified by the crash diagnosis
//! (docs/mpc_full_crash_diagnosis.md) to all four stacks:
//!
//! - tilt fence OFF (`tilt_barrier_tau = 0`, `tilt_max_deg = 180`);
//! - `q_ref` built with the missions' own `flatness_map: tilt_yaw`
//!   (the firmware default, robust through 90° — the sim's frozen
//!   cross-product map is singular there, campaign cause C2);
//! - controller-side rate limits raised to [16, 16, 8] rad/s so the
//!   fast missions' 10.1–12.8 rad/s demand is not infeasible by
//!   construction (`max_rate_rad_s` is not read by the plant);
//! - `thrust_frac` 0.75 → 1.0 (0.75 capped the reduced stack at 4.7 g
//!   vs the 5.4 g the fast missions demand);
//! - mpc_full: S1 reference completion (Ω_r + u_r feedforward) and the
//!   body-rate barrier (τ=0.5) on the raised bounds.
//!
//! Completion criterion printed per run: no early exit (geofence /
//! ground / NaN) and terminal error < 0.15 m.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test mission_campaign_completion -- \
//!       --nocapture --test-threads=14

use std::path::PathBuf;

use cybflight_core::params::FirmwareConfig;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};
use cybflight_sim::{
    controller::{Controller, MpcFullIndiController, MpcIndiController},
    plant::QuadPlant,
    runner::{MissionRunner, RunnerConfig},
    scenario::{default_sim_params, tweaked_vehicle, PassCriteria, Scenario},
    trajectory::MissionSetpoints,
};
use nalgebra::{UnitQuaternion, Vector3};
use vehicle_yaml::mission::{load_mission, Mission};

fn out_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("campaign_completion");
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
    let head = [Vec3::from(m.start), ZERO3, ZERO3, ZERO3];
    let tail = [Vec3::from(m.waypoints[n - 1]), ZERO3, ZERO3, ZERO3];
    let mut minco = Box::new(MincoSnap::new(&head, &tail, n));
    minco.solve(&intermediate, &durations);
    MissionSetpoints::from_trajectory(minco.get_trajectory())
}

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

/// Completion-campaign vehicle config — see module docs for rationale.
fn completion_vehicle(full_model: bool) -> FirmwareConfig {
    tweaked_vehicle(|vp| {
        vp.mpc.tilt_max_deg = 180.0;
        vp.mpc.tilt_barrier_tau = 0.0;
        vp.airframe.body.max_rate_rad_s = [16.0, 16.0, 8.0];
        vp.mpc.thrust_frac = 1.0;
        if full_model {
            vp.mpc.rate_barrier_tau = 0.5;
            vp.mpc.rate_barrier_delta = 0.5;
        }
    })
}

fn run_case(
    mission_name: &str,
    case: &str,
    mut scenario: Scenario,
    controller: &mut dyn Controller,
) {
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
    let completed = s.early_exit_reason.is_none() && s.terminal_pos_err_m < 0.15;
    println!(
        "{:<26} [{:<12}] rms={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {}",
        mission_name,
        case,
        s.rms_pos_err_m,
        s.terminal_pos_err_m,
        s.peak_tilt_rad.to_degrees(),
        s.peak_motor_saturation_pct,
        if completed {
            "COMPLETED".to_string()
        } else {
            format!("INCOMPLETE ({:?})", s.early_exit_reason)
        }
    );
    let blob = serde_json::json!({
        "mission": mission_name,
        "case": case,
        "completed": completed,
        "rms_pos_err_m": s.rms_pos_err_m,
        "terminal_pos_err_m": s.terminal_pos_err_m,
        "peak_pos_err_m": s.peak_pos_err_m,
        "peak_tilt_deg": s.peak_tilt_rad.to_degrees(),
        "peak_motor_saturation_pct": s.peak_motor_saturation_pct,
        "early_exit": s.early_exit_reason,
    });
    let path = out_dir().join(format!("{mission_name}__{case}.json"));
    std::fs::write(path, serde_json::to_string_pretty(&blob).unwrap()).unwrap();
}

fn run_mission(name: &str) {
    let m = load(name);
    let tilt_map = m.flatness_map == vehicle_yaml::mission::FlatnessMapKind::TiltYaw;

    let vp_a = completion_vehicle(false);
    for (case, rate) in [("mpc_8k", 8000.0f32), ("mpc_1k", 1000.0)] {
        let mut c = MpcIndiController::from_params_at_indi_rate(&vp_a, rate);
        c.use_tilt_map = tilt_map;
        c.complete_references = true;
        run_case(name, case, scenario_for(name, &m, vp_a.clone()), &mut c);
    }

    let vp_b = completion_vehicle(true);
    for (case, rate) in [("mpc_full_8k", 8000.0f32), ("mpc_full_1k", 1000.0)] {
        let mut c = MpcFullIndiController::with_options_at_rate(&vp_b, 100.0, true, rate);
        c.use_tilt_map = tilt_map;
        c.complete_references = true;
        run_case(name, case, scenario_for(name, &m, vp_b.clone()), &mut c);
    }
}

/// Demand probe: sample each mission's planned trajectory and report the
/// peak thrust demand (m·‖ξ̈+g‖ plus the plant's rotor-drag force at the
/// reference velocity) against the 34 N ceiling, and the peak reference
/// body-rate (finite-differenced tilt-yaw q_ref). Dumps 100 Hz demand
/// timelines for offline correlation with crash times.
#[test]
fn demand_probe() {
    use cybflight_core::rotation::quaternion_from_zb_and_yaw;
    let sim = default_sim_params();
    let vp = completion_vehicle(false);
    let mass = vp.airframe.body.mass_kg;
    let g = 9.81f32;
    // ω_sum at high thrust ≈ 4·√(T/4/c_T); c_T from the YAML comment.
    let c_t = 1.3695e-6f32;
    for name in [
        "indoor_splits_slow", "indoor_splits_mid", "indoor_splits_fast",
        "outdoor_drag-super_mid", "outdoor_splits_mid",
        "outdoor_splits-large_mid", "outdoor_splits-large_fast",
        "outdoor_splits-super_slow", "outdoor_splits-super_fast",
    ] {
        let m = load(name);
        let mut sp_src = plan_fixed_time(&m);
        use cybflight_sim::trajectory::SetpointSource;
        let dur = sp_src.duration_s();
        let dt = 0.01f32;
        let n = (dur / dt) as usize;
        let mut rows = Vec::with_capacity(n);
        let mut q_prev: Option<nalgebra::UnitQuaternion<f32>> = None;
        let (mut peak_thrust_n, mut peak_rate, mut peak_v) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let t = i as f32 * dt;
            let s = sp_src.sample(t);
            let acc_cmd = Vector3::new(s.acceleration.x, s.acceleration.y, s.acceleration.z + g);
            let thrust_n = mass * acc_cmd.norm();
            // Rotor-drag H-force at reference speed, assuming rotors at
            // the speed this thrust needs.
            let omega_sum = 4.0 * ((thrust_n / 4.0).max(0.0) / c_t).sqrt();
            let drag_n = (Vector3::new(
                sim.aero_drag[0] * s.velocity.x,
                sim.aero_drag[1] * s.velocity.y,
                sim.aero_drag[2] * s.velocity.z,
            ) * omega_sum)
                .norm();
            let z_b = acc_cmd / acc_cmd.norm().max(1e-6);
            let q = quaternion_from_zb_and_yaw(&z_b, 0.0, true);
            let rate = q_prev
                .map(|qp| (qp.inverse() * q).scaled_axis().norm() / dt)
                .unwrap_or(0.0);
            q_prev = Some(q);
            peak_thrust_n = peak_thrust_n.max(thrust_n + drag_n);
            peak_rate = peak_rate.max(rate);
            peak_v = peak_v.max(s.velocity.norm());
            rows.push(serde_json::json!({
                "t": t,
                "thrust_n": thrust_n + drag_n,
                "rate": rate,
                "v": s.velocity.norm(),
            }));
        }
        println!(
            "{:<26} peak demand: thrust {:.1} N ({:.0}% of 34 N ceiling, {:.2} g) rate {:.1} rad/s v {:.1} m/s",
            name,
            peak_thrust_n,
            peak_thrust_n / 34.0 * 100.0,
            peak_thrust_n / (mass * g),
            peak_rate,
            peak_v,
        );
        let blob = serde_json::json!({"mission": name, "rows": rows});
        std::fs::write(
            out_dir().join(format!("demand__{name}.json")),
            serde_json::to_string(&blob).unwrap(),
        )
        .unwrap();
    }
}

/// Regression matrix for mpc_full on outdoor_splits_mid: which
/// ingredient of the completion config broke it (it completed with
/// S1 + cross-product map + frozen rate limits in the diag campaign)?
#[test]
fn splits_mid_full_matrix() {
    let m = load("outdoor_splits_mid");
    for (label, tilt_map, s1, raised_limits) in [
        ("m_cross_s1_frozen", false, true, false),
        ("m_cross_s1_raised", false, true, true),
        ("m_tilt_s1_frozen", true, true, false),
        ("m_tilt_s1_raised", true, true, true),
        ("m_tilt_nos1_raised", true, false, true),
        ("m_tilt_nos1_frozen", true, false, false),
    ] {
        let mut vp = completion_vehicle(true);
        if !raised_limits {
            vp.airframe.body.max_rate_rad_s = [10.0, 10.0, 6.0];
            vp.mpc.thrust_frac = 0.75;
        }
        let mut c = MpcFullIndiController::with_options_at_rate(&vp, 100.0, true, 8000.0);
        c.use_tilt_map = tilt_map;
        c.complete_references = s1;
        run_case("outdoor_splits_mid", label, scenario_for("outdoor_splits_mid", &m, vp), &mut c);
    }
}

/// Like `run_case` but also dumps the 100 Hz state timeline for offline
/// analysis of a failure.
fn run_case_trace(
    mission_name: &str,
    case: &str,
    mut scenario: Scenario,
    controller: &mut dyn Controller,
) {
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
    let completed = s.early_exit_reason.is_none() && s.terminal_pos_err_m < 0.15;
    println!(
        "{:<26} [{:<20}] rms={:.4}m term={:.4}m tilt={:.1}° sat={:.0}% {}",
        mission_name,
        case,
        s.rms_pos_err_m,
        s.terminal_pos_err_m,
        s.peak_tilt_rad.to_degrees(),
        s.peak_motor_saturation_pct,
        if completed { "COMPLETED".into() } else { format!("INCOMPLETE ({:?})", s.early_exit_reason) }
    );
    let hist: Vec<serde_json::Value> = out
        .history
        .iter()
        .map(|r| {
            serde_json::json!({
                "t": r.t,
                "pos": [r.position.x, r.position.y, r.position.z],
                "vel": [r.velocity.x, r.velocity.y, r.velocity.z],
                "rate": [r.body_rate.x, r.body_rate.y, r.body_rate.z],
                "tilt_rad": r.tilt_rad,
                "sp_pos": [r.setpoint.position.x, r.setpoint.position.y, r.setpoint.position.z],
                "sp_acc": [r.setpoint.acceleration.x, r.setpoint.acceleration.y, r.setpoint.acceleration.z],
                "cmd": r.motor_commands,
            })
        })
        .collect();
    let blob = serde_json::json!({"mission": mission_name, "case": case, "history": hist});
    std::fs::write(
        out_dir().join(format!("trace__{mission_name}__{case}.json")),
        serde_json::to_string(&blob).unwrap(),
    )
    .unwrap();
}

/// mpc_full on the head-whip mission: does a faster outer solve close
/// the held-(T, τ) gap that the reduced stack's 8 kHz rate loop covers?
#[test]
fn fast_mission_solve_rate() {
    let m = load("indoor_splits_fast");
    for (label, yaw_w, barrier_tau) in [
        ("r_full_yaw20", 20.0f32, 0.5f32),
        ("r_full_nobarrier", 200.0, 0.0),
        ("r_full_yaw20_nobarrier", 20.0, 0.0),
    ] {
        let mut vp = completion_vehicle(true);
        vp.mpc.att_weight[2] = yaw_w;
        vp.mpc.rate_barrier_tau = barrier_tau;
        let mut c = MpcFullIndiController::with_options_at_rate(&vp, 100.0, true, 8000.0);
        c.use_tilt_map = true;
        c.complete_references = true;
        run_case_trace("indoor_splits_fast", label, scenario_for("indoor_splits_fast", &m, vp), &mut c);
    }
}

/// The two thrust-infeasible missions (demand 102 % / 107 % of the 34 N
/// ceiling): flown with the flight vehicles' 12 N motors (plant AND
/// controller change consistently, as in the original campaign's C3
/// experiment) — 48 N ceiling brings demand to 73 % / 75 %.
#[test]
fn thrust_infeasible_on_12n() {
    for name in ["outdoor_splits-large_fast", "outdoor_splits-super_fast"] {
        let m = load(name);
        let mk = |full: bool| {
            let mut vp = completion_vehicle(full);
            for mo in &mut vp.airframe.motors {
                mo.max_thrust_n = 12.0;
            }
            vp
        };
        let vp = mk(false);
        for (case, rate, err_sat) in [
            ("n12_mpc_8k", 8000.0f32, 1.0f32),
            ("n12_mpc_1k", 1000.0, 1.0),
            ("n12_mpc_8k_s2tight", 8000.0, 0.4),
        ] {
            let mut c = MpcIndiController::from_params_at_indi_rate(&vp, rate);
            c.use_tilt_map = true;
            c.complete_references = true;
            c.err_sat_m = err_sat;
            c.set_thrust_floor(0.1);
            run_case(name, case, scenario_for(name, &m, vp.clone()), &mut c);
        }
        let vp = mk(true);
        for (case, rate) in [("n12_full_8k", 8000.0f32), ("n12_full_1k", 1000.0)] {
            let mut c = MpcFullIndiController::with_options_at_rate(&vp, 100.0, true, rate);
            c.use_tilt_map = true;
            c.complete_references = true;
            c.err_sat_m = 1.0;
            c.set_thrust_floor(0.1);
            run_case(name, case, scenario_for(name, &m, vp.clone()), &mut c);
        }
    }
}

/// Focused iteration on the remaining incomplete missions.
#[test]
fn focus_experiments() {
    // Does large_mid still complete without S1? (S1-on completed at 0.33.)
    {
        let m = load("outdoor_splits-large_mid");
        let vp = completion_vehicle(true);
        let mut c = MpcFullIndiController::with_options_at_rate(&vp, 100.0, true, 8000.0);
        c.use_tilt_map = true;
        c.complete_references = false;
        run_case("outdoor_splits-large_mid", "f_full_nos1", scenario_for("outdoor_splits-large_mid", &m, vp), &mut c);
    }
    // indoor_splits_fast: S2 governor × thrust floor, both stacks
    // (S1 + tilt map always on, matching the campaign config).
    {
        let m = load("indoor_splits_fast");
        for (label, err_sat, floor) in [
            ("f_full_s2", 1.0f32, 0.0f32),
            ("f_full_floor", 0.0, 0.1),
            ("f_full_s2_floor", 1.0, 0.1),
            ("f_full_s2tight_floor", 0.5, 0.1),
        ] {
            let vp = completion_vehicle(true);
            let mut c = MpcFullIndiController::with_options_at_rate(&vp, 100.0, true, 8000.0);
            c.use_tilt_map = true;
            c.complete_references = true;
            c.err_sat_m = err_sat;
            if floor > 0.0 {
                c.set_thrust_floor(floor);
            }
            run_case_trace("indoor_splits_fast", label, scenario_for("indoor_splits_fast", &m, vp), &mut c);
        }
        for (label, err_sat, floor) in [
            ("f_mpc_s2", 1.0f32, 0.0f32),
            ("f_mpc_floor", 0.0, 0.1),
            ("f_mpc_s2_floor", 1.0, 0.1),
            ("f_mpc_s2tight_floor", 0.5, 0.1),
        ] {
            let vp = completion_vehicle(false);
            let mut c = MpcIndiController::from_params_at_indi_rate(&vp, 8000.0);
            c.use_tilt_map = true;
            c.complete_references = true;
            c.err_sat_m = err_sat;
            if floor > 0.0 {
                c.set_thrust_floor(floor);
            }
            run_case_trace("indoor_splits_fast", label, scenario_for("indoor_splits_fast", &m, vp), &mut c);
        }
    }
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
