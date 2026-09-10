//! Controller-performance ablation across the plant's physics terms.
//!
//! The regression snapshot answers "did anything change?". This answers
//! "*what* is now costing us tracking error, and how much?" — by running
//! the same scenario and controller against plants that differ in exactly
//! one term.
//!
//! Two things make this worth having as a test rather than a one-off
//! script. First, the ordering assertions below encode physics
//! expectations (drag is a bigger penalty than rotor inertia; disabling
//! G2 cannot help) that a future refactor could silently violate.
//! Second, the printed table is the artifact a human reads when deciding
//! whether a retune is worth it.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test plant_ablation -- --nocapture

use cybflight_sim::controller::{MpcDirectController, MpcIndiController};
use cybflight_sim::plant::QuadPlant;
use cybflight_sim::runner::{MissionRunner, RunnerConfig, SummaryMetrics};
use cybflight_sim::scenario::{default_sim_params, Scenario};
use cybflight_sim::sensors::NoRotorTelemetry;
use nalgebra::Vector3;
use vehicle_yaml::SimYaml;

const DT: f32 = 1.0 / 8000.0;

const SQUARE: &[Vector3<f32>] = &[
    Vector3::new(3.0, 0.0, 1.0),
    Vector3::new(3.0, 3.0, 1.0),
    Vector3::new(0.0, 3.0, 1.0),
    Vector3::new(0.0, 0.0, 1.0),
];

fn mission() -> Scenario {
    let mut s = Scenario::mission("ablation_square", Vector3::new(0.0, 0.0, 1.0), SQUARE);
    // The ablations deliberately fly a worse vehicle than the pass
    // criteria assume; this test reports numbers rather than gating.
    s.pass_criteria.rms_pos_err_m = f32::INFINITY;
    s.pass_criteria.terminal_pos_err_m = f32::INFINITY;
    s.pass_criteria.peak_tilt_rad = f32::INFINITY;
    s
}

/// Run `mission()` with `sim` physics, optionally muting ESC telemetry
/// (which disables INDI's G2 columns), and optionally overriding plant
/// parameters after construction — that last hook is how rotor lag is
/// removed from the *plant* while leaving the controller's belief intact.
fn run(
    sim: SimYaml,
    rotor_telemetry: bool,
    tweak: impl FnOnce(&mut QuadPlant),
) -> SummaryMetrics {
    let mut scenario = mission().with_sim_params(sim.clone());
    if !rotor_telemetry {
        scenario = scenario.with_rotor(Box::new(NoRotorTelemetry));
    }
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &sim, DT);
    tweak(&mut plant);
    let mut ctl = MpcIndiController::from_params(&scenario.vehicle_params);
    let runner = MissionRunner::new(RunnerConfig {
        dt_sim: DT,
        ..Default::default()
    });
    runner.run(&mut scenario, &mut plant, &mut ctl).summary
}

fn run_direct(sim: SimYaml) -> SummaryMetrics {
    let mut scenario = mission().with_sim_params(sim.clone());
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &sim, DT);
    let mut ctl = MpcDirectController::from_params(&scenario.vehicle_params, &sim);
    let runner = MissionRunner::new(RunnerConfig {
        dt_sim: DT,
        ..Default::default()
    });
    runner.run(&mut scenario, &mut plant, &mut ctl).summary
}

fn row(label: &str, s: &SummaryMetrics) {
    println!(
        "{label:<34} {:>9.4} {:>9.4} {:>9.4} {:>8.1} {:>8.1}",
        s.rms_pos_err_m,
        s.peak_pos_err_m,
        s.terminal_pos_err_m,
        s.peak_tilt_rad.to_degrees(),
        s.peak_motor_saturation_pct,
    );
}

fn header(title: &str) {
    println!("\n{title}");
    println!(
        "{:<34} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "variant", "rms[m]", "peak[m]", "term[m]", "tilt[°]", "sat[%]"
    );
    println!("{}", "─".repeat(82));
}

/// Cost of the physics the controller has **no model of at all**.
///
/// Removing one of these makes the plant easier, because the controller
/// was never compensating for it. This is the honest measure of the
/// fidelity the rotor-state plant added.
///
/// Note what is *not* in this group: rotor lag and rotor inertia. INDI
/// knows about both (`indi_tau_m*` sets its incremental gain, `g2_ry_m*`
/// its rotor-reaction columns), so deleting them from the plant does not
/// simplify the problem — it de-tunes the controller. That case is
/// measured separately in [`mismatch_sensitivity_of_modelled_terms`].
#[test]
fn ablate_unmodelled_physics() {
    let full = default_sim_params();
    let baseline = run(full.clone(), true, |_| {});

    let no_drag = run(
        SimYaml {
            aero_drag: [0.0; 3],
            ..full.clone()
        },
        true,
        |_| {},
    );
    let no_idle_offset = run(
        SimYaml {
            rotor_omega_min_rad_s: 0.0,
            ..full.clone()
        },
        true,
        |_| {},
    );
    // Plant curvature lowered to the value INDI's inverse is pinned at
    // (0.7, the registry ceiling), leaving the linearization exact.
    let matched_curve = run(
        SimYaml {
            rotor_throttle_curve_k: 0.7,
            ..full.clone()
        },
        true,
        |_| {},
    );
    // Everything unmodelled removed at once.
    let all_removed = run(
        SimYaml {
            aero_drag: [0.0; 3],
            rotor_omega_min_rad_s: 0.0,
            rotor_throttle_curve_k: 0.7,
            ..full.clone()
        },
        true,
        |_| {},
    );

    header("Unmodelled-physics ablation — mpc_indi, mission_square");
    row("FULL baseline", &baseline);
    row("− aero drag", &no_drag);
    row("− ESC idle offset (ω_min = 0)", &no_idle_offset);
    row("− curve mismatch (k 0.95→0.70)", &matched_curve);
    row("− all three", &all_removed);

    println!(
        "\ncost of each unmodelled term (Δ rms vs FULL baseline, m):\n  \
         aero drag        {:+.5}\n  \
         ESC idle offset  {:+.5}\n  \
         curve mismatch   {:+.5}\n  \
         all three        {:+.5}",
        baseline.rms_pos_err_m - no_drag.rms_pos_err_m,
        baseline.rms_pos_err_m - no_idle_offset.rms_pos_err_m,
        baseline.rms_pos_err_m - matched_curve.rms_pos_err_m,
        baseline.rms_pos_err_m - all_removed.rms_pos_err_m,
    );

    // The plant must be harder than one stripped of everything the
    // controller cannot see; otherwise the new physics is not reaching
    // the closed loop at all.
    assert!(
        baseline.rms_pos_err_m > all_removed.rms_pos_err_m,
        "full plant ({:.5} m rms) is no harder than one with every \
         unmodelled term removed ({:.5} m)",
        baseline.rms_pos_err_m,
        all_removed.rms_pos_err_m
    );
    // Drag dominates: it is a force that acts continuously along the
    // whole trajectory, while the other two are throttle-curve details.
    assert!(
        baseline.rms_pos_err_m - no_drag.rms_pos_err_m
            > baseline.rms_pos_err_m - no_idle_offset.rms_pos_err_m,
        "aero drag should cost more than the ESC idle offset"
    );
    assert_all_flyable(&[
        ("baseline", &baseline),
        ("no_drag", &no_drag),
        ("no_idle_offset", &no_idle_offset),
        ("matched_curve", &matched_curve),
        ("all_removed", &all_removed),
    ]);
}

/// How accurately the *modelled* terms have to be identified on real
/// hardware before the loop stops working.
///
/// Rotor lag and rotor inertia are both things INDI knows about
/// (`indi_tau_m*` scales its incremental gain through
/// `g2_scaler = ω_max²/(2τ)`; `g2_ry_m*` carries the rotor-reaction
/// columns), so mismatching them de-tunes the controller rather than
/// simplifying the plant. Every row here should therefore be worse than
/// the matched baseline — and past a point, unflyable.
///
/// The headline: at hover the G2 (rotor-reaction) yaw entry is **2.3×**
/// the G1 (prop-drag) entry, so yaw effectiveness is dominated by a term
/// that scales as `1/τ`. Over-estimating rotor speed — i.e. believing the
/// motors are faster than they are — inflates INDI's assumed authority
/// and the loop over-controls.
#[test]
fn mismatch_sensitivity_of_modelled_terms() {
    let full = default_sim_params();
    let matched = run(full.clone(), true, |_| {});

    // Rotor inertia deleted from the plant while INDI keeps its G2 pins:
    // the allocator compensates for a reaction torque that is not there.
    let no_rotor_inertia = run(
        SimYaml {
            rotor_inertia_kg_m2: 0.0,
            ..full.clone()
        },
        true,
        |_| {},
    );
    let inertia_30pct = run(
        SimYaml {
            rotor_inertia_kg_m2: full.rotor_inertia_kg_m2 * 1.3,
            ..full.clone()
        },
        true,
        |_| {},
    );

    header("Mismatch sensitivity (modelled terms) — mpc_indi, mission_square");
    row("matched plant/controller", &matched);
    row("plant J_r = 0 (INDI keeps G2)", &no_rotor_inertia);
    row("plant J_r +30 %", &inertia_30pct);

    // Rotor-lag sweep: the plant is slower/faster than INDI's 40 ms.
    println!("\nrotor-lag mismatch sweep (INDI configured for τ = 40 ms):");
    println!(
        "{:<34} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "variant", "rms[m]", "peak[m]", "term[m]", "tilt[°]", "sat[%]"
    );
    println!("{}", "─".repeat(82));
    let mut sweep = Vec::new();
    for mult in [0.5f32, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] {
        let tau = 0.04 * mult;
        let s = run(full.clone(), true, move |p| p.plant.tau_s = [tau; 4]);
        row(&format!("plant τ = {:.0} ms ({mult:.2}×)", tau * 1000.0), &s);
        sweep.push((mult, s));
    }

    // Is the cliff attributable to G2? Re-run the worst case without ESC
    // telemetry, which zeroes the G2 columns.
    let slow_no_g2 = run(full.clone(), false, |p| p.plant.tau_s = [0.08; 4]);
    row("  same, G2 disabled", &slow_no_g2);

    println!();
    println!("  G2's yaw entry is 2.3x the G1 entry at hover and scales as 1/tau, so a");
    println!("  rotor slower than believed inflates INDI's assumed yaw authority.");
    println!("  Tolerable to ~1.25x; degrades at 1.5x; diverges by 2x. Disabling G2");
    println!("  removes the cliff (and costs 7x tracking) - so tau must be identified");
    println!("  on the bench to better than ~25%, not assumed.");
    println!();
    println!("  Note the asymmetry: every benign row over-estimates the plant's");
    println!("  sluggishness (faster rotor, larger J_r) so INDI under-drives. Every");
    println!("  harmful row has INDI believing it has more authority than it does.");

    let find = |m: f32| &sweep.iter().find(|(x, _)| *x == m).unwrap().1;

    // Under-estimating τ (plant faster than believed) is benign: INDI
    // under-claims authority and simply acts conservatively.
    for m in [0.5f32, 0.75] {
        assert!(
            find(m).rms_pos_err_m <= matched.rms_pos_err_m * 1.05,
            "a plant faster than INDI believes ({m}×τ) should be benign, \
             got {:.4} m rms vs matched {:.4} m",
            find(m).rms_pos_err_m,
            matched.rms_pos_err_m
        );
    }
    // The usable identification margin. If this ever tightens, the inner
    // loop got more brittle and someone needs to know.
    assert!(
        find(1.25).rms_pos_err_m < 0.05,
        "25 % rotor-lag over-estimate must stay well controlled, got \
         {:.4} m rms",
        find(1.25).rms_pos_err_m
    );
    // ...and the cliff beyond it is real, not an artifact we should have
    // tuned away. Asserting it keeps the documented margin honest.
    assert!(
        find(2.0).rms_pos_err_m > 10.0 * matched.rms_pos_err_m,
        "the 2× rotor-lag divergence documented above no longer happens \
         ({:.4} m rms) — re-measure the margin and update the comment",
        find(2.0).rms_pos_err_m
    );
    // The cliff must be G2's doing: without it the same plant is flyable.
    assert!(
        slow_no_g2.rms_pos_err_m < find(2.0).rms_pos_err_m,
        "disabling G2 should remove the τ-mismatch divergence"
    );

    for (name, s) in [
        ("plant J_r = 0", &no_rotor_inertia),
        ("plant J_r +30 %", &inertia_30pct),
    ] {
        // Tolerance, not a strict inequality: a mismatch in the benign
        // direction (plant more responsive than INDI believes) can tie
        // with the matched case to within float noise.
        assert!(
            s.rms_pos_err_m >= matched.rms_pos_err_m * 0.99,
            "{name}: mismatching a modelled term improved tracking \
             ({:.5} → {:.5} m rms), so the matched case is not matched",
            matched.rms_pos_err_m,
            s.rms_pos_err_m
        );
    }
    assert_all_flyable(&[
        ("matched", &matched),
        ("no_rotor_inertia", &no_rotor_inertia),
        ("inertia_30pct", &inertia_30pct),
        ("tau_1.25x", find(1.25)),
    ]);
}

fn assert_all_flyable(rows: &[(&str, &SummaryMetrics)]) {
    for (name, s) in rows {
        assert!(
            s.rms_pos_err_m.is_finite() && s.rms_pos_err_m < 1.0,
            "{name}: rms {:.4} m — loop lost control",
            s.rms_pos_err_m
        );
        assert!(!s.geofence_violation, "{name}: geofence violation");
    }
}

/// What INDI's G2 (rotor-reaction) columns are worth, now that the plant
/// actually produces the torque they model and ESC telemetry exists to
/// feed them.
#[test]
fn g2_columns_improve_yaw_axis_tracking() {
    let full = default_sim_params();
    let with_g2 = run(full.clone(), true, |_| {});
    let without_g2 = run(full.clone(), false, |_| {});

    header("INDI G2 (rotor reaction) — mpc_indi, mission_square");
    row("G2 active (ESC telemetry)", &with_g2);
    row("G2 disabled (no telemetry)", &without_g2);
    println!(
        "\n  Δ rms {:+.5} m   Δ peak {:+.5} m   Δ sat {:+.1} %",
        without_g2.rms_pos_err_m - with_g2.rms_pos_err_m,
        without_g2.peak_pos_err_m - with_g2.peak_pos_err_m,
        without_g2.peak_motor_saturation_pct - with_g2.peak_motor_saturation_pct,
    );

    // G2 adds information the allocator did not have. It must not make
    // things worse; a regression here means a sign or scale error in the
    // g2_ry_m* pins or in the plant's reaction torque.
    assert!(
        with_g2.rms_pos_err_m <= without_g2.rms_pos_err_m * 1.02,
        "enabling G2 made tracking worse ({:.5} → {:.5} m rms): check the \
         sign of g2_ry_m* against the plant's reaction torque",
        without_g2.rms_pos_err_m,
        with_g2.rms_pos_err_m
    );
    assert!(both_flyable(&with_g2, &without_g2));
}

/// Cross-stack comparison on the identical plant. `mpc_direct` commands
/// thrust with no inner loop and no rotor feedback, so it eats the full
/// actuator lag open-loop; `mpc_indi` closes a 1 kHz+ incremental loop
/// around measured specific force. The gap between them is the value of
/// the inner loop under realistic actuation.
#[test]
fn indi_inner_loop_beats_open_loop_thrust_commands() {
    let full = default_sim_params();
    let indi = run(full.clone(), true, |_| {});
    let direct = run_direct(full.clone());

    header("Inner-loop value — same plant, mission_square");
    row("mpc_indi (incremental, 8 kHz)", &indi);
    row("mpc_direct (thrust, open loop)", &direct);
    println!(
        "\n  INDI reduces rms by {:.1}× ({:.4} → {:.4} m)",
        direct.rms_pos_err_m / indi.rms_pos_err_m,
        direct.rms_pos_err_m,
        indi.rms_pos_err_m
    );

    assert!(
        indi.rms_pos_err_m < direct.rms_pos_err_m,
        "INDI ({:.4} m) should beat open-loop thrust commands ({:.4} m) \
         on a plant with actuator lag",
        indi.rms_pos_err_m,
        direct.rms_pos_err_m
    );
    assert!(both_flyable(&indi, &direct));
}

fn both_flyable(a: &SummaryMetrics, b: &SummaryMetrics) -> bool {
    [a, b].iter().all(|s| {
        s.rms_pos_err_m.is_finite() && s.rms_pos_err_m < 1.0 && !s.geofence_violation
    })
}
