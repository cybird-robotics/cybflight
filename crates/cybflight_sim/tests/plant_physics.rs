//! Plant physics validation — open-loop assertions on `QuadPlant` alone.
//!
//! These do not run a controller. They exist to pin the *unit conversions*
//! that produced `vehicles/sim_baseline.yaml`, so a future edit that
//! rescales a coefficient without rescaling its partners fails here rather
//! than showing up as a mysterious 3 % drift in the regression snapshot.
//!
//! The reference numbers come from a system identification of a 5-inch
//! quadrotor (`optimal_quad_control_RL/randomization.py`, `params_5inch`),
//! whose coefficients are all *specific* quantities — fitted against
//! accelerometer and gyro data, so each carries a hidden 1/m or 1/I:
//!
//! ```text
//!   k_w  = 2.49e-6   m·s⁻²/(rad/s)²   thrust      → c_T   = m·k_w
//!   k_x  = 4.85e-5   rad⁻¹            drag x      → c_dx  = m·k_x
//!   k_y  = 7.28e-5   rad⁻¹            drag y      → c_dy  = m·k_y
//!   k_p  = 6.55e-5   rad⁻¹            roll accel  → l_y·m·k_w/I_xx
//!   k_q  ≈ 5.52e-5   rad⁻¹            pitch accel → l_x·m·k_w/I_yy
//!   k_r  = 1.07e-2   s⁻¹              yaw accel   → I_zz·k_r/(c_T·ω_hov)
//!   k_rd = 1.97e-3   —                yaw spin-up → J_r = I_zz·k_rd
//! ```
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test plant_physics

use cybflight_sim::plant::{PlantParams, QuadPlant};
use cybflight_sim::scenario::{default_sim_params, default_vehicle};
use nalgebra::{SVector, UnitQuaternion, Vector3};

const DT: f32 = 1.0 / 8000.0;

// Identified 5-inch reference values.
const K_W: f32 = 2.49e-6;
const K_X: f32 = 4.85e-5;
const K_Y: f32 = 7.28e-5;
const K_P_ID: f32 = 6.55e-5;
const K_Q_ID: f32 = 5.52e-5;
const K_R_ID: f32 = 1.07e-2;
const K_RD_ID: f32 = 1.97e-3;

fn plant_params() -> PlantParams {
    PlantParams::from_config(&default_vehicle(), &default_sim_params())
}

fn fresh_plant() -> QuadPlant {
    let mut p = QuadPlant::new(default_vehicle(), &default_sim_params(), DT);
    p.reset(
        Vector3::zeros(),
        Vector3::zeros(),
        UnitQuaternion::identity(),
    );
    p
}

fn rel_err(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs()
}

/// The identified moment coefficients are *derived* quantities — they
/// absorb the arm length, the thrust coefficient and 1/I. Reconstructing
/// them from the YAML's own geometry must reproduce the identified values,
/// otherwise the sim's airframe is not the airframe those coefficients
/// describe and importing any of them was invalid.
#[test]
fn identified_coefficients_reconstruct_from_yaml_geometry() {
    let vp = default_vehicle();
    let p = plant_params();
    let m = vp.airframe.body.mass_kg;
    let ixx = vp.airframe.body.inertia_kg_m2[0];
    let iyy = vp.airframe.body.inertia_kg_m2[4];
    let izz = vp.airframe.body.inertia_kg_m2[8];
    // Motor 0 sits at (−lx, −ly); magnitudes are what the moment arms use.
    let lx = p.motor_pos[0][0].abs();
    let ly = p.motor_pos[0][1].abs();

    // c_T must equal m·k_w — this is what ties `max_thrust_n` and
    // `indi_omega_m*` together.
    let c_t = p.thrust_coeff[0];
    assert!(
        rel_err(c_t, m * K_W) < 0.02,
        "c_T = {c_t:.4e} but m·k_w = {:.4e}: max_thrust_n and indi_omega_m* \
         no longer imply the identified thrust coefficient",
        m * K_W
    );

    let k_p = ly * m * K_W / ixx;
    assert!(
        rel_err(k_p, K_P_ID) < 0.05,
        "reconstructed roll coefficient {k_p:.4e} vs identified {K_P_ID:.4e}"
    );

    let k_q = lx * m * K_W / iyy;
    assert!(
        rel_err(k_q, K_Q_ID) < 0.06,
        "reconstructed pitch coefficient {k_q:.4e} vs identified {K_Q_ID:.4e}"
    );

    // Yaw: c_Q · c_T · ω_hover = I_zz · k_r at the hover operating point.
    let w_hover = p.hover_omega()[0];
    let k_r = p.torque_coeff_m[0] * c_t * w_hover / izz;
    assert!(
        rel_err(k_r, K_R_ID) < 0.05,
        "reconstructed yaw coefficient {k_r:.4e} vs identified {K_R_ID:.4e}"
    );

    // Rotor inertia: J_r / I_zz is exactly the identified k_rd, and is
    // also what `g2_ry_m*` must carry for INDI's G2 column to be right.
    let k_rd = p.rotor_inertia / izz;
    assert!(
        rel_err(k_rd, K_RD_ID) < 0.02,
        "J_r/I_zz = {k_rd:.4e} vs identified k_rd {K_RD_ID:.4e}"
    );
    let g2_yaw = default_vehicle().airframe.motors[0].g2[2];
    assert!(
        rel_err(g2_yaw.abs(), k_rd) < 0.02,
        "g2_ry_m0 = {g2_yaw:.4e} disagrees with J_r/I_zz = {k_rd:.4e}: \
         INDI's G2 column and the plant's reaction torque have drifted apart"
    );
    // Signs must alternate with spin direction, matching the G1 yaw row.
    for i in 0..4 {
        let g2 = default_vehicle().airframe.motors[i].g2[2];
        assert_eq!(
            g2.signum(),
            p.spin_sign[i],
            "motor {i}: g2_ry sign must follow spin direction"
        );
    }
}

/// Drag coefficients must equal `m · k` — the specific-force → force
/// conversion. Checked through the plant's own force balance rather than
/// the stored constant, so a sign or axis error also fails.
#[test]
fn drag_matches_identified_specific_force() {
    let mut plant = fresh_plant();
    // Level attitude, 10 m/s along body-x. Rotors held at hover speed by
    // the reset, so Σω is known.
    let v = 10.0;
    plant.reset(
        Vector3::zeros(),
        Vector3::new(v, 0.0, 0.0),
        UnitQuaternion::identity(),
    );
    let omega_sum: f32 = plant.rotor_omega().iter().sum();
    let spf = plant.specific_force_body();

    let expect_ax = -K_X * v * omega_sum;
    assert!(
        rel_err(spf.x, expect_ax) < 0.02,
        "body-x specific force {:.4} vs identified −k_x·v·Σω = {expect_ax:.4}",
        spf.x
    );

    plant.reset(
        Vector3::zeros(),
        Vector3::new(0.0, v, 0.0),
        UnitQuaternion::identity(),
    );
    let spf = plant.specific_force_body();
    let expect_ay = -K_Y * v * omega_sum;
    assert!(
        rel_err(spf.y, expect_ay) < 0.02,
        "body-y specific force {:.4} vs identified −k_y·v·Σω = {expect_ay:.4}",
        spf.y
    );
}

/// The hover command must actually hover. This is the end-to-end
/// consistency check on {c_T, ω_max, ω_min, k, mass}: if any one of them
/// is rescaled without the others, the vehicle climbs or sinks.
#[test]
fn hover_command_holds_altitude() {
    let mut plant = fresh_plant();
    let d = plant.plant.hover_command();
    // Sanity: a TWR-6.3 airframe on a k=0.95 curve should hover near a
    // third throttle. A value near 0 or 1 means the curve is broken.
    assert!(
        (0.20..0.50).contains(&d[0]),
        "hover command {:.3} is not a plausible throttle",
        d[0]
    );

    for _ in 0..(2.0 / DT) as usize {
        plant.step(&d);
    }
    let pos = plant.position();
    let vel = plant.velocity();
    assert!(
        pos.z.abs() < 0.02,
        "hover drifted {:.4} m in 2 s — the actuator map and the thrust \
         coefficient disagree about what hover is",
        pos.z
    );
    assert!(vel.norm() < 0.02, "hover velocity {:.4} m/s", vel.norm());
    assert!(
        plant.body_rate().norm() < 1e-3,
        "hover induced body rate {:.5} rad/s",
        plant.body_rate().norm()
    );
}

/// First-order rotor lag: a step from hover to full throttle must cross
/// 63.2 % of the span in exactly one time constant.
#[test]
fn rotor_step_response_matches_time_constant() {
    let p = plant_params();
    let tau = p.tau_s[0];
    let mut plant = fresh_plant();

    let w0 = plant.rotor_omega()[0];
    let d_full = SVector::<f32, 4>::from_element(1.0);
    let w_target = p.omega_command(0, 1.0);

    for _ in 0..(tau / DT).round() as usize {
        plant.step(&d_full);
    }
    let w = plant.rotor_omega()[0];
    let frac = (w - w0) / (w_target - w0);
    assert!(
        (frac - 0.632).abs() < 0.01,
        "rotor reached {:.1}% of the step in one τ, expected 63.2%",
        frac * 100.0
    );
}

/// `command_for_omega` must invert `omega_command` over the usable range,
/// including the ω_min offset. The baseline controllers rely on this to
/// convert their thrust output into an ESC command.
#[test]
fn actuator_map_round_trips() {
    let p = plant_params();
    for step in 0..=20 {
        let d = step as f32 / 20.0;
        let w = p.omega_command(0, d);
        let back = p.command_for_omega(0, w);
        assert!(
            (back - d).abs() < 1e-4,
            "d={d:.3} → ω={w:.2} → d={back:.5}: actuator map is not invertible"
        );
    }
}

/// Rotor spin-up must produce a yaw reaction torque with the sign of the
/// spinning-up rotors, and none when the four rotors accelerate
/// symmetrically in opposing pairs.
#[test]
fn rotor_reaction_torque_yaws_the_airframe() {
    let p = plant_params();
    let mut plant = fresh_plant();
    let hover = p.hover_command();

    // Spin up the two CW rotors, spin down the two CCW ones, keeping
    // collective roughly constant. Net yaw torque should be positive
    // (CW reaction is +z in FLU).
    let mut d = hover;
    for i in 0..4 {
        d[i] = if p.spin_sign[i] > 0.0 {
            (hover[i] + 0.10).min(1.0)
        } else {
            (hover[i] - 0.10).max(0.0)
        };
    }
    for _ in 0..(0.05 / DT) as usize {
        plant.step(&d);
    }
    assert!(
        plant.body_rate().z > 0.05,
        "differential CW spin-up produced yaw rate {:.4} rad/s, expected \
         a clear positive yaw",
        plant.body_rate().z
    );

    // Symmetric collective change: the reaction torques cancel, so any
    // residual yaw is a sign error somewhere in the pattern.
    let mut plant = fresh_plant();
    let d = SVector::<f32, 4>::from_fn(|i, _| (hover[i] + 0.10).min(1.0));
    for _ in 0..(0.05 / DT) as usize {
        plant.step(&d);
    }
    assert!(
        plant.body_rate().z.abs() < 1e-4,
        "symmetric spin-up produced yaw rate {:.5} rad/s — the reaction \
         torque signs do not cancel across the CW/CCW pairs",
        plant.body_rate().z
    );
}

/// With drag present, a vehicle held at a fixed tilt must reach a finite
/// terminal velocity — and it must be the one the drag coefficient
/// predicts, not merely "some finite number".
#[test]
fn drag_produces_predicted_terminal_velocity() {
    let p = plant_params();
    let mut plant = fresh_plant();
    // Lean the thrust axis 15° toward +x (rotation about body-y maps z_b
    // to (sinθ, 0, cosθ)), hold a vertically-balanced collective, and let
    // it accelerate until drag catches up.
    let tilt = 15.0_f32.to_radians();
    let att = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), tilt);
    plant.reset(Vector3::zeros(), Vector3::zeros(), att);

    // Collective that keeps vertical force in balance at this tilt.
    let per_motor = p.mass_kg * p.grav / (4.0 * tilt.cos());
    let d = SVector::<f32, 4>::from_fn(|i, _| {
        p.command_for_omega(i, p.omega_for_thrust(i, per_motor))
    });
    // Terminal-velocity time constant is m/(c_dx·Σω) ≈ 5 s; 40 s leaves
    // well under 1 % of the initial gap.
    for _ in 0..(40.0 / DT) as usize {
        plant.step(&d);
    }

    let v = plant.velocity();
    assert!(v.norm().is_finite() && v.norm() > 1.0, "no forward motion");

    // At terminal velocity along body-x: thrust·sin(tilt) = c_dx·Σω·v_bx.
    let omega_sum: f32 = plant.rotor_omega().iter().sum();
    let thrust: f32 = plant.motor_thrusts().iter().sum();
    let v_body = plant
        .attitude()
        .to_rotation_matrix()
        .inverse_transform_vector(&v);
    let drag_force = p.aero_drag.x * omega_sum * v_body.x;
    let thrust_along_x = thrust * tilt.sin();
    assert!(
        rel_err(drag_force, thrust_along_x) < 0.10,
        "at steady state drag {drag_force:.3} N should balance the \
         in-plane thrust component {thrust_along_x:.3} N"
    );
}

/// A drag-free, lag-free, reaction-free plant must reproduce the old
/// behaviour: neutral `sim:` defaults collapse the new model back onto
/// the previous one. Guards against the new terms leaking in when a
/// vehicle declares no `sim:` section.
#[test]
fn neutral_sim_params_are_drag_and_reaction_free() {
    let vp = default_vehicle();
    let neutral = vehicle_yaml::SimYaml::default();
    let p = PlantParams::from_config(&vp, &neutral);
    assert_eq!(p.aero_drag, Vector3::zeros());
    assert_eq!(p.rotor_inertia, 0.0);
    assert_eq!(p.omega_min[0], 0.0);
    assert_eq!(p.throttle_curve_k, 0.0);

    let mut plant = QuadPlant::new(vp, &neutral, DT);
    plant.reset(
        Vector3::zeros(),
        Vector3::new(10.0, 0.0, 0.0),
        UnitQuaternion::identity(),
    );
    let spf = plant.specific_force_body();
    assert_eq!(spf.x, 0.0, "neutral params must produce no drag");
    assert_eq!(spf.y, 0.0);
}
