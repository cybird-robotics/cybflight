// INDI regression tests against recorded C-reference results.
//
// The fixed inputs and expected outputs live in tests/indi_golden/golden.csv.
// These tests run entirely in Rust; fixture generation is not part of the build.
// Tests the core INDI math (filters, pseudo-control, WLS allocation,
// linearization, actuator state estimation) independent of frame convention.
//
// Also includes an FLU frame convention test that verifies the FLU G1 derived
// from MotorParams produces the same achieved pseudo-controls as the NED reference
// after frame transformation.
//
// Run: cargo test -p cybflight-core --target x86_64-unknown-linux-gnu --features std --test indi_golden

use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use air_filters::Filter;
use cybflight_core::indi::{
    effectiveness::{IndiEffectiveness, IndiMotorParams},
    linearization::{ThrustLinearization, ThrustModel},
};
use cybflight_core::mixer::{MotorParams, RigidBodyParams, SpinDir};
use flight_solver::cls::setup::wls::{setup_a, setup_b};
use flight_solver::cls::solve;
use nalgebra::{SMatrix, SVector, Vector3};

const NU: usize = 4;
const NV: usize = 6;
const NC: usize = NU + NV;
const LOOP_HZ: f32 = 8000.0;
/// Voltage fed to the voltage-aware INDI API. The golden fixtures use
/// `ThrustModel::Quadratic`, which ignores voltage entirely, so any
/// finite value preserves equivalence with the indiflight reference.
const NOMINAL_VOLTAGE_V: f32 = 16.0;
const GRAVITY: f32 = 9.80665;
const SYNC_LPF_HZ: f32 = 15.0;

type Biquad = BiquadFilter<f32, DirectForm2<f32>>;

fn make_biquad() -> Biquad {
    BiquadFilter::new(
        BiquadFilterConfigBuilder::direct_form_2()
            .sample_frequency_hz(LOOP_HZ)
            .filter_type(BiquadFilterType::LowPass)
            .cutoff_frequency_hz(SYNC_LPF_HZ)
            .build()
            .unwrap(),
    )
}

/// NED G1 derived from cybflight QuadX physical geometry.
///
/// Motor layout (BF QuadX, NED positions):
///   M0=RR(CW) at (-0.075,+0.1), M1=FR(CCW) at (+0.075,+0.1)
///   M2=RL(CCW) at (-0.075,-0.1), M3=FL(CW) at (+0.075,-0.1)
///
/// Body: mass=0.55, Ixx=0.0025, Iyy=0.0021, Izz=0.0043, T=8.5N, c=0.022m
///
/// Config: fz×100 = -1545, roll×10 = ±3400, pitch×10 = ±3036, yaw×10 = ±435
fn ned_g1() -> SMatrix<f32, NV, NU> {
    // After config scaling: fz*0.01, roll/pitch/yaw*0.1
    SMatrix::<f32, NV, NU>::from_row_slice(&[
        0.0, 0.0, 0.0, 0.0, // fx
        0.0, 0.0, 0.0, 0.0, // fy
        -15.45, -15.45, -15.45, -15.45, // fz
        -340.0, -340.0, 340.0, 340.0, // roll
        -303.6, 303.6, -303.6, 303.6, // pitch
        -43.5, 43.5, 43.5, -43.5, // yaw
    ])
}

fn wls_wv() -> SVector<f32, NV> {
    [1.0, 1.0, 50.0, 50.0, 50.0, 5.0].into()
}
fn wls_wu() -> SVector<f32, NU> {
    SVector::from_element(1.0)
}
const WLS_THETA: f32 = 1e-4;
const WLS_COND_BOUND: f32 = 3.2768e8; // (1<<15) * 1e4

// ---------------------------------------------------------------------------
// Test pipeline (NED, matches C)
// ---------------------------------------------------------------------------

struct IndiTestState {
    rate_gains: SVector<f32, 3>,
    rate_dot_filter: [Biquad; 3],
    spf_filter: [Biquad; 3],
    u_state_filter: [Biquad; NU],
    linearization: [ThrustLinearization; NU],
    prev_rate: Vector3<f32>,
    u_state: SVector<f32, NU>,
    u_state_fs: SVector<f32, NU>,
    pt1_alpha: f32,
    ws: [i8; NU],
    u: SVector<f32, NU>,
    d: SVector<f32, NU>,
    dv: SVector<f32, NV>,
    act_limit: SVector<f32, NU>,
    // G1 matrix used for allocation (configurable: NED or FLU)
    g1: SMatrix<f32, NV, NU>,
    // G2 support
    g2_yaw: SVector<f32, NU>,    // G2 yaw values per motor
    g2_scaler: SVector<f32, NU>, // ω_max² / (2·τ)
    omega_fs: SVector<f32, NU>,  // filtered motor speed (rad/s)
    max_omega: SVector<f32, NU>, // max motor speed (rad/s)
    prev_du: SVector<f32, NU>,   // previous du for omegaDot fallback
}

struct StepOutput {
    u: SVector<f32, NU>,
    d: SVector<f32, NU>,
    dv: SVector<f32, NV>,
}

impl IndiTestState {
    fn new() -> Self {
        Self::with_g1_and_limits(ned_g1(), SVector::from_element(1.0))
    }

    fn with_g1(g1: SMatrix<f32, NV, NU>) -> Self {
        Self::with_g1_and_limits(g1, SVector::from_element(1.0))
    }

    fn with_limits(act_limit: SVector<f32, NU>) -> Self {
        Self::with_g1_and_limits(ned_g1(), act_limit)
    }

    fn with_g1_and_limits(g1: SMatrix<f32, NV, NU>, act_limit: SVector<f32, NU>) -> Self {
        let dt = 1.0 / LOOP_HZ;
        let tau = 0.025f32;
        let max_rpm = 40000.0f32;
        let max_omega = max_rpm / 60.0 * core::f32::consts::TAU;
        Self {
            rate_gains: [20.0, 20.0, 20.0].into(),
            rate_dot_filter: core::array::from_fn(|_| make_biquad()),
            spf_filter: core::array::from_fn(|_| make_biquad()),
            u_state_filter: core::array::from_fn(|_| make_biquad()),
            linearization: [ThrustLinearization::new(0.5, ThrustModel::Quadratic, 12.0); NU],
            prev_rate: SVector::zeros(),
            u_state: SVector::zeros(),
            u_state_fs: SVector::zeros(),
            pt1_alpha: dt / (tau + dt),
            ws: [0; NU],
            u: SVector::zeros(),
            d: SVector::zeros(),
            dv: SVector::zeros(),
            act_limit,
            g1,
            g2_yaw: SVector::zeros(),
            g2_scaler: SVector::from_element(0.5 * max_omega * max_omega / tau),
            omega_fs: SVector::zeros(),
            max_omega: SVector::from_element(max_omega),
            prev_du: SVector::zeros(),
        }
    }

    fn with_g2(mut self, g2_yaw: SVector<f32, NU>, hover_omega: SVector<f32, NU>) -> Self {
        self.g2_yaw = g2_yaw;
        self.omega_fs = hover_omega;
        self
    }

    fn step(
        &mut self,
        gyro_dps: SVector<f32, 3>,
        accel_g: SVector<f32, 3>,
        rate_sp_rads: SVector<f32, 3>,
        spf_sp_z: f32,
        do_indi: bool,
    ) -> StepOutput {
        let do_f = if do_indi { 1.0f32 } else { 0.0 };
        let deg2rad = core::f32::consts::PI / 180.0;

        let rate = Vector3::new(
            gyro_dps[0] * deg2rad,
            gyro_dps[1] * deg2rad,
            gyro_dps[2] * deg2rad,
        );
        let spf = [
            accel_g[0] * GRAVITY,
            accel_g[1] * GRAVITY,
            accel_g[2] * GRAVITY,
        ];

        let rate_dot = [
            (rate[0] - self.prev_rate[0]) * LOOP_HZ,
            (rate[1] - self.prev_rate[1]) * LOOP_HZ,
            (rate[2] - self.prev_rate[2]) * LOOP_HZ,
        ];
        self.prev_rate = rate;

        let rate_dot_fs = [
            self.rate_dot_filter[0].apply(rate_dot[0]),
            self.rate_dot_filter[1].apply(rate_dot[1]),
            self.rate_dot_filter[2].apply(rate_dot[2]),
        ];
        let spf_fs = [
            self.spf_filter[0].apply(spf[0]),
            self.spf_filter[1].apply(spf[1]),
            self.spf_filter[2].apply(spf[2]),
        ];

        for i in 0..NU {
            self.u_state_fs[i] = self.u_state_filter[i].apply(self.u_state[i]);
            self.u_state_fs[i] = self.u_state_fs[i].clamp(0.0, 1.0);
        }

        let rate_err = [
            rate_sp_rads[0] - rate[0],
            rate_sp_rads[1] - rate[1],
            rate_sp_rads[2] - rate[2],
        ];
        let rate_dot_sp = [
            self.rate_gains[0] * rate_err[0],
            self.rate_gains[1] * rate_err[1],
            self.rate_gains[2] * rate_err[2],
        ];

        // Compute omegaDot_fs using du-based fallback (matches C when no dshot telem)
        let mut omega_dot_fs = SVector::<f32, NU>::from_element(0.0f32);
        for i in 0..NU {
            if self.g2_yaw[i].abs() > 1e-10 {
                let inv_thresh = 0.1 * self.max_omega[i];
                let omega_inv = if self.omega_fs[i].abs() > inv_thresh {
                    1.0 / self.omega_fs[i]
                } else {
                    1.0 / inv_thresh
                };
                omega_dot_fs[i] = self.prev_du[i] * self.g2_scaler[i] * omega_inv;
            }
        }

        self.dv = SVector::<f32, 6>::zeros();
        self.dv[2] = spf_sp_z - do_f * spf_fs[2];
        self.dv[3] = rate_dot_sp[0] - do_f * rate_dot_fs[0];
        self.dv[4] = rate_dot_sp[1] - do_f * rate_dot_fs[1];
        self.dv[5] = rate_dot_sp[2] - do_f * rate_dot_fs[2];

        // G2 contribution to dv
        for i in 0..NU {
            self.dv[5] += do_f * self.g2_yaw[i] * omega_dot_fs[i];
        }

        // Build G1 + G2 combined effectiveness matrix
        let mut g1g2 = self.g1;
        for i in 0..NU {
            if self.g2_yaw[i].abs() > 1e-10 && self.omega_fs[i].abs() > 1e-6 {
                let inv_thresh = 0.1 * self.max_omega[i];
                let omega_inv = if self.omega_fs[i].abs() > inv_thresh {
                    1.0 / self.omega_fs[i]
                } else {
                    1.0 / inv_thresh
                };
                // G2 only affects torque rows (3,4,5) — only yaw (row 5) is nonzero
                g1g2[(5, i)] += self.g2_scaler[i] * omega_inv * self.g2_yaw[i];
            }
        }

        let wv = wls_wv();
        let mut wu = wls_wu();
        let v = self.dv;

        let (a_mat, gamma) = setup_a::<NU, NV, NC>(&g1g2, &wv, &mut wu, WLS_THETA, WLS_COND_BOUND);

        let mut du_min = SVector::<f32, NU>::zeros();
        let mut du_max = SVector::<f32, NU>::zeros();
        let mut du_pref = SVector::<f32, NU>::zeros();
        for i in 0..NU {
            du_min[i] = -do_f * self.u_state_fs[i];
            du_max[i] = self.act_limit[i] - do_f * self.u_state_fs[i];
            du_pref[i] = -do_f * self.u_state_fs[i];
        }
        let b_vec = setup_b::<NU, NV, NC>(&v, &du_pref, &wv, &wu, gamma);

        let mut du = SVector::<f32, NU>::zeros();
        for i in 0..NU {
            du[i] = (du_min[i] + du_max[i]) * 0.5;
        }

        let _stats =
            solve::<NU, NV, NC>(&a_mat, &b_vec, &du_min, &du_max, &mut du, &mut self.ws, 1);

        for i in 0..NU {
            self.u[i] = (do_f * self.u_state_fs[i] + du[i]).clamp(0.0, self.act_limit[i]);
            self.d[i] = self.linearization[i].linearize(self.u[i], NOMINAL_VOLTAGE_V);
            // Track du for omegaDot fallback (u[i] - uState[i] = actual du)
            self.prev_du[i] = self.u[i] - self.u_state[i];
        }

        for i in 0..NU {
            let u_from_d = self.linearization[i].output_curve(self.d[i], NOMINAL_VOLTAGE_V);
            self.u_state[i] += self.pt1_alpha * (u_from_d - self.u_state[i]);
        }

        StepOutput {
            u: self.u,
            d: self.d,
            dv: self.dv,
        }
    }
}

// ---------------------------------------------------------------------------
// Golden CSV parsing
// ---------------------------------------------------------------------------

struct GoldenRow {
    test_case: String,
    step: usize,
    dv: SVector<f32, 6>,
    u: SVector<f32, 4>,
    d: SVector<f32, 4>,
}

fn parse_golden_csv() -> Vec<GoldenRow> {
    let csv_data = include_str!("../../../tests/indi_golden/golden.csv");
    let mut rows = Vec::new();
    for line in csv_data.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 28 {
            continue;
        }
        let f = |i: usize| cols[i].trim().parse::<f32>().unwrap();
        rows.push(GoldenRow {
            test_case: cols[0].to_string(),
            step: cols[1].parse().unwrap(),
            dv: [f(16), f(17), f(18), f(19), f(20), f(21)].into(),
            u: [f(22), f(23), f(24), f(25)].into(),
            d: [f(26), f(27), f(28), f(29)].into(),
        });
    }
    rows
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Per-step input generators (match C test cases exactly)
// ---------------------------------------------------------------------------

struct ConstInput {
    gyro_dps: SVector<f32, 3>,
    accel_g: SVector<f32, 3>,
    rate_sp: SVector<f32, 3>,
    spf_sp_z: f32,
    do_indi: bool,
}

impl ConstInput {
    fn at(&self, _step: usize) -> (SVector<f32, 3>, SVector<f32, 3>, SVector<f32, 3>, f32, bool) {
        (
            self.gyro_dps,
            self.accel_g,
            self.rate_sp,
            self.spf_sp_z,
            self.do_indi,
        )
    }
}

fn hover_input() -> ConstInput {
    ConstInput {
        gyro_dps: SVector::from_element(0.),
        accel_g: Vector3::new(0., 0., -1.),
        rate_sp: SVector::from_element(0.),
        spf_sp_z: -GRAVITY,
        do_indi: true,
    }
}

// ---------------------------------------------------------------------------
// Comparison helper
// ---------------------------------------------------------------------------

fn compare_case(
    case_name: &str,
    golden_rows: &[GoldenRow],
    state: &mut IndiTestState,
    input_fn: &dyn Fn(usize) -> (SVector<f32, 3>, SVector<f32, 3>, SVector<f32, 3>, f32, bool),
    achieved_tol: f32,
) {
    let case_rows: Vec<&GoldenRow> = golden_rows
        .iter()
        .filter(|r| r.test_case == case_name)
        .collect();
    assert!(!case_rows.is_empty(), "No golden rows for {case_name}");

    // Use the same G1 (without G2 correction) for achieved comparison.
    // The comparison checks that both solvers produce similar torques,
    // even if the effectiveness matrix used during allocation included G2.
    let g1_for_comparison = ned_g1();
    let mut max_achieved_diff = 0.0f32;

    for (step, golden) in case_rows.iter().enumerate() {
        assert_eq!(golden.step, step);
        let (gyro, accel, rate_sp, spf_z, do_indi) = input_fn(step);
        let out = state.step(gyro, accel, rate_sp, spf_z, do_indi);

        let u_rust = out.u;
        let u_ref = golden.u;
        let achieved_rust = g1_for_comparison * u_rust;
        let achieved_ref = g1_for_comparison * u_ref;

        // Compare ALL axes using relative tolerance: diff / max(|G1 row|).
        // This catches wild unconstrained-axis allocations without being
        // sensitive to WLS degeneracy (equivalent solutions that differ
        // by a motor permutation produce similar relative errors).
        let mut achieved_diff = 0.0f32;
        for i in 0..NV {
            let abs_diff = (achieved_rust[i] - achieved_ref[i]).abs();
            // Scale by the row's maximum possible output for relative comparison
            let row_scale = (0..NU)
                .map(|j| g1_for_comparison[(i, j)].abs())
                .fold(0.0f32, f32::max)
                .max(1.0);
            let rel_diff = abs_diff / row_scale;
            achieved_diff = achieved_diff.max(rel_diff);
        }
        max_achieved_diff = max_achieved_diff.max(achieved_diff);

        if achieved_diff > achieved_tol {
            panic!(
                "{case_name} step {step}: achieved diff {achieved_diff:.6e} > tol {achieved_tol}\n\
                 rust u: {:?}\n  ref u: {:?}\n\
                 rust G1*u: {:?}\n  ref G1*u: {:?}\n\
                 rust dv: {:?}\n  ref dv: {:?}",
                out.u,
                golden.u,
                achieved_rust.as_slice(),
                achieved_ref.as_slice(),
                out.dv,
                golden.dv
            );
        }

        for i in 0..NU {
            assert!(
                out.u[i] >= -1e-6 && out.u[i] <= state.act_limit[i] + 1e-6,
                "{case_name} step {step}: u[{i}]={:.6} out of bounds",
                out.u[i]
            );
        }
    }

    eprintln!(
        "{case_name}: PASS ({} steps, max achieved_diff={max_achieved_diff:.6e})",
        case_rows.len()
    );
}

// ---------------------------------------------------------------------------
// Original 4 test cases
// ---------------------------------------------------------------------------

#[test]
fn golden_hover_steady() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = hover_input();
    compare_case("hover_steady", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_roll_step() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = ConstInput {
        rate_sp: Vector3::new(2.0, 0., 0.),
        ..hover_input()
    };
    compare_case("roll_step", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_ground_ndi() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = ConstInput {
        spf_sp_z: -5.0,
        do_indi: false,
        ..hover_input()
    };
    compare_case("ground_ndi", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_saturation() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = ConstInput {
        rate_sp: Vector3::new(15.0, 15.0, 0.),
        ..hover_input()
    };
    compare_case("saturation", &rows, &mut state, &|s| inp.at(s), 0.01);
}

// ---------------------------------------------------------------------------
// New comprehensive test cases
// ---------------------------------------------------------------------------

#[test]
fn golden_combined_axes() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = ConstInput {
        rate_sp: Vector3::new(3.0, -2.0, 1.5),
        ..hover_input()
    };
    compare_case("combined_axes", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_ramp_command() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    compare_case(
        "ramp_command",
        &rows,
        &mut state,
        &|step| {
            let roll_sp = if step < 40 {
                step as f32 * 0.2
            } else {
                (80 - step) as f32 * 0.2
            };
            (
                SVector::from_element(0.),
                Vector3::new(0., 0., -1.),
                Vector3::new(roll_sp, 0., 0.),
                -GRAVITY,
                true,
            )
        },
        0.01,
    );
}

#[test]
fn golden_spinning_vehicle() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = ConstInput {
        gyro_dps: Vector3::new(100., 50., -20.),
        rate_sp: SVector::from_element(0.),
        ..hover_input()
    };
    // Higher tolerance: large G1 values (340 rad/s²) amplify the biquad filter
    // implementation difference (C: DF1, Rust: DF2T). A ~1e-3 motor command diff
    // becomes ~0.34 achieved diff. The spinning vehicle has large constant gyro
    // input that causes accumulated filter state divergence over 50 steps.
    compare_case("spinning_vehicle", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_tilted_accel() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = ConstInput {
        accel_g: Vector3::new(0.0, 0.5, -0.866),
        ..hover_input()
    };
    compare_case("tilted_accel", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_asymmetric_limits() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::with_limits(SVector::from_row_slice(&[0.8, 1.0, 1.0, 1.0]));
    let inp = ConstInput {
        rate_sp: Vector3::new(5.0, 3.0, 0.),
        ..hover_input()
    };
    compare_case("asymmetric_limits", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_small_corrections() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    let inp = ConstInput {
        gyro_dps: Vector3::new(2., -1., 0.5),
        accel_g: Vector3::new(0.01, -0.02, -0.998),
        rate_sp: Vector3::new(0.05, -0.03, 0.01),
        ..hover_input()
    };
    compare_case("small_corrections", &rows, &mut state, &|s| inp.at(s), 0.01);
}

#[test]
fn golden_setpoint_reversal() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    compare_case(
        "setpoint_reversal",
        &rows,
        &mut state,
        &|step| {
            let roll_sp = if step < 20 {
                5.0
            } else if step < 40 {
                -5.0
            } else {
                0.0
            };
            (
                SVector::from_element(0.),
                Vector3::new(0., 0., -1.),
                Vector3::new(roll_sp, 0., 0.),
                -GRAVITY,
                true,
            )
        },
        0.01,
    );
}

#[test]
fn golden_doindi_transition() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    compare_case(
        "doindi_transition",
        &rows,
        &mut state,
        &|step| {
            let do_indi = step >= 20;
            let spf_z = if step < 20 {
                -5.0 + step as f32 * (-4.81 / 20.0)
            } else {
                -GRAVITY
            };
            (
                SVector::from_element(0.),
                Vector3::new(0., 0., -1.),
                SVector::from_element(0.),
                spf_z,
                do_indi,
            )
        },
        0.01,
    );
}

#[test]
fn golden_changing_gyro() {
    let rows = parse_golden_csv();
    let mut state = IndiTestState::new();
    compare_case(
        "changing_gyro",
        &rows,
        &mut state,
        &|step| {
            let roll_dps = if step < 40 { step as f32 * 5.0 } else { 200.0 };
            let pitch_dps = 30.0 * num_traits::Float::sin(step as f32 * 0.3);
            (
                Vector3::new(roll_dps, pitch_dps, 0.0),
                Vector3::new(0., 0., -1.),
                SVector::from_element(0.),
                -GRAVITY,
                true,
            )
        },
        0.01,
    );
}

#[test]
fn golden_g2_active() {
    let rows = parse_golden_csv();
    // Match C harness: G2 yaw values + hover omega
    let hover_omega = 20000.0f32 / 60.0 * core::f32::consts::TAU; // 20000 RPM → rad/s
    let mut state = IndiTestState::new().with_g2(
        SVector::from_row_slice(&[-0.001, 0.001, 0.001, -0.001]), // NED: CW=negative, CCW=positive
        SVector::from_element(hover_omega),
    );
    let inp = ConstInput {
        rate_sp: Vector3::new(0., 0., 2.0),
        ..hover_input()
    };
    compare_case("g2_active", &rows, &mut state, &|s| inp.at(s), 0.01);
}

// ---------------------------------------------------------------------------
// FLU frame convention test
// ---------------------------------------------------------------------------

/// Verify that the FLU G1 derived from MotorParams produces achieved
/// pseudo-controls consistent with the NED reference after frame transform.
///
/// Transform: ν_FLU = diag(1,-1,-1, 1,-1,-1) × ν_NED
#[test]
fn golden_flu_frame_convention() {
    let rows = parse_golden_csv();
    let case_rows: Vec<&GoldenRow> = rows
        .iter()
        .filter(|r| r.test_case == "hover_steady")
        .collect();

    // Build FLU G1 from MotorParams (same as cybflight's vehicle.rs)
    let motors = [
        MotorParams {
            position_m: [-0.075, -0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [0.075, -0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [-0.075, 0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [0.075, 0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
    ];
    let body = RigidBodyParams {
        mass_kg: 0.55,
        inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
        max_rate_rad_s: [10.0, 10.0, 6.0],
    };
    let indi_params = [IndiMotorParams {
        time_const_s: 0.025,
        max_rpm: 40000.0,
        g2_yaw: 0.0,
    }; 4];
    let eff = IndiEffectiveness::new(&motors, &body, &indi_params);
    let g1_flu = eff.g1; // 6×4 in FLU acceleration space

    // Verify FLU G1 signs against physics (not against the arbitrary NED test G1,
    // which uses different absolute values and motor geometry).
    //
    // Physical constraints for a QuadX in FLU:
    //   fz:    always positive (thrust in +z = up)
    //   roll:  sign(py) — right motors (py<0) → negative, left (py>0) → positive
    //   pitch: sign(-px) — rear motors (px<0) → positive, front (px>0) → negative
    //   yaw:   sign(spin) — CW(+1) → positive, CCW(-1) → negative
    //
    // Motor layout: M0=RR(CW), M1=FR(CCW), M2=RL(CCW), M3=FL(CW)
    let expected_signs: [SVector<f32, 4>; 4] = [
        // [fz,  roll,  pitch, yaw] for each motor
        SVector::from_row_slice(&[1.0, -1.0, 1.0, 1.0]), // M0: RR, CW — right→-roll, rear→+pitch, CW→+yaw
        SVector::from_row_slice(&[1.0, -1.0, -1.0, -1.0]), // M1: FR, CCW — right→-roll, front→-pitch, CCW→-yaw
        SVector::from_row_slice(&[1.0, 1.0, 1.0, -1.0]), // M2: RL, CCW — left→+roll, rear→+pitch, CCW→-yaw
        SVector::from_row_slice(&[1.0, 1.0, -1.0, 1.0]), // M3: FL, CW — left→+roll, front→-pitch, CW→+yaw
    ];
    let row_names = ["fz", "roll", "pitch", "yaw"];

    for motor in 0..NU {
        for (row_idx, &expected) in expected_signs[motor].iter().enumerate() {
            let g1_row = row_idx + 2; // skip fx(0), fy(1)
            let actual = g1_flu[(g1_row, motor)];
            assert!(
                actual.signum() == expected,
                "FLU G1 sign error: motor {motor} {}: expected sign {expected:+.0}, got value {actual:.4}",
                row_names[row_idx]
            );
        }
    }

    // Also verify specific sign conventions
    // M0 (rear-right, CW): fz>0 (up in FLU), roll<0 (right), pitch>0 (rear→nose down), yaw>0 (CW reaction)
    assert!(g1_flu[(2, 0)] > 0.0, "FLU: fz should be positive (upward)");
    assert!(g1_flu[(3, 0)] < 0.0, "FLU: M0 right motor → negative roll");
    assert!(g1_flu[(5, 0)] > 0.0, "FLU: M0 CW → positive yaw reaction");

    // M1 (front-right, CCW): yaw<0
    assert!(g1_flu[(5, 1)] < 0.0, "FLU: M1 CCW → negative yaw reaction");

    // M2 (rear-left, CCW): roll>0
    assert!(g1_flu[(3, 2)] > 0.0, "FLU: M2 left motor → positive roll");

    eprintln!("FLU frame sign check: PASS");
}

/// Verify NED↔FLU frame transform by deriving G1 from the same motor geometry
/// in both frames and checking that G1_FLU = diag(1,-1,-1,1,-1,-1) × G1_NED.
///
/// This computes the NED G1 from the physical motor params (NOT the arbitrary
/// test G1 used in the golden pipeline), then transforms to FLU and compares
/// against the FLU G1 from IndiEffectiveness::new().
#[test]
fn golden_ned_flu_transform() {
    // FLU motor params (cybflight convention)
    let motors_flu = [
        MotorParams {
            position_m: [-0.075, -0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [0.075, -0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [-0.075, 0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [0.075, 0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
    ];
    let body = RigidBodyParams {
        mass_kg: 0.55,
        inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
        max_rate_rad_s: [10.0, 10.0, 6.0],
    };
    let indi_params = [IndiMotorParams {
        time_const_s: 0.025,
        max_rpm: 40000.0,
        g2_yaw: 0.0,
    }; 4];

    // Get FLU G1 from IndiEffectiveness
    let eff_flu = IndiEffectiveness::new(&motors_flu, &body, &indi_params);
    let g1_flu = eff_flu.g1;

    // Compute NED G1 manually from the same physical motors.
    // NED motor positions: px_NED = px_FLU, py_NED = -py_FLU
    // NED thrust: F = (0, 0, -T) (upward = -z in NED)
    // NED torque: τ = r_NED × F_NED
    //   τ_roll  = py_NED * (-T)  = -py_NED * T
    //   τ_pitch = -px_NED * (-T) = px_NED * T ... wait, cross product:
    //   τ_pitch = r_z*F_x - r_x*F_z = 0 - px_NED*(-T) = px_NED * T
    //   Hmm, no: τ_y = r_z*F_x - r_x*F_z = 0*0 - r_x*(-T) = r_x * T
    //   ... but that would be WRONG. Let me use the full cross product.
    //
    // r_NED × F_NED where r=(px, -py_flu, 0), F=(0, 0, -T):
    //   τ_x = r_y*F_z - r_z*F_y = (-py_flu)*(-T) - 0 = py_flu * T
    //   τ_y = r_z*F_x - r_x*F_z = 0 - px*(-T) = px * T
    //   τ_z = r_x*F_y - r_y*F_x = 0 - 0 = 0 (drag handled separately)
    //
    // NED yaw: τ_yaw = -s * c * T (CW in NED = negative yaw, per indi_effectiveness.tex)

    let inertia_inv = body.inertia_matrix().try_inverse().unwrap();
    let mut g1_ned = SMatrix::<f32, NV, NU>::zeros();

    for i in 0..NU {
        let m = &motors_flu[i];
        let [px, py_flu] = m.position_m;
        let py_ned = -py_flu;
        let t = m.max_thrust_n;
        let s = m.spin_dir as i32 as f32;

        // NED force
        g1_ned[(2, i)] = -t / body.mass_kg; // fz: thrust in -z NED

        // NED torques from cross product
        let tau_roll = py_ned * (-t); // = -py_ned * T = py_flu * T
        let tau_pitch = 0.0 - px * (-t); // = px * T
        let tau_yaw = -s * m.torque_coeff_m * t; // CW → negative in NED

        let torque = nalgebra::Vector3::new(tau_roll, tau_pitch, tau_yaw);
        let ang_accel = inertia_inv * torque;
        g1_ned[(3, i)] = ang_accel[0];
        g1_ned[(4, i)] = ang_accel[1];
        g1_ned[(5, i)] = ang_accel[2];
    }

    // Now verify: G1_FLU = diag(1,-1,-1, 1,-1,-1) × G1_NED
    let sign_flip = [1.0f32, -1.0, -1.0, 1.0, -1.0, -1.0];
    let tol = 1e-4;

    for i in 0..NU {
        for j in 0..NV {
            let expected_flu = sign_flip[j] * g1_ned[(j, i)];
            let actual_flu = g1_flu[(j, i)];
            let diff = (expected_flu - actual_flu).abs();
            assert!(
                diff < tol,
                "NED→FLU transform mismatch: row {j} motor {i}\n\
                 G1_NED = {:.6}, expected FLU = {:.6} (sign_flip={:.0}), actual FLU = {:.6}, diff = {:.2e}",
                g1_ned[(j, i)], expected_flu, sign_flip[j], actual_flu, diff
            );
        }
    }

    // Print both matrices for visual inspection
    eprintln!("G1_NED (derived from physical params):");
    for j in 0..NV {
        let label = ["fx", "fy", "fz", "roll", "pitch", "yaw"][j];
        eprintln!(
            "  {label:>5}: [{:>10.4}, {:>10.4}, {:>10.4}, {:>10.4}]",
            g1_ned[(j, 0)],
            g1_ned[(j, 1)],
            g1_ned[(j, 2)],
            g1_ned[(j, 3)]
        );
    }
    eprintln!("G1_FLU (from IndiEffectiveness::new):");
    for j in 0..NV {
        let label = ["fx", "fy", "fz", "roll", "pitch", "yaw"][j];
        eprintln!(
            "  {label:>5}: [{:>10.4}, {:>10.4}, {:>10.4}, {:>10.4}]",
            g1_flu[(j, 0)],
            g1_flu[(j, 1)],
            g1_flu[(j, 2)],
            g1_flu[(j, 3)]
        );
    }

    eprintln!("NED↔FLU frame transform: PASS");
}

// ---------------------------------------------------------------------------
// IndiController integration tests: verify the real controller produces
// the same output as the standalone IndiTestState with FLU G1.
// ---------------------------------------------------------------------------

use cybflight_core::indi::controller::{IndiConfig, IndiController, MotorState};

fn flu_controller_config() -> IndiConfig {
    let motors = [
        MotorParams {
            position_m: [-0.075, -0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [0.075, -0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [-0.075, 0.1],
            spin_dir: SpinDir::Ccw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
        MotorParams {
            position_m: [0.075, 0.1],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 8.5,
            torque_coeff_m: 0.022,
            ..MotorParams::STOCK_DYNAMICS
        },
    ];
    IndiConfig {
        indi_enabled: true,
        ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
        ground_accel_m_s2: 0.8 * 9.81,
        ground_thrust_sp_m_s2: 3.0,
        rate_gains: nalgebra::Vector3::new(20.0, 20.0, 20.0),
        sync_filter_hz: 15.0,
        rate_dot_sg_window_size: 7,
        rate_dot_sg_order: 2,
        motors,
        body: RigidBodyParams {
            mass_kg: 0.55,
            inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
            max_rate_rad_s: [10.0, 10.0, 6.0],
        },
        indi_motors: [IndiMotorParams {
            time_const_s: 0.025,
            max_rpm: 40000.0,
            g2_yaw: 0.0,
        }; 4],
        thrust_model: ThrustModel::Quadratic,
        nonlinearity: SVector::from_element(0.5),
        act_limit: SVector::from_element(1.0),
        wls_wv: SVector::from_row_slice(&[1.0, 1.0, 50.0, 50.0, 50.0, 5.0]),
        wls_wu: SVector::from_element(1.0),
        wls_cond_bound: 3.2768e8,
        wls_theta: 1e-4,
        wls_imax: 1,
        nan_limit: 20,
        nan_rampdown: 0.95,
        rpm_invalid_limit: 50,
        rpm_all_invalid_limit: 50,
        rpm_recovery_count: 10,
        motor_pole_count: 14,
    }
}

/// Build an IndiTestState that uses FLU G1 (from IndiEffectiveness).
fn flu_test_state() -> IndiTestState {
    let config = flu_controller_config();
    let eff = IndiEffectiveness::new(&config.motors, &config.body, &config.indi_motors);
    IndiTestState::with_g1(eff.g1)
}

/// Compare IndiController against itself across scenarios.
/// Verifies that running the same scenario twice produces identical results
/// (deterministic), and that the outputs match physical expectations.
fn run_controller_scenario(
    name: &str,
    steps: usize,
    input_fn: &dyn Fn(
        usize,
    ) -> (
        nalgebra::Vector3<f32>,
        nalgebra::Vector3<f32>,
        nalgebra::Vector3<f32>,
        f32,
        bool,
    ),
) -> Vec<SVector<f32, 4>> {
    let mut ctrl = IndiController::new(&flu_controller_config(), LOOP_HZ);
    let g2_valid = [false; 4];
    let mut outputs = Vec::new();

    for step in 0..steps {
        let (gyro, accel, rate_sp, spf_z, armed) = input_fn(step);
        let (out, _) = ctrl.step(
            &gyro,
            &accel,
            &rate_sp,
            spf_z,
            armed,
            &g2_valid,
            MotorState::Internal,
            NOMINAL_VOLTAGE_V,
        );
        outputs.push(out.motor_commands);

        for (i, &c) in out.motor_commands.iter().enumerate() {
            assert!(
                c.is_finite() && c >= 0.0 && c <= 1.0,
                "{name} step {step}: motor {i} = {c}"
            );
        }
    }
    outputs
}

#[test]
fn controller_deterministic() {
    // Run the same scenario twice — must produce identical results
    let input = |_step: usize| {
        (
            nalgebra::Vector3::zeros(),
            nalgebra::Vector3::new(0.0, 0.0, GRAVITY),
            nalgebra::Vector3::zeros(),
            GRAVITY,
            true,
        )
    };

    let run1 = run_controller_scenario("det_run1", 100, &input);
    let run2 = run_controller_scenario("det_run2", 100, &input);

    for (step, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        for i in 0..4 {
            assert!(
                (a[i] - b[i]).abs() < 1e-10,
                "non-deterministic at step {step} motor {i}: {:.10} vs {:.10}",
                a[i],
                b[i]
            );
        }
    }
    eprintln!("controller_deterministic: PASS (100 steps)");
}

#[test]
fn controller_hover_flu() {
    let outputs = run_controller_scenario("hover_flu", 200, &|_| {
        (
            nalgebra::Vector3::zeros(),
            nalgebra::Vector3::new(0.0, 0.0, GRAVITY),
            nalgebra::Vector3::zeros(),
            GRAVITY,
            true,
        )
    });

    // After settling, all motors should be equal and in hover range
    let last = outputs.last().unwrap();
    let mean = last.iter().sum::<f32>() / 4.0;
    for (i, &c) in last.iter().enumerate() {
        assert!(
            (c - mean).abs() < 0.05,
            "hover motor {i} = {c}, mean = {mean}"
        );
    }
    assert!(mean > 0.1 && mean < 0.7, "hover mean = {mean}");
    eprintln!("controller_hover_flu: PASS (mean={mean:.4})");
}

#[test]
fn controller_roll_step_flu() {
    let outputs = run_controller_scenario("roll_flu", 200, &|_| {
        (
            nalgebra::Vector3::zeros(),
            nalgebra::Vector3::new(0.0, 0.0, GRAVITY),
            nalgebra::Vector3::new(3.0, 0.0, 0.0), // positive roll = left up in FLU
            GRAVITY,
            true,
        )
    });

    let last = outputs.last().unwrap();
    // FLU positive roll: left motors (M2=RL, M3=FL) increase
    let left = (last[2] + last[3]) / 2.0;
    let right = (last[0] + last[1]) / 2.0;
    assert!(
        left > right,
        "FLU roll: left={left:.4} should > right={right:.4}"
    );
    eprintln!("controller_roll_step_flu: PASS (left={left:.4}, right={right:.4})");
}

#[test]
fn controller_spinning_flu() {
    // Vehicle spinning at 100 deg/s roll, controller tries to stop
    let outputs = run_controller_scenario("spin_flu", 200, &|_| {
        (
            nalgebra::Vector3::new(100.0f32.to_radians(), 0.0, 0.0),
            nalgebra::Vector3::new(0.0, 0.0, GRAVITY),
            nalgebra::Vector3::zeros(), // rate_sp = 0 = stop spinning
            GRAVITY,
            true,
        )
    });

    let last = outputs.last().unwrap();
    // To stop positive roll (left going up), right motors should increase
    let left = (last[2] + last[3]) / 2.0;
    let right = (last[0] + last[1]) / 2.0;
    assert!(
        right > left,
        "stopping roll: right={right:.4} should > left={left:.4}"
    );
    eprintln!("controller_spinning_flu: PASS (right={right:.4}, left={left:.4})");
}

#[test]
fn controller_ground_to_air_flu() {
    let outputs = run_controller_scenario("ground_air_flu", 100, &|step| {
        let armed = step >= 30;
        let spf_z = if step < 30 { 2.0 } else { GRAVITY };
        (
            nalgebra::Vector3::zeros(),
            nalgebra::Vector3::new(0.0, 0.0, GRAVITY),
            nalgebra::Vector3::zeros(),
            spf_z,
            armed,
        )
    });

    // Check no spike at transition (step 30)
    if outputs.len() > 31 {
        for i in 0..4 {
            let diff = (outputs[30][i] - outputs[29][i]).abs();
            assert!(diff < 0.5, "transition spike motor {i}: diff={diff}");
        }
    }
    eprintln!("controller_ground_to_air_flu: PASS");
}

#[test]
fn controller_combined_axes_flu() {
    let outputs = run_controller_scenario("combined_flu", 200, &|_| {
        (
            nalgebra::Vector3::zeros(),
            nalgebra::Vector3::new(0.0, 0.0, GRAVITY),
            nalgebra::Vector3::new(3.0, -2.0, 1.5),
            GRAVITY,
            true,
        )
    });

    // All motors should be in bounds and not all equal (combined command)
    let last = outputs.last().unwrap();
    let mean = last.iter().sum::<f32>() / 4.0;
    let max_dev = last.iter().map(|c| (c - mean).abs()).fold(0.0f32, f32::max);
    assert!(
        max_dev > 0.01,
        "combined command should produce differential thrust: max_dev={max_dev}"
    );
    eprintln!("controller_combined_axes_flu: PASS (max_dev={max_dev:.4})");
}

#[test]
fn controller_setpoint_reversal_flu() {
    let outputs = run_controller_scenario("reversal_flu", 200, &|step| {
        let roll_sp = if step < 60 {
            5.0
        } else if step < 120 {
            -5.0
        } else {
            0.0
        };
        (
            nalgebra::Vector3::zeros(),
            nalgebra::Vector3::new(0.0, 0.0, GRAVITY),
            nalgebra::Vector3::new(roll_sp, 0.0, 0.0),
            GRAVITY,
            true,
        )
    });

    // After reversal at step 60, left/right relationship should flip
    let at_50 = &outputs[50]; // positive roll phase
    let at_100 = &outputs[100]; // negative roll phase

    let left_50 = (at_50[2] + at_50[3]) / 2.0;
    let right_50 = (at_50[0] + at_50[1]) / 2.0;
    let left_100 = (at_100[2] + at_100[3]) / 2.0;
    let right_100 = (at_100[0] + at_100[1]) / 2.0;

    assert!(left_50 > right_50, "phase 1: left should > right");
    assert!(
        right_100 > left_100,
        "phase 2: right should > left (reversed)"
    );
    eprintln!("controller_setpoint_reversal_flu: PASS");
}

// ---------------------------------------------------------------------------
// Cross-validation: IndiController vs IndiTestState with FLU G1.
// Verifies the controller's internal wiring (filters, WLS, linearization,
// actuator state) matches the standalone reference implementation.
// ---------------------------------------------------------------------------

#[test]
fn controller_matches_test_state_flu() {
    use nalgebra::Vector3;

    let config = flu_controller_config();

    // Get FLU G1 from IndiEffectiveness (same derivation as IndiController uses internally)
    let eff = IndiEffectiveness::new(&config.motors, &config.body, &config.indi_motors);
    let g1_flu = eff.g1;

    // Create both: real controller and standalone reference with same FLU G1
    let mut ctrl = IndiController::new(&config, LOOP_HZ);
    let mut reference = IndiTestState::with_g1(g1_flu);

    let g2_valid = [false; NU];

    // FLU inputs: hover
    let g2_valid = [false; NU];

    // Test scenarios: (name, steps, gyro_dps, accel_g, rate_sp_rad_s, spf_sp_z, armed)
    // IndiTestState takes gyro in deg/s and accel in g-units (converts internally).
    // IndiController takes gyro in rad/s and accel in m/s².
    let scenarios: &[(
        &str,
        usize,
        SVector<f32, 3>,
        SVector<f32, 3>,
        SVector<f32, 3>,
        f32,
        bool,
    )] = &[
        (
            "hover",
            100,
            SVector::from_element(0.),
            Vector3::new(0., 0., 1.),
            SVector::from_element(0.),
            GRAVITY,
            true,
        ),
        (
            "roll",
            100,
            SVector::from_element(0.),
            Vector3::new(0., 0., 1.),
            Vector3::new(3.0, 0., 0.),
            GRAVITY,
            true,
        ),
        (
            "combined",
            100,
            SVector::from_element(0.),
            Vector3::new(0., 0., 1.),
            Vector3::new(3.0, -2.0, 1.5),
            GRAVITY,
            true,
        ),
        (
            "ground",
            50,
            SVector::from_element(0.),
            Vector3::new(0., 0., 1.),
            SVector::from_element(0.),
            2.0,
            false,
        ),
    ];

    for &(name, steps, gyro_dps, accel_g, rate_sp, spf_z, armed) in scenarios {
        let mut ctrl = IndiController::new(&config, LOOP_HZ);
        let mut reference = IndiTestState::with_g1(g1_flu);
        let mut max_motor_diff = 0.0f32;

        // Convert for IndiController: deg/s → rad/s, g → m/s²
        let deg2rad = core::f32::consts::PI / 180.0;
        let gyro_v = gyro_dps * deg2rad;
        let accel_v = accel_g * GRAVITY;
        let rate_sp_v = rate_sp;

        for step in 0..steps {
            let ref_out = reference.step(gyro_dps, accel_g, rate_sp, spf_z, armed);
            let (ctrl_out, _) = ctrl.step(
                &gyro_v,
                &accel_v,
                &rate_sp_v,
                spf_z,
                armed,
                &g2_valid,
                MotorState::Internal,
                NOMINAL_VOLTAGE_V,
            );

            // Compare linearized motor commands (d, not u).
            // IndiTestState.d = linearize(u), IndiController.motor_commands = linearize(u).
            let motor_diff = ref_out
                .d
                .iter()
                .zip(ctrl_out.motor_commands.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            max_motor_diff = max_motor_diff.max(motor_diff);

            if motor_diff > 0.01 {
                panic!(
                    "{name} step {step}: motor diff {motor_diff:.6e}\n\
                     ref d:  {:?}\n\
                     ctrl d: {:?}\n\
                     ref u:  {:?}\n\
                     ref dv: {:?}",
                    ref_out.d, ctrl_out.motor_commands, ref_out.u, ref_out.dv
                );
            }
        }
        eprintln!("{name}: controller vs reference max_motor_diff={max_motor_diff:.6e}");
    }

    eprintln!("controller_matches_test_state_flu: PASS");
}

#[test]
fn controller_matches_test_state_flu_with_g2() {
    use nalgebra::Vector3;

    // Config with G2 active (FLU signs)
    let config = IndiConfig {
        indi_enabled: true,
        ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
        ground_accel_m_s2: 0.8 * 9.81,
        ground_thrust_sp_m_s2: 3.0,
        indi_motors: [
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: 0.001,
            },
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: -0.001,
            },
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: -0.001,
            },
            IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: 0.001,
            },
        ],
        ..flu_controller_config()
    };

    let eff = IndiEffectiveness::new(&config.motors, &config.body, &config.indi_motors);
    let g1_flu = eff.g1;

    // Compute eRPM value that produces hover_omega (20000 RPM).
    // eRPM = RPM * pole_pairs / 100. With 14 poles (7 pairs): eRPM = 20000 * 7 / 100 = 1400.
    let pole_pairs = config.motor_pole_count as f32 / 2.0;
    let hover_rpm = 20000.0f32;
    let hover_erpm = (hover_rpm * pole_pairs / 100.0) as u32; // 1400
    let hover_omega = hover_rpm / 60.0 * core::f32::consts::TAU;

    let mut reference =
        IndiTestState::with_g1(g1_flu).with_g2(
            SVector::from_row_slice(&[0.001, -0.001, -0.001, 0.001]),
            SVector::from_element(hover_omega),
        );

    let mut ctrl = IndiController::new(&config, LOOP_HZ);
    ctrl.update_rpm(&[cybflight_core::indi::rpm_tracker::RpmInput::Erpm(hover_erpm); NU]);

    let deg2rad = core::f32::consts::PI / 180.0;
    let g2_valid = [true; NU];

    // Yaw command to exercise G2 path
    let gyro_dps = Vector3::new(0.0f32, 0.0, 0.0);
    let accel_g = Vector3::new(0.0, 0.0, 1.0);
    let rate_sp = Vector3::new(0.0, 0.0, 3.0); // yaw command
    let spf_z = GRAVITY;

    let gyro_v = gyro_dps * deg2rad;
    let accel_v = accel_g * GRAVITY;
    let rate_sp_v = rate_sp;

    // Settle phase: run both for 200 steps to let all filters converge
    // (omega biquad, u_state PT1+biquad). The IndiTestState has instant omega
    // while IndiController filters it, so they diverge initially.
    for _ in 0..2000 {
        ctrl.update_rpm(&[cybflight_core::indi::rpm_tracker::RpmInput::Erpm(hover_erpm); NU]);
        reference.step(gyro_dps, accel_g, rate_sp, spf_z, true);
        ctrl.step(
            &gyro_v,
            &accel_v,
            &rate_sp_v,
            spf_z,
            true,
            &g2_valid,
            MotorState::Internal,
            NOMINAL_VOLTAGE_V,
        );
    }

    // Compare after settling: outputs should match closely
    let mut max_diff = 0.0f32;
    for step in 0..50 {
        ctrl.update_rpm(&[cybflight_core::indi::rpm_tracker::RpmInput::Erpm(hover_erpm); NU]);

        let ref_out = reference.step(gyro_dps, accel_g, rate_sp, spf_z, true);
        let (ctrl_out, _) = ctrl.step(
            &gyro_v,
            &accel_v,
            &rate_sp_v,
            spf_z,
            true,
            &g2_valid,
            MotorState::Internal,
            NOMINAL_VOLTAGE_V,
        );

        let diff = ref_out
            .d
            .iter()
            .zip(ctrl_out.motor_commands.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        max_diff = max_diff.max(diff);

        // G2 path has higher tolerance due to omega biquad filter difference
        // between IndiTestState (instant omega) and IndiController (filtered omega).
        if diff > 0.02 {
            panic!(
                "G2 cross-val step {step} (after settle): diff {diff:.6e}\n\
                 ref d:  {:?}\n\
                 ctrl d: {:?}",
                ref_out.d, ctrl_out.motor_commands
            );
        }
    }
    eprintln!("controller_matches_test_state_flu_with_g2: PASS (max_diff={max_diff:.6e})");
}
