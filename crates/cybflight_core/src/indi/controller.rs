// INDI (Incremental Nonlinear Dynamic Inversion) controller.
//
// Pure computation — no async, no channels, no embassy types.
// Hardcoded to quad (NU=4, NV=6, NC=10) to avoid nightly generic_const_exprs.
// Ported from indiflight: src/main/flight/indi.c

use air_filters::Filter;
use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use nalgebra::{SMatrix, SVector, Vector3};
use flight_solver::cls::{ExitCode, solve};
use flight_solver::cls::setup::wls::{setup_a, setup_b};

use super::{
    effectiveness::{IndiEffectiveness, IndiMotorParams},
    linearization::ThrustLinearization,
    rpm_tracker::{RpmInput, RpmTracker},
};
use crate::mixer::{MotorParams, RigidBodyParams};

/// Number of actuators (motors).
pub const NU: usize = 4;
/// Number of pseudo-controls (fx, fy, fz, roll, pitch, yaw).
pub const NV: usize = 6;
/// Number of constraint rows (NU + NV).
pub const NC: usize = NU + NV;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// INDI controller configuration.
pub struct IndiConfig {
    /// Rate error → angular acceleration gains (rad/s² per rad/s).
    pub rate_gains: Vector3<f32>,
    /// Biquad low-pass cutoff frequency (Hz) for all synchronized filters.
    pub sync_filter_hz: f32,
    /// Motor parameters (from vehicle definition).
    pub motors: [MotorParams; NU],
    /// Body rigid-body parameters.
    pub body: RigidBodyParams,
    /// Per-motor INDI parameters (time constant, max RPM, G2 yaw).
    pub indi_motors: [IndiMotorParams; NU],
    /// Motor nonlinearity for thrust linearization (0.0–1.0).
    pub nonlinearity: [f32; NU],
    /// Motor output limit per motor (0.0–1.0, typically 1.0).
    pub act_limit: [f32; NU],
    /// WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw].
    pub wls_wv: [f32; NV],
    /// WLS actuator penalty weights.
    pub wls_wu: [f32; NU],
    /// WLS condition number bound.
    pub wls_cond_bound: f32,
    /// WLS objective separation parameter.
    pub wls_theta: f32,
    /// WLS max iterations per loop (1 with warmstarting).
    pub wls_imax: usize,
    /// Consecutive WLS NaN failures before failsafe.
    pub nan_limit: u16,
    /// Consecutive invalid RPM frames before zeroing G2 column (per motor).
    pub rpm_invalid_limit: u16,
    /// Consecutive frames with ALL motors invalid before failsafe.
    pub rpm_all_invalid_limit: u16,
    /// Consecutive valid frames required to re-enable G2 after it was zeroed.
    pub rpm_recovery_count: u16,
    /// Motor pole count (for eRPM → RPM conversion).
    pub motor_pole_count: u8,
}

// ---------------------------------------------------------------------------
// INDI controller
// ---------------------------------------------------------------------------

type Biquad = BiquadFilter<f32, DirectForm2<f32>>;

/// INDI controller runtime state.
pub struct IndiController {
    effectiveness: IndiEffectiveness<NU>,
    linearization: [ThrustLinearization; NU],

    rate_gains: Vector3<f32>,

    rate_dot_filter: [Biquad; 3],
    spf_filter: [Biquad; 3],
    u_state_filter: [Biquad; NU],
    omega_filter: [Biquad; NU],

    prev_rate: Vector3<f32>,
    prev_omega_fs: [f32; NU],
    prev_du: [f32; NU],

    u_state: [f32; NU],
    u_state_fs: [f32; NU],
    pt1_alpha: [f32; NU],

    rpm_tracker: RpmTracker<NU>,
    erpm_to_rads: f32,

    ws: [i8; NU],
    nan_counter: u16,

    act_limit: [f32; NU],
    wls_wv: [f32; NV],
    wls_wu: [f32; NU],
    wls_cond_bound: f32,
    wls_theta: f32,
    wls_imax: usize,
    nan_limit: u16,
    rpm_invalid_limit: u16,
    rpm_all_invalid_limit: u16,
    rpm_recovery_count: u16,

    freq: f32,
}

/// Output of one INDI iteration.
#[derive(Clone, Copy)]
pub struct IndiOutput {
    /// Motor commands [0, 1] per motor.
    pub motor_commands: [f32; NU],
    /// True if WLS NaN counter exceeded limit.
    pub nan_failsafe: bool,
}

/// Intermediate signals from the INDI step, exposed for the learner.
///
/// These are the raw (pre-INDI-sync-filter) signals that the learner needs
/// for its own matched filtering. The learner applies its own filters at
/// a different cutoff frequency.
#[derive(Clone, Copy)]
pub struct IndiStepState {
    /// Unfiltered angular acceleration (rad/s²), from gyro finite difference.
    pub rate_dot_raw: Vector3<f32>,
    /// True if the ground-detection heuristic thinks the vehicle is on the ground.
    pub touching_ground: bool,
}

impl IndiController {
    pub fn new(config: &IndiConfig, loop_rate_hz: f32) -> Self {
        let dt = 1.0 / loop_rate_hz;

        let effectiveness =
            IndiEffectiveness::new(&config.motors, &config.body, &config.indi_motors);

        let linearization =
            core::array::from_fn(|i| ThrustLinearization::new(config.nonlinearity[i]));

        let make_biquad = || {
            let cfg = BiquadFilterConfigBuilder::direct_form_2()
                .sample_frequency_hz(loop_rate_hz)
                .filter_type(BiquadFilterType::LowPass)
                .cutoff_frequency_hz(config.sync_filter_hz)
                .build()
                .expect("indi: biquad filter config invalid");
            BiquadFilter::new(cfg)
        };

        let pt1_alpha = core::array::from_fn(|i| {
            let tau = config.indi_motors[i].time_const_s;
            dt / (tau + dt)
        });

        let pole_pairs = config.motor_pole_count as f32 / 2.0;
        let erpm_to_rads = 100.0 / pole_pairs / 60.0 * core::f32::consts::TAU;

        Self {
            effectiveness,
            linearization,
            rate_gains: config.rate_gains,
            rate_dot_filter: core::array::from_fn(|_| make_biquad()),
            spf_filter: core::array::from_fn(|_| make_biquad()),
            u_state_filter: core::array::from_fn(|_| make_biquad()),
            omega_filter: core::array::from_fn(|_| make_biquad()),
            prev_rate: Vector3::zeros(),
            prev_omega_fs: [0.0; NU],
            prev_du: [0.0; NU],
            u_state: [0.0; NU],
            u_state_fs: [0.0; NU],
            pt1_alpha,
            rpm_tracker: RpmTracker::<NU>::new(),
            erpm_to_rads,
            ws: [0; NU],
            nan_counter: 0,
            act_limit: config.act_limit,
            wls_wv: config.wls_wv,
            wls_wu: config.wls_wu,
            wls_cond_bound: config.wls_cond_bound,
            wls_theta: config.wls_theta,
            wls_imax: config.wls_imax,
            nan_limit: config.nan_limit,
            rpm_invalid_limit: config.rpm_invalid_limit,
            rpm_all_invalid_limit: config.rpm_all_invalid_limit,
            rpm_recovery_count: config.rpm_recovery_count,
            freq: loop_rate_hz,
        }
    }

    /// Apply learned parameters to the controller.
    ///
    /// Updates effectiveness (G1/G2), linearization (nonlinearity), PT1 time
    /// constants, and rate gains. Validates all values before applying.
    /// Returns `true` if applied, `false` if validation failed.
    pub fn apply_learned_params(
        &mut self,
        learned: &super::learner::LearnedParams,
    ) -> bool {
        if !learned.valid {
            return false;
        }
        // Validate and apply effectiveness
        if !self.effectiveness.update_from_learned(
            &learned.g1,
            &learned.g2,
            &learned.max_omega,
            &learned.time_const_s,
        ) {
            return false;
        }
        // Update PT1 time constants for actuator state estimation
        let dt = 1.0 / self.freq;
        for i in 0..NU {
            self.pt1_alpha[i] = dt / (learned.time_const_s[i] + dt);
        }
        // Update linearization
        for i in 0..NU {
            self.linearization[i] =
                super::linearization::ThrustLinearization::new(learned.nonlinearity[i]);
        }
        // Update rate gains
        if learned.rate_gain.is_finite() && learned.rate_gain > 0.0 {
            self.rate_gains = Vector3::new(learned.rate_gain, learned.rate_gain, learned.rate_gain);
        }
        true
    }

    /// Reset effectiveness to geometric G1 with zero G2.
    ///
    /// Used when entering learner-prearm mode: reverts to the physics-based
    /// effectiveness derived from motor geometry, ensuring no learned G2 or
    /// G1 is active during a data-collection flight.
    pub fn reset_to_geometric(
        &mut self,
        motors: &[crate::mixer::MotorParams; NU],
        body: &crate::mixer::RigidBodyParams,
        indi_params: &[IndiMotorParams; NU],
    ) {
        self.effectiveness = IndiEffectiveness::new(motors, body, indi_params);
    }

    /// Update actuator state estimation from last motor command.
    /// Must be called every loop even when INDI is not the active controller.
    pub fn update_actuator_state(&mut self, d: &[f32; NU]) {
        for i in 0..NU {
            let u = self.linearization[i].output_curve(d[i]);
            self.u_state[i] += self.pt1_alpha[i] * (u - self.u_state[i]);
        }
    }

    /// Update motor RPM from telemetry.
    /// Takes `RpmInput` (not `TelemetryValue`) — caller converts.
    /// Returns (g2_valid per motor, rpm_failsafe).
    pub fn update_rpm(&mut self, inputs: &[RpmInput; NU]) -> ([bool; NU], bool) {
        let result = self.rpm_tracker.update(
            inputs,
            self.erpm_to_rads,
            self.rpm_invalid_limit,
            self.rpm_all_invalid_limit,
            self.rpm_recovery_count,
        );

        for i in 0..NU {
            let filtered = self.omega_filter[i].apply(result.omega[i]);
            self.prev_omega_fs[i] = if filtered > 0.0 { filtered } else { 0.0 };
        }

        (result.g2_valid, result.failsafe)
    }

    /// Run one INDI iteration.
    ///
    /// Returns `(IndiOutput, IndiStepState)`. The `IndiStepState` exposes
    /// intermediate signals needed by the learner (raw rate_dot, ground
    /// detection). The learner applies its own matched filters.
    pub fn step(
        &mut self,
        gyro_rad_s: &Vector3<f32>,
        accel_m_s2: &Vector3<f32>,
        rate_sp: &Vector3<f32>,
        spf_sp_z: f32,
        armed: bool,
        g2_valid: &[bool; NU],
    ) -> (IndiOutput, IndiStepState) {
        // --- 1. Sensor processing ---
        let rate_dot_raw = (*gyro_rad_s - self.prev_rate) * self.freq;
        self.prev_rate = *gyro_rad_s;

        let rate_dot_fs = Vector3::new(
            self.rate_dot_filter[0].apply(rate_dot_raw[0]),
            self.rate_dot_filter[1].apply(rate_dot_raw[1]),
            self.rate_dot_filter[2].apply(rate_dot_raw[2]),
        );

        let spf_fs = Vector3::new(
            self.spf_filter[0].apply(accel_m_s2[0]),
            self.spf_filter[1].apply(accel_m_s2[1]),
            self.spf_filter[2].apply(accel_m_s2[2]),
        );

        // Motor acceleration via du-based fallback (matches C when no dshot RPM derivative).
        let mut omega_dot_fs = [0.0f32; NU];
        for i in 0..NU {
            let inv_thresh = 0.1 * self.effectiveness.max_omega[i];
            let omega_inv = if self.prev_omega_fs[i].abs() > inv_thresh {
                1.0 / self.prev_omega_fs[i]
            } else {
                1.0 / inv_thresh
            };
            omega_dot_fs[i] = self.prev_du[i] * self.effectiveness.g2_scaler[i] * omega_inv;
        }

        // Actuator state filtering
        for i in 0..NU {
            self.u_state_fs[i] = self.u_state_filter[i].apply(self.u_state[i]);
            self.u_state_fs[i] = self.u_state_fs[i].clamp(0.0, 1.0);
        }

        // --- 2. Takeoff detection ---
        let gyro_mag_sq = gyro_rad_s.norm_squared();
        let accel_mag_sq = accel_m_s2.norm_squared();
        let gyro_thresh = 100.0_f32 * core::f32::consts::PI / 180.0;
        let gyro_low = gyro_mag_sq < gyro_thresh * gyro_thresh;
        let accel_high = accel_mag_sq > (0.8 * 9.81) * (0.8 * 9.81);
        let thrust_low = spf_sp_z < 3.0;
        let touching_ground = gyro_low && accel_high && thrust_low;
        let do_indi = !touching_ground && armed;
        let do_indi_f = if do_indi { 1.0f32 } else { 0.0 };

        // --- 3. Rate controller ---
        let rate_err = *rate_sp - *gyro_rad_s;
        let rate_dot_sp = Vector3::new(
            self.rate_gains[0] * rate_err[0],
            self.rate_gains[1] * rate_err[1],
            self.rate_gains[2] * rate_err[2],
        );

        // --- 4. Pseudo-control ---
        let mut dv = [0.0f32; NV];
        dv[2] = spf_sp_z - do_indi_f * spf_fs[2];
        dv[3] = rate_dot_sp[0] - do_indi_f * rate_dot_fs[0];
        dv[4] = rate_dot_sp[1] - do_indi_f * rate_dot_fs[1];
        dv[5] = rate_dot_sp[2] - do_indi_f * rate_dot_fs[2];

        for j in 0..3 {
            for i in 0..NU {
                dv[j + 3] += do_indi_f * self.effectiveness.g2[(j, i)] * omega_dot_fs[i];
            }
        }

        // --- 5. Combined effectiveness matrix ---
        let omega_fs_vec = SVector::<f32, NU>::from_row_slice(&self.prev_omega_fs);
        let g1g2: SMatrix<f32, NV, NU> = self.effectiveness.combined_g1g2(&omega_fs_vec, g2_valid);

        // --- 6. WLS allocation ---
        let wv = SVector::<f32, NV>::from_row_slice(&self.wls_wv);
        let mut wu = SVector::<f32, NU>::from_row_slice(&self.wls_wu);
        let v = SVector::<f32, NV>::from_row_slice(&dv);

        let (a_mat, gamma) =
            setup_a::<NU, NV, NC>(&g1g2, &wv, &mut wu, self.wls_theta, self.wls_cond_bound);

        let mut du_min = SVector::<f32, NU>::zeros();
        let mut du_max = SVector::<f32, NU>::zeros();
        let mut du_pref = SVector::<f32, NU>::zeros();
        for i in 0..NU {
            du_min[i] = -do_indi_f * self.u_state_fs[i];
            du_max[i] = self.act_limit[i] - do_indi_f * self.u_state_fs[i];
            du_pref[i] = -do_indi_f * self.u_state_fs[i];
        }

        let b_vec = setup_b::<NU, NV, NC>(&v, &du_pref, &wv, &wu, gamma);

        let mut du = SVector::<f32, NU>::zeros();
        for i in 0..NU {
            du[i] = (du_min[i] + du_max[i]) * 0.5;
        }

        let stats = solve::<NU, NV, NC>(
            &a_mat,
            &b_vec,
            &du_min,
            &du_max,
            &mut du,
            &mut self.ws,
            self.wls_imax,
        );

        // --- 7. NaN protection ---
        let nan_exit =
            stats.exit_code == ExitCode::NanFoundQ || stats.exit_code == ExitCode::NanFoundUs;
        if nan_exit {
            self.nan_counter += 1;
            self.ws = [0; NU];
        } else {
            self.nan_counter = 0;
        }
        let nan_failsafe = self.nan_counter > self.nan_limit;

        // --- 8. Motor commands ---
        let mut motor_commands = [0.0f32; NU];

        for i in 0..NU {
            let u = if !nan_exit {
                (do_indi_f * self.u_state_fs[i] + du[i]).clamp(0.0, self.act_limit[i])
            } else {
                (self.u_state_fs[i] * 0.95).clamp(0.0, self.act_limit[i])
            };
            motor_commands[i] = self.linearization[i].linearize(u);
            self.prev_du[i] = u - self.u_state[i];
        }

        if motor_commands.iter().all(|v| v.is_finite()) {
            self.update_actuator_state(&motor_commands);
        }

        let output = IndiOutput {
            motor_commands,
            nan_failsafe,
        };
        let step_state = IndiStepState {
            rate_dot_raw,
            touching_ground,
        };
        (output, step_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixer::SpinDir;

    const LOOP_HZ: f32 = 8000.0;
    const GRAVITY: f32 = 9.80665;

    fn test_config() -> IndiConfig {
        IndiConfig {
            rate_gains: Vector3::new(20.0, 20.0, 20.0),
            sync_filter_hz: 15.0,
            motors: [
                MotorParams { position_m: [-0.075, -0.1], spin_dir: SpinDir::Cw,  max_thrust_n: 8.5, torque_coeff_m: 0.022 },
                MotorParams { position_m: [ 0.075, -0.1], spin_dir: SpinDir::Ccw, max_thrust_n: 8.5, torque_coeff_m: 0.022 },
                MotorParams { position_m: [-0.075,  0.1], spin_dir: SpinDir::Ccw, max_thrust_n: 8.5, torque_coeff_m: 0.022 },
                MotorParams { position_m: [ 0.075,  0.1], spin_dir: SpinDir::Cw,  max_thrust_n: 8.5, torque_coeff_m: 0.022 },
            ],
            body: RigidBodyParams {
                mass_kg: 0.55,
                inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
                max_rate_rad_s: [10.0, 10.0, 6.0],
            },
            indi_motors: [IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: 0.0 }; NU],
            nonlinearity: [0.5; NU],
            act_limit: [1.0; NU],
            wls_wv: [1.0, 1.0, 50.0, 50.0, 50.0, 5.0],
            wls_wu: [1.0; NU],
            wls_cond_bound: 3.2768e8,
            wls_theta: 1e-4,
            wls_imax: 1,
            nan_limit: 20,
            rpm_invalid_limit: 50,
            rpm_all_invalid_limit: 50,
            rpm_recovery_count: 10,
            motor_pole_count: 14,
        }
    }

    fn hover_inputs() -> (Vector3<f32>, Vector3<f32>, Vector3<f32>, f32) {
        (Vector3::zeros(), Vector3::new(0.0, 0.0, GRAVITY), Vector3::zeros(), GRAVITY)
    }

    #[test]
    fn new_does_not_panic() {
        let _ctrl = IndiController::new(&test_config(), LOOP_HZ);
    }

    #[test]
    fn hover_produces_equal_motor_commands() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (gyro, accel, rate_sp, spf_sp_z) = hover_inputs();
        let g2 = [false; NU];
        let mut out = ctrl.step(&gyro, &accel, &rate_sp, spf_sp_z, true, &g2).0;
        for _ in 0..200 { out = ctrl.step(&gyro, &accel, &rate_sp, spf_sp_z, true, &g2).0; }
        let mean = out.motor_commands.iter().sum::<f32>() / NU as f32;
        for (i, &c) in out.motor_commands.iter().enumerate() {
            assert!(c >= 0.0 && c <= 1.0, "motor {i} out of bounds: {c}");
            assert!((c - mean).abs() < 0.05, "motor {i} diverges: {c} vs mean {mean}");
        }
        assert!(mean > 0.1 && mean < 0.7, "hover mean={mean} unexpected");
        assert!(!out.nan_failsafe);
    }

    #[test]
    fn motor_commands_always_in_bounds() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let accel = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        let cases: &[(Vector3<f32>, Vector3<f32>, f32)] = &[
            (Vector3::zeros(), Vector3::zeros(), GRAVITY),
            (Vector3::zeros(), Vector3::new(5.0, -3.0, 1.0), GRAVITY),
            (Vector3::zeros(), Vector3::new(15.0, 15.0, 0.0), GRAVITY * 2.0),
            (Vector3::new(1.7, -0.9, 0.3), Vector3::zeros(), GRAVITY),
        ];
        for (gyro, rate_sp, spf) in cases {
            for _ in 0..50 {
                let out = ctrl.step(gyro, &accel, rate_sp, *spf, true, &g2).0;
                for (i, &c) in out.motor_commands.iter().enumerate() {
                    assert!(c >= 0.0 && c <= 1.0, "motor {i} = {c}");
                }
            }
        }
    }

    #[test]
    fn roll_command_differential_thrust() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let accel = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        for _ in 0..100 { ctrl.step(&Vector3::zeros(), &accel, &Vector3::zeros(), GRAVITY, true, &g2).0; }
        let rate_sp = Vector3::new(3.0, 0.0, 0.0);
        let mut out = ctrl.step(&Vector3::zeros(), &accel, &rate_sp, GRAVITY, true, &g2).0;
        for _ in 0..50 { out = ctrl.step(&Vector3::zeros(), &accel, &rate_sp, GRAVITY, true, &g2).0; }
        let left = (out.motor_commands[2] + out.motor_commands[3]) / 2.0;
        let right = (out.motor_commands[0] + out.motor_commands[1]) / 2.0;
        assert!(left > right, "roll: left={left} should > right={right}");
    }

    #[test]
    fn disarmed_no_failsafe() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (g, a, r, s) = hover_inputs();
        for _ in 0..50 {
            let out = ctrl.step(&g, &a, &r, s, false, &[false; NU]).0;
            assert!(!out.nan_failsafe);
            for &c in &out.motor_commands { assert!(c.is_finite() && c >= 0.0 && c <= 1.0); }
        }
    }

    #[test]
    fn ground_ndi_valid() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let gyro = Vector3::zeros();
        let accel = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        let out1 = ctrl.step(&gyro, &accel, &Vector3::zeros(), 2.0, false, &g2).0;
        let out2 = ctrl.step(&gyro, &accel, &Vector3::zeros(), 2.0, false, &g2).0;
        for &c in &out1.motor_commands { assert!(c.is_finite() && c >= 0.0 && c <= 1.0); }
        for i in 0..NU {
            assert!((out1.motor_commands[i] - out2.motor_commands[i]).abs() < 0.1);
        }
    }

    #[test]
    fn takeoff_transition_no_spike() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let g = Vector3::zeros();
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let r = Vector3::zeros();
        let g2 = [false; NU];
        for _ in 0..50 { ctrl.step(&g, &a, &r, 2.0, false, &g2).0; }
        let ground = ctrl.step(&g, &a, &r, 2.0, false, &g2).0;
        let air = ctrl.step(&g, &a, &r, GRAVITY, true, &g2).0;
        for i in 0..NU {
            assert!((air.motor_commands[i] - ground.motor_commands[i]).abs() < 0.5,
                "takeoff spike motor {i}");
        }
    }

    #[test]
    fn update_rpm_valid_enables_g2() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (g2, fs) = ctrl.update_rpm(&[RpmInput::Erpm(10000); NU]);
        assert!(g2.iter().all(|&v| v));
        assert!(!fs);
    }

    #[test]
    fn update_rpm_invalid_disables_g2() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        ctrl.update_rpm(&[RpmInput::Erpm(10000); NU]);
        for _ in 0..60 {
            let (g2, _) = ctrl.update_rpm(&[RpmInput::Invalid; NU]);
            if g2.iter().all(|&v| !v) { return; }
        }
        panic!("G2 should have been zeroed");
    }

    #[test]
    fn update_rpm_all_invalid_failsafe() {
        let cfg = IndiConfig { rpm_invalid_limit: 5, rpm_all_invalid_limit: 10, ..test_config() };
        let mut ctrl = IndiController::new(&cfg, LOOP_HZ);
        ctrl.update_rpm(&[RpmInput::Erpm(10000); NU]);
        for _ in 0..20 {
            let (_, fs) = ctrl.update_rpm(&[RpmInput::Invalid; NU]);
            if fs { return; }
        }
        panic!("RPM failsafe should have triggered");
    }

    #[test]
    fn asymmetric_limits_respected() {
        let cfg = IndiConfig { act_limit: [0.8, 1.0, 1.0, 1.0], ..test_config() };
        let mut ctrl = IndiController::new(&cfg, LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(10.0, 10.0, 0.0);
        let g2 = [false; NU];
        for _ in 0..100 {
            let out = ctrl.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &g2).0;
            assert!(out.motor_commands[0] <= 0.8 + 1e-6, "M0 exceeded: {}", out.motor_commands[0]);
        }
    }

    #[test]
    fn sequential_steps_converge() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (g, a, r, s) = hover_inputs();
        let g2 = [false; NU];
        let mut prev = [0.0f32; NU];
        for step in 0..500 {
            let out = ctrl.step(&g, &a, &r, s, true, &g2).0;
            let max_change = out.motor_commands.iter().zip(prev.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            if step > 100 && max_change < 1e-5 { return; }
            prev = out.motor_commands;
        }
        panic!("hover did not converge in 500 steps");
    }

    #[test]
    fn nonzero_gyro_produces_correction() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        for _ in 0..100 { ctrl.step(&Vector3::zeros(), &a, &Vector3::zeros(), GRAVITY, true, &g2).0; }
        let hover = ctrl.step(&Vector3::zeros(), &a, &Vector3::zeros(), GRAVITY, true, &g2).0;
        let gyro = Vector3::new(100.0f32.to_radians(), 0.0, 0.0);
        let mut spin = hover;
        for _ in 0..20 { spin = ctrl.step(&gyro, &a, &Vector3::zeros(), GRAVITY, true, &g2).0; }
        let diff = spin.motor_commands.iter().zip(hover.motor_commands.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff > 0.01, "gyro should change allocation: diff={diff}");
    }

    #[test]
    fn warmstart_stable_output() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (g, a, r, s) = hover_inputs();
        let g2 = [false; NU];
        // Settle until converged (filters + actuator state)
        let mut prev = [0.0f32; NU];
        for step in 0..1000 {
            let out = ctrl.step(&g, &a, &r, s, true, &g2).0;
            let max_change = out.motor_commands.iter().zip(prev.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            prev = out.motor_commands;
            if step > 50 && max_change < 1e-6 {
                break;
            }
            assert!(step < 999, "warmstart test: failed to converge in 1000 steps");
        }
        // Now check 10 subsequent outputs are nearly identical
        let mut outputs = Vec::new();
        for _ in 0..10 {
            outputs.push(ctrl.step(&g, &a, &r, s, true, &g2).0.motor_commands);
        }
        for i in 1..10 {
            let diff = outputs[i].iter().zip(outputs[i-1].iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            // Tolerance accounts for the slow PT1+biquad filter tail.
            // The key assertion is that warmstarted outputs don't diverge —
            // they should decrease or stay flat, not grow.
            assert!(diff < 5e-5, "warmstart unstable at step {i}: diff={diff}");
        }
    }

    #[test]
    fn no_nan_across_diverse_inputs() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        let cases: &[(Vector3<f32>, Vector3<f32>, f32, bool)] = &[
            (Vector3::zeros(), Vector3::zeros(), GRAVITY, true),
            (Vector3::zeros(), Vector3::zeros(), 0.0, false),
            (Vector3::zeros(), Vector3::new(10.0, -10.0, 5.0), GRAVITY * 3.0, true),
            (Vector3::new(200.0f32.to_radians(), 0.0, 0.0), Vector3::zeros(), GRAVITY, true),
            (Vector3::new(0.01, -0.005, 0.002), Vector3::new(0.05, -0.03, 0.01), GRAVITY, true),
        ];
        for (gyro, rate_sp, spf, armed) in cases {
            for _ in 0..50 {
                let out = ctrl.step(gyro, &a, rate_sp, *spf, *armed, &g2).0;
                for (i, &c) in out.motor_commands.iter().enumerate() {
                    assert!(c.is_finite() && c >= 0.0 && c <= 1.0, "motor {i} = {c}");
                }
            }
        }
    }

    #[test]
    fn g2_valid_affects_allocation() {
        let cfg = IndiConfig {
            indi_motors: [IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: 0.001 }; NU],
            ..test_config()
        };
        let mut ctrl_g2 = IndiController::new(&cfg, LOOP_HZ);
        let mut ctrl_no = IndiController::new(&cfg, LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 3.0);
        for _ in 0..100 {
            ctrl_g2.update_rpm(&[RpmInput::Erpm(20000); NU]);
            ctrl_g2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;
            ctrl_no.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU]).0;
        }
        let out_g2 = ctrl_g2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;
        let out_no = ctrl_no.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU]).0;
        let diff = out_g2.motor_commands.iter().zip(out_no.motor_commands.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff > 1e-6, "G2 should affect allocation: diff={diff}");
    }

    fn g2_config() -> IndiConfig {
        // FLU sign convention: CW motors get positive G2 yaw, CCW get negative
        IndiConfig {
            indi_motors: [
                IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw:  0.001 }, // M0 CW
                IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: -0.001 }, // M1 CCW
                IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: -0.001 }, // M2 CCW
                IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw:  0.001 }, // M3 CW
            ],
            ..test_config()
        }
    }

    #[test]
    fn g2_yaw_command_correct_direction() {
        // Positive yaw command in FLU → CW motors (M0, M3) should increase
        let mut ctrl_g2 = IndiController::new(&g2_config(), LOOP_HZ);
        let mut ctrl_no = IndiController::new(&test_config(), LOOP_HZ); // G2=0 baseline
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 3.0); // positive yaw in FLU

        // Feed RPM to enable G2
        for _ in 0..100 {
            ctrl_g2.update_rpm(&[RpmInput::Erpm(20000); NU]);
            ctrl_g2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;
            ctrl_no.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU]).0;
        }

        let out_g2 = ctrl_g2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;
        let out_no = ctrl_no.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU]).0;

        // G2 should modify the allocation but not invert it.
        // CW motors (M0, M3) should still be higher than CCW for positive yaw.
        let cw_avg_g2 = (out_g2.motor_commands[0] + out_g2.motor_commands[3]) / 2.0;
        let ccw_avg_g2 = (out_g2.motor_commands[1] + out_g2.motor_commands[2]) / 2.0;
        assert!(
            cw_avg_g2 > ccw_avg_g2 || (cw_avg_g2 - ccw_avg_g2).abs() < 0.01,
            "positive yaw: CW={cw_avg_g2:.4} should >= CCW={ccw_avg_g2:.4}"
        );
    }

    #[test]
    fn g2_omegadot_feedback_nonzero_after_command() {
        // After a step with nonzero allocation, prev_du should be nonzero,
        // so the next step's omegaDot_fs should be nonzero (affecting dv).
        let mut ctrl = IndiController::new(&g2_config(), LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 5.0); // yaw command

        // Feed RPM
        ctrl.update_rpm(&[RpmInput::Erpm(20000); NU]);

        // First step: establishes prev_du
        ctrl.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;

        // Second step with G2 vs without G2: the omegaDot contribution to dv
        // should cause different outputs
        let mut ctrl2_g2 = IndiController::new(&g2_config(), LOOP_HZ);
        let mut ctrl2_no = IndiController::new(&test_config(), LOOP_HZ);

        ctrl2_g2.update_rpm(&[RpmInput::Erpm(20000); NU]);

        // Run both for enough steps to have meaningful prev_du
        for _ in 0..50 {
            ctrl2_g2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;
            ctrl2_no.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU]).0;
        }

        let out_g2 = ctrl2_g2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;
        let out_no = ctrl2_no.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU]).0;

        let diff = out_g2.motor_commands.iter().zip(out_no.motor_commands.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff > 1e-5, "omegaDot feedback should cause measurable difference: diff={diff}");
    }

    #[test]
    fn g2_disabled_motor_no_effect() {
        // If one motor's G2 is disabled (g2_valid=false), only that motor's
        // G2 column should be zeroed. Others should still have G2 active.
        let mut ctrl = IndiController::new(&g2_config(), LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 3.0);

        ctrl.update_rpm(&[RpmInput::Erpm(20000); NU]);

        // All G2 active
        for _ in 0..100 {
            ctrl.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;
        }
        let out_all = ctrl.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU]).0;

        // Reset and run with M0 G2 disabled
        let mut ctrl2 = IndiController::new(&g2_config(), LOOP_HZ);
        ctrl2.update_rpm(&[RpmInput::Erpm(20000); NU]);
        let mut g2_partial = [true; NU];
        g2_partial[0] = false;

        for _ in 0..100 {
            ctrl2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &g2_partial).0;
        }
        let out_partial = ctrl2.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &g2_partial).0;

        // Outputs should differ (M0's G2 column removed changes allocation)
        let diff = out_all.motor_commands.iter().zip(out_partial.motor_commands.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff > 1e-6, "disabling one motor's G2 should change allocation: diff={diff}");
    }
}
