// INDI (Incremental Nonlinear Dynamic Inversion) controller.
//
// Runs at IMU rate (~8 kHz). Subscribes to raw IMU data and DShot telemetry.
// Replaces the rate PID + linear allocator pipeline with:
//   rate error → angular accel demand → pseudo-control → WLS allocation → motor commands
//
// Hardcoded to quad (NU=4, NV=6, NC=10) to avoid nightly generic_const_exprs.
// Ported from indiflight: src/main/flight/indi.c

use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use air_filters::Filter;
use cybflight_core::indi::{
    effectiveness::{IndiEffectiveness, IndiMotorParams},
    linearization::ThrustLinearization,
    rpm_tracker::RpmTracker,
};
use cybflight_core::mixer::{MotorParams, RigidBodyParams};
use nalgebra::{SMatrix, SVector, Vector3};
use wls_alloc::{setup_a, setup_b, solve, ExitCode};

use crate::msgs::dshot::TelemetryValue;

/// Number of actuators (motors).
const NU: usize = 4;
/// Number of pseudo-controls (fx, fy, fz, roll, pitch, yaw).
const NV: usize = 6;
/// Number of constraint rows (NU + NV).
const NC: usize = NU + NV;

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

// RPM tracker: uses cybflight_core::indi::rpm_tracker

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
pub struct IndiOutput {
    /// Motor commands [0, 1] per motor.
    pub motor_commands: [f32; NU],
    /// True if WLS NaN counter exceeded limit.
    pub nan_failsafe: bool,
}

impl IndiController {
    pub fn new(config: &IndiConfig, loop_rate_hz: f32) -> Self {
        let dt = 1.0 / loop_rate_hz;

        let effectiveness =
            IndiEffectiveness::new(&config.motors, &config.body, &config.indi_motors);

        let linearization = core::array::from_fn(|i| {
            ThrustLinearization::new(config.nonlinearity[i])
        });

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

    /// Update actuator state estimation from last motor command.
    /// Must be called every loop even when INDI is not the active controller.
    pub fn update_actuator_state(&mut self, d: &[f32; NU]) {
        for i in 0..NU {
            let u = self.linearization[i].output_curve(d[i]);
            self.u_state[i] += self.pt1_alpha[i] * (u - self.u_state[i]);
        }
    }

    /// Update motor RPM from DShot telemetry.
    /// Returns (g2_valid per motor, rpm_failsafe).
    pub fn update_rpm(&mut self, telem: &[TelemetryValue; NU]) -> ([bool; NU], bool) {
        use cybflight_core::indi::rpm_tracker::RpmInput;
        let inputs: [RpmInput; NU] = core::array::from_fn(|i| match telem[i] {
            TelemetryValue::Erpm(erpm) => RpmInput::Erpm(erpm),
            TelemetryValue::Stopped => RpmInput::Stopped,
            TelemetryValue::Invalid | TelemetryValue::Edt(_) => RpmInput::Invalid,
        });
        let result = self.rpm_tracker.update(
            &inputs,
            self.erpm_to_rads,
            self.rpm_invalid_limit,
            self.rpm_all_invalid_limit,
            self.rpm_recovery_count,
        );
        let omega_raw = result.omega;
        let g2_valid = result.g2_valid;
        let failsafe = result.failsafe;

        for i in 0..NU {
            let filtered = self.omega_filter[i].apply(omega_raw[i]);
            self.prev_omega_fs[i] = if filtered > 0.0 { filtered } else { 0.0 };
        }

        (g2_valid, failsafe)
    }

    /// Run one INDI iteration.
    pub fn step(
        &mut self,
        gyro_rad_s: &Vector3<f32>,
        accel_m_s2: &Vector3<f32>,
        rate_sp: &Vector3<f32>,
        spf_sp_z: f32,
        armed: bool,
        g2_valid: &[bool; NU],
    ) -> IndiOutput {
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
        // omegaDot_fs = du * G2_scaler * omega_inv
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
        // 100 deg/s threshold squared (in rad/s)
        let gyro_thresh_sq = 100.0_f32 * core::f32::consts::PI / 180.0;
        let gyro_thresh_sq = gyro_thresh_sq * gyro_thresh_sq;
        let gyro_low = gyro_mag_sq < gyro_thresh_sq;
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

        // setup_a::<NU=4, NV=6, NC=10>
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
            &a_mat, &b_vec, &du_min, &du_max, &mut du, &mut self.ws, self.wls_imax,
        );

        // --- 7. NaN protection ---
        let nan_exit = stats.exit_code == ExitCode::NanFoundQ
            || stats.exit_code == ExitCode::NanFoundUs;
        if nan_exit {
            self.nan_counter += 1;
            // Reset WLS warmstart to prevent corrupted working set from
            // preventing recovery. Matches indiflight indi.c:471.
            self.ws = [0; NU];
        } else {
            // Reset on any successful solve (whether armed or not).
            // Prevents counter accumulating during NDI ground phase.
            self.nan_counter = 0;
        }
        let nan_failsafe = self.nan_counter > self.nan_limit;

        // --- 8. Motor commands ---
        let mut motor_commands = [0.0f32; NU];

        for i in 0..NU {
            let u = if !nan_exit {
                (do_indi_f * self.u_state_fs[i] + du[i]).clamp(0.0, self.act_limit[i])
            } else {
                // On NaN: ramp toward idle (0) rather than holding stale command.
                // Blend 95% previous + 5% zero each frame — reaches near-zero
                // in ~60 frames (7.5 ms at 8 kHz), well within nan_limit.
                (self.u_state_fs[i] * 0.95).clamp(0.0, self.act_limit[i])
            };
            motor_commands[i] = self.linearization[i].linearize(u);
            // Track du for omegaDot fallback (u - uState = actual du)
            self.prev_du[i] = u - self.u_state[i];
        }

        // Update actuator state only with finite commands (Issue 3: prevent
        // corrupting u_state with non-finite values from defense-in-depth path).
        if motor_commands.iter().all(|v| v.is_finite()) {
            self.update_actuator_state(&motor_commands);
        }

        IndiOutput { motor_commands, nan_failsafe }
    }
}
