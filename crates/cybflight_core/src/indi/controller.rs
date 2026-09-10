// INDI (Incremental Nonlinear Dynamic Inversion) controller.
//
// Pure computation — no async, no channels, no embassy types.
// Hardcoded to quad (NU=4, NV=6, NC=10) to avoid nightly generic_const_exprs.
// Ported from indiflight: src/main/flight/indi.c

use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use air_filters::Filter;
use flight_solver::cls::setup::wls::{setup_a, setup_b};
use flight_solver::cls::{solve, ExitCode};
use nalgebra::{stack, SMatrix, SVector, Vector3};

use super::{
    effectiveness::{IndiEffectiveness, IndiMotorParams},
    linearization::{ThrustLinearization, ThrustModel},
    rate_dot_estimator::{RateDotEstimator, RateDotEstimatorConfig},
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
    /// Master INDI switch. `true` (the normal build) runs the full
    /// incremental law. `false` permanently drops the incremental terms —
    /// the same `do_indi = false` path the controller already takes on the
    /// ground — leaving a plain rate controller: rate error × `rate_gains`
    /// → desired angular acceleration → WLS allocation through G1 →
    /// thrust linearization. This is indiflight's `useIncrement = false`
    /// (NDI, or more precisely linDI); see `src/main/flight/indi.h`.
    ///
    /// Selected by the vehicle YAML `build: indi:` knob (cargo feature
    /// `indi_off`), so on a normal build the `false` branch folds away.
    /// Note the degraded law is proportional-only: without the increment
    /// there is no integral action, so a steady disturbance (mass
    /// imbalance, wind) leaves a standing rate error the outer loop has
    /// to absorb.
    pub indi_enabled: bool,
    /// Rate error → angular acceleration gains (rad/s² per rad/s).
    pub rate_gains: Vector3<f32>,
    /// Biquad low-pass cutoff frequency (Hz) for `spf`, `u_state`, `omega`
    /// sync filters, and for the `rate_dot` post-biquad that follows the
    /// Savitzky–Golay derivative filter.
    pub sync_filter_hz: f32,
    /// Savitzky–Golay window size (odd, ≥3, ≤19). Used for the `rate_dot`
    /// first-derivative filter.
    pub rate_dot_sg_window_size: i32,
    /// Savitzky–Golay polynomial order (1 ≤ n ≤ 3, n < window_size).
    pub rate_dot_sg_order: i32,
    /// Motor parameters (from vehicle definition).
    pub motors: [MotorParams; NU],
    /// Body rigid-body parameters.
    pub body: RigidBodyParams,
    /// Per-motor INDI parameters (time constant, max RPM, G2 yaw).
    pub indi_motors: [IndiMotorParams; NU],
    /// Thrust-to-command model (shared across all motors).
    pub thrust_model: ThrustModel,
    /// Motor nonlinearity for thrust linearization (0.0–1.0).
    pub nonlinearity: SVector<f32, NU>,
    /// Motor output limit per motor (0.0–1.0, typically 1.0).
    pub act_limit: SVector<f32, NU>,
    /// WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw].
    pub wls_wv: SVector<f32, NV>,
    /// WLS actuator penalty weights.
    pub wls_wu: SVector<f32, NU>,
    /// WLS condition number bound.
    pub wls_cond_bound: f32,
    /// WLS objective separation parameter.
    pub wls_theta: f32,
    /// WLS max iterations per loop (1 with warmstarting).
    pub wls_imax: usize,
    /// Consecutive WLS NaN failures before failsafe.
    pub nan_limit: u16,
    /// Per-tick multiplier applied to the held actuator state while the
    /// WLS allocator is returning NaN.
    ///
    /// Ramps the motors down instead of holding the last good command
    /// indefinitely or cutting them dead. Its effective time constant is
    /// per *control* tick, so it is coupled to `indi_ctrl_div` and the
    /// loop rate: the same fraction decays faster at a higher rate.
    pub nan_rampdown: f32,
    /// Consecutive invalid RPM frames before zeroing G2 column (per motor).
    pub rpm_invalid_limit: u16,
    /// Consecutive frames with ALL motors invalid before failsafe.
    pub rpm_all_invalid_limit: u16,
    /// Consecutive valid frames required to re-enable G2 after it was zeroed.
    pub rpm_recovery_count: u16,
    /// Motor pole count (for eRPM → RPM conversion).
    pub motor_pole_count: u8,
    /// Ground-contact detection: gyro magnitude below this counts as
    /// "not flying" [rad/s].
    pub ground_gyro_rad_s: f32,
    /// Ground-contact detection: specific-force magnitude above this
    /// counts as "resting on its skids" [m/s²]. Normally slightly below
    /// 1 g — a vehicle in free flight reads less.
    pub ground_accel_m_s2: f32,
    /// Ground-contact detection: vertical thrust setpoint below this
    /// counts as "not commanding flight" [m/s²].
    pub ground_thrust_sp_m_s2: f32,
}

// ---------------------------------------------------------------------------
// INDI controller
// ---------------------------------------------------------------------------

type Biquad = BiquadFilter<f32, DirectForm2<f32>>;

/// INDI controller runtime state.
pub struct IndiController {
    effectiveness: IndiEffectiveness<NU>,
    linearization: [ThrustLinearization; NU],
    thrust_model: ThrustModel,

    indi_enabled: bool,
    rate_gains: Vector3<f32>,
    /// Effective (Nyquist-clamped) cutoff shared by every sync filter.
    sync_filter_hz: f32,

    /// Ground-contact detection thresholds (gyro [rad/s], specific
    /// force [m/s²], vertical thrust setpoint [m/s²]).
    ground_gyro_rad_s: f32,
    ground_accel_m_s2: f32,
    ground_thrust_sp_m_s2: f32,

    rate_dot_estimator: RateDotEstimator,
    spf_filter: [Biquad; 3],
    u_state_filter: [Biquad; NU],
    omega_filter: [Biquad; NU],

    prev_omega_fs: SVector<f32, NU>,
    prev_du: SVector<f32, NU>,

    u_state: SVector<f32, NU>,
    u_state_fs: SVector<f32, NU>,
    pt1_alpha: SVector<f32, NU>,

    rpm_tracker: RpmTracker<NU>,
    erpm_to_rads: f32,

    ws: [i8; NU],
    nan_counter: u16,

    act_limit: SVector<f32, NU>,
    wls_wv: SVector<f32, NV>,
    wls_wu: SVector<f32, NU>,
    wls_cond_bound: f32,
    wls_theta: f32,
    wls_imax: usize,
    nan_limit: u16,
    nan_rampdown: f32,
    rpm_invalid_limit: u16,
    rpm_all_invalid_limit: u16,
    rpm_recovery_count: u16,

    freq: f32,
}

/// Source of filtered motor speed and acceleration used inside `step`.
///
/// `Internal`: ω_fs comes from `update_rpm` (internal biquad on the
///   `RpmTracker` output); ω̇_fs comes from the du-based fallback
///   `prev_du · g2_scaler · omega_inv`.
/// `External`: caller supplies pre-filtered ω, ω̇ (e.g. from a dshot LPF +
///   finite difference in the task layer). The internal `update_rpm`
///   path is bypassed for this step's `dv` and `combined_g1g2` inputs.
///
/// Safety: if any element of an `External` payload is non-finite, `step`
/// silently falls back to the `Internal` computation for that call. Don't
/// rely on this as your only line of defense — the caller should still
/// reject obviously bad inputs upstream.
#[derive(Clone, Copy)]
pub enum MotorState<'a> {
    Internal,
    External {
        omega_fs: &'a SVector<f32, NU>,
        omega_dot_fs: &'a SVector<f32, NU>,
    },
}

/// Output of one INDI iteration.
#[derive(Clone, Copy)]
pub struct IndiOutput {
    /// Motor commands [0, 1] per motor.
    pub motor_commands: SVector<f32, NU>,
    /// True if WLS NaN counter exceeded limit.
    pub nan_failsafe: bool,
}

/// Intermediate signals from the INDI step, exposed for external
/// observers (raw, pre-INDI-sync-filter).
#[derive(Clone, Copy)]
pub struct IndiStepState {
    /// SG first-derivative output (rad/s²), pre-post-biquad.
    pub rate_dot_raw: Vector3<f32>,
    /// True if the ground-detection heuristic thinks the vehicle is on the ground.
    pub touching_ground: bool,
}

/// How the G1 block of [`crate::params::IndiEffectivenessParams`] resolved
/// in [`IndiController::apply_effectiveness_params`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum G1Application {
    /// G1 block all-zero (the "not configured" sentinel): the
    /// geometry-derived G1 was (re)stored.
    Geometric,
    /// Configured G1 validated and applied.
    Configured,
    /// Configured G1 present but invalid: geometric restored. Degraded but
    /// flyable — zero control authority is structurally impossible.
    RejectedKeptGeometric,
}

/// Per-call outcome of [`IndiController::apply_effectiveness_params`]. The
/// firmware maps these flags to `defmt` warnings; core stays log-free.
#[derive(Clone, Copy, Debug)]
pub struct EffectivenessApplyReport {
    /// How the G1 block resolved.
    pub g1: G1Application,
    /// False if any G2 entry was non-finite or beyond the magnitude bound
    /// (previous G2 kept).
    pub g2_ok: bool,
    /// Per-motor: false if tau/omega was non-finite or ≤ 0 (previous value
    /// kept for that motor).
    pub motor_dynamics_ok: [bool; NU],
}

impl IndiController {
    pub fn new(config: &IndiConfig, loop_rate_hz: f32) -> Self {
        let dt = 1.0 / loop_rate_hz;

        let effectiveness =
            IndiEffectiveness::new(&config.motors, &config.body, &config.indi_motors);

        let linearization = core::array::from_fn(|i| {
            ThrustLinearization::new(
                config.nonlinearity[i],
                config.thrust_model,
                config.motors[i].max_thrust_n,
            )
        });

        // One clamped cutoff for every synchronized filter. The whole point
        // of `sync_filter_hz` is that `spf_fs`, `u_state_fs`, `omega_fs` and
        // the `rate_dot` post-biquad share a group delay, so they must share
        // a cutoff — including after the Nyquist clamp, which only binds on
        // low loop rates (see `super::clamp_cutoff_hz`).
        let sync_filter_hz = super::clamp_cutoff_hz(config.sync_filter_hz, loop_rate_hz);

        let make_biquad = || {
            let cfg = BiquadFilterConfigBuilder::direct_form_2()
                .sample_frequency_hz(loop_rate_hz)
                .filter_type(BiquadFilterType::LowPass)
                .cutoff_frequency_hz(sync_filter_hz)
                .build()
                .expect("indi: biquad filter config invalid");
            BiquadFilter::new(cfg)
        };

        let pt1_alpha = SVector::<f32, 4>::from_fn(|i, _| {
            let tau = config.indi_motors[i].time_const_s;
            dt / (tau + dt)
        });

        let rate_dot_estimator = RateDotEstimator::new(
            loop_rate_hz,
            &RateDotEstimatorConfig {
                sg_window_size: config.rate_dot_sg_window_size,
                sg_order: config.rate_dot_sg_order,
                post_cutoff_hz: sync_filter_hz,
            },
        );

        let pole_pairs = config.motor_pole_count as f32 / 2.0;
        let erpm_to_rads = 100.0 / pole_pairs / 60.0 * core::f32::consts::TAU;

        Self {
            effectiveness,
            linearization,
            thrust_model: config.thrust_model,
            indi_enabled: config.indi_enabled,
            rate_gains: config.rate_gains,
            sync_filter_hz,
            ground_gyro_rad_s: config.ground_gyro_rad_s,
            ground_accel_m_s2: config.ground_accel_m_s2,
            ground_thrust_sp_m_s2: config.ground_thrust_sp_m_s2,
            rate_dot_estimator,
            spf_filter: core::array::from_fn(|_| make_biquad()),
            u_state_filter: core::array::from_fn(|_| make_biquad()),
            omega_filter: core::array::from_fn(|_| make_biquad()),
            prev_omega_fs: SVector::zeros(),
            prev_du: SVector::zeros(),
            u_state: SVector::zeros(),
            u_state_fs: SVector::zeros(),
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
            nan_rampdown: config.nan_rampdown,
            rpm_invalid_limit: config.rpm_invalid_limit,
            rpm_all_invalid_limit: config.rpm_all_invalid_limit,
            rpm_recovery_count: config.rpm_recovery_count,
            freq: loop_rate_hz,
        }
    }

    /// Apply the effectiveness param block to the controller. Call while
    /// disarmed only (boot + the disarmed param hot-reload) — it swaps the
    /// allocator's B matrix and the actuator-state dynamics mid-loop.
    ///
    /// Range authority is the schema `ParamMeta` (enforced at shell set,
    /// YAML bake, and flash replay); this function performs only structural
    /// safety checks (finite, positive, magnitude) and therefore can never
    /// disagree with the schema ranges — the old `update_from_learned`
    /// duplicated a *tighter* tau range and silently rejected legal values.
    ///
    /// The actuator facts (`tau`, `omega_max`, `g2_*`, `nonlin`) come from
    /// `motors` — the airframe group — and only the identified G1 override
    /// comes from `p`. They are separate arguments rather than one struct
    /// because they answer to different owners: `motors` describes the
    /// hardware, `p` supersedes a computation over it.
    ///
    /// Per-block semantics (each block independent — a bad value in one
    /// never blocks the others):
    /// - **tau/omega** per motor: accepted iff finite and > 0; updates
    ///   `max_omega`, the G2 scaler, and the PT1 actuator-state alpha.
    /// - **G2**: copied verbatim from `motors` (it IS the G2 source; there
    ///   is no const seed) iff every entry is finite and bounded.
    /// - **G1**: all-zero block = "derive geometrically" sentinel → the
    ///   geometry-derived matrix is (re)stored. A configured block is
    ///   validated; on failure the geometric matrix is restored. Zero
    ///   control authority is structurally impossible.
    /// - **nonlinearity** per motor: param if > 0, else `fallback_k` (the
    ///   thrust-model-matched compile-time default).
    pub fn apply_effectiveness_params(
        &mut self,
        motors: &[MotorParams; NU],
        p: &crate::params::IndiEffectivenessParams,
        fallback_k: f32,
    ) -> EffectivenessApplyReport {
        // Magnitude bound: ~30× the largest geometric G1 entry for a
        // typical micro-quad. Anything beyond is a corrupted config, not a
        // real vehicle.
        const G_MAG_MAX: f32 = 1e4;
        let g_valid = |v: f32| v.is_finite() && v.abs() <= G_MAG_MAX;

        // Motor dynamics (tau/omega), per motor.
        let dt = 1.0 / self.freq;
        let mut motor_dynamics_ok = [true; NU];
        for i in 0..NU {
            let tau = motors[i].time_const_s;
            let omega = motors[i].max_omega_rad_s;
            if tau.is_finite() && tau > 0.0 && omega.is_finite() && omega > 0.0 {
                self.effectiveness.max_omega[i] = omega;
                self.effectiveness.g2_scaler[i] = 0.5 * omega * omega / tau;
                self.pt1_alpha[i] = dt / (tau + dt);
            } else {
                motor_dynamics_ok[i] = false;
            }
        }

        // G2: airframe matrix verbatim.
        let g2_ok = motors.iter().flat_map(|m| m.g2.iter()).all(|&v| g_valid(v));
        if g2_ok {
            for i in 0..NU {
                for j in 0..3 {
                    self.effectiveness.g2[(j, i)] = motors[i].g2[j];
                }
            }
        }

        // G1: zero-sentinel → geometric; configured → validate or restore
        // geometric.
        let g1_entries = || p.g1_force.iter().flatten().chain(p.g1_torque.iter().flatten());
        let g1 = if g1_entries().all(|&v| v == 0.0) {
            self.effectiveness.g1 = self.effectiveness.g1_geometric;
            G1Application::Geometric
        } else if g1_entries().all(|&v| g_valid(v)) {
            for i in 0..NU {
                for j in 0..3 {
                    self.effectiveness.g1[(j, i)] = p.g1_force[i][j];
                    self.effectiveness.g1[(j + 3, i)] = p.g1_torque[i][j];
                }
            }
            G1Application::Configured
        } else {
            self.effectiveness.g1 = self.effectiveness.g1_geometric;
            G1Application::RejectedKeptGeometric
        };

        // Nonlinearity: per-motor param with model-matched fallback.
        for i in 0..NU {
            let k = motors[i].nonlinearity;
            let k = if k.is_finite() && k > 0.0 { k } else { fallback_k };
            self.linearization[i] = super::linearization::ThrustLinearization::new(
                k,
                self.thrust_model,
                self.linearization[i].per_motor_max_n(),
            );
        }

        EffectivenessApplyReport {
            g1,
            g2_ok,
            motor_dynamics_ok,
        }
    }

    /// The cutoff actually in use by every synchronized filter, after the
    /// Nyquist clamp against the loop rate. Callers that build their own
    /// filters feeding [`MotorState::External`] must use THIS value, not
    /// the requested `sync_filter_hz`, or the delay match INDI depends on
    /// silently breaks. Also lets the firmware log when a parameter was
    /// clamped (core stays log-free).
    pub fn effective_sync_filter_hz(&self) -> f32 {
        self.sync_filter_hz
    }

    /// Update actuator state estimation from last motor command.
    /// Must be called every loop even when INDI is not the active controller.
    /// `voltage_v` is consumed only by `ThrustModel::Table`; pass any value
    /// for the analytic models.
    pub fn update_actuator_state(&mut self, d: &SVector<f32, NU>, voltage_v: f32) {
        for i in 0..NU {
            let u = self.linearization[i].output_curve(d[i], voltage_v);
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

    /// Run one INDI iteration from a body-rate setpoint (legacy entry).
    ///
    /// The rate-error stage `rate_dot_sp = rate_gains ∘ (rate_sp − gyro)`
    /// runs here; everything downstream is shared with [`Self::step_alpha`]
    /// via `step_rate_dot`.
    ///
    /// Returns `(IndiOutput, IndiStepState)`. The `IndiStepState` exposes
    /// raw intermediate signals (pre-sync-filter rate_dot, ground
    /// detection) for external observers.
    pub fn step(
        &mut self,
        gyro_rad_s: &Vector3<f32>,
        accel_m_s2: &Vector3<f32>,
        rate_sp: &Vector3<f32>,
        spf_sp_z: f32,
        armed: bool,
        g2_valid: &[bool; NU],
        motor_state: MotorState<'_>,
        voltage_v: f32,
    ) -> (IndiOutput, IndiStepState) {
        // --- Rate controller (the ONLY consumer of `rate_gains`) ---
        let rate_err = *rate_sp - *gyro_rad_s;
        let rate_dot_sp = self.rate_gains.component_mul(&rate_err);
        self.step_rate_dot(
            gyro_rad_s,
            accel_m_s2,
            &rate_dot_sp,
            spf_sp_z,
            armed,
            g2_valid,
            motor_state,
            voltage_v,
        )
    }

    /// Run one INDI iteration from an angular-acceleration setpoint.
    ///
    /// Entry point for the `mpc_full` architecture (Sun et al., T-RO 2022,
    /// Fig. 3): the outer full-model NMPC supplies the desired angular
    /// acceleration `α_d` directly (see
    /// `mpc::FullQuadModel::inner_setpoint`), so **no rate gains are
    /// involved** — the rate loop lives inside the optimizer. The torque
    /// loop below still runs at full IMU rate: with INDI enabled the
    /// increment `α_d − ω̇_f` plus rotor-speed feedback rejects unmodeled
    /// torque (paper eq. 33–35); with INDI disabled (`indi_enabled =
    /// false` / on ground / disarmed) this degrades to the paper's
    /// "NMPC w/o INDI" baseline — model-based static inversion of `α_d`
    /// through G1 alone (eq. 29–30).
    ///
    /// `α_d` is expected to be *held* between outer-loop solves; the
    /// incremental correction re-evaluates against fresh gyro/rotor data
    /// every call.
    #[allow(clippy::too_many_arguments)]
    pub fn step_alpha(
        &mut self,
        gyro_rad_s: &Vector3<f32>,
        accel_m_s2: &Vector3<f32>,
        alpha_sp_rad_s2: &Vector3<f32>,
        spf_sp_z: f32,
        armed: bool,
        g2_valid: &[bool; NU],
        motor_state: MotorState<'_>,
        voltage_v: f32,
    ) -> (IndiOutput, IndiStepState) {
        self.step_rate_dot(
            gyro_rad_s,
            accel_m_s2,
            alpha_sp_rad_s2,
            spf_sp_z,
            armed,
            g2_valid,
            motor_state,
            voltage_v,
        )
    }

    /// Shared INDI pipeline downstream of the pseudo-control input:
    /// sensor processing, takeoff detection, incremental pseudo-control,
    /// WLS allocation, NaN protection, motor-command linearization.
    #[allow(clippy::too_many_arguments)]
    fn step_rate_dot(
        &mut self,
        gyro_rad_s: &Vector3<f32>,
        accel_m_s2: &Vector3<f32>,
        rate_dot_sp: &Vector3<f32>,
        spf_sp_z: f32,
        armed: bool,
        g2_valid: &[bool; NU],
        motor_state: MotorState<'_>,
        voltage_v: f32,
    ) -> (IndiOutput, IndiStepState) {
        // --- 1. Sensor processing ---
        self.rate_dot_estimator.update(gyro_rad_s);
        let rate_dot_raw = self.rate_dot_estimator.raw();
        let rate_dot_fs = self.rate_dot_estimator.filtered();

        let spf_fs = Vector3::from(self.spf_filter.apply((*accel_m_s2).into()));

        // Resolve filtered motor speed and acceleration. `Internal` matches the
        // C reference (du-based fallback); `External` consumes a caller-supplied
        // dshot-derived pair. A non-finite element in either External vector
        // forces the same-step fallback to Internal — protects WLS from NaN
        // injection if the upstream LPF + finite-diff produces a bad sample.
        let used_external = matches!(motor_state, MotorState::External { .. })
            && match motor_state {
                MotorState::External {
                    omega_fs,
                    omega_dot_fs,
                } => {
                    omega_fs.iter().all(|v| v.is_finite())
                        && omega_dot_fs.iter().all(|v| v.is_finite())
                }
                MotorState::Internal => false,
            };
        let (omega_fs_vec, omega_dot_fs) = match motor_state {
            MotorState::External {
                omega_fs,
                omega_dot_fs,
            } if used_external => (*omega_fs, *omega_dot_fs),
            _ => {
                let mut omega_dot = SVector::<f32, NU>::zeros();
                for i in 0..NU {
                    let omega_inv = super::effectiveness::omega_inv_guarded(
                        self.prev_omega_fs[i],
                        self.effectiveness.max_omega[i],
                    );
                    omega_dot[i] = self.prev_du[i] * self.effectiveness.g2_scaler[i] * omega_inv;
                }
                (self.prev_omega_fs, omega_dot)
            }
        };

        // Actuator state filtering
        self.u_state_fs = self
            .u_state_filter
            .apply(self.u_state.into())
            .map(|w| w.clamp(0.0, 1.0))
            .into();

        // --- 2. Takeoff detection ---
        let gyro_mag_sq = gyro_rad_s.norm_squared();
        let accel_mag_sq = accel_m_s2.norm_squared();
        let gyro_thresh = self.ground_gyro_rad_s;
        let gyro_low = gyro_mag_sq < gyro_thresh * gyro_thresh;
        let accel_thresh = self.ground_accel_m_s2;
        let accel_high = accel_mag_sq > accel_thresh * accel_thresh;
        let thrust_low = spf_sp_z < self.ground_thrust_sp_m_s2;
        let touching_ground = gyro_low && accel_high && thrust_low;
        let do_indi = self.indi_enabled && !touching_ground && armed;
        let do_indi_f = do_indi as u32 as f32;

        // G2 models the yaw reaction torque of a motor *acceleration*, so
        // its columns are only meaningful against an incremental `du`.
        // Whenever the increment is off — INDI disabled, disarmed, or
        // sitting on the ground — `du` IS the absolute command, and adding
        // G2 to G1 would mix two different input definitions in one B
        // matrix. Allocate through G1 alone in all three cases; `do_indi`
        // is exactly that condition.
        let g2_eff: [bool; NU] = if do_indi { *g2_valid } else { [false; NU] };

        // --- 3. Pseudo-control ---
        // (The rate controller, when used, ran in `step` — `rate_dot_sp`
        // arrives here as the α-space pseudo-control input.)
        let dv = stack![
            Vector3::new(0.0, 0.0, spf_sp_z - do_indi_f * spf_fs.z);
            *rate_dot_sp - do_indi_f * rate_dot_fs
                + do_indi_f * self.effectiveness.g2 * omega_dot_fs
        ];

        // --- 4. Combined effectiveness matrix ---
        let g1g2: SMatrix<f32, NV, NU> = self.effectiveness.combined_g1g2(&omega_fs_vec, &g2_eff);

        // --- 5. WLS allocation ---
        let wv = self.wls_wv;
        let mut wu = self.wls_wu;

        let (a_mat, gamma) =
            setup_a::<NU, NV, NC>(&g1g2, &wv, &mut wu, self.wls_theta, self.wls_cond_bound);

        let du_min = -do_indi_f * self.u_state_fs;
        let du_max = self.act_limit - do_indi_f * self.u_state_fs;
        let du_pref = -do_indi_f * self.u_state_fs;

        let b_vec = setup_b::<NU, NV, NC>(&dv, &du_pref, &wv, &wu, gamma);

        let mut du = (du_min + du_max) / 2.0;

        let stats = solve::<NU, NV, NC>(
            &a_mat,
            &b_vec,
            &du_min,
            &du_max,
            &mut du,
            &mut self.ws,
            self.wls_imax,
        );

        // --- 6. NaN protection ---
        let nan_exit =
            stats.exit_code == ExitCode::NanFoundQ || stats.exit_code == ExitCode::NanFoundUs;
        if nan_exit {
            self.nan_counter += 1;
            self.ws = [0; NU];
        } else {
            self.nan_counter = 0;
        }
        let nan_failsafe = self.nan_counter > self.nan_limit;

        // --- 7. Motor commands ---
        let u = if !nan_exit {
            self.u_state_fs
                .zip_map(&du, |u_fs, du| (do_indi_f * u_fs + du).max(0.0))
        } else {
            self.u_state_fs
                .map(|u_fs| (u_fs * self.nan_rampdown).max(0.0))
        }
        .inf(&self.act_limit);

        let motor_commands =
            SVector::<_, NU>::from_fn(|i, _| self.linearization[i].linearize(u[i], voltage_v));
        // `prev_du` only feeds the Internal du-based ω̇ fallback. While in
        // External mode, ω̇ comes from telemetry and prev_du isn't read — we
        // also zero it here so a transition back to Internal starts the next
        // step with `omega_dot = 0` instead of a value that reflects the
        // External-mode WLS history (which was solved with a different ω̇
        // source). The first Internal step after a fallback then has no
        // model-based ω̇ contribution; subsequent steps recompute prev_du
        // normally from the latest u/u_state.
        self.prev_du = if used_external {
            SVector::zeros()
        } else {
            u - self.u_state
        };

        if motor_commands.iter().all(|v| v.is_finite()) {
            self.update_actuator_state(&motor_commands, voltage_v);
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

    /// Nominal voltage used by tests — analytic thrust models ignore it,
    /// so any finite value works; the value here matches a 6S mid-pack.
    const V_NOM: f32 = 23.0;

    const LOOP_HZ: f32 = 8000.0;
    const GRAVITY: f32 = 9.80665;

    fn test_config() -> IndiConfig {
        IndiConfig {
            indi_enabled: true,
            ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
            ground_accel_m_s2: 0.8 * 9.81,
            ground_thrust_sp_m_s2: 3.0,
            rate_gains: Vector3::new(20.0, 20.0, 20.0),
            sync_filter_hz: 15.0,
            rate_dot_sg_window_size: 7,
            rate_dot_sg_order: 2,
            motors: [
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
            ],
            body: RigidBodyParams {
                mass_kg: 0.55,
                inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
                max_rate_rad_s: [10.0, 10.0, 6.0],
            },
            indi_motors: [IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: 0.0,
            }; NU],
            thrust_model: ThrustModel::Quadratic,
            nonlinearity: SVector::from_element(0.5),
            act_limit: SVector::from_element(1.0),
            wls_wv: [1.0, 1.0, 50.0, 50.0, 50.0, 5.0].into(),
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

    fn hover_inputs() -> (Vector3<f32>, Vector3<f32>, Vector3<f32>, f32) {
        (
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, GRAVITY),
            Vector3::zeros(),
            GRAVITY,
        )
    }

    #[test]
    fn new_does_not_panic() {
        let _ctrl = IndiController::new(&test_config(), LOOP_HZ);
    }

    /// `step_alpha` must be bit-identical to `step` when fed the exact
    /// pseudo-control the legacy rate stage would have produced —
    /// guarantees the refactor changed no numerics for existing users.
    #[test]
    fn step_alpha_equals_step_for_equivalent_pseudo_control() {
        let cfg = test_config();
        let mut ctrl_rate = IndiController::new(&cfg, LOOP_HZ);
        let mut ctrl_alpha = IndiController::new(&cfg, LOOP_HZ);
        let g2 = [true; NU];

        // Drive both controllers through an identical, non-trivial input
        // sequence: varying gyro + rate setpoints so filters and actuator
        // state evolve away from init.
        for k in 0..200 {
            let t = k as f32 / LOOP_HZ;
            let gyro = Vector3::new(
                0.6 * libm::sinf(40.0 * t),
                -0.4 * libm::cosf(25.0 * t),
                0.2 * libm::sinf(10.0 * t),
            );
            let accel = Vector3::new(0.3, -0.2, GRAVITY + 0.5 * libm::sinf(30.0 * t));
            let rate_sp = Vector3::new(1.0, -0.5, 0.25);
            let spf = GRAVITY + 1.0;

            let (out_rate, _) = ctrl_rate.step(
                &gyro, &accel, &rate_sp, spf, true, &g2, MotorState::Internal, V_NOM,
            );
            // The equivalent pseudo-control the legacy stage computes.
            let alpha_sp = cfg.rate_gains.component_mul(&(rate_sp - gyro));
            let (out_alpha, _) = ctrl_alpha.step_alpha(
                &gyro, &accel, &alpha_sp, spf, true, &g2, MotorState::Internal, V_NOM,
            );
            assert_eq!(
                out_rate.motor_commands, out_alpha.motor_commands,
                "diverged at step {k}"
            );
        }
    }

    /// With INDI disabled, `step_alpha` degrades to model-based static
    /// inversion of the commanded angular acceleration through G1 (the
    /// paper's "NMPC w/o INDI" baseline): a positive roll-α command must
    /// load the left motors (M2/M3, +y in FLU) more than the right pair.
    #[test]
    fn step_alpha_static_inversion_roll_sign() {
        let cfg = IndiConfig {
            indi_enabled: false,
            ..test_config()
        };
        let mut ctrl = IndiController::new(&cfg, LOOP_HZ);
        let g2 = [false; NU];
        let gyro = Vector3::zeros();
        let accel = Vector3::new(0.0, 0.0, GRAVITY);

        // Positive roll angular-acceleration demand at hover thrust.
        let alpha_sp = Vector3::new(200.0, 0.0, 0.0);
        let mut out = ctrl
            .step_alpha(&gyro, &accel, &alpha_sp, GRAVITY, true, &g2, MotorState::Internal, V_NOM)
            .0;
        for _ in 0..50 {
            out = ctrl
                .step_alpha(
                    &gyro, &accel, &alpha_sp, GRAVITY, true, &g2, MotorState::Internal, V_NOM,
                )
                .0;
        }
        let m = out.motor_commands;
        assert!(m.iter().all(|v| v.is_finite()), "commands must be finite");
        let left = m[2] + m[3];
        let right = m[0] + m[1];
        assert!(
            left > right + 1e-4,
            "positive roll α must load left motors: left={left}, right={right}"
        );
    }

    /// The `imu_1khz` firmware setup: 1 kHz loop, SG window 5, SG target =
    /// loop rate, SG window 5 (its delay-budgeted value). Mirrors
    /// `hover_produces_equal_motor_commands` so both flight rates get a
    /// construction + convergence smoke test.
    #[test]
    fn hover_converges_at_1khz() {
        let mut config = test_config();
        config.rate_dot_sg_window_size = 5;
        let mut ctrl = IndiController::new(&config, 1000.0);
        let (gyro, accel, rate_sp, spf_sp_z) = hover_inputs();
        let g2 = [false; NU];
        let mut out = ctrl.step(&gyro, &accel, &rate_sp, spf_sp_z, true, &g2, MotorState::Internal, V_NOM).0;
        for _ in 0..200 {
            out = ctrl.step(&gyro, &accel, &rate_sp, spf_sp_z, true, &g2, MotorState::Internal, V_NOM).0;
        }
        let mean = out.motor_commands.iter().sum::<f32>() / NU as f32;
        for (i, &c) in out.motor_commands.iter().enumerate() {
            assert!(c >= 0.0 && c <= 1.0, "motor {i} out of bounds: {c}");
            assert!(
                (c - mean).abs() < 0.05,
                "motor {i} diverges: {c} vs mean {mean}"
            );
        }
        assert!(mean > 0.1 && mean < 0.7, "hover mean={mean} unexpected");
        assert!(!out.nan_failsafe);
    }

    #[test]
    fn hover_produces_equal_motor_commands() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (gyro, accel, rate_sp, spf_sp_z) = hover_inputs();
        let g2 = [false; NU];
        let mut out = ctrl.step(&gyro, &accel, &rate_sp, spf_sp_z, true, &g2, MotorState::Internal, V_NOM).0;
        for _ in 0..200 {
            out = ctrl.step(&gyro, &accel, &rate_sp, spf_sp_z, true, &g2, MotorState::Internal, V_NOM).0;
        }
        let mean = out.motor_commands.iter().sum::<f32>() / NU as f32;
        for (i, &c) in out.motor_commands.iter().enumerate() {
            assert!(c >= 0.0 && c <= 1.0, "motor {i} out of bounds: {c}");
            assert!(
                (c - mean).abs() < 0.05,
                "motor {i} diverges: {c} vs mean {mean}"
            );
        }
        assert!(mean > 0.1 && mean < 0.7, "hover mean={mean} unexpected");
        assert!(!out.nan_failsafe);
    }

    /// `indi_enabled: false` still hovers and still tracks a rate command
    /// — it is a rate controller, just not an incremental one.
    #[test]
    fn indi_disabled_hovers_and_tracks_rate() {
        let cfg = IndiConfig {
            indi_enabled: false,
            ..test_config()
        };
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [true; NU]; // G2 is dropped internally; assert it's harmless

        let mut ctrl = IndiController::new(&cfg, LOOP_HZ);
        let mut out = ctrl.step(&Vector3::zeros(), &a, &Vector3::zeros(), GRAVITY, true, &g2, MotorState::Internal, V_NOM).0;
        for _ in 0..200 {
            out = ctrl
                .step(&Vector3::zeros(), &a, &Vector3::zeros(), GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                .0;
        }
        let mean = out.motor_commands.iter().sum::<f32>() / NU as f32;
        for (i, &c) in out.motor_commands.iter().enumerate() {
            assert!((0.0..=1.0).contains(&c), "motor {i} out of bounds: {c}");
            assert!((c - mean).abs() < 0.05, "motor {i} diverges: {c} vs {mean}");
        }
        assert!(mean > 0.1 && mean < 0.7, "hover mean={mean} unexpected");
        assert!(!out.nan_failsafe);

        // Roll-rate command → differential thrust, correct direction.
        let rate_sp = Vector3::new(3.0, 0.0, 0.0);
        for _ in 0..50 {
            out = ctrl
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                .0;
        }
        let left = (out.motor_commands[2] + out.motor_commands[3]) / 2.0;
        let right = (out.motor_commands[0] + out.motor_commands[1]) / 2.0;
        assert!(left > right, "INDI off: roll left={left} should > right={right}");
    }

    /// The discriminator between the two laws: a persistent gap between the
    /// commanded and measured specific force. INDI reads it as "I need this
    /// much MORE, every tick" and integrates the command to saturation;
    /// without the increment the same gap is an absolute demand that settles
    /// at hover. If `indi_enabled` ever stopped cutting the feedback path,
    /// this test would see both controllers saturate.
    #[test]
    fn indi_disabled_drops_the_incremental_feedback() {
        let a_free_fall = Vector3::zeros(); // spf reads 0, setpoint asks for 1 g
        let g2 = [false; NU];
        let run = |enabled: bool| {
            let cfg = IndiConfig {
                indi_enabled: enabled,
                ..test_config()
            };
            let mut ctrl = IndiController::new(&cfg, LOOP_HZ);
            let mut out = ctrl
                .step(&Vector3::zeros(), &a_free_fall, &Vector3::zeros(), GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                .0;
            // 1 s at LOOP_HZ — long enough for the actuator PT1 + 15 Hz
            // sync filter to close the u_fs ← u loop the increment rides on.
            for _ in 0..(LOOP_HZ as usize) {
                out = ctrl
                    .step(&Vector3::zeros(), &a_free_fall, &Vector3::zeros(), GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                    .0;
            }
            out.motor_commands.iter().sum::<f32>() / NU as f32
        };
        let on = run(true);
        let off = run(false);
        assert!(on > 0.95, "INDI on: sustained spf gap should saturate, got {on}");
        assert!(off < 0.7, "INDI off: should settle near hover, got {off}");
    }

    #[test]
    fn motor_commands_always_in_bounds() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let accel = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        let cases: &[(Vector3<f32>, Vector3<f32>, f32)] = &[
            (Vector3::zeros(), Vector3::zeros(), GRAVITY),
            (Vector3::zeros(), Vector3::new(5.0, -3.0, 1.0), GRAVITY),
            (
                Vector3::zeros(),
                Vector3::new(15.0, 15.0, 0.0),
                GRAVITY * 2.0,
            ),
            (Vector3::new(1.7, -0.9, 0.3), Vector3::zeros(), GRAVITY),
        ];
        for (gyro, rate_sp, spf) in cases {
            for _ in 0..50 {
                let out = ctrl.step(gyro, &accel, rate_sp, *spf, true, &g2, MotorState::Internal, V_NOM).0;
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
        for _ in 0..100 {
            ctrl.step(
                &Vector3::zeros(),
                &accel,
                &Vector3::zeros(),
                GRAVITY,
                true,
                &g2,
                MotorState::Internal,
                V_NOM,
            )
            .0;
        }
        let rate_sp = Vector3::new(3.0, 0.0, 0.0);
        let mut out = ctrl
            .step(&Vector3::zeros(), &accel, &rate_sp, GRAVITY, true, &g2, MotorState::Internal, V_NOM)
            .0;
        for _ in 0..50 {
            out = ctrl
                .step(&Vector3::zeros(), &accel, &rate_sp, GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                .0;
        }
        let left = (out.motor_commands[2] + out.motor_commands[3]) / 2.0;
        let right = (out.motor_commands[0] + out.motor_commands[1]) / 2.0;
        assert!(left > right, "roll: left={left} should > right={right}");
    }

    #[test]
    fn disarmed_no_failsafe() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (g, a, r, s) = hover_inputs();
        for _ in 0..50 {
            let out = ctrl.step(&g, &a, &r, s, false, &[false; NU], MotorState::Internal, V_NOM).0;
            assert!(!out.nan_failsafe);
            for &c in &out.motor_commands {
                assert!(c.is_finite() && c >= 0.0 && c <= 1.0);
            }
        }
    }

    #[test]
    fn ground_ndi_valid() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let gyro = Vector3::zeros();
        let accel = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        let out1 = ctrl
            .step(&gyro, &accel, &Vector3::zeros(), 2.0, false, &g2, MotorState::Internal, V_NOM)
            .0;
        let out2 = ctrl
            .step(&gyro, &accel, &Vector3::zeros(), 2.0, false, &g2, MotorState::Internal, V_NOM)
            .0;
        for &c in &out1.motor_commands {
            assert!(c.is_finite() && c >= 0.0 && c <= 1.0);
        }
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
        for _ in 0..50 {
            ctrl.step(&g, &a, &r, 2.0, false, &g2, MotorState::Internal, V_NOM).0;
        }
        let ground = ctrl.step(&g, &a, &r, 2.0, false, &g2, MotorState::Internal, V_NOM).0;
        let air = ctrl.step(&g, &a, &r, GRAVITY, true, &g2, MotorState::Internal, V_NOM).0;
        for i in 0..NU {
            assert!(
                (air.motor_commands[i] - ground.motor_commands[i]).abs() < 0.5,
                "takeoff spike motor {i}"
            );
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
            if g2.iter().all(|&v| !v) {
                return;
            }
        }
        panic!("G2 should have been zeroed");
    }

    #[test]
    fn update_rpm_all_invalid_failsafe() {
        let cfg = IndiConfig {
            ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
            ground_accel_m_s2: 0.8 * 9.81,
            ground_thrust_sp_m_s2: 3.0,
            rpm_invalid_limit: 5,
            rpm_all_invalid_limit: 10,
            ..test_config()
        };
        let mut ctrl = IndiController::new(&cfg, LOOP_HZ);
        ctrl.update_rpm(&[RpmInput::Erpm(10000); NU]);
        for _ in 0..20 {
            let (_, fs) = ctrl.update_rpm(&[RpmInput::Invalid; NU]);
            if fs {
                return;
            }
        }
        panic!("RPM failsafe should have triggered");
    }

    #[test]
    fn asymmetric_limits_respected() {
        use nalgebra::Vector4;
        let cfg = IndiConfig {
            ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
            ground_accel_m_s2: 0.8 * 9.81,
            ground_thrust_sp_m_s2: 3.0,
            act_limit: Vector4::new(0.8, 1.0, 1.0, 1.0),
            ..test_config()
        };
        let mut ctrl = IndiController::new(&cfg, LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(10.0, 10.0, 0.0);
        let g2 = [false; NU];
        for _ in 0..100 {
            let out = ctrl
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                .0;
            assert!(
                out.motor_commands[0] <= 0.8 + 1e-6,
                "M0 exceeded: {}",
                out.motor_commands[0]
            );
        }
    }

    #[test]
    fn sequential_steps_converge() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (g, a, r, s) = hover_inputs();
        let g2 = [false; NU];
        let mut prev = SVector::from_element(0.0f32);
        for step in 0..500 {
            let out = ctrl.step(&g, &a, &r, s, true, &g2, MotorState::Internal, V_NOM).0;
            let max_change = (out.motor_commands - prev).abs().max();

            if step > 100 && max_change < 1e-5 {
                return;
            }
            prev = out.motor_commands;
        }
        panic!("hover did not converge in 500 steps");
    }

    #[test]
    fn nonzero_gyro_produces_correction() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let g2 = [false; NU];
        for _ in 0..100 {
            ctrl.step(&Vector3::zeros(), &a, &Vector3::zeros(), GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                .0;
        }
        let hover = ctrl
            .step(&Vector3::zeros(), &a, &Vector3::zeros(), GRAVITY, true, &g2, MotorState::Internal, V_NOM)
            .0;
        let gyro = Vector3::new(100.0f32.to_radians(), 0.0, 0.0);
        let mut spin = hover;
        for _ in 0..20 {
            spin = ctrl
                .step(&gyro, &a, &Vector3::zeros(), GRAVITY, true, &g2, MotorState::Internal, V_NOM)
                .0;
        }
        let diff = spin
            .motor_commands
            .iter()
            .zip(hover.motor_commands.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 0.01, "gyro should change allocation: diff={diff}");
    }

    #[test]
    fn warmstart_stable_output() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let (g, a, r, s) = hover_inputs();
        let g2 = [false; NU];
        // Settle until converged (filters + actuator state)
        let mut prev = SVector::<f32, NU>::zeros();
        for step in 0..1000 {
            let out = ctrl.step(&g, &a, &r, s, true, &g2, MotorState::Internal, V_NOM).0;
            let max_change = out
                .motor_commands
                .iter()
                .zip(prev.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            prev = out.motor_commands;
            if step > 50 && max_change < 1e-6 {
                break;
            }
            assert!(
                step < 999,
                "warmstart test: failed to converge in 1000 steps"
            );
        }
        // Now check 10 subsequent outputs are nearly identical
        let mut outputs = Vec::new();
        for _ in 0..10 {
            outputs.push(ctrl.step(&g, &a, &r, s, true, &g2, MotorState::Internal, V_NOM).0.motor_commands);
        }
        for i in 1..10 {
            let diff = outputs[i]
                .iter()
                .zip(outputs[i - 1].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
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
            (
                Vector3::zeros(),
                Vector3::new(10.0, -10.0, 5.0),
                GRAVITY * 3.0,
                true,
            ),
            (
                Vector3::new(200.0f32.to_radians(), 0.0, 0.0),
                Vector3::zeros(),
                GRAVITY,
                true,
            ),
            (
                Vector3::new(0.01, -0.005, 0.002),
                Vector3::new(0.05, -0.03, 0.01),
                GRAVITY,
                true,
            ),
        ];
        for (gyro, rate_sp, spf, armed) in cases {
            for _ in 0..50 {
                let out = ctrl.step(gyro, &a, rate_sp, *spf, *armed, &g2, MotorState::Internal, V_NOM).0;
                for (i, &c) in out.motor_commands.iter().enumerate() {
                    assert!(c.is_finite() && c >= 0.0 && c <= 1.0, "motor {i} = {c}");
                }
            }
        }
    }

    #[test]
    fn g2_valid_affects_allocation() {
        let cfg = IndiConfig {
            ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
            ground_accel_m_s2: 0.8 * 9.81,
            ground_thrust_sp_m_s2: 3.0,
            indi_motors: [IndiMotorParams {
                time_const_s: 0.025,
                max_rpm: 40000.0,
                g2_yaw: 0.001,
            }; NU],
            ..test_config()
        };
        let mut ctrl_g2 = IndiController::new(&cfg, LOOP_HZ);
        let mut ctrl_no = IndiController::new(&cfg, LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 3.0);
        for _ in 0..100 {
            ctrl_g2.update_rpm(&[RpmInput::Erpm(20000); NU]);
            ctrl_g2
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
                .0;
            ctrl_no
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU], MotorState::Internal, V_NOM)
                .0;
        }
        let out_g2 = ctrl_g2
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
            .0;
        let out_no = ctrl_no
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU], MotorState::Internal, V_NOM)
            .0;
        let diff = out_g2
            .motor_commands
            .iter()
            .zip(out_no.motor_commands.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-6, "G2 should affect allocation: diff={diff}");
    }

    fn g2_config() -> IndiConfig {
        // FLU sign convention: CW motors get positive G2 yaw, CCW get negative
        IndiConfig {
            ground_gyro_rad_s: 100.0_f32 * core::f32::consts::PI / 180.0,
            ground_accel_m_s2: 0.8 * 9.81,
            ground_thrust_sp_m_s2: 3.0,
            indi_motors: [
                IndiMotorParams {
                    time_const_s: 0.025,
                    max_rpm: 40000.0,
                    g2_yaw: 0.001,
                }, // M0 CW
                IndiMotorParams {
                    time_const_s: 0.025,
                    max_rpm: 40000.0,
                    g2_yaw: -0.001,
                }, // M1 CCW
                IndiMotorParams {
                    time_const_s: 0.025,
                    max_rpm: 40000.0,
                    g2_yaw: -0.001,
                }, // M2 CCW
                IndiMotorParams {
                    time_const_s: 0.025,
                    max_rpm: 40000.0,
                    g2_yaw: 0.001,
                }, // M3 CW
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
            ctrl_g2
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
                .0;
            ctrl_no
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU], MotorState::Internal, V_NOM)
                .0;
        }

        let out_g2 = ctrl_g2
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
            .0;
        let out_no = ctrl_no
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU], MotorState::Internal, V_NOM)
            .0;

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
        ctrl.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
            .0;

        // Second step with G2 vs without G2: the omegaDot contribution to dv
        // should cause different outputs
        let mut ctrl2_g2 = IndiController::new(&g2_config(), LOOP_HZ);
        let mut ctrl2_no = IndiController::new(&test_config(), LOOP_HZ);

        ctrl2_g2.update_rpm(&[RpmInput::Erpm(20000); NU]);

        // Run both for enough steps to have meaningful prev_du
        for _ in 0..50 {
            ctrl2_g2
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
                .0;
            ctrl2_no
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU], MotorState::Internal, V_NOM)
                .0;
        }

        let out_g2 = ctrl2_g2
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
            .0;
        let out_no = ctrl2_no
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[false; NU], MotorState::Internal, V_NOM)
            .0;

        let diff = out_g2
            .motor_commands
            .iter()
            .zip(out_no.motor_commands.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff > 1e-5,
            "omegaDot feedback should cause measurable difference: diff={diff}"
        );
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
            ctrl.step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
                .0;
        }
        let out_all = ctrl
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &[true; NU], MotorState::Internal, V_NOM)
            .0;

        // Reset and run with M0 G2 disabled
        let mut ctrl2 = IndiController::new(&g2_config(), LOOP_HZ);
        ctrl2.update_rpm(&[RpmInput::Erpm(20000); NU]);
        let mut g2_partial = [true; NU];
        g2_partial[0] = false;

        for _ in 0..100 {
            ctrl2
                .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &g2_partial, MotorState::Internal, V_NOM)
                .0;
        }
        let out_partial = ctrl2
            .step(&Vector3::zeros(), &a, &rate_sp, GRAVITY, true, &g2_partial, MotorState::Internal, V_NOM)
            .0;

        // Outputs should differ (M0's G2 column removed changes allocation)
        let diff = out_all
            .motor_commands
            .iter()
            .zip(out_partial.motor_commands.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff > 1e-6,
            "disabling one motor's G2 should change allocation: diff={diff}"
        );
    }

    #[test]
    fn external_motor_state_matches_internal_when_fed_internal_values() {
        // Drive two controllers in lockstep with identical Internal-mode steps.
        // On the final step, the External branch is fed the exact same ω_fs and
        // du-fallback ω̇_fs that Internal would compute. Outputs must match.
        let cfg = g2_config();
        let mut ctrl_int = IndiController::new(&cfg, LOOP_HZ);
        let mut ctrl_ext = IndiController::new(&cfg, LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 3.0);

        // Settle both with G2 active. Identical inputs → identical state.
        for _ in 0..50 {
            ctrl_int.update_rpm(&[RpmInput::Erpm(20000); NU]);
            ctrl_ext.update_rpm(&[RpmInput::Erpm(20000); NU]);
            ctrl_int
                .step(
                    &Vector3::zeros(),
                    &a,
                    &rate_sp,
                    GRAVITY,
                    true,
                    &[true; NU],
                    MotorState::Internal,
                    V_NOM,
                )
                .0;
            ctrl_ext
                .step(
                    &Vector3::zeros(),
                    &a,
                    &rate_sp,
                    GRAVITY,
                    true,
                    &[true; NU],
                    MotorState::Internal,
                    V_NOM,
                )
                .0;
        }

        // Final update_rpm BEFORE the snapshot — Internal will see this same
        // post-update prev_omega_fs when its own step runs.
        ctrl_int.update_rpm(&[RpmInput::Erpm(20000); NU]);
        ctrl_ext.update_rpm(&[RpmInput::Erpm(20000); NU]);

        // Reconstruct what Internal would compute for ω_fs / ω̇_fs at the start
        // of the next step from ctrl_ext's own state.
        let omega_fs = ctrl_ext.prev_omega_fs;
        let mut omega_dot_fs = SVector::<f32, NU>::zeros();
        for i in 0..NU {
            let inv_thresh = 0.1 * ctrl_ext.effectiveness.max_omega[i];
            let omega_inv = if ctrl_ext.prev_omega_fs[i].abs() > inv_thresh {
                1.0 / ctrl_ext.prev_omega_fs[i]
            } else {
                1.0 / inv_thresh
            };
            omega_dot_fs[i] =
                ctrl_ext.prev_du[i] * ctrl_ext.effectiveness.g2_scaler[i] * omega_inv;
        }

        let out_int = ctrl_int
            .step(
                &Vector3::zeros(),
                &a,
                &rate_sp,
                GRAVITY,
                true,
                &[true; NU],
                MotorState::Internal,
                V_NOM,
            )
            .0;
        let out_ext = ctrl_ext
            .step(
                &Vector3::zeros(),
                &a,
                &rate_sp,
                GRAVITY,
                true,
                &[true; NU],
                MotorState::External {
                    omega_fs: &omega_fs,
                    omega_dot_fs: &omega_dot_fs,
                },
                V_NOM,
            )
            .0;
        for i in 0..NU {
            let d = (out_int.motor_commands[i] - out_ext.motor_commands[i]).abs();
            assert!(
                d < 1e-6,
                "External fed Internal-equivalent values diverged on motor {i}: \
                 int={} ext={} diff={d}",
                out_int.motor_commands[i],
                out_ext.motor_commands[i],
            );
        }
    }

    #[test]
    fn external_non_finite_inputs_fall_back_to_internal() {
        // A non-finite element in either External vector must NOT propagate
        // into motor_commands. The controller silently falls back to the
        // Internal computation for that step. WLS should not see NaN, the
        // NaN counter should not increment, and outputs stay in [0, 1].
        let mut ctrl = IndiController::new(&g2_config(), LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 3.0);

        // Settle on the Internal path so prev_du / prev_omega_fs are healthy.
        for _ in 0..100 {
            ctrl.update_rpm(&[RpmInput::Erpm(20000); NU]);
            ctrl.step(
                &Vector3::zeros(),
                &a,
                &rate_sp,
                GRAVITY,
                true,
                &[true; NU],
                MotorState::Internal,
                V_NOM,
            )
            .0;
        }

        let omega_fs_clean = SVector::<f32, NU>::from_element(2000.0);
        let omega_dot_fs_clean = SVector::<f32, NU>::zeros();

        // Case 1: NaN in omega_fs.
        let mut omega_fs_nan = omega_fs_clean;
        omega_fs_nan[2] = f32::NAN;
        let out = ctrl
            .step(
                &Vector3::zeros(),
                &a,
                &rate_sp,
                GRAVITY,
                true,
                &[true; NU],
                MotorState::External {
                    omega_fs: &omega_fs_nan,
                    omega_dot_fs: &omega_dot_fs_clean,
                },
                V_NOM,
            )
            .0;
        for (i, &c) in out.motor_commands.iter().enumerate() {
            assert!(
                c.is_finite() && (0.0..=1.0).contains(&c),
                "NaN omega_fs leaked into motor {i}: {c}"
            );
        }
        assert!(!out.nan_failsafe);

        // Case 2: +inf in omega_dot_fs.
        let mut omega_dot_fs_inf = omega_dot_fs_clean;
        omega_dot_fs_inf[0] = f32::INFINITY;
        let out2 = ctrl
            .step(
                &Vector3::zeros(),
                &a,
                &rate_sp,
                GRAVITY,
                true,
                &[true; NU],
                MotorState::External {
                    omega_fs: &omega_fs_clean,
                    omega_dot_fs: &omega_dot_fs_inf,
                },
                V_NOM,
            )
            .0;
        for (i, &c) in out2.motor_commands.iter().enumerate() {
            assert!(
                c.is_finite() && (0.0..=1.0).contains(&c),
                "inf omega_dot_fs leaked into motor {i}: {c}"
            );
        }
        assert!(!out2.nan_failsafe);

        // Case 3: -inf in omega_fs.
        let mut omega_fs_ninf = omega_fs_clean;
        omega_fs_ninf[3] = f32::NEG_INFINITY;
        let out3 = ctrl
            .step(
                &Vector3::zeros(),
                &a,
                &rate_sp,
                GRAVITY,
                true,
                &[true; NU],
                MotorState::External {
                    omega_fs: &omega_fs_ninf,
                    omega_dot_fs: &omega_dot_fs_clean,
                },
                V_NOM,
            )
            .0;
        for (i, &c) in out3.motor_commands.iter().enumerate() {
            assert!(
                c.is_finite() && (0.0..=1.0).contains(&c),
                "-inf omega_fs leaked into motor {i}: {c}"
            );
        }
        assert!(!out3.nan_failsafe);
    }

    #[test]
    fn external_zeros_prev_du_for_clean_internal_fallback() {
        // After a clean External step, prev_du must be zero so that a
        // subsequent transition to Internal starts with no model-based ω̇
        // from External-mode WLS history.
        let mut ctrl = IndiController::new(&g2_config(), LOOP_HZ);
        let a = Vector3::new(0.0, 0.0, GRAVITY);
        let rate_sp = Vector3::new(0.0, 0.0, 3.0);

        // Settle on Internal so prev_du is non-zero.
        for _ in 0..50 {
            ctrl.update_rpm(&[RpmInput::Erpm(20000); NU]);
            ctrl.step(
                &Vector3::zeros(),
                &a,
                &rate_sp,
                GRAVITY,
                true,
                &[true; NU],
                MotorState::Internal,
                V_NOM,
            )
            .0;
        }
        let prev_du_after_internal = ctrl.prev_du;
        assert!(
            prev_du_after_internal.iter().any(|&v| v.abs() > 1e-9),
            "expected prev_du to be non-zero after Internal-mode settling"
        );

        // One External step with valid inputs.
        let omega_fs = SVector::<f32, NU>::from_element(2000.0);
        let omega_dot_fs = SVector::<f32, NU>::zeros();
        ctrl.update_rpm(&[RpmInput::Erpm(20000); NU]);
        ctrl.step(
            &Vector3::zeros(),
            &a,
            &rate_sp,
            GRAVITY,
            true,
            &[true; NU],
            MotorState::External {
                omega_fs: &omega_fs,
                omega_dot_fs: &omega_dot_fs,
            },
            V_NOM,
        )
        .0;
        for (i, &v) in ctrl.prev_du.iter().enumerate() {
            assert_eq!(v, 0.0, "prev_du[{i}] should be zeroed after External step, got {v}");
        }
    }

    // ── apply_effectiveness_params ─────────────────────────────────────

    use crate::mixer::{DEFAULT_MOTOR_MAX_OMEGA_RAD_S, DEFAULT_MOTOR_TAU_S};
    use crate::params::IndiEffectivenessParams;

    /// The airframe's motors at their schema defaults — the actuator side
    /// of an apply call when nothing has been identified.
    fn stock_motors() -> [MotorParams; NU] {
        test_config().motors
    }

    /// Motors carrying a full identified dynamics set: tuned tau/omega,
    /// per-motor G2 yaw (sign follows spin), explicit nonlinearity.
    fn configured_motors() -> [MotorParams; NU] {
        let mut m = stock_motors();
        let g2_yaw = [0.002, -0.002, -0.002, 0.002];
        for (i, motor) in m.iter_mut().enumerate() {
            motor.time_const_s = 0.03;
            motor.max_omega_rad_s = 4000.0;
            motor.g2 = [0.0, 0.0, g2_yaw[i]];
            motor.nonlinearity = 0.4;
        }
        m
    }

    /// A fully-populated, valid configured G1 block distinct from geometric.
    fn configured_effectiveness() -> IndiEffectivenessParams {
        IndiEffectivenessParams {
            g1_force: [[0.0, 0.0, 14.0]; 4],
            g1_torque: [
                [-300.0, 280.0, 40.0],
                [-300.0, -280.0, -40.0],
                [300.0, 280.0, -40.0],
                [300.0, -280.0, 40.0],
            ],
        }
    }

    #[test]
    fn apply_zero_g1_block_restores_geometric() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let geometric = ctrl.effectiveness.g1;
        let p = IndiEffectivenessParams::default(); // G1 zero
        let report = ctrl.apply_effectiveness_params(&stock_motors(), &p, 0.5);
        assert_eq!(report.g1, G1Application::Geometric);
        assert!(report.g2_ok);
        assert!(report.motor_dynamics_ok.iter().all(|&ok| ok));
        assert_eq!(ctrl.effectiveness.g1, geometric, "G1 must stay geometric");
        // Motor dynamics from the airframe defaults were applied.
        let (omega, tau) = (DEFAULT_MOTOR_MAX_OMEGA_RAD_S, DEFAULT_MOTOR_TAU_S);
        for i in 0..NU {
            assert!((ctrl.effectiveness.max_omega[i] - omega).abs() < 1e-3);
            let expect_scaler = 0.5 * omega * omega / tau;
            assert!((ctrl.effectiveness.g2_scaler[i] - expect_scaler).abs() / expect_scaler < 1e-5);
        }
    }

    #[test]
    fn apply_valid_configured_g1() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let p = configured_effectiveness();
        let report = ctrl.apply_effectiveness_params(&configured_motors(), &p, 0.5);
        assert_eq!(report.g1, G1Application::Configured);
        assert!((ctrl.effectiveness.g1[(2, 0)] - 14.0).abs() < 1e-6);
        assert!((ctrl.effectiveness.g1[(3, 0)] - (-300.0)).abs() < 1e-6);
        // G2 verbatim from the airframe motors.
        assert!((ctrl.effectiveness.g2[(2, 0)] - 0.002).abs() < 1e-9);
        assert!((ctrl.effectiveness.g2[(2, 1)] - (-0.002)).abs() < 1e-9);
    }

    #[test]
    fn apply_invalid_g1_restores_geometric_even_after_configured() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let geometric = ctrl.effectiveness.g1;

        // First apply a valid configured G1...
        let p = configured_effectiveness();
        let motors = configured_motors();
        assert_eq!(
            ctrl.apply_effectiveness_params(&motors, &p, 0.5).g1,
            G1Application::Configured
        );
        assert_ne!(ctrl.effectiveness.g1, geometric);

        // ...then an invalid one: geometric must be RESTORED, not the stale
        // configured matrix kept and not zero authority.
        let mut bad = configured_effectiveness();
        bad.g1_torque[2][1] = f32::NAN;
        let report = ctrl.apply_effectiveness_params(&motors, &bad, 0.5);
        assert_eq!(report.g1, G1Application::RejectedKeptGeometric);
        assert_eq!(ctrl.effectiveness.g1, geometric);
    }

    #[test]
    fn apply_partial_config_never_zeroes_authority() {
        // The old plumbing's failure mode: tau/omega set, G1 left zero →
        // all-zero G1 applied to the allocator. Must resolve to Geometric.
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let geometric = ctrl.effectiveness.g1;
        let p = IndiEffectivenessParams::default();
        let report = ctrl.apply_effectiveness_params(&configured_motors(), &p, 0.5);
        assert_eq!(report.g1, G1Application::Geometric);
        assert_eq!(ctrl.effectiveness.g1, geometric);
        assert!(ctrl.effectiveness.g1.iter().any(|&v| v != 0.0));
        // And the motor dynamics WERE applied (per-block independence).
        assert!((ctrl.effectiveness.max_omega[0] - 4000.0).abs() < 1e-6);
    }

    #[test]
    fn apply_invalid_g2_keeps_previous() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let before = ctrl.effectiveness.g2;
        let mut motors = configured_motors();
        motors[1].g2[2] = f32::INFINITY;
        let report = ctrl.apply_effectiveness_params(&motors, &IndiEffectivenessParams::default(), 0.5);
        assert!(!report.g2_ok);
        assert_eq!(ctrl.effectiveness.g2, before);
    }

    #[test]
    fn apply_invalid_motor_dynamics_keeps_previous_for_that_motor() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let omega_before = ctrl.effectiveness.max_omega;
        let alpha_before = ctrl.pt1_alpha;
        let mut motors = configured_motors();
        motors[1].time_const_s = 0.0; // structurally invalid
        motors[2].max_omega_rad_s = f32::NAN;
        let report =
            ctrl.apply_effectiveness_params(&motors, &IndiEffectivenessParams::default(), 0.5);
        assert!(report.motor_dynamics_ok[0]);
        assert!(!report.motor_dynamics_ok[1]);
        assert!(!report.motor_dynamics_ok[2]);
        assert!(report.motor_dynamics_ok[3]);
        assert_eq!(ctrl.effectiveness.max_omega[1], omega_before[1]);
        assert_eq!(ctrl.pt1_alpha[1], alpha_before[1]);
        assert_eq!(ctrl.effectiveness.max_omega[2], omega_before[2]);
    }

    #[test]
    fn apply_pt1_alpha_recomputed_from_tau() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let mut motors = stock_motors();
        for m in motors.iter_mut() {
            m.time_const_s = 0.04;
        }
        ctrl.apply_effectiveness_params(&motors, &IndiEffectivenessParams::default(), 0.5);
        let dt = 1.0 / LOOP_HZ;
        let expect = dt / (0.04 + dt);
        for i in 0..NU {
            assert!((ctrl.pt1_alpha[i] - expect).abs() < 1e-9);
        }
    }

    #[test]
    fn apply_nonlinearity_zero_uses_fallback() {
        let mut ctrl = IndiController::new(&test_config(), LOOP_HZ);
        let mut motors = stock_motors();
        for (m, k) in motors.iter_mut().zip([0.0, 0.7, 0.0, 0.3]) {
            m.nonlinearity = k;
        }
        ctrl.apply_effectiveness_params(&motors, &IndiEffectivenessParams::default(), 0.55);
        // Compare against freshly-built linearizations: fallback for the
        // zero entries, param value otherwise. output_curve at a probe
        // point discriminates the k values.
        let probe = |ctrl: &IndiController, i: usize| ctrl.linearization[i].output_curve(0.5, V_NOM);
        let expect_k = [0.55, 0.7, 0.55, 0.3];
        for i in 0..NU {
            let reference = super::super::linearization::ThrustLinearization::new(
                expect_k[i],
                ThrustModel::Quadratic,
                8.5,
            );
            assert!(
                (probe(&ctrl, i) - reference.output_curve(0.5, V_NOM)).abs() < 1e-6,
                "motor {i} nonlinearity mismatch"
            );
        }
    }
}
