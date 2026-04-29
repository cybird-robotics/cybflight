// INDI task: pure rate controller running at IMU rate (~8 kHz).
//
// Subscribes to IMU_1 (raw gyro+accel), RATE_COMMAND (from outer loop),
// and DSHOT_TELEMETRY (motor RPM).
// Publishes ACTUATOR_MOTORS and telemetry.
//
// The outer loop (cascade, MPC, or RC rate mode) publishes rate_ref +
// collective_thrust to RATE_COMMAND. INDI tracks the rate reference using
// bias-corrected gyro and WLS motor allocation.
//
// Safety principle: when any input is stale or output is non-finite, the task
// stops publishing ACTUATOR_MOTORS (goes silent). The failsafe controller
// watchdog detects the silence and disarms — the same pattern as RC loss.

#[cfg(not(feature = "outer_mpc"))]
use cybflight_core::position_control::{self, pd_ff_control};

#[cfg(not(feature = "outer_mpc"))]
use crate::sensors::VEHICLE_ODOMETRY;

use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use air_filters::{Filter, nonlinear::slew::SlewFilter};
#[cfg(not(feature = "outer_mpc"))]
use cybflight_core::attitude_control::{self, AttitudeControlOutput, geometric_controller};
use cybflight_core::{
    indi::{
        controller::{IndiConfig, IndiController, MotorState, NU},
        effectiveness::IndiMotorParams,
        learner::{LearnedParams, Learner, LearnerConfig, LearnerInput},
        linearization::ThrustModel,
        rpm_notch::RpmNotchBank,
        rpm_tracker::RpmInput,
    },
    params::IndiEffectivenessParams,
};
use embassy_time::{Duration, Instant};
use nalgebra::{Matrix2, SMatrix, SVector, UnitQuaternion, Vector3};

use crate::estimation::rpm_estimator::{
    NormalizedThrottle as EstNormalizedThrottle, RpmEstimator, RpmEstimatorConfigBuilder,
    StateAndCov,
};

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs::{self, dshot::TelemetryValue},
    sensors::{DSHOT_TELEMETRY, IMU_1, POWER_STATUS},
    vehicle::{QUADROTOR_BODY, QUADROTOR_MOTORS},
};

/// Auto-save flag: set on disarm when learned params are committed,
/// consumed in the disarmed idle path to write to flash.
static LEARNED_SAVE_PENDING: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Thrust-to-command model used by the INDI linearization.
///
/// `Quadratic`: u = k·d² + (1−k)·d  (indiflight port; firmware default).
/// `SqrtSquared`: u = (k·d + (1−k)·√d)²  (steady-state ω mix, T ∝ ω²;
///   often fits thrust-stand data better — see `tmp/thrust_map/`).
/// `Table(...)`: 2D bench-data lookup `(thrust_N, voltage_V) → command`,
///   where `thrust_N` is *per-rotor* force. Compensates for battery sag
///   automatically. Not the default — needs bench data and validated
///   voltage telemetry first. Bench rigs that report collective thrust are
///   converted to per-rotor at build time (see `build.rs`); the runtime
///   `ThrustTable` always carries per-rotor units.
///
/// Changing this is a global decision for the airframe; the meaning of `k`
/// differs between models, so `indi_effectiveness.nonlinearity` typically
/// needs re-identification after switching.
// const THRUST_MODEL: ThrustModel = ThrustModel::Quadratic;
// const THRUST_MODEL: ThrustModel = ThrustModel::SqrtSquared;
const THRUST_MODEL: ThrustModel = ThrustModel::Table(&crate::thrust_tables::A2RL_0114);

/// Default motor nonlinearity `k`, matched to `THRUST_MODEL`.
///
/// Identified from the A2RL 6S thrust map in `tmp/thrust_map/a2rl_0114.csv`
/// by `identify_indi_k.py` (pooled fit across 21.7–24.8 V, 2500 samples):
///   Quadratic:    k = 0.518  (RMS 0.272 N, R² 0.9946)
///   SqrtSquared:  k = 0.458  (RMS 0.230 N, R² 0.9961)
///
/// Only used when no learned params exist in flash (fresh install).
/// Unused for `Table` mode — the table itself encodes the curve.
const THRUST_NONLINEARITY: f32 = match THRUST_MODEL {
    ThrustModel::Quadratic => 0.518,
    ThrustModel::SqrtSquared => 0.458,
    ThrustModel::Table(_) => 0.0,
};

/// Bootstrap pack voltage, used as the initial value of `last_voltage_v`
/// before `power_task` publishes its first frame. Matches the mid-range of
/// the bench thrust map at `tmp/thrust_map/a2rl_0114.csv` (21.7–24.8 V →
/// 23.0 V mid). Once any valid voltage frame arrives we always hold the
/// last reading rather than fall back here — a freshly-stale value tracks
/// truth far better than a fixed nominal, especially at end-of-flight
/// when sag is largest.
const NOMINAL_VOLTAGE_V: f32 = 23.0;
/// Soft staleness threshold: past this many ms without a fresh
/// `POWER_STATUS`, we still hold `last_voltage_v` (it tracks slowly under
/// heavy load) but enter the "stale" state for logging + failsafe
/// accounting. Power task publishes at 100 Hz, so 500 ms covers ~50
/// missed frames — well past any plausible scheduling hiccup.
const VOLTAGE_STALE_TIMEOUT: Duration = Duration::from_millis(500);
/// Hard failsafe in `Table` mode: if voltage stays stale this long while
/// armed, the inner loop goes silent and the watchdog disarms — same
/// pattern as `CMD_STALE_TIMEOUT`. The 500 ms NOMINAL fallback is meant to
/// ride out a transient `power_task` stall; if it persists beyond 2 s the
/// linearization is unreliable enough that flying further is more
/// dangerous than landing. Analytic models ignore voltage, so the failsafe
/// is suppressed for them.
const VOLTAGE_FAILSAFE_TIMEOUT: Duration = Duration::from_millis(2000);
/// Plausibility gate — anything outside this is treated as a glitched
/// frame and dropped. 12 V floor (4S empty) to 30 V ceiling (6S full)
/// covers every battery this airframe will see; tighten per-airframe if
/// needed.
const VOLTAGE_MIN_PLAUSIBLE: f32 = 12.0;
const VOLTAGE_MAX_PLAUSIBLE: f32 = 30.0;

/// Default INDI motor parameters.
const MAX_RPM: f32 = 40000.0;
const TIME_CONSTANT: f32 = 0.02;

const INDI_MOTOR_PARAMS: [IndiMotorParams; NU] = [
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: 0.001,
    },
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: -0.001,
    },
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: -0.001,
    },
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: 0.001,
    },
];

/// Convert `IndiEffectivenessParams` to a `LearnedParams` that the INDI
/// controller can consume. Returns `None` if all values are zero (meaning
/// "use geometric fallback").
fn learned_from_indi_params(p: &IndiEffectivenessParams) -> Option<LearnedParams> {
    // All-zero check: if nothing is configured, signal "no saved params".
    let all_zero = p
        .g1_force
        .iter()
        .flatten()
        .chain(p.g1_torque.iter().flatten())
        .chain(p.g2.iter().flatten())
        .chain(p.max_omega.iter())
        .chain(p.time_const_s.iter())
        .chain(p.nonlinearity.iter())
        .all(|&v| v == 0.0);
    if all_zero {
        return None;
    }

    let mut g1 = SMatrix::<f32, 6, NU>::zeros();
    for col in 0..NU {
        g1[(0, col)] = p.g1_force[col][0]; // fx
        g1[(1, col)] = p.g1_force[col][1]; // fy
        g1[(2, col)] = p.g1_force[col][2]; // fz
        g1[(3, col)] = p.g1_torque[col][0]; // roll
        g1[(4, col)] = p.g1_torque[col][1]; // pitch
        g1[(5, col)] = p.g1_torque[col][2]; // yaw
    }
    let mut g2 = SMatrix::<f32, 3, NU>::zeros();
    for col in 0..NU {
        g2[(0, col)] = p.g2[col][0]; // roll
        g2[(1, col)] = p.g2[col][1]; // pitch
        g2[(2, col)] = p.g2[col][2]; // yaw
    }

    Some(LearnedParams {
        g1,
        g2,
        max_omega: p.max_omega.into(),
        time_const_s: p.time_const_s.into(),
        nonlinearity: p.nonlinearity.into(),
        rate_gain: 0.0, // will be recomputed by controller if needed
        attitude_gain: 0.0,
        valid: true,
    })
}

/// Write learned parameters back into the in-memory vehicle params.
fn write_learned_to_params(learned: &LearnedParams) {
    let mut params = crate::params::get();
    let ie = &mut params.indi_effectiveness;
    for col in 0..NU {
        ie.g1_force[col][0] = learned.g1[(0, col)];
        ie.g1_force[col][1] = learned.g1[(1, col)];
        ie.g1_force[col][2] = learned.g1[(2, col)];
        ie.g1_torque[col][0] = learned.g1[(3, col)];
        ie.g1_torque[col][1] = learned.g1[(4, col)];
        ie.g1_torque[col][2] = learned.g1[(5, col)];
        ie.g2[col][0] = learned.g2[(0, col)];
        ie.g2[col][1] = learned.g2[(1, col)];
        ie.g2[col][2] = learned.g2[(2, col)];
    }
    ie.max_omega = learned.max_omega.into();
    ie.time_const_s = learned.time_const_s.into();
    ie.nonlinearity = learned.nonlinearity.into();
    crate::params::set(params);
}

/// Convert an estimated mechanical omega (rad/s) to wire-safe eRPM (u32).
///
/// Guards the saturating `f32 as u32` cast against non-finite and
/// out-of-range inputs: `+inf as u32` saturates to `u32::MAX`, which would
/// otherwise leak through telemetry as a ~4.3 billion eRPM spike. Anything
/// non-finite, non-positive, or beyond `max_omega * 1.2` (the range-gate
/// headroom used elsewhere) collapses to 0.
fn omega_to_safe_erpm(omega: f32, erpm_to_rads: f32, max_omega: f32) -> u32 {
    if omega.is_finite() && omega > 0.0 && omega <= max_omega * 1.2 {
        libm::roundf(omega / erpm_to_rads) as u32
    } else {
        0
    }
}

#[embassy_executor::task]
pub async fn indi_task() {
    // --- Load params for INDI controller and learner ---
    let params = crate::params::get();
    let ic = &params.indi_controller;
    let lp = &params.learner;

    // --- Build INDI controller ---
    let config = IndiConfig {
        rate_gains: ic.rate_gains.into(),
        sync_filter_hz: ic.sync_filter_hz,
        rate_dot_sg_window_size: 13,
        rate_dot_sg_order: 2,
        rate_dot_sg_target_rate_hz: 8000.0,
        motors: QUADROTOR_MOTORS,
        body: QUADROTOR_BODY,
        indi_motors: INDI_MOTOR_PARAMS,
        thrust_model: THRUST_MODEL,
        nonlinearity: SVector::from_element(THRUST_NONLINEARITY),
        act_limit: SVector::from_element(1.0),
        wls_wv: ic.wls_wv.into(),
        wls_wu: ic.wls_wu.into(),
        wls_cond_bound: 3.2768e8, // (1<<15) * 1e4
        wls_theta: 1e-4,
        wls_imax: 1,
        nan_limit: 20,
        rpm_invalid_limit: 50,
        rpm_all_invalid_limit: 50,
        rpm_recovery_count: 10,
        motor_pole_count: ic.motor_pole_count,
    };

    // IMU sample rate — nominal 8 kHz
    let loop_rate_hz = 8000.0f32;
    let mut indi = IndiController::new(&config, loop_rate_hz);

    // --- Per-motor RPM estimators (FOPDT EKF, one per motor) ---
    // If learned motor dynamics exist in flash, use them; otherwise fall back
    // to the hardcoded INDI_MOTOR_PARAMS defaults.
    let pole_pairs = config.motor_pole_count as f32 / 2.0;
    let erpm_to_rads = core::f32::consts::TAU * 100.0 / (pole_pairs * 60.0);
    let saved_learned = learned_from_indi_params(&params.indi_effectiveness);
    let mut rpm_estimators: [RpmEstimator; NU] = core::array::from_fn(|i| {
        let (tau_m, c_m) = if let Some(ref saved) = saved_learned {
            (saved.time_const_s[i], saved.max_omega[i])
        } else {
            (
                INDI_MOTOR_PARAMS[i].time_const_s,
                INDI_MOTOR_PARAMS[i].max_rpm / 60.0 * core::f32::consts::TAU,
            )
        };
        let est_config = RpmEstimatorConfigBuilder::new()
            .tau_m_up(tau_m)
            .tau_m_down(tau_m)
            .build()
            .unwrap();
        let init_state = StateAndCov::new(0.0, c_m, Matrix2::new(1000.0, 0.0, 0.0, c_m * c_m));
        RpmEstimator::new(est_config, init_state)
    });

    // --- G1/G2 online learner (optional, activated via RC switch) ---
    let learner_config = LearnerConfig {
        fx_filt_hz: lp.fx_filt_hz,
        motor_filt_hz: lp.motor_filt_hz,
        acc_offset_m: lp.acc_offset_m,
        rls_gamma: lp.rls_gamma,
        rls_t_char_s: lp.rls_t_char_s,
        zeta_rate: lp.zeta_rate,
        zeta_attitude: lp.zeta_attitude,
    };
    let mut learner = Learner::new(&learner_config, loop_rate_hz);

    // Load previously saved INDI effectiveness from vehicle params (if non-zero).
    if let Some(ref saved) = saved_learned
        && indi.apply_learned_params(saved)
    {
        defmt::info!("INDI: loaded learned G1/G2 from params");
    }

    // Track whether we need to save learned params on disarm
    let mut was_armed = false;

    // --- Learner prearm state ---
    let mut learner_prearm_latched: bool = false;
    // Per-motor sample-and-hold for raw eRPM (rad/s) — used when KF is off
    let mut raw_omega_hold = SVector::<f32, NU>::zeros();

    // --- Slew outlier filter (always on, protects both KF and raw path) ---
    //
    // Per-motor `SlewFilter` with a fixed max per-sample delta (ZOH on reject). The delta is
    // derived from a physics rate bound × a worst- case inter-sample interval, NOT a live dt lookup
    // — `SlewFilter` is intentionally time-unaware, so we pre-size the gate for the slowest
    // tolerable telemetry cadence and accept that at nominal rates the gate is loose by that same
    // ratio.
    let default_max_omega = INDI_MOTOR_PARAMS[0].max_rpm / 60.0 * core::f32::consts::TAU;
    let max_omega_bound: f32 = saved_learned
        .as_ref()
        .map(|s| s.max_omega.iter().cloned().fold(0.0f32, f32::max))
        .filter(|&v| v > 0.0)
        .unwrap_or(default_max_omega);
    // Physics bound: max plausible dω/dt for a first-order motor with
    // time constant TAU_MIN_S driven toward max_omega_bound.
    const TAU_MIN_S: f32 = 0.005;
    // Worst-case interval between valid per-motor telemetry frames. DShot
    // bidir updates each motor at roughly command_rate / 4 (~2 kHz at
    // 8 kHz commands); 1 ms is ~2× nominal — enough headroom to absorb
    // single-frame dropouts without opening the gate to real outliers.
    const WORST_DT_S: f32 = 0.001;
    let slew_max_delta: f32 = (max_omega_bound / TAU_MIN_S) * WORST_DT_S;
    let mut slew_filters: [SlewFilter<f32>; NU] =
        core::array::from_fn(|_| SlewFilter::new(slew_max_delta).unwrap());
    // Local param version — re-read params when global version changes.
    let mut local_param_ver =
        crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
    // --- Subscribe to channels ---
    let mut imu_sub = IMU_1.subscriber().unwrap();
    let mut dshot_sub = DSHOT_TELEMETRY.subscriber().unwrap();
    // Battery voltage from `power_task` (100 Hz). Consumed by the thrust
    // map in `Table` mode; ignored by the analytic models. Held between
    // updates with a staleness fallback to nominal — see VOLTAGE_* consts.
    let mut power_sub = POWER_STATUS.subscriber().unwrap();
    // Armed state read from IS_ARMED atomic (set by DShot task).
    let att_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
    let motor_telem_pub = super::ACTUATOR_MOTORS_TELEM.immediate_publisher();
    let processed_dshot_pub = super::PROCESSED_DSHOT_TELEM.immediate_publisher();
    let processed_motor_pub = super::PROCESSED_MOTOR_STATE.immediate_publisher();

    // --- State ---
    // Rate command from outer loop (cascade, MPC, or RC rate mode).
    let mut collective_thrust_n: f32 = 0.0;
    let mut rate_ref = Vector3::<f32>::zeros();
    let mut spf_sp_z: f32 = 0.0; // thrust / mass in body z
    let mut telem_attitude = UnitQuaternion::<f32>::identity();

    // --- Rate command tracking ---
    // Cache the latest RATE_COMMAND from the outer loop (cascade, MPC, or
    // RC rate mode) plus its arrival time for the staleness gate.
    let mut last_cmd_time: Option<Instant> = None;
    /// Failsafe timeout on the rate command Signal. After this many ms
    /// without a fresh command from the outer loop, the inner loop goes
    /// silent and the watchdog disarms.
    const CMD_STALE_TIMEOUT: Duration = Duration::from_millis(100);

    // --- Battery voltage tracking (for Table thrust model) ---
    // Hold-last-value with staleness/plausibility fallback to nominal.
    // power_task runs at 100 Hz; INDI runs at 8 kHz, so 99% of ticks
    // reuse the held value — that's expected, not a problem.
    let mut last_voltage_v: f32 = NOMINAL_VOLTAGE_V;
    let mut last_voltage_time: Option<Instant> = None;
    // Tracks the start of the current voltage-staleness episode for
    // one-shot stale/recover logging and the Table-mode armed failsafe
    // gate. `Some(t)` ⇒ stale since `t`; `None` ⇒ fresh.
    let mut voltage_stale_since: Option<Instant> = None;

    // RPM estimator timestamp tracking (seconds, f32 relative to task start)
    let mut est_prev_ts: Option<Instant> = None;
    let mut est_current_ts: f32 = 0.0;

    // Decimation counter for position/attitude controller
    let mut outer_counter: u32 = 0;
    // Position controller runs at IMU_rate / OUTER_DECIMATION
    // 8000 / 80 = 100 Hz (matching the old inner_loop rate)
    const OUTER_DECIMATION: u32 = 80;

    // INDI starts immediately. Gyro bias begins at zero and improves as
    // Mahony's ki integral converges (~1–2 s). Rate control runs from
    // the first IMU tick; the outer loop's first RATE_COMMAND activates
    // motor output.
    defmt::info!("INDI task started ({}Hz)", loop_rate_hz as u32);

    // ── Airframe ↔ thrust-table binding check ──────────────────────────
    //
    // `THRUST_MODEL = Table(...)` requires the table's per-rotor thrust
    // axis to match the airframe's per-motor `max_thrust_n`; otherwise the
    // u-axis scaling is silently wrong and the drone gets a fraction of
    // the throttle authority WLS thinks it has. Hard-fail at startup
    // rather than fly with the mismatch — `defmt::panic!` halts the
    // firmware before motors arm. Voltage range is logged so an operator
    // can sanity-check the active table against the pack in use.
    if let ThrustModel::Table(t) = THRUST_MODEL {
        let pm = QUADROTOR_MOTORS[0].max_thrust_n;
        let tm = t.thrust_max_n();
        defmt::info!(
            "INDI Table: thrust [{}, {}] N/rotor, voltage [{}, {}] V, per_motor_max={} N",
            t.thrust_min_n(),
            tm,
            t.voltage_min_v(),
            t.voltage_max_v(),
            pm,
        );
        // 10% relative tolerance: the bench rig's per-rotor max and the
        // configured `max_thrust_n` should agree to well within this; any
        // larger gap means the table was baked from a different airframe.
        let rel_err = libm::fabsf(pm - tm) / tm;
        if rel_err >= 0.10 {
            defmt::panic!(
                "INDI: Table thrust_max ({} N/rotor) and vehicle.rs max_thrust_n ({} N) disagree by {}% — re-bake the table or fix vehicle.rs",
                tm,
                pm,
                rel_err * 100.0,
            );
        }
    }

    /****************************/
    let motor_filter_hz = 15.0;
    let make_biquad = || {
        let cfg = BiquadFilterConfigBuilder::direct_form_2()
            .sample_frequency_hz(loop_rate_hz)
            .filter_type(BiquadFilterType::LowPass)
            .cutoff_frequency_hz(motor_filter_hz)
            .build()
            .expect("indi: biquad filter config invalid");
        BiquadFilter::new(cfg)
    };
    let mut motor_omega_filter: [BiquadFilter<f32, DirectForm2<f32>>; NU] =
        core::array::from_fn(|_| make_biquad());

    let mut omega_fs = SVector::<f32, NU>::zeros();
    let mut omega_dot_fs = SVector::<f32, NU>::zeros();
    // True ZOH on the *input*: on a missed telemetry tick we feed the last
    // valid raw omega, not the filter output. Feeding the output back forms
    // a feedback loop that is only marginally stable (pole at z=1), so the
    // filter would drift under sustained telemetry loss.
    let mut last_y_meas_hold = SVector::<f32, NU>::zeros();
    let mut omega_fs_has_prev = false;
    /****************************/

    // ── RPM-tracking notch filters on gyro and accel ────────────────────
    //
    // Each motor's known rotational frequency drives a cascade of biquad
    // notches placed on the IMU signal *just before* INDI consumes it.
    // Suppresses the narrow-band vibration the motors inject into gyro &
    // accel, which would otherwise close a positive-feedback loop through
    // INDI's high-bandwidth rate path (motor → vibration → gyro → motor)
    // and force `sync_filter_hz` to stay too low to track aggressive
    // trajectories.
    //
    // Defaults match Betaflight's `rpm_filter` defaults; promote to
    // `params.rs` only if tuning data argues for it. NU=4, NH_GYRO=3,
    // NH_ACCEL=1 mirrors `tmp/indi_c/rpm_filter.c` and `acceleration.c`.
    //
    // Safety: the bank fades to passthrough when motor freq < min_hz or
    // is non-finite (see `RpmNotchBank::update`), so a dshot dropout or
    // disarmed state never injects NaN or stale notches into the IMU
    // signal — matches the silence-as-failure protocol in
    // docs/safety_protocol.md (this stage doesn't *go silent* itself; it
    // just degrades gracefully, leaving the upstream/downstream silence
    // signals untouched).
    //
    // Test/A-B flag: flip ENABLE_RPM_NOTCH to disable the entire RPM-notch
    // path (PT1 motor-freq tracker, bank update, bank apply, per-arm reset).
    // When false, `gyro_corrected` / `accel_corrected` pass straight through
    // from the IMU as before. The bank storage and PT1 state are still
    // allocated (~10 KB BSS) but the per-loop work is dead-code eliminated
    // by the compiler since this is a `const bool`. Recompile + reflash to
    // toggle; matches the `THRUST_MODEL` const pattern used above.
    const ENABLE_RPM_NOTCH: bool = false;
    const RPM_NOTCH_Q: f32 = 5.0; // 2.0
    const RPM_NOTCH_MIN_HZ: f32 = 100.0; // 200
    const RPM_NOTCH_FADE_HZ: f32 = 50.0;
    let mut gyro_rpm_notch = RpmNotchBank::<NU, 3>::new(
        loop_rate_hz,
        RPM_NOTCH_Q,
        RPM_NOTCH_MIN_HZ,
        RPM_NOTCH_FADE_HZ,
    );
    let mut accel_rpm_notch = RpmNotchBank::<NU, 1>::new(
        loop_rate_hz,
        RPM_NOTCH_Q,
        RPM_NOTCH_MIN_HZ,
        RPM_NOTCH_FADE_HZ,
    );
    // Per-motor rotational frequency in Hz, fed into both notch banks
    // each loop after passing through the dedicated PT1 below.
    let mut motor_freq_hz = [0.0f32; NU];
    /// Rad/s → Hz: divide by 2π. Pre-computed as a multiply for speed.
    const RAD_S_TO_HZ: f32 = 0.5 * core::f32::consts::FRAC_1_PI;

    // ── RPM-notch frequency tracker (separate from `motor_omega_filter`) ──
    //
    // Mirrors Indiflight's `motorFreqLpf` in `tmp/indi_c/rpm_filter.c:75`,
    // a 1st-order PT1 dedicated to *notch frequency tracking*. It runs in
    // PARALLEL with `motor_omega_filter` (the 15 Hz biquad that feeds INDI's
    // sync-required `omega_fs` / `omega_dot_fs`). The two filters have
    // different consumers, different lag/noise tradeoffs, and so different
    // cutoffs:
    //
    //   * INDI sync filter @ 15 Hz: matches the delay of `rate_dot_fs`,
    //     `spf_fs`, `u_state_fs` so `dv = sp − fs` is computed at a
    //     consistent time.
    //   * Notch frequency filter @ MOTOR_FREQ_LPF_HZ: needs to track motor
    //     1P during throttle transients without lagging more than the notch
    //     half-width (~motor_freq / (2·Q)). Reusing the 15 Hz output here
    //     would smear the notch off the motor harmonic for ~10–15 ms after
    //     every throttle change, defeating the whole point of the notch.
    //
    // 150 Hz matches Betaflight's upstream `rpm_filter_lpf_hz` default.
    // Hardcoded; promote to params if tuning ever needs it.
    const MOTOR_FREQ_LPF_HZ: f32 = 150.0;
    let motor_freq_pt1_alpha: f32 = {
        let dt = 1.0 / loop_rate_hz;
        let tau = 1.0 / (2.0 * core::f32::consts::PI * MOTOR_FREQ_LPF_HZ);
        dt / (tau + dt)
    };
    let mut motor_freq_lpf_state = [0.0f32; NU];
    // Mirrors `omega_fs_has_prev`: the first sample seeds the PT1 directly
    // instead of running a step from zero, avoiding a startup transient
    // that would tilt the first ~τ ms of notch tracking.
    let mut motor_freq_lpf_has_prev = false;

    loop {
        // 1. Await IMU sample — this drives the loop at ~8 kHz.
        let imu = imu_sub.next_message_pure().await;
        let est_dt = if let Some(prev) = est_prev_ts {
            imu.timestamp.duration_since(prev).as_micros() as f32 / 1_000_000.0
        } else {
            1.0 / loop_rate_hz
        };
        est_prev_ts = Some(imu.timestamp);
        est_current_ts += est_dt;

        // INDI runs on raw IMU without gyro/accel bias correction.
        // Rationale: ESKF bias degrades under external-sensor loss (mocap/
        // GPS), and Mahony has been removed. The outer loop (cascade /
        // MPC / RC rate mode) compensates for steady-state bias via its
        // attitude/position feedback — any constant gyro offset shows up
        // as a small trim on rate_ref and is absorbed naturally.
        let gyro_corrected = imu.gyro_rad_s;
        let accel_corrected = imu.accel_m_s2;

        // 2. Non-blocking reads of other channels.
        // Arming state — read from atomic (set by DShot task, single source of truth)
        let armed = crate::motors::IS_ARMED.load(core::sync::atomic::Ordering::Acquire);

        // ── Arm/disarm transitions ──────────────────────────────────────
        //
        // On ARM: latch learner prearm, reset KF state for fresh convergence.
        //   If prearm latched: reset to geometric G1, zero G2, reset learner.
        //
        // On DISARM: if learner was active, commit learned params → write to
        //   VehicleParams → signal flash auto-save. PARAM_VERSION re-read
        //   (below) handles applying to INDI + KF on next iteration.
        if !was_armed && armed {
            // ARM transition; true if the PRE_ARM switch is HIGH/DOWN
            learner_prearm_latched =
                super::LEARNER_PREARM.load(core::sync::atomic::Ordering::Acquire);

            // Always reset KF state — fresh start every flight
            for est in rpm_estimators.iter_mut() {
                est.reset_state();
            }

            // Always re-seed the slew filters to 0 so re-arm doesn't inherit stale omega from
            // the previous flight. Motors are at rest at arm time, so 0 is the correct prior.
            slew_filters.reset([0.0; NU]).unwrap();

            // Reset motor-omega LPF, ZOH hold, and derivative state so re-arm
            // starts fresh at 0. (The biquad has no in-place reset API; replace
            // the array with freshly-constructed filters.)
            motor_omega_filter = core::array::from_fn(|_| make_biquad());
            last_y_meas_hold = SVector::zeros();
            omega_fs = SVector::zeros();
            omega_dot_fs = SVector::zeros();
            omega_fs_has_prev = false;

            if ENABLE_RPM_NOTCH {
                // Clear RPM-notch delay lines and weights so the previous
                // flight's coefficient state doesn't bleed into this one.
                gyro_rpm_notch.reset();
                accel_rpm_notch.reset();

                // Re-seed the notch frequency tracker on the next valid
                // dshot frame so a re-arm starts from the current motor
                // state, not from a stale frequency belonging to the
                // previous flight.
                motor_freq_lpf_state = [0.0; NU];
                motor_freq_lpf_has_prev = false;
            }

            if learner_prearm_latched {
                // Learner prearm: throw away any previously-learned G1/G2 so
                // this learning flight starts from the analytic geometric
                // model. The unstable-prearm bug was not in this reset — it
                // was in the motor-state source (see `motor_state` below).
                indi.reset_to_geometric(&QUADROTOR_MOTORS, &QUADROTOR_BODY, &INDI_MOTOR_PARAMS);
                learner.reset();
                raw_omega_hold = SVector::zeros();
                defmt::info!("INDI: learner prearm LATCHED — effectiveness reset to geometric");
            }
        }
        if was_armed && !armed {
            // DISARM transition
            if learner_prearm_latched && learner.samples() > 0 {
                let snap = learner.update(&LearnerInput {
                    rate_rad_s: gyro_corrected,
                    rate_dot_rad_s2: nalgebra::Vector3::zeros(),
                    spf_m_s2: accel_corrected,
                    omega_rad_s: SVector::zeros(),
                    d_commands: SVector::zeros(),
                    armed: false,
                    touching_ground: true,
                });
                if snap.valid {
                    write_learned_to_params(&snap);
                    LEARNED_SAVE_PENDING.store(true, core::sync::atomic::Ordering::Release);
                    defmt::info!("INDI: learned G1/G2 committed — auto-saving to flash");
                }
            }
            learner_prearm_latched = false;
        }
        was_armed = armed;

        // ── DShot telemetry + slew outlier filter ──────────────────────
        //
        // Decode eRPM → rad/s, then gate on a fixed per-sample delta to
        // reject GCR decode errors (ZOH on reject, no interpolation).
        // Always on — protects both the KF path and the raw-hold path.
        let mut y_meas: [Option<f32>; NU] = [None; NU];
        if let Some(telem) = dshot_sub.try_next_message_pure() {
            for i in 0..NU {
                let raw_omega = match telem.motors[i].value {
                    TelemetryValue::Erpm(erpm) => Some(erpm as f32 * erpm_to_rads),
                    TelemetryValue::Stopped => Some(0.0),
                    TelemetryValue::Invalid | TelemetryValue::Edt(_) => None,
                };

                if let Some(omega) = raw_omega {
                    // Hard range gate
                    if omega < 0.0 || omega > max_omega_bound * 1.5 {
                        continue;
                    }
                    // if omega < 0.0 {
                    //     continue;
                    // }

                    // SlewFilter returns `input` on accept and the held state on reject; equality
                    // to input uniquely identifies the accept branch (an equal state can only occur
                    // with a zero-delta input, which also accepts).
                    let out = slew_filters[i].apply(omega);
                    if out == omega {
                        y_meas[i] = Some(omega);
                    }
                }
            }
        }

        // ── Conditional KF / raw-hold ────────────────────────────────
        //
        // KF only runs when armed AND not in learner-prearm mode.
        // Disarmed: KF idle (avoids divergence without corrections).
        // Learner prearm: raw eRPM with sample-and-hold (no KF).
        let g2_valid = if armed && !learner_prearm_latched {
            // Normal flight: run KF, feed smoothed omega to RpmTracker
            for i in 0..NU {
                rpm_estimators[i].step(est_current_ts, est_dt, y_meas[i]);
            }
            let estimated_inputs: [RpmInput; NU] = core::array::from_fn(|i| {
                let erpm = omega_to_safe_erpm(
                    rpm_estimators[i].state().omega(),
                    erpm_to_rads,
                    max_omega_bound,
                );
                RpmInput::Erpm(erpm)
            });
            let (valid, _rpm_failsafe) = indi.update_rpm(&estimated_inputs);
            valid
        } else if armed && learner_prearm_latched {
            // Learner prearm: KF off, sample-and-hold raw eRPM
            for i in 0..NU {
                if let Some(y) = y_meas[i] {
                    raw_omega_hold[i] = y;
                }
            }
            [false; NU]
        } else {
            // Disarmed: KF idle, G2 inactive
            [false; NU]
        };

        // Step the per-motor LPF every IMU tick. On a missed telemetry sample
        // (y_meas[i] == None) hold the last valid *input* (true ZOH), not the
        // filter output — the filter then smoothly settles to that constant.
        // omega_dot_fs is a finite difference of the filter output, scaled by
        // the loop rate. Skip the first iteration to avoid a startup spike.
        for i in 0..NU {
            let x = match y_meas[i] {
                Some(y) => {
                    last_y_meas_hold[i] = y;
                    y
                }
                None => last_y_meas_hold[i],
            };
            let new_fs = motor_omega_filter[i].apply(x);
            omega_dot_fs[i] = if omega_fs_has_prev {
                (new_fs - omega_fs[i]) * loop_rate_hz
            } else {
                0.0
            };
            omega_fs[i] = new_fs;
        }
        omega_fs_has_prev = true;

        // ── RPM-tracking notches on gyro + accel ───────────────────────
        //
        // Gated by `ENABLE_RPM_NOTCH` (compile-time const). When false,
        // every line in this block is dead-code eliminated and
        // `gyro_corrected` / `accel_corrected` keep the values they had
        // out of the IMU sub block above. Use this for A/B comparison
        // flights against the no-notch baseline.
        //
        // When true:
        //   1. Update the per-motor frequency tracker (PT1 @ 150 Hz) with
        //      the freshest *raw* ω available — `y_meas` is the post-slew,
        //      pre-KF dshot value, the analogue of Indiflight's
        //      `getDshotTelemetry()` tap (rpm_filter.c:116). Feeding the
        //      PT1 from `omega_fs` (15 Hz biquad) instead would lag notch
        //      tracking by ~10–15 ms during throttle transients and let
        //      the motor 1P walk out from under the notch.
        //   2. Refresh notch coefficients (round-robin batched, full bank
        //      in ~1 ms) and apply the cascade. Shadow `gyro_corrected`
        //      and `accel_corrected` so the rest of the loop (INDI step,
        //      learner) consumes the notched signals.
        //
        // ZOH on missed dshot frames: hold the previous filter state, do
        // not push a stale or non-finite value. Per docs/safety_protocol.md
        // rule 3, this stage degrades gracefully and never injects NaN
        // into the IMU signal.
        let (gyro_corrected, accel_corrected) = if ENABLE_RPM_NOTCH {
            for i in 0..NU {
                let raw_hz = match y_meas[i] {
                    Some(omega_rad_s) if omega_rad_s.is_finite() => omega_rad_s * RAD_S_TO_HZ,
                    _ => motor_freq_lpf_state[i], // ZOH (no fresh input)
                };
                if motor_freq_lpf_has_prev {
                    motor_freq_lpf_state[i] +=
                        motor_freq_pt1_alpha * (raw_hz - motor_freq_lpf_state[i]);
                } else if y_meas[i].is_some() && raw_hz.is_finite() {
                    // Seed on the first valid frame to avoid a startup
                    // ramp that would tilt notch tracking for ~τ ms.
                    motor_freq_lpf_state[i] = raw_hz;
                }
                motor_freq_hz[i] = motor_freq_lpf_state[i];
            }
            // Latch only after at least one motor saw a fresh frame —
            // keeps the seed-on-first-valid path active until telemetry
            // actually arrives.
            if !motor_freq_lpf_has_prev && y_meas.iter().any(|s| s.is_some()) {
                motor_freq_lpf_has_prev = true;
            }

            // Disarmed → motor_freq_hz < min_hz → notches fade to
            // passthrough; no explicit armed gate needed.
            gyro_rpm_notch.update(&motor_freq_hz);
            accel_rpm_notch.update(&motor_freq_hz);
            (
                gyro_rpm_notch.apply_xyz(gyro_corrected),
                accel_rpm_notch.apply_xyz(accel_corrected),
            )
        } else {
            (gyro_corrected, accel_corrected)
        };

        // 3. Drain latest battery voltage. Plausibility-gate at the
        //    boundary so a glitched ADC frame can't poison the thrust
        //    map for the rest of the flight.
        if let Some(power) = power_sub.try_next_message_pure() {
            let v = power.voltage_cv as f32 * 0.01; // centivolts → volts
            if v.is_finite() && (VOLTAGE_MIN_PLAUSIBLE..=VOLTAGE_MAX_PLAUSIBLE).contains(&v) {
                last_voltage_v = v;
                last_voltage_time = Some(power.timestamp);
            }
            // Implausible/NaN frames silently drop. Repeated drops trip
            // the staleness fallback below.
        }

        // 4. Drain latest rate command from outer loop.
        let now = Instant::now();
        outer_counter += 1;
        if outer_counter >= OUTER_DECIMATION {
            outer_counter = 0;
        }
        if let Some(cmd) = super::RATE_COMMAND.try_take() {
            rate_ref = cmd.body_rate_rad_s;
            collective_thrust_n = cmd.collective_thrust_n;
            spf_sp_z = collective_thrust_n / QUADROTOR_BODY.mass_kg;
            telem_attitude = cmd.attitude_quaternion;
            last_cmd_time = Some(now);
        }

        // 5. Stale command while armed — go silent, let watchdog handle it.
        let cmd_fresh = match last_cmd_time {
            Some(t) => now.duration_since(t) < CMD_STALE_TIMEOUT,
            None => false,
        };
        if armed && !cmd_fresh {
            continue;
        }

        // 5b. Auto-save learned params to flash when disarmed.
        //     The flash erase stalls the CPU for ~1-2s — safe here because
        //     motors are stopped and no control is needed.
        if !armed
            && LEARNED_SAVE_PENDING
                .compare_exchange(
                    true,
                    false,
                    core::sync::atomic::Ordering::AcqRel,
                    core::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
        {
            match crate::params::save_to_flash() {
                Ok(()) => defmt::info!("INDI: auto-saved learned params to flash"),
                Err(e) => defmt::warn!("INDI: flash save failed: {}", e),
            }
            // save_to_flash() bumps PARAM_VERSION — the re-read below will
            // pick it up on the next iteration.
        }

        // 5c. Re-read params when version changes (disarmed only).
        //     Catches: learning auto-save, shell `param set` + `param save`,
        //     `param defaults`, or any other param writer.
        //     Updates INDI effectiveness AND KF motor dynamics from the same
        //     persistent params — single source of truth.
        if !armed {
            let current_ver =
                crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
            if current_ver != local_param_ver {
                local_param_ver = current_ver;
                if let Some(saved) =
                    learned_from_indi_params(&crate::params::get().indi_effectiveness)
                {
                    indi.apply_learned_params(&saved);
                    for (i, est) in rpm_estimators.iter_mut().enumerate() {
                        est.reconfigure(saved.time_const_s[i], saved.max_omega[i]);
                    }
                }
                defmt::info!("INDI: params reloaded (ver {})", current_ver);
            }
        }

        // 7. INDI step (8 kHz) — uses bias-corrected gyro. Voltage feeds
        //    the `Table` thrust model; analytic models ignore it. Single
        //    source of truth: never sample VBAT here — power_task owns it.
        //
        //    Voltage-staleness machine:
        //      - Always hold `last_voltage_v` (battery sag is slow vs. the
        //        soft staleness window; at boot the field is initialized to
        //        NOMINAL_VOLTAGE_V, which carries us until power_task's
        //        first frame).
        //      - Log warn on stale entry, info on recovery — once per
        //        episode so the defmt log isn't spammed at 8 kHz.
        //      - In Table mode, if stale persists past
        //        VOLTAGE_FAILSAFE_TIMEOUT while armed, go silent and let
        //        the controller watchdog disarm. The analytic models
        //        ignore voltage so this gate is suppressed for them; the
        //        failsafe also bounds how long a frozen reading can lie.
        let voltage_fresh = match last_voltage_time {
            Some(t) => now.duration_since(t) < VOLTAGE_STALE_TIMEOUT,
            None => false,
        };
        let voltage_v = last_voltage_v;
        if voltage_fresh {
            if let Some(t0) = voltage_stale_since.take() {
                defmt::info!(
                    "INDI: voltage recovered after {}ms",
                    now.duration_since(t0).as_millis() as u32,
                );
            }
        } else if voltage_stale_since.is_none() {
            voltage_stale_since = Some(now);
            defmt::warn!(
                "INDI: voltage stale, holding last reading {}V",
                last_voltage_v,
            );
        }
        if armed
            && matches!(THRUST_MODEL, ThrustModel::Table(_))
            && voltage_stale_since
                .map(|t0| now.duration_since(t0) >= VOLTAGE_FAILSAFE_TIMEOUT)
                .unwrap_or(false)
        {
            continue;
        }
        //
        // Motor-state source:
        //   - Armed + LPF has a sample → feed dshot-derived ω, ω̇ from the
        //     task-level biquad + finite difference. Used in both normal and
        //     prearm flights: the LPF runs unconditionally so External data
        //     is always available, and the Internal du-based fallback reads
        //     `prev_du` / `prev_omega_fs` which are not zeroed by the arm
        //     transition — feeding stale values into `effectiveness.g2 *
        //     omega_dot_fs` inside `step`'s `dv` and destabilizing the
        //     prearm flight on lift-off.
        //   - Disarmed or first iteration: fall back to Internal.
        let motor_state = if armed && omega_fs_has_prev {
            MotorState::External {
                omega_fs: &omega_fs,
                omega_dot_fs: &omega_dot_fs,
            }
        } else {
            MotorState::Internal
        };
        let (output, step_state) = indi.step(
            &gyro_corrected,
            &accel_corrected,
            &rate_ref,
            spf_sp_z,
            armed,
            &g2_valid,
            motor_state,
            voltage_v,
        );

        // 6b. Online learner.
        //     Omega source depends on mode:
        //       - Learner prearm: raw ESC eRPM (sample-and-hold), no KF
        //       - Normal: KF-smoothed omega (existing)
        //     RLS only updates when: prearm latched + toggle on + armed.
        //     Toggle alone without prearm does nothing.
        let omega_for_learner = if learner_prearm_latched {
            raw_omega_hold
        } else {
            SVector::from_fn(|i, _| rpm_estimators[i].state().omega())
        };
        let learning_on = super::LEARNING_ENABLED.load(core::sync::atomic::Ordering::Acquire);
        let learn_active = learner_prearm_latched && learning_on && armed;
        let learner_input = LearnerInput {
            rate_rad_s: gyro_corrected,
            rate_dot_rad_s2: step_state.rate_dot_raw,
            spf_m_s2: accel_corrected,
            omega_rad_s: omega_for_learner,
            d_commands: output.motor_commands,
            armed: learn_active,
            touching_ground: step_state.touching_ground,
        };
        let _learned = learner.update(&learner_input);

        // 7. Non-finite guard — skip publishing, stay alive.
        //    Transient NaN from WLS is recoverable; sustained NaN causes
        //    the watchdog heartbeat to stop → failsafe disarm.
        if !output.motor_commands.iter().all(|v| v.is_finite()) {
            continue;
        }

        // 8. Publish motor commands + watchdog heartbeat.
        //    Heartbeat is ONLY updated when a valid command is published.
        //    Any failure path above that hits `continue` goes silent.
        let motor_commands = [
            msgs::NormalizedThrottle::new_saturating(output.motor_commands[0]),
            msgs::NormalizedThrottle::new_saturating(output.motor_commands[1]),
            msgs::NormalizedThrottle::new_saturating(output.motor_commands[2]),
            msgs::NormalizedThrottle::new_saturating(output.motor_commands[3]),
        ];
        let publish_time = Instant::now();
        ACTUATOR_MOTORS.signal(msgs::ActuatorMotors {
            timestamp: publish_time,
            motor_commands,
        });
        super::LAST_CONTROLLER_PUBLISH.lock(|c| c.set(Some(publish_time)));

        // Feed this frame's throttle commands into the estimators so the
        // FOPDT model can account for transport delay on the next decode.
        // Only when KF is active (armed + not learner prearm).
        if armed && !learner_prearm_latched {
            for (i, est) in rpm_estimators.iter_mut().enumerate() {
                est.push_throttle(
                    est_current_ts,
                    EstNormalizedThrottle::new_clamped(output.motor_commands[i]),
                );
            }
        }

        // 9. Publish telemetry (at reduced rate — every OUTER_DECIMATION frames).
        if outer_counter == 1 {
            att_pub.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: publish_time,
                collective_thrust_n,
                attitude_quaternion: telem_attitude,
                body_rate_rad_s: rate_ref,
                torque_n_m: Vector3::zeros(), // INDI doesn't compute explicit torque
            });
            motor_telem_pub.publish_immediate(msgs::ActuatorMotors {
                timestamp: publish_time,
                motor_commands,
            });
            // Processed motor RPM: post-slew (learner prearm) or post-KF (normal).
            let processed_motors: [msgs::DshotMotorTelemetry; 4] = core::array::from_fn(|i| {
                let omega = if armed && learner_prearm_latched {
                    raw_omega_hold[i]
                } else if armed {
                    rpm_estimators[i].state().omega()
                } else {
                    0.0
                };
                let erpm = omega_to_safe_erpm(omega, erpm_to_rads, max_omega_bound);
                msgs::DshotMotorTelemetry {
                    value: if erpm > 0 {
                        TelemetryValue::Erpm(erpm)
                    } else {
                        TelemetryValue::Stopped
                    },
                    raw: None,
                }
            });
            processed_dshot_pub.publish_immediate(msgs::DshotTelemetry {
                timestamp: publish_time,
                motors: processed_motors,
            });

            let processed_motor_dynamics: [msgs::MotorDynamicsTelemetry; 4] =
                core::array::from_fn(|i| msgs::MotorDynamicsTelemetry {
                    omega: omega_fs[i],
                    omega_dot: omega_dot_fs[i],
                    raw: y_meas[i],
                });
            processed_motor_pub.publish_immediate(msgs::MotorStateTelemetry {
                timestamp: publish_time,
                motors: processed_motor_dynamics,
            });
        }
    }
}
