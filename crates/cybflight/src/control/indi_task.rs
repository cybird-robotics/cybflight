// INDI task: pure rate controller running at the IMU rate divided by the
// `indi_ctrl_div` vehicle param (`rates::IMU_ODR_HZ` / div: 8 kHz / 4 =
// 2 kHz on 8 kHz vehicles, 1 kHz / 1 on `imu_1khz` builds). The IMU keeps
// its full ODR; the controller steps on every div-th sample.
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

use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use air_filters::{Filter, nonlinear::slew::SlewFilter};
use cybflight_core::{
    indi::{
        controller::{
            EffectivenessApplyReport, G1Application, IndiConfig, IndiController, MotorState, NU,
        },
        effectiveness::IndiMotorParams,
        linearization::ThrustModel,
        rpm_notch::RpmNotchBank,
        rpm_tracker::RpmInput,
    },
    mixer::{DEFAULT_MOTOR_MAX_OMEGA_RAD_S, DEFAULT_MOTOR_TAU_S},
};
use embassy_time::{Duration, Instant};
use nalgebra::{Matrix2, SVector, UnitQuaternion, Vector3};

use crate::estimation::rpm_estimator::{
    NormalizedThrottle as EstNormalizedThrottle, RpmEstimator, RpmEstimatorConfigBuilder,
    StateAndCov, StepOutcome,
};

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs::{self, dshot::TelemetryValue},
    sensors::{DSHOT_TELEMETRY, IMU_1, POWER_STATUS},
};

// Thrust-to-command model used by the INDI linearization.
//
// Declared per-vehicle in `vehicles/<VEHICLE>.yaml` (`airframe.thrust_model`,
// REQUIRED — same "no default mass" rule as the rest of the identity) and
// resolved at build time to `crate::vehicle::BAKED_THRUST_MODEL` +
// `BAKED_THRUST_NONLINEARITY`:
//
// `quadratic`: u = k·d² + (1−k)·d  (indiflight port).
// `sqrt_squared`: u = (k·d + (1−k)·√d)²  (steady-state ω mix, T ∝ ω²;
//   often fits thrust-stand data better — see `tmp/thrust_map/`).
// `table`: 2D bench-data lookup `(thrust_N, voltage_V) → command`, where
//   `thrust_N` is *per-rotor* force. Compensates for battery sag
//   automatically. Needs bench data (`data/thrust_tables/*.csv`) and
//   validated voltage telemetry. Bench rigs that report collective thrust
//   are converted to per-rotor at build time (see `build.rs`); the runtime
//   `ThrustTable` always carries per-rotor units.
//
// Changing the model is a global decision for the airframe; the meaning of
// `k` differs between models, so `indi_nonlin_m*` typically needs
// re-identification after switching. (A2RL 6S map fits, for reference:
// Quadratic k = 0.518, SqrtSquared k = 0.458 — see `identify_indi_k.py`.)

// Battery facts (bootstrap voltage, plausibility window) are params now —
// `batt_nominal_v` / `batt_min_v` / `batt_max_v` in `BatteryParams`, read
// once at task start (the group is reboot-flagged). Once any valid voltage
// frame arrives we always hold the last reading rather than fall back to
// nominal — a freshly-stale value tracks truth far better, especially at
// end-of-flight when sag is largest.

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

/// Map an [`EffectivenessApplyReport`] onto `defmt` logs (core is
/// log-free). Shared by boot and the disarmed hot-reload so the two paths
/// cannot drift.
fn log_effectiveness_report(report: &EffectivenessApplyReport) {
    match report.g1 {
        G1Application::Geometric => {
            defmt::info!("INDI: G1 derived geometrically from airframe identity");
        }
        G1Application::Configured => {
            defmt::info!("INDI: configured G1 applied from params");
        }
        G1Application::RejectedKeptGeometric => {
            defmt::warn!("INDI: configured G1 invalid — keeping geometric G1");
        }
    }
    if !report.g2_ok {
        defmt::warn!("INDI: G2 params invalid — keeping previous G2");
    }
    for (i, ok) in report.motor_dynamics_ok.iter().enumerate() {
        if !ok {
            defmt::warn!(
                "INDI: motor {} dynamics params invalid (tau/omega), keeping previous",
                i,
            );
        }
    }
}

/// Convert an estimated mechanical omega (rad/s) to wire-safe eRPM (u32).
///
/// Guards the saturating `f32 as u32` cast against non-finite and
/// out-of-range inputs: `+inf as u32` saturates to `u32::MAX`, which would
/// otherwise leak through telemetry as a ~4.3 billion eRPM spike. Anything
/// non-finite, non-positive, or beyond `max_omega * 1.5` (the same headroom
/// as the hard range gate on raw measurements) collapses to 0.
///
/// 1.5 deliberately matches the range gate: with an *identified*
/// `m*_omega_max` the rotor really operates near the bound, and eRPM
/// measurement noise (σ ≈ 9 % of ω) reaches past `1.2·ω_max` — the old
/// 1.2 headroom clipped legitimate top-speed estimates to 0 (which
/// `RpmTracker` would then treat as a VALID zero, handing G2 a floored
/// 1/ω at exactly full throttle).
fn omega_to_safe_erpm(omega: f32, erpm_to_rads: f32, max_omega: f32) -> u32 {
    if omega.is_finite() && omega > 0.0 && omega <= max_omega * 1.5 {
        libm::roundf(omega / erpm_to_rads) as u32
    } else {
        0
    }
}

/// Shortest gap we can *observe* between fresh RPM samples: the FC cannot
/// see telemetry faster than it polls. `dshot_task` is a free-running loop
/// (~150–200 µs per frame including the 80 µs receive window), so the
/// expected-interval floor is the DShot frame period — unless this task's
/// own polling is slower, see [`rpm_observation_period_s`].
const DSHOT_FRAME_PERIOD_S: f32 = 200e-6;
/// The actual observation period: this task drains `DSHOT_TELEMETRY` once
/// per *control* tick (`1 / loop_rate_hz`, i.e. every `indi_ctrl_div`-th
/// IMU sample), so whenever that tick is longer than the DShot frame
/// (1 ms on `imu_1khz`, 500 µs at 8 kHz / 4) the tick is the polling
/// floor — most ESC replies are never seen. Flooring the staleness clamp
/// at 200 µs there made the minimum threshold (10 gaps × 200 µs = 2 ms)
/// just TWO observations, and at low ω — where "no new commutation"
/// replies are common — two consecutive no-fresh observations are
/// routine, so the gate flickered `RpmInput::Invalid` (G2 churn,
/// `rpm_all_stale` log spam) on a healthy link. Computed at task start
/// from the control rate (runtime param) — see [`rpm_observation_period_s`].
fn rpm_observation_period_s(loop_rate_hz: f32) -> f32 {
    let tick_s = 1.0 / loop_rate_hz;
    if tick_s > DSHOT_FRAME_PERIOD_S { tick_s } else { DSHOT_FRAME_PERIOD_S }
}
/// Ceiling on the expected RPM interval, so a near-zero ω estimate cannot
/// make the staleness gate arbitrarily permissive. With the default 10
/// gaps this bounds the worst-case staleness verdict at 50 ms.
const RPM_EXPECTED_GAP_MAX_S: f32 = 5e-3;

/// Wall-clock interval at which the ESC is expected to produce a *new*
/// speed value, given the current estimate of motor speed.
///
/// Bidirectional DShot answers every frame, but the payload only changes
/// once the ESC has measured another commutation — one 60° electrical
/// step, i.e. one electrical revolution / 6. Between those it reports "no
/// new commutation" (period 0), which is expected traffic rather than a
/// fault, so the honest staleness question is "have we missed more
/// intervals than this motor should have produced", not "how many frames
/// were empty". Hence `π / (3 · pole_pairs · ω)`: ~120 µs at hover, but
/// milliseconds during spin-up.
///
/// Deliberately fed the *estimator's* ω rather than a measured one: when
/// telemetry stops the estimator coasts toward `c_m · u`, so the gate
/// tightens to the commanded speed instead of loosening — a dead ESC
/// cannot widen its own staleness window.
fn expected_rpm_gap_s(omega: f32, pole_pairs: f32, observation_period_s: f32) -> f32 {
    if !omega.is_finite() || omega <= 0.0 || pole_pairs <= 0.0 {
        return RPM_EXPECTED_GAP_MAX_S;
    }
    let gap = core::f32::consts::PI / (3.0 * pole_pairs * omega);
    gap.clamp(observation_period_s, RPM_EXPECTED_GAP_MAX_S)
}

/// Build one motor's RPM-estimator config from params.
///
/// Shared by boot and the disarmed hot-reload so the two cannot drift —
/// the same one-code-path rule as `apply_effectiveness_params`. A builder
/// rejection degrades to the schema defaults rather than panicking the
/// inner loop into a flash-persistent boot-panic cycle
/// (docs/safety_protocol.md rule 2); every write path is range-gated by
/// the registry, but a programmatic `params::set()` is not.
fn build_rpm_estimator_config(
    motor_index: usize,
    tau_m: f32,
    curve_k: f32,
    p: &cybflight_core::params::RpmEstimatorParams,
    control_rate_hz: f32,
) -> crate::estimation::rpm_estimator::RpmEstimatorConfig {
    // The throttle history spans `MAX_HISTORY` control ticks, so at this
    // build's control rate it can only represent a delay up to
    // `max_representable_tau_d_s`. Asking for more used to fall back to
    // the oldest entry silently, i.e. the filter modelled a delay the
    // operator never set. Clamp and say so instead — the schema range is
    // rate-independent, but what the buffer can hold is not.
    let tau_d_max =
        crate::estimation::rpm_estimator::max_representable_tau_d_s(control_rate_hz);
    let tau_d = if p.tau_d > tau_d_max {
        // Log once per motor at construction, not per tick.
        defmt::warn!(
            "INDI: motor {} rpm_est_tau_d {} s exceeds the {} s the throttle history spans at {} Hz — clamped",
            motor_index,
            p.tau_d,
            tau_d_max,
            control_rate_hz,
        );
        tau_d_max
    } else {
        p.tau_d
    };
    RpmEstimatorConfigBuilder::new()
        .tau_m_up(tau_m)
        .tau_m_down(tau_m)
        .tau_d(tau_d)
        .throttle_noise_cov(p.throttle_noise_cov)
        .c_m_noise_cov(p.c_m_noise_cov)
        .omega_noise_cov(p.omega_noise_cov)
        .nis_gate(p.nis_gate)
        // m*_nonlin: the identified throttle-curve k; the zero sentinel
        // (and any out-of-range value) selects the legacy linear model
        // inside the builder.
        .curve_k(curve_k)
        .omega_noise_ref(p.omega_noise_ref)
        .init_omega_var(p.init_omega_var)
        .escape_consecutive_rejects(p.escape_consecutive_rejects as u16)
        .plausible_omega_frac(p.plausible_omega_frac)
        .build()
        .unwrap_or_else(|_| {
            defmt::error!(
                "INDI: motor {} RPM-estimator config rejected, using defaults",
                motor_index,
            );
            let d = cybflight_core::params::RpmEstimatorParams::default();
            RpmEstimatorConfigBuilder::new()
                .tau_m_up(DEFAULT_MOTOR_TAU_S)
                .tau_m_down(DEFAULT_MOTOR_TAU_S)
                .tau_d(d.tau_d)
                .throttle_noise_cov(d.throttle_noise_cov)
                .c_m_noise_cov(d.c_m_noise_cov)
                .omega_noise_cov(d.omega_noise_cov)
                .nis_gate(d.nis_gate)
                .curve_k(1.0)
                .omega_noise_ref(d.omega_noise_ref)
                .init_omega_var(d.init_omega_var)
                .escape_consecutive_rejects(d.escape_consecutive_rejects as u16)
                .plausible_omega_frac(d.plausible_omega_frac)
                .build()
                .expect("compile-time default RPM-estimator config must be valid")
        })
}

/// One-shot flag for the Table-mode voltage failsafe trip log: the 8 kHz
/// loop would otherwise emit the error every tick while going silent.
static VOLTAGE_FAILSAFE_TRIPPED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Why the inner loop stopped publishing motor commands, latched for
/// the rest of the boot. 0 = still publishing; otherwise one of
/// `blackbox::topics::events::SILENT_CAUSE_*`.
///
/// Read by the blackbox recorder, which emits `KIND_INNER_SILENT` on
/// the 0 -> non-zero edge. Both trips below go silent and let the
/// controller watchdog disarm, so what lands in the log is a
/// `ControllerTimeout` failsafe — true, but it names the symptom. This
/// atomic names the cause. Set alongside the existing one-shot log
/// latches so the two cannot disagree about whether a trip happened.
pub static INNER_SILENT_CAUSE: core::sync::atomic::AtomicU8 =
    core::sync::atomic::AtomicU8::new(0);

/// `true` while `POWER_STATUS` has been silent longer than
/// [`VOLTAGE_STALE_TIMEOUT`] and INDI is flying the held voltage.
///
/// The *soft* staleness state, distinct from the hard trip above: it is
/// the early warning that precedes it by up to
/// `VOLTAGE_FAILSAFE_TIMEOUT - VOLTAGE_STALE_TIMEOUT`, and it also
/// fires on episodes that recover. Edge-polled by the recorder for
/// `KIND_POWER_STALE` / `KIND_POWER_OK`.
pub static VOLTAGE_STALE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Count of staleness episodes entered since boot. Emitted as the
/// `data` of `KIND_POWER_STALE` so that an episode the recorder's poll
/// aliased away shows up as a gap in the sequence instead of vanishing.
pub static VOLTAGE_STALE_EPISODES: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(0);

/// Duration of the most recently *ended* staleness episode, in ms.
/// Measured here rather than by the recorder, whose poll would fold its
/// own iteration latency (p99 77-212 ms in flight) into the number.
pub static VOLTAGE_STALE_LAST_MS: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(0);

/// One-shot flag for the sustained-WLS-NaN failsafe trip log (same
/// rationale as [`VOLTAGE_FAILSAFE_TRIPPED`]: report the transition once,
/// not at 8 kHz while going silent).
static WLS_NAN_FAILSAFE_TRIPPED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Per-iteration cost probe for the INDI loop, in DWT cycle counts.
/// `indistat` on the shell prints avg/max in µs and resets the sums.
/// One iteration = everything after the decimation gate passes up to the
/// motor publish; sysclk is 480 MHz, so the 8 kHz budget (125 µs) is
/// 60,000 cycles and the 2 kHz budget 240,000.
pub mod step_stats {
    use core::sync::atomic::{AtomicU32, Ordering};
    pub static COUNT: AtomicU32 = AtomicU32::new(0);
    pub static CYCLES_SUM: AtomicU32 = AtomicU32::new(0);
    pub static CYCLES_MAX: AtomicU32 = AtomicU32::new(0);
    pub static LOOP_PERIOD_MAX: AtomicU32 = AtomicU32::new(0);
    /// Blackbox-owned copies of the two maxima, reset by the `/health`
    /// encoder every record. Separate from the shell's pair because both
    /// readers reset on read: sharing one would let an `indistat` poll
    /// during a flight silently zero the window the log was about to
    /// record, and vice versa. Two extra relaxed `fetch_max` per
    /// iteration — a handful of cycles against a 60,000-cycle budget at
    /// 8 kHz.
    pub static CYCLES_MAX_LOG: AtomicU32 = AtomicU32::new(0);
    pub static LOOP_PERIOD_MAX_LOG: AtomicU32 = AtomicU32::new(0);
    /// SYSCLK, from the selected BSP rather than assumed, since the DWT
    /// cycle counter this converts is clocked by it.
    pub const SYSCLK_HZ: u32 = crate::bsp::SYSCLK_HZ;

    #[inline]
    pub fn record(step_cycles: u32, period_cycles: u32) {
        COUNT.fetch_add(1, Ordering::Relaxed);
        CYCLES_SUM.fetch_add(step_cycles, Ordering::Relaxed);
        CYCLES_MAX.fetch_max(step_cycles, Ordering::Relaxed);
        LOOP_PERIOD_MAX.fetch_max(period_cycles, Ordering::Relaxed);
        CYCLES_MAX_LOG.fetch_max(step_cycles, Ordering::Relaxed);
        LOOP_PERIOD_MAX_LOG.fetch_max(period_cycles, Ordering::Relaxed);
    }

    /// Snapshot `(step_max, period_max)` in cycles since the last call
    /// and reset — the blackbox's window. Leaves the shell's stats alone.
    pub fn take_log_max() -> (u32, u32) {
        (
            CYCLES_MAX_LOG.swap(0, Ordering::Relaxed),
            LOOP_PERIOD_MAX_LOG.swap(0, Ordering::Relaxed),
        )
    }

    /// Snapshot `(count, sum, max, period_max)` and reset.
    pub fn take() -> (u32, u32, u32, u32) {
        (
            COUNT.swap(0, Ordering::Relaxed),
            CYCLES_SUM.swap(0, Ordering::Relaxed),
            CYCLES_MAX.swap(0, Ordering::Relaxed),
            LOOP_PERIOD_MAX.swap(0, Ordering::Relaxed),
        )
    }
}

#[embassy_executor::task]
pub async fn indi_task() {
    // --- Load params for INDI controller ---
    let params = crate::params::get();
    let ic = &params.indi.controller;
    // Control rate: INDI steps on every `ctrl_div`-th IMU sample. Read once
    // here (the param is `reboot`): every filter below is designed from
    // the resulting `loop_rate_hz`.
    let ctrl_div = (ic.ctrl_decimation as u32).max(1);
    let loop_rate_hz = crate::rates::IMU_ODR_HZ / ctrl_div as f32;
    let rpm_obs_period_s = rpm_observation_period_s(loop_rate_hz);
    let mass_kg = params.airframe.body.mass_kg;
    // Inertia tensor (+ inverse) for the `outer_mpc_full` α inner loop:
    // the outer loop ships the RAW model torque τ(u0) in `torque_n_m`, and
    // this task recovers the pseudo-control at IMU rate with the FRESH
    // gyro — α = I⁻¹·(τ_d − ω×Iω) — so the gyroscopic correction tracks
    // the actual rates instead of being frozen at the solve instant
    // (paper Fig. 3 evaluates eq. 32 inside the inner loop for the same
    // reason). Same validated full-tensor construction as
    // `FullQuadModel::from_vehicle_params` (falls back to the floored
    // diagonal on a non-SPD tensor — outer_loop warns on that case).
    // Captured once at boot: `inertia_kg_m2` is reboot-flagged, and the
    // geometric G1 torque rows (built from the same tensor at boot) are
    // never rebuilt on hot-reload — so neither is this, keeping α, G1,
    // and the (also boot-pinned) MPC model on one tensor.
    #[cfg(feature = "outer_mpc_full")]
    let (inertia, inv_inertia) = {
        let (i, i_inv, _) = cybflight_core::mpc::full_quad_model::inertia_from_array(
            &params.airframe.body.inertia_kg_m2,
        );
        (i, i_inv)
    };
    // Pre-launch idle throttle — prop/ESC coupled, so a parameter rather
    // than a firmware constant.
    #[cfg(any(feature = "outer_mpc", feature = "outer_geometric"))]
    let idle_normalized = ic.idle_normalized;

    // --- Build INDI controller ---
    let config = IndiConfig {
        // Vehicle YAML `build: indi:` (cargo feature `indi_off`). `no`
        // drops the incremental terms and leaves a plain rate controller
        // — see `IndiConfig::indi_enabled`. Compile-time constant, so the
        // unused branch folds away in either build.
        indi_enabled: !cfg!(feature = "indi_off"),
        rate_gains: ic.rate_gains.into(),
        sync_filter_hz: ic.sync_filter_hz,
        rate_dot_sg_window_size: crate::rates::rate_dot_sg_window(loop_rate_hz),
        rate_dot_sg_order: 2,
        ground_gyro_rad_s: ic.ground_gyro_dps * core::f32::consts::PI / 180.0,
        ground_accel_m_s2: ic.ground_accel_g * params.site.gravity_m_s2,
        ground_thrust_sp_m_s2: ic.ground_thrust_sp_m_s2,
        // Airframe identity from params (YAML-baked + flash overrides) —
        // the same source the MPC and planner read, ending the split-brain
        // where `param set mass` changed the outer loop but not INDI.
        // Captured at task start; body/motor changes need a reboot.
        motors: params.airframe.motors,
        body: params.airframe.body,
        // Motor dynamics + G2 yaw from the airframe params (previously the
        // hardcoded INDI_MOTOR_PARAMS consts). The schema carries real
        // defaults (0.02 s, 40 000 RPM); g2_yaw defaults 0 — vehicles pin
        // `m*_g2_ry` in their YAML. Every write path is range-gated, but
        // `IndiEffectiveness::new` divides by tau, so guard structurally
        // (degrade to schema defaults, never an inf G2 scaler).
        indi_motors: core::array::from_fn(|i| {
            let m = &params.airframe.motors[i];
            let tau = m.time_const_s;
            let tau = if tau.is_finite() && tau > 0.0 { tau } else { DEFAULT_MOTOR_TAU_S };
            let omega = m.max_omega_rad_s;
            let omega = if omega.is_finite() && omega > 0.0 {
                omega
            } else {
                DEFAULT_MOTOR_MAX_OMEGA_RAD_S
            };
            let g2_yaw = m.g2[2];
            IndiMotorParams {
                time_const_s: tau,
                max_rpm: omega * 60.0 / core::f32::consts::TAU,
                // Same magnitude bound as apply_effectiveness_params'
                // G_MAG_MAX — the constructor seed must not admit a value
                // the apply path would reject (at boot "keep previous"
                // would keep exactly this seed).
                g2_yaw: if g2_yaw.is_finite() && g2_yaw.abs() <= 1e4 {
                    g2_yaw
                } else {
                    0.0
                },
            }
        }),
        thrust_model: crate::vehicle::BAKED_THRUST_MODEL,
        // Per-motor nonlinearity: a configured `m*_nonlin` param wins;
        // zero (unset) falls back to the model-matched compile-time default.
        // This closes the old trap where setting one motor's nonlinearity
        // alone did nothing unless the whole effectiveness block was non-zero.
        nonlinearity: SVector::from_fn(|i, _| {
            let k = params.airframe.motors[i].nonlinearity;
            if k > 0.0 { k } else { crate::vehicle::BAKED_THRUST_NONLINEARITY }
        }),
        act_limit: SVector::from_element(1.0),
        wls_wv: ic.wls_wv.into(),
        wls_wu: ic.wls_wu.into(),
        wls_cond_bound: 3.2768e8, // (1<<15) * 1e4
        wls_theta: 1e-4,
        wls_imax: 1,
        // Consecutive WLS-NaN ticks before the armed go-silent failsafe
        // (step 7b). The counter ticks once per *control* step, so it is
        // scaled from `loop_rate_hz` (IMU ODR / `indi_ctrl_div`), not the
        // IMU ODR, so it means the same ~2.5 ms of wall-clock on every
        // build — a flat 20 ticks was 2.6 ms at 8 kHz but 21 ms at
        // 1 kHz, all of it spent flying the decayed hold while armed;
        // deriving from the ODR made it 10 ms at 8 kHz / div 4.
        // Floor of 2 keeps "consecutive" meaningful at low rates.
        nan_limit: ((loop_rate_hz * 0.0025) as u16).max(2),
        // Per-tick decay of the held actuator state during that window
        // (`indi_nan_rampdown`). Rate-coupled the same way `nan_limit`
        // is, so retune the two together.
        nan_rampdown: ic.nan_rampdown,
        // Debounce only. The staleness *decision* is made in the task from
        // a physically-derived expected sample interval (see
        // `expected_rpm_gap_s` and `indi_rpm_stale_gaps`); these counters
        // just add a short, uniform confirmation delay on top. Derived from
        // the control rate (they tick per control step) so they mean the
        // same wall-clock time on every build — a flat 50 ticks was 6 ms
        // at 8 kHz but 50 ms at 1 kHz.
        rpm_invalid_limit: ((loop_rate_hz * 0.006) as u16).max(1),
        rpm_all_invalid_limit: ((loop_rate_hz * 0.006) as u16).max(1),
        rpm_recovery_count: ((loop_rate_hz * 0.00125) as u16).max(3),
        motor_pole_count: params.airframe.motor_pole_count,
    };

    // INDI runs once per primary-gyro sample, so the loop rate IS the gyro
    // ODR. Sourced from `rates::IMU_ODR_HZ` (the build's effective ODR: ICM
    // boards 8 kHz, or 1 kHz with the `imu_1khz` knob; BMI270 boards 3.2 kHz)
    // so the SG rate-derivative, omega_dot finite-diff, RPM-notch coefficients
    // and outer-loop decimation all scale to the real sample rate. The IMU
    // driver's `sample_rate_hz()` agreement is asserted in board_init.
    let mut indi = IndiController::new(&config, loop_rate_hz);

    // --- Per-motor RPM estimators (FOPDT EKF, one per motor) ---
    // Motor dynamics come from the same params block as the controller
    // (single source). Range gates cover every write path; the structural
    // finite/>0 guard degrades to the schema defaults, never tau=0 in the KF.
    let pole_pairs = config.motor_pole_count as f32 / 2.0;
    let erpm_to_rads = core::f32::consts::TAU * 100.0 / (pole_pairs * 60.0);
    let mut rpm_estimators: [RpmEstimator; NU] = core::array::from_fn(|i| {
        let m = &params.airframe.motors[i];
        let default_tau = DEFAULT_MOTOR_TAU_S;
        let default_omega = DEFAULT_MOTOR_MAX_OMEGA_RAD_S;
        let (tau_m, c_m) = if m.time_const_s.is_finite()
            && m.time_const_s > 0.0
            && m.max_omega_rad_s.is_finite()
            && m.max_omega_rad_s > 0.0
        {
            (m.time_const_s, m.max_omega_rad_s)
        } else {
            defmt::warn!(
                "INDI: motor {} dynamics params invalid (tau/omega), using defaults",
                i,
            );
            (default_tau, default_omega)
        };
        let est_config =
            build_rpm_estimator_config(
                i,
                tau_m,
                m.nonlinearity,
                &params.indi.rpm_estimator,
                loop_rate_hz,
            );
        // Initial ω variance from `rpm_est_init_omega_var`, the same
        // value the filter re-seeds with. It was the one part of the
        // estimator's configuration that stayed a literal here.
        let init_omega_var = params.indi.rpm_estimator.init_omega_var;
        let init_state =
            StateAndCov::new(0.0, c_m, Matrix2::new(init_omega_var, 0.0, 0.0, c_m * c_m));
        RpmEstimator::new(est_config, init_state)
    });

    // Apply the effectiveness param block (G1 sentinel/configured, G2
    // verbatim, motor dynamics, nonlinearity). Same call as the disarmed
    // hot-reload below — one code path.
    let report = indi.apply_effectiveness_params(
        &params.airframe.motors,
        &params.indi.effectiveness,
        crate::vehicle::BAKED_THRUST_NONLINEARITY,
    );
    log_effectiveness_report(&report);

    // Arm-edge tracking for KF resets.
    let mut was_armed = false;

    // --- Slew outlier filter (always on, protects both KF and raw path) ---
    //
    // Per-motor `SlewFilter` with a fixed max per-sample delta (ZOH on reject). The delta is
    // derived from a physics rate bound × a worst- case inter-sample interval, NOT a live dt lookup
    // — `SlewFilter` is intentionally time-unaware, so we pre-size the gate for the slowest
    // tolerable telemetry cadence and accept that at nominal rates the gate is loose by that same
    // ratio.
    let max_omega_bound: f32 = {
        let m = params
            .airframe
            .motors
            .iter()
            .map(|m| m.max_omega_rad_s)
            .filter(|v| v.is_finite())
            .fold(0.0f32, f32::max);
        if m > 0.0 { m } else { DEFAULT_MOTOR_MAX_OMEGA_RAD_S }
    };
    // Physics bound: max plausible dω/dt for a first-order motor with
    // time constant TAU_MIN_S driven toward max_omega_bound.
    const TAU_MIN_S: f32 = 0.005;
    // Worst-case interval between valid per-motor telemetry frames.
    //
    // Bidirectional DShot has every ESC answer every command frame — the
    // driver decodes all four per frame (`motors/dshot.rs`) — so telemetry
    // cadence follows the DShot frame period (~180 µs for DSHOT600
    // including the GCR reply and line turnaround), NOT the IMU rate. 1 ms
    // is ~5 frames of headroom, enough to ride out a burst of dropouts
    // without opening the gate to real outliers.
    //
    // Deliberately a constant. Deriving it from `loop_rate_hz` made the
    // gate 8× looser on `imu_1khz` builds, where the resulting delta
    // (`ω_max/TAU_MIN · 8 ms` ≈ 6700 rad/s) exceeded the hard range gate
    // below (`1.5 · ω_max` ≈ 6280 rad/s) — no admissible sample could
    // trip it, so the outlier filter was a provable no-op at 1 kHz. This
    // value is what 8 kHz builds have always flown.
    const TELEM_WORST_DT_S: f32 = 1.0e-3;
    let slew_max_delta: f32 = (max_omega_bound / TAU_MIN_S) * TELEM_WORST_DT_S;
    let mut slew_filters: [SlewFilter<f32>; NU] =
        core::array::from_fn(|_| SlewFilter::new(slew_max_delta).unwrap());
    // Local param version — re-read params when global version changes.
    let mut local_param_ver =
        crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
    // --- Subscribe to channels ---
    let mut imu_sub = crate::subscribe_or_park!(IMU_1, "IMU_1");
    let mut dshot_sub = crate::subscribe_or_park!(DSHOT_TELEMETRY, "DSHOT_TELEMETRY");
    // Battery voltage from `power_task` (100 Hz). Consumed by the thrust
    // map in `Table` mode; ignored by the analytic models. Held between
    // updates with a staleness fallback to nominal — see VOLTAGE_* consts.
    let mut power_sub = crate::subscribe_or_park!(POWER_STATUS, "POWER_STATUS");
    // Armed state read from IS_ARMED atomic (set by DShot task).
    let att_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
    let motor_telem_pub = super::ACTUATOR_MOTORS_TELEM.immediate_publisher();
    let processed_dshot_pub = super::PROCESSED_DSHOT_TELEM.immediate_publisher();
    let processed_motor_pub = super::PROCESSED_MOTOR_STATE.immediate_publisher();
    let tracking_err_pub = super::TRACKING_ERROR.immediate_publisher();
    let health_pub = super::DSHOT_HEALTH.immediate_publisher();

    // --- State ---
    // Rate command from outer loop (cascade, MPC, or RC rate mode).
    let mut collective_thrust_n: f32 = 0.0;
    let mut rate_ref = Vector3::<f32>::zeros();
    // Held body-torque setpoint τ_d (`outer_mpc_full`): refreshed on each
    // RATE_COMMAND, held between outer solves, and covered by the same
    // 100 ms staleness gate as the rest of the command. The α pseudo-
    // control is re-derived from it every IMU tick with the fresh gyro.
    #[cfg(feature = "outer_mpc_full")]
    let mut torque_ref = Vector3::<f32>::zeros();
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
    //
    // Battery facts come from BatteryParams (reboot-flagged, read once
    // here). The registry range-checks each key individually; the
    // cross-field relation (min < max, nominal inside the window) can't
    // be expressed per-key, so sanity-check it here and degrade to the
    // schema defaults rather than run with an inverted gate.
    let (v_nominal, v_min_plausible, v_max_plausible) = {
        let b = &params.battery;
        if b.min_plausible_v < b.max_plausible_v
            && (b.min_plausible_v..=b.max_plausible_v).contains(&b.nominal_v)
        {
            (b.nominal_v, b.min_plausible_v, b.max_plausible_v)
        } else {
            defmt::warn!(
                "INDI: battery params inconsistent (nominal {} window {}..{}), using defaults",
                b.nominal_v,
                b.min_plausible_v,
                b.max_plausible_v,
            );
            let d = cybflight_core::params::BatteryParams::default();
            (d.nominal_v, d.min_plausible_v, d.max_plausible_v)
        }
    };
    let mut last_voltage_v: f32 = v_nominal;
    let mut last_voltage_time: Option<Instant> = None;
    // Tracks the start of the current voltage-staleness episode for
    // one-shot stale/recover logging and the Table-mode armed failsafe
    // gate. `Some(t)` ⇒ stale since `t`; `None` ⇒ fresh.
    let mut voltage_stale_since: Option<Instant> = None;

    // RPM estimator timestamp tracking (seconds, f32 relative to task start)
    let mut est_prev_ts: Option<Instant> = None;
    let mut est_current_ts: f32 = 0.0;

    // --- DShot telemetry health accounting ---
    // Cumulative since boot, never reset: differencing two published
    // snapshots gives the rate over any window, and monotonic counters have
    // no reset-semantics ambiguity across arm/launch edges. See
    // `super::DshotMotorHealth` for what each bucket means and why
    // `no_fresh` is expected traffic rather than an error.
    let mut health = [super::DshotMotorHealth::default(); NU];
    let mut health_frames: u32 = 0;

    // --- RPM liveness ---
    // Time of the last *usable* ω per motor (survived decode + range +
    // slew). `None` means none since the last reset, which counts as
    // stale: G2 stays off until telemetry has proven itself, rather than
    // being trusted by default.
    let mut rpm_last_fresh: [Option<Instant>; NU] = [None; NU];
    let rpm_stale_gaps = ic.rpm_stale_gaps;
    let use_kf_omega = ic.use_kf_omega;
    // Edge-tracking for the all-motors-lost log, so the 8 kHz loop reports
    // the transition rather than every tick of the condition.
    let mut rpm_all_stale_prev = false;
    // Rate-limited defmt surfacing: every HEALTH_LOG_DECIMATION telemetry
    // publishes (100 Hz → 5 s), report only motors whose *error* buckets
    // advanced since the last report, so a healthy link stays silent.
    const HEALTH_LOG_DECIMATION: u32 = 500;
    let mut health_log_counter: u32 = 0;
    let mut health_log_prev = [0u32; NU];

    // Decimation counter for position/attitude controller
    let mut outer_counter: u32 = 0;
    // Position controller runs at loop_rate_hz / OUTER_DECIMATION ≈ 100 Hz,
    // derived from the actual gyro ODR (8000/80 on ICM boards, 3200/32 on
    // BMI270 boards) so the outer loop stays at ~100 Hz regardless of board.
    // `.max(1)`: below a 200 Hz control rate (large `indi_ctrl_div` on a
    // 1 kHz vehicle) the truncation would give 0 and the `== 1` gate
    // below would never fire, silencing all 100 Hz telemetry.
    let outer_decimation: u32 = ((loop_rate_hz / 100.0) as u32).max(1);
    // Sysid fast-telemetry decimation: ≥500 Hz for the blackbox
    // `sysid` record-set tier (16 at 8 kHz, 2 at 1 kHz). Only the
    // three blackbox mirrors (/motors, /motor_state, /tracking_error)
    // run at this rate — ATTITUDE_CONTROL_SETPOINT / DshotTelemetry /
    // DshotHealth stay at 100 Hz, they feed the ESP bridge and shell,
    // not sysid.
    let sysid_telem_decimation: u32 = ((loop_rate_hz / 500.0) as u32).max(1);
    let mut sysid_telem_counter: u32 = 0;

    // INDI starts immediately. Gyro bias begins at zero and improves as
    // Mahony's ki integral converges (~1–2 s). Rate control runs from
    // the first IMU tick; the outer loop's first RATE_COMMAND activates
    // motor output.
    defmt::info!("INDI task started ({}Hz)", loop_rate_hz as u32);
    // `indi_sync_hz` is schema-bounded to 500 Hz, which is legal at 8 kHz
    // and exactly Nyquist at 1 kHz. The controller clamps rather than
    // panicking; say so, because the flown filter is then not the tuned one.
    if indi.effective_sync_filter_hz() != ic.sync_filter_hz {
        defmt::warn!(
            "INDI: indi_sync_hz {} Hz is too high for a {} Hz loop — clamped to {} Hz",
            ic.sync_filter_hz,
            loop_rate_hz as u32,
            indi.effective_sync_filter_hz(),
        );
    }
    if ic.use_kf_omega {
        defmt::info!(
            "INDI: motor ω from the RPM Kalman filter (indi_omega_kf=1) — depends on \
             m*_tau / m*_omega_max being identified, not defaults"
        );
    } else {
        defmt::info!("INDI: motor ω from held dshot telemetry (indi_omega_kf=0)");
    }
    if !config.indi_enabled {
        defmt::warn!(
            "INDI DISABLED (build: indi: no) — inner loop is a proportional rate \
             controller (no incremental feedback, no G2); re-tune indi_rate_* before flight"
        );
    }

    // ── Airframe ↔ thrust-table binding ────────────────────────────────
    //
    // The consistency CHECK (table per-rotor thrust axis vs the airframe's
    // `max_thrust_n`, 10% tolerance) moved to `build.rs::vehicle_bake` —
    // fail-loud at bake, no boot-panic cycle on the vehicle. What remains
    // here is the bench-visible info log so an operator can sanity-check
    // the active table against the pack in use.
    if let ThrustModel::Table(t) = crate::vehicle::BAKED_THRUST_MODEL {
        defmt::info!(
            "INDI Table: thrust [{}, {}] N/rotor, voltage [{}, {}] V, per_motor_max={} N",
            t.thrust_min_n(),
            t.thrust_max_n(),
            t.voltage_min_v(),
            t.voltage_max_v(),
            params.airframe.motors[0].max_thrust_n,
        );
    }

    /****************************/
    // The ω / ω̇ this task hands INDI as `MotorState::External` must carry
    // the SAME group delay as the signals INDI compares them against
    // (`spf_fs`, `u_state_fs`, `rate_dot_fs`) — that delay match is the
    // entire purpose of `indi_sync_hz`. So this filter takes its cutoff
    // from the controller's effective (Nyquist-clamped) sync cutoff rather
    // than a literal. It used to be pinned at 15 Hz against a 12 Hz sync
    // filter, which put 15.0 ms of delay on ω against 18.8 ms on
    // everything else and made `indi_sync_hz` silently stop meaning "the
    // common group delay" the moment anyone tuned it.
    let motor_filter_hz = indi.effective_sync_filter_hz();
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
    // Values come from `RpmNotchParams` (`rpm_notch_*` keys, reboot-flagged
    // — the banks bake q/min/fade into their biquads at construction, so
    // everything is read once here). Defaults match Betaflight's
    // `rpm_filter` defaults with the notch disabled. NU=4, NH_GYRO=3,
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
    // When `rpm_notch_en` is false the bank storage and PT1 state are
    // still allocated (~10 KB BSS, unchanged — they always were) and the
    // per-loop cost is one predicted branch at 8 kHz. Bench-check CPU
    // headroom before first enabling on a flight vehicle.
    //
    // Structural guard: `RpmNotchBank::new` panics on an invalid biquad
    // config. Every write path is range-gated by the registry, but a
    // programmatic `params::set()` is not — degrade to the schema
    // defaults rather than panic the inner loop into a boot-panic cycle.
    //
    // 1 kHz (`imu_1khz`) builds: the bank's usable band tops out at
    // 0.48 × loop rate = 480 Hz, and that ceiling bites the *fundamental*,
    // not just the harmonics. On a 40 krpm airframe 1P reaches 667 Hz, so
    // it leaves the band above ~52 % thrust and the bank fades to
    // passthrough exactly in the high-vibration regime it exists for.
    // Below that, 2P sits in the upper fade band (430–480 Hz), where the
    // RBJ Q-mapping has collapsed the notch to a few Hz wide with a ~25 ms
    // ringing constant — a resonator sweeping the gyro path, not a notch.
    // (The bank's achievable −3 dB width is capped at sample_hz / 2πQ,
    // ≈ 32 Hz at 1 kHz / Q=5 versus 255 Hz at 8 kHz.)
    //
    // The division of labor at 1 kHz is therefore the chip's 258 Hz AAF +
    // 227 Hz UI filter plus the software `imu_gyro_lpf_hz` / `imu_accel_lpf_hz`
    // biquads — not this bank. The coverage guard below warns when the
    // band cannot span the airframe's throttle range; it does not
    // force-disable, so a bench experiment is still possible (it needs
    // `rpm_notch_q` and the IMU LPF cutoffs retuned to be worth anything).
    let notch = {
        let n = params.rpm_notch.clone();
        let nyquist_ok = n.min_hz + n.fade_hz < 0.48 * loop_rate_hz;
        let finite_pos =
            |v: f32| v.is_finite() && v > 0.0;
        if finite_pos(n.q) && finite_pos(n.min_hz) && finite_pos(n.fade_hz)
            && finite_pos(n.freq_lpf_hz) && nyquist_ok
        {
            n
        } else {
            defmt::warn!("INDI: rpm_notch params invalid, using defaults (disabled)");
            cybflight_core::params::RpmNotchParams::default()
        }
    };
    let rpm_notch_enabled = notch.enable;
    let mut gyro_rpm_notch =
        RpmNotchBank::<NU, 3>::new(loop_rate_hz, notch.q, notch.min_hz, notch.fade_hz);
    let mut accel_rpm_notch =
        RpmNotchBank::<NU, 1>::new(loop_rate_hz, notch.q, notch.min_hz, notch.fade_hz);
    // Per-motor rotational frequency in Hz, fed into both notch banks
    // each loop after passing through the dedicated PT1 below.
    let mut motor_freq_hz = [0.0f32; NU];
    /// Rad/s → Hz: divide by 2π. Pre-computed as a multiply for speed.
    const RAD_S_TO_HZ: f32 = 0.5 * core::f32::consts::FRAC_1_PI;

    // Band-coverage guard. Derived from `loop_rate_hz` and the airframe's
    // own `m*_omega_max` rather than the `imu_1khz` feature, so it states
    // the physical fact ("this bank cannot follow these motors") instead of
    // a build knob — and so it also catches a fast airframe on the 3.2 kHz
    // BMI270 boards, or a re-propped vehicle whose 1P outgrew an 8 kHz
    // band. Thrust ∝ ω², so the fraction of full thrust at which 1P runs
    // off the end of the band is (ceiling / max_1p)².
    if rpm_notch_enabled {
        let ceiling_hz = 0.48 * loop_rate_hz;
        let max_1p_hz = params
            .airframe
            .motors
            .iter()
            .map(|m| m.max_omega_rad_s)
            .filter(|w| w.is_finite() && *w > 0.0)
            .fold(0.0f32, f32::max)
            * RAD_S_TO_HZ;
        if max_1p_hz > ceiling_hz {
            let ratio = ceiling_hz / max_1p_hz;
            defmt::warn!(
                "INDI: rpm_notch band tops out at {} Hz but motor 1P reaches {} Hz \
                 — the fundamental leaves the band above ~{} % thrust and the bank \
                 fades to passthrough there. Designed for 8 kHz loops; this build \
                 runs at {} Hz.",
                ceiling_hz as u32,
                max_1p_hz as u32,
                (ratio * ratio * 100.0) as u32,
                loop_rate_hz as u32,
            );
        }
    }

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
    // The 150 Hz default matches Betaflight's upstream `rpm_filter_lpf_hz`;
    // tunable via `rpm_notch_lpf_hz`.
    let motor_freq_pt1_alpha: f32 = {
        let dt = 1.0 / loop_rate_hz;
        let tau = 1.0 / (2.0 * core::f32::consts::PI * notch.freq_lpf_hz);
        dt / (tau + dt)
    };
    let mut motor_freq_lpf_state = [0.0f32; NU];
    // Mirrors `omega_fs_has_prev`: the first sample seeds the PT1 directly
    // instead of running a step from zero, avoiding a startup transient
    // that would tilt the first ~τ ms of notch tracking.
    let mut motor_freq_lpf_has_prev = false;

    // Tracks the previous tick's [`super::LAUNCHED`] value so we can
    // detect the !was_launched → launched edge and reset INDI internal
    // state on it (mirrors the existing arm-edge reset). Only meaningful
    // in position-mode builds where the LAUNCHED static exists.
    #[cfg(any(feature = "outer_mpc", feature = "outer_geometric"))]
    let mut was_launched = false;

    // ── INDI state-reset macro ────────────────────────────────────────
    //
    // Resets every piece of inner-loop state that should start fresh on
    // a clean handover (KF, slew gates, motor-omega LPF, RPM-notch
    // bank). Used at the arm edge AND at the !LAUNCHED → LAUNCHED
    // edge below — both edges represent the same underlying invariant:
    // "motors transition from rest/idle to closed-loop control, prior
    // filter state is irrelevant or misleading."
    //
    // Macro rather than closure to avoid the `&mut`-capture lifetime
    // dance that would otherwise tie up half the locals for the rest of
    // the loop body.
    macro_rules! reset_indi_state {
        () => {{
            for est in rpm_estimators.iter_mut() {
                est.reset_state();
            }
            // Reconstruct rather than `reset([0.0; NU])`. `SlewFilter::reset`
            // seeds `state = Some(0.0)`, which defeats the crate's
            // "the very first sample is always accepted" cold start: a first
            // frame further than `slew_max_delta` from zero would be rejected,
            // and because rejection is a *freeze* (the state never advances
            // toward the input) there is no path back — every later sample is
            // measured against the same stale 0.0 and rejected too. A fresh
            // filter starts at `state = None` and takes its seed from the
            // first real sample, which is the right prior at both the arm and
            // the LAUNCHED edge (motors at rest, or at a uniform idle ω we do
            // not know a priori). Same reconstruct-don't-reset reasoning as
            // the biquad below.
            slew_filters = core::array::from_fn(|_| SlewFilter::new(slew_max_delta).unwrap());
            motor_omega_filter = core::array::from_fn(|_| make_biquad());
            last_y_meas_hold = SVector::zeros();
            // Liveness must be re-earned after a reset, not inherited: the
            // pre-launch bypass skips the telemetry block entirely, so a
            // timestamp from before the LAUNCHED edge would vouch for data
            // the new filter state never saw.
            rpm_last_fresh = [None; NU];
            omega_fs = SVector::zeros();
            omega_dot_fs = SVector::zeros();
            omega_fs_has_prev = false;

            if rpm_notch_enabled {
                gyro_rpm_notch.reset();
                accel_rpm_notch.reset();
                motor_freq_lpf_state = [0.0; NU];
                motor_freq_lpf_has_prev = false;
            }

        }};
    }

    // DWT cycle counter for the step-cost probe (`step_stats`).
    {
        let mut cp = unsafe { cortex_m::Peripherals::steal() };
        cp.DCB.enable_trace();
        cp.DWT.enable_cycle_counter();
    }
    let mut probe_prev_start: u32 = 0;
    let mut ctrl_decim_ctr: u32 = 0;

    loop {
        // 1. Await IMU sample. The reader publishes at IMU_ODR_HZ; INDI
        //    steps on every `ctrl_div`-th one (`indi_ctrl_div`). Skipped
        //    samples are already band-limited by the reader's biquad, so
        //    this is decimation, not aliasing. `next_message_pure` drops
        //    `Lagged` silently, so the counter is over *received* samples;
        //    `est_dt` below is timestamp-based and self-corrects.
        let imu = imu_sub.next_message_pure().await;
        ctrl_decim_ctr += 1;
        if ctrl_decim_ctr < ctrl_div {
            continue;
        }
        ctrl_decim_ctr = 0;
        let probe_start = cortex_m::peripheral::DWT::cycle_count();
        let est_dt = if let Some(prev) = est_prev_ts {
            imu.timestamp.saturating_duration_since(prev).as_micros() as f32 / 1_000_000.0
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
        // On ARM: reset KF state for fresh convergence.
        if !was_armed && armed {
            // Reset every inner-loop filter / KF / slew gate. Comment
            // history: the KF reset gives a fresh-per-flight start; the
            // slew + motor-omega resets prevent stale omega from the
            // previous flight bleeding into this one (motors are at rest
            // at arm time, so 0 is the right prior); the optional
            // RPM-notch resets clear delay lines and re-seed the
            // frequency tracker.
            reset_indi_state!();
        }
        was_armed = armed;

        // ── Launch edge: mirror the arm-edge state reset ──────────────
        //
        // While `armed && !LAUNCHED`, the bypass branch below `continue`s
        // before the dshot/IMU/INDI processing path runs, so the
        // controller's filters / KF / slew gates accumulate no state
        // during pre-launch. On the !was_launched → launched edge we
        // reset them again so the first closed-loop tick sees the same
        // fresh prior the arm edge already established — uniform-idle
        // motors are the new "rest" prior for the slew/omega filters.
        #[cfg(any(feature = "outer_mpc", feature = "outer_geometric"))]
        let launched = super::LAUNCHED.load(core::sync::atomic::Ordering::Acquire);
        #[cfg(any(feature = "outer_mpc", feature = "outer_geometric"))]
        {
            if armed && launched && !was_launched {
                reset_indi_state!();
                defmt::info!("INDI: LAUNCHED edge — state reset");
            }
            was_launched = launched;
        }

        // ── Pre-launch idle bypass ────────────────────────────────────
        //
        // While armed but not yet launched, write a uniform
        // the `indi_idle_norm` throttle to all four motors and skip the
        // entire INDI / KF stack. Guarantees motors spin at
        // the same speed regardless of (a) drone tilt on the ground,
        // (b) WLS allocator asymmetry from configured G1, or (c) any
        // upstream MPC / cascade output (which is silently dropped
        // here).
        //
        // We update `LAST_CONTROLLER_PUBLISH` (DShot's 10-ms
        // motor-cmd-stale watchdog needs this to not drop motors to
        // forced idle) but DELIBERATELY DO NOT update `last_cmd_time`.
        // Letting the cmd-staleness gate trip naturally on the launch
        // tick is the safety mechanism: if `outer_loop` was silent
        // during pre-launch (e.g. ESKF stale), the first launched
        // tick will see `try_take() = None`, `cmd_fresh = false`, and
        // INDI will silently skip → the controller watchdog (500 ms)
        // disarms. If we bumped `last_cmd_time` here, INDI would
        // instead run WLS with stale local-cache values
        // (`rate_ref`/`collective_thrust_n`) for up to 100 ms — a
        // dangerous moment of unknown thrust at the most critical
        // tick. See safety review note H1.
        #[cfg(any(feature = "outer_mpc", feature = "outer_geometric"))]
        if armed && !launched {
            let idle = msgs::NormalizedThrottle::new_saturating(idle_normalized);
            let now = Instant::now();
            ACTUATOR_MOTORS.signal(msgs::ActuatorMotors {
                timestamp: now,
                motor_commands: [idle; NU],
            });
            super::LAST_CONTROLLER_PUBLISH.lock(|c| c.set(Some(now)));
            continue;
        }

        // ── DShot telemetry + slew outlier filter ──────────────────────
        //
        // Decode eRPM → rad/s, then gate on a fixed per-sample delta to
        // reject GCR decode errors (ZOH on reject, no interpolation).
        // Always on — protects both the KF path and the raw-hold path.
        let mut y_meas: [Option<f32>; NU] = [None; NU];
        if let Some(telem) = dshot_sub.try_next_message_pure() {
            health_frames = health_frames.saturating_add(1);
            for i in 0..NU {
                let m = &telem.motors[i];
                let raw_omega = match m.value {
                    TelemetryValue::Erpm(erpm) => Some(erpm as f32 * erpm_to_rads),
                    TelemetryValue::Stopped => Some(0.0),
                    TelemetryValue::Edt(_) => {
                        health[i].edt = health[i].edt.saturating_add(1);
                        None
                    }
                    // `Invalid` covers two conditions the driver cannot
                    // distinguish by value alone, but `raw` can:
                    //   raw == Some(_) → the frame decoded, and carried
                    //     period 0 — this ESC firmware's explicit "no new
                    //     commutation since my last reply". Expected, and
                    //     common at low RPM.
                    //   raw == None    → nothing decodable came back at all
                    //     (GCR/checksum failure, or a reply outside the
                    //     receive window). A real error rate.
                    TelemetryValue::Invalid => {
                        if m.raw.is_some() {
                            health[i].no_fresh = health[i].no_fresh.saturating_add(1);
                        } else {
                            health[i].no_reply = health[i].no_reply.saturating_add(1);
                        }
                        None
                    }
                };

                if let Some(omega) = raw_omega {
                    // Hard range gate
                    if omega < 0.0 || omega > max_omega_bound * 1.5 {
                        health[i].range_reject = health[i].range_reject.saturating_add(1);
                        continue;
                    }

                    // SlewFilter returns `input` on accept and the held state on reject; equality
                    // to input uniquely identifies the accept branch (an equal state can only occur
                    // with a zero-delta input, which also accepts).
                    let out = slew_filters[i].apply(omega);
                    if out == omega {
                        y_meas[i] = Some(omega);
                        rpm_last_fresh[i] = Some(imu.timestamp);
                        health[i].passed = health[i].passed.saturating_add(1);
                    } else {
                        health[i].slew_reject = health[i].slew_reject.saturating_add(1);
                    }
                }
            }
        }

        // ── Conditional KF ───────────────────────────────────────────
        //
        // KF only runs when armed.
        // Disarmed: KF idle (avoids divergence without corrections).
        let (g2_valid, rpm_all_stale) = if armed {
            // Normal flight: run KF, feed smoothed omega to RpmTracker
            for i in 0..NU {
                // A rejection here means telemetry arrived and passed every
                // task-level gate, but the model and the measurement
                // disagreed — a different failure from having no data, and
                // one that used to be invisible.
                if rpm_estimators[i].step(est_current_ts, est_dt, y_meas[i])
                    == StepOutcome::Rejected
                {
                    health[i].nis_reject = health[i].nis_reject.saturating_add(1);
                }
            }
            // Validity comes from the *wire*, the value from the estimator.
            //
            // This used to pass `RpmInput::Erpm(kf_omega)` unconditionally,
            // which made `RpmInput::Invalid` unreachable: `g2_valid` was a
            // constant `true`, the per-motor G2 gating and the all-motors
            // failsafe in `RpmTracker` could never fire, and four dead ESCs
            // would have left INDI applying G2 against a purely fabricated
            // ω. The estimator always produces a finite number — that is
            // its job — so it can never be the source of liveness.
            //
            // The gate is time-based rather than frame-counting because
            // this ESC firmware legitimately reports "no new commutation"
            // between electrical steps: at idle roughly half of all frames,
            // and dozens consecutively during spin-up. Counting those as
            // faults would trip the failsafe every takeoff.
            let inputs: [RpmInput; NU] = core::array::from_fn(|i| {
                let omega_hat = rpm_estimators[i].state().omega();
                let stale = match rpm_last_fresh[i] {
                    // `rpm_last_fresh` is only ever written from
                    // `imu.timestamp`, so this subtraction cannot underflow.
                    Some(t) => {
                        let elapsed =
                            imu.timestamp.saturating_duration_since(t).as_micros() as f32 * 1e-6;
                        elapsed > rpm_stale_gaps * expected_rpm_gap_s(omega_hat, pole_pairs, rpm_obs_period_s)
                    }
                    None => true,
                };
                if stale {
                    RpmInput::Invalid
                } else {
                    match omega_to_safe_erpm(omega_hat, erpm_to_rads, max_omega_bound) {
                        // 0 here means the guard collapsed a non-finite or
                        // out-of-range estimate — that is "do not trust",
                        // not "rotor stopped". Erpm(0) would be recorded by
                        // RpmTracker as a VALID zero and G2 would run on a
                        // floored 1/omega; Invalid takes the last-good /
                        // hysteresis path instead.
                        0 => RpmInput::Invalid,
                        erpm => RpmInput::Erpm(erpm),
                    }
                }
            });
            indi.update_rpm(&inputs)
        } else {
            // Disarmed: KF idle, G2 inactive
            ([false; NU], false)
        };

        // Step the per-motor LPF every IMU tick. On a missed telemetry sample
        // (y_meas[i] == None) hold the last valid *input* (true ZOH), not the
        // filter output — the filter then smoothly settles to that constant.
        // omega_dot_fs is a finite difference of the filter output over the
        // MEASURED sample interval, the same `est_dt` the RPM KF above uses:
        // dividing by the nominal loop rate turns IMU scheduling jitter
        // straight into ω̇ error, and the jitter is a larger fraction of dt
        // the faster the loop runs (125 µs ticks at 8 kHz). Guarded because
        // est_dt comes from hardware timestamps. Skip the first iteration to
        // avoid a startup spike.
        let est_dt_inv = if est_dt.is_finite() && est_dt > 0.0 {
            1.0 / est_dt
        } else {
            loop_rate_hz
        };
        for i in 0..NU {
            // ω source (`indi_omega_kf`). Both branches feed the SAME
            // biquad, so the group delay INDI's increment depends on is
            // identical either way and the two are directly comparable.
            //
            // KF: dropouts are bridged by the motor model rather than by a
            // constant, and the single-step commutation noise this ESC
            // sends is attenuated before the finite difference amplifies
            // it. The estimator idles while disarmed, so its ω is only
            // meaningful when armed — but so is `MotorState::External`,
            // which is the only consumer.
            //
            // ZOH: the faithful indiflight port (`indiUpdateActuatorState`
            // re-filters Betaflight's held last-valid value every loop).
            // Model-free, so a wrong `m*_tau` / `m*_omega_max` cannot
            // fabricate an ω — it can only go stale.
            let x = if use_kf_omega && armed {
                rpm_estimators[i].state().omega()
            } else {
                match y_meas[i] {
                    Some(y) => {
                        last_y_meas_hold[i] = y;
                        y
                    }
                    None => last_y_meas_hold[i],
                }
            };
            let new_fs = motor_omega_filter[i].apply(x);
            omega_dot_fs[i] = if omega_fs_has_prev {
                (new_fs - omega_fs[i]) * est_dt_inv
            } else {
                0.0
            };
            omega_fs[i] = new_fs;
        }
        omega_fs_has_prev = true;

        // ── RPM-tracking notches on gyro + accel ───────────────────────
        //
        // Gated by `rpm_notch_en` (reboot-flagged param). When false,
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
        //      and `accel_corrected` so the rest of the loop (INDI step)
        //      consumes the notched signals.
        //
        // ZOH on missed dshot frames: hold the previous filter state, do
        // not push a stale or non-finite value. Per docs/safety_protocol.md
        // rule 3, this stage degrades gracefully and never injects NaN
        // into the IMU signal.
        let (gyro_corrected, accel_corrected) = if rpm_notch_enabled {
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
            if v.is_finite() && (v_min_plausible..=v_max_plausible).contains(&v) {
                last_voltage_v = v;
                last_voltage_time = Some(power.timestamp);
            }
            // Implausible/NaN frames silently drop. Repeated drops trip
            // the staleness fallback below.
        }

        // 4. Drain latest rate command from outer loop.
        let now = Instant::now();
        outer_counter += 1;
        if outer_counter >= outer_decimation {
            outer_counter = 0;
        }
        if let Some(cmd) = super::RATE_COMMAND.try_take() {
            rate_ref = cmd.body_rate_rad_s;
            collective_thrust_n = cmd.collective_thrust_n;
            spf_sp_z = collective_thrust_n / mass_kg;
            telem_attitude = cmd.attitude_quaternion;
            // `outer_mpc_full` ships the raw model torque τ(u0) in
            // `torque_n_m` (outer_loop publish step 9); hold it — the α
            // pseudo-control is derived per-tick below with fresh gyro.
            #[cfg(feature = "outer_mpc_full")]
            {
                torque_ref = cmd.torque_n_m;
            }
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

        // 5b. Re-read params when version changes (disarmed only).
        //     Catches: shell `param set` + `param save`,
        //     `param defaults`, or any other param writer.
        //     Updates INDI effectiveness AND KF motor dynamics from the same
        //     persistent params — single source of truth.
        if !armed {
            let current_ver =
                crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
            if current_ver != local_param_ver {
                local_param_ver = current_ver;
                let reloaded = crate::params::get();
                let motors = reloaded.airframe.motors;
                // Same application function as boot — one code path.
                let report = indi.apply_effectiveness_params(
                    &motors,
                    &reloaded.indi.effectiveness,
                    crate::vehicle::BAKED_THRUST_NONLINEARITY,
                );
                log_effectiveness_report(&report);
                // The α-mode inertia is deliberately NOT reloaded here:
                // `inertia_kg_m2` is reboot-flagged, and the geometric G1
                // torque rows above were derived from the boot tensor —
                // swapping only the α inertia would put the two halves of
                // the same inner loop on different tensors. The outer
                // loop's model rebuild pins the boot airframe for the
                // same reason.
                for (i, est) in rpm_estimators.iter_mut().enumerate() {
                    // Filter tuning first, preserving the state estimate —
                    // sweeping a noise value on the bench should not throw
                    // away a converged c_m.
                    let tau = if report.motor_dynamics_ok[i] {
                        motors[i].time_const_s
                    } else {
                        DEFAULT_MOTOR_TAU_S
                    };
                    est.set_config(build_rpm_estimator_config(
                        i,
                        tau,
                        motors[i].nonlinearity,
                        &reloaded.indi.rpm_estimator,
                        loop_rate_hz,
                    ));
                    // Same per-motor validity gate as boot: never feed a
                    // zero/NaN tau or omega into the KF. This one re-seeds
                    // the covariance, so it stays gated.
                    if report.motor_dynamics_ok[i] {
                        est.reconfigure(
                            motors[i].time_const_s,
                            motors[i].max_omega_rad_s,
                            motors[i].nonlinearity,
                        );
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
        //        `batt_nominal_v`, which carries us until power_task's
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
                let held_ms = now.saturating_duration_since(t0).as_millis() as u32;
                defmt::info!("INDI: voltage recovered after {}ms", held_ms);
                // Duration first, then the flag: a recorder that
                // observes the true -> false edge is then guaranteed to
                // read this episode's length, not the previous one's.
                VOLTAGE_STALE_LAST_MS.store(held_ms, core::sync::atomic::Ordering::Release);
                VOLTAGE_STALE.store(false, core::sync::atomic::Ordering::Release);
            }
        } else if voltage_stale_since.is_none() {
            voltage_stale_since = Some(now);
            defmt::warn!(
                "INDI: voltage stale, holding last reading {}V",
                last_voltage_v,
            );
            // Same ordering rule as recovery: bump the index before
            // publishing the flag the recorder edge-detects on.
            VOLTAGE_STALE_EPISODES.fetch_add(1, core::sync::atomic::Ordering::Release);
            VOLTAGE_STALE.store(true, core::sync::atomic::Ordering::Release);
        }
        // Table-mode voltage failsafe (re-enabled — this was commented out
        // while its doc comment described it as active): a linearization
        // driven by a voltage reading stale past VOLTAGE_FAILSAFE_TIMEOUT
        // is unreliable enough that flying further is more dangerous than
        // landing. Going silent lets the controller watchdog disarm, same
        // pattern as CMD_STALE_TIMEOUT. Analytic models ignore voltage —
        // suppressed for them (compile-time constant fold).
        if armed
            && matches!(crate::vehicle::BAKED_THRUST_MODEL, ThrustModel::Table(_))
            && voltage_stale_since
                .map(|t0| now.duration_since(t0) >= VOLTAGE_FAILSAFE_TIMEOUT)
                .unwrap_or(false)
        {
            if !VOLTAGE_FAILSAFE_TRIPPED.swap(true, core::sync::atomic::Ordering::Relaxed) {
                defmt::error!(
                    "INDI: voltage stale > {}ms while armed (Table mode) — going silent, \
                     watchdog will disarm",
                    VOLTAGE_FAILSAFE_TIMEOUT.as_millis() as u32,
                );
                INNER_SILENT_CAUSE.store(
                    crate::blackbox::topics::events::SILENT_CAUSE_VOLTAGE_STALE as u8,
                    core::sync::atomic::Ordering::Release,
                );
            }
            continue;
        }
        // //
        // Motor-state source:
        //   - Armed + LPF has a sample → feed dshot-derived ω, ω̇ from the
        //     task-level biquad + finite difference. The LPF runs
        //     unconditionally so External data is always available, and the
        //     Internal du-based fallback reads `prev_du` / `prev_omega_fs`
        //     which are not zeroed by the arm transition.
        //   - Disarmed or first iteration: fall back to Internal.
        //   - All motors stale → Internal as well: `omega_fs` is then a
        //     biquad settling onto a ZOH of values old enough that the
        //     tracker gave up on them, which is worse than the du-based
        //     model it replaces.
        //
        // Deliberately NOT the go-silent protocol. Losing RPM telemetry
        // does not make the vehicle uncontrollable — `MotorState::Internal`
        // is a complete fallback that reconstructs ω̇ from the commanded
        // increment and the motor model (the same du-based path indiflight
        // takes with `useRpmDotFeedback = 0`). Going silent here would
        // trade a degraded but flyable controller for a guaranteed
        // disarm-in-flight. Genuine loss of control is still caught by the
        // command-staleness and non-finite-output paths above.
        if rpm_all_stale != rpm_all_stale_prev {
            if rpm_all_stale {
                defmt::error!(
                    "INDI: all motor RPM telemetry stale — G2 off, ω̇ from motor model"
                );
            } else {
                defmt::info!("INDI: motor RPM telemetry recovered");
            }
            rpm_all_stale_prev = rpm_all_stale;
        }
        let motor_state = if armed && omega_fs_has_prev && !rpm_all_stale {
            MotorState::External {
                omega_fs: &omega_fs,
                omega_dot_fs: &omega_dot_fs,
            }
        } else {
            MotorState::Internal
        };
        // Reduced outer loop: rate-setpoint entry (rate gains inside
        // INDI). Full outer loop: α entry — the rate loop lives in the
        // NMPC, no rate gains anywhere in this path; with INDI disabled
        // (`build: indi: no`) this degrades to model-based static
        // inversion of α_d (the "NMPC w/o INDI" ablation).
        #[cfg(not(feature = "outer_mpc_full"))]
        let (output, _step_state) = indi.step(
            &gyro_corrected,
            &accel_corrected,
            &rate_ref,
            spf_sp_z,
            armed,
            &g2_valid,
            motor_state,
            voltage_v,
        );
        // α pseudo-control from the held torque with the FRESH gyro:
        // α = I⁻¹·(τ_d − ω×Iω), full-tensor Euler form. Evaluating
        // ω×Iω here (not at the solve instant) matches paper Fig. 3's
        // inner-loop placement of eq. (32).
        #[cfg(feature = "outer_mpc_full")]
        let alpha_sp = {
            let g = &gyro_corrected;
            inv_inertia * (torque_ref - g.cross(&(inertia * g)))
        };
        #[cfg(feature = "outer_mpc_full")]
        let (output, _step_state) = indi.step_alpha(
            &gyro_corrected,
            &accel_corrected,
            &alpha_sp,
            spf_sp_z,
            armed,
            &g2_valid,
            motor_state,
            voltage_v,
        );

        // 7. Non-finite guard — skip publishing, stay alive.
        //    Transient NaN from WLS is recoverable; sustained NaN causes
        //    the watchdog heartbeat to stop → failsafe disarm.
        if !output.motor_commands.iter().all(|v| v.is_finite()) {
            continue;
        }

        // 7b. Sustained-WLS-NaN failsafe. On a WLS NaN exit the controller
        //     substitutes a *finite* decayed hold (u_state·0.95), so the
        //     guard above never sees the failure. That hold is the right
        //     bridge for a transient glitch, but once `nan_failsafe`
        //     asserts (nan_limit consecutive NaN exits, ~2.5 ms at 8 kHz)
        //     it is no longer a controller output — publishing it would
        //     keep the watchdog heartbeat alive while the motors decay to
        //     idle *armed*, and the failsafe would never fire. Go silent
        //     instead and let the watchdog disarm, per the "WLS produces
        //     NaN" path in docs/safety_protocol.md. Armed-gated like
        //     CMD_STALE_TIMEOUT: while disarmed the heartbeat must keep
        //     running or the vehicle becomes un-armable (rule 7 — no
        //     implicit arming gates), and the decayed hold is harmless
        //     behind DShot's armed gate.
        if armed && output.nan_failsafe {
            if !WLS_NAN_FAILSAFE_TRIPPED.swap(true, core::sync::atomic::Ordering::Relaxed) {
                defmt::error!(
                    "INDI: WLS NaN persisted past nan_limit while armed — going silent, \
                     watchdog will disarm"
                );
                INNER_SILENT_CAUSE.store(
                    crate::blackbox::topics::events::SILENT_CAUSE_WLS_NAN as u8,
                    core::sync::atomic::Ordering::Release,
                );
            }
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
        {
            let end = cortex_m::peripheral::DWT::cycle_count();
            let period = if probe_prev_start == 0 {
                0
            } else {
                probe_start.wrapping_sub(probe_prev_start)
            };
            probe_prev_start = probe_start;
            step_stats::record(end.wrapping_sub(probe_start), period);
        }

        // Feed this frame's throttle commands into the estimators so the
        // FOPDT model can account for transport delay on the next decode.
        // Only when the KF is active (armed).
        if armed {
            for (i, est) in rpm_estimators.iter_mut().enumerate() {
                est.push_throttle(
                    est_current_ts,
                    EstNormalizedThrottle::new_clamped(output.motor_commands[i]),
                );
            }
        }

        // 9. Publish telemetry (at reduced rate — every OUTER_DECIMATION
        //    frames, or every SYSID_TELEM frames for the blackbox mirrors
        //    when the `sysid` record-set tier is active).
        //
        // The tier is re-read once per fast-telemetry wrap (~500 Hz), so
        // a `blackbox set sysid` from the shell takes effect within ~2 ms
        // without re-arming; the load is an AtomicU8 + match, invisible
        // at that cadence. A fast tick coinciding with the 100 Hz tick
        // collapses into the single `||` below — never a double publish.
        sysid_telem_counter += 1;
        let blackbox_telem_tick = if sysid_telem_counter >= sysid_telem_decimation {
            sysid_telem_counter = 0;
            crate::blackbox::record_set::current().fast_indi_telem()
        } else {
            false
        };
        let blackbox_telem_tick = blackbox_telem_tick || outer_counter == 1;

        if blackbox_telem_tick {
            motor_telem_pub.publish_immediate(msgs::ActuatorMotors {
                timestamp: publish_time,
                motor_commands,
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

            // Body-rate tracking error: `rate_ref - gyro_corrected`.
            // `gyro_corrected` is the post-RPM-notch shadow — the same
            // signal the WLS allocator saw, so the error matches what
            // INDI actually acted on.
            tracking_err_pub.publish_immediate(super::TrackingError {
                timestamp: publish_time,
                pos_err: Vector3::zeros(),
                vel_err: Vector3::zeros(),
                attitude_err: Vector3::zeros(),
                body_rate_err: rate_ref - gyro_corrected,
                source: super::TRACKING_ERROR_SOURCE_INDI,
            });
        }

        if outer_counter == 1 {
            att_pub.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: publish_time,
                collective_thrust_n,
                attitude_quaternion: telem_attitude,
                body_rate_rad_s: rate_ref,
                torque_n_m: Vector3::zeros(), // INDI doesn't compute explicit torque
            });
            // Processed motor RPM: post-KF.
            let processed_motors: [msgs::DshotMotorTelemetry; 4] = core::array::from_fn(|i| {
                let omega = if armed {
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

            health_pub.publish_immediate(super::DshotHealth {
                timestamp: publish_time,
                frames: health_frames,
                motors: health,
            });
            // Surface health over defmt without needing a channel consumer.
            // Only motors whose error buckets advanced are reported, so a
            // clean link produces no output at all. `no_fresh` is excluded
            // from the trigger (expected traffic) but printed for context —
            // it is the denominator that makes the others interpretable.
            health_log_counter = health_log_counter.saturating_add(1);
            if health_log_counter >= HEALTH_LOG_DECIMATION {
                health_log_counter = 0;
                for i in 0..NU {
                    let errs = health[i]
                        .no_reply
                        .saturating_add(health[i].range_reject)
                        .saturating_add(health[i].slew_reject)
                        .saturating_add(health[i].nis_reject)
                        .saturating_add(health[i].edt);
                    if errs != health_log_prev[i] {
                        defmt::info!(
                            "dshot m{}: frames {} passed {} | no_reply {} no_fresh {} \
                             range {} slew {} nis {} edt {}",
                            i,
                            health_frames,
                            health[i].passed,
                            health[i].no_reply,
                            health[i].no_fresh,
                            health[i].range_reject,
                            health[i].slew_reject,
                            health[i].nis_reject,
                            health[i].edt,
                        );
                        health_log_prev[i] = errs;
                    }
                }
            }
        }
    }
}
