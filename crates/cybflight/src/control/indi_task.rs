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

#[cfg(not(feature = "outer_mpc"))]
use cybflight_core::attitude_control::{self, AttitudeControlOutput, geometric_controller};
use cybflight_core::{
    indi::{
        controller::{IndiConfig, IndiController, NU},
        effectiveness::IndiMotorParams,
        learner::{LearnedParams, Learner, LearnerConfig, LearnerInput},
        linearization::ThrustModel,
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
    sensors::{DSHOT_TELEMETRY, IMU_1},
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
///
/// Changing this is a global decision for the airframe; the meaning of `k`
/// differs between models, so `indi_effectiveness.nonlinearity` typically
/// needs re-identification after switching.
// const THRUST_MODEL: ThrustModel = ThrustModel::Quadratic;
const THRUST_MODEL: ThrustModel = ThrustModel::SqrtSquared;

/// Default motor nonlinearity `k`, matched to `THRUST_MODEL`.
///
/// Identified from the A2RL 6S thrust map in `tmp/thrust_map/a2rl_0114.csv`
/// by `identify_indi_k.py` (pooled fit across 21.7–24.8 V, 2500 samples):
///   Quadratic:    k = 0.518  (RMS 0.272 N, R² 0.9946)
///   SqrtSquared:  k = 0.458  (RMS 0.230 N, R² 0.9961)
///
/// Only used when no learned params exist in flash (fresh install).
const THRUST_NONLINEARITY: f32 = match THRUST_MODEL {
    ThrustModel::Quadratic => 0.518,
    ThrustModel::SqrtSquared => 0.458,
};

/// Default INDI motor parameters.
const MAX_RPM: f32 = 27000.0;
const TIME_CONSTANT: f32 = 0.015;

const INDI_MOTOR_PARAMS: [IndiMotorParams; NU] = [
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: 0.0,
    },
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: 0.0,
    },
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: 0.0,
    },
    IndiMotorParams {
        time_const_s: TIME_CONSTANT,
        max_rpm: MAX_RPM,
        g2_yaw: 0.0,
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

    // --- Slew rate limiter state (always on, protects both KF and raw path) ---
    // Max omega bound from configured motor params (use learned if available)
    let max_omega_bound: f32 = if let Some(ref saved) = saved_learned {
        saved.max_omega.iter().cloned().fold(0.0f32, f32::max)
    } else {
        INDI_MOTOR_PARAMS[0].max_rpm / 60.0 * core::f32::consts::TAU
    };
    // Minimum plausible motor time constant (conservative lower bound)
    const TAU_MIN_BOUND: f32 = 0.005;
    let slew_max_rate: f32 = max_omega_bound / TAU_MIN_BOUND; // rad/s²
    let mut slew_prev_omega = SVector::<f32, NU>::zeros();
    let mut slew_prev_time: [Option<Instant>; NU] = [None; NU];
    // Local param version — re-read params when global version changes.
    let mut local_param_ver =
        crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
    // --- Subscribe to channels ---
    let mut imu_sub = IMU_1.subscriber().unwrap();
    let mut dshot_sub = DSHOT_TELEMETRY.subscriber().unwrap();
    // Armed state read from IS_ARMED atomic (set by DShot task).
    let att_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
    let motor_telem_pub = super::ACTUATOR_MOTORS_TELEM.immediate_publisher();
    let processed_dshot_pub = super::PROCESSED_DSHOT_TELEM.immediate_publisher();

    // --- State ---
    // Rate command from outer loop (cascade, MPC, or RC rate mode).
    let mut collective_thrust_n: f32 = 0.0;
    let mut rate_ref = Vector3::<f32>::zeros();
    let mut spf_sp_z: f32 = 0.0; // thrust / mass in body z
    let mut telem_attitude = UnitQuaternion::<f32>::identity();

    // armed is read from IS_ARMED atomic each frame (no channel subscription needed)
    let mut g2_valid = [false; NU];

    // --- Rate command tracking ---
    // Cache the latest RATE_COMMAND from the outer loop (cascade, MPC, or
    // RC rate mode) plus its arrival time for the staleness gate.
    let mut last_cmd_time: Option<Instant> = None;
    /// Failsafe timeout on the rate command Signal. After this many ms
    /// without a fresh command from the outer loop, the inner loop goes
    /// silent and the watchdog disarms.
    const CMD_STALE_TIMEOUT: Duration = Duration::from_millis(100);

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
            // ARM transition
            learner_prearm_latched =
                super::LEARNER_PREARM.load(core::sync::atomic::Ordering::Acquire);

            // Always reset KF state — fresh start every flight
            for est in rpm_estimators.iter_mut() {
                est.reset_state();
            }

            if learner_prearm_latched {
                // Learner prearm: geometric G1, zero G2, reset learner
                indi.reset_to_geometric(&QUADROTOR_MOTORS, &QUADROTOR_BODY, &INDI_MOTOR_PARAMS);
                learner.reset();
                raw_omega_hold = SVector::zeros();
                slew_prev_omega = SVector::zeros();
                slew_prev_time = [None; NU];
                defmt::info!("INDI: learner prearm LATCHED — KF off, G2 zeroed");
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

        // ── DShot telemetry + slew rate limiter ────────────────────────
        //
        // Decode eRPM → rad/s, then apply a physics-based slew rate limiter
        // to reject GCR decode errors. The limiter runs always, protecting
        // both the KF path and the raw-hold path.
        let mut y_meas: [Option<f32>; NU] = [None; NU];
        if let Some(telem) = dshot_sub.try_next_message_pure() {
            let now = imu.timestamp;
            for i in 0..NU {
                let raw_omega = match telem.motors[i].value {
                    TelemetryValue::Erpm(erpm) => Some(erpm as f32 * erpm_to_rads),
                    TelemetryValue::Stopped => Some(0.0),
                    TelemetryValue::Invalid | TelemetryValue::Edt(_) => None,
                };

                // Slew rate limiter: reject if change exceeds physical limit
                if let Some(omega) = raw_omega {
                    // Hard range gate
                    if omega < 0.0 || omega > max_omega_bound * 1.2 {
                        continue;
                    }
                    // Rate gate
                    if let Some(prev_t) = slew_prev_time[i] {
                        let dt_slew = now.duration_since(prev_t).as_micros() as f32 / 1_000_000.0;
                        let max_delta = slew_max_rate * dt_slew;
                        if (omega - slew_prev_omega[i]).abs() > max_delta {
                            continue;
                        }
                    }
                    slew_prev_omega[i] = omega;
                    slew_prev_time[i] = Some(now);
                    y_meas[i] = Some(omega);
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
                let erpm = libm::roundf(rpm_estimators[i].state().omega() / erpm_to_rads) as u32;
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

        // 3. Drain latest rate command from outer loop.
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

        // 4. Stale command while armed — go silent, let watchdog handle it.
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

        // 6. INDI step (8 kHz) — uses bias-corrected gyro.
        let (output, step_state) = indi.step(
            &gyro_corrected,
            &accel_corrected,
            &rate_ref,
            spf_sp_z,
            armed,
            &g2_valid,
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
                let erpm = if omega > 0.0 {
                    libm::roundf(omega / erpm_to_rads) as u32
                } else {
                    0
                };
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
        }
    }
}
