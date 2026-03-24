// INDI task: runs at IMU rate (~8 kHz).
//
// Subscribes to IMU_1 (raw gyro+accel), VEHICLE_ODOMETRY (for position/attitude
// controller), DSHOT_TELEMETRY (motor RPM), and AUTO_SETPOINT (position commands).
// Publishes ACTUATOR_MOTORS and telemetry.
//
// Replaces inner_loop_task as the sole controller. The position controller and
// geometric attitude controller still run here (decimated), but the rate PIDs
// and linear allocator are replaced by INDI.
//
// Safety principle: when any input is stale or output is non-finite, the task
// stops publishing ACTUATOR_MOTORS (goes silent). The failsafe controller
// watchdog detects the silence and disarms — the same pattern as RC loss.

use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    indi::{
        controller::{IndiConfig, IndiController, NU},
        effectiveness::IndiMotorParams,
        learner::{Learner, LearnerConfig, LearnerInput},
        rpm_tracker::RpmInput,
    },
    position_control::{self, pd_ff_control},
};
use embassy_time::{Duration, Instant};
use nalgebra::{Matrix2, UnitQuaternion, Vector3};

use crate::estimation::rpm_estimator::{
    NormalizedThrottle as EstNormalizedThrottle, RpmEstimator, RpmEstimatorConfigBuilder,
    StateAndCov,
};

use crate::{
    motors::ACTUATOR_MOTORS,
    msgs::{self, dshot::TelemetryValue},
    sensors::{DSHOT_TELEMETRY, IMU_1, VEHICLE_ODOMETRY},
    vehicle::{QUADROTOR_BODY, QUADROTOR_MOTORS},
};

/// Default INDI motor parameters.
const INDI_MOTOR_PARAMS: [IndiMotorParams; NU] = [
    IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: 0.0 },
    IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: 0.0 },
    IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: 0.0 },
    IndiMotorParams { time_const_s: 0.025, max_rpm: 40000.0, g2_yaw: 0.0 },
];

fn extract_yaw(q: &UnitQuaternion<f32>) -> f32 {
    let (_roll, _pitch, yaw) = q.euler_angles();
    yaw
}

fn odom_is_valid(odom: &msgs::VehicleOdometry) -> bool {
    let fin = |v: &Vector3<f32>| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let q = odom.pose.orientation.as_vector();
    fin(&odom.pose.position)
        && q.x.is_finite()
        && q.y.is_finite()
        && q.z.is_finite()
        && q.w.is_finite()
        && fin(&odom.twist.linear)
        && fin(&odom.twist.angular)
}

#[embassy_executor::task]
pub async fn indi_task() {
    // --- Build INDI controller ---
    let config = IndiConfig {
        rate_gains: Vector3::new(20.0, 20.0, 20.0),
        sync_filter_hz: 15.0,
        motors: QUADROTOR_MOTORS,
        body: QUADROTOR_BODY,
        indi_motors: INDI_MOTOR_PARAMS,
        nonlinearity: [0.5; NU],
        act_limit: [1.0; NU],
        wls_wv: [1.0, 1.0, 50.0, 50.0, 50.0, 5.0],
        wls_wu: [1.0; NU],
        wls_cond_bound: 3.2768e8,  // (1<<15) * 1e4
        wls_theta: 1e-4,
        wls_imax: 1,
        nan_limit: 20,
        rpm_invalid_limit: 50,
        rpm_all_invalid_limit: 50,
        rpm_recovery_count: 10,
        motor_pole_count: 14,
    };

    // IMU sample rate — nominal 8 kHz
    let loop_rate_hz = 8000.0f32;
    let mut indi = IndiController::new(&config, loop_rate_hz);

    // --- Per-motor RPM estimators (FOPDT EKF, one per motor) ---
    let pole_pairs = config.motor_pole_count as f32 / 2.0;
    let erpm_to_rads = core::f32::consts::TAU * 100.0 / (pole_pairs * 60.0);
    let est_config = RpmEstimatorConfigBuilder::new()
        .tau_m_up(INDI_MOTOR_PARAMS[0].time_const_s)
        .tau_m_down(INDI_MOTOR_PARAMS[0].time_const_s)
        .build()
        .unwrap();
    let max_omega = INDI_MOTOR_PARAMS[0].max_rpm * erpm_to_rads;
    let init_state = StateAndCov::new(
        0.0,
        max_omega,
        Matrix2::new(1000.0, 0.0, 0.0, max_omega * max_omega),
    );
    let mut rpm_estimators: [RpmEstimator; NU] =
        core::array::from_fn(|_| RpmEstimator::new(est_config, init_state));

    // --- G1/G2 online learner (optional, activated via RC switch) ---
    let learner_config = LearnerConfig::default();
    let mut learner = Learner::new(&learner_config, loop_rate_hz);

    // Load previously learned params from flash (if available) and apply.
    if let Some(saved_learned) = crate::params::load_learned_from_flash() {
        if indi.apply_learned_params(&saved_learned) {
            defmt::info!("INDI: loaded learned G1/G2 from flash");
        }
    }

    // Track whether we need to save learned params on disarm
    let mut was_armed = false;

    // --- Position + attitude controllers (gains from params, same as inner_loop) ---
    let params = crate::params::get();
    let g = &params.control;
    let pc = pd_ff_control::PositionController::new(
        Vector3::new(g.pos_kp[0], g.pos_kp[1], g.pos_kp[2]),
        Vector3::new(g.pos_kd[0], g.pos_kd[1], g.pos_kd[2]),
        position_control::VehicleParams {
            mass: QUADROTOR_BODY.mass_kg,
            gravity: 9.81,
        },
    );
    let ac = geometric_controller::GeometricAttitudeController::new(
        Vector3::new(g.att_k_rate[0], g.att_k_rate[1], g.att_k_rate[2]),
        Vector3::new(1.0, 1.0, 0.2),
    )
    .with_inertia(QUADROTOR_BODY.inertia_matrix());

    // --- Subscribe to channels ---
    let mut imu_sub = IMU_1.subscriber().unwrap();
    let mut odom_sub = VEHICLE_ODOMETRY.subscriber().unwrap();
    let mut dshot_sub = DSHOT_TELEMETRY.subscriber().unwrap();
    // Armed state read from IS_ARMED atomic (set by DShot task).
    let att_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
    let pos_pub = super::POSITION_CONTROL_SETPOINT.immediate_publisher();
    let motor_telem_pub = super::ACTUATOR_MOTORS_TELEM.immediate_publisher();

    // --- State ---
    let mut pos_state = position_control::PositionControlState::<f32>::default();
    let mut att_state = attitude_control::AttitudeControlState::<f32>::default();
    let mut att_ref = attitude_control::AttitudeControlSetpoint::<f32>::default();
    let mut collective_thrust_n: f32 = 0.0;
    let mut rate_ref = Vector3::<f32>::zeros();
    // armed is read from IS_ARMED atomic each frame (no channel subscription needed)
    let mut g2_valid = [false; NU];
    let mut gyro_bias = Vector3::<f32>::zeros();
    let mut accel_bias = Vector3::<f32>::zeros();

    // Specific force setpoint from position controller (thrust / mass in body z)
    let mut spf_sp_z: f32 = 0.0;

    // Latest valid odometry — persisted across inner loop iterations so the
    // decimated outer loop always has a recent sample regardless of phase offset.
    let mut latest_odom: Option<msgs::VehicleOdometry> = None;
    let mut last_odom_time: Option<Instant> = None;
    // Skip outer-loop control when odometry is older than this.
    const ODOM_STALE_TIMEOUT: Duration = Duration::from_millis(100);

    // RPM estimator timestamp tracking (seconds, f32 relative to task start)
    let mut est_prev_ts: Option<Instant> = None;
    let mut est_current_ts: f32 = 0.0;

    // Decimation counter for position/attitude controller
    let mut outer_counter: u32 = 0;
    // Position controller runs at IMU_rate / OUTER_DECIMATION
    // 8000 / 80 = 100 Hz (matching the old inner_loop rate)
    const OUTER_DECIMATION: u32 = 80;

    // Wait for first setpoint
    let sp = super::AUTO_SETPOINT.wait().await;
    let mut pos_setpoint = position_control::PositionControlSetpoint {
        position: sp.pose.position,
        velocity: sp.twist.linear,
        yaw: extract_yaw(&sp.pose.orientation),
        ..Default::default()
    };

    // Wait for ESKF to converge before entering the control loop.
    // Before convergence, gyro bias is unreliable and odometry may be off.
    defmt::info!("INDI: waiting for ESKF convergence...");
    while !crate::estimation::ESTIMATOR_READY.load(core::sync::atomic::Ordering::Acquire) {
        embassy_time::Timer::after_millis(100).await;
    }
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

        // Bias-correct raw gyro with latest ESKF estimate.
        // GYRO_BIAS is a Signal — try_take returns the latest value if updated,
        // otherwise we keep using the previous bias (changes slowly, ~1 Hz).
        if let Some(bias) = crate::estimation::GYRO_BIAS.try_take() {
            gyro_bias = bias;
        }
        if let Some(bias) = crate::estimation::ACCEL_BIAS.try_take() {
            accel_bias = bias;
        }
        let gyro_corrected = imu.gyro_rad_s - gyro_bias;
        let accel_corrected = imu.accel_m_s2 - accel_bias;

        // 2. Non-blocking reads of other channels.
        // Arming state — read from atomic (set by DShot task, single source of truth)
        let armed = crate::motors::IS_ARMED.load(core::sync::atomic::Ordering::Acquire);

        // Save learned params to flash on disarm transition (if learning was active).
        if was_armed && !armed && learner.samples() > 0 {
            let learned = learner.update(&LearnerInput {
                rate_rad_s: gyro_corrected,
                rate_dot_rad_s2: nalgebra::Vector3::zeros(),
                spf_m_s2: accel_corrected,
                omega_rad_s: [0.0; NU],
                d_commands: [0.0; NU],
                armed: false,
                touching_ground: true,
            });
            if learned.valid {
                match crate::params::save_learned_to_flash(&learned) {
                    Ok(()) => defmt::info!("INDI: saved learned G1/G2 to flash"),
                    Err(e) => defmt::warn!("INDI: failed to save learned params: {}", e),
                }
            }
        }
        was_armed = armed;

        // DShot telemetry (RPM)
        // When a new DShot frame arrives, update the per-motor estimators with
        // the measured omega.  On invalid frames the estimators coast on the
        // FOPDT model instead of reporting Invalid to the RpmTracker.
        let mut y_meas: [Option<f32>; NU] = [None; NU];
        if let Some(telem) = dshot_sub.try_next_message_pure() {
            for i in 0..NU {
                y_meas[i] = match telem.motors[i].value {
                    TelemetryValue::Erpm(erpm) => Some(erpm as f32 * erpm_to_rads),
                    TelemetryValue::Stopped => Some(0.0),
                    TelemetryValue::Invalid | TelemetryValue::Edt(_) => None,
                };
            }
        }

        // Predict (and optionally update) every IMU tick.
        for i in 0..NU {
            rpm_estimators[i].step(est_current_ts, est_dt, y_meas[i]);
        }

        // Feed estimated omega to the RpmTracker.  The estimator coasts through
        // invalid frames, so the tracker sees a smooth signal and its invalid
        // counter only increments when the estimator itself is diverging.
        let estimated_inputs: [RpmInput; NU] = core::array::from_fn(|i| {
            let erpm = libm::roundf(rpm_estimators[i].state().omega() / erpm_to_rads) as u32;
            RpmInput::Erpm(erpm)
        });
        let (valid, _rpm_failsafe) = indi.update_rpm(&estimated_inputs);
        g2_valid = valid;
        // RPM failsafe (all motors lost) is tracked by the controller core
        // but not acted on here — if RPM loss degrades output quality, the
        // controller will produce bad output → go silent → watchdog disarms.

        // Latest odometry for position/attitude controllers.
        // Persist across iterations so the decimated outer loop always has a
        // recent sample regardless of phase offset between odom and outer tick.
        // Drain all queued messages to get the most recent.
        while let Some(o) = odom_sub.try_next_message_pure() {
            if odom_is_valid(&o) {
                att_state.attitude_quaternion = o.pose.orientation;
                latest_odom = Some(o);
                last_odom_time = Some(Instant::now());
            }
        }

        // Use bias-corrected gyro for body rate (8kHz, not 100Hz odom).
        att_state.body_rate_rad_s = gyro_corrected;

        // New position setpoint
        if let Some(sp) = super::AUTO_SETPOINT.try_take() {
            pos_setpoint.position = sp.pose.position;
            pos_setpoint.velocity = sp.twist.linear;
            pos_setpoint.yaw = extract_yaw(&sp.pose.orientation);
        }

        // 3. Input recency — check all critical inputs before computing.
        //    If any is stale, skip publishing (go silent). The failsafe
        //    controller watchdog detects silence and disarms.
        let now = Instant::now();
        let odom_fresh = match last_odom_time {
            Some(t) => now.duration_since(t) < ODOM_STALE_TIMEOUT,
            None => false,
        };

        // 4. Outer loop (position + attitude) — decimated to ~100 Hz.
        //    Skipped when odometry is stale (ESKF diverged or VICON lost).
        outer_counter += 1;
        if outer_counter >= OUTER_DECIMATION {
            outer_counter = 0;

            if odom_fresh {
                if let Some(ref odom) = latest_odom {
                    pos_state.position = odom.pose.position;
                    pos_state.velocity = odom.twist.linear;
                    pos_state.attitude = odom.pose.orientation;

                    let pc_out = pc.compute(&pos_state, &pos_setpoint);
                    collective_thrust_n = pc_out.collective_thrust_n;
                    att_ref.attitude_quaternion = Some(pc_out.desired_attitude_quaternion);
                    att_ref.body_rate_rad_s = pc_out.desired_body_rate_rad_s;

                    // Convert collective thrust to specific force in body z (m/s²)
                    // In FLU: positive = up. spf = thrust / mass.
                    spf_sp_z = collective_thrust_n / QUADROTOR_BODY.mass_kg;
                }

                // Attitude controller → rate reference
                let AttitudeControlOutput {
                    body_rate_rad_s,
                    torque_n_m: _,
                } = ac.compute(&att_state, &att_ref);
                rate_ref = body_rate_rad_s;
            }
        }

        // 5. Stale odometry while armed — go silent, let watchdog handle it.
        //    INDI can't produce meaningful output without recent state feedback.
        if armed && !odom_fresh {
            continue;
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

        // 6b. Online learner (runs every frame, gated internally on armed+airborne).
        //     Uses RPM estimator predicted omega and the INDI step state.
        let omega_for_learner: [f32; NU] =
            core::array::from_fn(|i| rpm_estimators[i].state().omega());
        let learner_input = LearnerInput {
            rate_rad_s: gyro_corrected,
            rate_dot_rad_s2: step_state.rate_dot_raw,
            spf_m_s2: accel_corrected,
            omega_rad_s: omega_for_learner,
            d_commands: output.motor_commands,
            armed,
            touching_ground: step_state.touching_ground,
        };
        let learned = learner.update(&learner_input);

        // Apply learned params when the RC learning switch is active and
        // the learner has produced valid estimates. Applies at most once
        // per outer-loop cycle (~100 Hz) to avoid redundant recomputation.
        if outer_counter == 0
            && super::LEARNING_ENABLED.load(core::sync::atomic::Ordering::Acquire)
        {
            indi.apply_learned_params(&learned);
        }

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
        for i in 0..NU {
            rpm_estimators[i].push_throttle(
                est_current_ts,
                EstNormalizedThrottle::new_clamped(output.motor_commands[i]),
            );
        }

        // 9. Publish telemetry (at reduced rate — every OUTER_DECIMATION frames).
        if outer_counter == 1 {
            att_pub.publish_immediate(msgs::AttitudeControlSetpoint {
                timestamp: publish_time,
                collective_thrust_n,
                attitude_quaternion: att_ref
                    .attitude_quaternion
                    .unwrap_or(UnitQuaternion::identity()),
                body_rate_rad_s: rate_ref,
                torque_n_m: Vector3::zeros(), // INDI doesn't compute explicit torque
            });
            pos_pub.publish_immediate(msgs::PositionControlSetpoint {
                timestamp: publish_time,
                position: pos_setpoint.position,
                velocity: pos_setpoint.velocity,
                yaw: pos_setpoint.yaw,
                collective_thrust_n,
            });
            motor_telem_pub.publish_immediate(msgs::ActuatorMotors {
                timestamp: publish_time,
                motor_commands,
            });
        }
    }
}
