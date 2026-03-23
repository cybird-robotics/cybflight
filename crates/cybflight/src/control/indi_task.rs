// INDI task: runs at IMU rate (~8 kHz).
//
// Subscribes to IMU_1 (raw gyro+accel), VEHICLE_ODOMETRY (for position/attitude
// controller), DSHOT_TELEMETRY (motor RPM), and AUTO_SETPOINT (position commands).
// Publishes ACTUATOR_MOTORS and telemetry.
//
// Replaces inner_loop_task as the sole controller. The position controller and
// geometric attitude controller still run here (decimated), but the rate PIDs
// and linear allocator are replaced by INDI.

use cybflight_core::{
    attitude_control::{self, geometric_controller, AttitudeControlOutput},
    indi::{
        controller::{IndiConfig, IndiController, NU},
        effectiveness::IndiMotorParams,
        rpm_tracker::RpmInput,
    },
    position_control::{self, pd_ff_control},
};
use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3};

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

        // DShot telemetry (RPM)
        if let Some(telem) = dshot_sub.try_next_message_pure() {
            let inputs: [RpmInput; NU] = core::array::from_fn(|i| {
                match telem.motors[i].value {
                    TelemetryValue::Erpm(erpm) => RpmInput::Erpm(erpm),
                    TelemetryValue::Stopped => RpmInput::Stopped,
                    TelemetryValue::Invalid | TelemetryValue::Edt(_) => RpmInput::Invalid,
                }
            });
            let (valid, rpm_failsafe) = indi.update_rpm(&inputs);
            g2_valid = valid;
            if rpm_failsafe {
                defmt::error!("INDI: all RPM telemetry lost — DISARMING");
                super::failsafe::FAILSAFE_ACTIVE.store(true, core::sync::atomic::Ordering::Release);
                crate::motors::ARM_STATE.signal(
                    msgs::ArmDisarm { timestamp: Instant::now(), armed: false },
                );
            }
        }

        // Latest odometry for position/attitude controllers
        let odom = match odom_sub.try_next_message_pure() {
            Some(o) if odom_is_valid(&o) => {
                att_state.body_rate_rad_s = o.twist.angular;
                att_state.attitude_quaternion = o.pose.orientation;
                Some(o)
            }
            _ => None,
        };

        // New position setpoint
        if let Some(sp) = super::AUTO_SETPOINT.try_take() {
            pos_setpoint.position = sp.pose.position;
            pos_setpoint.velocity = sp.twist.linear;
            pos_setpoint.yaw = extract_yaw(&sp.pose.orientation);
        }

        // 3. Outer loop (position + attitude) — decimated to ~100 Hz.
        outer_counter += 1;
        if outer_counter >= OUTER_DECIMATION {
            outer_counter = 0;

            if let Some(ref odom) = odom {
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

        // 4. INDI step (8 kHz) — uses bias-corrected gyro.
        let output = indi.step(
            &gyro_corrected,
            &accel_corrected,
            &rate_ref,
            spf_sp_z,
            armed,
            &g2_valid,
        );

        // 5. Non-finite guard.
        let any_nan = output.motor_commands.iter().any(|v| !v.is_finite());
        if any_nan {
            defmt::warn!("INDI: non-finite motor output, skipping frame");
            continue;
        }

        // 6. NaN failsafe check.
        if output.nan_failsafe {
            defmt::error!("INDI: WLS NaN limit exceeded — DISARMING");
            super::failsafe::FAILSAFE_ACTIVE.store(true, core::sync::atomic::Ordering::Release);
            crate::motors::ARM_STATE.signal(
                msgs::ArmDisarm { timestamp: Instant::now(), armed: false },
            );
        }

        // 7. Publish motor commands.
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

        // 8. Controller watchdog heartbeat.
        super::LAST_CONTROLLER_PUBLISH.lock(|c| c.set(Some(publish_time)));

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
