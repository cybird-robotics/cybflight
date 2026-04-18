//! MPC outer-loop task: 100 Hz position/attitude controller using
//! `SimpleSqpSolver` over `QuadModel`. Publishes body-rate + collective-thrust
//! commands to `super::RATE_COMMAND` for the INDI inner loop to consume.
//!
//! Lives in its own embassy task to avoid blocking the 8 kHz INDI loop.
//! Stack-allocated locals are tiny (~1 KB); the ~32 KB SQP workspace lives
//! in BSS via `static_cell::StaticCell`.
//!
//! Gated on `cfg(feature = "outer_mpc")`.

#![cfg(feature = "outer_mpc")]

use cybflight_core::mpc::quad_model::{N as MPC_N, NU as MPC_NU, NX as MPC_NX};
use cybflight_core::mpc::{QuadModel, SimpleQuadProblem, SimpleSqpSolver};
use cybflight_core::rotation::quaternion_to_yaw;
use embassy_time::{Duration, Instant, Ticker};
use nalgebra::{SVector, Vector3};
use static_cell::StaticCell;

type MpcStateVec = SVector<f32, MPC_NX>;
type MpcInputVec = SVector<f32, MPC_NU>;

use crate::msgs;
use crate::sensors::VEHICLE_ODOMETRY;
use crate::vehicle::QUADROTOR_BODY;
use nalgebra::UnitQuaternion;

/// Static-allocated SQP workspace (~32 KB in BSS, init-once at task startup).
static MPC_SOLVER: StaticCell<SimpleSqpSolver> = StaticCell::new();

/// Maximum age of an odometry sample (against its own timestamp) we will use
/// as the MPC initial state. ESKF divergence often produces valid-looking
/// (finite) but stale odometry; without this gate the MPC would happily plan
/// from an ancient pose. Matches the cascade path's `ODOM_STALE_TIMEOUT` in
/// `indi_task.rs`.
const ODOM_STALE_TIMEOUT: Duration = Duration::from_millis(100);

/// Wall-clock budget for one SQP solve. The solver is synchronous (no
/// `with_timeout` possible) so this is a *post-hoc* check: if a solve exceeds
/// the budget we discard its output and refuse to publish, on the theory that
/// (a) the command is now stale relative to the 100 Hz tick, and (b) a solve
/// that ran long is more likely to have diverged. Persistent overruns will
/// trip the inner loop's `MPC_CMD_STALE_TIMEOUT` and the failsafe watchdog.
// const MPC_SOLVE_BUDGET: Duration = Duration::from_millis(8);

/// Reject odometry with any non-finite component.
fn odom_is_valid(odom: &msgs::VehicleOdometry) -> bool {
    let fin = |v: &Vector3<f32>| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let q = odom.pose.orientation.as_vector();
    fin(&odom.pose.position)
        && q.x.is_finite()
        && q.y.is_finite()
        && q.z.is_finite()
        && q.w.is_finite()
        && fin(&odom.twist.linear)
}

/// Clamp each component of `u0` into the model's per-channel control bounds.
fn clamp_mpc_output(u0: &mut MpcInputVec, u_bounds: &[[f32; 2]; MPC_NU]) {
    for i in 0..MPC_NU {
        u0[i] = u0[i].clamp(u_bounds[i][0], u_bounds[i][1]);
    }
}

#[embassy_executor::task]
pub async fn control_loop_task() {
    // ── Construct MPC ──────────────────────────────────────────────────
    let params = crate::params::get();
    let mpc_solver: &mut SimpleSqpSolver = MPC_SOLVER.init(SimpleSqpSolver::new());
    let mut mpc_problem =
        SimpleQuadProblem::with_rk4(QuadModel::from_vehicle_params(&params), MPC_N);

    // ── Reference + warm-start trajectories ────────────────────────────
    let mut hover_thrust = QUADROTOR_BODY.mass_kg * 9.81;
    let identity_x = {
        let mut x = MpcStateVec::zeros();
        x[6] = 1.0; // qw = 1 (identity quaternion in xyzw layout)
        x
    };
    let mut x_refs: [MpcStateVec; MPC_N + 1] = [identity_x; MPC_N + 1];
    let hover_u = MpcInputVec::from_row_slice(&[hover_thrust, 0.0, 0.0, 0.0]);
    let mut u_refs: [MpcInputVec; MPC_N] = [hover_u; MPC_N];
    let mut u_warm: [MpcInputVec; MPC_N] = u_refs;

    // ── Publishers / Subscribers ──────────────────────────────────────
    let pos_ctrl_pub = super::POSITION_CONTROL_SETPOINT.immediate_publisher();
    let att_ctrl_pub = super::ATTITUDE_CONTROL_SETPOINT.immediate_publisher();
    let ocp_pub = super::OCP_SOLVER_OUTPUT.immediate_publisher();
    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("outer_loop: VEHICLE_ODOMETRY subscriber");

    // ── Wait for the first AUTO_SETPOINT and ESKF convergence ──────────
    let sp = super::AUTO_SETPOINT.wait().await;
    let mut pos_setpoint: Vector3<f32> = sp.pose.position;
    let mut att_setpoint: nalgebra::UnitQuaternion<f32> = sp.pose.orientation;
    while !crate::estimation::ESTIMATOR_READY.load(core::sync::atomic::Ordering::Acquire) {
        embassy_time::Timer::after_millis(100).await;
    }
    defmt::info!("MPC outer loop task started (100 Hz)");

    // ── Param hot-reload bookkeeping (mirror of indi_task pattern) ────
    let mut local_param_ver =
        crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);

    // ── 100 Hz tick loop ───────────────────────────────────────────────
    let mut ticker = Ticker::every(Duration::from_millis(10));
    loop {
        ticker.next().await;

        // 1. Drain new position setpoint (RC joystick → target).
        if let Some(sp) = super::AUTO_SETPOINT.try_take() {
            pos_setpoint = sp.pose.position;
            att_setpoint = sp.pose.orientation;
        }

        // 2. Hot-reload params when disarmed (mirrors indi_task's pattern).
        let armed = crate::motors::IS_ARMED.load(core::sync::atomic::Ordering::Acquire);
        if !armed {
            let cur = crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
            if cur != local_param_ver {
                local_param_ver = cur;
                let np = crate::params::get();
                mpc_problem =
                    SimpleQuadProblem::with_rk4(QuadModel::from_vehicle_params(&np), MPC_N);
                hover_thrust = np.body.mass_kg * 9.81;
                let hover_u = MpcInputVec::from_row_slice(&[hover_thrust, 0.0, 0.0, 0.0]);
                u_refs = [hover_u; MPC_N];
                u_warm = u_refs;
                defmt::info!("MPC outer loop: params reloaded (ver {})", cur);
            }
        }

        // 3. Drain latest valid odometry (skip the tick if none arrived).
        //    Validity = finite components AND timestamp within
        //    `ODOM_STALE_TIMEOUT` of now (and not future-dated). The timestamp
        //    gate is the C3 fix: an ESKF that hangs while still publishing
        //    finite values must not feed the MPC an ancient initial state.
        let now_for_odom = Instant::now();
        let mut latest = None;
        while let Some(o) = odom_sub.try_next_message_pure() {
            if !odom_is_valid(&o) {
                continue;
            }
            if o.timestamp > now_for_odom {
                // Future-dated → clock skew or corruption. Reject.
                continue;
            }
            if now_for_odom.duration_since(o.timestamp) > ODOM_STALE_TIMEOUT {
                continue;
            }
            latest = Some(o);
        }
        let Some(odom) = latest else {
            defmt::warn!("MPC outer loop: no fresh odometry, skipping tick");
            continue;
        };

        // 4. Build MPC initial state from odometry.
        // QuadModel state = [px, py, pz, qx, qy, qz, qw, vx, vy, vz].
        let q = odom.pose.orientation;
        let mpc_x0 = MpcStateVec::from_row_slice(&[
            odom.pose.position.x,
            odom.pose.position.y,
            odom.pose.position.z,
            q.i,
            q.j,
            q.k,
            q.w,
            odom.twist.linear.x,
            odom.twist.linear.y,
            odom.twist.linear.z,
        ]);

        // 5. Refresh reference position (identity quat + zero velocity were
        //    set at task init).
        for x_ref in x_refs.iter_mut() {
            x_ref[0] = pos_setpoint.x;
            x_ref[1] = pos_setpoint.y;
            x_ref[2] = pos_setpoint.z;
        }

        // 6. Solve one SQP iteration (max_iters = 1, matches host benchmark).
        let solve_start = Instant::now();
        let result = mpc_solver.solve(&mpc_problem, &mpc_x0, &x_refs, &u_refs, &u_warm, 1, 1e-3);
        let solve_time_us = Instant::now().duration_since(solve_start).as_micros();
        u_warm = *mpc_solver.u_bar();
        let mut u0 = mpc_solver.u_bar()[0];

        // 7. Non-finite guard — skip publishing on NaN/Inf. The inner loop's
        //    last_mpc_cmd_time staleness check will eventually trip the
        //    failsafe if this persists.
        if !u0.iter().all(|v| v.is_finite()) {
            defmt::warn!("MPC outer loop: non-finite output, skipping tick");
            // Reset warm-start so a transient NaN does not poison the next
            // iteration via u_warm.
            u_warm = u_refs;
            continue;
        }

        // 7b. Bounds clamp (C1) — even with a finite solve, a diverged or
        //     numerically degraded interior can emit values outside the
        //     physical envelope. The SQP penalty (rho) makes constraint
        //     violation costly but does not enforce hard feasibility, so we
        //     project explicitly onto the model's `u_bounds` before letting
        //     the command reach INDI.
        clamp_mpc_output(&mut u0, &mpc_problem.model.u_bounds);

        // 8. Extract predicted attitude from the SECOND MPC state (index 1).
        //    x_bar[0] is the current/measured state; x_bar[1] is the
        //    one-step-ahead prediction — the attitude reference the MPC is
        //    driving toward.
        let x1 = &mpc_solver.x_bar()[1];
        let att_ref = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            x1[6], // qw (scalar-last layout in state, scalar-first in nalgebra ctor)
            x1[3], // qx
            x1[4], // qy
            x1[5], // qz
        ));
        let yaw_ref = quaternion_to_yaw(&att_ref, 0.0);

        let publish_time = Instant::now();

        // 9. Publish to the inner loop.
        super::RATE_COMMAND.signal(msgs::AttitudeControlSetpoint {
            timestamp: publish_time,
            collective_thrust_n: u0[0],
            attitude_quaternion: att_setpoint,
            body_rate_rad_s: Vector3::new(u0[1], u0[2], u0[3]),
            torque_n_m: Vector3::zeros(),
        });

        // 10. Publish telemetry for downlink (fulfils the promise in indi_task's
        //     comment that the MPC path delegates these to outer_loop).
        pos_ctrl_pub.publish_immediate(msgs::PositionControlSetpoint {
            timestamp: publish_time,
            position: pos_setpoint,
            velocity: Vector3::new(x1[7], x1[8], x1[9]),
            yaw: yaw_ref,
        });
        att_ctrl_pub.publish_immediate(msgs::AttitudeControlSetpoint {
            timestamp: publish_time,
            collective_thrust_n: u0[0],
            attitude_quaternion: att_ref,
            body_rate_rad_s: Vector3::new(u0[1], u0[2], u0[3]),
            torque_n_m: Vector3::zeros(),
        });
        ocp_pub.publish_immediate(msgs::OcpSolverOutput {
            timestamp: publish_time,
            command: nalgebra::SVector::from(u0),
            iterations: result.iters as i32,
            converged: result.converged,
            solve_time_us,
        });
    }
}
