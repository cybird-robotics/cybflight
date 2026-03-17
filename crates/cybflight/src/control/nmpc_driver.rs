use cybflight_core::nmpc::{self, NmpcCommand, NmpcState, N};
use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::{SVector, UnitQuaternion, Vector3};

use cybflight_msgs as msgs;
use crate::sensors;

pub struct NmpcDriver {
    solver: nmpc::NmpcSolver,
}

impl NmpcDriver {
    pub fn new() -> Self {
        Self {
            solver: nmpc::NmpcSolver::new().with_timer(|| Instant::now().as_micros()),
        }
    }

    pub async fn run(&mut self) -> ! {
        let mut att_sub = sensors::VEHICLE_ATTITUDE.subscriber().unwrap();
        let publisher = super::OCP_SOLVER_OUTPUT.immediate_publisher();

        let mass = self.solver.model.mass;
        let grav = self.solver.model.grav;

        // Stub position and velocity — zeros until VICON_POSE is wired up.
        // TODO: subscribe to sensors::VICON_POSE and track last-known state.
        let position = Vector3::zeros();
        let velocity = Vector3::zeros();

        // Stub reference: hover at z = 1 m, level attitude, zero velocity.
        // TODO: subscribe to super::NMPC_SETPOINT for a live target.
        let x_ref = NmpcState {
            position: Vector3::new(0.0, 0.0, 1.0),
            orientation: UnitQuaternion::identity(),
            velocity: Vector3::zeros(),
        };
        let u_ref = NmpcCommand {
            thrust: mass * grav,
            omega: Vector3::zeros(),
        };
        let x_refs: [NmpcState; N + 1] = core::array::from_fn(|_| x_ref.clone());
        let u_refs: [NmpcCommand; N] = core::array::from_fn(|_| u_ref.clone());

        loop {
            let att = match att_sub.next_message().await {
                WaitResult::Message(m) => m,
                WaitResult::Lagged(n) => {
                    defmt::warn!("NMPC: dropped {} attitude updates", n);
                    continue;
                }
            };

            let x_init = NmpcState {
                position,
                orientation: att.orientation,
                velocity,
            };

            let t0 = Instant::now();
            let (u_cmd, _cost, iterations, converged, timing) =
                self.solver.solve(&x_init, &x_refs, &u_refs);
            let solve_us =
                timing.us_fwd + timing.us_bwd_jac + timing.us_bwd_cost + timing.us_bwd_mat;

            publisher.publish_immediate(msgs::OcpSolverOutput {
                timestamp: t0,
                command: SVector::from([
                    u_cmd.thrust,
                    u_cmd.omega[0],
                    u_cmd.omega[1],
                    u_cmd.omega[2],
                ]),
                converged,
                iterations,
                solve_time_us: solve_us,
            });

            embassy_futures::yield_now().await;
        }
    }
}

impl Default for NmpcDriver {
    fn default() -> Self {
        Self::new()
    }
}

#[embassy_executor::task(pool_size = 2)]
pub async fn nmpc_task() {
    let mut driver = NmpcDriver::new();
    driver.run().await;
}
