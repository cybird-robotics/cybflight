use cybflight_core::nmpc;
use embassy_time::Instant;

use crate::msgs;

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
        let publisher = super::OCP_SOLVER_OUTPUT.immediate_publisher();
        loop {
            let res = nmpc::run_once(&mut self.solver);
            let solve_us = res.timing.us_fwd
                + res.timing.us_bwd_jac
                + res.timing.us_bwd_cost
                + res.timing.us_bwd_mat;
            publisher.publish_immediate(msgs::OcpSolverOutput {
                timestamp: Instant::now(),
                command: res.u_opt.into(),
                converged: res.converged,
                iterations: res.iterations,
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
