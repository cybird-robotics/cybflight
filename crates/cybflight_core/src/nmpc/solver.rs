// NMPC solver: adjoint backward pass + L-BFGS orchestration.
// Ported from NMPCSolver.h (FSC Lab / cyblib).

use super::lbfgs::{self, DIM, LbfgsParams, LbfgsWorkspace};
use super::model::{Control, CtrlJac, NU, QuadModel, State, StateJac};

pub const N: usize = 10; // prediction horizon

const _: () = assert!(lbfgs::DIM == N * NU, "lbfgs::DIM must equal N * NU");

// ── Persistent solver state ───────────────────────────────────────────────────

pub struct NmpcSolver {
    pub model: QuadModel,
    pub ws: LbfgsWorkspace,
    warmstart: bool,
    prev_u: [f32; DIM],
    /// Persistent simulated state — propagated forward after each solve so the
    /// next call sees a different x0 (closed-loop simulation).
    pub sim_x0: Option<State>,
}

impl NmpcSolver {
    pub fn new() -> Self {
        let mass = 1.0_f32;
        let grav = 9.81_f32;
        let mut prev_u = [0.0_f32; DIM];
        for i in 0..N {
            prev_u[i * NU] = mass * grav; // hover thrust for every step
        }
        Self {
            model: QuadModel::new(mass, grav, 0.1),
            ws: LbfgsWorkspace::zeroed(),
            warmstart: true,
            prev_u,
            sim_x0: None,
        }
    }

    // Solve the NMPC problem.
    // Returns (u_opt_first, final_cost, lbfgs_iterations, converged).
    pub fn solve(
        &mut self,
        x_init: &State,
        x_refs: &[State; N + 1],
        u_refs: &[Control; N],
    ) -> (Control, f32, i32, bool) {
        let mut u_flat = [0.0_f32; DIM];
        if self.warmstart {
            // Receding-horizon shift: u[i] = prev_u[i+1], repeat last step
            for i in 0..(N - 1) {
                for j in 0..NU {
                    u_flat[i * NU + j] = self.prev_u[(i + 1) * NU + j];
                }
            }
            for j in 0..NU {
                u_flat[(N - 1) * NU + j] = self.prev_u[(N - 1) * NU + j];
            }
            self.warmstart = false;
        } else {
            for i in 0..N {
                for j in 0..NU {
                    u_flat[i * NU + j] = u_refs[i][j];
                }
            }
        }

        let params = LbfgsParams::default_nmpc();
        let mut f_opt = 0.0_f32;

        // Borrow disjoint fields before creating the closure so the
        // borrow checker can see they don't overlap with self.ws.
        let model = &mut self.model;
        let ws = &mut self.ws;

        // Stack-allocated trajectory scratch (1680 bytes — acceptable)
        let mut traj_x: [State; N + 1] = [State::zeros(); N + 1];
        let mut grad_fx = StateJac::zeros();
        let mut grad_fu = CtrlJac::zeros();

        let x0 = *x_init;
        let xr = *x_refs;
        let ur = *u_refs;

        let mut eval = |u_flat: &[f32; DIM], grad_u_flat: &mut [f32; DIM]| -> f32 {
            // Forward pass
            traj_x[0] = x0;
            for i in 0..N {
                let uk = Control::from_fn(|j, _| u_flat[i * NU + j]);
                traj_x[i + 1] = model.propagate_rk4(&traj_x[i], &uk);
            }

            // Backward pass (adjoint)
            let mut lambda = State::zeros();
            let mut total_cost = model.terminal_cost_grad(&traj_x[N], &xr[N], &mut lambda);

            for i in (0..N).rev() {
                let uk = Control::from_fn(|j, _| u_flat[i * NU + j]);
                let mut gx_cost = State::zeros();
                let mut gu_cost = Control::zeros();
                let mut gx_con = State::zeros();
                let mut gu_con = Control::zeros();

                total_cost += model.path_cost_grad(
                    &traj_x[i],
                    &uk,
                    &xr[i],
                    &ur[i],
                    &mut gx_cost,
                    &mut gu_cost,
                );
                total_cost += model.constraint_grad(&uk, &mut gx_con, &mut gu_con);

                model.propagate_rk4_grad(&traj_x[i], &uk, &mut grad_fx, &mut grad_fu);

                let gu = gu_cost + gu_con + grad_fu.transpose() * lambda;
                for j in 0..NU {
                    grad_u_flat[i * NU + j] = gu[j];
                }

                lambda = grad_fx.transpose() * lambda + gx_cost + gx_con;
            }

            total_cost
        };

        let ret = lbfgs::lbfgs_optimize(&mut u_flat, &mut f_opt, &params, ws, &mut eval);

        self.prev_u.copy_from_slice(&u_flat);
        self.warmstart = true;

        let u_first = Control::from_fn(|j, _| u_flat[j]);
        let iters = self.ws.last_k;
        let converged = ret == lbfgs::LBFGS_CONVERGENCE || ret == lbfgs::LBFGS_STOP;
        (u_first, f_opt, iters, converged)
    }
}
