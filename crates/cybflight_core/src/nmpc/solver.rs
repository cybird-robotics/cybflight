// NMPC solver: adjoint backward pass + L-BFGS orchestration.
// Ported from NMPCSolver.h (FSC Lab / cyblib).

use super::lbfgs::{self, LbfgsParams, LbfgsWorkspace, DIM};
use super::model::{Control, CtrlJac, QuadModel, State, StateJac, NU};
use core::assert;
use core::default::Default;
use core::option::{
    Option,
    Option::{None, Some},
};

/// Per-phase timing from one `solve()` call (cumulative across all eval invocations).
pub struct NmpcTiming {
    /// µs spent in forward pass (propagate_rk4 × N, all eval calls combined).
    pub us_fwd: u64,
    /// µs spent in propagate_rk4_grad (adjoint Jacobians) in the backward pass.
    pub us_bwd_jac: u64,
    /// µs spent in path_cost_grad + constraint_grad + terminal_cost_grad.
    pub us_bwd_cost: u64,
    /// µs spent in matrix multiplies inside the backward pass.
    pub us_bwd_mat: u64,
    /// Total number of eval() invocations (1 initial + line-search count).
    pub eval_count: u32,
}

pub const N: usize = 20; // prediction horizon

/// If true, use the fast single-stage Euler sensitivity in the backward pass
/// (~10× cheaper than RK4 sensitivity, slight gradient approximation error).
/// If false, use the full 4-stage RK4 sensitivity (accurate but slow).
const USE_EULER_GRAD: bool = true;

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
    /// Xorshift32 PRNG state for sensor noise simulation.
    pub rng_state: u32,
    pub timer_us: Option<fn() -> u64>, // Optional timer function for benchmarking (returns µs)
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
            model: QuadModel::new(mass, grav, 0.05),
            ws: LbfgsWorkspace::zeroed(),
            warmstart: true,
            prev_u,
            sim_x0: None,
            rng_state: 0xdeadbeef_u32,
            timer_us: None,
        }
    }

    pub fn with_timer(mut self, timer_fn: fn() -> u64) -> Self {
        self.timer_us = Some(timer_fn);
        self
    }

    // Solve the NMPC problem.
    // Returns (u_opt_first, final_cost, lbfgs_iterations, converged, timing).
    pub fn solve(
        &mut self,
        x_init: &State,
        x_refs: &[State; N + 1],
        u_refs: &[Control; N],
    ) -> (Control, f32, i32, bool, NmpcTiming) {
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

        let mut us_fwd = 0u64;
        let mut us_bwd_jac = 0u64;
        let mut us_bwd_cost = 0u64;
        let mut us_bwd_mat = 0u64;
        let mut eval_count = 0u32;

        let mut eval = |u_flat: &[f32; DIM], grad_u_flat: &mut [f32; DIM]| -> f32 {
            eval_count += 1;

            // Forward pass
            traj_x[0] = x0;
            let t = self.timer_us.map_or(0, |f| f());
            for i in 0..N {
                let uk = Control::from_fn(|j, _| u_flat[i * NU + j]);
                traj_x[i + 1] = model.propagate_rk4(&traj_x[i], &uk);
            }
            us_fwd += self.timer_us.map_or(0, |f| f()) - t;

            // Backward pass (adjoint)
            let mut lambda = State::zeros();
            let t = self.timer_us.map_or(0, |f| f());
            let mut total_cost = model.terminal_cost_grad(&traj_x[N], &xr[N], &mut lambda);
            us_bwd_cost += self.timer_us.map_or(0, |f| f()) - t;

            for i in (0..N).rev() {
                let uk = Control::from_fn(|j, _| u_flat[i * NU + j]);
                let mut gx_cost = State::zeros();
                let mut gu_cost = Control::zeros();
                let mut gx_con = State::zeros();
                let mut gu_con = Control::zeros();

                let t = self.timer_us.map_or(0, |f| f());
                total_cost += model.path_cost_grad(
                    &traj_x[i],
                    &uk,
                    &xr[i],
                    &ur[i],
                    &mut gx_cost,
                    &mut gu_cost,
                );
                total_cost += model.constraint_grad(&uk, &mut gx_con, &mut gu_con);
                us_bwd_cost += self.timer_us.map_or(0, |f| f()) - t;

                let t = self.timer_us.map_or(0, |f| f());
                if USE_EULER_GRAD {
                    model.propagate_euler_grad(&traj_x[i], &uk, &mut grad_fx, &mut grad_fu);
                } else {
                    model.propagate_rk4_grad(&traj_x[i], &uk, &mut grad_fx, &mut grad_fu);
                }
                us_bwd_jac += self.timer_us.map_or(0, |f| f()) - t;

                let t = self.timer_us.map_or(0, |f| f());
                let gu = gu_cost + gu_con + grad_fu.transpose() * lambda;
                for j in 0..NU {
                    grad_u_flat[i * NU + j] = gu[j];
                }
                lambda = grad_fx.transpose() * lambda + gx_cost + gx_con;
                us_bwd_mat += self.timer_us.map_or(0, |f| f()) - t;
            }

            total_cost
        };

        let ret = lbfgs::lbfgs_optimize(&mut u_flat, &mut f_opt, &params, ws, &mut eval);

        self.prev_u.copy_from_slice(&u_flat);
        self.warmstart = true;

        let u_first = Control::from_fn(|j, _| u_flat[j]);
        let iters = self.ws.last_k;
        let converged = ret == lbfgs::LBFGS_CONVERGENCE || ret == lbfgs::LBFGS_STOP;
        let timing = NmpcTiming {
            us_fwd,
            us_bwd_jac,
            us_bwd_cost,
            us_bwd_mat,
            eval_count,
        };
        (u_first, f_opt, iters, converged, timing)
    }
}

impl Default for NmpcSolver {
    fn default() -> Self {
        Self::new()
    }
}
