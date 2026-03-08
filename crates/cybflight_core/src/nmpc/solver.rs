// NMPC solver: supports both multiple-shooting SQP (Gauss-Newton) and
// single-shooting L-BFGS via the USE_SQP flag.

use super::lbfgs::{self, DIM, LbfgsParams, LbfgsWorkspace};
use super::model::{Control, CtrlJac, NU, QuadModel, State, StateJac};
use super::qp::QpWorkspace;

/// Per-phase timing from one `solve()` call.
pub struct NmpcTiming {
    /// µs spent in forward simulation (propagate_rk4 × N).
    pub us_fwd: u64,
    /// µs spent computing linearizations (Euler sensitivities).
    pub us_bwd_jac: u64,
    /// µs spent computing cost Hessians and gradients.
    pub us_bwd_cost: u64,
    /// µs spent in QP backward/forward sweeps (SQP) or matrix multiplies (L-BFGS).
    pub us_bwd_mat: u64,
    /// Number of SQP iterations or eval() invocations.
    pub eval_count: u32,
}

pub const N: usize = 20; // prediction horizon

/// If true, use multiple-shooting SQP with Gauss-Newton Hessian.
/// If false, use single-shooting L-BFGS (original solver).
const USE_SQP: bool = true;

/// If true, use the fast single-stage Euler sensitivity in the L-BFGS backward pass.
const USE_EULER_GRAD: bool = true;

const MAX_SQP_ITERS: usize = 2;
const KKT_TOL: f32 = 1e-3;

const _: () = assert!(DIM == N * NU, "lbfgs::DIM must equal N * NU");

// ── Persistent solver state ───────────────────────────────────────────────────

pub struct NmpcSolver {
    pub model: QuadModel,
    pub qp: QpWorkspace,
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
        let model = QuadModel::new(mass, grav, 0.05);
        let mut qp = QpWorkspace::new();
        for k in 0..N {
            qp.u_bar[k] = Control::from([mass * grav, 0.0, 0.0, 0.0]);
        }
        let mut prev_u = [0.0_f32; DIM];
        for i in 0..N {
            prev_u[i * NU] = mass * grav;
        }
        Self {
            model,
            qp,
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
    // Returns (u_opt_first, final_cost, iterations, converged, timing).
    pub fn solve(
        &mut self,
        x_init: &State,
        x_refs: &[State; N + 1],
        u_refs: &[Control; N],
    ) -> (Control, f32, i32, bool, NmpcTiming) {
        if USE_SQP {
            self.solve_sqp(x_init, x_refs, u_refs)
        } else {
            self.solve_lbfgs(x_init, x_refs, u_refs)
        }
    }

    // ── SQP solver ──────────────────────────────────────────────────────────

    /// Compute total cost for current trajectory in qp workspace.
    fn eval_cost(&self, x_refs: &[State; N + 1], u_refs: &[Control; N]) -> f32 {
        let mut cost = 0.0_f32;
        let mut gx = State::zeros();
        let mut gu = Control::zeros();
        for k in 0..N {
            cost += self
                .model
                .state_cost_grad(&self.qp.x_bar[k], &x_refs[k], &mut gx);
            cost += self
                .model
                .input_cost_grad(&self.qp.u_bar[k], &u_refs[k], &mut gu);
            let mut gx_con = State::zeros();
            let mut gu_con = Control::zeros();
            cost += self
                .model
                .constraint_grad(&self.qp.u_bar[k], &mut gx_con, &mut gu_con);
        }
        cost += self
            .model
            .state_cost_grad(&self.qp.x_bar[N], &x_refs[N], &mut gx);
        cost
    }

    fn solve_sqp(
        &mut self,
        x_init: &State,
        x_refs: &[State; N + 1],
        u_refs: &[Control; N],
    ) -> (Control, f32, i32, bool, NmpcTiming) {
        if self.warmstart {
            let mut shifted = [Control::zeros(); N];
            for k in 0..(N - 1) {
                shifted[k] = self.qp.u_bar[k + 1];
            }
            shifted[N - 1] = self.qp.u_bar[N - 1];
            self.qp.u_bar = shifted;
            self.warmstart = false;
        }

        let mut us_fwd = 0u64;
        let mut us_bwd_jac = 0u64;
        let mut us_bwd_cost = 0u64;
        let mut us_bwd_mat = 0u64;
        let mut converged = false;
        let mut sqp_iter = 0i32;
        let mut final_cost = 0.0_f32;

        let mut grad_fx = StateJac::zeros();
        let mut grad_fu = CtrlJac::zeros();

        for iter in 0..MAX_SQP_ITERS {
            sqp_iter = (iter + 1) as i32;

            // Step 1: Forward simulate trajectory
            let t = self.timer_us.map_or(0, |f| f());
            self.qp.x_bar[0] = *x_init;
            for k in 0..N {
                self.qp.x_bar[k + 1] = self
                    .model
                    .propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
            }
            us_fwd += self.timer_us.map_or(0, |f| f()) - t;

            // Step 2: Linearize dynamics + compute cost Hessians/gradients
            let t = self.timer_us.map_or(0, |f| f());
            for k in 0..N {
                self.model.propagate_euler_grad(
                    &self.qp.x_bar[k],
                    &self.qp.u_bar[k],
                    &mut grad_fx,
                    &mut grad_fu,
                );
                self.qp.a[k] = grad_fx;
                self.qp.b[k] = grad_fu;
                self.qp.d[k] = State::zeros(); // consistent trajectory → zero defect
            }
            us_bwd_jac += self.timer_us.map_or(0, |f| f()) - t;

            let t = self.timer_us.map_or(0, |f| f());
            for k in 0..N {
                self.model.stage_cost_hess_grad(
                    &self.qp.x_bar[k],
                    &self.qp.u_bar[k],
                    &x_refs[k],
                    &u_refs[k],
                    &mut self.qp.qm[k],
                    &mut self.qp.rm[k],
                    &mut self.qp.q[k],
                    &mut self.qp.r[k],
                );
            }
            self.model.state_cost_hess_grad(
                &self.qp.x_bar[N],
                &x_refs[N],
                &mut self.qp.q[N],
                &mut self.qp.qm[N],
            );
            us_bwd_cost += self.timer_us.map_or(0, |f| f()) - t;

            // Step 3: Backward sweep (Riccati block elimination)
            let t = self.timer_us.map_or(0, |f| f());
            self.qp.backward_sweep();
            us_bwd_mat += self.timer_us.map_or(0, |f| f()) - t;

            // Step 4: Check convergence via feedforward norm
            let mut kkt_norm = 0.0_f32;
            for k in 0..N {
                for j in 0..4 {
                    let v = self.qp.gain_kk[k][j].abs();
                    if v > kkt_norm {
                        kkt_norm = v;
                    }
                }
            }

            // Step 5: Forward sweep with line search
            let t_fwd = self.timer_us.map_or(0, |f| f());
            let u_bar_prev = self.qp.u_bar;
            let cost_before = self.eval_cost(x_refs, u_refs);

            let mut alpha = 1.0_f32;
            self.qp.forward_sweep(x_init, alpha);

            self.qp.x_bar[0] = *x_init;
            for k in 0..N {
                self.qp.x_bar[k + 1] = self
                    .model
                    .propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
            }
            let mut cost_after = self.eval_cost(x_refs, u_refs);

            for _ in 0..4 {
                if cost_after <= cost_before {
                    break;
                }
                alpha *= 0.5;
                self.qp.u_bar = u_bar_prev;
                self.qp.forward_sweep(x_init, alpha);
                self.qp.x_bar[0] = *x_init;
                for k in 0..N {
                    self.qp.x_bar[k + 1] = self
                        .model
                        .propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
                }
                cost_after = self.eval_cost(x_refs, u_refs);
            }

            if cost_after > cost_before {
                self.qp.u_bar = u_bar_prev;
                self.qp.x_bar[0] = *x_init;
                for k in 0..N {
                    self.qp.x_bar[k + 1] = self
                        .model
                        .propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
                }
                final_cost = cost_before;
            } else {
                final_cost = cost_after;
            }
            us_fwd += self.timer_us.map_or(0, |f| f()) - t_fwd;

            if kkt_norm < KKT_TOL {
                converged = true;
                break;
            }
        }

        self.warmstart = true;

        let u_first = self.qp.u_bar[0];
        let timing = NmpcTiming {
            us_fwd,
            us_bwd_jac,
            us_bwd_cost,
            us_bwd_mat,
            eval_count: sqp_iter as u32,
        };
        (u_first, final_cost, sqp_iter, converged, timing)
    }

    // ── L-BFGS solver ───────────────────────────────────────────────────────

    fn solve_lbfgs(
        &mut self,
        x_init: &State,
        x_refs: &[State; N + 1],
        u_refs: &[Control; N],
    ) -> (Control, f32, i32, bool, NmpcTiming) {
        let mut u_flat = [0.0_f32; DIM];
        if self.warmstart {
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

        let model = &mut self.model;
        let ws = &mut self.ws;

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
