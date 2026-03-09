// NMPC solver: supports both multiple-shooting SQP (Gauss-Newton) and
// single-shooting L-BFGS via the USE_SQP flag.

use crate::nmpc::lbfgs::ValueAndGrad;

use super::lbfgs::{self, LbfgsParams, LbfgsWorkspace, DIM};
use super::model::{Control, CtrlJac, QuadModel, State, StateJac, NU};
use super::qp::QpWorkspace;
use core::convert::From;
use core::option::{
    Option,
    Option::{None, Some},
};
use nalgebra as na;

#[derive(Clone)]
pub struct NmpcState {
    pub position: na::Vector3<f32>,
    pub orientation: na::UnitQuaternion<f32>,
    pub velocity: na::Vector3<f32>,
}

impl From<State> for NmpcState {
    fn from(state: State) -> Self {
        Self {
            position: state.fixed_rows::<3>(0).into(),
            orientation: na::UnitQuaternion::from_quaternion(na::Quaternion::from_vector(
                state.fixed_rows::<4>(3).into(),
            )),
            velocity: state.fixed_rows::<3>(7).into(),
        }
    }
}

#[derive(Clone)]
pub struct NmpcCommand {
    pub thrust: f32,
    pub omega: na::Vector3<f32>,
}

impl From<Control> for NmpcCommand {
    fn from(control: Control) -> Self {
        Self {
            thrust: control[0],
            omega: control.fixed_rows::<3>(1).into(),
        }
    }
}

impl From<NmpcState> for State {
    fn from(s: NmpcState) -> Self {
        let mut out = State::zeros();
        out.fixed_rows_mut::<3>(0).copy_from(&s.position);
        out.fixed_rows_mut::<4>(3).copy_from(&s.orientation.coords);
        out.fixed_rows_mut::<3>(7).copy_from(&s.velocity);
        out
    }
}

impl From<&NmpcState> for State {
    fn from(s: &NmpcState) -> Self {
        State::from(s.clone())
    }
}

impl From<NmpcCommand> for Control {
    fn from(c: NmpcCommand) -> Self {
        Control::from([c.thrust, c.omega[0], c.omega[1], c.omega[2]])
    }
}

impl From<&NmpcCommand> for Control {
    fn from(c: &NmpcCommand) -> Self {
        Control::from(c.clone())
    }
}

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
    pub ws: LbfgsWorkspace<f32, DIM>,
    warmstart: bool,
    prev_u: na::SVector<f32, DIM>,
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
        let mut prev_u = na::SVector::<f32, DIM>::zeros();
        for i in 0..N {
            prev_u[i * NU] = mass * grav;
        }
        Self {
            model,
            qp,
            ws: LbfgsWorkspace::default(),
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

    pub fn solve(
        &mut self,
        x_init: &NmpcState,
        x_refs: &[NmpcState; N + 1],
        u_refs: &[NmpcCommand; N],
    ) -> (NmpcCommand, f32, i32, bool, NmpcTiming) {
        // 1. Convert the initial state and bind it to a local variable
        let x0: State = x_init.into();

        // 2. Use core::array::from_fn to map over the reference arrays cleanly
        let x_refs_arr: [State; N + 1] = core::array::from_fn(|i| (&x_refs[i]).into());
        let u_refs_arr: [Control; N] = core::array::from_fn(|i| (&u_refs[i]).into());

        // 3. Pass references to the newly created local arrays/variables
        let (u, v, i, b, t) = self.solve_with_array(&x0, &x_refs_arr, &u_refs_arr);

        // 4. Convert the output back to the wrapper
        let u_cmd = NmpcCommand::from(u);

        (u_cmd, v, i, b, t)
    }

    // Solve the NMPC problem.
    // Returns (u_opt_first, final_cost, iterations, converged, timing).
    pub fn solve_with_array(
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
            cost += self.model.state_cost_grad(&self.qp.x_bar[k], &x_refs[k], &mut gx);
            cost += self.model.input_cost_grad(&self.qp.u_bar[k], &u_refs[k], &mut gu);
            let mut gx_con = State::zeros();
            let mut gu_con = Control::zeros();
            cost += self.model.constraint_grad(&self.qp.u_bar[k], &mut gx_con, &mut gu_con);
        }
        cost += self.model.state_cost_grad(&self.qp.x_bar[N], &x_refs[N], &mut gx);
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
                self.qp.x_bar[k + 1] =
                    self.model.propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
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
                self.qp.x_bar[k + 1] =
                    self.model.propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
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
                    self.qp.x_bar[k + 1] =
                        self.model.propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
                }
                cost_after = self.eval_cost(x_refs, u_refs);
            }

            if cost_after > cost_before {
                self.qp.u_bar = u_bar_prev;
                self.qp.x_bar[0] = *x_init;
                for k in 0..N {
                    self.qp.x_bar[k + 1] =
                        self.model.propagate_rk4(&self.qp.x_bar[k], &self.qp.u_bar[k]);
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
        let mut u_flat = na::SVector::<f32, DIM>::zeros();
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

        let params = LbfgsParams::default();

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

        let mut eval = |u_flat: &na::SVector<f32, DIM>| -> lbfgs::ValueAndGrad<f32, DIM> {
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

            let mut grad_u_flat = na::SVector::<f32, DIM>::zeros();
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

            ValueAndGrad(total_cost, grad_u_flat)
        };

        let ret = lbfgs::lbfgs_optimize(&u_flat, &params, ws, &mut eval);

        let (u_first, f_opt, converged) = if let Ok(lbfgs::Solution { x, f, .. }) = ret {
            (x, f, true)
        } else {
            (u_flat, f32::INFINITY, false)
        };
        let u_first = Control::from_fn(|j, _| u_first[j]);

        self.prev_u = u_flat;
        self.warmstart = true;

        let iters = self.ws.last_k;
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
