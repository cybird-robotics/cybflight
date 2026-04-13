//! Cost function evaluator for trajectory optimization.
//!
//! Decomposes the monolithic cost+gradient closure into a struct with
//! named pipeline stages and individual penalty methods.

#[allow(unused_imports)]
use num_traits::Float;

use super::flatness::{self, FlatnessState};
use super::minco_jerk::MincoJerk;
use super::penalties::{
    self, eval_dynamics_derivatives, forward_t_log, jerk_basis_vectors, smoothed_l1,
    DynDerivatives,
};
use super::piecewise_polynomial::PiecewisePolynomial;
use super::quad_planning_config::QuadPlanningConfig;
use super::types::*;
use super::MAX_PIECES;
use crate::params::PlannerParams;

const JERK_COEFFS: usize = 6;

/// Precomputed constraint bounds derived from config.
struct ConstraintBounds {
    max_vel_sq: f32,
    thr_mean: f32,
    thr_radi_sq: f32,
    cos_max_tilt: f32,
    max_rate_sq: [f32; 3],
    gravity: f32,
    mass: f32,
}

/// Accumulated derivative gradients at a single sample point.
///
/// Stack-local per point; folded into coefficient gradients after
/// all penalties are evaluated. Avoids `&mut self` borrow conflicts.
pub struct PointGradients {
    pub vel: Vec3,
    pub acc: Vec3,
    pub jer: Vec3,
}

impl PointGradients {
    fn zero() -> Self {
        Self {
            vel: ZERO3,
            acc: ZERO3,
            jer: ZERO3,
        }
    }
}

/// Cost function evaluator. Owns the MINCO solver and all working buffers.
///
/// Construct once per optimization run, then call `evaluate()` on each
/// iteration from the BFGS optimizer.
pub struct CostEvaluator {
    // Config (immutable after construction)
    params: PlannerParams,
    bounds: ConstraintBounds,
    n_pieces: usize,
    n_waypoints: usize,
    dim_k: usize,

    // MINCO solver (mutated on each evaluate call)
    minco: MincoJerk,

    // Working buffers (reused across evaluate calls, cleared each time)
    times: [f32; MAX_PIECES],
    waypoints: [Vec3; MAX_PIECES],
    partial_grad_c: [[f32; 3]; JERK_COEFFS * MAX_PIECES],
    partial_grad_t: [f32; MAX_PIECES],
    grad_points: [Vec3; MAX_PIECES],
    grad_times: [f32; MAX_PIECES],
    eg_c: [[f32; 3]; JERK_COEFFS * MAX_PIECES],
    eg_t: [f32; MAX_PIECES],
}

impl CostEvaluator {
    /// Construct evaluator from config and boundary conditions.
    pub fn new(
        config: &QuadPlanningConfig,
        n_pieces: usize,
        head: &PVA3D,
        tail: &PVA3D,
    ) -> Self {
        let p = &config.planner;
        let max_vel = p.max_vel_m_s;
        let thr_mean = 0.5 * (config.max_collective_thrust_n + config.min_collective_thrust_n);
        let thr_radi = 0.5 * (config.max_collective_thrust_n - config.min_collective_thrust_n);
        let cos_max_tilt = p.max_tilt_rad.cos();
        let mr = config.max_rate_rad_s;

        let bounds = ConstraintBounds {
            max_vel_sq: max_vel * max_vel,
            thr_mean,
            thr_radi_sq: thr_radi * thr_radi,
            cos_max_tilt,
            max_rate_sq: [mr[0] * mr[0], mr[1] * mr[1], mr[2] * mr[2]],
            gravity: config.grav,
            mass: config.mass,
        };

        let n_wp = n_pieces - 1;
        Self {
            params: p.clone(),
            bounds,
            n_pieces,
            n_waypoints: n_wp,
            dim_k: n_pieces,
            minco: MincoJerk::new(head, tail, n_pieces),
            times: [0.0; MAX_PIECES],
            waypoints: [ZERO3; MAX_PIECES],
            partial_grad_c: [[0.0; 3]; JERK_COEFFS * MAX_PIECES],
            partial_grad_t: [0.0; MAX_PIECES],
            grad_points: [ZERO3; MAX_PIECES],
            grad_times: [0.0; MAX_PIECES],
            eg_c: [[0.0; 3]; JERK_COEFFS * MAX_PIECES],
            eg_t: [0.0; MAX_PIECES],
        }
    }

    /// Evaluate cost and gradient at decision vector `x`.
    ///
    /// `x = [K_0..K_{n-1}, d_0x, d_0y, d_0z, ..., d_{nwp-1}z]`
    ///
    /// Returns cost value. Fills `grad[..dim_k + dim_d]`.
    pub fn evaluate(&mut self, x: &[f32], grad: &mut [f32]) -> f32 {
        self.decode_decision_vars(x);
        self.solve_minco();
        let traj = self.minco.get_trajectory();

        self.zero_grad_accumulators();

        let mut cost = 0.0;
        cost += self.accumulate_energy_cost();
        cost += self.accumulate_dynamics_penalties(&traj);
        self.propagate_through_minco();
        cost += self.accumulate_time_cost();
        self.encode_gradient(x, grad);

        cost
    }

    // -----------------------------------------------------------------------
    // Private pipeline stages
    // -----------------------------------------------------------------------

    /// Decode decision vector into times[] and waypoints[].
    fn decode_decision_vars(&mut self, x: &[f32]) {
        for i in 0..self.n_pieces {
            self.times[i] = forward_t_log(x[i]);
        }
        for i in 0..self.n_waypoints {
            self.waypoints[i] = [
                x[self.dim_k + 3 * i],
                x[self.dim_k + 3 * i + 1],
                x[self.dim_k + 3 * i + 2],
            ];
        }
    }

    /// Solve MINCO with current times and waypoints.
    fn solve_minco(&mut self) {
        self.minco.solve(
            &self.waypoints[..self.n_waypoints],
            &self.times[..self.n_pieces],
        );
    }

    /// Zero all gradient accumulator buffers.
    fn zero_grad_accumulators(&mut self) {
        let sys = JERK_COEFFS * self.n_pieces;
        self.partial_grad_c[..sys].fill([0.0; 3]);
        self.partial_grad_t[..self.n_pieces].fill(0.0);
    }

    /// Energy cost: weight_energy * minco.get_energy().
    fn accumulate_energy_cost(&mut self) -> f32 {
        let w = self.params.weight_energy;
        if w < 1e-6 {
            return 0.0;
        }

        let energy = self.minco.get_energy();
        let sys = JERK_COEFFS * self.n_pieces;

        self.minco
            .get_energy_partial_grad_by_coeffs(&mut self.eg_c[..sys]);
        self.minco
            .get_energy_partial_grad_by_times(&mut self.eg_t[..self.n_pieces]);

        for j in 0..sys {
            for d in 0..3 {
                self.partial_grad_c[j][d] += w * self.eg_c[j][d];
            }
        }
        for j in 0..self.n_pieces {
            self.partial_grad_t[j] += w * self.eg_t[j];
        }

        w * energy
    }

    /// Sample dynamics penalties across all segments and sample points.
    fn accumulate_dynamics_penalties(&mut self, traj: &PiecewisePolynomial) -> f32 {
        let n_check = self.params.num_check_per_piece;
        let need_flatness =
            self.params.weight_tilt > 1e-6 || self.params.weight_body_rate > 1e-6;

        let mut total_cost = 0.0;

        for seg in 0..self.n_pieces {
            let seg_dur = self.times[seg];
            let step = seg_dur / n_check as f32;
            let inv_n = 1.0 / n_check as f32;

            for j in 0..=n_check {
                let t_local = j as f32 * step;
                let node = if j == 0 || j == n_check { 0.5 } else { 1.0 };
                let t_frac = j as f32 * inv_n;
                let base = seg * JERK_COEFFS;

                let dd = eval_dynamics_derivatives(traj, seg, t_local);
                let mut grads = PointGradients::zero();

                // Compute flatness state (shared by tilt + body rate penalties)
                let fs = if need_flatness {
                    flatness::compute_flatness_state(dd.acc, dd.jer, self.bounds.gravity)
                } else {
                    None
                };

                let mut penalty = 0.0;

                // -- Individual penalties --
                penalty += self.velocity_penalty(&dd, &mut grads);

                if let Some(ref fs) = fs {
                    penalty += self.thrust_penalty(fs, &mut grads);
                    penalty += self.tilt_penalty(fs, &mut grads);
                    penalty += self.body_rate_penalty(fs, &mut grads);
                } else {
                    // Without flatness, still compute collective thrust penalty
                    penalty += self.thrust_penalty_collective(&dd, &mut grads);
                }

                // -- Assemble basis gradients --
                self.assemble_basis_gradients(
                    base, &dd, &grads, penalty, step, node, t_frac,
                );

                total_cost += node * step * penalty;
            }
        }

        total_cost
    }

    /// Propagate partial_grad_c/t through MINCO to get grad_points/times.
    fn propagate_through_minco(&mut self) {
        let sys = JERK_COEFFS * self.n_pieces;
        self.minco.propagate_grad(
            &self.partial_grad_c[..sys],
            &self.partial_grad_t[..self.n_pieces],
            &mut self.grad_points[..self.n_waypoints],
            &mut self.grad_times[..self.n_pieces],
        );
    }

    /// Time cost: weight_time * sum(T_i).
    fn accumulate_time_cost(&mut self) -> f32 {
        let w = self.params.weight_time;
        let mut cost = 0.0;
        for i in 0..self.n_pieces {
            cost += w * self.times[i];
            self.grad_times[i] += w;
        }
        cost
    }

    /// Transform grad_times/points back to decision variable space.
    fn encode_gradient(&self, x: &[f32], grad: &mut [f32]) {
        // Time gradients: ∂L/∂K = T · ∂L/∂T (log parameterization: ∂T/∂K = exp(K) = T)
        for i in 0..self.n_pieces {
            grad[i] = self.grad_times[i] * self.times[i];
        }
        // Waypoint gradients: identity
        for i in 0..self.n_waypoints {
            grad[self.dim_k + 3 * i] = self.grad_points[i][0];
            grad[self.dim_k + 3 * i + 1] = self.grad_points[i][1];
            grad[self.dim_k + 3 * i + 2] = self.grad_points[i][2];
        }
    }

    // -----------------------------------------------------------------------
    // Individual penalties (take &self, write to stack-local PointGradients)
    // -----------------------------------------------------------------------

    /// Velocity penalty: smoothed_l1(‖v‖² - max_vel²).
    #[inline]
    fn velocity_penalty(&self, dd: &DynDerivatives, grads: &mut PointGradients) -> f32 {
        let w = self.params.weight_vel;
        if w < 1e-6 {
            return 0.0;
        }
        let v_sq = norm_sq3(dd.vel);
        let violation = v_sq - self.bounds.max_vel_sq;
        let (f, df) = smoothed_l1(violation, self.params.smoothing_eps);
        if f > 0.0 {
            let scale = w * df * 2.0;
            for d in 0..3 {
                grads.vel[d] += scale * dd.vel[d];
            }
            w * f
        } else {
            0.0
        }
    }

    /// Collective thrust penalty (used when flatness state is available).
    /// F = mass * ‖α‖, penalty on (F - F_mean)² - F_radius².
    #[inline]
    fn thrust_penalty(&self, fs: &FlatnessState, grads: &mut PointGradients) -> f32 {
        let w = self.params.weight_thrust;
        if w < 1e-6 {
            return 0.0;
        }
        let collective = self.bounds.mass * fs.norm_alpha;
        let thr_violation =
            (collective - self.bounds.thr_mean) * (collective - self.bounds.thr_mean)
                - self.bounds.thr_radi_sq;
        let (f, df) = smoothed_l1(thr_violation, self.params.smoothing_eps);
        if f > 0.0 {
            let d_violation =
                2.0 * (collective - self.bounds.thr_mean) * self.bounds.mass * fs.inv_norm_alpha;
            let scale = w * df * d_violation;
            for d in 0..3 {
                grads.acc[d] += scale * fs.alpha[d];
            }
            w * f
        } else {
            0.0
        }
    }

    /// Collective thrust penalty without full flatness state (fallback).
    #[inline]
    fn thrust_penalty_collective(
        &self,
        dd: &DynDerivatives,
        grads: &mut PointGradients,
    ) -> f32 {
        let w = self.params.weight_thrust;
        if w < 1e-6 {
            return 0.0;
        }
        let alpha = [dd.acc[0], dd.acc[1], dd.acc[2] + self.bounds.gravity];
        let norm_alpha = norm_sq3(alpha).sqrt().max(1e-8);
        let inv_norm = 1.0 / norm_alpha;
        let collective = self.bounds.mass * norm_alpha;
        let thr_violation =
            (collective - self.bounds.thr_mean) * (collective - self.bounds.thr_mean)
                - self.bounds.thr_radi_sq;
        let (f, df) = smoothed_l1(thr_violation, self.params.smoothing_eps);
        if f > 0.0 {
            let d_violation =
                2.0 * (collective - self.bounds.thr_mean) * self.bounds.mass * inv_norm;
            let scale = w * df * d_violation;
            for d in 0..3 {
                grads.acc[d] += scale * alpha[d];
            }
            w * f
        } else {
            0.0
        }
    }

    /// Tilt angle penalty: smoothed_l1(cos_max_tilt - zb_z).
    ///
    /// Works in cosine domain to avoid 1/sin singularity.
    #[inline]
    fn tilt_penalty(&self, fs: &FlatnessState, grads: &mut PointGradients) -> f32 {
        let w = self.params.weight_tilt;
        if w < 1e-6 {
            return 0.0;
        }
        let cos_tilt = fs.zb[2];
        let violation = self.bounds.cos_max_tilt - cos_tilt;
        let (f, df) = smoothed_l1(violation, self.params.smoothing_eps);
        if f > 0.0 {
            let inv3 = fs.inv_norm_alpha * fs.inv_norm_alpha * fs.inv_norm_alpha;
            let common = w * df;
            let axy_sq = fs.alpha[0] * fs.alpha[0] + fs.alpha[1] * fs.alpha[1];
            grads.acc[0] += common * fs.alpha[0] * fs.alpha[2] * inv3;
            grads.acc[1] += common * fs.alpha[1] * fs.alpha[2] * inv3;
            grads.acc[2] -= common * axy_sq * inv3;
            w * f
        } else {
            0.0
        }
    }

    /// Body rate penalty: smoothed_l1(ω_xy² - max²) + smoothed_l1(ω_z² - max²).
    #[inline]
    fn body_rate_penalty(&self, fs: &FlatnessState, grads: &mut PointGradients) -> f32 {
        let w = self.params.weight_body_rate;
        if w < 1e-6 {
            return 0.0;
        }
        let omega_xy_sq = fs.omega[0] * fs.omega[0] + fs.omega[1] * fs.omega[1];
        let (f_xy, df_xy) =
            smoothed_l1(omega_xy_sq - self.bounds.max_rate_sq[0], self.params.smoothing_eps);
        let (f_z, df_z) = smoothed_l1(
            fs.omega[2] * fs.omega[2] - self.bounds.max_rate_sq[2],
            self.params.smoothing_eps,
        );

        if f_xy > 0.0 || f_z > 0.0 {
            let mut g_omega = ZERO3;
            if f_xy > 0.0 {
                g_omega[0] = w * df_xy * 2.0 * fs.omega[0];
                g_omega[1] = w * df_xy * 2.0 * fs.omega[1];
            }
            if f_z > 0.0 {
                g_omega[2] = w * df_z * 2.0 * fs.omega[2];
            }
            flatness::body_rate_grad_backprop(&g_omega, fs, &mut grads.acc, &mut grads.jer);
            w * (f_xy + f_z)
        } else {
            0.0
        }
    }

    // -----------------------------------------------------------------------
    // Gradient assembly
    // -----------------------------------------------------------------------

    /// Assemble monomial basis gradients and accumulate into partial_grad_c/t.
    fn assemble_basis_gradients(
        &mut self,
        base: usize,
        dd: &DynDerivatives,
        grads: &PointGradients,
        penalty: f32,
        step: f32,
        node: f32,
        t_frac: f32,
    ) {
        let (beta_vel, beta_acc, beta_jer) = jerk_basis_vectors(dd);
        let scale = step * node;

        for k in 0..JERK_COEFFS {
            for d in 0..3 {
                self.partial_grad_c[base + k][d] += (beta_vel[k] * grads.vel[d]
                    + beta_acc[k] * grads.acc[d]
                    + beta_jer[k] * grads.jer[d])
                    * scale;
            }
        }

        // Time gradient: ∂penalty/∂t via chain rule through derivatives
        let time_chain =
            dot3(grads.vel, dd.acc) + dot3(grads.acc, dd.jer) + dot3(grads.jer, dd.sna);
        let seg = base / JERK_COEFFS;
        let inv_n = 1.0 / self.params.num_check_per_piece as f32;
        self.partial_grad_t[seg] += time_chain * scale * t_frac + node * inv_n * penalty;
    }
}
