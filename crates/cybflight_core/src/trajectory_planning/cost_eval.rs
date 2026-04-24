//! Cost function evaluator for trajectory optimization.

#[allow(unused_imports)]
use num_traits::Float;

use nalgebra::Vector3;

use super::flatness::{self, AlphaState, FlatnessState};
use super::minco_jerk::MincoJerk;
use super::penalties::{
    back_propagate_t, eval_dynamics_derivatives, forward_t, smoothed_l1_inv, DynDerivatives,
};
use super::quad_planning_config::QuadPlanningConfig;
use super::types::{Vec3, ZERO3, PVA3D};
use super::MAX_PIECES;
use crate::params::PlannerParams;

const JERK_COEFFS: usize = 6;

struct ConstraintBounds {
    max_vel_sq: f32,
    thr_mean: f32,
    thr_radi_sq: f32,
    cos_max_tilt: f32,
    /// Squared maximum pitch/roll rate magnitude.
    max_rate_xy_sq: f32,
    /// Squared maximum yaw rate magnitude.
    max_rate_z_sq: f32,
    gravity: f32,
    mass: f32,
    mu: f32,
    inv_mu: f32,
}

/// Accumulated derivative gradients at a single sample point.
pub struct PointGradients {
    pub vel: Vector3<f32>,
    pub acc: Vector3<f32>,
    pub jer: Vector3<f32>,
}

impl PointGradients {
    fn zero() -> Self {
        Self {
            vel: Vector3::zeros(),
            acc: Vector3::zeros(),
            jer: Vector3::zeros(),
        }
    }
}

/// Cost function evaluator. Owns the MINCO solver and all working buffers.
pub struct CostEvaluator {
    params: PlannerParams,
    bounds: ConstraintBounds,
    n_pieces: usize,
    n_waypoints: usize,
    dim_k: usize,

    nominal_waypoints: [Vec3; MAX_PIECES],
    waypoint_radius: f32,

    minco: MincoJerk,

    times: [f32; MAX_PIECES],
    waypoints: [Vec3; MAX_PIECES],
    partial_grad_c: [Vector3<f32>; JERK_COEFFS * MAX_PIECES],
    partial_grad_t: [f32; MAX_PIECES],
    grad_points: [Vector3<f32>; MAX_PIECES],
    grad_times: [f32; MAX_PIECES],
}

impl CostEvaluator {
    pub fn new(
        config: &QuadPlanningConfig,
        n_pieces: usize,
        head: &PVA3D,
        tail: &PVA3D,
        nominal_waypoints: &[Vec3; MAX_PIECES],
        waypoint_radius: f32,
    ) -> Self {
        let p = &config.planner;
        let max_vel = p.max_vel_m_s;
        let thr_mean = 0.5 * (config.max_collective_thrust_n + config.min_collective_thrust_n);
        let thr_radi = 0.5 * (config.max_collective_thrust_n - config.min_collective_thrust_n);
        let cos_max_tilt = p.max_tilt_rad.cos();
        let mr = config.max_rate_rad_s;

        let mu = p.smoothing_eps;
        let bounds = ConstraintBounds {
            max_vel_sq: max_vel * max_vel,
            thr_mean,
            thr_radi_sq: thr_radi * thr_radi,
            cos_max_tilt,
            max_rate_xy_sq: mr[0] * mr[0],
            max_rate_z_sq: mr[2] * mr[2],
            gravity: config.grav,
            mass: config.mass,
            mu,
            inv_mu: 1.0 / mu,
        };

        let n_wp = n_pieces - 1;
        Self {
            params: p.clone(),
            bounds,
            n_pieces,
            n_waypoints: n_wp,
            dim_k: n_pieces,
            nominal_waypoints: *nominal_waypoints,
            waypoint_radius,
            minco: MincoJerk::new(head, tail, n_pieces),
            times: [0.0; MAX_PIECES],
            waypoints: [ZERO3; MAX_PIECES],
            partial_grad_c: [Vector3::zeros(); JERK_COEFFS * MAX_PIECES],
            partial_grad_t: [0.0; MAX_PIECES],
            grad_points: [Vector3::zeros(); MAX_PIECES],
            grad_times: [0.0; MAX_PIECES],
        }
    }

    /// Evaluate cost and gradient at decision vector `x`.
    pub fn evaluate(&mut self, x: &[f32], grad: &mut [f32]) -> f32 {
        self.decode_decision_vars(x);
        self.solve_minco();
        self.zero_grad_accumulators();

        let mut cost = 0.0;
        cost += self.accumulate_energy_cost();
        cost += self.accumulate_dynamics_penalties();
        self.propagate_through_minco();
        cost += self.accumulate_time_cost();
        self.encode_gradient(x, grad);

        cost
    }

    fn decode_decision_vars(&mut self, x: &[f32]) {
        for i in 0..self.n_pieces {
            self.times[i] = forward_t(x[i]);
        }
        let r = self.waypoint_radius;
        for i in 0..self.n_waypoints {
            let dx = x[self.dim_k + 3 * i];
            let dy = x[self.dim_k + 3 * i + 1];
            let dz = x[self.dim_k + 3 * i + 2];
            let d = Vector3::new(dx, dy, dz);
            let s = 2.0 * r / (d.norm_squared() + 1.0);
            self.waypoints[i] = self.nominal_waypoints[i] + d * s;
        }
    }

    fn solve_minco(&mut self) {
        self.minco.solve(
            &self.waypoints[..self.n_waypoints],
            &self.times[..self.n_pieces],
        );
    }

    fn zero_grad_accumulators(&mut self) {
        let sys = JERK_COEFFS * self.n_pieces;
        self.partial_grad_c[..sys].fill(Vector3::zeros());
        self.partial_grad_t[..self.n_pieces].fill(0.0);
    }

    fn accumulate_energy_cost(&mut self) -> f32 {
        let w = self.params.weight_energy;
        if w < 1e-6 {
            return 0.0;
        }

        let sys = JERK_COEFFS * self.n_pieces;
        self.minco
            .add_energy_grad_by_coeffs(&mut self.partial_grad_c[..sys], w);
        self.minco
            .add_energy_grad_by_times(&mut self.partial_grad_t[..self.n_pieces], w);

        w * self.minco.get_energy()
    }

    fn accumulate_dynamics_penalties(&mut self) -> f32 {
        let n_check = self.params.num_check_per_piece;
        let inv_n = 1.0 / n_check as f32;

        let need_vel = self.params.weight_vel > 1e-6;
        let need_thrust = self.params.weight_thrust > 1e-6;
        let need_tilt = self.params.weight_tilt > 1e-6;
        let need_body_rate = self.params.weight_body_rate > 1e-6;
        let need_alpha = need_thrust || need_tilt || need_body_rate;

        if !(need_vel || need_alpha) {
            return 0.0;
        }

        let mut total_cost = 0.0;

        for seg in 0..self.n_pieces {
            let seg_dur = self.times[seg];
            let step = seg_dur / n_check as f32;

            for j in 0..=n_check {
                let t_local = j as f32 * step;
                let node = if j == 0 || j == n_check { 0.5 } else { 1.0 };
                let t_frac = j as f32 * inv_n;
                let base = seg * JERK_COEFFS;

                let dd = eval_dynamics_derivatives(self.minco.piece_coeffs(seg), t_local);
                let mut grads = PointGradients::zero();
                let mut penalty = 0.0;

                if need_vel {
                    penalty += self.velocity_penalty(&dd, &mut grads);
                }

                if need_alpha {
                    let alpha = flatness::compute_alpha_state(dd.acc, self.bounds.gravity);

                    if need_thrust {
                        penalty += self.thrust_penalty(&alpha, &mut grads);
                    }
                    if need_tilt {
                        penalty += self.tilt_penalty(&alpha, &mut grads);
                    }
                    if need_body_rate && alpha.zb[2] > -0.9 {
                        let fs = flatness::extend_to_flatness(&alpha, dd.jer);
                        penalty += self.body_rate_penalty(&fs, &mut grads);
                    }
                }

                if penalty > 0.0 {
                    self.assemble_basis_gradients(
                        base, &dd, &grads, penalty, step, node, t_frac, inv_n,
                    );
                    total_cost += node * step * penalty;
                }
            }
        }

        total_cost
    }

    fn propagate_through_minco(&mut self) {
        let sys = JERK_COEFFS * self.n_pieces;
        self.minco.propagate_grad(
            &self.partial_grad_c[..sys],
            &self.partial_grad_t[..self.n_pieces],
            &mut self.grad_points[..self.n_waypoints],
            &mut self.grad_times[..self.n_pieces],
        );
    }

    fn accumulate_time_cost(&mut self) -> f32 {
        let w = self.params.weight_time;
        let mut cost = 0.0;
        for i in 0..self.n_pieces {
            cost += w * self.times[i];
            self.grad_times[i] += w;
        }
        cost
    }

    fn encode_gradient(&self, x: &[f32], grad: &mut [f32]) {
        for i in 0..self.n_pieces {
            grad[i] = back_propagate_t(x[i], self.grad_times[i]);
        }
        let r = self.waypoint_radius;
        for i in 0..self.n_waypoints {
            let dx = x[self.dim_k + 3 * i];
            let dy = x[self.dim_k + 3 * i + 1];
            let dz = x[self.dim_k + 3 * i + 2];
            let d = Vector3::new(dx, dy, dz);
            let s = 2.0 * r / (d.norm_squared() + 1.0);
            let g = self.grad_points[i];
            let dot_dg = d.dot(&g);
            let coeff = s * s / r;
            let out = g * s - d * (coeff * dot_dg);
            grad[self.dim_k + 3 * i] = out.x;
            grad[self.dim_k + 3 * i + 1] = out.y;
            grad[self.dim_k + 3 * i + 2] = out.z;
        }
    }

    #[inline]
    fn velocity_penalty(&self, dd: &DynDerivatives, grads: &mut PointGradients) -> f32 {
        let violation = dd.vel.norm_squared() - self.bounds.max_vel_sq;
        let (f, df) = smoothed_l1_inv(violation, self.bounds.mu, self.bounds.inv_mu);
        if f > 0.0 {
            let w = self.params.weight_vel;
            let scale = w * df * 2.0;
            grads.vel += dd.vel * scale;
            w * f
        } else {
            0.0
        }
    }

    #[inline]
    fn thrust_penalty(&self, alpha: &AlphaState, grads: &mut PointGradients) -> f32 {
        let collective = self.bounds.mass * alpha.norm_alpha;
        let delta = collective - self.bounds.thr_mean;
        let (f, df) = smoothed_l1_inv(
            delta * delta - self.bounds.thr_radi_sq,
            self.bounds.mu,
            self.bounds.inv_mu,
        );
        if f > 0.0 {
            let w = self.params.weight_thrust;
            let d_violation = 2.0 * delta * self.bounds.mass * alpha.inv_norm_alpha;
            let scale = w * df * d_violation;
            grads.acc += alpha.alpha * scale;
            w * f
        } else {
            0.0
        }
    }

    #[inline]
    fn tilt_penalty(&self, alpha: &AlphaState, grads: &mut PointGradients) -> f32 {
        let violation = self.bounds.cos_max_tilt - alpha.zb[2];
        let (f, df) = smoothed_l1_inv(violation, self.bounds.mu, self.bounds.inv_mu);
        if f > 0.0 {
            let w = self.params.weight_tilt;
            let inv = alpha.inv_norm_alpha;
            let inv3 = inv * inv * inv;
            let common = w * df;
            let a = alpha.alpha;
            let axy_sq = a.x * a.x + a.y * a.y;
            grads.acc.x += common * a.x * a.z * inv3;
            grads.acc.y += common * a.y * a.z * inv3;
            grads.acc.z -= common * axy_sq * inv3;
            w * f
        } else {
            0.0
        }
    }

    #[inline]
    fn body_rate_penalty(&self, fs: &FlatnessState, grads: &mut PointGradients) -> f32 {
        let omega_xy_sq = fs.omega.x * fs.omega.x + fs.omega.y * fs.omega.y;
        let (f_xy, df_xy) = smoothed_l1_inv(
            omega_xy_sq - self.bounds.max_rate_xy_sq,
            self.bounds.mu,
            self.bounds.inv_mu,
        );
        let (f_z, df_z) = smoothed_l1_inv(
            fs.omega.z * fs.omega.z - self.bounds.max_rate_z_sq,
            self.bounds.mu,
            self.bounds.inv_mu,
        );

        if f_xy > 0.0 || f_z > 0.0 {
            let w = self.params.weight_body_rate;
            let mut g_omega = Vector3::<f32>::zeros();
            if f_xy > 0.0 {
                g_omega.x = w * df_xy * 2.0 * fs.omega.x;
                g_omega.y = w * df_xy * 2.0 * fs.omega.y;
            }
            if f_z > 0.0 {
                g_omega.z = w * df_z * 2.0 * fs.omega.z;
            }
            flatness::body_rate_grad_backprop(&g_omega, fs, &mut grads.acc, &mut grads.jer);
            w * (f_xy + f_z)
        } else {
            0.0
        }
    }

    fn assemble_basis_gradients(
        &mut self,
        base: usize,
        dd: &DynDerivatives,
        grads: &PointGradients,
        penalty: f32,
        step: f32,
        node: f32,
        t_frac: f32,
        inv_n: f32,
    ) {
        let s1 = dd.s1;
        let s2 = dd.s2;
        let s3 = dd.s3;
        let s4 = dd.s4;
        let scale = step * node;
        let bv1 = 1.0;
        let bv2 = 2.0 * s1;
        let bv3 = 3.0 * s2;
        let bv4 = 4.0 * s3;
        let bv5 = 5.0 * s4;
        let ba2 = 2.0;
        let ba3 = 6.0 * s1;
        let ba4 = 12.0 * s2;
        let ba5 = 20.0 * s3;
        let bj3 = 6.0;
        let bj4 = 24.0 * s1;
        let bj5 = 60.0 * s2;

        let gv = grads.vel;
        let ga = grads.acc;
        let gj = grads.jer;

        self.partial_grad_c[base + 1] += gv * (bv1 * scale);
        self.partial_grad_c[base + 2] += (gv * bv2 + ga * ba2) * scale;
        self.partial_grad_c[base + 3] += (gv * bv3 + ga * ba3 + gj * bj3) * scale;
        self.partial_grad_c[base + 4] += (gv * bv4 + ga * ba4 + gj * bj4) * scale;
        self.partial_grad_c[base + 5] += (gv * bv5 + ga * ba5 + gj * bj5) * scale;

        let time_chain = gv.dot(&dd.acc) + ga.dot(&dd.jer) + gj.dot(&dd.sna);
        let seg = base / JERK_COEFFS;
        self.partial_grad_t[seg] += time_chain * scale * t_frac + node * inv_n * penalty;
    }
}
