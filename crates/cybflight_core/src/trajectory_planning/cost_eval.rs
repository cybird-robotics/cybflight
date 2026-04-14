//! Cost function evaluator for trajectory optimization.
//!
//! Decomposes the monolithic cost+gradient closure into a struct with
//! named pipeline stages and individual penalty methods.

#[allow(unused_imports)]
use num_traits::Float;

use super::flatness::{self, AlphaState, FlatnessState};
use super::minco_jerk::MincoJerk;
use super::penalties::{
    back_propagate_t, eval_dynamics_derivatives, forward_t, jerk_basis_vectors,
    smoothed_l1, DynDerivatives,
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
    /// Squared maximum pitch/roll rate magnitude: ω_xy_max² (bound on ‖ω_xy‖²).
    max_rate_xy_sq: f32,
    /// Squared maximum yaw rate magnitude: ω_z_max².
    max_rate_z_sq: f32,
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

    /// Nominal (user-specified) waypoint positions — ball centers.
    nominal_waypoints: [Vec3; MAX_PIECES],
    /// Stereographic ball radius around each nominal waypoint.
    waypoint_radius: f32,

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
    ///
    /// `nominal_waypoints` holds the user-specified waypoint positions (only
    /// the first `n_pieces - 1` entries are consulted). `waypoint_radius`
    /// sets the stereographic ball radius around each nominal waypoint.
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

        let bounds = ConstraintBounds {
            max_vel_sq: max_vel * max_vel,
            thr_mean,
            thr_radi_sq: thr_radi * thr_radi,
            cos_max_tilt,
            // max_rate_rad_s[0] is the pitch/roll rate limit (a scalar bound
            // on the xy-plane body rate magnitude, matching the C++
            // `maxOmgXY`). max_rate_rad_s[2] is the yaw rate limit.
            max_rate_xy_sq: mr[0] * mr[0],
            max_rate_z_sq: mr[2] * mr[2],
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
            nominal_waypoints: *nominal_waypoints,
            waypoint_radius,
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
    ///
    /// Stereographic forward map: `P = P̂ + 2·r·D / (‖D‖² + 1)`.
    fn decode_decision_vars(&mut self, x: &[f32]) {
        for i in 0..self.n_pieces {
            self.times[i] = forward_t(x[i]);
        }
        let r = self.waypoint_radius;
        for i in 0..self.n_waypoints {
            let dx = x[self.dim_k + 3 * i];
            let dy = x[self.dim_k + 3 * i + 1];
            let dz = x[self.dim_k + 3 * i + 2];
            let norm_sq = dx * dx + dy * dy + dz * dz;
            let s = 2.0 * r / (norm_sq + 1.0);
            self.waypoints[i] = [
                self.nominal_waypoints[i][0] + s * dx,
                self.nominal_waypoints[i][1] + s * dy,
                self.nominal_waypoints[i][2] + s * dz,
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
    ///
    /// Precomputes which penalties are active once per evaluation. Within the
    /// hot loop:
    /// - Zero-weight penalties are skipped via outer flags (no function call).
    /// - `AlphaState` (α, zB) is computed only when any flatness-dependent
    ///   penalty is active. It is always well-defined (unlike the old
    ///   "None if zb_z ≤ -0.9" guard, which incorrectly skipped tilt).
    /// - `FlatnessState` (adds dzB, ω) is computed only when the body-rate
    ///   penalty is active *and* zB is away from the inversion singularity.
    /// - `assemble_basis_gradients` and the per-sample cost term are skipped
    ///   entirely when no penalty activated at this point (feasible sample).
    fn accumulate_dynamics_penalties(&mut self, traj: &PiecewisePolynomial) -> f32 {
        let n_check = self.params.num_check_per_piece;
        let inv_n = 1.0 / n_check as f32;

        // Active-penalty flags, cached once per evaluation.
        let need_vel = self.params.weight_vel > 1e-6;
        let need_thrust = self.params.weight_thrust > 1e-6;
        let need_tilt = self.params.weight_tilt > 1e-6;
        let need_body_rate = self.params.weight_body_rate > 1e-6;
        let need_alpha = need_thrust || need_tilt || need_body_rate;

        // Nothing to do if no penalty is active.
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

                let dd = eval_dynamics_derivatives(traj, seg, t_local);
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
                    // Body rate requires the 1/(1+zb_z) factor — guard against
                    // near-inversion where that term blows up.
                    if need_body_rate && alpha.zb[2] > -0.9 {
                        let fs = flatness::extend_to_flatness(&alpha, dd.jer);
                        penalty += self.body_rate_penalty(&fs, &mut grads);
                    }
                }

                // Feasible sample: every penalty returned (0, 0) and grads
                // are all zero. Skip basis assembly + time chain (all zero).
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
        // Time gradients: ∂L/∂K = ∂T/∂K · ∂L/∂T (quadratic parameterization).
        for i in 0..self.n_pieces {
            grad[i] = back_propagate_t(x[i], self.grad_times[i]);
        }
        // Waypoint gradients via stereographic Jacobian:
        //   P = P̂ + s·D,   s = 2r / (‖D‖² + 1)
        //   ∂P/∂D = s·I − (s²/r)·D Dᵀ   (symmetric)
        //   ∂L/∂D = (∂P/∂D)ᵀ · ∂L/∂P = s·g − (s²/r) · (D·g) · D
        let r = self.waypoint_radius;
        for i in 0..self.n_waypoints {
            let dx = x[self.dim_k + 3 * i];
            let dy = x[self.dim_k + 3 * i + 1];
            let dz = x[self.dim_k + 3 * i + 2];
            let norm_sq = dx * dx + dy * dy + dz * dz;
            let s = 2.0 * r / (norm_sq + 1.0);
            let g = self.grad_points[i];
            let dot_dg = dx * g[0] + dy * g[1] + dz * g[2];
            let coeff = s * s / r;
            grad[self.dim_k + 3 * i]     = s * g[0] - coeff * dx * dot_dg;
            grad[self.dim_k + 3 * i + 1] = s * g[1] - coeff * dy * dot_dg;
            grad[self.dim_k + 3 * i + 2] = s * g[2] - coeff * dz * dot_dg;
        }
    }

    // -----------------------------------------------------------------------
    // Individual penalties (take &self, write to stack-local PointGradients)
    // -----------------------------------------------------------------------

    // Per-penalty methods: the outer loop guards with `need_*` flags, so
    // these are only called when the corresponding weight is > 1e-6. Each
    // still returns 0.0 when the constraint is satisfied (no violation).

    /// Velocity penalty: smoothed_l1(‖v‖² - max_vel²).
    #[inline]
    fn velocity_penalty(&self, dd: &DynDerivatives, grads: &mut PointGradients) -> f32 {
        let violation = norm_sq3(dd.vel) - self.bounds.max_vel_sq;
        let (f, df) = smoothed_l1(violation, self.params.smoothing_eps);
        if f > 0.0 {
            let w = self.params.weight_vel;
            let scale = w * df * 2.0;
            for d in 0..3 {
                grads.vel[d] += scale * dd.vel[d];
            }
            w * f
        } else {
            0.0
        }
    }

    /// Collective thrust penalty: F = mass·‖α‖, penalty on (F − F_mean)² − F_radius².
    #[inline]
    fn thrust_penalty(&self, alpha: &AlphaState, grads: &mut PointGradients) -> f32 {
        let collective = self.bounds.mass * alpha.norm_alpha;
        let delta = collective - self.bounds.thr_mean;
        let (f, df) = smoothed_l1(delta * delta - self.bounds.thr_radi_sq, self.params.smoothing_eps);
        if f > 0.0 {
            let w = self.params.weight_thrust;
            let d_violation = 2.0 * delta * self.bounds.mass * alpha.inv_norm_alpha;
            let scale = w * df * d_violation;
            for d in 0..3 {
                grads.acc[d] += scale * alpha.alpha[d];
            }
            w * f
        } else {
            0.0
        }
    }

    /// Tilt angle penalty: smoothed_l1(cos_max_tilt - zb_z).
    ///
    /// Works in cosine domain to avoid 1/sin singularity; safe to evaluate
    /// across the full attitude range (including near inversion).
    #[inline]
    fn tilt_penalty(&self, alpha: &AlphaState, grads: &mut PointGradients) -> f32 {
        let violation = self.bounds.cos_max_tilt - alpha.zb[2];
        let (f, df) = smoothed_l1(violation, self.params.smoothing_eps);
        if f > 0.0 {
            let w = self.params.weight_tilt;
            let inv = alpha.inv_norm_alpha;
            let inv3 = inv * inv * inv;
            let common = w * df;
            let axy_sq = alpha.alpha[0] * alpha.alpha[0] + alpha.alpha[1] * alpha.alpha[1];
            grads.acc[0] += common * alpha.alpha[0] * alpha.alpha[2] * inv3;
            grads.acc[1] += common * alpha.alpha[1] * alpha.alpha[2] * inv3;
            grads.acc[2] -= common * axy_sq * inv3;
            w * f
        } else {
            0.0
        }
    }

    /// Body rate penalty: smoothed_l1(ω_xy² − max²) + smoothed_l1(ω_z² − max²).
    ///
    /// Matches the C++ `addBodyratePenalities`: ‖ω_xy‖² bounded by a single
    /// scalar `maxOmgXYSqr` (pitch/roll scalar limit).
    #[inline]
    fn body_rate_penalty(&self, fs: &FlatnessState, grads: &mut PointGradients) -> f32 {
        let omega_xy_sq = fs.omega[0] * fs.omega[0] + fs.omega[1] * fs.omega[1];
        let (f_xy, df_xy) = smoothed_l1(
            omega_xy_sq - self.bounds.max_rate_xy_sq,
            self.params.smoothing_eps,
        );
        let (f_z, df_z) = smoothed_l1(
            fs.omega[2] * fs.omega[2] - self.bounds.max_rate_z_sq,
            self.params.smoothing_eps,
        );

        if f_xy > 0.0 || f_z > 0.0 {
            let w = self.params.weight_body_rate;
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
    ///
    /// Only called when `penalty > 0.0` (outer loop guard), so `grads` is
    /// guaranteed non-trivial.
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
        self.partial_grad_t[seg] += time_chain * scale * t_frac + node * inv_n * penalty;
    }
}
