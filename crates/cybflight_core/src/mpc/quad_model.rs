//! Reduced quadrotor dynamics model for NMPC (ENU + FLU).
//!
//! State   x\[10\]: \[px, py, pz,  qx, qy, qz, qw,  vx, vy, vz\]
//! Control u\[4\]:  \[thrust_collective, ωx, ωy, ωz\]   — collective thrust [N] + body-rate setpoint [rad/s]
//!
//! This is the "outer-loop only" companion to [`super::full_quad_model::FullQuadModel`].
//! The body-rate dynamics (Euler rigid-body equations) and per-motor mixing are
//! delegated to an inner-loop controller; the MPC commands a body-rate setpoint
//! directly. As a consequence the state has no `[wx, wy, wz]` rows and the
//! control vector replaces the four motor thrusts with one collective thrust
//! plus three body-rate commands.
//!
//! Inertial frame : ENU (x = East, y = North, z = Up).
//! Body frame     : FLU (x = Forward, y = Left, z = Up).
//! Quaternion     : \[qx, qy, qz, qw\] scalar-LAST, FLU-body → ENU-world.
//! Thrust direction: +body z (upward in FLU).
//! Gravity        : −g along world z  (subtracted, so `grav` field is positive 9.81).
//!
//! Conventions match `full_quad_model.rs` exactly: plain `SVector<f32, NX>` arrays,
//! Default + from_vehicle_params + new constructors, owned cost weights pulled
//! from `VehicleParams.mpc`, projection-method RK4 with quaternion normalization
//! between every stage, and a public `normalize_quat` boundary helper.

use super::model_utils;
use nalgebra::{SMatrix, SVector, Vector4, vector};

pub const NX: usize = 10;
pub const NU: usize = 4;
pub const N: usize = 20;

/// Project the quaternion components of a 10-state vector back onto the
/// unit 3-sphere. Thin wrapper around the const-generic
/// [`super::model_utils::normalize_quat`] kept here for backward-compatible API.
///
/// Called by `propagate_rk4` / `propagate_euler` automatically — callers do
/// not need to invoke it themselves. Public so external boundary code can
/// normalize once before handing data into the solver.
#[inline]
pub fn normalize_quat(x: &mut SVector<f32, NX>) {
    model_utils::normalize_quat(x);
}

/// Position-error cost mode.
///
/// `Quadratic` is the standard `w_pos · ‖p − p_ref‖²` cost — the default,
/// byte-identical to the pre-MPCTC behavior.
///
/// `Contouring` activates **MPCTC** (Model Predictive Contouring Tracking
/// Control): the position error is decomposed along the path tangent (read
/// from `xref[7..10]` — the velocity reference written by the sampler).
/// The two scalar weights are pulled from the existing `w_pos` array so the
/// flash-tunable surface stays a single 3-element vector:
///
/// - `w_pos[0]` is the **contour weight** (penalizes orthogonal-to-path error)
/// - `w_pos[2]` is the **lag weight** (penalizes along-path lag)
/// - `w_pos[1]` is **unused** in this mode
///
/// When the two are equal, the cost is mathematically identical to the
/// `Quadratic` mode by orthogonal decomposition. Lowering `w_pos[2]` lets
/// the drone slide along the path under saturation while still pulling
/// hard in the contour direction.
///
/// The tangent normalization is guarded by [`PosCostMode::VEL_EPS`]: when
/// `‖vel_ref‖ < VEL_EPS` (hover, mission start, terminal), the cost
/// degenerates to the `w_contour · ‖e‖²` form (lag term vanishes).
///
/// **Sampler-pairing safety invariant.** `Contouring` requires
/// [`PositionSampler`] (firmware feature `position_sampler`). The position
/// sampler's closest-point search produces an `xref[7..10]` tangent that
/// is meaningful at every horizon step — so the contour/lag decomposition
/// reflects the geometric path. Pairing `Contouring` with `TimeSampler`
/// is unsafe in firmware operation: a time-anchored sample at hover, in a
/// stall, or after a planner timeout produces zero or near-zero `vel_ref`
/// for stretches of the horizon, dragging cost stages through the
/// `VEL_EPS` fallback inconsistently and chattering between contour and
/// isotropic regimes. The firmware enforces this invariant in
/// `outer_loop::build_outer_quad_model`, which clamps `pos_cost_mode` to
/// `Quadratic` whenever the `position_sampler` feature is off,
/// regardless of `vp.mpc.pos_cost_mode`.
///
/// [`PositionSampler`]: crate::trajectory_planning::sampler::PositionSampler
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PosCostMode {
    Quadratic,
    Contouring,
}

impl PosCostMode {
    /// Tangent-normalization guard for the contour/lag decomposition.
    /// Below this `‖vel_ref‖` threshold the lag term vanishes and the
    /// cost falls back to `w_contour · ‖e‖²` (correct hover behavior).
    /// Fixed because retuning it bench-side is rare; if you need it
    /// configurable, promote to a `MpcParams` field.
    pub const VEL_EPS: f32 = 0.1;
}

impl Default for PosCostMode {
    fn default() -> Self {
        PosCostMode::Quadratic
    }
}

#[derive(Clone)]
pub struct QuadModel {
    pub mass: f32,
    pub grav: f32,
    pub dt: f32,
    /// Per-control box bounds: index 0 = thrust [N], 1..4 = body rates [rad/s].
    pub u_bounds: [[f32; 2]; NU],
    /// Precomputed 1.0 / mass.
    pub mass_inv: f32,
    // ── Stage cost weights (mirrored from VehicleParams.mpc) ───────────
    pub w_pos: [f32; 3],
    pub w_vel: [f32; 3],
    pub w_att: [f32; 3],
    /// Per-component input weight: `w_input[0]` weights collective thrust [N²],
    /// `w_input[1..4]` weight body-rate commands [(rad/s)²]. Per-element
    /// (rather than scalar) because the four control channels are physically
    /// heterogeneous (one force, three angular rates).
    pub w_input: SVector<f32, NU>,
    /// Cubic constraint penalty weight (input bound enforcement).
    pub rho: f32,
    /// Position-cost formulation. Default [`PosCostMode::Quadratic`] is
    /// byte-identical to the pre-MPCTC behavior; switch to
    /// [`PosCostMode::Contouring`] to activate MPCTC. See the enum doc.
    pub pos_cost_mode: PosCostMode,
}

impl Default for QuadModel {
    /// Default mass matches `vehicle.rs::QUADROTOR_BODY` (0.55 kg). Default
    /// weights are taken from `MpcParams::default()` for position / velocity /
    /// attitude blocks; the input-weight default is uniform 1.0. Default
    /// control bounds:
    /// - Thrust ceiling = 4 × 8.5 N = 34 N (sum of per-motor `max_thrust_n`
    ///   from `vehicle.rs::QUADROTOR_MOTORS`); thrust floor = m·g·0.1.
    /// - Body-rate ceiling matches `QUADROTOR_BODY.max_rate_rad_s` =
    ///   [10, 10, 6] rad/s (roll, pitch, yaw).
    fn default() -> Self {
        let mass = 0.58;
        let grav = 9.81;
        // Sum of per-motor max thrusts from QUADROTOR_MOTORS (4 × 8.5 N).
        let max_collective_thrust_n: f32 = 4.0 * 12.0;
        Self {
            mass,
            grav,
            dt: 0.05,
            u_bounds: [
                [0.0_f32, max_collective_thrust_n],
                [-10.0, 10.0],
                [-10.0, 10.0],
                [-6.0, 6.0],
            ],
            mass_inv: 1.0 / mass,
            w_pos: [200.0, 200.0, 200.0],
            w_vel: [10.0, 10.0, 10.0],
            w_att: [5.0, 5.0, 200.0],
            w_input: Vector4::new(1.0, 20.0, 20.0, 20.0),
            rho: 1e4,
            pos_cost_mode: PosCostMode::Quadratic,
        }
    }
}

impl QuadModel {
    /// Construct from firmware vehicle parameters.
    ///
    /// Mass and `grav` come from `vp.body`. Cost weights, integration timestep,
    /// and constraint penalty are sourced from `vp.mpc`. Control bounds:
    /// - Collective thrust upper bound = sum of per-motor `max_thrust_n` from
    ///   `vp.motors[*]` — the actual physical ceiling of the airframe.
    /// - Collective thrust lower bound = `mass·g·0.1` (10% of hover) as a
    ///   minimum-throttle margin so the model never commands cut-off.
    /// - Per-axis body-rate bounds = `±vp.body.max_rate_rad_s[i]`.
    pub fn from_vehicle_params(vp: &crate::params::VehicleParams) -> Self {
        let mass = vp.body.mass_kg;
        let grav = 9.81;
        let mr = vp.body.max_rate_rad_s;
        // Total collective thrust ceiling = Σ per-motor max thrusts.
        let mut max_collective_thrust_n = 0.0_f32;
        for m in &vp.motors {
            max_collective_thrust_n += m.max_thrust_n;
        }
        let thrust_percentage = 0.75;
        Self {
            mass,
            grav,
            dt: vp.mpc.dt,
            u_bounds: [
                [0.0_f32, max_collective_thrust_n * thrust_percentage],
                // [0.0_f32, 36.0_f32],
                [-mr[0], mr[0]],
                [-mr[1], mr[1]],
                [-mr[2], mr[2]],
            ],
            mass_inv: 1.0 / mass,
            w_pos: vp.mpc.pos_weight,
            w_vel: vp.mpc.vel_weight,
            w_att: vp.mpc.att_weight,
            // Input cost is heterogeneous in this model (thrust + 3 rates),
            // so we replicate `vp.mpc.thrust_weight` (a scalar) across all four
            // channels. Override the struct field directly if asymmetric tuning
            // is needed.
            w_input: Vector4::new(
                vp.mpc.thrust_weight,
                vp.mpc.rate_weight[0],
                vp.mpc.rate_weight[1],
                vp.mpc.rate_weight[2],
            ),
            rho: vp.mpc.rho,
            pos_cost_mode: vp.mpc.pos_cost_mode,
        }
    }

    /// Construct with overridden mass / gravity / dt; everything else
    /// (including the per-motor-sum thrust ceiling) is inherited from
    /// `Default::default()`. Only the thrust *floor* is rescaled to track
    /// the new mass (`mass·g·0.1` minimum-throttle margin).
    pub fn new(mass: f32, grav: f32, dt: f32) -> Self {
        let base = Self {
            mass,
            grav,
            dt,
            ..Default::default()
        };
        Self {
            u_bounds: [
                [0.0_f32, base.u_bounds[0][1]],
                base.u_bounds[1],
                base.u_bounds[2],
                base.u_bounds[3],
            ],
            mass_inv: 1.0 / mass,
            ..base
        }
    }

    /// Continuous-time dynamics: ẋ = f(x, u).
    ///
    /// Thrust acts along +body z (FLU), rotated to ENU via R(q).
    /// Gravity is subtracted: `a = R[:,2] * c/m − [0, 0, grav]`.
    pub fn dynamics(&self, x: &SVector<f32, NX>, u: &SVector<f32, NU>) -> SVector<f32, NX> {
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (vx, vy, vz) = (x[7], x[8], x[9]);
        let (c, wx, wy, wz) = (u[0], u[1], u[2], u[3]);
        let m_inv = self.mass_inv;

        let ct_m = c * m_inv;

        let mut xdot = SVector::<f32, NX>::zeros();
        xdot[0] = vx;
        xdot[1] = vy;
        xdot[2] = vz;
        // Quaternion kinematics: q_dot = 0.5 * q ⊗ [wx, wy, wz, 0]
        xdot[3] = 0.5 * (qw * wx + qy * wz - qz * wy);
        xdot[4] = 0.5 * (qw * wy - qx * wz + qz * wx);
        xdot[5] = 0.5 * (qw * wz + qx * wy - qy * wx);
        xdot[6] = 0.5 * (-qx * wx - qy * wy - qz * wz);
        // Translational dynamics: a = +R[:,2] * c/m − [0,0,g]
        xdot[7] = 2.0 * (qw * qy + qx * qz) * ct_m;
        xdot[8] = 2.0 * (qy * qz - qw * qx) * ct_m;
        xdot[9] = (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * ct_m - self.grav;
        xdot
    }

    /// Compute xdot, df/dx (NX x NX), df/du (NX x NU).
    pub fn dynamics_jac(
        &self,
        x: &SVector<f32, NX>,
        u: &SVector<f32, NU>,
    ) -> (SVector<f32, NX>, SMatrix<f32, NX, NX>, SMatrix<f32, NX, NU>) {
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (vx, vy, vz) = (x[7], x[8], x[9]);
        let (c, wx, wy, wz) = (u[0], u[1], u[2], u[3]);
        let m_inv = self.mass_inv;

        let ct_m = c * m_inv;

        let xdot = vector![
            vx,
            vy,
            vz,
            0.5 * (qw * wx + qy * wz - qz * wy),
            0.5 * (qw * wy - qx * wz + qz * wx),
            0.5 * (qw * wz + qx * wy - qy * wx),
            0.5 * (-qx * wx - qy * wy - qz * wz),
            2.0 * (qw * qy + qx * qz) * ct_m,
            2.0 * (qy * qz - qw * qx) * ct_m,
            (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * ct_m - self.grav
        ];

        // df/dx (10 × 10)
        let mut jx = SMatrix::<f32, NX, NX>::zeros();
        // Rows 0-2: dp/dt = v
        jx[(0, 7)] = 1.0;
        jx[(1, 8)] = 1.0;
        jx[(2, 9)] = 1.0;

        // Rows 3-6: quaternion kinematics (only ∂/∂q; ∂/∂ω is in df/du since
        // ω is an input here, not a state)
        let (hw_x, hw_y, hw_z) = (0.5 * wx, 0.5 * wy, 0.5 * wz);

        jx[(3, 4)] = hw_z;
        jx[(3, 5)] = -hw_y;
        jx[(3, 6)] = hw_x;

        jx[(4, 3)] = -hw_z;
        jx[(4, 5)] = hw_x;
        jx[(4, 6)] = hw_y;

        jx[(5, 3)] = hw_y;
        jx[(5, 4)] = -hw_x;
        jx[(5, 6)] = hw_z;

        jx[(6, 3)] = -hw_x;
        jx[(6, 4)] = -hw_y;
        jx[(6, 5)] = -hw_z;

        // Rows 7-9: d(accel)/d(quaternion)  (thrust rotation Jacobian)
        let dc_qw = 2.0 * ct_m * qw;
        let dc_qx = 2.0 * ct_m * qx;
        let dc_qy = 2.0 * ct_m * qy;
        let dc_qz = 2.0 * ct_m * qz;

        jx[(7, 3)] = dc_qz;
        jx[(7, 4)] = dc_qw;
        jx[(7, 5)] = dc_qx;
        jx[(7, 6)] = dc_qy;
        jx[(8, 3)] = -dc_qw;
        jx[(8, 4)] = dc_qz;
        jx[(8, 5)] = dc_qy;
        jx[(8, 6)] = -dc_qx;
        jx[(9, 3)] = -2.0 * dc_qx;
        jx[(9, 4)] = -2.0 * dc_qy;

        // df/du (10 × 4)
        // Column 0 (collective thrust c) drives only translational accel (rows 7-9).
        // Columns 1-3 (ωx, ωy, ωz) drive quaternion kinematics (rows 3-6).
        let mut ju = SMatrix::<f32, NX, NU>::zeros();
        let a1_m = 2.0 * (qw * qy + qx * qz) * m_inv;
        let a2_m = 2.0 * (qy * qz - qw * qx) * m_inv;
        let a3_m = (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * m_inv;
        ju[(7, 0)] = a1_m;
        ju[(8, 0)] = a2_m;
        ju[(9, 0)] = a3_m;

        let (hq_w, hq_x, hq_y, hq_z) = (0.5 * qw, 0.5 * qx, 0.5 * qy, 0.5 * qz);
        ju[(3, 1)] = hq_w;
        ju[(3, 2)] = -hq_z;
        ju[(3, 3)] = hq_y;
        ju[(4, 1)] = hq_z;
        ju[(4, 2)] = hq_w;
        ju[(4, 3)] = -hq_x;
        ju[(5, 1)] = -hq_y;
        ju[(5, 2)] = hq_x;
        ju[(5, 3)] = hq_w;
        ju[(6, 1)] = -hq_x;
        ju[(6, 2)] = -hq_y;
        ju[(6, 3)] = -hq_z;

        (xdot, jx, ju)
    }

    /// RK4 integration with projection-method quaternion handling.
    ///
    /// Each RK4 intermediate state is projected back onto the unit sphere
    /// before being fed to the next `dynamics` evaluation, and the final
    /// result is also projected. This bounds intermediate-stage drift to
    /// one normalization's worth of error per stage instead of letting it
    /// compound across the four sub-steps.
    pub fn propagate_rk4(&self, xk: &SVector<f32, NX>, uk: &SVector<f32, NU>) -> SVector<f32, NX> {
        let dt = self.dt;
        let half_dt = 0.5 * dt;

        let k0 = self.dynamics(xk, uk);
        let mut x1 = xk + k0 * half_dt;
        normalize_quat(&mut x1);
        let k1 = self.dynamics(&x1, uk);

        x1 = xk + k1 * half_dt;
        normalize_quat(&mut x1);
        let k2 = self.dynamics(&x1, uk);

        x1 = xk + k2 * dt;
        normalize_quat(&mut x1);
        let k3 = self.dynamics(&x1, uk);

        let mut result = xk + (k0 + 2.0 * k1 + 2.0 * k2 + k3) * (dt / 6.0);
        normalize_quat(&mut result);
        result
    }

    /// Forward Euler integration with quaternion projection.
    pub fn propagate_euler(
        &self,
        xk: &SVector<f32, NX>,
        uk: &SVector<f32, NU>,
    ) -> SVector<f32, NX> {
        let xdot = self.dynamics(xk, uk);
        let mut result = xk + xdot * self.dt;
        normalize_quat(&mut result);
        result
    }

    /// Euler sensitivity: F_x = I + dt*df/dx,  F_u = dt*df/du.
    pub fn propagate_euler_grad(
        &self,
        xk: &SVector<f32, NX>,
        uk: &SVector<f32, NU>,
    ) -> (SMatrix<f32, NX, NX>, SMatrix<f32, NX, NU>) {
        let (_, jac_x, jac_u) = self.dynamics_jac(xk, uk);
        let dt = self.dt;
        let fx = SMatrix::<f32, NX, NX>::identity() + jac_x * dt;
        let fu = jac_u * dt;
        (fx, fu)
    }

    // ── Cost functions ──────────────────────────────────────────────────
    //
    // All cost-function components are computed via `super::model_utils` so the
    // math is identical to `FullQuadModel`. This model has no body-rate
    // state, so there is no body-rate cost block to inline locally.

    /// Stage state cost + gradient. Returns cost, writes grad_x.
    pub fn state_cost_grad(
        &self,
        x: &SVector<f32, NX>,
        xref: &SVector<f32, NX>,
        grad_x: &mut SVector<f32, NX>,
    ) -> f32 {
        let dt = self.dt;
        let mut cost = match self.pos_cost_mode {
            PosCostMode::Quadratic => {
                model_utils::write_pos_vel_cost_grad(x, xref, &self.w_pos, &self.w_vel, dt, grad_x)
            }
            PosCostMode::Contouring => {
                // w_pos[0] = contour weight, w_pos[2] = lag weight (see PosCostMode doc).
                let pos = model_utils::write_contour_lag_cost_grad(
                    x,
                    xref,
                    self.w_pos[0],
                    self.w_pos[2],
                    PosCostMode::VEL_EPS,
                    dt,
                    grad_x,
                );
                pos + model_utils::write_vel_cost_grad(x, xref, &self.w_vel, dt, grad_x)
            }
        };
        let (ea, de, dqa_dq) = model_utils::attitude_error(x, xref);
        cost += model_utils::write_quat_cost_grad(&ea, &de, &dqa_dq, &self.w_att, dt, grad_x);
        cost
    }

    /// Gauss-Newton Hessian + gradient in a single pass.
    pub fn state_cost_hess_grad(
        &self,
        x: &SVector<f32, NX>,
        xref: &SVector<f32, NX>,
        grad_x: &mut SVector<f32, NX>,
        hess_xx: &mut SMatrix<f32, NX, NX>,
    ) -> f32 {
        let dt = self.dt;

        // ── Pos + vel cost/gradient (mode-dependent for pos) ──
        let mut cost = match self.pos_cost_mode {
            PosCostMode::Quadratic => {
                model_utils::write_pos_vel_cost_grad(x, xref, &self.w_pos, &self.w_vel, dt, grad_x)
            }
            PosCostMode::Contouring => {
                let pos = model_utils::write_contour_lag_cost_grad(
                    x,
                    xref,
                    self.w_pos[0],
                    self.w_pos[2],
                    PosCostMode::VEL_EPS,
                    dt,
                    grad_x,
                );
                pos + model_utils::write_vel_cost_grad(x, xref, &self.w_vel, dt, grad_x)
            }
        };

        // ── Quaternion cost/gradient via shared helpers ──
        let (ea, de, dqa_dq) = model_utils::attitude_error(x, xref);
        cost += model_utils::write_quat_cost_grad(&ea, &de, &dqa_dq, &self.w_att, dt, grad_x);

        // ── Hessian ──
        hess_xx.fill(0.0);
        match self.pos_cost_mode {
            PosCostMode::Quadratic => {
                model_utils::write_pos_vel_hess(&self.w_pos, &self.w_vel, dt, hess_xx)
            }
            PosCostMode::Contouring => {
                model_utils::write_contour_lag_hess(
                    xref,
                    self.w_pos[0],
                    self.w_pos[2],
                    PosCostMode::VEL_EPS,
                    dt,
                    hess_xx,
                );
                model_utils::write_vel_hess(&self.w_vel, dt, hess_xx);
            }
        };
        model_utils::write_quat_hess(&de, &dqa_dq, &self.w_att, dt, hess_xx);

        cost
    }

    /// Input cost + gradient.
    pub fn input_cost_grad(
        &self,
        u: &SVector<f32, NU>,
        uref: &SVector<f32, NU>,
        grad_u: &mut SVector<f32, NU>,
    ) -> f32 {
        let dt = self.dt;
        let eu = u - uref;
        *grad_u = eu.component_mul(&self.w_input) * (2.0 * dt);
        dt * eu.component_mul(&eu).dot(&self.w_input)
    }

    /// Cubic box-constraint penalty.
    pub fn constraint_hess_grad(
        &self,
        u: &SVector<f32, NU>,
        grad_u: &mut SVector<f32, NU>,
        r_diag: &mut SVector<f32, NU>,
    ) -> f32 {
        model_utils::constraint_hess_grad(u, &self.u_bounds, self.rho, grad_u, r_diag)
    }

    /// Full stage cost = state cost + input cost + constraint penalty.
    pub fn stage_cost_hess_grad(
        &self,
        x: &SVector<f32, NX>,
        u: &SVector<f32, NU>,
        xref: &SVector<f32, NX>,
        uref: &SVector<f32, NU>,
        hess_xx: &mut SMatrix<f32, NX, NX>,
        r_diag: &mut SVector<f32, NU>,
        grad_x: &mut SVector<f32, NX>,
        grad_u: &mut SVector<f32, NU>,
    ) -> f32 {
        let dt = self.dt;
        let mut cost = self.state_cost_hess_grad(x, xref, grad_x, hess_xx);

        let eu = u - uref;
        *grad_u = eu.component_mul(&self.w_input) * (2.0 * dt);
        *r_diag = self.w_input * (2.0 * dt);
        cost += dt * eu.component_mul(&eu).dot(&self.w_input);

        cost += self.constraint_hess_grad(u, grad_u, r_diag);
        cost
    }

    /// Clamp control to bounds.
    pub fn clamp_control(&self, u: &SVector<f32, NU>) -> SVector<f32, NU> {
        model_utils::clamp_control(u, &self.u_bounds)
    }
}
