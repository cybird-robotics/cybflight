//! Full quadrotor rigid-body dynamics model for NMPC (ENU + FLU).
//!
//! State   x\[13\]: \[px, py, pz,  qx, qy, qz, qw,  vx, vy, vz,  wx, wy, wz\]
//! Control u\[4\]:  \[f1, f2, f3, f4\]  per-motor thrusts \[N\], Betaflight QuadX ordering
//!
//! Inertial frame : ENU (x = East, y = North, z = Up).
//! Body frame     : FLU (x = Forward, y = Left, z = Up).
//! Quaternion     : \[qx, qy, qz, qw\] scalar-LAST, FLU-body → ENU-world.
//! Thrust direction: +body z (upward in FLU).
//! Gravity        : −g along world z  (subtracted, so `grav` field is positive 9.81).
//!
//! Motor ordering (Betaflight QuadX, FLU frame):
//!   0 = REAR\_RIGHT  (CW)  pos (−d, −d)
//!   1 = FRONT\_RIGHT (CCW) pos (+d, −d)
//!   2 = REAR\_LEFT   (CCW) pos (−d, +d)
//!   3 = FRONT\_LEFT  (CW)  pos (+d, +d)

use super::model_utils;

use nalgebra::{SMatrix, SVector, Vector3};

pub const NX: usize = 13;
pub const NU: usize = 4;
pub const N: usize = 20;

/// Project the quaternion components of a 13-state vector back onto the
/// unit 3-sphere. Thin wrapper around the const-generic
/// [`super::model_utils::normalize_quat`] kept here for backward-compatible API.
///
/// Called by `propagate_rk4` / `propagate_euler` automatically — callers do
/// not need to invoke it themselves. Public so external boundary code (e.g.
/// the MPC driver task pulling state from the estimator) can normalize once
/// before handing data into the solver.
#[inline]
pub fn normalize_quat(x: &mut SVector<f32, NX>) {
    model_utils::normalize_quat(x);
}

#[derive(Clone)]
pub struct FullQuadModel {
    pub mass: f32,
    pub grav: f32,
    pub dt: f32,
    pub inertia: [f32; 3],
    /// Per-motor position \[x, y\] in the body FLU frame (metres).
    pub motor_pos: [[f32; 2]; NU],
    /// Per-motor signed yaw coefficient: `spin_sign * torque_coeff`.
    /// Positive for CW (from above), negative for CCW.
    pub motor_yaw_coeff: SVector<f32, NU>,
    pub u_bounds: [[f32; 2]; NU],
    /// Precomputed 1.0 / mass.
    pub mass_inv: f32,
    /// Precomputed [1.0/Ixx, 1.0/Iyy, 1.0/Izz].
    pub inertia_inv: [f32; 3],
    // ── Stage cost weights (mirrored from VehicleParams.mpc) ───────────
    pub w_pos: [f32; 3],
    pub w_vel: [f32; 3],
    pub w_att: [f32; 3],
    pub w_rate: [f32; 3],
    /// Control effort weight (uniform across motors).
    pub w_thrust: f32,
    /// Cubic constraint penalty weight (input bound enforcement).
    pub rho: f32,
}

impl Default for FullQuadModel {
    /// Default matches the Betaflight QuadX layout in `vehicle.rs` and the
    /// `MpcParams::default()` cost weights in `params.rs`.
    fn default() -> Self {
        Self {
            mass: 0.55,
            grav: 9.81,
            dt: 0.05,
            inertia: [0.0025, 0.0021, 0.0043],
            motor_pos: [
                [-0.075, -0.1], // M0: rear-right
                [0.075, -0.1],  // M1: front-right
                [-0.075, 0.1],  // M2: rear-left
                [0.075, 0.1],   // M3: front-left
            ],
            motor_yaw_coeff: [
                0.022,  // M0: CW  → positive yaw reaction
                -0.022, // M1: CCW → negative yaw reaction
                -0.022, // M2: CCW → negative yaw reaction
                0.022,  // M3: CW  → positive yaw reaction
            ]
            .into(),
            u_bounds: [[0.0, 8.5]; NU],
            mass_inv: 1.0 / 0.55,
            inertia_inv: [1.0 / 0.0025, 1.0 / 0.0021, 1.0 / 0.0043],
            w_pos: [200.0, 200.0, 200.0],
            w_vel: [1.0, 1.0, 1.0],
            w_att: [5.0, 5.0, 200.0],
            w_rate: [1.0, 1.0, 1.0],
            w_thrust: 6.0,
            rho: 1e4,
        }
    }
}

impl FullQuadModel {
    /// Construct from firmware vehicle parameters.
    ///
    /// Extracts mass, diagonal inertia, motor geometry, and thrust bounds
    /// from the canonical `VehicleParams`, ensuring consistency with the
    /// inner-loop controller and mixer. Cost weights, integration timestep,
    /// and constraint penalty are sourced from `vp.mpc` so a single
    /// `crate::params::set` call retunes the entire MPC.
    pub fn from_vehicle_params(vp: &crate::params::VehicleParams) -> Self {
        let mass = vp.body.mass_kg;
        let inertia = [
            vp.body.inertia_kg_m2[0], // Ixx
            vp.body.inertia_kg_m2[4], // Iyy
            vp.body.inertia_kg_m2[8], // Izz
        ];
        let mut motor_pos = [[0.0f32; 2]; NU];
        let mut motor_yaw_coeff = SVector::<f32, NU>::zeros();
        let mut u_bounds = [[0.0f32; 2]; NU];
        for i in 0..NU {
            motor_pos[i] = vp.motors[i].position_m;
            motor_yaw_coeff[i] =
                (vp.motors[i].spin_dir as i32 as f32) * vp.motors[i].torque_coeff_m;
            u_bounds[i] = [0.0, vp.motors[i].max_thrust_n];
        }
        Self {
            mass,
            grav: 9.81,
            dt: vp.mpc.dt,
            inertia,
            motor_pos,
            motor_yaw_coeff,
            u_bounds,
            mass_inv: 1.0 / mass,
            inertia_inv: [1.0 / inertia[0], 1.0 / inertia[1], 1.0 / inertia[2]],
            w_pos: vp.mpc.pos_weight,
            w_vel: vp.mpc.vel_weight,
            w_att: vp.mpc.att_weight,
            w_rate: vp.mpc.rate_weight,
            w_thrust: vp.mpc.thrust_weight,
            rho: vp.mpc.rho,
        }
    }

    pub fn new(mass: f32, grav: f32, dt: f32) -> Self {
        let base = Self {
            mass,
            grav,
            dt,
            ..Default::default()
        };
        Self {
            mass_inv: 1.0 / mass,
            inertia_inv: [
                1.0 / base.inertia[0],
                1.0 / base.inertia[1],
                1.0 / base.inertia[2],
            ],
            ..base
        }
    }

    /// Motor allocation: returns (f_total, tau_x, tau_y, tau_z).
    ///
    /// Uses per-motor positions and yaw coefficients so the allocation
    /// matrix is identical to the firmware's `MotorEffectiveness` / G1.
    ///
    /// `tau = r × F`:  tau\_x = py·T,  tau\_y = −px·T,  tau\_z = yaw\_coeff·T.
    #[inline]
    pub fn alloc(&self, u: &SVector<f32, NU>) -> (f32, f32, f32, f32) {
        let pos_x = SVector::<f32, NU>::from_fn(|i, _| self.motor_pos[i][0]);
        let pos_y = SVector::<f32, NU>::from_fn(|i, _| self.motor_pos[i][1]);
        let f_total = u.sum();
        let tau_x = pos_y.dot(u);
        let tau_y = -pos_x.dot(u);
        let tau_z = self.motor_yaw_coeff.dot(u);
        (f_total, tau_x, tau_y, tau_z)
    }

    /// Continuous-time ENU dynamics: xdot = f(x, u).
    ///
    /// Thrust acts along +body z (FLU), rotated to ENU via R(q).
    /// Gravity is subtracted: `a = R[:,2] * T/m − [0, 0, grav]`.
    pub fn dynamics(&self, x: &SVector<f32, NX>, u: &SVector<f32, NU>) -> SVector<f32, NX> {
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (vx, vy, vz) = (x[7], x[8], x[9]);
        let (wx, wy, wz) = (x[10], x[11], x[12]);
        let m_inv = self.mass_inv;
        let [ixx_inv, iyy_inv, izz_inv] = self.inertia_inv;
        let [ixx, iyy, izz] = self.inertia;

        let (f_total, tau_x, tau_y, tau_z) = self.alloc(u);
        let ct_m = f_total * m_inv;

        let mut xdot = SVector::<f32, NX>::zeros();
        xdot[0] = vx;
        xdot[1] = vy;
        xdot[2] = vz;
        // Quaternion kinematics: q_dot = 0.5 * q ⊗ [wx, wy, wz, 0]
        xdot[3] = 0.5 * (qw * wx + qy * wz - qz * wy);
        xdot[4] = 0.5 * (qw * wy - qx * wz + qz * wx);
        xdot[5] = 0.5 * (qw * wz + qx * wy - qy * wx);
        xdot[6] = 0.5 * (-qx * wx - qy * wy - qz * wz);
        // Translational dynamics: a = +R[:,2] * T/m − [0,0,g]
        xdot[7] = 2.0 * (qw * qy + qx * qz) * ct_m;
        xdot[8] = 2.0 * (qy * qz - qw * qx) * ct_m;
        xdot[9] = (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * ct_m - self.grav;
        // Euler rigid-body equations
        xdot[10] = (tau_x - (izz - iyy) * wy * wz) * ixx_inv;
        xdot[11] = (tau_y - (ixx - izz) * wx * wz) * iyy_inv;
        xdot[12] = (tau_z - (iyy - ixx) * wx * wy) * izz_inv;
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
        let (wx, wy, wz) = (x[10], x[11], x[12]);
        let m_inv = self.mass_inv;
        let [ixx_inv, iyy_inv, izz_inv] = self.inertia_inv;
        let [ixx, iyy, izz] = self.inertia;

        let (f_total, tau_x, tau_y, tau_z) = self.alloc(u);
        let ct_m = f_total * m_inv;

        let xdot: SVector<f32, NX> = [
            vx,
            vy,
            vz,
            0.5 * (qw * wx + qy * wz - qz * wy),
            0.5 * (qw * wy - qx * wz + qz * wx),
            0.5 * (qw * wz + qx * wy - qy * wx),
            0.5 * (-qx * wx - qy * wy - qz * wz),
            2.0 * (qw * qy + qx * qz) * ct_m,
            2.0 * (qy * qz - qw * qx) * ct_m,
            (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * ct_m - self.grav,
            (tau_x - (izz - iyy) * wy * wz) * ixx_inv,
            (tau_y - (ixx - izz) * wx * wz) * iyy_inv,
            (tau_z - (iyy - ixx) * wx * wy) * izz_inv,
        ]
        .into();

        // df/dx
        let mut jx = SMatrix::<f32, NX, NX>::zeros();
        // Rows 0-2: dp/dt = v
        jx[(0, 7)] = 1.0;
        jx[(1, 8)] = 1.0;
        jx[(2, 9)] = 1.0;

        // Rows 3-6: quaternion kinematics
        let (hw_x, hw_y, hw_z) = (0.5 * wx, 0.5 * wy, 0.5 * wz);
        let (hq_w, hq_x, hq_y, hq_z) = (0.5 * qw, 0.5 * qx, 0.5 * qy, 0.5 * qz);

        jx[(3, 4)] = hw_z;
        jx[(3, 5)] = -hw_y;
        jx[(3, 6)] = hw_x;
        jx[(3, 10)] = hq_w;
        jx[(3, 11)] = -hq_z;
        jx[(3, 12)] = hq_y;

        jx[(4, 3)] = -hw_z;
        jx[(4, 5)] = hw_x;
        jx[(4, 6)] = hw_y;
        jx[(4, 10)] = hq_z;
        jx[(4, 11)] = hq_w;
        jx[(4, 12)] = -hq_x;

        jx[(5, 3)] = hw_y;
        jx[(5, 4)] = -hw_x;
        jx[(5, 6)] = hw_z;
        jx[(5, 10)] = -hq_y;
        jx[(5, 11)] = hq_x;
        jx[(5, 12)] = hq_w;

        jx[(6, 3)] = -hw_x;
        jx[(6, 4)] = -hw_y;
        jx[(6, 5)] = -hw_z;
        jx[(6, 10)] = -hq_x;
        jx[(6, 11)] = -hq_y;
        jx[(6, 12)] = -hq_z;

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

        // Rows 10-12: body-rate coupling
        jx[(10, 11)] = -(izz - iyy) * wz * ixx_inv;
        jx[(10, 12)] = -(izz - iyy) * wy * ixx_inv;
        jx[(11, 10)] = -(ixx - izz) * wz * iyy_inv;
        jx[(11, 12)] = -(ixx - izz) * wx * iyy_inv;
        jx[(12, 10)] = -(iyy - ixx) * wy * izz_inv;
        jx[(12, 11)] = -(iyy - ixx) * wx * izz_inv;

        // df/du
        let mut ju = SMatrix::<f32, NX, NU>::zeros();
        let a1_m = 2.0 * (qw * qy + qx * qz) * m_inv;
        let a2_m = 2.0 * (qy * qz - qw * qx) * m_inv;
        let a3_m = (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * m_inv;
        // Translation thrust rows — constant across motors (collective thrust only).
        ju.row_mut(7).fill(a1_m);
        ju.row_mut(8).fill(a2_m);
        ju.row_mut(9).fill(a3_m);
        // Torque allocation rows (per-motor, consistent with alloc()).
        let pos_x = SVector::<f32, NU>::from_fn(|i, _| self.motor_pos[i][0]);
        let pos_y = SVector::<f32, NU>::from_fn(|i, _| self.motor_pos[i][1]);
        ju.row_mut(10).copy_from(&(pos_y * ixx_inv).transpose());
        ju.row_mut(11).copy_from(&(pos_x * -iyy_inv).transpose());
        ju.row_mut(12)
            .copy_from(&(self.motor_yaw_coeff * izz_inv).transpose());

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

    /// Forward Euler integration.
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
    // Position, velocity, quaternion-attitude, and box-constraint terms are
    // computed via `super::model_utils` so the math is identical to `QuadModel`.
    // Only the FullQuadModel-specific body-rate state cost (indices 10..13)
    // is inlined locally.

    /// Stage state cost + gradient. Returns cost, writes grad_x.
    pub fn state_cost_grad(
        &self,
        x: &SVector<f32, NX>,
        xref: &SVector<f32, NX>,
        grad_x: &mut SVector<f32, NX>,
    ) -> f32 {
        let dt = self.dt;
        // Pos + vel cost/gradient via shared helper.
        let mut cost =
            model_utils::write_pos_vel_cost_grad(x, xref, &self.w_pos, &self.w_vel, dt, grad_x);
        // Body-rate cost/gradient — full-model only (rates are STATE here).
        let w_rate = Vector3::from(self.w_rate);
        let rate_err = x.fixed_rows::<3>(10) - xref.fixed_rows::<3>(10);
        cost += dt * rate_err.component_mul(&rate_err).dot(&w_rate);
        grad_x
            .fixed_rows_mut::<3>(10)
            .copy_from(&(rate_err.component_mul(&w_rate) * (2.0 * dt)));
        // Quaternion cost/gradient via shared helpers.
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

        // ── Pos + vel cost/gradient via shared helper ──
        let mut cost =
            model_utils::write_pos_vel_cost_grad(x, xref, &self.w_pos, &self.w_vel, dt, grad_x);

        // ── Body-rate cost/gradient (full-model only) ──
        let w_rate = Vector3::from(self.w_rate);
        let rate_err = x.fixed_rows::<3>(10) - xref.fixed_rows::<3>(10);
        cost += dt * rate_err.component_mul(&rate_err).dot(&w_rate);
        grad_x
            .fixed_rows_mut::<3>(10)
            .copy_from(&(rate_err.component_mul(&w_rate) * (2.0 * dt)));

        // ── Quaternion cost/gradient via shared helpers ──
        let (ea, de, dqa_dq) = model_utils::attitude_error(x, xref);
        cost += model_utils::write_quat_cost_grad(&ea, &de, &dqa_dq, &self.w_att, dt, grad_x);

        // ── Hessian ──
        hess_xx.fill(0.0);
        // Pos + vel diagonal block via shared helper.
        model_utils::write_pos_vel_hess(&self.w_pos, &self.w_vel, dt, hess_xx);
        // Body-rate diagonal block — full-model only.
        let rate_diag = w_rate * (2.0 * dt);
        for (offset, &v) in rate_diag.iter().enumerate() {
            hess_xx[(10 + offset, 10 + offset)] = v;
        }
        // Quaternion Hessian block via shared helper.
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
        *grad_u = eu * (2.0 * self.w_thrust * dt);
        dt * self.w_thrust * eu.norm_squared()
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
        *grad_u = eu * (2.0 * self.w_thrust * dt);
        r_diag.fill(2.0 * dt * self.w_thrust);
        cost += dt * self.w_thrust * eu.norm_squared();

        cost += self.constraint_hess_grad(u, grad_u, r_diag);
        cost
    }

    /// Clamp control to bounds.
    pub fn clamp_control(&self, u: &SVector<f32, NU>) -> SVector<f32, NU> {
        model_utils::clamp_control(u, &self.u_bounds)
    }
}
