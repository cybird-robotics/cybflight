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
pub fn normalize_quat(x: &mut [f32; NX]) {
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
    pub motor_yaw_coeff: [f32; NU],
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
            ],
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
        let mut motor_yaw_coeff = [0.0f32; NU];
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
    pub fn alloc(&self, u: &[f32; NU]) -> (f32, f32, f32, f32) {
        let mut f_total = 0.0;
        let mut tau_x = 0.0;
        let mut tau_y = 0.0;
        let mut tau_z = 0.0;
        for i in 0..NU {
            f_total += u[i];
            tau_x += self.motor_pos[i][1] * u[i];
            tau_y += -self.motor_pos[i][0] * u[i];
            tau_z += self.motor_yaw_coeff[i] * u[i];
        }
        (f_total, tau_x, tau_y, tau_z)
    }

    /// Continuous-time ENU dynamics: xdot = f(x, u).
    ///
    /// Thrust acts along +body z (FLU), rotated to ENU via R(q).
    /// Gravity is subtracted: `a = R[:,2] * T/m − [0, 0, grav]`.
    pub fn dynamics(&self, x: &[f32; NX], u: &[f32; NU]) -> [f32; NX] {
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (vx, vy, vz) = (x[7], x[8], x[9]);
        let (wx, wy, wz) = (x[10], x[11], x[12]);
        let m_inv = self.mass_inv;
        let [ixx_inv, iyy_inv, izz_inv] = self.inertia_inv;
        let [ixx, iyy, izz] = self.inertia;

        let (f_total, tau_x, tau_y, tau_z) = self.alloc(u);
        let ct_m = f_total * m_inv;

        let mut xdot = [0.0f32; NX];
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
        x: &[f32; NX],
        u: &[f32; NU],
    ) -> ([f32; NX], [[f32; NX]; NX], [[f32; NU]; NX]) {
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (vx, vy, vz) = (x[7], x[8], x[9]);
        let (wx, wy, wz) = (x[10], x[11], x[12]);
        let m_inv = self.mass_inv;
        let [ixx_inv, iyy_inv, izz_inv] = self.inertia_inv;
        let [ixx, iyy, izz] = self.inertia;

        let (f_total, tau_x, tau_y, tau_z) = self.alloc(u);
        let ct_m = f_total * m_inv;

        let xdot = [
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
        ];

        // df/dx
        let mut jx = [[0.0f32; NX]; NX];
        // Rows 0-2: dp/dt = v
        jx[0][7] = 1.0;
        jx[1][8] = 1.0;
        jx[2][9] = 1.0;

        // Rows 3-6: quaternion kinematics
        let (hw_x, hw_y, hw_z) = (0.5 * wx, 0.5 * wy, 0.5 * wz);
        let (hq_w, hq_x, hq_y, hq_z) = (0.5 * qw, 0.5 * qx, 0.5 * qy, 0.5 * qz);

        jx[3][4] = hw_z;
        jx[3][5] = -hw_y;
        jx[3][6] = hw_x;
        jx[3][10] = hq_w;
        jx[3][11] = -hq_z;
        jx[3][12] = hq_y;

        jx[4][3] = -hw_z;
        jx[4][5] = hw_x;
        jx[4][6] = hw_y;
        jx[4][10] = hq_z;
        jx[4][11] = hq_w;
        jx[4][12] = -hq_x;

        jx[5][3] = hw_y;
        jx[5][4] = -hw_x;
        jx[5][6] = hw_z;
        jx[5][10] = -hq_y;
        jx[5][11] = hq_x;
        jx[5][12] = hq_w;

        jx[6][3] = -hw_x;
        jx[6][4] = -hw_y;
        jx[6][5] = -hw_z;
        jx[6][10] = -hq_x;
        jx[6][11] = -hq_y;
        jx[6][12] = -hq_z;

        // Rows 7-9: d(accel)/d(quaternion)  (thrust rotation Jacobian)
        let dc_qw = 2.0 * ct_m * qw;
        let dc_qx = 2.0 * ct_m * qx;
        let dc_qy = 2.0 * ct_m * qy;
        let dc_qz = 2.0 * ct_m * qz;

        jx[7][3] = dc_qz;
        jx[7][4] = dc_qw;
        jx[7][5] = dc_qx;
        jx[7][6] = dc_qy;
        jx[8][3] = -dc_qw;
        jx[8][4] = dc_qz;
        jx[8][5] = dc_qy;
        jx[8][6] = -dc_qx;
        jx[9][3] = -2.0 * dc_qx;
        jx[9][4] = -2.0 * dc_qy;

        // Rows 10-12: body-rate coupling
        jx[10][11] = -(izz - iyy) * wz * ixx_inv;
        jx[10][12] = -(izz - iyy) * wy * ixx_inv;
        jx[11][10] = -(ixx - izz) * wz * iyy_inv;
        jx[11][12] = -(ixx - izz) * wx * iyy_inv;
        jx[12][10] = -(iyy - ixx) * wy * izz_inv;
        jx[12][11] = -(iyy - ixx) * wx * izz_inv;

        // df/du
        let mut ju = [[0.0f32; NU]; NX];
        let a1_m = 2.0 * (qw * qy + qx * qz) * m_inv;
        let a2_m = 2.0 * (qy * qz - qw * qx) * m_inv;
        let a3_m = (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * m_inv;
        for j in 0..NU {
            ju[7][j] = a1_m;
            ju[8][j] = a2_m;
            ju[9][j] = a3_m;
        }
        // Torque allocation rows (per-motor, consistent with alloc())
        for j in 0..NU {
            ju[10][j] = self.motor_pos[j][1] * ixx_inv;
            ju[11][j] = -self.motor_pos[j][0] * iyy_inv;
            ju[12][j] = self.motor_yaw_coeff[j] * izz_inv;
        }

        (xdot, jx, ju)
    }

    /// RK4 integration with projection-method quaternion handling.
    ///
    /// Each RK4 intermediate state is projected back onto the unit sphere
    /// before being fed to the next `dynamics` evaluation, and the final
    /// result is also projected. This bounds intermediate-stage drift to
    /// one normalization's worth of error per stage instead of letting it
    /// compound across the four sub-steps.
    pub fn propagate_rk4(&self, xk: &[f32; NX], uk: &[f32; NU]) -> [f32; NX] {
        let dt = self.dt;
        let k0 = self.dynamics(xk, uk);

        let mut x1 = [0.0; NX];
        for i in 0..NX {
            x1[i] = xk[i] + k0[i] * (dt * 0.5);
        }
        normalize_quat(&mut x1);
        let k1 = self.dynamics(&x1, uk);

        for i in 0..NX {
            x1[i] = xk[i] + k1[i] * (dt * 0.5);
        }
        normalize_quat(&mut x1);
        let k2 = self.dynamics(&x1, uk);

        for i in 0..NX {
            x1[i] = xk[i] + k2[i] * dt;
        }
        normalize_quat(&mut x1);
        let k3 = self.dynamics(&x1, uk);

        let mut result = [0.0; NX];
        let s = dt / 6.0;
        for i in 0..NX {
            result[i] = xk[i] + (k0[i] + 2.0 * k1[i] + 2.0 * k2[i] + k3[i]) * s;
        }
        normalize_quat(&mut result);
        result
    }

    /// Forward Euler integration.
    pub fn propagate_euler(&self, xk: &[f32; NX], uk: &[f32; NU]) -> [f32; NX] {
        let xdot = self.dynamics(xk, uk);
        let mut result = [0.0; NX];
        for i in 0..NX {
            result[i] = xk[i] + self.dt * xdot[i];
        }
        normalize_quat(&mut result);
        result
    }

    /// Euler sensitivity: F_x = I + dt*df/dx,  F_u = dt*df/du.
    pub fn propagate_euler_grad(
        &self,
        xk: &[f32; NX],
        uk: &[f32; NU],
    ) -> ([[f32; NX]; NX], [[f32; NU]; NX]) {
        let (_, jac_x, jac_u) = self.dynamics_jac(xk, uk);
        let dt = self.dt;

        let mut fx = [[0.0f32; NX]; NX];
        for i in 0..NX {
            for j in 0..NX {
                fx[i][j] = jac_x[i][j] * dt;
            }
            fx[i][i] += 1.0; // I + dt*Jx
        }

        let mut fu = [[0.0f32; NU]; NX];
        for i in 0..NX {
            for j in 0..NU {
                fu[i][j] = jac_u[i][j] * dt;
            }
        }
        (fx, fu)
    }

    // ── Cost functions ──────────────────────────────────────────────────
    //
    // Position, velocity, quaternion-attitude, and box-constraint terms are
    // computed via `super::model_utils` so the math is identical to `QuadModel`.
    // Only the FullQuadModel-specific body-rate state cost (indices 10..13)
    // is inlined locally.

    /// Stage state cost + gradient. Returns cost, writes grad_x.
    pub fn state_cost_grad(&self, x: &[f32; NX], xref: &[f32; NX], grad_x: &mut [f32; NX]) -> f32 {
        let dt = self.dt;
        // Pos + vel cost/gradient via shared helper.
        let mut cost = model_utils::write_pos_vel_cost_grad(x, xref, &self.w_pos, &self.w_vel, dt, grad_x);
        // Body-rate cost/gradient — full-model only (rates are STATE here).
        for i in 0..3 {
            let er = x[10 + i] - xref[10 + i];
            cost += dt * er * er * self.w_rate[i];
            grad_x[10 + i] = 2.0 * er * self.w_rate[i] * dt;
        }
        // Quaternion cost/gradient via shared helpers.
        let (ea, de, dqa_dq) = model_utils::attitude_error(x, xref);
        cost += model_utils::write_quat_cost_grad(&ea, &de, &dqa_dq, &self.w_att, dt, grad_x);
        cost
    }

    /// Gauss-Newton Hessian + gradient in a single pass.
    pub fn state_cost_hess_grad(
        &self,
        x: &[f32; NX],
        xref: &[f32; NX],
        grad_x: &mut [f32; NX],
        hess_xx: &mut [[f32; NX]; NX],
    ) -> f32 {
        let dt = self.dt;

        // ── Pos + vel cost/gradient via shared helper ──
        let mut cost = model_utils::write_pos_vel_cost_grad(x, xref, &self.w_pos, &self.w_vel, dt, grad_x);

        // ── Body-rate cost/gradient (full-model only) ──
        for i in 0..3 {
            let er = x[10 + i] - xref[10 + i];
            cost += dt * er * er * self.w_rate[i];
            grad_x[10 + i] = 2.0 * er * self.w_rate[i] * dt;
        }

        // ── Quaternion cost/gradient via shared helpers ──
        let (ea, de, dqa_dq) = model_utils::attitude_error(x, xref);
        cost += model_utils::write_quat_cost_grad(&ea, &de, &dqa_dq, &self.w_att, dt, grad_x);

        // ── Hessian ──
        for row in hess_xx.iter_mut() {
            row.fill(0.0);
        }
        // Pos + vel diagonal block via shared helper.
        model_utils::write_pos_vel_hess(&self.w_pos, &self.w_vel, dt, hess_xx);
        // Body-rate diagonal block — full-model only.
        for i in 0..3 {
            hess_xx[10 + i][10 + i] = 2.0 * dt * self.w_rate[i];
        }
        // Quaternion Hessian block via shared helper.
        model_utils::write_quat_hess(&de, &dqa_dq, &self.w_att, dt, hess_xx);

        cost
    }

    /// Input cost + gradient.
    pub fn input_cost_grad(&self, u: &[f32; NU], uref: &[f32; NU], grad_u: &mut [f32; NU]) -> f32 {
        let dt = self.dt;
        let mut cost = 0.0;
        for i in 0..NU {
            let eu = u[i] - uref[i];
            cost += dt * eu * eu * self.w_thrust;
            grad_u[i] = 2.0 * eu * self.w_thrust * dt;
        }
        cost
    }

    /// Cubic box-constraint penalty.
    pub fn constraint_hess_grad(
        &self,
        u: &[f32; NU],
        grad_u: &mut [f32; NU],
        r_diag: &mut [f32; NU],
    ) -> f32 {
        model_utils::constraint_hess_grad(u, &self.u_bounds, self.rho, grad_u, r_diag)
    }

    /// Full stage cost = state cost + input cost + constraint penalty.
    pub fn stage_cost_hess_grad(
        &self,
        x: &[f32; NX],
        u: &[f32; NU],
        xref: &[f32; NX],
        uref: &[f32; NU],
        hess_xx: &mut [[f32; NX]; NX],
        r_diag: &mut [f32; NU],
        grad_x: &mut [f32; NX],
        grad_u: &mut [f32; NU],
    ) -> f32 {
        let dt = self.dt;
        let mut cost = self.state_cost_hess_grad(x, xref, grad_x, hess_xx);

        for i in 0..NU {
            let eu = u[i] - uref[i];
            cost += dt * eu * eu * self.w_thrust;
            grad_u[i] = 2.0 * eu * self.w_thrust * dt;
            r_diag[i] = 2.0 * dt * self.w_thrust;
        }

        cost += self.constraint_hess_grad(u, grad_u, r_diag);
        cost
    }

    /// Clamp control to bounds.
    pub fn clamp_control(&self, u: &[f32; NU]) -> [f32; NU] {
        model_utils::clamp_control(u, &self.u_bounds)
    }
}
