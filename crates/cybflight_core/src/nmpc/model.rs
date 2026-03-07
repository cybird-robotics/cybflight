// Quadrotor dynamics model for NMPC.
// Ported from NMPCModel.h (FSC Lab / cyblib).
//
// State  x[10]: [px, py, pz, qx, qy, qz, qw, vx, vy, vz]
// Control u[4]: [thrust_c, omega_x, omega_y, omega_z]

use nalgebra::{ComplexField as _, SMatrix, SVector};

pub type State = SVector<f32, 10>;
pub type Control = SVector<f32, 4>;
pub type StateJac = SMatrix<f32, 10, 10>;
pub type CtrlJac = SMatrix<f32, 10, 4>;

pub const NU: usize = 4;

pub struct QuadModel {
    pub mass: f32,
    pub grav: f32,
    pub dt: f32,

    // RK4 scratch — pre-allocated to keep stack pressure low
    df_x0: StateJac,
    df_x1: StateJac,
    df_x2: StateJac,
    df_x3: StateJac,
    df_u0: CtrlJac,
    df_u1: CtrlJac,
    df_u2: CtrlJac,
    df_u3: CtrlJac,
    gk0_x: StateJac,
    gk1_x: StateJac,
    gk2_x: StateJac,
    gk3_x: StateJac,
    gk0_u: CtrlJac,
    gk1_u: CtrlJac,
    gk2_u: CtrlJac,
    gk3_u: CtrlJac,
}

// Free function so propagate_rk4_grad can call it without a `&self` borrow
// conflicting with `&mut self.df_x*` borrows.
fn dynamics_jac_impl(
    x: &State,
    u: &Control,
    mass: f32,
    grav: f32,
    jac_x: &mut StateJac,
    jac_u: &mut CtrlJac,
) -> State {
    let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
    let (vx, vy, vz) = (x[7], x[8], x[9]);
    let (c, wx, wy, wz) = (u[0], u[1], u[2], u[3]);
    let m = mass;

    let a1 = 2.0 * (qw * qy + qx * qz) / m;
    let a2 = 2.0 * (qy * qz - qw * qx) / m;
    let a3 = (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) / m;

    let xdot = State::from([
        vx,
        vy,
        vz,
        0.5 * (wx * qw + wz * qy - wy * qz),
        0.5 * (wy * qw - wz * qx + wx * qz),
        0.5 * (wz * qw + wy * qx - wx * qy),
        0.5 * (-wx * qx - wy * qy - wz * qz),
        a1 * c,
        a2 * c,
        a3 * c - grav,
    ]);

    let hw_x = 0.5 * wx;
    let hw_y = 0.5 * wy;
    let hw_z = 0.5 * wz;
    let hq_w = 0.5 * qw;
    let hq_x = 0.5 * qx;
    let hq_y = 0.5 * qy;
    let hq_z = 0.5 * qz;
    let dc_qw = 2.0 * c * qw / m;
    let dc_qx = 2.0 * c * qx / m;
    let dc_qy = 2.0 * c * qy / m;
    let dc_qz = 2.0 * c * qz / m;

    *jac_x = StateJac::zeros();
    jac_x[(0, 7)] = 1.0;
    jac_x[(1, 8)] = 1.0;
    jac_x[(2, 9)] = 1.0;
    jac_x[(3, 4)] = hw_z;
    jac_x[(3, 5)] = -hw_y;
    jac_x[(3, 6)] = hw_x;
    jac_x[(4, 3)] = -hw_z;
    jac_x[(4, 5)] = hw_x;
    jac_x[(4, 6)] = hw_y;
    jac_x[(5, 3)] = hw_y;
    jac_x[(5, 4)] = -hw_x;
    jac_x[(5, 6)] = hw_z;
    jac_x[(6, 3)] = -hw_x;
    jac_x[(6, 4)] = -hw_y;
    jac_x[(6, 5)] = -hw_z;
    jac_x[(7, 3)] = dc_qz;
    jac_x[(7, 4)] = dc_qw;
    jac_x[(7, 5)] = dc_qx;
    jac_x[(7, 6)] = dc_qy;
    jac_x[(8, 3)] = -dc_qw;
    jac_x[(8, 4)] = dc_qz;
    jac_x[(8, 5)] = dc_qy;
    jac_x[(8, 6)] = -dc_qx;
    jac_x[(9, 3)] = -2.0 * dc_qx;
    jac_x[(9, 4)] = -2.0 * dc_qy;

    *jac_u = CtrlJac::zeros();
    jac_u[(3, 1)] = hq_w;
    jac_u[(3, 2)] = -hq_z;
    jac_u[(3, 3)] = hq_y;
    jac_u[(4, 1)] = hq_z;
    jac_u[(4, 2)] = hq_w;
    jac_u[(4, 3)] = -hq_x;
    jac_u[(5, 1)] = -hq_y;
    jac_u[(5, 2)] = hq_x;
    jac_u[(5, 3)] = hq_w;
    jac_u[(6, 1)] = -hq_x;
    jac_u[(6, 2)] = -hq_y;
    jac_u[(6, 3)] = -hq_z;
    jac_u[(7, 0)] = a1;
    jac_u[(8, 0)] = a2;
    jac_u[(9, 0)] = a3;

    xdot
}

// ── Sparse Jacobian helpers ────────────────────────────────────────────────────
//
// dynamics_jac_impl fills a 10×10 StateJac with ~20 nonzero entries.
// The dense nalgebra `*` operator would do 1000 MACs; these helpers do ~250.
//
// Nonzero pattern of jac_x (row → nonzero cols):
//   0 → [7],  1 → [8],  2 → [9]                     (velocity)
//   3 → [4,5,6], 4 → [3,5,6], 5 → [3,4,6], 6 → [3,4,5] (quaternion kinematics)
//   7 → [3,4,5,6], 8 → [3,4,5,6], 9 → [3,4]         (thrust acceleration)

/// Compute `j * b` (10×10 × 10×10) exploiting jac_x sparsity.
#[inline]
fn sparse_jx_mul(j: &StateJac, b: &StateJac) -> StateJac {
    let mut r = StateJac::zeros();
    for c in 0..10 { r[(0,c)] = j[(0,7)] * b[(7,c)]; }
    for c in 0..10 { r[(1,c)] = j[(1,8)] * b[(8,c)]; }
    for c in 0..10 { r[(2,c)] = j[(2,9)] * b[(9,c)]; }
    for c in 0..10 { r[(3,c)] = j[(3,4)]*b[(4,c)] + j[(3,5)]*b[(5,c)] + j[(3,6)]*b[(6,c)]; }
    for c in 0..10 { r[(4,c)] = j[(4,3)]*b[(3,c)] + j[(4,5)]*b[(5,c)] + j[(4,6)]*b[(6,c)]; }
    for c in 0..10 { r[(5,c)] = j[(5,3)]*b[(3,c)] + j[(5,4)]*b[(4,c)] + j[(5,6)]*b[(6,c)]; }
    for c in 0..10 { r[(6,c)] = j[(6,3)]*b[(3,c)] + j[(6,4)]*b[(4,c)] + j[(6,5)]*b[(5,c)]; }
    for c in 0..10 { r[(7,c)] = j[(7,3)]*b[(3,c)] + j[(7,4)]*b[(4,c)] + j[(7,5)]*b[(5,c)] + j[(7,6)]*b[(6,c)]; }
    for c in 0..10 { r[(8,c)] = j[(8,3)]*b[(3,c)] + j[(8,4)]*b[(4,c)] + j[(8,5)]*b[(5,c)] + j[(8,6)]*b[(6,c)]; }
    for c in 0..10 { r[(9,c)] = j[(9,3)]*b[(3,c)] + j[(9,4)]*b[(4,c)]; }
    r
}

/// Compute `j * b` (10×10 × 10×4) exploiting jac_x sparsity — for gk_u propagation.
#[inline]
fn sparse_jx_mul_u(j: &StateJac, b: &CtrlJac) -> CtrlJac {
    let mut r = CtrlJac::zeros();
    for c in 0..4 { r[(0,c)] = j[(0,7)] * b[(7,c)]; }
    for c in 0..4 { r[(1,c)] = j[(1,8)] * b[(8,c)]; }
    for c in 0..4 { r[(2,c)] = j[(2,9)] * b[(9,c)]; }
    for c in 0..4 { r[(3,c)] = j[(3,4)]*b[(4,c)] + j[(3,5)]*b[(5,c)] + j[(3,6)]*b[(6,c)]; }
    for c in 0..4 { r[(4,c)] = j[(4,3)]*b[(3,c)] + j[(4,5)]*b[(5,c)] + j[(4,6)]*b[(6,c)]; }
    for c in 0..4 { r[(5,c)] = j[(5,3)]*b[(3,c)] + j[(5,4)]*b[(4,c)] + j[(5,6)]*b[(6,c)]; }
    for c in 0..4 { r[(6,c)] = j[(6,3)]*b[(3,c)] + j[(6,4)]*b[(4,c)] + j[(6,5)]*b[(5,c)]; }
    for c in 0..4 { r[(7,c)] = j[(7,3)]*b[(3,c)] + j[(7,4)]*b[(4,c)] + j[(7,5)]*b[(5,c)] + j[(7,6)]*b[(6,c)]; }
    for c in 0..4 { r[(8,c)] = j[(8,3)]*b[(3,c)] + j[(8,4)]*b[(4,c)] + j[(8,5)]*b[(5,c)] + j[(8,6)]*b[(6,c)]; }
    for c in 0..4 { r[(9,c)] = j[(9,3)]*b[(3,c)] + j[(9,4)]*b[(4,c)]; }
    r
}

impl QuadModel {
    pub fn new(mass: f32, grav: f32, dt: f32) -> Self {
        Self {
            mass,
            grav,
            dt,
            df_x0: StateJac::zeros(),
            df_x1: StateJac::zeros(),
            df_x2: StateJac::zeros(),
            df_x3: StateJac::zeros(),
            df_u0: CtrlJac::zeros(),
            df_u1: CtrlJac::zeros(),
            df_u2: CtrlJac::zeros(),
            df_u3: CtrlJac::zeros(),
            gk0_x: StateJac::zeros(),
            gk1_x: StateJac::zeros(),
            gk2_x: StateJac::zeros(),
            gk3_x: StateJac::zeros(),
            gk0_u: CtrlJac::zeros(),
            gk1_u: CtrlJac::zeros(),
            gk2_u: CtrlJac::zeros(),
            gk3_u: CtrlJac::zeros(),
        }
    }

    // Continuous-time dynamics: ẋ = f(x, u)
    pub fn dynamics(&self, x: &State, u: &Control) -> State {
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (vx, vy, vz) = (x[7], x[8], x[9]);
        let (c, wx, wy, wz) = (u[0], u[1], u[2], u[3]);
        let m = self.mass;
        State::from([
            vx,
            vy,
            vz,
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy),
            0.5 * (-wx * qx - wy * qy - wz * qz),
            2.0 * (qw * qy + qx * qz) * c / m,
            2.0 * (qy * qz - qw * qx) * c / m,
            (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * c / m - self.grav,
        ])
    }

    // RK4 integration (no Jacobians).
    pub fn propagate_rk4(&self, xk: &State, uk: &Control) -> State {
        let dt = self.dt;
        let k0 = self.dynamics(xk, uk);
        let k1 = self.dynamics(&(xk + k0 * (dt * 0.5)), uk);
        let k2 = self.dynamics(&(xk + k1 * (dt * 0.5)), uk);
        let k3 = self.dynamics(&(xk + k2 * dt), uk);
        xk + (k0 + k1 * 2.0 + k2 * 2.0 + k3) * (dt / 6.0)
    }

    // RK4 + sensitivity equations. Matches NMPCModel.h:PropagateRK4Grad.
    pub fn propagate_rk4_grad(
        &mut self,
        xk: &State,
        uk: &Control,
        grad_fx: &mut StateJac,
        grad_fu: &mut CtrlJac,
    ) -> State {
        // Copy scalars out first so the borrow checker sees no aliasing with
        // the mutable borrows of self.df_x* / self.df_u* below.
        let mass = self.mass;
        let grav = self.grav;
        let dt = self.dt;
        let dth = dt * 0.5;
        let dt6 = dt / 6.0;
        let ii = StateJac::identity();

        // Forward: compute k0..k3 and their Jacobians
        let k0 = dynamics_jac_impl(xk, uk, mass, grav, &mut self.df_x0, &mut self.df_u0);
        let xk1h = xk + k0 * dth;
        let k1 = dynamics_jac_impl(&xk1h, uk, mass, grav, &mut self.df_x1, &mut self.df_u1);
        let xk2h = xk + k1 * dth;
        let k2 = dynamics_jac_impl(&xk2h, uk, mass, grav, &mut self.df_x2, &mut self.df_u2);
        let xk3 = xk + k2 * dt;
        let k3 = dynamics_jac_impl(&xk3, uk, mass, grav, &mut self.df_x3, &mut self.df_u3);

        let xout = xk + (k0 + k1 * 2.0 + k2 * 2.0 + k3) * dt6;

        // Sensitivity propagation (chain rule through RK4 stages).
        // Dense df_xN * mat is replaced by sparse_jx_mul which exploits the
        // ~20-nonzero structure of the dynamics Jacobian (~250 MACs vs 1000).
        self.gk0_x = self.df_x0;
        self.gk0_u = self.df_u0;
        let tmp1x = ii + self.gk0_x * dth;
        self.gk1_x = sparse_jx_mul(&self.df_x1, &tmp1x);
        self.gk1_u = self.df_u1 + sparse_jx_mul_u(&self.df_x1, &self.gk0_u) * dth;
        let tmp2x = ii + self.gk1_x * dth;
        self.gk2_x = sparse_jx_mul(&self.df_x2, &tmp2x);
        self.gk2_u = self.df_u2 + sparse_jx_mul_u(&self.df_x2, &self.gk1_u) * dth;
        let tmp3x = ii + self.gk2_x * dt;
        self.gk3_x = sparse_jx_mul(&self.df_x3, &tmp3x);
        self.gk3_u = self.df_u3 + sparse_jx_mul_u(&self.df_x3, &self.gk2_u) * dt;

        *grad_fx = ii + (self.gk0_x + self.gk1_x * 2.0 + self.gk2_x * 2.0 + self.gk3_x) * dt6;
        *grad_fu = (self.gk0_u + self.gk1_u * 2.0 + self.gk2_u * 2.0 + self.gk3_u) * dt6;

        xout
    }

    // State cost + gradient wrt x. Matches NMPCModel.h:StateCostGrad.
    pub fn state_cost_grad(&self, x: &State, xref: &State, grad_x: &mut State) -> f32 {
        let wp = [50.0_f32, 50.0, 100.0];
        let wa = [5.0_f32, 5.0, 200.0];
        let wv = [1.0_f32, 1.0, 1.0];
        let dt = self.dt;

        let ep = [x[0] - xref[0], x[1] - xref[1], x[2] - xref[2]];
        let ev = [x[7] - xref[7], x[8] - xref[8], x[9] - xref[9]];

        // q_aux = q ⊗ q_ref⁻¹
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (rx, ry, rz, rw) = (xref[3], xref[4], xref[5], xref[6]);
        let mut qa = [
            -qx * rw + qw * rx + qz * ry - qy * rz,
            -qy * rw - qz * rx + qw * ry + qx * rz,
            -qz * rw + qy * rx - qx * ry + qw * rz,
            qw * rw + qx * rx + qy * ry + qz * rz,
        ];
        let sign_flip = if qa[3] < 0.0 {
            qa[0] = -qa[0];
            qa[1] = -qa[1];
            qa[2] = -qa[2];
            qa[3] = -qa[3];
            -1.0_f32
        } else {
            1.0_f32
        };

        const EPS: f32 = 1e-3;
        let denom = (qa[3] * qa[3] + qa[2] * qa[2] + EPS).sqrt();
        let inv_d = 1.0 / denom;
        let nr = qa[3] * qa[0] - qa[1] * qa[2];
        let np = qa[3] * qa[1] + qa[0] * qa[2];
        let ny = qa[2];
        let ea = [2.0 * nr * inv_d, 2.0 * np * inv_d, 2.0 * ny * inv_d];

        let cost = dt * (ep[0] * ep[0] * wp[0] + ep[1] * ep[1] * wp[1] + ep[2] * ep[2] * wp[2])
            + dt * (ev[0] * ev[0] * wv[0] + ev[1] * ev[1] * wv[1] + ev[2] * ev[2] * wv[2])
            + dt * (ea[0] * ea[0] * wa[0] + ea[1] * ea[1] * wa[1] + ea[2] * ea[2] * wa[2]);

        // Gradient wrt position and velocity
        grad_x[0] = 2.0 * ep[0] * wp[0] * dt;
        grad_x[1] = 2.0 * ep[1] * wp[1] * dt;
        grad_x[2] = 2.0 * ep[2] * wp[2] * dt;
        grad_x[7] = 2.0 * ev[0] * wv[0] * dt;
        grad_x[8] = 2.0 * ev[1] * wv[1] * dt;
        grad_x[9] = 2.0 * ev[2] * wv[2] * dt;

        // Gradient wrt quaternion (chain rule)
        let inv_d2 = inv_d * inv_d;
        let dd2 = qa[2] * inv_d; // ∂denom/∂qa[2] * (1/denom)
        let dd3 = qa[3] * inv_d; // ∂denom/∂qa[3] * (1/denom)

        // de_att/dq_aux (3×4)
        let de = [
            [
                2.0 * qa[3] * inv_d,
                2.0 * (-qa[2]) * inv_d,
                2.0 * (-qa[1] * inv_d - nr * dd2 * inv_d2),
                2.0 * (qa[0] * inv_d - nr * dd3 * inv_d2),
            ],
            [
                2.0 * qa[2] * inv_d,
                2.0 * qa[3] * inv_d,
                2.0 * (qa[0] * inv_d - np * dd2 * inv_d2),
                2.0 * (qa[1] * inv_d - np * dd3 * inv_d2),
            ],
            [
                0.0,
                0.0,
                2.0 * (inv_d - ny * dd2 * inv_d2),
                2.0 * (-ny * dd3 * inv_d2),
            ],
        ];
        // dq_aux/dq (4×4), with sign_flip applied.
        // Row i = ∂q_aux[i]/∂[qx, qy, qz, qw]. Matches NMPCModel.h:dqaux_dq exactly.
        let dqadq = [
            [
                -rw * sign_flip,
                -rz * sign_flip,
                ry * sign_flip,
                rx * sign_flip,
            ], // row 0
            [
                rz * sign_flip,
                -rw * sign_flip,
                -rx * sign_flip,
                ry * sign_flip,
            ], // row 1
            [
                -ry * sign_flip,
                rx * sign_flip,
                -rw * sign_flip,
                rz * sign_flip,
            ], // row 2
            [
                rx * sign_flip,
                ry * sign_flip,
                rz * sign_flip,
                rw * sign_flip,
            ], // row 3
        ];
        let wea = [ea[0] * wa[0], ea[1] * wa[1], ea[2] * wa[2]];

        // grad_qaux = de^T * wea  (length 4)
        let mut gqa = [0.0_f32; 4];
        for col in 0..4 {
            for row in 0..3 {
                gqa[col] += de[row][col] * wea[row];
            }
        }

        // grad_q = 2*dt * dqadq^T * gqa
        for qi in 0..4 {
            let mut v = 0.0_f32;
            for j in 0..4 {
                v += dqadq[j][qi] * gqa[j];
            }
            grad_x[3 + qi] = 2.0 * dt * v;
        }

        cost
    }

    // Input cost + gradient. Matches NMPCModel.h:InputCostGrad.
    pub fn input_cost_grad(&self, u: &Control, uref: &Control, grad_u: &mut Control) -> f32 {
        let wu = [1.0_f32; 4];
        let dt = self.dt;
        let mut cost = 0.0;
        for i in 0..4 {
            let e = u[i] - uref[i];
            cost += e * e * wu[i] * dt;
            grad_u[i] = 2.0 * e * wu[i] * dt;
        }
        cost
    }

    // Combined path cost + gradients.
    pub fn path_cost_grad(
        &self,
        x: &State,
        u: &Control,
        xref: &State,
        uref: &Control,
        grad_x: &mut State,
        grad_u: &mut Control,
    ) -> f32 {
        self.state_cost_grad(x, xref, grad_x) + self.input_cost_grad(u, uref, grad_u)
    }

    // Terminal cost (same formula as state cost, no separate scaling).
    pub fn terminal_cost_grad(&self, x: &State, xref: &State, grad_x: &mut State) -> f32 {
        self.state_cost_grad(x, xref, grad_x)
    }

    // Cubic box constraint penalty for one scalar value.
    fn box_constraint(val: f32, lb: f32, ub: f32, grad: &mut f32) -> f32 {
        const RHO: f32 = 1e4;
        let lpen = lb - val;
        if lpen > 0.0 {
            let lpen2 = lpen * lpen;
            *grad = -RHO * 3.0 * lpen2;
            return RHO * lpen2 * lpen;
        }
        let upen = val - ub;
        if upen > 0.0 {
            let upen2 = upen * upen;
            *grad = RHO * 3.0 * upen2;
            return RHO * upen2 * upen;
        }
        *grad = 0.0;
        0.0
    }

    // Control box constraints + gradients. Matches NMPCModel.h:AddGeneralConstraintGrad.
    // Derived from NMPCExample.cpp: mass=1.0 kg, grav=9.8 m/s², throttle_mult=3.0
    //   thrust ∈ [mass*g*0.1, mass*g*3.0] = [0.98, 29.4] N
    //   roll/pitch ∈ ±600 deg/s = ±10.47 rad/s,  yaw ∈ ±500 deg/s = ±8.73 rad/s
    pub fn constraint_grad(&self, u: &Control, grad_x: &mut State, grad_u: &mut Control) -> f32 {
        *grad_x = State::zeros();
        *grad_u = Control::zeros();
        let bounds = [
            (0.98_f32, 29.4_f32),
            (-10.47, 10.47),
            (-10.47, 10.47),
            (-8.73, 8.73),
        ];
        let mut penalty = 0.0;
        for i in 0..4 {
            let mut g = 0.0;
            penalty += Self::box_constraint(u[i], bounds[i].0, bounds[i].1, &mut g);
            grad_u[i] += g;
        }
        penalty
    }
}
