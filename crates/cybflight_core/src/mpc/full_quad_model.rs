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
use crate::rotation::hat;

use nalgebra::{vector, Matrix3, SMatrix, SVector, Vector3};

pub const NX: usize = 13;
pub const NU: usize = 4;
pub const N: usize = 20;

/// Build `(I, I⁻¹)` from the row-major 3×3 airframe inertia array.
///
/// The full tensor is symmetrized (`½(I + Iᵀ)`) and its diagonal floored at
/// `1e-6` — the same divide-by-zero belt-and-braces the diagonal-only code
/// applied (identity params are validated at the YAML bake and on flash
/// replay, but the shell's shared array metadata cannot bound diagonal and
/// off-diagonal terms separately). The result must be positive-definite
/// (Sylvester's criterion) with a finite inverse; any failure — non-finite
/// off-diagonals, indefinite tensor, singular inverse — falls back to the
/// floored **diagonal**, discarding the off-diagonal terms.
///
/// Returns `(inertia, inertia_inv, fell_back_to_diagonal)`. Callers that
/// care about the fallback (it means the MPC model disagrees with INDI's
/// full-matrix G1) should surface the flag; see
/// `outer_loop::build_outer_quad_model`.
pub fn inertia_from_array(arr: &[f32; 9]) -> (Matrix3<f32>, Matrix3<f32>, bool) {
    let floor = |v: f32| if v.is_finite() && v > 1e-6 { v } else { 1e-6 };
    let (ixx, iyy, izz) = (floor(arr[0]), floor(arr[4]), floor(arr[8]));
    let diag = Matrix3::from_diagonal(&Vector3::new(ixx, iyy, izz));
    let diag_inv = Matrix3::from_diagonal(&Vector3::new(1.0 / ixx, 1.0 / iyy, 1.0 / izz));

    // Symmetrized off-diagonals (products of inertia).
    let ixy = 0.5 * (arr[1] + arr[3]);
    let ixz = 0.5 * (arr[2] + arr[6]);
    let iyz = 0.5 * (arr[5] + arr[7]);
    if !(ixy.is_finite() && ixz.is_finite() && iyz.is_finite()) {
        return (diag, diag_inv, true);
    }
    if ixy == 0.0 && ixz == 0.0 && iyz == 0.0 {
        // Pure diagonal — skip the SPD checks and keep the element-wise
        // inverse (exact, no cancellation from a general 3×3 inversion).
        return (diag, diag_inv, false);
    }
    #[rustfmt::skip]
    let full = Matrix3::new(
        ixx, ixy, ixz,
        ixy, iyy, iyz,
        ixz, iyz, izz,
    );
    // Positive-definiteness via Sylvester's criterion (leading principal
    // minors; the 1×1 minor ixx > 0 is guaranteed by the floor).
    let minor2 = ixx * iyy - ixy * ixy;
    let det = full.determinant();
    if !(minor2 > 0.0 && det > 0.0 && det.is_finite()) {
        return (diag, diag_inv, true);
    }
    match full.try_inverse() {
        Some(inv) if inv.iter().all(|v| v.is_finite()) => (full, inv, false),
        _ => (diag, diag_inv, true),
    }
}

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
    /// Full 3×3 body inertia tensor \[kg·m²\] (FLU frame, symmetric
    /// positive-definite — see [`inertia_from_array`]).
    pub inertia: Matrix3<f32>,
    /// Per-motor position \[x, y\] in the body FLU frame (metres).
    pub motor_pos: [[f32; 2]; NU],
    /// Per-motor signed yaw coefficient: `spin_sign * torque_coeff`.
    /// Positive for CW (from above), negative for CCW.
    pub motor_yaw_coeff: SVector<f32, NU>,
    pub u_bounds: [[f32; 2]; NU],
    /// Precomputed 1.0 / mass.
    pub mass_inv: f32,
    /// Precomputed inverse of `inertia`.
    pub inertia_inv: Matrix3<f32>,
    // ── Stage cost weights (mirrored from VehicleParams.mpc) ───────────
    pub w_pos: [f32; 3],
    pub w_vel: [f32; 3],
    pub w_att: [f32; 3],
    pub w_rate: [f32; 3],
    /// Control effort weight (uniform across motors).
    pub w_thrust: f32,
    /// Cubic constraint penalty weight (input bound enforcement).
    pub rho: f32,
    // ── Body-rate STATE constraints (relaxed log-barrier) ──────────────
    //
    // Unlike `QuadModel`, whose body rates are *inputs* bounded by
    // `u_bounds`, the rates here are states 10..13 — enforcing the
    // vehicle's rate limits requires state constraints. Implemented per
    // Frey et al., arXiv:2505.01353v2 (App. A.2): the constraints enter the
    // stage cost as relaxed log-barrier terms whose gradient/Hessian flow
    // into the Riccati sweep via `q`/`qm`. See
    // `model_utils::write_rate_barrier_cost_grad`.
    /// Per-axis `[lower, upper]` bounds on body rates (states 10..13) [rad/s].
    pub rate_bounds: [[f32; 2]; 3],
    /// Barrier weight τ. **`0.0` disables the state constraints entirely**
    /// (byte-identical to the pre-barrier solver); τ > 0 enables them, and
    /// smaller τ approximates the hard constraint more closely (paper
    /// Thm. 3) at the price of a stiffer subproblem.
    pub rate_barrier_tau: f32,
    /// Relaxation margin δ [rad/s]: below this constraint margin the log
    /// branch switches to its quadratic extension, keeping the barrier
    /// defined (and strongly repulsive) for infeasible iterates.
    pub rate_barrier_delta: f32,
    // ── Maximum-tilt STATE constraint (relaxed log-barrier, cos space) ──
    //
    // Envelope fence keeping the *predicted* trajectory away from deep
    // tilt, where the local SQP has a free-fall stationary point (all
    // motors cornered at zero for tilt ≳ 100°). A fence on what the
    // controller chooses — it cannot recover states that disturbances
    // push beyond it. See `model_utils::write_tilt_barrier_cost_grad`.
    /// cos(θ_max): the constraint is `1 − 2(qx²+qy²) ≥ cos_max_tilt`.
    pub tilt_cos_max: f32,
    /// Tilt barrier weight τ. **`0.0` disables the constraint entirely**
    /// (byte-identical to the pre-barrier solver).
    pub tilt_barrier_tau: f32,
    /// Relaxation margin δ in **cos units** (the constraint lives in cos
    /// space; 0.05 ≈ 3.3° of margin at a 60° limit).
    pub tilt_barrier_delta: f32,
}

impl Default for FullQuadModel {
    /// Default matches the Betaflight QuadX layout in `vehicle.rs` and the
    /// `MpcParams::default()` cost weights in `params.rs`.
    fn default() -> Self {
        Self {
            mass: 0.55,
            grav: 9.81,
            dt: 0.05,
            inertia: Matrix3::from_diagonal(&Vector3::new(0.0025, 0.0021, 0.0043)),
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
            inertia_inv: Matrix3::from_diagonal(&Vector3::new(
                1.0 / 0.0025,
                1.0 / 0.0021,
                1.0 / 0.0043,
            )),
            w_pos: [200.0, 200.0, 200.0],
            w_vel: [1.0, 1.0, 1.0],
            w_att: [5.0, 5.0, 200.0],
            w_rate: [1.0, 1.0, 1.0],
            w_thrust: 6.0,
            rho: 1e4,
            // Matches the reduced model's default body-rate input bounds.
            // τ = 0 → state constraints OFF by default (legacy behavior).
            rate_bounds: [[-10.0, 10.0], [-10.0, 10.0], [-6.0, 6.0]],
            rate_barrier_tau: 0.0,
            rate_barrier_delta: 0.1,
            // τ = 0 → tilt constraint OFF by default (legacy behavior).
            tilt_cos_max: 0.5, // 60°
            tilt_barrier_tau: 0.0,
            tilt_barrier_delta: 0.05,
        }
    }
}

impl FullQuadModel {
    /// Construct from firmware vehicle parameters.
    ///
    /// Extracts mass, the full inertia tensor, motor geometry, and thrust
    /// bounds from the canonical `VehicleParams`, ensuring consistency with
    /// the inner-loop controller and mixer (INDI's G1 inverts the same full
    /// 3×3 — `IndiEffectiveness::new`). Cost weights, integration timestep,
    /// and constraint penalty are sourced from `vp.mpc` so a single
    /// `crate::params::set` call retunes the entire MPC.
    pub fn from_vehicle_params(vp: &crate::params::FirmwareConfig) -> Self {
        let mass = vp.airframe.body.mass_kg;
        // Validation + divide-by-zero belt-and-braces live in
        // `inertia_from_array`; a non-SPD tensor silently degrades to its
        // diagonal here — `outer_loop::build_outer_quad_model` surfaces
        // that case with a boot warning.
        let (inertia, inertia_inv, _fell_back) =
            inertia_from_array(&vp.airframe.body.inertia_kg_m2);
        let mr = vp.airframe.body.max_rate_rad_s;
        let mut motor_pos = [[0.0f32; 2]; NU];
        let mut motor_yaw_coeff = SVector::<f32, NU>::zeros();
        let mut u_bounds = [[0.0f32; 2]; NU];
        for i in 0..NU {
            motor_pos[i] = vp.airframe.motors[i].position_m;
            motor_yaw_coeff[i] =
                (vp.airframe.motors[i].spin_dir as i32 as f32) * vp.airframe.motors[i].torque_coeff_m;
            u_bounds[i] = [0.0, vp.airframe.motors[i].max_thrust_n];
        }
        Self {
            mass,
            grav: vp.site.gravity_m_s2,
            dt: vp.mpc.dt,
            inertia,
            motor_pos,
            motor_yaw_coeff,
            u_bounds,
            mass_inv: 1.0 / mass,
            inertia_inv,
            w_pos: vp.mpc.pos_weight,
            w_vel: vp.mpc.vel_weight,
            w_att: vp.mpc.att_weight,
            w_rate: vp.mpc.rate_weight,
            w_thrust: vp.mpc.thrust_weight,
            rho: vp.mpc.rho,
            // Same per-axis rate limits the reduced model applies as input
            // bounds. τ/δ come from the param plane (`mpc_rate_barrier_tau`,
            // `mpc_rate_barrier_delta`); the schema default τ = 0 keeps the
            // constraints OFF unless the vehicle YAML (or a `param set`)
            // opts in.
            rate_bounds: [[-mr[0], mr[0]], [-mr[1], mr[1]], [-mr[2], mr[2]]],
            rate_barrier_tau: vp.mpc.rate_barrier_tau,
            rate_barrier_delta: vp.mpc.rate_barrier_delta,
            // Tilt fence from the param plane (`mpc_tilt_max_deg`,
            // `mpc_tilt_barrier_tau/delta`); schema default τ = 0 keeps it
            // OFF unless the vehicle YAML (or a `param set`) opts in.
            tilt_cos_max: libm::cosf(vp.mpc.tilt_max_deg.to_radians()),
            tilt_barrier_tau: vp.mpc.tilt_barrier_tau,
            tilt_barrier_delta: vp.mpc.tilt_barrier_delta,
        }
    }

    pub fn new(mass: f32, grav: f32, dt: f32) -> Self {
        Self {
            mass,
            grav,
            dt,
            mass_inv: 1.0 / mass,
            ..Default::default()
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
        // Euler rigid-body equations: α = I⁻¹·(τ − ω×Iω), full tensor.
        let w = Vector3::new(wx, wy, wz);
        let tau = Vector3::new(tau_x, tau_y, tau_z);
        let alpha = self.inertia_inv * (tau - w.cross(&(self.inertia * w)));
        xdot[10] = alpha.x;
        xdot[11] = alpha.y;
        xdot[12] = alpha.z;
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

        let (f_total, tau_x, tau_y, tau_z) = self.alloc(u);
        let ct_m = f_total * m_inv;

        // Euler rigid-body equations: α = I⁻¹·(τ − ω×Iω), full tensor.
        let w = Vector3::new(wx, wy, wz);
        let iw = self.inertia * w;
        let tau = Vector3::new(tau_x, tau_y, tau_z);
        let alpha = self.inertia_inv * (tau - w.cross(&iw));

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
            (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * ct_m - self.grav,
            alpha.x,
            alpha.y,
            alpha.z
        ];

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
        // ∂α/∂ω = −I⁻¹·(∂[ω×Iω]/∂ω) = −I⁻¹·([ω]ₓ·I − [Iω]ₓ)
        let dalpha_dw = -(self.inertia_inv * (hat(&w) * self.inertia - hat(&iw)));
        jx.fixed_view_mut::<3, 3>(10, 10).copy_from(&dalpha_dw);

        // df/du
        let mut ju = SMatrix::<f32, NX, NU>::zeros();
        let a1_m = 2.0 * (qw * qy + qx * qz) * m_inv;
        let a2_m = 2.0 * (qy * qz - qw * qx) * m_inv;
        let a3_m = (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * m_inv;
        // Translation thrust rows — constant across motors (collective thrust only).
        ju.row_mut(7).fill(a1_m);
        ju.row_mut(8).fill(a2_m);
        ju.row_mut(9).fill(a3_m);
        // Torque allocation rows: ∂α/∂u = I⁻¹·G_τ, with G_τ the per-motor
        // torque allocation (rows = [pos_y; −pos_x; yaw_coeff], consistent
        // with alloc()).
        let pos_x = SVector::<f32, NU>::from_fn(|i, _| self.motor_pos[i][0]);
        let pos_y = SVector::<f32, NU>::from_fn(|i, _| self.motor_pos[i][1]);
        let mut g_tau = SMatrix::<f32, 3, NU>::zeros();
        g_tau.row_mut(0).copy_from(&pos_y.transpose());
        g_tau.row_mut(1).copy_from(&(-pos_x).transpose());
        g_tau.row_mut(2).copy_from(&self.motor_yaw_coeff.transpose());
        ju.fixed_view_mut::<3, NU>(10, 0)
            .copy_from(&(self.inertia_inv * g_tau));

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
        // Body-rate state constraints (relaxed log-barrier). τ = 0 → off.
        if self.rate_barrier_tau > 0.0 {
            cost += model_utils::write_rate_barrier_cost_grad(
                x,
                &self.rate_bounds,
                self.rate_barrier_tau,
                self.rate_barrier_delta,
                grad_x,
            );
        }
        // Maximum-tilt state constraint (relaxed log-barrier). τ = 0 → off.
        if self.tilt_barrier_tau > 0.0 {
            cost += model_utils::write_tilt_barrier_cost_grad(
                x,
                self.tilt_cos_max,
                self.tilt_barrier_tau,
                self.tilt_barrier_delta,
                grad_x,
            );
        }
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

        // ── Body-rate state constraints (relaxed log-barrier, τ = 0 → off) ──
        if self.rate_barrier_tau > 0.0 {
            cost += model_utils::write_rate_barrier_cost_grad(
                x,
                &self.rate_bounds,
                self.rate_barrier_tau,
                self.rate_barrier_delta,
                grad_x,
            );
            model_utils::write_rate_barrier_hess(
                x,
                &self.rate_bounds,
                self.rate_barrier_tau,
                self.rate_barrier_delta,
                hess_xx,
            );
        }

        // ── Maximum-tilt state constraint (relaxed log-barrier, τ = 0 → off) ──
        if self.tilt_barrier_tau > 0.0 {
            cost += model_utils::write_tilt_barrier_cost_grad(
                x,
                self.tilt_cos_max,
                self.tilt_barrier_tau,
                self.tilt_barrier_delta,
                grad_x,
            );
            model_utils::write_tilt_barrier_hess(
                x,
                self.tilt_cos_max,
                self.tilt_barrier_tau,
                self.tilt_barrier_delta,
                hess_xx,
            );
        }

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

    /// Reduce the NMPC's first control to the inner-loop setpoint pair
    /// `(T_d, α_d)` — desired collective thrust [N] and desired body
    /// angular acceleration [rad/s²].
    ///
    /// Implements eq. (32) of Sun et al. (T-RO 2022,
    /// arXiv:2109.01365v6): `[T_d; I·α_d + ω×Iω] = G1·u`, i.e.
    ///
    /// `T_d = Σ uᵢ`,  `α_d = I⁻¹·(τ(u) − ω×Iω)`
    ///
    /// with the gyroscopic term evaluated at the caller-supplied body
    /// rate `ω`. By construction `α_d` equals `dynamics(x, u)[10..13]`
    /// when `ω` matches the state's rate entries.
    ///
    /// **Freshness caveat.** The paper evaluates this reduction *inside
    /// the inner loop* (Fig. 3), so its `ω×Iω` tracks the current gyro.
    /// If a caller computes `α_d` once per NMPC solve and holds it, the
    /// gyroscopic correction is frozen at the solve-time ω and the
    /// realized torque drifts by `(ω×Iω)|now − (ω×Iω)|solve` between
    /// solves. The firmware therefore does NOT hold this function's
    /// output: `outer_loop` ships the raw allocation torque `τ(u)`
    /// (`alloc`) and `indi_task` performs the `− ω×Iω` subtraction at
    /// IMU rate with the fresh gyro (as does the sim's
    /// `MpcFullIndiController`). This helper remains the single-call
    /// form of the same reduction for tests and one-shot consumers.
    pub fn inner_setpoint(
        &self,
        u: &SVector<f32, NU>,
        omega_rad_s: &Vector3<f32>,
    ) -> (f32, Vector3<f32>) {
        let (f_total, tau_x, tau_y, tau_z) = self.alloc(u);
        // Euler rigid-body equations, matching `dynamics` rows 10..13.
        let tau = Vector3::new(tau_x, tau_y, tau_z);
        let alpha =
            self.inertia_inv * (tau - omega_rad_s.cross(&(self.inertia * omega_rad_s)));
        (f_total, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// `inner_setpoint` must agree exactly with the rigid-body rows of
    /// `dynamics` when evaluated at the state's own rates — they encode
    /// the same Euler equations (paper eq. 32 is the allocation identity).
    #[test]
    fn inner_setpoint_matches_dynamics_rate_rows() {
        let model = FullQuadModel::default();
        let u = SVector::<f32, NU>::from_row_slice(&[1.2, 2.4, 0.8, 3.1]);
        let omega = Vector3::new(2.0, -3.5, 1.25);
        let mut x = SVector::<f32, NX>::zeros();
        // Non-identity attitude to prove attitude does not enter α_d.
        x[3] = 0.3;
        x[6] = libm::sqrtf(1.0 - 0.3 * 0.3);
        x[10] = omega.x;
        x[11] = omega.y;
        x[12] = omega.z;

        let (t_d, alpha_d) = model.inner_setpoint(&u, &omega);
        let xdot = model.dynamics(&x, &u);

        assert_eq!(t_d, u.sum());
        for i in 0..3 {
            assert!(
                (alpha_d[i] - xdot[10 + i]).abs() < 1e-6,
                "axis {i}: inner_setpoint α={} vs dynamics={}",
                alpha_d[i],
                xdot[10 + i]
            );
        }
    }

    /// A physically plausible tensor with non-zero products of inertia
    /// (SPD: diag dominant, det > 0).
    const COUPLED_INERTIA: [f32; 9] = [
        0.0025, 0.0004, -0.0002, //
        0.0004, 0.0021, 0.0003, //
        -0.0002, 0.0003, 0.0043,
    ];

    fn coupled_model() -> FullQuadModel {
        let (inertia, inertia_inv, fell_back) = inertia_from_array(&COUPLED_INERTIA);
        assert!(!fell_back, "COUPLED_INERTIA must be accepted as SPD");
        FullQuadModel {
            inertia,
            inertia_inv,
            ..Default::default()
        }
    }

    /// With a full (coupled) tensor, the analytic `dynamics_jac` must match
    /// central finite differences of `dynamics` — this exercises the
    /// ∂α/∂ω = −I⁻¹([ω]ₓI − [Iω]ₓ) block and the I⁻¹·G_τ input rows, which
    /// have no off-diagonal contribution in the diagonal-only FD test
    /// (`mpc_runtime_comparison.rs`).
    #[test]
    fn coupled_inertia_jacobian_matches_finite_differences() {
        let model = coupled_model();
        let u = SVector::<f32, NU>::from_row_slice(&[1.2, 2.4, 0.8, 3.1]);
        let mut x = SVector::<f32, NX>::zeros();
        x[3] = 0.3;
        x[6] = libm::sqrtf(1.0 - 0.3 * 0.3);
        x[10] = 2.0;
        x[11] = -3.5;
        x[12] = 1.25;

        let (_, jx, ju) = model.dynamics_jac(&x, &u);
        let eps = 1e-3;
        // f32 central differences on rate rows (|α| up to ~170 rad/s²)
        // carry ~1e-2 rounding error at this eps — use a relative
        // tolerance; structural errors are O(1) relative.
        let tol = |analytic: f32| 0.05_f32.max(0.02 * analytic.abs());
        // Rate rows only — the rest of the Jacobian is inertia-free and
        // covered by the existing full-matrix FD test.
        for col in 0..NX {
            let mut xp = x;
            let mut xm = x;
            xp[col] += eps;
            xm[col] -= eps;
            let fd = (model.dynamics(&xp, &u) - model.dynamics(&xm, &u)) / (2.0 * eps);
            for row in 10..NX {
                assert!(
                    (jx[(row, col)] - fd[row]).abs() < tol(jx[(row, col)]),
                    "jx[({row},{col})]: analytic {} vs FD {}",
                    jx[(row, col)],
                    fd[row]
                );
            }
        }
        for col in 0..NU {
            let mut up = u;
            let mut um = u;
            up[col] += eps;
            um[col] -= eps;
            let fd = (model.dynamics(&x, &up) - model.dynamics(&x, &um)) / (2.0 * eps);
            for row in 10..NX {
                assert!(
                    (ju[(row, col)] - fd[row]).abs() < tol(ju[(row, col)]),
                    "ju[({row},{col})]: analytic {} vs FD {}",
                    ju[(row, col)],
                    fd[row]
                );
            }
        }
    }

    /// `inner_setpoint` ≡ `dynamics` rate rows must also hold for a
    /// coupled tensor (both now go through the same matrix expression).
    #[test]
    fn coupled_inertia_inner_setpoint_matches_dynamics() {
        let model = coupled_model();
        let u = SVector::<f32, NU>::from_row_slice(&[2.0, 1.0, 3.0, 0.5]);
        let omega = Vector3::new(-1.5, 2.5, 0.75);
        let mut x = SVector::<f32, NX>::zeros();
        x[6] = 1.0;
        x[10] = omega.x;
        x[11] = omega.y;
        x[12] = omega.z;

        let (_, alpha_d) = model.inner_setpoint(&u, &omega);
        let xdot = model.dynamics(&x, &u);
        for i in 0..3 {
            assert!((alpha_d[i] - xdot[10 + i]).abs() < 1e-6);
        }
    }

    /// Zero off-diagonals must reproduce the element-wise diagonal inverse
    /// exactly (no fallback, no general-inverse rounding).
    #[test]
    fn inertia_from_array_diagonal_is_exact() {
        let arr = [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043];
        let (i, i_inv, fell_back) = inertia_from_array(&arr);
        assert!(!fell_back);
        assert_eq!(i, Matrix3::from_diagonal(&Vector3::new(0.0025, 0.0021, 0.0043)));
        assert_eq!(i_inv[(0, 0)], 1.0 / 0.0025);
        assert_eq!(i_inv[(1, 1)], 1.0 / 0.0021);
        assert_eq!(i_inv[(2, 2)], 1.0 / 0.0043);
        assert_eq!(i_inv[(0, 1)], 0.0);
    }

    /// A coupled tensor round-trips: I·I⁻¹ ≈ identity, and the result is
    /// the symmetrized input.
    #[test]
    fn inertia_from_array_coupled_roundtrip() {
        let (i, i_inv, fell_back) = inertia_from_array(&COUPLED_INERTIA);
        assert!(!fell_back);
        assert_eq!(i[(0, 1)], 0.0004);
        assert_eq!(i[(1, 0)], 0.0004);
        let eye = i * i_inv;
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((eye[(r, c)] - want).abs() < 1e-4);
            }
        }
    }

    /// Indefinite (non-SPD) input must fall back to the floored diagonal.
    #[test]
    fn inertia_from_array_rejects_non_spd() {
        // |Ixy| > sqrt(Ixx·Iyy) → 2nd leading principal minor < 0.
        let arr = [0.0025, 0.005, 0.0, 0.005, 0.0021, 0.0, 0.0, 0.0, 0.0043];
        let (i, i_inv, fell_back) = inertia_from_array(&arr);
        assert!(fell_back);
        assert_eq!(i, Matrix3::from_diagonal(&Vector3::new(0.0025, 0.0021, 0.0043)));
        assert_eq!(i_inv[(1, 1)], 1.0 / 0.0021);
        // Non-finite off-diagonal → same fallback.
        let arr = [0.0025, f32::NAN, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043];
        let (_, _, fell_back) = inertia_from_array(&arr);
        assert!(fell_back);
        // Degenerate diagonal is floored, as before.
        let arr = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, f32::NAN];
        let (i, _, fell_back) = inertia_from_array(&arr);
        assert!(!fell_back);
        assert_eq!(i, Matrix3::from_diagonal(&Vector3::new(1e-6, 1e-6, 1e-6)));
    }
}
