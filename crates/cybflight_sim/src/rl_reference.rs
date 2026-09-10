//! ENU reproduction of the upstream RL training environment's plant.
//!
//! **This is a validation fixture, not cybflight physics.** Its only job is
//! to reproduce `optimal_quad_control_RL/quad_race_env.py` closely enough
//! that a policy trained there behaves here as it did in the trainer. If a
//! ported policy flies this plant but not [`crate::plant::QuadPlant`], the
//! gap is modelling, not the port — which is the whole reason it exists.
//!
//! # Differences from [`crate::plant::QuadPlant`], deliberately preserved
//!
//! * **Moments are angular accelerations, not torques.** The upstream system
//!   identification regressed `ṗ, q̇, ṙ` directly against `ωᵢ²`, so the
//!   identified coefficients already absorb `1/I` *and* whatever gyroscopic
//!   and `ω × Iω` coupling was present in the flight data. There is
//!   therefore no inertia tensor here and no rigid-body cross-coupling term:
//!   adding one would double-count what the fit already contains.
//! * **Yaw is linear in ω, not quadratic.** Also a fitting choice upstream
//!   (`analyze.py` regresses `ṙ` on `±ωᵢ` and `±ω̇ᵢ`). Real prop drag torque
//!   goes as `ω²`; over the identified speed range the linear fit is close,
//!   but the two diverge by ~3.3× at full throttle.
//! * **Forward Euler**, matching the trainer's `x + dt·f(x, u)`.
//!
//! # Frames
//!
//! State is ENU world / FLU body throughout, with attitude carried as a
//! quaternion rather than the upstream ZYX Euler triple. The upstream ODE is
//! written in NED/FRD; the sign flips that follow from re-expressing it here
//! are marked at each term. Roll is unchanged (shared x axis), pitch and yaw
//! negate (y and z flip).

use cybflight_core::acmpc::CtbrCommand;
use cybflight_core::nn::race_policy::{Gate, VehicleState, NUM_MOTORS};
use cybflight_core::rotation::euler_angles_rpy_to_quaternion;
use nalgebra::{Matrix3, Matrix4, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4};

/// The upstream 8-gate figure-of-eight, **in ENU**.
///
/// Upstream defines this track in NED (`train.py`, `r = 1.5`). These are the
/// same eight gates re-expressed in cybflight's native frame:
/// `(E,N,U) = (y_ned, x_ned, −z_ned)` and `ψ_enu = π/2 − ψ_ned`. The
/// conversion is pinned by `enu_track_matches_upstream_ned_table`, so the
/// literals below cannot drift from their source.
pub const RL_TRACK_ENU: [([f32; 3], f32); 8] = [
    ([-1.5, 1.5, 1.5], 0.0),
    ([0.0, 0.0, 1.5], -core::f32::consts::FRAC_PI_2),
    ([1.5, -1.5, 1.5], 0.0),
    ([3.0, 0.0, 1.5], core::f32::consts::FRAC_PI_2),
    ([1.5, 1.5, 1.5], core::f32::consts::PI),
    ([0.0, 0.0, 1.5], -core::f32::consts::FRAC_PI_2),
    ([-1.5, -1.5, 1.5], core::f32::consts::PI),
    ([-3.0, 0.0, 1.5], core::f32::consts::FRAC_PI_2),
];

/// Upstream start pose, ENU: one metre behind gate 0, facing it.
pub const RL_START_POS_ENU: [f32; 3] = [-2.5, 1.5, 1.5];
pub const RL_START_YAW_ENU: f32 = 0.0;

/// Full gate aperture [m] (`gate_size` upstream).
pub const RL_GATE_SIZE_M: f32 = 1.5;

/// Control period the policy was trained at [s].
pub const RL_DT_S: f32 = 0.01;

/// Rotor speed the upstream env normalizes its motor state against
/// [rad/s]. **Not** the airframe's `ω_max` (3295.5): it is the scale of
/// the `[-1, 1]` motor-state observation, and the allocation targets it as
/// a ceiling because a command above it is unrepresentable in that state.
pub const RL_OMEGA_NORM_MAX: f32 = 3000.0;

/// Build the track as [`Gate`]s.
pub fn rl_track() -> [Gate; 8] {
    core::array::from_fn(|i| {
        let (p, yaw) = RL_TRACK_ENU[i];
        Gate::new(Vector3::new(p[0], p[1], p[2]), yaw)
    })
}

/// Decode one upstream 16-element NED world state into cybflight's ENU/FLU
/// [`VehicleState`].
///
/// Layout: `[x y z | vx vy vz | φ θ ψ | p q r | w₁..w₄]`, with the rotor
/// entries **normalized** to `[-1, 1]` against `[0, RL_OMEGA_NORM_MAX]`.
pub fn world_state_ned_to_enu(ws: &[f32]) -> VehicleState {
    let to_enu = |v: Vector3<f32>| Vector3::new(v.y, v.x, -v.z);
    let q_ned = euler_angles_rpy_to_quaternion(&Vector3::new(ws[6], ws[7], ws[8]));
    // R_enu_flu = M·R_ned_frd·N with M, N involutive reflections, so the
    // same expression that builds the NED attitude inverts it.
    let (m, n) = (
        Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0),
    );
    VehicleState {
        position_m: to_enu(Vector3::new(ws[0], ws[1], ws[2])),
        velocity_m_s: to_enu(Vector3::new(ws[3], ws[4], ws[5])),
        attitude: UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
            m * q_ned.to_rotation_matrix().into_inner() * n,
        )),
        body_rate_rad_s: Vector3::new(ws[9], -ws[10], -ws[11]),
        rotor_omega_rad_s: core::array::from_fn(|i| {
            (ws[12 + i] + 1.0) * 0.5 * RL_OMEGA_NORM_MAX
        }),
    }
}

/// The upstream env's CTBR action interface: a proportional body-rate loop
/// feeding a control allocation over the identified effectiveness model.
///
/// The ACMPC actor commands collective thrust and body rates; this is what
/// the training environment put between that command and the plant's
/// native per-motor input, so a policy trained on it must be flown through
/// it to be reproduced.
#[derive(Clone, Copy, Debug)]
pub struct RateLoopConfig {
    /// Proportional gain per axis, FLU [1/s]: `α_des = k·(ω_sp − ω)`.
    pub rate_gains: Vector3<f32>,
    /// Newton iterations of the allocation solve.
    pub newton_iters: usize,
}

impl Default for RateLoopConfig {
    fn default() -> Self {
        Self { rate_gains: Vector3::new(15.0, 15.0, 8.0), newton_iters: 2 }
    }
}

/// Identified plant coefficients, in the upstream parameterization.
///
/// Every coefficient is **specific** — fit against accelerometer and gyro
/// data, so each carries a hidden `1/m` or `1/I`. They are therefore
/// airframe-independent in the sense that matters here: reproducing them
/// requires no mass or inertia at all.
#[derive(Clone, Copy, Debug)]
pub struct RlParams {
    /// Specific thrust [m/s² per (rad/s)²]: `a_z = k_w · Σωᵢ²`.
    pub k_w: f32,
    /// Specific rotor drag, body x and y [1/rad]: `a_x = −k_x · v_bx · Σωᵢ`.
    pub k_x: f32,
    pub k_y: f32,
    /// Roll/pitch angular acceleration per `ωᵢ²`, per motor [1/(rad·s)].
    pub k_p: [f32; NUM_MOTORS],
    pub k_q: [f32; NUM_MOTORS],
    /// Yaw angular acceleration per `ωᵢ` (linear) and per `ω̇ᵢ`.
    pub k_r: [f32; NUM_MOTORS],
    pub k_rd: [f32; NUM_MOTORS],
    /// Actuator: `ω_c = (ω_max−ω_min)·√(k·u² + (1−k)·u) + ω_min`,
    /// `ω̇ = (ω_c − ω)/τ`.
    pub omega_min: f32,
    pub omega_max: f32,
    pub curve_k: f32,
    pub tau_s: f32,
    pub gravity: f32,
}

impl RlParams {
    /// The identified 5-inch racing quad (`randomization.py::params_5inch`).
    pub fn five_inch() -> Self {
        Self {
            k_w: 2.49e-6,
            k_x: 4.85e-5,
            k_y: 7.28e-5,
            k_p: [6.55e-5, 6.61e-5, 6.36e-5, 6.67e-5],
            k_q: [5.28e-5, 5.86e-5, 5.05e-5, 5.89e-5],
            k_r: [1.07e-2; NUM_MOTORS],
            k_rd: [1.97e-3; NUM_MOTORS],
            omega_min: 238.49,
            omega_max: 3295.50,
            curve_k: 0.95,
            tau_s: 0.04,
            gravity: 9.81,
        }
    }

    /// Rotor speed at which the four rotors exactly cancel gravity.
    pub fn hover_omega(&self) -> f32 {
        (self.gravity / (4.0 * self.k_w)).sqrt()
    }

    /// Steady-state rotor speed commanded by normalized throttle `u ∈ [0,1]`.
    pub fn omega_command(&self, u: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        let s = (self.curve_k * u * u + (1.0 - self.curve_k) * u).max(0.0);
        (self.omega_max - self.omega_min) * s.sqrt() + self.omega_min
    }

    /// Peak collective acceleration, as a thrust-to-weight ratio.
    pub fn thrust_to_weight(&self) -> f32 {
        self.k_w * 4.0 * self.omega_max * self.omega_max / self.gravity
    }

    /// Exact inverse of [`RlParams::omega_command`]: the throttle whose
    /// steady state is `omega`, from the positive root of
    /// `k·u² + (1−k)·u − σ² = 0`.
    pub fn command_for_omega(&self, omega: f32) -> f32 {
        let s = ((omega - self.omega_min) / (self.omega_max - self.omega_min)).clamp(0.0, 1.0);
        let (target, b) = (s * s, 1.0 - self.curve_k);
        if self.curve_k <= 1e-6 {
            target
        } else {
            ((b * b + 4.0 * self.curve_k * target).sqrt() - b) / (2.0 * self.curve_k)
        }
    }
}

/// Per-motor sign pattern of the identified roll / pitch / yaw moment
/// rows, **FLU** — read straight off [`RlReferencePlant::step`], so the
/// allocation is by construction the inverse of the plant it feeds.
const S_ROLL: [f32; NUM_MOTORS] = [-1.0, -1.0, 1.0, 1.0];
const S_PITCH: [f32; NUM_MOTORS] = [1.0, -1.0, 1.0, -1.0];
const S_YAW: [f32; NUM_MOTORS] = [1.0, -1.0, -1.0, 1.0];

/// The reference plant. ENU world, FLU body.
#[derive(Clone, Debug)]
pub struct RlReferencePlant {
    pub position_m: Vector3<f32>,
    pub velocity_m_s: Vector3<f32>,
    /// FLU body → ENU world.
    pub attitude: UnitQuaternion<f32>,
    pub body_rate_rad_s: Vector3<f32>,
    pub rotor_omega_rad_s: [f32; NUM_MOTORS],
    params: RlParams,
    time_s: f32,
}

impl RlReferencePlant {
    /// Start hovering at `position_m`, level, heading `yaw_rad` (ENU: from
    /// +x East toward +y North), rotors already at hover speed.
    pub fn hovering_at(params: RlParams, position_m: Vector3<f32>, yaw_rad: f32) -> Self {
        let w = params.hover_omega();
        Self {
            position_m,
            velocity_m_s: Vector3::zeros(),
            attitude: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw_rad),
            body_rate_rad_s: Vector3::zeros(),
            rotor_omega_rad_s: [w; NUM_MOTORS],
            params,
            time_s: 0.0,
        }
    }

    pub fn params(&self) -> &RlParams {
        &self.params
    }

    pub fn time_s(&self) -> f32 {
        self.time_s
    }

    /// State in the form the policy consumes.
    pub fn vehicle_state(&self) -> VehicleState {
        VehicleState {
            position_m: self.position_m,
            velocity_m_s: self.velocity_m_s,
            attitude: self.attitude,
            body_rate_rad_s: self.body_rate_rad_s,
            rotor_omega_rad_s: self.rotor_omega_rad_s,
        }
    }

    /// Track a CTBR command: proportional rate loop, then allocation.
    ///
    /// Returns normalized motor commands `u ∈ [0, 1]` for [`Self::step`].
    ///
    /// Thrust, roll and pitch are quadratic in rotor speed while yaw is
    /// linear, so no linear mixer exists and the allocation is a short
    /// Newton solve seeded from the current rotor speeds. The transient
    /// yaw terms (`k_rd·ω̇`) are left out: they vanish at the equilibrium
    /// being solved for, and including them would make the target depend
    /// on the answer.
    pub fn track_ctbr(
        &self,
        cmd: &CtbrCommand,
        cfg: &RateLoopConfig,
    ) -> [f32; NUM_MOTORS] {
        let p = &self.params;
        let alpha = cfg
            .rate_gains
            .component_mul(&(cmd.body_rate_rad_s - self.body_rate_rad_s));
        let target = Vector4::new(
            cmd.specific_thrust_m_s2,
            alpha.x,
            alpha.y,
            alpha.z,
        );
        // Rows: collective, roll, pitch, yaw. The first three are `c·ω²`,
        // the last `c·ω`; `quad` selects which residual/derivative pair a
        // row uses.
        let coeff: [[f32; NUM_MOTORS]; 4] = [
            [p.k_w; NUM_MOTORS],
            core::array::from_fn(|i| S_ROLL[i] * p.k_p[i]),
            core::array::from_fn(|i| S_PITCH[i] * p.k_q[i]),
            core::array::from_fn(|i| S_YAW[i] * p.k_r[i]),
        ];

        let mut w = self.rotor_omega_rad_s.map(|v| v.clamp(p.omega_min, RL_OMEGA_NORM_MAX));
        for _ in 0..cfg.newton_iters {
            let residual = Vector4::from_fn(|r, _| {
                let quad = r < 3;
                (0..NUM_MOTORS)
                    .map(|i| coeff[r][i] * if quad { w[i] * w[i] } else { w[i] })
                    .sum::<f32>()
                    - target[r]
            });
            let jac = Matrix4::from_fn(|r, i| {
                if r < 3 { 2.0 * coeff[r][i] * w[i] } else { coeff[r][i] }
            });
            let Some(step) = jac.lu().solve(&residual) else {
                break;
            };
            for i in 0..NUM_MOTORS {
                w[i] = (w[i] - step[i]).clamp(p.omega_min, RL_OMEGA_NORM_MAX);
            }
        }
        w.map(|v| p.command_for_omega(v))
    }

    /// One forward-Euler step under normalized motor commands `u ∈ [0,1]`,
    /// indexed in cybflight motor order (rear-right, front-right, rear-left,
    /// front-left).
    pub fn step(&mut self, u: &[f32; NUM_MOTORS], dt: f32) {
        let p = &self.params;
        let w = self.rotor_omega_rad_s;

        // ── Actuator: first-order lag toward the commanded steady state ──
        let mut w_dot = [0.0f32; NUM_MOTORS];
        for i in 0..NUM_MOTORS {
            w_dot[i] = (p.omega_command(u[i]) - w[i]) / p.tau_s;
        }

        let sum_w: f32 = w.iter().sum();
        let sum_w2: f32 = w.iter().map(|v| v * v).sum();

        // ── Specific force, body FLU ──
        // Thrust acts along +z_b (up in FLU; the upstream −z_b in FRD).
        // Drag is form-invariant under the FRD↔FLU flip: both v_by and a_by
        // negate, so the coefficient and sign carry over unchanged.
        let rot = self.attitude.to_rotation_matrix();
        let v_body = rot.inverse_transform_vector(&self.velocity_m_s);
        let a_body = Vector3::new(
            -p.k_x * v_body.x * sum_w,
            -p.k_y * v_body.y * sum_w,
            p.k_w * sum_w2,
        );
        let accel_world =
            rot.transform_vector(&a_body) - Vector3::new(0.0, 0.0, p.gravity);

        // ── Angular acceleration, body FLU ──
        // Upstream (FRD): ṗ = −k_p₁ω₁² − k_p₂ω₂² + k_p₃ω₃² + k_p₄ω₄²
        //                 q̇ = −k_q₁ω₁² + k_q₂ω₂² − k_q₃ω₃² + k_q₄ω₄²
        //                 ṙ = −k_r₁ω₁ + k_r₂ω₂ + k_r₃ω₃ − k_r₄ω₄  (+ ω̇ terms)
        // Roll shares its axis with FLU, so it carries over verbatim; pitch
        // and yaw negate. No `ω × Iω` term — see the module note.
        let sq = |i: usize| w[i] * w[i];
        let rate_dot = Vector3::new(
            -p.k_p[0] * sq(0) - p.k_p[1] * sq(1) + p.k_p[2] * sq(2) + p.k_p[3] * sq(3),
            p.k_q[0] * sq(0) - p.k_q[1] * sq(1) + p.k_q[2] * sq(2) - p.k_q[3] * sq(3),
            p.k_r[0] * w[0] - p.k_r[1] * w[1] - p.k_r[2] * w[2] + p.k_r[3] * w[3]
                + p.k_rd[0] * w_dot[0]
                - p.k_rd[1] * w_dot[1]
                - p.k_rd[2] * w_dot[2]
                + p.k_rd[3] * w_dot[3],
        );

        // ── Integrate (forward Euler, matching the trainer) ──
        let q = self.attitude.as_ref();
        let (qx, qy, qz, qw) = (q.i, q.j, q.k, q.w);
        let (wx, wy, wz) = (
            self.body_rate_rad_s.x,
            self.body_rate_rad_s.y,
            self.body_rate_rad_s.z,
        );
        let dq = Quaternion::new(
            0.5 * (-qx * wx - qy * wy - qz * wz),
            0.5 * (qw * wx + qy * wz - qz * wy),
            0.5 * (qw * wy - qx * wz + qz * wx),
            0.5 * (qw * wz + qx * wy - qy * wx),
        );

        self.position_m += self.velocity_m_s * dt;
        self.velocity_m_s += accel_world * dt;
        self.attitude =
            UnitQuaternion::from_quaternion(self.attitude.as_ref() + dq * dt);
        self.body_rate_rad_s += rate_dot * dt;
        for i in 0..NUM_MOTORS {
            self.rotor_omega_rad_s[i] = (w[i] + w_dot[i] * dt).max(0.0);
        }
        self.time_s += dt;
    }
}
