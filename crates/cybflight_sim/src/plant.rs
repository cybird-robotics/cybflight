//! Quadrotor plant: rotor-state rigid-body simulator (ENU + FLU).
//!
//! **This is deliberately NOT `cybflight_core::mpc::FullQuadModel`.** That
//! struct is the *prediction* model the MPC carries internally; before this
//! module owned its own dynamics, the plant propagated the very same struct
//! with the very same parameters, so every MPC metric was measured against
//! the controller's own assumptions. The plant now models the physics the
//! controller does not know about:
//!
//!   * rotor speed as state, with a first-order lag and a curved
//!     command→speed map (the controller sees only its analytic inverse),
//!   * per-rotor thrust `T = c_T·ω²` rather than commanded force,
//!   * rotor-speed-proportional aerodynamic drag,
//!   * yaw reaction torque from rotor angular acceleration,
//!   * rotor gyroscopic precession,
//!   * the full inertia tensor (not just its diagonal).
//!
//! State layout (`NX_PLANT` = 17):
//! ```text
//!   [ px py pz | qx qy qz qw | vx vy vz | wx wy wz | ω0 ω1 ω2 ω3 ]
//!     0  1  2    3  4  5  6    7  8  9    10 11 12   13 14 15 16
//! ```
//! The leading 13 entries are byte-compatible with `FullQuadModel`'s state,
//! so [`QuadPlant::control_state`] hands controllers exactly what they took
//! before.
//!
//! Control input (`NU` = 4): per-motor **normalized command** `d ∈ [0, 1]` —
//! what actually goes out to the ESC. Betaflight QuadX ordering.
//!
//! Frames: inertial ENU (z up), body FLU (x fwd, y left, z up), quaternion
//! `[qx, qy, qz, qw]` scalar-last mapping FLU-body → ENU-world. Thrust acts
//! along +body z.
//!
//! ## Equations
//! ```text
//!   ω_c,i = (ω_max − ω_min)·√(k·dᵢ² + (1−k)·dᵢ) + ω_min      (ZOH command)
//!   ω̇ᵢ    = (ω_c,i − ωᵢ) / τ
//!   Tᵢ    = εᵢ · c_T · ωᵢ²                                    (per-rotor thrust)
//!   Σω    = Σᵢ ωᵢ
//!   v_b   = R(q)ᵀ · v_w
//!   F_b   = [ −c_dx·Σω·v_bx, −c_dy·Σω·v_by, ΣTᵢ − c_dz·Σω·v_bz ]
//!   a_w   = R(q)·F_b/m − [0, 0, g]
//!   τ_x   = Σᵢ  l_y,i·Tᵢ
//!   τ_y   = Σᵢ −l_x,i·Tᵢ
//!   τ_z   = Σᵢ  sᵢ·( c_Q,i·Tᵢ + J_r·ω̇ᵢ )
//!   τ_gyr = J_r·(Σᵢ sᵢ·ωᵢ)·(ω_b × ẑ_b)
//!   ω̇_b   = I⁻¹·( τ + τ_gyr − ω_b × I·ω_b )
//!   q̇     = ½·q ⊗ [ω_b, 0]
//! ```
//!
//! ## Sign conventions
//! `sᵢ = +1` for CW-from-above (matching `SpinDir::Cw as i32`). A CW rotor's
//! angular momentum points along **−z_b**, so accelerating it produces a
//! body reaction torque along **+z_b** — the *same* sign as its steady-state
//! aerodynamic drag torque. Both yaw terms therefore carry `sᵢ`, and
//! `h_rotor = −ẑ_b·J_r·Σ sᵢωᵢ` gives `τ_gyr = −ω_b × h_rotor` as written.

use cybflight_core::mpc::{NU, NX};
use cybflight_core::params::FirmwareConfig;
use nalgebra::{Matrix3, Quaternion, SVector, UnitQuaternion, Vector3};
use vehicle_yaml::SimYaml;

/// Plant state dimension: the 13-state rigid body plus 4 rotor speeds.
pub const NX_PLANT: usize = NX + NU;

/// Index of the first rotor-speed entry in the plant state vector.
pub const ROTOR_OFFSET: usize = NX;

/// Canonical host-side vehicle parameters — loaded from the FROZEN
/// `vehicles/sim_baseline.yaml` through the same `vehicle-yaml` loader
/// the firmware bake uses (one schema, one parser).
///
/// **Snapshot-stability contract** (see `CLAUDE.md` "Sim regression
/// snapshot"): the baseline deliberately lags the flight tune
/// (`vehicles/sakura_bench.yaml`); firmware retunes of schema `Default`s
/// or the flight YAML must NOT shift the sim implicitly. New schema
/// fields adopt their `Default` here automatically — if a new field is
/// tuning-sensitive, pin it in `sim_baseline.yaml` and regenerate the
/// snapshot in the same commit.
pub const VEHICLE: FirmwareConfigBuilder = FirmwareConfigBuilder;

pub struct FirmwareConfigBuilder;

/// Compile-embedded so the sim binary is hermetic; parsed at call time.
const SIM_BASELINE_YAML: &str = include_str!("../../../vehicles/sim_baseline.yaml");

impl FirmwareConfigBuilder {
    pub fn build(&self) -> FirmwareConfig {
        self.load().0
    }

    /// Params + the plant-only `sim:` section. Callers that construct a
    /// [`QuadPlant`] want both halves; `build()` is the params-only view
    /// kept for controller/scenario construction.
    pub fn load(&self) -> (FirmwareConfig, SimYaml) {
        let v = vehicle_yaml::load("sim_baseline", SIM_BASELINE_YAML)
            .expect("vehicles/sim_baseline.yaml invalid");
        (v.params, v.sim)
    }

    pub fn sim(&self) -> SimYaml {
        self.load().1
    }
}

/// Physical plant parameters, resolved from a [`FirmwareConfig`] plus the
/// vehicle's plant-only [`SimYaml`] section.
///
/// Public and `Clone` so a scenario can perturb any single quantity after
/// construction — that is how deliberate plant/controller mismatch (a
/// slower rotor than INDI assumes, a weak motor, more drag than modelled)
/// is introduced without a second YAML.
#[derive(Clone, Debug)]
pub struct PlantParams {
    pub mass_kg: f32,
    pub grav: f32,
    pub inertia: Matrix3<f32>,
    pub inertia_inv: Matrix3<f32>,
    /// Per-motor `[x, y]` position in body FLU [m].
    pub motor_pos: [[f32; 2]; NU],
    /// Per-motor spin sign: `+1` CW (from above), `−1` CCW.
    pub spin_sign: [f32; NU],
    /// Per-motor yaw torque per newton of thrust [m].
    pub torque_coeff_m: [f32; NU],
    /// Per-motor thrust coefficient `c_T` [N·s²/rad²], already including
    /// the `motor_thrust_scale` multiplier.
    pub thrust_coeff: [f32; NU],
    /// Per-motor rotor speed at full throttle [rad/s].
    pub omega_max: [f32; NU],
    /// Per-motor rotor speed at zero throttle [rad/s].
    pub omega_min: [f32; NU],
    /// Per-motor first-order rotor time constant [s].
    pub tau_s: [f32; NU],
    /// Throttle-curve curvature `k ∈ [0, 1]`.
    pub throttle_curve_k: f32,
    /// Rotor + prop polar inertia [kg·m²].
    pub rotor_inertia: f32,
    /// Body-frame rotor drag coefficients [N·s²/(m·rad)].
    pub aero_drag: Vector3<f32>,
    /// Body-frame quadratic drag `½ρC_dA` per axis [N·s²/m²].
    pub body_drag: Vector3<f32>,
    /// External disturbance force in the WORLD frame [N] — wind, tether,
    /// contact. Zero by default; the RL environment's domain
    /// randomization drives it per tick. Deliberately kept out of
    /// [`QuadPlant::specific_force_body`] (like `body_drag`) so the
    /// disturbance is unmodeled by every controller input path.
    pub external_force_n: Vector3<f32>,
}

impl PlantParams {
    /// Resolve plant parameters from the vehicle definition.
    ///
    /// `c_T` is *derived*, not declared: `c_T = T_max / ω_max²`. Declaring
    /// it separately would let a YAML specify a thrust coefficient that
    /// disagrees with the `max_thrust_n` the controller's G1 is built
    /// from — the motor would saturate somewhere other than where every
    /// allocator believes it does.
    pub fn from_config(vp: &FirmwareConfig, sim: &SimYaml) -> Self {
        let inertia = vp.airframe.body.inertia_matrix();
        let inertia_inv = inertia
            .try_inverse()
            .expect("plant: inertia tensor is singular");

        let scale = sim.thrust_scale();

        let mut motor_pos = [[0.0f32; 2]; NU];
        let mut spin_sign = [0.0f32; NU];
        let mut torque_coeff_m = [0.0f32; NU];
        let mut thrust_coeff = [0.0f32; NU];
        let mut omega_max = [0.0f32; NU];
        let mut omega_min = [0.0f32; NU];
        let mut tau_s = [0.0f32; NU];

        for i in 0..NU {
            let m = &vp.airframe.motors[i];
            motor_pos[i] = m.position_m;
            spin_sign[i] = m.spin_dir as i32 as f32;
            torque_coeff_m[i] = m.torque_coeff_m;

            let w_max = if m.max_omega_rad_s.is_finite() && m.max_omega_rad_s > 1.0 {
                m.max_omega_rad_s
            } else {
                // Same fallback the firmware INDI task uses (40 000 RPM).
                4188.79
            };
            omega_max[i] = w_max;
            omega_min[i] = sim.rotor_omega_min_rad_s.clamp(0.0, 0.9 * w_max);
            thrust_coeff[i] = scale[i] * m.max_thrust_n / (w_max * w_max);
            tau_s[i] = if m.time_const_s.is_finite() && m.time_const_s > 1e-4 {
                m.time_const_s
            } else {
                0.02
            };
        }

        Self {
            mass_kg: vp.airframe.body.mass_kg,
            // The simulated world's gravity, from the same `site` param the
            // controllers read — so a non-standard value moves plant and
            // controller together rather than manufacturing a model error.
            grav: vp.site.gravity_m_s2,
            inertia,
            inertia_inv,
            motor_pos,
            spin_sign,
            torque_coeff_m,
            thrust_coeff,
            omega_max,
            omega_min,
            tau_s,
            throttle_curve_k: sim.rotor_throttle_curve_k.clamp(0.0, 1.0),
            rotor_inertia: sim.rotor_inertia_kg_m2.max(0.0),
            aero_drag: Vector3::from(sim.aero_drag),
            body_drag: Vector3::from(sim.body_drag),
            external_force_n: Vector3::zeros(),
        }
    }

    /// Steady-state rotor speed [rad/s] for a normalized command
    /// `d ∈ [0, 1]`, per the identified actuator map.
    #[inline]
    pub fn omega_command(&self, i: usize, d: f32) -> f32 {
        let d = d.clamp(0.0, 1.0);
        let k = self.throttle_curve_k;
        let s = (k * d * d + (1.0 - k) * d).max(0.0).sqrt();
        (self.omega_max[i] - self.omega_min[i]) * s + self.omega_min[i]
    }

    /// Rotor speed [rad/s] that produces `thrust_n` on motor `i`.
    #[inline]
    pub fn omega_for_thrust(&self, i: usize, thrust_n: f32) -> f32 {
        (thrust_n.max(0.0) / self.thrust_coeff[i]).sqrt()
    }

    /// Inverse of [`Self::omega_command`]: the normalized command whose
    /// steady state is `omega`. Used to seed the plant at hover and by
    /// the thrust-commanding baseline controllers.
    #[inline]
    pub fn command_for_omega(&self, i: usize, omega: f32) -> f32 {
        let span = self.omega_max[i] - self.omega_min[i];
        if span <= 0.0 {
            return 0.0;
        }
        let s = ((omega - self.omega_min[i]) / span).clamp(0.0, 1.0);
        let target = s * s; // = k·d² + (1−k)·d
        let k = self.throttle_curve_k;
        if k <= 1e-6 {
            target.clamp(0.0, 1.0)
        } else {
            // Positive root of k·d² + (1−k)·d − target = 0.
            let b = 1.0 - k;
            (((b * b + 4.0 * k * target).max(0.0).sqrt() - b) / (2.0 * k)).clamp(0.0, 1.0)
        }
    }

    /// Normalized command that holds a hover on all four motors.
    pub fn hover_command(&self) -> SVector<f32, NU> {
        let per_motor = self.mass_kg * self.grav / NU as f32;
        SVector::<f32, NU>::from_fn(|i, _| {
            self.command_for_omega(i, self.omega_for_thrust(i, per_motor))
        })
    }

    /// Rotor speeds at hover — the plant's reset condition.
    pub fn hover_omega(&self) -> SVector<f32, NU> {
        let per_motor = self.mass_kg * self.grav / NU as f32;
        SVector::<f32, NU>::from_fn(|i, _| self.omega_for_thrust(i, per_motor))
    }
}

/// Rotor-state quadrotor simulator.
pub struct QuadPlant {
    pub dt_sim: f32,
    pub params: FirmwareConfig,
    pub plant: PlantParams,
    state: SVector<f32, NX_PLANT>,
    /// Last evaluated rotor angular acceleration [rad/s²]. Reported for
    /// telemetry/diagnostics; the yaw reaction torque uses the value
    /// computed inside each RK4 stage, not this cached one.
    omega_dot: SVector<f32, NU>,
    time_s: f32,
}

/// Project the quaternion block of a plant state back onto the unit
/// 3-sphere, and clamp rotor speeds to be non-negative.
#[inline]
fn normalize_state(x: &mut SVector<f32, NX_PLANT>) {
    let n = (x[3] * x[3] + x[4] * x[4] + x[5] * x[5] + x[6] * x[6]).sqrt();
    if n > 1e-9 {
        let inv = 1.0 / n;
        x[3] *= inv;
        x[4] *= inv;
        x[5] *= inv;
        x[6] *= inv;
    } else {
        x[3] = 0.0;
        x[4] = 0.0;
        x[5] = 0.0;
        x[6] = 1.0;
    }
    for i in 0..NU {
        if x[ROTOR_OFFSET + i] < 0.0 {
            x[ROTOR_OFFSET + i] = 0.0;
        }
    }
}

impl QuadPlant {
    pub fn new(params: FirmwareConfig, sim: &SimYaml, dt_sim: f32) -> Self {
        let plant = PlantParams::from_config(&params, sim);
        let mut state = SVector::<f32, NX_PLANT>::zeros();
        state[6] = 1.0;
        state
            .fixed_rows_mut::<NU>(ROTOR_OFFSET)
            .copy_from(&plant.hover_omega());

        Self {
            dt_sim,
            params,
            plant,
            state,
            omega_dot: SVector::zeros(),
            time_s: 0.0,
        }
    }

    /// Construct from the frozen `sim_baseline.yaml`.
    pub fn default_vehicle(dt_sim: f32) -> Self {
        let (params, sim) = VEHICLE.load();
        Self::new(params, &sim, dt_sim)
    }

    /// Overwrite the rigid-body state. Quaternion is re-normalized, and
    /// rotor speeds are seeded at hover so a scenario does not open with a
    /// spin-up transient that would read as free fall to INDI's takeoff
    /// detector.
    pub fn reset(&mut self, pos: Vector3<f32>, vel: Vector3<f32>, attitude: UnitQuaternion<f32>) {
        let q = attitude.into_inner();
        self.state = SVector::<f32, NX_PLANT>::zeros();
        self.state.fixed_rows_mut::<3>(0).copy_from(&pos);
        self.state.fixed_rows_mut::<4>(3).copy_from(&q.coords);
        self.state.fixed_rows_mut::<3>(7).copy_from(&vel);
        self.state
            .fixed_rows_mut::<NU>(ROTOR_OFFSET)
            .copy_from(&self.plant.hover_omega());
        normalize_state(&mut self.state);
        self.omega_dot = SVector::zeros();
        self.time_s = 0.0;
    }

    /// Continuous-time dynamics. `d` is the per-motor normalized command,
    /// held constant across the step (ZOH, matching an ESC frame).
    ///
    /// Also returns the rotor angular accelerations, which the caller
    /// needs both for the yaw reaction torque (already applied here) and
    /// for telemetry.
    fn dynamics(
        &self,
        x: &SVector<f32, NX_PLANT>,
        d: &SVector<f32, NU>,
    ) -> (SVector<f32, NX_PLANT>, SVector<f32, NU>) {
        let p = &self.plant;
        let q = UnitQuaternion::from_quaternion(Quaternion::new(x[6], x[3], x[4], x[5]));
        let vel = Vector3::new(x[7], x[8], x[9]);
        let rate = Vector3::new(x[10], x[11], x[12]);

        // ── Actuator: first-order lag toward the commanded steady state ──
        let mut omega = SVector::<f32, NU>::zeros();
        let mut omega_dot = SVector::<f32, NU>::zeros();
        let mut thrust = SVector::<f32, NU>::zeros();
        for i in 0..NU {
            let w = x[ROTOR_OFFSET + i].max(0.0);
            omega[i] = w;
            omega_dot[i] = (p.omega_command(i, d[i]) - w) / p.tau_s[i];
            thrust[i] = p.thrust_coeff[i] * w * w;
        }
        let omega_sum: f32 = omega.iter().sum();
        let thrust_sum: f32 = thrust.iter().sum();

        // ── Forces: thrust along +z_b, rotor drag opposing body velocity ──
        let rot = q.to_rotation_matrix();
        let v_body = rot.inverse_transform_vector(&vel);
        let force_body = Vector3::new(
            -p.aero_drag.x * omega_sum * v_body.x - p.body_drag.x * v_body.x.abs() * v_body.x,
            -p.aero_drag.y * omega_sum * v_body.y - p.body_drag.y * v_body.y.abs() * v_body.y,
            thrust_sum
                - p.aero_drag.z * omega_sum * v_body.z
                - p.body_drag.z * v_body.z.abs() * v_body.z,
        );
        let accel_world = rot.transform_vector(&force_body) / p.mass_kg
            + p.external_force_n / p.mass_kg
            - Vector3::new(0.0, 0.0, p.grav);

        // ── Torques ──
        let mut torque = Vector3::zeros();
        let mut spin_momentum = 0.0f32;
        for i in 0..NU {
            let [lx, ly] = p.motor_pos[i];
            let s = p.spin_sign[i];
            torque.x += ly * thrust[i];
            torque.y += -lx * thrust[i];
            // Steady-state prop drag reaction AND rotor spin-up reaction
            // share the sign of `s` — see the module-level sign note.
            torque.z += s * (p.torque_coeff_m[i] * thrust[i] + p.rotor_inertia * omega_dot[i]);
            spin_momentum += s * omega[i];
        }
        // Rotor gyroscopic precession: τ = −ω_b × h_rotor with
        // h_rotor = −ẑ_b·J_r·Σ sᵢωᵢ, and ω_b × ẑ_b = (wy, −wx, 0).
        let gyro_scale = p.rotor_inertia * spin_momentum;
        torque += Vector3::new(gyro_scale * rate.y, -gyro_scale * rate.x, 0.0);

        let rate_dot = p.inertia_inv * (torque - rate.cross(&(p.inertia * rate)));

        // ── Assemble ──
        let (qx, qy, qz, qw) = (x[3], x[4], x[5], x[6]);
        let (wx, wy, wz) = (rate.x, rate.y, rate.z);
        let mut xdot = SVector::<f32, NX_PLANT>::zeros();
        xdot[0] = vel.x;
        xdot[1] = vel.y;
        xdot[2] = vel.z;
        xdot[3] = 0.5 * (qw * wx + qy * wz - qz * wy);
        xdot[4] = 0.5 * (qw * wy - qx * wz + qz * wx);
        xdot[5] = 0.5 * (qw * wz + qx * wy - qy * wx);
        xdot[6] = 0.5 * (-qx * wx - qy * wy - qz * wz);
        xdot.fixed_rows_mut::<3>(7).copy_from(&accel_world);
        xdot.fixed_rows_mut::<3>(10).copy_from(&rate_dot);
        xdot.fixed_rows_mut::<NU>(ROTOR_OFFSET).copy_from(&omega_dot);
        (xdot, omega_dot)
    }

    /// Advance one substep given per-motor normalized commands `d ∈ [0,1]`.
    ///
    /// RK4 with the quaternion projected back onto the unit sphere at every
    /// intermediate stage, so stage drift is bounded by one normalization
    /// rather than compounding across the four sub-steps.
    /// Overwrite the body-rate block only (recovery-episode disturbances).
    pub fn set_body_rate(&mut self, w: Vector3<f32>) {
        self.state.fixed_rows_mut::<3>(10).copy_from(&w);
    }

    pub fn step(&mut self, d: &SVector<f32, NU>) {
        let d = SVector::<f32, NU>::from_fn(|i, _| {
            let v = d[i];
            if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
        });
        let dt = self.dt_sim;
        let half_dt = 0.5 * dt;

        let (k0, omega_dot) = self.dynamics(&self.state, &d);
        self.omega_dot = omega_dot;

        let mut xs = self.state + k0 * half_dt;
        normalize_state(&mut xs);
        let (k1, _) = self.dynamics(&xs, &d);

        xs = self.state + k1 * half_dt;
        normalize_state(&mut xs);
        let (k2, _) = self.dynamics(&xs, &d);

        xs = self.state + k2 * dt;
        normalize_state(&mut xs);
        let (k3, _) = self.dynamics(&xs, &d);

        self.state += (k0 + 2.0 * k1 + 2.0 * k2 + k3) * (dt / 6.0);
        normalize_state(&mut self.state);
        self.time_s += dt;
    }

    pub fn time_s(&self) -> f32 {
        self.time_s
    }

    /// Set the world-frame external disturbance force [N] applied from
    /// the next `step` on (see [`PlantParams::external_force_n`]).
    pub fn set_external_force(&mut self, f: Vector3<f32>) {
        self.plant.external_force_n = f;
    }

    /// Full 17-state plant vector (rigid body + rotors).
    pub fn raw_state(&self) -> &SVector<f32, NX_PLANT> {
        &self.state
    }

    /// The 13-state rigid-body slice controllers consume.
    pub fn control_state(&self) -> SVector<f32, NX> {
        self.state.fixed_rows::<NX>(0).into_owned()
    }

    pub fn position(&self) -> Vector3<f32> {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }

    pub fn velocity(&self) -> Vector3<f32> {
        Vector3::new(self.state[7], self.state[8], self.state[9])
    }

    pub fn body_rate(&self) -> Vector3<f32> {
        Vector3::new(self.state[10], self.state[11], self.state[12])
    }

    pub fn attitude(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::from_quaternion(Quaternion::from_vector(
            self.state.fixed_rows::<4>(3).into(),
        ))
    }

    /// Rotor speeds [rad/s].
    pub fn rotor_omega(&self) -> SVector<f32, NU> {
        self.state.fixed_rows::<NU>(ROTOR_OFFSET).into_owned()
    }

    /// Rotor angular accelerations [rad/s²] from the most recent step.
    pub fn rotor_omega_dot(&self) -> SVector<f32, NU> {
        self.omega_dot
    }

    /// Per-rotor thrust [N] at the current rotor speeds.
    pub fn motor_thrusts(&self) -> SVector<f32, NU> {
        SVector::<f32, NU>::from_fn(|i, _| {
            let w = self.state[ROTOR_OFFSET + i].max(0.0);
            self.plant.thrust_coeff[i] * w * w
        })
    }

    /// Body-frame specific force [m/s²] — proper acceleration, i.e. what a
    /// strapdown accelerometer reads. Includes aerodynamic drag; excludes
    /// gravity. This is `F_body / m` with `F_body` the total
    /// non-gravitational force.
    pub fn specific_force_body(&self) -> Vector3<f32> {
        let p = &self.plant;
        let omega_sum: f32 = self.rotor_omega().iter().sum();
        let thrust_sum: f32 = self.motor_thrusts().iter().sum();
        let v_body = self
            .attitude()
            .to_rotation_matrix()
            .inverse_transform_vector(&self.velocity());
        Vector3::new(
            -p.aero_drag.x * omega_sum * v_body.x,
            -p.aero_drag.y * omega_sum * v_body.y,
            thrust_sum - p.aero_drag.z * omega_sum * v_body.z,
        ) / p.mass_kg
    }

    /// Fraction of full throttle each motor's current speed corresponds to
    /// — the saturation metric, now measured on rotor state rather than on
    /// the commanded force.
    pub fn rotor_saturation(&self) -> SVector<f32, NU> {
        SVector::<f32, NU>::from_fn(|i, _| {
            (self.state[ROTOR_OFFSET + i] / self.plant.omega_max[i]).clamp(0.0, 1.0)
        })
    }

    /// Tilt angle between body-z and world-z (radians, ≥0).
    pub fn tilt_rad(&self) -> f32 {
        let q = self.attitude();
        let zb_world_z = q.to_rotation_matrix()[(2, 2)];
        zb_world_z.clamp(-1.0, 1.0).acos()
    }
}
