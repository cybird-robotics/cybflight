//! Recovery-episode environment for training the situation-conditioned
//! cost policy (docs/learned_mpc_cost.md, PLAN B).
//!
//! One environment step is one outer-loop period: the agent's action `z`
//! is written into the [`MpcIndiController`]'s cost, the controller runs
//! its SQP solve and the INDI ticks of that period against the plant, and
//! the observation handed back is the situation the *next* solve will
//! see. The MPC, INDI, plant and sampler are the exact objects the sim
//! harness scores, so the policy trains against the solver it flies with.
//!
//! An episode is a random [`primitives`] trajectory with the vehicle
//! initialized *off* it (position, velocity, attitude and rate
//! disturbances drawn independently of the maneuver). Nothing about the
//! trajectory reaches the policy except through the local situation
//! observation.

use cybflight_core::mpc::cost_adapt::{NZ, OBS_DIM};
use cybflight_core::mpc::quad_model::{N as SIMPLE_N, PosCostMode};
use cybflight_core::params::FirmwareConfig;
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::types::Vec3;
use nalgebra::{UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use vehicle_yaml::SimYaml;

use crate::controller::{Controller, MpcIndiController};
use crate::plant::QuadPlant;
use crate::primitives::{random_primitive, Envelope, Primitive};
use crate::sensors::{ImuModel, NoisyImu, PerfectRotorTelemetry, RotorModel};
use crate::trajectory::{MissionSetpoints, Setpoint, SetpointSource};

/// Horizon spacing of the sim's SQP stack (`controller::SIMPLE_MPC_DT`).
const MPC_DT: f32 = 0.05;

#[derive(Clone, Copy, Debug)]
pub struct RewardWeights {
    /// On the squared closest-point (geometric) error [1/m²].
    pub geom: f32,
    /// On the squared time-indexed error [1/m²].
    pub time: f32,
    /// On the squared normalized body rate.
    pub rate: f32,
    /// On the mean squared change of `z` between consecutive steps.
    pub dz: f32,
    /// One-off penalty on crash (ground, >3 m off the path, non-finite).
    pub crash: f32,
    /// On the squared change of the commanded body rates between
    /// consecutive solves, normalized by `ω_max` — the cost that the MPC's
    /// rate input weight proxies, made explicit so learning that weight
    /// is a trade-off rather than free.
    pub du_rate: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self { geom: 10.0, time: 1.0, rate: 0.2, dz: 0.1, crash: 20.0, du_rate: 25.0 }
    }
}

#[derive(Clone)]
pub struct EnvConfig {
    /// Controller's vehicle (nominal); the plant gets a jittered copy.
    pub vehicle: FirmwareConfig,
    pub sim: SimYaml,
    pub mpc_rate_hz: f32,
    pub indi_rate_hz: f32,
    pub dt_sim: f32,
    /// Episode cap [s]; the primitive's own duration usually ends it first.
    pub max_episode_s: f32,
    /// Zero the observation (learns a constant `z`: the "blind" control).
    pub blind: bool,
    /// Scale on every initial-disturbance magnitude (0 = start on the
    /// reference).
    pub disturb: f32,
    /// Plant mass jitter, fraction (uniform ±).
    pub mass_jitter: f32,
    /// Domain randomization (0 = off, 1 = full). Per episode, plant only
    /// (the controller keeps the nominal model): inertia ×[0.7, 1.5] per
    /// axis, per-motor thrust ceiling ×[0.85, 1.15] (common) ×[0.95, 1.05]
    /// (per motor), motor τ ×[0.5, 2.0] (log-uniform — spans the τ×2
    /// robustness condition and, since v17, a faster-than-nominal rotor
    /// too), rotor drag ×[0.5, 3.0], body drag ×[0.5, 2.0]
    /// around the configured value, plus an unmodeled wind force (mean up
    /// to `WIND_FORCE_MAX_N` with an OU gust — see the `WIND_*`
    /// constants). Per tick, localization noise on the state the outer
    /// loop sees (INDI keeps the IMU): white position σ ∈ [0, 5] cm,
    /// velocity σ ∈ [0, 20] cm/s, attitude σ ∈ [0, 3°], plus a
    /// 1 s-correlated position bias σ ∈ [0, 10] cm, drawn per episode.
    pub domain_rand: f32,
    /// Gyro / accel white noise into INDI [rad/s, m/s²] — sensor noise the
    /// vehicle really has; it is what makes an aggressive rate command
    /// cost something (motor chatter through the rate loop).
    pub gyro_sigma: f32,
    pub accel_sigma: f32,
    /// Nominal body-rate input weight the policy modulates (thrust stays 1).
    pub rate_weight_nominal: f32,
    /// Put the identified rotor drag (`sim: aero_drag`, per-motor `c_T`)
    /// into the controller's prediction model.
    pub model_drag: bool,
    /// Quadratic body drag `[x, y, z]` in the plant **and** (when
    /// `model_drag`) the controller model — the outdoor regime.
    pub body_drag: [f32; 3],
    /// Position-sampler lag allowance [s] (YAML: 0.1). The reward's
    /// closest-point window follows it (`max(0.5, lag + 0.1)`).
    pub sampler_max_lag_s: f32,
    pub reward: RewardWeights,
    /// Where primitives start (the flight volume centre); each episode
    /// offsets this randomly by ±1 m.
    pub volume_centre: Vec3,
}

impl EnvConfig {
    pub fn new(vehicle: FirmwareConfig, sim: SimYaml) -> Self {
        Self {
            vehicle,
            sim,
            mpc_rate_hz: 100.0,
            indi_rate_hz: 1000.0,
            dt_sim: 1.0 / 8000.0,
            max_episode_s: 4.0,
            blind: false,
            disturb: 1.0,
            mass_jitter: 0.2,
            domain_rand: 0.0,
            gyro_sigma: 0.03,
            accel_sigma: 0.3,
            rate_weight_nominal: 5.0,
            sampler_max_lag_s: 0.1,
            model_drag: false,
            body_drag: [0.0; 3],
            reward: RewardWeights::default(),
            volume_centre: Vec3::new(0.0, 0.0, 1.5),
        }
    }

    pub fn envelope(&self) -> Envelope {
        let vp = &self.vehicle;
        let thrust_max_n: f32 = vp.airframe.motors.iter().map(|m| m.max_thrust_n).sum::<f32>()
            * vp.mpc.thrust_frac;
        let mr = vp.airframe.body.max_rate_rad_s;
        Envelope {
            mass_kg: vp.airframe.body.mass_kg,
            grav: vp.site.gravity_m_s2,
            thrust_max_n,
            rate_max: Vec3::new(mr[0], mr[1], mr[2]),
        }
    }
}

/// Per-step diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepInfo {
    pub e_geom: f32,
    pub e_time: f32,
    /// Squared normalized rate-command change this step.
    pub du_rate2: f32,
    pub tilt_rad: f32,
    pub crashed: bool,
    /// Episode ended by the trajectory finishing (a truncation, not a
    /// terminal state — bootstrap the value).
    pub truncated: bool,
    pub diverged: bool,
}

/// Provenance of the current episode (coverage statistics).
#[derive(Clone, Copy, Debug, Default)]
pub struct EpisodeInfo {
    pub kind: u8,
    pub speed: f32,
    pub peak_thrust_frac: f32,
    pub peak_rate_frac: f32,
    pub duration_s: f32,
}

/// Wind disturbance envelope (domain randomization): per-episode mean
/// force magnitude uniform in `[0, WIND_FORCE_MAX_N]` — 1.5 N ≈ 0.25 g
/// lateral on the 0.6 kg airframe, a strong-but-flyable steady wind —
/// with an added OU gust of `WIND_GUST_FRAC · mean` 1-σ per axis and
/// `WIND_GUST_TAU_S` correlation. The mean direction is drawn with its
/// vertical component damped by `WIND_VERTICAL_FRAC` (real wind is
/// mostly horizontal).
const WIND_FORCE_MAX_N: f32 = 1.5;
const WIND_GUST_FRAC: f32 = 0.4;
const WIND_GUST_TAU_S: f32 = 0.5;
const WIND_VERTICAL_FRAC: f32 = 0.3;

pub struct RecoveryEnv {
    cfg: EnvConfig,
    rng: ChaCha8Rng,
    plant: QuadPlant,
    ctrl: MpcIndiController,
    traj: PiecewisePolynomial,
    setpoints: MissionSetpoints,
    duration_s: f32,
    horizon: Vec<Setpoint>,
    imu: NoisyImu,
    rotor: PerfectRotorTelemetry,
    last_rate_cmd: Vector3<f32>,
    tick: u32,
    ticks_per_step: u32,
    substeps: usize,
    last_z: [f32; NZ],
    /// Per-episode localization-noise draw `(σ_p, σ_v, σ_θ, σ_bias)`.
    loc_sigma: [f32; 4],
    loc_bias: Vector3<f32>,
    /// Per-episode wind draw: mean world-frame force [N], OU gust state
    /// and per-axis gust σ (see the `WIND_*` constants).
    wind_mean: Vector3<f32>,
    wind_gust: Vector3<f32>,
    wind_sigma: f32,
    pub episode: EpisodeInfo,
}

/// Closest-point distance from `p` to the whole trajectory: 20 ms coarse
/// grid, then [`closest_point_dist`] refinement around the best coarse
/// sample. Lag-independent (the ±window variant is the reward's, where
/// the window is the sampler's own allowance).
pub fn closest_point_dist_global(traj: &PiecewisePolynomial, p: Vector3<f32>) -> f32 {
    let dur = traj.total_duration();
    if !dur.is_finite() || dur <= 0.0 {
        return f32::NAN;
    }
    let (mut best_t, mut best) = (0.0f32, f32::INFINITY);
    let mut t = 0.0f32;
    while t <= dur {
        let d = (traj.get_pos(t) - p).norm_squared();
        if d < best {
            best = d;
            best_t = t;
        }
        t += 0.02;
    }
    closest_point_dist(traj, p, best_t, 0.03)
}

/// Closest-point distance from `p` to the trajectory within
/// `t_c ± half_window` (5 ms grid then a parabolic refinement).
pub fn closest_point_dist(
    traj: &PiecewisePolynomial,
    p: Vector3<f32>,
    t_c: f32,
    half_window: f32,
) -> f32 {
    let dur = traj.total_duration();
    if !dur.is_finite() || dur < 0.0 {
        return f32::NAN;
    }
    let lo = (t_c - half_window).max(0.0);
    let hi = (t_c + half_window).min(dur);
    let d2 = |t: f32| (traj.get_pos(t) - p).norm_squared();
    let mut best_t = lo;
    let mut best = d2(lo);
    let mut t = lo;
    while t <= hi {
        let v = d2(t);
        if v < best {
            best = v;
            best_t = t;
        }
        t += 0.005;
    }
    // Parabolic refinement on the grid neighbours.
    let h = 0.005;
    if best_t - h >= lo && best_t + h <= hi {
        let (fm, f0, fp) = (d2(best_t - h), best, d2(best_t + h));
        let denom = fm - 2.0 * f0 + fp;
        if denom > 1e-12 {
            let dt = 0.5 * h * (fm - fp) / denom;
            let v = d2(best_t + dt);
            if v < best {
                best = v;
            }
        }
    }
    best.sqrt()
}

impl RecoveryEnv {
    pub fn new(cfg: EnvConfig, seed: u64) -> Self {
        let plant = QuadPlant::new(cfg.vehicle.clone(), &cfg.sim, cfg.dt_sim);
        let ctrl = Self::build_controller(&cfg);
        let ticks_per_step = (cfg.indi_rate_hz / cfg.mpc_rate_hz).round().max(1.0) as u32;
        let substeps = ((1.0 / cfg.indi_rate_hz) / cfg.dt_sim).round().max(1.0) as usize;
        // A valid (if unused) trajectory so every field is well-formed
        // before the first `reset`.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let traj = random_primitive(&mut rng, &cfg.envelope(), cfg.volume_centre).traj;
        let setpoints = MissionSetpoints::from_trajectory(traj.clone());
        Self {
            rng,
            plant,
            ctrl,
            traj,
            setpoints,
            duration_s: 0.0,
            horizon: Vec::with_capacity(SIMPLE_N + 1),
            imu: NoisyImu::isotropic(seed ^ 0xA5A5, cfg.gyro_sigma, cfg.accel_sigma),
            rotor: PerfectRotorTelemetry,
            last_rate_cmd: Vector3::zeros(),
            tick: 0,
            ticks_per_step,
            substeps,
            last_z: [0.0; NZ],
            loc_sigma: [0.0; 4],
            loc_bias: Vector3::zeros(),
            wind_mean: Vector3::zeros(),
            wind_gust: Vector3::zeros(),
            wind_sigma: 0.0,
            episode: EpisodeInfo::default(),
            cfg,
        }
    }

    /// The flight configuration of the SQP stack (see
    /// `tests/figure8_tinympc_compare.rs`): tilt-yaw `q_ref`, flatness
    /// `u_ref` feedforward, position sampler, full horizon.
    fn build_controller(cfg: &EnvConfig) -> MpcIndiController {
        let vp = &cfg.vehicle;
        let mode = if vp.mpc.pos_cost_mode == PosCostMode::Contouring {
            PosCostMode::Contouring
        } else {
            PosCostMode::Quadratic
        };
        let mut c =
            MpcIndiController::with_options(vp, mode, cfg.mpc_rate_hz, cfg.indi_rate_hz, SIMPLE_N);
        c.use_tilt_map = true;
        c.flatness_feedforward = true;
        c.observe_situation = true;
        // Thrust weight from the YAML (fixed); the nominal rate weight the
        // policy modulates is a config choice.
        let r = cfg.rate_weight_nominal;
        c.set_input_weights([vp.mpc.thrust_weight, r, r, r]);
        if cfg.model_drag {
            let m = &vp.airframe.motors[0];
            c.set_rotor_drag(cfg.sim.aero_drag, m.max_thrust_n / (m.max_omega_rad_s * m.max_omega_rad_s));
            c.set_body_drag(cfg.body_drag);
        }
        c
    }

    pub fn obs_dim() -> usize {
        OBS_DIM
    }
    pub fn act_dim() -> usize {
        NZ
    }

    fn t_now(&self) -> f32 {
        self.tick as f32 / self.cfg.indi_rate_hz
    }

    fn fill_horizon(&mut self, t: f32) {
        self.horizon.clear();
        for k in 0..=SIMPLE_N {
            self.horizon.push(self.setpoints.sample(t + k as f32 * MPC_DT));
        }
    }

    /// Standard normal draw (Box–Muller on the episode RNG).
    fn gauss(&mut self) -> f32 {
        let u1: f32 = self.rng.random_range(1e-7..1.0);
        let u2: f32 = self.rng.random_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }

    /// The plant state as the outer loop sees it: ground truth corrupted
    /// by this episode's localization-noise draw (white + a 1 s-correlated
    /// position bias). INDI keeps the IMU; this models the estimator.
    fn observed_state(&mut self) -> nalgebra::SVector<f32, { cybflight_core::mpc::NX }> {
        let mut x = self.plant.control_state();
        let [sp, sv, sth, sb] = self.loc_sigma;
        if sp == 0.0 && sv == 0.0 && sth == 0.0 && sb == 0.0 {
            return x;
        }
        let a = (-1.0 / self.cfg.indi_rate_hz).exp();
        let s = sb * (1.0 - a * a).sqrt();
        for i in 0..3 {
            let g = self.gauss();
            self.loc_bias[i] = a * self.loc_bias[i] + s * g;
            x[i] += sp * self.gauss() + self.loc_bias[i];
            x[7 + i] += sv * self.gauss();
        }
        if sth > 0.0 {
            let dth = Vector3::new(self.gauss(), self.gauss(), self.gauss()) * sth;
            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(x[6], x[3], x[4], x[5]));
            let qn = q * UnitQuaternion::from_scaled_axis(dth);
            x[3] = qn.i;
            x[4] = qn.j;
            x[5] = qn.k;
            x[6] = qn.w;
        }
        x
    }

    fn observe(&mut self) -> [f32; OBS_DIM] {
        if self.cfg.blind {
            return [0.0; OBS_DIM];
        }
        let t = self.t_now();
        self.fill_horizon(t);
        let x = self.observed_state();
        self.ctrl.peek_situation(&x, &self.horizon)
    }

    /// Start a new episode: fresh primitive, jittered plant, disturbed
    /// initial state, fresh controller. Returns the first observation.
    pub fn reset(&mut self) -> [f32; OBS_DIM] {
        let env = self.cfg.envelope();
        let centre = self.cfg.volume_centre
            + Vec3::new(
                self.rng.random_range(-1.0..1.0),
                self.rng.random_range(-1.0..1.0),
                self.rng.random_range(-0.5..0.5),
            );
        let prim: Primitive = random_primitive(&mut self.rng, &env, centre);
        self.episode = EpisodeInfo {
            kind: prim.kind as u8,
            speed: prim.speed,
            peak_thrust_frac: prim.peak_thrust_frac,
            peak_rate_frac: prim.peak_rate_frac,
            duration_s: prim.traj.total_duration(),
        };
        self.duration_s = prim.traj.total_duration().min(self.cfg.max_episode_s);
        self.traj = prim.traj;
        self.setpoints = MissionSetpoints::from_trajectory(self.traj.clone());

        // Plant with jittered mass; controller on the nominal vehicle.
        let mut vp_plant = self.cfg.vehicle.clone();
        let j = self.cfg.mass_jitter;
        if j > 0.0 {
            vp_plant.airframe.body.mass_kg *= 1.0 + self.rng.random_range(-j..j);
        }
        let mut sim = self.cfg.sim.clone();
        sim.body_drag = self.cfg.body_drag;
        let d = self.cfg.domain_rand;
        if d > 0.0 {
            // Plant-side dynamics randomization; the controller model stays
            // nominal, so every draw is a model mismatch to be survived.
            let lu = |rng: &mut ChaCha8Rng, lo: f32, hi: f32| -> f32 {
                (rng.random_range(lo.ln()..hi.ln())).exp()
            };
            for k in [0usize, 4, 8] {
                vp_plant.airframe.body.inertia_kg_m2[k] *= 1.0 + d * (lu(&mut self.rng, 0.7, 1.5) - 1.0);
            }
            let thrust_common = 1.0 + d * (self.rng.random_range(-0.15..0.15f32));
            let tau_f = 1.0 + d * (lu(&mut self.rng, 0.5, 2.0) - 1.0);
            for m in vp_plant.airframe.motors.iter_mut() {
                m.max_thrust_n *= thrust_common * (1.0 + d * self.rng.random_range(-0.05..0.05f32));
                m.time_const_s *= tau_f;
            }
            let rotor_drag_f = 1.0 + d * (lu(&mut self.rng, 0.5, 3.0) - 1.0);
            for c in sim.aero_drag.iter_mut() {
                *c *= rotor_drag_f;
            }
            let body_drag_f = 1.0 + d * (lu(&mut self.rng, 0.5, 2.0) - 1.0);
            for c in sim.body_drag.iter_mut() {
                *c *= body_drag_f;
            }
            self.loc_sigma = [
                d * self.rng.random_range(0.0..0.05f32),
                d * self.rng.random_range(0.0..0.20f32),
                d * self.rng.random_range(0.0..0.052f32),
                d * self.rng.random_range(0.0..0.10f32),
            ];
            // Wind: a per-episode mean force (direction mostly horizontal)
            // plus an OU gust evolved per control tick in `step`. Applied
            // to the plant in the WORLD frame and modeled by nothing in
            // the controller — pure disturbance rejection.
            let mag = d * self.rng.random_range(0.0..WIND_FORCE_MAX_N);
            let mut dir = Vector3::new(
                self.rng.random_range(-1.0..1.0f32),
                self.rng.random_range(-1.0..1.0f32),
                WIND_VERTICAL_FRAC * self.rng.random_range(-1.0..1.0f32),
            );
            if dir.norm() < 1e-3 {
                dir = Vector3::x();
            }
            self.wind_mean = dir / dir.norm() * mag;
            self.wind_sigma = WIND_GUST_FRAC * mag;
            self.wind_gust = Vector3::zeros();
        } else {
            self.loc_sigma = [0.0; 4];
            self.wind_mean = Vector3::zeros();
            self.wind_sigma = 0.0;
            self.wind_gust = Vector3::zeros();
        }
        self.loc_bias = Vector3::zeros();
        self.plant = QuadPlant::new(vp_plant, &sim, self.cfg.dt_sim);
        self.ctrl = Self::build_controller(&self.cfg);
        let mut sp = self.cfg.vehicle.trajectory.sampler.to_position_sampler_params();
        sp.max_lag_s = self.cfg.sampler_max_lag_s;
        self.ctrl.attach_position_sampler(self.traj.clone(), sp);

        // Initial state = reference head + independent disturbances. Each
        // magnitude is log-uniform (small errors are common, not drowned
        // out) and each channel is zero 30 % of the time.
        let grav = env.grav;
        let p0 = self.traj.get_pos(0.0);
        let v0 = self.traj.get_vel(0.0);
        let a0 = self.traj.get_acc(0.0);
        let zb = {
            let a = Vector3::new(a0.x, a0.y, a0.z + grav);
            a / a.norm().max(1e-6)
        };
        let q0 = cybflight_core::rotation::quaternion_from_zb_and_yaw(&zb, 0.0, true);
        let d = self.cfg.disturb;
        let mut mag = |lo: f32, hi: f32| -> f32 {
            if d <= 0.0 || self.rng.random_bool(0.3) {
                0.0
            } else {
                d * self.rng.random_range(lo.ln()..hi.ln()).exp()
            }
        };
        let dp = mag(0.02, 1.5);
        // Velocity disturbance to a quarter of the envelope speed: large
        // velocity errors are ordinary in outdoor flight, not failures.
        let dv = mag(0.05, 15.0);
        let dth = mag(0.02, 1.2);
        let dw = mag(0.05, 5.0);
        let unit = |rng: &mut ChaCha8Rng| -> Vector3<f32> {
            let v = Vector3::new(
                rng.random_range(-1.0..1.0f32),
                rng.random_range(-1.0..1.0f32),
                rng.random_range(-1.0..1.0f32),
            );
            if v.norm() < 1e-3 { Vector3::x() } else { v / v.norm() }
        };
        let p = p0 + unit(&mut self.rng) * dp;
        let v = v0 + unit(&mut self.rng) * dv;
        let q = q0 * UnitQuaternion::from_scaled_axis(unit(&mut self.rng) * dth);
        let w = unit(&mut self.rng) * dw;
        self.plant.reset(p, v, q);
        self.plant.set_body_rate(w);

        self.tick = 0;
        self.last_z = [0.0; NZ];
        self.imu = NoisyImu::isotropic(self.rng.random::<u64>(), self.cfg.gyro_sigma, self.cfg.accel_sigma);
        self.last_rate_cmd = Vector3::zeros();
        self.observe()
    }

    /// Apply `z` for one outer-loop period. Returns `(obs, reward, done, info)`.
    pub fn step(&mut self, z: &[f32; NZ]) -> ([f32; OBS_DIM], f32, bool, StepInfo) {
        let zc: [f32; NZ] = core::array::from_fn(|i| {
            if z[i].is_finite() { z[i].clamp(-1.0, 1.0) } else { 0.0 }
        });
        self.ctrl.set_cost_z(&zc);
        // OU gust update per control tick (`α = e^{−Δt/τ_w}`; the
        // stationary per-axis σ is `wind_sigma`).
        let wind_alpha = (-1.0 / (self.cfg.indi_rate_hz * WIND_GUST_TAU_S)).exp();
        let wind_s = self.wind_sigma * (1.0 - wind_alpha * wind_alpha).sqrt();
        for _ in 0..self.ticks_per_step {
            if self.wind_sigma > 0.0 || self.wind_mean != Vector3::zeros() {
                for i in 0..3 {
                    self.wind_gust[i] = wind_alpha * self.wind_gust[i] + wind_s * self.gauss();
                }
                let f = self.wind_mean + self.wind_gust;
                self.plant.set_external_force(f);
            }
            let t = self.t_now();
            self.fill_horizon(t);
            let x = self.observed_state();
            let imu = self.imu.sample(&self.plant);
            let rotor = self.rotor.sample(&self.plant);
            let u = self.ctrl.step(&x, &imu, &rotor, &self.horizon);
            for _ in 0..self.substeps {
                self.plant.step(&u);
            }
            self.tick += 1;
        }

        let t = self.t_now();
        let p = self.plant.position();
        let vel = self.plant.velocity();
        let w = self.plant.body_rate();
        let mr = self.cfg.vehicle.airframe.body.max_rate_rad_s;
        let t_ref = t.min(self.traj.total_duration());
        let e_time = (self.traj.get_pos(t_ref) - p).norm();
        let e_geom = closest_point_dist(&self.traj, p, t_ref, (self.cfg.sampler_max_lag_s + 0.1).max(0.5));
        let rate2 = (w.x / mr[0]).powi(2) + (w.y / mr[1]).powi(2) + (w.z / mr[2]).powi(2);
        let dz2 = (0..NZ).map(|i| (zc[i] - self.last_z[i]).powi(2)).sum::<f32>() / NZ as f32;
        self.last_z = zc;
        let u = self.ctrl.last_command();
        let rate_cmd = Vector3::new(u[1], u[2], u[3]);
        let du = rate_cmd - self.last_rate_cmd;
        let du2 = (du.x / mr[0]).powi(2) + (du.y / mr[1]).powi(2) + (du.z / mr[2]).powi(2);
        self.last_rate_cmd = rate_cmd;

        let rw = self.cfg.reward;
        let mut reward = -(rw.geom * e_geom * e_geom + rw.time * e_time * e_time)
            - rw.rate * rate2
            - rw.dz * dz2
            - rw.du_rate * du2;
        let finite = p.iter().all(|v| v.is_finite()) && vel.iter().all(|v| v.is_finite());
        // No speed cap: a fast reference is not a crash. Leaving the
        // trajectory by more than 3 m is what marks a runaway.
        let crashed = !finite || p.z < 0.0 || e_geom > 3.0;
        let truncated = !crashed && t >= self.duration_s;
        if crashed {
            reward -= rw.crash;
        }
        let info = StepInfo {
            e_geom,
            e_time,
            du_rate2: du2,
            tilt_rad: self.plant.tilt_rad(),
            crashed,
            truncated,
            diverged: self.ctrl.last_diverged(),
        };
        let obs = if crashed { [0.0; OBS_DIM] } else { self.observe() };
        (obs, reward, crashed || truncated, info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plant::VEHICLE;

    /// A nominal (`z = 0`) episode must run to its truncation without a
    /// crash and produce finite, bounded observations.
    #[test]
    fn nominal_episode_runs_to_truncation() {
        let (vp, sim) = VEHICLE.load();
        let mut cfg = EnvConfig::new(vp, sim);
        cfg.disturb = 0.3;
        let mut env = RecoveryEnv::new(cfg, 3);
        let obs = env.reset();
        assert!(obs.iter().all(|v| v.is_finite()));
        let mut steps = 0;
        loop {
            let (obs, r, done, info) = env.step(&[0.0; NZ]);
            assert!(r.is_finite());
            assert!(obs.iter().all(|v| v.is_finite() && v.abs() < 50.0));
            steps += 1;
            if done {
                assert!(!info.crashed, "crashed at step {steps}: {info:?}");
                break;
            }
            assert!(steps < 1000);
        }
        assert!(steps > 20);
    }
}
