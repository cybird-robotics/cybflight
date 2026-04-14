//! Persistent vehicle parameter container with manual serialization.
//!
//! On-flash layout (little-endian, 640 bytes, aligned to 32-byte flash words):
//!
//! ```text
//! [0x00]  magic:   u32 = 0x43594250 ("CYBP")
//! [0x04]  version: u32 = 7
//! [0x08]  length:  u32 = PAYLOAD_SIZE (624)
//! [0x0C]  crc32:   u32 (over payload only)
//! [0x10]  payload: 624 bytes
//!   Body:               mass(4) + inertia(36) + max_rate(12) = 52 bytes
//!   Motors (x4):        px(4) + py(4) + spin_dir(4) + max_thrust(4) + torque_coeff(4) = 80 bytes
//!   Control gains:      pos_kp(12) + pos_kd(12) + att_k_rate(12) + rate_kp(12) + rate_ki(12) + rate_kd(12) = 72 bytes
//!   INDI effectiveness: g1_force(48) + g1_torque(48) + g2(48) + max_omega(16) + time_const(16) + nonlinearity(16) = 192 bytes
//!   INDI controller:    rate_gains(12) + sync_filter_hz(4) + wls_wv(24) + wls_wu(16) + motor_pole_count(4 as f32) = 60 bytes
//!   Learner:            fx_filt_hz(4) + motor_filt_hz(4) + acc_offset_m(12) + rls_gamma(4) + rls_t_char_s(4) + zeta_rate(4) + zeta_attitude(4) = 36 bytes
//!   MpcParams:          pos(12) + vel(12) + att(12) + rate(12) + thrust(4) + dt(4) + rho(4) = 60 bytes
//!   PlannerParams:      max_vel_m_s(4) + max_tilt_rad(4) + weight_time(4) + weight_energy(4) + weight_pos(4) + weight_vel(4) + weight_tilt(4) + weight_body_rate(4) + weight_thrust(4) + smoothing_eps(4) + num_check_per_piece(4 as f32) = 44 bytes
//!   BfgsTrustParams:    delta_init(4) + delta_max(4) + eta(4) + g_epsilon(4) + max_iterations(4 as f32) + past(4 as f32) + delta_conv(4) = 28 bytes
//! [0x280] padding: 0 bytes
//! ```

use crate::mixer::{MotorParams, RigidBodyParams, SpinDir};

const MAGIC: u32 = 0x4359_4250; // "CYBP"
const VERSION: u32 = 7;
const HEADER_SIZE: usize = 16; // magic + version + length + crc
/// Total payload: 52 + 80 + 72 + 192 + 60 + 36 + 60 + 44 + 28 = 624 bytes
const PAYLOAD_SIZE: usize = 624;
/// Padded to 32-byte flash word boundary: ceil((16+624)/32)*32 = 640
pub const PADDED_SIZE: usize = 640;

/// MPC tuning parameters: cost weights, discretization, and constraint penalty.
///
/// Bundles the entire MPC tuning surface into one struct, mirroring the
/// `IndiControllerParams` precedent (which similarly groups gains, filter
/// cutoffs, and cost weights).
#[derive(Clone, Debug)]
pub struct MpcParams {
    /// Position tracking weights [x, y, z].
    pub pos_weight: [f32; 3],
    /// Velocity tracking weights [x, y, z].
    pub vel_weight: [f32; 3],
    /// Attitude tracking weights [roll, pitch, yaw].
    pub att_weight: [f32; 3],
    /// Body-rate tracking weights [roll, pitch, yaw].
    pub rate_weight: [f32; 3],
    /// Control effort weight (uniform across motors).
    pub thrust_weight: f32,
    /// Integration timestep [s] for the prediction horizon.
    pub dt: f32,
    /// Cubic constraint penalty weight (input bound enforcement).
    pub rho: f32,
}

impl Default for MpcParams {
    fn default() -> Self {
        Self {
            pos_weight: [200.0, 200.0, 200.0],
            vel_weight: [1.0, 1.0, 1.0],
            att_weight: [5.0, 5.0, 200.0],
            rate_weight: [1.0, 1.0, 1.0],
            thrust_weight: 6.0,
            dt: 0.05,
            rho: 1e4,
        }
    }
}

/// BFGS trust-region parameters.
#[derive(Clone, Debug)]
pub struct BfgsTrustParams {
    pub delta_init: f32,
    pub delta_max: f32,
    /// Acceptance threshold: accept step if actual/predicted > eta.
    pub eta: f32,
    pub g_epsilon: f32,
    pub max_iterations: usize,
    /// Delta-based convergence: check cost stagnation over `past` iterations.
    pub past: usize,
    pub delta_conv: f32,
}

impl Default for BfgsTrustParams {
    fn default() -> Self {
        Self {
            delta_init: 1.0,
            delta_max: 100.0,
            eta: 0.1,
            // Tight enough to avoid premature termination in multi-waypoint
            // trajectory optimization problems (empirically the loose 1e-5
            // caused non-monotonic weight-time behavior).
            g_epsilon: 1.0e-7,
            max_iterations: 500,
            past: 3,
            delta_conv: 1.0e-8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlannerParams {
    pub max_vel_m_s: f32,
    pub max_tilt_rad: f32,
    /// Weight on total trajectory time Σ T_i (higher → faster trajectories).
    pub weight_time: f32,
    /// Weight on energy (∫‖jerk‖² or ∫‖snap‖²) — controls smoothness.
    pub weight_energy: f32,
    /// Weight on position constraint penalty
    pub weight_pos: f32,
    /// Weight on velocity constraint penalty (soft ‖v‖ ≤ max_vel).
    pub weight_vel: f32,
    /// Weight on tilt angle penalty. Set to 0 to disable.
    pub weight_tilt: f32,
    /// Weight on body rate penalty. Set to 0 to disable.
    pub weight_body_rate: f32,
    /// Weight on thrust constraint penalty (soft thrust bounds).
    pub weight_thrust: f32,
    /// Smoothing parameter ε for the smoothed L1 penalty function.
    pub smoothing_eps: f32,
    /// Number of Gauss–Legendre sample points per piece for constraint evaluation.
    pub num_check_per_piece: usize,
    /// Which nonlinear solver backend to use.
    pub bfgs_trust: BfgsTrustParams,
}

impl Default for PlannerParams {
    fn default() -> Self {
        Self {
            max_vel_m_s: 5.0,
            max_tilt_rad: core::f32::consts::FRAC_PI_3,
            weight_time: 1.0,
            weight_energy: 0.1,
            weight_pos: 0.0,
            weight_vel: 0.0,
            weight_tilt: 0.0,
            weight_body_rate: 10.0,
            weight_thrust: 10.0,
            smoothing_eps: 0.01,
            num_check_per_piece: 8,
            bfgs_trust: BfgsTrustParams::default(),
        }
    }
}

/// PID gain triplet.
#[derive(Clone, Copy, Debug)]
pub struct PidGains {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
}

/// Control gains for position, attitude, and rate loops.
///
/// All vector gains are dimension-major: `[roll/x, pitch/y, yaw/z]`.
#[derive(Clone, Debug)]
pub struct ControlGains {
    /// Position proportional gains [x, y, z].
    pub pos_kp: [f32; 3],
    /// Position derivative (velocity) gains [x, y, z].
    pub pos_kd: [f32; 3],
    /// Attitude error to body-rate gains [roll, pitch, yaw].
    pub att_k_rate: [f32; 3],
    /// Rate PID proportional gains [roll, pitch, yaw].
    pub rate_kp: [f32; 3],
    /// Rate PID integral gains [roll, pitch, yaw].
    pub rate_ki: [f32; 3],
    /// Rate PID derivative gains [roll, pitch, yaw].
    pub rate_kd: [f32; 3],
}

impl ControlGains {
    /// Extract per-axis PID gains for the rate controller.
    pub fn rate_pid(&self, axis: usize) -> PidGains {
        PidGains {
            kp: self.rate_kp[axis],
            ki: self.rate_ki[axis],
            kd: self.rate_kd[axis],
        }
    }
}

/// INDI effectiveness parameters (learned or manually configured).
/// G1 is stored per-motor, per-axis. G2 is per-motor, torque-axes only.
///
/// Default is all zeros, meaning "not configured / use geometric fallback".
#[derive(Clone, Debug)]
pub struct IndiEffectivenessParams {
    /// G1 force effectiveness per motor [fx, fy, fz] -- 4 motors x 3 axes = 12 values
    pub g1_force: [[f32; 3]; 4],
    /// G1 torque effectiveness per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values
    pub g1_torque: [[f32; 3]; 4],
    /// G2 gyroscopic coupling per motor [roll, pitch, yaw] -- 4 motors x 3 axes = 12 values
    pub g2: [[f32; 3]; 4],
    /// Motor max speed per motor (rad/s) -- 4 values
    pub max_omega: [f32; 4],
    /// Motor time constant per motor (seconds) -- 4 values
    pub time_const_s: [f32; 4],
    /// Motor nonlinearity per motor [0,1] -- 4 values
    pub nonlinearity: [f32; 4],
}

impl Default for IndiEffectivenessParams {
    fn default() -> Self {
        Self {
            g1_force: [[0.0; 3]; 4],
            g1_torque: [[0.0; 3]; 4],
            g2: [[0.0; 3]; 4],
            max_omega: [0.0; 4],
            time_const_s: [0.0; 4],
            nonlinearity: [0.0; 4],
        }
    }
}

/// INDI controller tuning parameters.
#[derive(Clone, Debug)]
pub struct IndiControllerParams {
    /// Rate error -> angular acceleration gains [roll, pitch, yaw] (rad/s^2 per rad/s).
    pub rate_gains: [f32; 3],
    /// Biquad low-pass cutoff for synchronized filters (Hz).
    pub sync_filter_hz: f32,
    /// WLS pseudo-control weights [fx, fy, fz, roll, pitch, yaw].
    pub wls_wv: [f32; 6],
    /// WLS actuator penalty weights [m0, m1, m2, m3].
    pub wls_wu: [f32; 4],
    /// Motor pole count (for eRPM -> RPM conversion).
    pub motor_pole_count: u8,
}

impl Default for IndiControllerParams {
    fn default() -> Self {
        Self {
            rate_gains: [20.0, 20.0, 20.0],
            sync_filter_hz: 5.0,
            wls_wv: [1.0, 1.0, 50.0, 50.0, 50.0, 5.0],
            wls_wu: [1.0, 1.0, 1.0, 1.0],
            motor_pole_count: 14,
        }
    }
}

/// G1/G2 learner tuning parameters.
#[derive(Clone, Debug)]
pub struct LearnerParams {
    /// Biquad cutoff for effectiveness learning filters (Hz).
    pub fx_filt_hz: f32,
    /// Biquad cutoff for motor dynamics learning filters (Hz).
    pub motor_filt_hz: f32,
    /// IMU offset from CoG [x, y, z] (metres).
    pub acc_offset_m: [f32; 3],
    /// RLS initial covariance diagonal.
    pub rls_gamma: f32,
    /// RLS characteristic forgetting time (seconds).
    pub rls_t_char_s: f32,
    /// Rate loop damping ratio for gain synthesis.
    pub zeta_rate: f32,
    /// Attitude loop damping ratio for gain synthesis.
    pub zeta_attitude: f32,
}

impl Default for LearnerParams {
    fn default() -> Self {
        Self {
            fx_filt_hz: 20.0,
            motor_filt_hz: 40.0,
            acc_offset_m: [0.0, 0.0, 0.0],
            rls_gamma: 100.0,
            rls_t_char_s: 0.25,
            zeta_rate: 0.8,
            zeta_attitude: 0.8,
        }
    }
}

/// Full vehicle parameter set.
#[derive(Clone, Debug)]
pub struct VehicleParams {
    pub body: RigidBodyParams,
    pub motors: [MotorParams; 4],
    pub control: ControlGains,
    pub indi_effectiveness: IndiEffectivenessParams,
    pub indi_controller: IndiControllerParams,
    pub learner: LearnerParams,
    pub mpc: MpcParams,
    pub planner: PlannerParams,
}

impl VehicleParams {
    /// Serialize to a flash-ready buffer with header and CRC.
    pub fn to_bytes(&self) -> [u8; PADDED_SIZE] {
        let mut buf = [0u8; PADDED_SIZE];
        // Write payload first (at offset HEADER_SIZE)
        let mut off = HEADER_SIZE;
        off = put_f32(&mut buf, off, self.body.mass_kg);
        for &v in &self.body.inertia_kg_m2 {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.body.max_rate_rad_s {
            off = put_f32(&mut buf, off, v);
        }
        for m in &self.motors {
            off = put_f32(&mut buf, off, m.position_m[0]);
            off = put_f32(&mut buf, off, m.position_m[1]);
            off = put_u32(
                &mut buf,
                off,
                match m.spin_dir {
                    SpinDir::Cw => 1,
                    SpinDir::Ccw => 0,
                },
            );
            off = put_f32(&mut buf, off, m.max_thrust_n);
            off = put_f32(&mut buf, off, m.torque_coeff_m);
        }
        for &v in &self.control.pos_kp {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.pos_kd {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.att_k_rate {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.rate_kp {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.rate_ki {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.rate_kd {
            off = put_f32(&mut buf, off, v);
        }
        // INDI effectiveness
        for m in &self.indi_effectiveness.g1_force {
            for &v in m {
                off = put_f32(&mut buf, off, v);
            }
        }
        for m in &self.indi_effectiveness.g1_torque {
            for &v in m {
                off = put_f32(&mut buf, off, v);
            }
        }
        for m in &self.indi_effectiveness.g2 {
            for &v in m {
                off = put_f32(&mut buf, off, v);
            }
        }
        for &v in &self.indi_effectiveness.max_omega {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.indi_effectiveness.time_const_s {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.indi_effectiveness.nonlinearity {
            off = put_f32(&mut buf, off, v);
        }
        // INDI controller
        for &v in &self.indi_controller.rate_gains {
            off = put_f32(&mut buf, off, v);
        }
        off = put_f32(&mut buf, off, self.indi_controller.sync_filter_hz);
        for &v in &self.indi_controller.wls_wv {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.indi_controller.wls_wu {
            off = put_f32(&mut buf, off, v);
        }
        off = put_f32(&mut buf, off, self.indi_controller.motor_pole_count as f32);
        // Learner
        off = put_f32(&mut buf, off, self.learner.fx_filt_hz);
        off = put_f32(&mut buf, off, self.learner.motor_filt_hz);
        for &v in &self.learner.acc_offset_m {
            off = put_f32(&mut buf, off, v);
        }
        off = put_f32(&mut buf, off, self.learner.rls_gamma);
        off = put_f32(&mut buf, off, self.learner.rls_t_char_s);
        off = put_f32(&mut buf, off, self.learner.zeta_rate);
        off = put_f32(&mut buf, off, self.learner.zeta_attitude);
        // MpcParams
        for &v in &self.mpc.pos_weight {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.mpc.vel_weight {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.mpc.att_weight {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.mpc.rate_weight {
            off = put_f32(&mut buf, off, v);
        }
        off = put_f32(&mut buf, off, self.mpc.thrust_weight);
        off = put_f32(&mut buf, off, self.mpc.dt);
        off = put_f32(&mut buf, off, self.mpc.rho);
        // PlannerParams
        off = put_f32(&mut buf, off, self.planner.max_vel_m_s);
        off = put_f32(&mut buf, off, self.planner.max_tilt_rad);
        off = put_f32(&mut buf, off, self.planner.weight_time);
        off = put_f32(&mut buf, off, self.planner.weight_energy);
        off = put_f32(&mut buf, off, self.planner.weight_pos);
        off = put_f32(&mut buf, off, self.planner.weight_vel);
        off = put_f32(&mut buf, off, self.planner.weight_tilt);
        off = put_f32(&mut buf, off, self.planner.weight_body_rate);
        off = put_f32(&mut buf, off, self.planner.weight_thrust);
        off = put_f32(&mut buf, off, self.planner.smoothing_eps);
        off = put_f32(&mut buf, off, self.planner.num_check_per_piece as f32);
        // BfgsTrustParams (nested in planner)
        off = put_f32(&mut buf, off, self.planner.bfgs_trust.delta_init);
        off = put_f32(&mut buf, off, self.planner.bfgs_trust.delta_max);
        off = put_f32(&mut buf, off, self.planner.bfgs_trust.eta);
        off = put_f32(&mut buf, off, self.planner.bfgs_trust.g_epsilon);
        off = put_f32(&mut buf, off, self.planner.bfgs_trust.max_iterations as f32);
        off = put_f32(&mut buf, off, self.planner.bfgs_trust.past as f32);
        off = put_f32(&mut buf, off, self.planner.bfgs_trust.delta_conv);
        debug_assert_eq!(off - HEADER_SIZE, PAYLOAD_SIZE);

        // Header
        let payload = &buf[HEADER_SIZE..HEADER_SIZE + PAYLOAD_SIZE];
        let crc = crc32fast::hash(payload);
        put_u32(&mut buf, 0, MAGIC);
        put_u32(&mut buf, 4, VERSION);
        put_u32(&mut buf, 8, PAYLOAD_SIZE as u32);
        put_u32(&mut buf, 12, crc);
        buf
    }

    /// Deserialize from a flash buffer. Returns `None` if magic, version, or CRC mismatch.
    pub fn from_bytes(buf: &[u8; PADDED_SIZE]) -> Option<Self> {
        let magic = get_u32(buf, 0);
        let version = get_u32(buf, 4);
        let length = get_u32(buf, 8);
        let stored_crc = get_u32(buf, 12);

        if magic != MAGIC || version != VERSION || length as usize != PAYLOAD_SIZE {
            return None;
        }

        let payload = &buf[HEADER_SIZE..HEADER_SIZE + PAYLOAD_SIZE];
        if crc32fast::hash(payload) != stored_crc {
            return None;
        }

        let mut off = HEADER_SIZE;
        let mass_kg = get_f32(buf, off);
        off += 4;
        let mut inertia_kg_m2 = [0.0f32; 9];
        for slot in &mut inertia_kg_m2 {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut max_rate_rad_s = [0.0f32; 3];
        for slot in &mut max_rate_rad_s {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let body = RigidBodyParams {
            mass_kg,
            inertia_kg_m2,
            max_rate_rad_s,
        };

        let mut motors = [MotorParams {
            position_m: [0.0; 2],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 0.0,
            torque_coeff_m: 0.0,
        }; 4];
        for m in &mut motors {
            let px = get_f32(buf, off);
            off += 4;
            let py = get_f32(buf, off);
            off += 4;
            let dir_val = get_u32(buf, off);
            off += 4;
            let max_thrust = get_f32(buf, off);
            off += 4;
            let torque_coeff = get_f32(buf, off);
            off += 4;
            m.position_m = [px, py];
            m.spin_dir = if dir_val == 1 {
                SpinDir::Cw
            } else {
                SpinDir::Ccw
            };
            m.max_thrust_n = max_thrust;
            m.torque_coeff_m = torque_coeff;
        }

        let mut pos_kp = [0.0f32; 3];
        for slot in &mut pos_kp {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut pos_kd = [0.0f32; 3];
        for slot in &mut pos_kd {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut att_k_rate = [0.0f32; 3];
        for slot in &mut att_k_rate {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut rate_kp = [0.0f32; 3];
        for slot in &mut rate_kp {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut rate_ki = [0.0f32; 3];
        for slot in &mut rate_ki {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut rate_kd = [0.0f32; 3];
        for slot in &mut rate_kd {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let control = ControlGains {
            pos_kp,
            pos_kd,
            att_k_rate,
            rate_kp,
            rate_ki,
            rate_kd,
        };

        // INDI effectiveness
        let mut g1_force = [[0.0f32; 3]; 4];
        for m in &mut g1_force {
            for slot in m.iter_mut() {
                *slot = get_f32(buf, off);
                off += 4;
            }
        }
        let mut g1_torque = [[0.0f32; 3]; 4];
        for m in &mut g1_torque {
            for slot in m.iter_mut() {
                *slot = get_f32(buf, off);
                off += 4;
            }
        }
        let mut g2 = [[0.0f32; 3]; 4];
        for m in &mut g2 {
            for slot in m.iter_mut() {
                *slot = get_f32(buf, off);
                off += 4;
            }
        }
        let mut max_omega = [0.0f32; 4];
        for slot in &mut max_omega {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut time_const_s = [0.0f32; 4];
        for slot in &mut time_const_s {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut nonlinearity = [0.0f32; 4];
        for slot in &mut nonlinearity {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let indi_effectiveness = IndiEffectivenessParams {
            g1_force,
            g1_torque,
            g2,
            max_omega,
            time_const_s,
            nonlinearity,
        };

        // INDI controller
        let mut rate_gains = [0.0f32; 3];
        for slot in &mut rate_gains {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let sync_filter_hz = get_f32(buf, off);
        off += 4;
        let mut wls_wv = [0.0f32; 6];
        for slot in &mut wls_wv {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut wls_wu = [0.0f32; 4];
        for slot in &mut wls_wu {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let motor_pole_count = get_f32(buf, off) as u8;
        off += 4;
        let indi_controller = IndiControllerParams {
            rate_gains,
            sync_filter_hz,
            wls_wv,
            wls_wu,
            motor_pole_count,
        };

        // Learner
        let fx_filt_hz = get_f32(buf, off);
        off += 4;
        let motor_filt_hz = get_f32(buf, off);
        off += 4;
        let mut acc_offset_m = [0.0f32; 3];
        for slot in &mut acc_offset_m {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let rls_gamma = get_f32(buf, off);
        off += 4;
        let rls_t_char_s = get_f32(buf, off);
        off += 4;
        let zeta_rate = get_f32(buf, off);
        off += 4;
        let zeta_attitude = get_f32(buf, off);
        off += 4;
        let learner = LearnerParams {
            fx_filt_hz,
            motor_filt_hz,
            acc_offset_m,
            rls_gamma,
            rls_t_char_s,
            zeta_rate,
            zeta_attitude,
        };

        // MpcParams
        let mut pos_weight = [0.0f32; 3];
        for slot in &mut pos_weight {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut vel_weight = [0.0f32; 3];
        for slot in &mut vel_weight {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut att_weight = [0.0f32; 3];
        for slot in &mut att_weight {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut rate_weight = [0.0f32; 3];
        for slot in &mut rate_weight {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let thrust_weight = get_f32(buf, off);
        off += 4;
        let dt = get_f32(buf, off);
        off += 4;
        let rho = get_f32(buf, off);
        off += 4;
        let mpc = MpcParams {
            pos_weight,
            vel_weight,
            att_weight,
            rate_weight,
            thrust_weight,
            dt,
            rho,
        };

        // PlannerParams
        let max_vel_m_s = get_f32(buf, off);
        off += 4;
        let max_tilt_rad = get_f32(buf, off);
        off += 4;
        let weight_time = get_f32(buf, off);
        off += 4;
        let weight_energy = get_f32(buf, off);
        off += 4;
        let weight_pos = get_f32(buf, off);
        off += 4;
        let weight_vel = get_f32(buf, off);
        off += 4;
        let weight_tilt = get_f32(buf, off);
        off += 4;
        let weight_body_rate = get_f32(buf, off);
        off += 4;
        let weight_thrust = get_f32(buf, off);
        off += 4;
        let smoothing_eps = get_f32(buf, off);
        off += 4;
        let num_check_per_piece = get_f32(buf, off) as usize;
        off += 4;
        // BfgsTrustParams (nested in planner)
        let delta_init = get_f32(buf, off);
        off += 4;
        let delta_max = get_f32(buf, off);
        off += 4;
        let eta = get_f32(buf, off);
        off += 4;
        let g_epsilon = get_f32(buf, off);
        off += 4;
        let max_iterations = get_f32(buf, off) as usize;
        off += 4;
        let past = get_f32(buf, off) as usize;
        off += 4;
        let delta_conv = get_f32(buf, off);
        off += 4;
        let planner = PlannerParams {
            max_vel_m_s,
            max_tilt_rad,
            weight_time,
            weight_energy,
            weight_pos,
            weight_vel,
            weight_tilt,
            weight_body_rate,
            weight_thrust,
            smoothing_eps,
            num_check_per_piece,
            bfgs_trust: BfgsTrustParams {
                delta_init,
                delta_max,
                eta,
                g_epsilon,
                max_iterations,
                past,
                delta_conv,
            },
        };

        let _ = off; // suppress unused warning

        Some(VehicleParams {
            body,
            motors,
            control,
            indi_effectiveness,
            indi_controller,
            learner,
            mpc,
            planner,
        })
    }

    /// Get a parameter value by key.
    pub fn get(&self, key: ParamKey) -> f32 {
        match key {
            ParamKey::Mass => self.body.mass_kg,
            ParamKey::Ixx => self.body.inertia_kg_m2[0],
            ParamKey::Ixy => self.body.inertia_kg_m2[1],
            ParamKey::Ixz => self.body.inertia_kg_m2[2],
            ParamKey::Iyx => self.body.inertia_kg_m2[3],
            ParamKey::Iyy => self.body.inertia_kg_m2[4],
            ParamKey::Iyz => self.body.inertia_kg_m2[5],
            ParamKey::Izx => self.body.inertia_kg_m2[6],
            ParamKey::Izy => self.body.inertia_kg_m2[7],
            ParamKey::Izz => self.body.inertia_kg_m2[8],
            ParamKey::M0Px => self.motors[0].position_m[0],
            ParamKey::M0Py => self.motors[0].position_m[1],
            ParamKey::M0Spin => self.motors[0].spin_dir as i32 as f32,
            ParamKey::M0Thrust => self.motors[0].max_thrust_n,
            ParamKey::M0Torque => self.motors[0].torque_coeff_m,
            ParamKey::M1Px => self.motors[1].position_m[0],
            ParamKey::M1Py => self.motors[1].position_m[1],
            ParamKey::M1Spin => self.motors[1].spin_dir as i32 as f32,
            ParamKey::M1Thrust => self.motors[1].max_thrust_n,
            ParamKey::M1Torque => self.motors[1].torque_coeff_m,
            ParamKey::M2Px => self.motors[2].position_m[0],
            ParamKey::M2Py => self.motors[2].position_m[1],
            ParamKey::M2Spin => self.motors[2].spin_dir as i32 as f32,
            ParamKey::M2Thrust => self.motors[2].max_thrust_n,
            ParamKey::M2Torque => self.motors[2].torque_coeff_m,
            ParamKey::M3Px => self.motors[3].position_m[0],
            ParamKey::M3Py => self.motors[3].position_m[1],
            ParamKey::M3Spin => self.motors[3].spin_dir as i32 as f32,
            ParamKey::M3Thrust => self.motors[3].max_thrust_n,
            ParamKey::M3Torque => self.motors[3].torque_coeff_m,
            ParamKey::PosKpX => self.control.pos_kp[0],
            ParamKey::PosKpY => self.control.pos_kp[1],
            ParamKey::PosKpZ => self.control.pos_kp[2],
            ParamKey::PosKdX => self.control.pos_kd[0],
            ParamKey::PosKdY => self.control.pos_kd[1],
            ParamKey::PosKdZ => self.control.pos_kd[2],
            ParamKey::AttKrX => self.control.att_k_rate[0],
            ParamKey::AttKrY => self.control.att_k_rate[1],
            ParamKey::AttKrZ => self.control.att_k_rate[2],
            ParamKey::RateKpR => self.control.rate_kp[0],
            ParamKey::RateKpP => self.control.rate_kp[1],
            ParamKey::RateKpY => self.control.rate_kp[2],
            ParamKey::RateKiR => self.control.rate_ki[0],
            ParamKey::RateKiP => self.control.rate_ki[1],
            ParamKey::RateKiY => self.control.rate_ki[2],
            ParamKey::RateKdR => self.control.rate_kd[0],
            ParamKey::RateKdP => self.control.rate_kd[1],
            ParamKey::RateKdY => self.control.rate_kd[2],
            // INDI effectiveness — G1 force
            ParamKey::G1FxM0 => self.indi_effectiveness.g1_force[0][0],
            ParamKey::G1FxM1 => self.indi_effectiveness.g1_force[1][0],
            ParamKey::G1FxM2 => self.indi_effectiveness.g1_force[2][0],
            ParamKey::G1FxM3 => self.indi_effectiveness.g1_force[3][0],
            ParamKey::G1FyM0 => self.indi_effectiveness.g1_force[0][1],
            ParamKey::G1FyM1 => self.indi_effectiveness.g1_force[1][1],
            ParamKey::G1FyM2 => self.indi_effectiveness.g1_force[2][1],
            ParamKey::G1FyM3 => self.indi_effectiveness.g1_force[3][1],
            ParamKey::G1FzM0 => self.indi_effectiveness.g1_force[0][2],
            ParamKey::G1FzM1 => self.indi_effectiveness.g1_force[1][2],
            ParamKey::G1FzM2 => self.indi_effectiveness.g1_force[2][2],
            ParamKey::G1FzM3 => self.indi_effectiveness.g1_force[3][2],
            // INDI effectiveness — G1 torque
            ParamKey::G1RrM0 => self.indi_effectiveness.g1_torque[0][0],
            ParamKey::G1RrM1 => self.indi_effectiveness.g1_torque[1][0],
            ParamKey::G1RrM2 => self.indi_effectiveness.g1_torque[2][0],
            ParamKey::G1RrM3 => self.indi_effectiveness.g1_torque[3][0],
            ParamKey::G1RpM0 => self.indi_effectiveness.g1_torque[0][1],
            ParamKey::G1RpM1 => self.indi_effectiveness.g1_torque[1][1],
            ParamKey::G1RpM2 => self.indi_effectiveness.g1_torque[2][1],
            ParamKey::G1RpM3 => self.indi_effectiveness.g1_torque[3][1],
            ParamKey::G1RyM0 => self.indi_effectiveness.g1_torque[0][2],
            ParamKey::G1RyM1 => self.indi_effectiveness.g1_torque[1][2],
            ParamKey::G1RyM2 => self.indi_effectiveness.g1_torque[2][2],
            ParamKey::G1RyM3 => self.indi_effectiveness.g1_torque[3][2],
            // INDI effectiveness — G2
            ParamKey::G2RrM0 => self.indi_effectiveness.g2[0][0],
            ParamKey::G2RrM1 => self.indi_effectiveness.g2[1][0],
            ParamKey::G2RrM2 => self.indi_effectiveness.g2[2][0],
            ParamKey::G2RrM3 => self.indi_effectiveness.g2[3][0],
            ParamKey::G2RpM0 => self.indi_effectiveness.g2[0][1],
            ParamKey::G2RpM1 => self.indi_effectiveness.g2[1][1],
            ParamKey::G2RpM2 => self.indi_effectiveness.g2[2][1],
            ParamKey::G2RpM3 => self.indi_effectiveness.g2[3][1],
            ParamKey::G2RyM0 => self.indi_effectiveness.g2[0][2],
            ParamKey::G2RyM1 => self.indi_effectiveness.g2[1][2],
            ParamKey::G2RyM2 => self.indi_effectiveness.g2[2][2],
            ParamKey::G2RyM3 => self.indi_effectiveness.g2[3][2],
            // INDI motor dynamics
            ParamKey::IndiOmegaM0 => self.indi_effectiveness.max_omega[0],
            ParamKey::IndiOmegaM1 => self.indi_effectiveness.max_omega[1],
            ParamKey::IndiOmegaM2 => self.indi_effectiveness.max_omega[2],
            ParamKey::IndiOmegaM3 => self.indi_effectiveness.max_omega[3],
            ParamKey::IndiTauM0 => self.indi_effectiveness.time_const_s[0],
            ParamKey::IndiTauM1 => self.indi_effectiveness.time_const_s[1],
            ParamKey::IndiTauM2 => self.indi_effectiveness.time_const_s[2],
            ParamKey::IndiTauM3 => self.indi_effectiveness.time_const_s[3],
            ParamKey::IndiNonlinM0 => self.indi_effectiveness.nonlinearity[0],
            ParamKey::IndiNonlinM1 => self.indi_effectiveness.nonlinearity[1],
            ParamKey::IndiNonlinM2 => self.indi_effectiveness.nonlinearity[2],
            ParamKey::IndiNonlinM3 => self.indi_effectiveness.nonlinearity[3],
            // INDI controller
            ParamKey::IndiRateR => self.indi_controller.rate_gains[0],
            ParamKey::IndiRateP => self.indi_controller.rate_gains[1],
            ParamKey::IndiRateY => self.indi_controller.rate_gains[2],
            ParamKey::IndiSyncHz => self.indi_controller.sync_filter_hz,
            ParamKey::WlsWvFx => self.indi_controller.wls_wv[0],
            ParamKey::WlsWvFy => self.indi_controller.wls_wv[1],
            ParamKey::WlsWvFz => self.indi_controller.wls_wv[2],
            ParamKey::WlsWvRr => self.indi_controller.wls_wv[3],
            ParamKey::WlsWvRp => self.indi_controller.wls_wv[4],
            ParamKey::WlsWvRy => self.indi_controller.wls_wv[5],
            ParamKey::WlsWuM0 => self.indi_controller.wls_wu[0],
            ParamKey::WlsWuM1 => self.indi_controller.wls_wu[1],
            ParamKey::WlsWuM2 => self.indi_controller.wls_wu[2],
            ParamKey::WlsWuM3 => self.indi_controller.wls_wu[3],
            ParamKey::MotorPoles => self.indi_controller.motor_pole_count as f32,
            // Learner
            ParamKey::LearnFxHz => self.learner.fx_filt_hz,
            ParamKey::LearnMotorHz => self.learner.motor_filt_hz,
            ParamKey::LearnAccX => self.learner.acc_offset_m[0],
            ParamKey::LearnAccY => self.learner.acc_offset_m[1],
            ParamKey::LearnAccZ => self.learner.acc_offset_m[2],
            ParamKey::LearnGamma => self.learner.rls_gamma,
            ParamKey::LearnTchar => self.learner.rls_t_char_s,
            ParamKey::LearnZetaRate => self.learner.zeta_rate,
            ParamKey::LearnZetaAtt => self.learner.zeta_attitude,
        }
    }

    /// Set a parameter value by key.
    pub fn set(&mut self, key: ParamKey, val: f32) {
        match key {
            ParamKey::Mass => self.body.mass_kg = val,
            ParamKey::Ixx => self.body.inertia_kg_m2[0] = val,
            ParamKey::Ixy => self.body.inertia_kg_m2[1] = val,
            ParamKey::Ixz => self.body.inertia_kg_m2[2] = val,
            ParamKey::Iyx => self.body.inertia_kg_m2[3] = val,
            ParamKey::Iyy => self.body.inertia_kg_m2[4] = val,
            ParamKey::Iyz => self.body.inertia_kg_m2[5] = val,
            ParamKey::Izx => self.body.inertia_kg_m2[6] = val,
            ParamKey::Izy => self.body.inertia_kg_m2[7] = val,
            ParamKey::Izz => self.body.inertia_kg_m2[8] = val,
            ParamKey::M0Px => self.motors[0].position_m[0] = val,
            ParamKey::M0Py => self.motors[0].position_m[1] = val,
            ParamKey::M0Spin => {
                self.motors[0].spin_dir = if val > 0.0 { SpinDir::Cw } else { SpinDir::Ccw }
            }
            ParamKey::M0Thrust => self.motors[0].max_thrust_n = val,
            ParamKey::M0Torque => self.motors[0].torque_coeff_m = val,
            ParamKey::M1Px => self.motors[1].position_m[0] = val,
            ParamKey::M1Py => self.motors[1].position_m[1] = val,
            ParamKey::M1Spin => {
                self.motors[1].spin_dir = if val > 0.0 { SpinDir::Cw } else { SpinDir::Ccw }
            }
            ParamKey::M1Thrust => self.motors[1].max_thrust_n = val,
            ParamKey::M1Torque => self.motors[1].torque_coeff_m = val,
            ParamKey::M2Px => self.motors[2].position_m[0] = val,
            ParamKey::M2Py => self.motors[2].position_m[1] = val,
            ParamKey::M2Spin => {
                self.motors[2].spin_dir = if val > 0.0 { SpinDir::Cw } else { SpinDir::Ccw }
            }
            ParamKey::M2Thrust => self.motors[2].max_thrust_n = val,
            ParamKey::M2Torque => self.motors[2].torque_coeff_m = val,
            ParamKey::M3Px => self.motors[3].position_m[0] = val,
            ParamKey::M3Py => self.motors[3].position_m[1] = val,
            ParamKey::M3Spin => {
                self.motors[3].spin_dir = if val > 0.0 { SpinDir::Cw } else { SpinDir::Ccw }
            }
            ParamKey::M3Thrust => self.motors[3].max_thrust_n = val,
            ParamKey::M3Torque => self.motors[3].torque_coeff_m = val,
            ParamKey::PosKpX => self.control.pos_kp[0] = val,
            ParamKey::PosKpY => self.control.pos_kp[1] = val,
            ParamKey::PosKpZ => self.control.pos_kp[2] = val,
            ParamKey::PosKdX => self.control.pos_kd[0] = val,
            ParamKey::PosKdY => self.control.pos_kd[1] = val,
            ParamKey::PosKdZ => self.control.pos_kd[2] = val,
            ParamKey::AttKrX => self.control.att_k_rate[0] = val,
            ParamKey::AttKrY => self.control.att_k_rate[1] = val,
            ParamKey::AttKrZ => self.control.att_k_rate[2] = val,
            ParamKey::RateKpR => self.control.rate_kp[0] = val,
            ParamKey::RateKpP => self.control.rate_kp[1] = val,
            ParamKey::RateKpY => self.control.rate_kp[2] = val,
            ParamKey::RateKiR => self.control.rate_ki[0] = val,
            ParamKey::RateKiP => self.control.rate_ki[1] = val,
            ParamKey::RateKiY => self.control.rate_ki[2] = val,
            ParamKey::RateKdR => self.control.rate_kd[0] = val,
            ParamKey::RateKdP => self.control.rate_kd[1] = val,
            ParamKey::RateKdY => self.control.rate_kd[2] = val,
            // INDI effectiveness — G1 force
            ParamKey::G1FxM0 => self.indi_effectiveness.g1_force[0][0] = val,
            ParamKey::G1FxM1 => self.indi_effectiveness.g1_force[1][0] = val,
            ParamKey::G1FxM2 => self.indi_effectiveness.g1_force[2][0] = val,
            ParamKey::G1FxM3 => self.indi_effectiveness.g1_force[3][0] = val,
            ParamKey::G1FyM0 => self.indi_effectiveness.g1_force[0][1] = val,
            ParamKey::G1FyM1 => self.indi_effectiveness.g1_force[1][1] = val,
            ParamKey::G1FyM2 => self.indi_effectiveness.g1_force[2][1] = val,
            ParamKey::G1FyM3 => self.indi_effectiveness.g1_force[3][1] = val,
            ParamKey::G1FzM0 => self.indi_effectiveness.g1_force[0][2] = val,
            ParamKey::G1FzM1 => self.indi_effectiveness.g1_force[1][2] = val,
            ParamKey::G1FzM2 => self.indi_effectiveness.g1_force[2][2] = val,
            ParamKey::G1FzM3 => self.indi_effectiveness.g1_force[3][2] = val,
            // INDI effectiveness — G1 torque
            ParamKey::G1RrM0 => self.indi_effectiveness.g1_torque[0][0] = val,
            ParamKey::G1RrM1 => self.indi_effectiveness.g1_torque[1][0] = val,
            ParamKey::G1RrM2 => self.indi_effectiveness.g1_torque[2][0] = val,
            ParamKey::G1RrM3 => self.indi_effectiveness.g1_torque[3][0] = val,
            ParamKey::G1RpM0 => self.indi_effectiveness.g1_torque[0][1] = val,
            ParamKey::G1RpM1 => self.indi_effectiveness.g1_torque[1][1] = val,
            ParamKey::G1RpM2 => self.indi_effectiveness.g1_torque[2][1] = val,
            ParamKey::G1RpM3 => self.indi_effectiveness.g1_torque[3][1] = val,
            ParamKey::G1RyM0 => self.indi_effectiveness.g1_torque[0][2] = val,
            ParamKey::G1RyM1 => self.indi_effectiveness.g1_torque[1][2] = val,
            ParamKey::G1RyM2 => self.indi_effectiveness.g1_torque[2][2] = val,
            ParamKey::G1RyM3 => self.indi_effectiveness.g1_torque[3][2] = val,
            // INDI effectiveness — G2
            ParamKey::G2RrM0 => self.indi_effectiveness.g2[0][0] = val,
            ParamKey::G2RrM1 => self.indi_effectiveness.g2[1][0] = val,
            ParamKey::G2RrM2 => self.indi_effectiveness.g2[2][0] = val,
            ParamKey::G2RrM3 => self.indi_effectiveness.g2[3][0] = val,
            ParamKey::G2RpM0 => self.indi_effectiveness.g2[0][1] = val,
            ParamKey::G2RpM1 => self.indi_effectiveness.g2[1][1] = val,
            ParamKey::G2RpM2 => self.indi_effectiveness.g2[2][1] = val,
            ParamKey::G2RpM3 => self.indi_effectiveness.g2[3][1] = val,
            ParamKey::G2RyM0 => self.indi_effectiveness.g2[0][2] = val,
            ParamKey::G2RyM1 => self.indi_effectiveness.g2[1][2] = val,
            ParamKey::G2RyM2 => self.indi_effectiveness.g2[2][2] = val,
            ParamKey::G2RyM3 => self.indi_effectiveness.g2[3][2] = val,
            // INDI motor dynamics
            ParamKey::IndiOmegaM0 => self.indi_effectiveness.max_omega[0] = val,
            ParamKey::IndiOmegaM1 => self.indi_effectiveness.max_omega[1] = val,
            ParamKey::IndiOmegaM2 => self.indi_effectiveness.max_omega[2] = val,
            ParamKey::IndiOmegaM3 => self.indi_effectiveness.max_omega[3] = val,
            ParamKey::IndiTauM0 => self.indi_effectiveness.time_const_s[0] = val,
            ParamKey::IndiTauM1 => self.indi_effectiveness.time_const_s[1] = val,
            ParamKey::IndiTauM2 => self.indi_effectiveness.time_const_s[2] = val,
            ParamKey::IndiTauM3 => self.indi_effectiveness.time_const_s[3] = val,
            ParamKey::IndiNonlinM0 => self.indi_effectiveness.nonlinearity[0] = val,
            ParamKey::IndiNonlinM1 => self.indi_effectiveness.nonlinearity[1] = val,
            ParamKey::IndiNonlinM2 => self.indi_effectiveness.nonlinearity[2] = val,
            ParamKey::IndiNonlinM3 => self.indi_effectiveness.nonlinearity[3] = val,
            // INDI controller
            ParamKey::IndiRateR => self.indi_controller.rate_gains[0] = val,
            ParamKey::IndiRateP => self.indi_controller.rate_gains[1] = val,
            ParamKey::IndiRateY => self.indi_controller.rate_gains[2] = val,
            ParamKey::IndiSyncHz => self.indi_controller.sync_filter_hz = val,
            ParamKey::WlsWvFx => self.indi_controller.wls_wv[0] = val,
            ParamKey::WlsWvFy => self.indi_controller.wls_wv[1] = val,
            ParamKey::WlsWvFz => self.indi_controller.wls_wv[2] = val,
            ParamKey::WlsWvRr => self.indi_controller.wls_wv[3] = val,
            ParamKey::WlsWvRp => self.indi_controller.wls_wv[4] = val,
            ParamKey::WlsWvRy => self.indi_controller.wls_wv[5] = val,
            ParamKey::WlsWuM0 => self.indi_controller.wls_wu[0] = val,
            ParamKey::WlsWuM1 => self.indi_controller.wls_wu[1] = val,
            ParamKey::WlsWuM2 => self.indi_controller.wls_wu[2] = val,
            ParamKey::WlsWuM3 => self.indi_controller.wls_wu[3] = val,
            ParamKey::MotorPoles => self.indi_controller.motor_pole_count = val as u8,
            // Learner
            ParamKey::LearnFxHz => self.learner.fx_filt_hz = val,
            ParamKey::LearnMotorHz => self.learner.motor_filt_hz = val,
            ParamKey::LearnAccX => self.learner.acc_offset_m[0] = val,
            ParamKey::LearnAccY => self.learner.acc_offset_m[1] = val,
            ParamKey::LearnAccZ => self.learner.acc_offset_m[2] = val,
            ParamKey::LearnGamma => self.learner.rls_gamma = val,
            ParamKey::LearnTchar => self.learner.rls_t_char_s = val,
            ParamKey::LearnZetaRate => self.learner.zeta_rate = val,
            ParamKey::LearnZetaAtt => self.learner.zeta_attitude = val,
        }
    }
}

/// Named parameter keys for shell access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKey {
    Mass,
    Ixx,
    Ixy,
    Ixz,
    Iyx,
    Iyy,
    Iyz,
    Izx,
    Izy,
    Izz,
    M0Px,
    M0Py,
    M0Spin,
    M0Thrust,
    M0Torque,
    M1Px,
    M1Py,
    M1Spin,
    M1Thrust,
    M1Torque,
    M2Px,
    M2Py,
    M2Spin,
    M2Thrust,
    M2Torque,
    M3Px,
    M3Py,
    M3Spin,
    M3Thrust,
    M3Torque,
    // Position control gains
    PosKpX,
    PosKpY,
    PosKpZ,
    PosKdX,
    PosKdY,
    PosKdZ,
    // Attitude control gains
    AttKrX,
    AttKrY,
    AttKrZ,
    // Rate PID gains (dimension-major)
    RateKpR,
    RateKpP,
    RateKpY,
    RateKiR,
    RateKiP,
    RateKiY,
    RateKdR,
    RateKdP,
    RateKdY,
    // INDI effectiveness — G1 force [fx, fy, fz] per motor
    G1FxM0,
    G1FxM1,
    G1FxM2,
    G1FxM3,
    G1FyM0,
    G1FyM1,
    G1FyM2,
    G1FyM3,
    G1FzM0,
    G1FzM1,
    G1FzM2,
    G1FzM3,
    // INDI effectiveness — G1 torque [roll, pitch, yaw] per motor
    G1RrM0,
    G1RrM1,
    G1RrM2,
    G1RrM3,
    G1RpM0,
    G1RpM1,
    G1RpM2,
    G1RpM3,
    G1RyM0,
    G1RyM1,
    G1RyM2,
    G1RyM3,
    // INDI effectiveness — G2 [roll, pitch, yaw] per motor
    G2RrM0,
    G2RrM1,
    G2RrM2,
    G2RrM3,
    G2RpM0,
    G2RpM1,
    G2RpM2,
    G2RpM3,
    G2RyM0,
    G2RyM1,
    G2RyM2,
    G2RyM3,
    // INDI motor dynamics
    IndiOmegaM0,
    IndiOmegaM1,
    IndiOmegaM2,
    IndiOmegaM3,
    IndiTauM0,
    IndiTauM1,
    IndiTauM2,
    IndiTauM3,
    IndiNonlinM0,
    IndiNonlinM1,
    IndiNonlinM2,
    IndiNonlinM3,
    // INDI controller
    IndiRateR,
    IndiRateP,
    IndiRateY,
    IndiSyncHz,
    WlsWvFx,
    WlsWvFy,
    WlsWvFz,
    WlsWvRr,
    WlsWvRp,
    WlsWvRy,
    WlsWuM0,
    WlsWuM1,
    WlsWuM2,
    WlsWuM3,
    MotorPoles,
    // Learner
    LearnFxHz,
    LearnMotorHz,
    LearnAccX,
    LearnAccY,
    LearnAccZ,
    LearnGamma,
    LearnTchar,
    LearnZetaRate,
    LearnZetaAtt,
}

/// All parameter keys in order, for iteration.
pub const ALL_KEYS: &[ParamKey] = &[
    ParamKey::Mass,
    ParamKey::Ixx,
    ParamKey::Ixy,
    ParamKey::Ixz,
    ParamKey::Iyx,
    ParamKey::Iyy,
    ParamKey::Iyz,
    ParamKey::Izx,
    ParamKey::Izy,
    ParamKey::Izz,
    ParamKey::M0Px,
    ParamKey::M0Py,
    ParamKey::M0Spin,
    ParamKey::M0Thrust,
    ParamKey::M0Torque,
    ParamKey::M1Px,
    ParamKey::M1Py,
    ParamKey::M1Spin,
    ParamKey::M1Thrust,
    ParamKey::M1Torque,
    ParamKey::M2Px,
    ParamKey::M2Py,
    ParamKey::M2Spin,
    ParamKey::M2Thrust,
    ParamKey::M2Torque,
    ParamKey::M3Px,
    ParamKey::M3Py,
    ParamKey::M3Spin,
    ParamKey::M3Thrust,
    ParamKey::M3Torque,
    ParamKey::PosKpX,
    ParamKey::PosKpY,
    ParamKey::PosKpZ,
    ParamKey::PosKdX,
    ParamKey::PosKdY,
    ParamKey::PosKdZ,
    ParamKey::AttKrX,
    ParamKey::AttKrY,
    ParamKey::AttKrZ,
    ParamKey::RateKpR,
    ParamKey::RateKpP,
    ParamKey::RateKpY,
    ParamKey::RateKiR,
    ParamKey::RateKiP,
    ParamKey::RateKiY,
    ParamKey::RateKdR,
    ParamKey::RateKdP,
    ParamKey::RateKdY,
    // INDI effectiveness — G1 force
    ParamKey::G1FxM0,
    ParamKey::G1FxM1,
    ParamKey::G1FxM2,
    ParamKey::G1FxM3,
    ParamKey::G1FyM0,
    ParamKey::G1FyM1,
    ParamKey::G1FyM2,
    ParamKey::G1FyM3,
    ParamKey::G1FzM0,
    ParamKey::G1FzM1,
    ParamKey::G1FzM2,
    ParamKey::G1FzM3,
    // INDI effectiveness — G1 torque
    ParamKey::G1RrM0,
    ParamKey::G1RrM1,
    ParamKey::G1RrM2,
    ParamKey::G1RrM3,
    ParamKey::G1RpM0,
    ParamKey::G1RpM1,
    ParamKey::G1RpM2,
    ParamKey::G1RpM3,
    ParamKey::G1RyM0,
    ParamKey::G1RyM1,
    ParamKey::G1RyM2,
    ParamKey::G1RyM3,
    // INDI effectiveness — G2
    ParamKey::G2RrM0,
    ParamKey::G2RrM1,
    ParamKey::G2RrM2,
    ParamKey::G2RrM3,
    ParamKey::G2RpM0,
    ParamKey::G2RpM1,
    ParamKey::G2RpM2,
    ParamKey::G2RpM3,
    ParamKey::G2RyM0,
    ParamKey::G2RyM1,
    ParamKey::G2RyM2,
    ParamKey::G2RyM3,
    // INDI motor dynamics
    ParamKey::IndiOmegaM0,
    ParamKey::IndiOmegaM1,
    ParamKey::IndiOmegaM2,
    ParamKey::IndiOmegaM3,
    ParamKey::IndiTauM0,
    ParamKey::IndiTauM1,
    ParamKey::IndiTauM2,
    ParamKey::IndiTauM3,
    ParamKey::IndiNonlinM0,
    ParamKey::IndiNonlinM1,
    ParamKey::IndiNonlinM2,
    ParamKey::IndiNonlinM3,
    // INDI controller
    ParamKey::IndiRateR,
    ParamKey::IndiRateP,
    ParamKey::IndiRateY,
    ParamKey::IndiSyncHz,
    ParamKey::WlsWvFx,
    ParamKey::WlsWvFy,
    ParamKey::WlsWvFz,
    ParamKey::WlsWvRr,
    ParamKey::WlsWvRp,
    ParamKey::WlsWvRy,
    ParamKey::WlsWuM0,
    ParamKey::WlsWuM1,
    ParamKey::WlsWuM2,
    ParamKey::WlsWuM3,
    ParamKey::MotorPoles,
    // Learner
    ParamKey::LearnFxHz,
    ParamKey::LearnMotorHz,
    ParamKey::LearnAccX,
    ParamKey::LearnAccY,
    ParamKey::LearnAccZ,
    ParamKey::LearnGamma,
    ParamKey::LearnTchar,
    ParamKey::LearnZetaRate,
    ParamKey::LearnZetaAtt,
];

impl ParamKey {
    /// Parse a parameter name string into a key.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "mass" => Some(Self::Mass),
            "ixx" => Some(Self::Ixx),
            "ixy" => Some(Self::Ixy),
            "ixz" => Some(Self::Ixz),
            "iyx" => Some(Self::Iyx),
            "iyy" => Some(Self::Iyy),
            "iyz" => Some(Self::Iyz),
            "izx" => Some(Self::Izx),
            "izy" => Some(Self::Izy),
            "izz" => Some(Self::Izz),
            "m0_px" => Some(Self::M0Px),
            "m0_py" => Some(Self::M0Py),
            "m0_spin" => Some(Self::M0Spin),
            "m0_thrust" => Some(Self::M0Thrust),
            "m0_torque" => Some(Self::M0Torque),
            "m1_px" => Some(Self::M1Px),
            "m1_py" => Some(Self::M1Py),
            "m1_spin" => Some(Self::M1Spin),
            "m1_thrust" => Some(Self::M1Thrust),
            "m1_torque" => Some(Self::M1Torque),
            "m2_px" => Some(Self::M2Px),
            "m2_py" => Some(Self::M2Py),
            "m2_spin" => Some(Self::M2Spin),
            "m2_thrust" => Some(Self::M2Thrust),
            "m2_torque" => Some(Self::M2Torque),
            "m3_px" => Some(Self::M3Px),
            "m3_py" => Some(Self::M3Py),
            "m3_spin" => Some(Self::M3Spin),
            "m3_thrust" => Some(Self::M3Thrust),
            "m3_torque" => Some(Self::M3Torque),
            "pos_kp_x" => Some(Self::PosKpX),
            "pos_kp_y" => Some(Self::PosKpY),
            "pos_kp_z" => Some(Self::PosKpZ),
            "pos_kd_x" => Some(Self::PosKdX),
            "pos_kd_y" => Some(Self::PosKdY),
            "pos_kd_z" => Some(Self::PosKdZ),
            "att_kr_x" => Some(Self::AttKrX),
            "att_kr_y" => Some(Self::AttKrY),
            "att_kr_z" => Some(Self::AttKrZ),
            "rate_kp_r" => Some(Self::RateKpR),
            "rate_kp_p" => Some(Self::RateKpP),
            "rate_kp_y" => Some(Self::RateKpY),
            "rate_ki_r" => Some(Self::RateKiR),
            "rate_ki_p" => Some(Self::RateKiP),
            "rate_ki_y" => Some(Self::RateKiY),
            "rate_kd_r" => Some(Self::RateKdR),
            "rate_kd_p" => Some(Self::RateKdP),
            "rate_kd_y" => Some(Self::RateKdY),
            // INDI effectiveness — G1 force
            "g1_fx_m0" => Some(Self::G1FxM0),
            "g1_fx_m1" => Some(Self::G1FxM1),
            "g1_fx_m2" => Some(Self::G1FxM2),
            "g1_fx_m3" => Some(Self::G1FxM3),
            "g1_fy_m0" => Some(Self::G1FyM0),
            "g1_fy_m1" => Some(Self::G1FyM1),
            "g1_fy_m2" => Some(Self::G1FyM2),
            "g1_fy_m3" => Some(Self::G1FyM3),
            "g1_fz_m0" => Some(Self::G1FzM0),
            "g1_fz_m1" => Some(Self::G1FzM1),
            "g1_fz_m2" => Some(Self::G1FzM2),
            "g1_fz_m3" => Some(Self::G1FzM3),
            // INDI effectiveness — G1 torque
            "g1_rr_m0" => Some(Self::G1RrM0),
            "g1_rr_m1" => Some(Self::G1RrM1),
            "g1_rr_m2" => Some(Self::G1RrM2),
            "g1_rr_m3" => Some(Self::G1RrM3),
            "g1_rp_m0" => Some(Self::G1RpM0),
            "g1_rp_m1" => Some(Self::G1RpM1),
            "g1_rp_m2" => Some(Self::G1RpM2),
            "g1_rp_m3" => Some(Self::G1RpM3),
            "g1_ry_m0" => Some(Self::G1RyM0),
            "g1_ry_m1" => Some(Self::G1RyM1),
            "g1_ry_m2" => Some(Self::G1RyM2),
            "g1_ry_m3" => Some(Self::G1RyM3),
            // INDI effectiveness — G2
            "g2_rr_m0" => Some(Self::G2RrM0),
            "g2_rr_m1" => Some(Self::G2RrM1),
            "g2_rr_m2" => Some(Self::G2RrM2),
            "g2_rr_m3" => Some(Self::G2RrM3),
            "g2_rp_m0" => Some(Self::G2RpM0),
            "g2_rp_m1" => Some(Self::G2RpM1),
            "g2_rp_m2" => Some(Self::G2RpM2),
            "g2_rp_m3" => Some(Self::G2RpM3),
            "g2_ry_m0" => Some(Self::G2RyM0),
            "g2_ry_m1" => Some(Self::G2RyM1),
            "g2_ry_m2" => Some(Self::G2RyM2),
            "g2_ry_m3" => Some(Self::G2RyM3),
            // INDI motor dynamics
            "indi_omega_m0" => Some(Self::IndiOmegaM0),
            "indi_omega_m1" => Some(Self::IndiOmegaM1),
            "indi_omega_m2" => Some(Self::IndiOmegaM2),
            "indi_omega_m3" => Some(Self::IndiOmegaM3),
            "indi_tau_m0" => Some(Self::IndiTauM0),
            "indi_tau_m1" => Some(Self::IndiTauM1),
            "indi_tau_m2" => Some(Self::IndiTauM2),
            "indi_tau_m3" => Some(Self::IndiTauM3),
            "indi_nonlin_m0" => Some(Self::IndiNonlinM0),
            "indi_nonlin_m1" => Some(Self::IndiNonlinM1),
            "indi_nonlin_m2" => Some(Self::IndiNonlinM2),
            "indi_nonlin_m3" => Some(Self::IndiNonlinM3),
            // INDI controller
            "indi_rate_r" => Some(Self::IndiRateR),
            "indi_rate_p" => Some(Self::IndiRateP),
            "indi_rate_y" => Some(Self::IndiRateY),
            "indi_sync_hz" => Some(Self::IndiSyncHz),
            "wls_wv_fx" => Some(Self::WlsWvFx),
            "wls_wv_fy" => Some(Self::WlsWvFy),
            "wls_wv_fz" => Some(Self::WlsWvFz),
            "wls_wv_rr" => Some(Self::WlsWvRr),
            "wls_wv_rp" => Some(Self::WlsWvRp),
            "wls_wv_ry" => Some(Self::WlsWvRy),
            "wls_wu_m0" => Some(Self::WlsWuM0),
            "wls_wu_m1" => Some(Self::WlsWuM1),
            "wls_wu_m2" => Some(Self::WlsWuM2),
            "wls_wu_m3" => Some(Self::WlsWuM3),
            "motor_poles" => Some(Self::MotorPoles),
            // Learner
            "learn_fx_hz" => Some(Self::LearnFxHz),
            "learn_motor_hz" => Some(Self::LearnMotorHz),
            "learn_acc_x" => Some(Self::LearnAccX),
            "learn_acc_y" => Some(Self::LearnAccY),
            "learn_acc_z" => Some(Self::LearnAccZ),
            "learn_gamma" => Some(Self::LearnGamma),
            "learn_tchar" => Some(Self::LearnTchar),
            "learn_zeta_rate" => Some(Self::LearnZetaRate),
            "learn_zeta_att" => Some(Self::LearnZetaAtt),
            _ => None,
        }
    }

    /// Return the canonical name for this key.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mass => "mass",
            Self::Ixx => "ixx",
            Self::Ixy => "ixy",
            Self::Ixz => "ixz",
            Self::Iyx => "iyx",
            Self::Iyy => "iyy",
            Self::Iyz => "iyz",
            Self::Izx => "izx",
            Self::Izy => "izy",
            Self::Izz => "izz",
            Self::M0Px => "m0_px",
            Self::M0Py => "m0_py",
            Self::M0Spin => "m0_spin",
            Self::M0Thrust => "m0_thrust",
            Self::M0Torque => "m0_torque",
            Self::M1Px => "m1_px",
            Self::M1Py => "m1_py",
            Self::M1Spin => "m1_spin",
            Self::M1Thrust => "m1_thrust",
            Self::M1Torque => "m1_torque",
            Self::M2Px => "m2_px",
            Self::M2Py => "m2_py",
            Self::M2Spin => "m2_spin",
            Self::M2Thrust => "m2_thrust",
            Self::M2Torque => "m2_torque",
            Self::M3Px => "m3_px",
            Self::M3Py => "m3_py",
            Self::M3Spin => "m3_spin",
            Self::M3Thrust => "m3_thrust",
            Self::M3Torque => "m3_torque",
            Self::PosKpX => "pos_kp_x",
            Self::PosKpY => "pos_kp_y",
            Self::PosKpZ => "pos_kp_z",
            Self::PosKdX => "pos_kd_x",
            Self::PosKdY => "pos_kd_y",
            Self::PosKdZ => "pos_kd_z",
            Self::AttKrX => "att_kr_x",
            Self::AttKrY => "att_kr_y",
            Self::AttKrZ => "att_kr_z",
            Self::RateKpR => "rate_kp_r",
            Self::RateKpP => "rate_kp_p",
            Self::RateKpY => "rate_kp_y",
            Self::RateKiR => "rate_ki_r",
            Self::RateKiP => "rate_ki_p",
            Self::RateKiY => "rate_ki_y",
            Self::RateKdR => "rate_kd_r",
            Self::RateKdP => "rate_kd_p",
            Self::RateKdY => "rate_kd_y",
            // INDI effectiveness — G1 force
            Self::G1FxM0 => "g1_fx_m0",
            Self::G1FxM1 => "g1_fx_m1",
            Self::G1FxM2 => "g1_fx_m2",
            Self::G1FxM3 => "g1_fx_m3",
            Self::G1FyM0 => "g1_fy_m0",
            Self::G1FyM1 => "g1_fy_m1",
            Self::G1FyM2 => "g1_fy_m2",
            Self::G1FyM3 => "g1_fy_m3",
            Self::G1FzM0 => "g1_fz_m0",
            Self::G1FzM1 => "g1_fz_m1",
            Self::G1FzM2 => "g1_fz_m2",
            Self::G1FzM3 => "g1_fz_m3",
            // INDI effectiveness — G1 torque
            Self::G1RrM0 => "g1_rr_m0",
            Self::G1RrM1 => "g1_rr_m1",
            Self::G1RrM2 => "g1_rr_m2",
            Self::G1RrM3 => "g1_rr_m3",
            Self::G1RpM0 => "g1_rp_m0",
            Self::G1RpM1 => "g1_rp_m1",
            Self::G1RpM2 => "g1_rp_m2",
            Self::G1RpM3 => "g1_rp_m3",
            Self::G1RyM0 => "g1_ry_m0",
            Self::G1RyM1 => "g1_ry_m1",
            Self::G1RyM2 => "g1_ry_m2",
            Self::G1RyM3 => "g1_ry_m3",
            // INDI effectiveness — G2
            Self::G2RrM0 => "g2_rr_m0",
            Self::G2RrM1 => "g2_rr_m1",
            Self::G2RrM2 => "g2_rr_m2",
            Self::G2RrM3 => "g2_rr_m3",
            Self::G2RpM0 => "g2_rp_m0",
            Self::G2RpM1 => "g2_rp_m1",
            Self::G2RpM2 => "g2_rp_m2",
            Self::G2RpM3 => "g2_rp_m3",
            Self::G2RyM0 => "g2_ry_m0",
            Self::G2RyM1 => "g2_ry_m1",
            Self::G2RyM2 => "g2_ry_m2",
            Self::G2RyM3 => "g2_ry_m3",
            // INDI motor dynamics
            Self::IndiOmegaM0 => "indi_omega_m0",
            Self::IndiOmegaM1 => "indi_omega_m1",
            Self::IndiOmegaM2 => "indi_omega_m2",
            Self::IndiOmegaM3 => "indi_omega_m3",
            Self::IndiTauM0 => "indi_tau_m0",
            Self::IndiTauM1 => "indi_tau_m1",
            Self::IndiTauM2 => "indi_tau_m2",
            Self::IndiTauM3 => "indi_tau_m3",
            Self::IndiNonlinM0 => "indi_nonlin_m0",
            Self::IndiNonlinM1 => "indi_nonlin_m1",
            Self::IndiNonlinM2 => "indi_nonlin_m2",
            Self::IndiNonlinM3 => "indi_nonlin_m3",
            // INDI controller
            Self::IndiRateR => "indi_rate_r",
            Self::IndiRateP => "indi_rate_p",
            Self::IndiRateY => "indi_rate_y",
            Self::IndiSyncHz => "indi_sync_hz",
            Self::WlsWvFx => "wls_wv_fx",
            Self::WlsWvFy => "wls_wv_fy",
            Self::WlsWvFz => "wls_wv_fz",
            Self::WlsWvRr => "wls_wv_rr",
            Self::WlsWvRp => "wls_wv_rp",
            Self::WlsWvRy => "wls_wv_ry",
            Self::WlsWuM0 => "wls_wu_m0",
            Self::WlsWuM1 => "wls_wu_m1",
            Self::WlsWuM2 => "wls_wu_m2",
            Self::WlsWuM3 => "wls_wu_m3",
            Self::MotorPoles => "motor_poles",
            // Learner
            Self::LearnFxHz => "learn_fx_hz",
            Self::LearnMotorHz => "learn_motor_hz",
            Self::LearnAccX => "learn_acc_x",
            Self::LearnAccY => "learn_acc_y",
            Self::LearnAccZ => "learn_acc_z",
            Self::LearnGamma => "learn_gamma",
            Self::LearnTchar => "learn_tchar",
            Self::LearnZetaRate => "learn_zeta_rate",
            Self::LearnZetaAtt => "learn_zeta_att",
        }
    }
}

// -- byte helpers --

fn put_f32(buf: &mut [u8], off: usize, val: f32) -> usize {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    off + 4
}

fn put_u32(buf: &mut [u8], off: usize, val: u32) -> usize {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    off + 4
}

fn get_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixer::SpinDir;

    fn test_params() -> VehicleParams {
        VehicleParams {
            body: RigidBodyParams {
                mass_kg: 0.5,
                inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
                max_rate_rad_s: [10.0, 10.0, 6.0],
            },
            motors: [
                MotorParams {
                    position_m: [-0.075, -0.1],
                    spin_dir: SpinDir::Cw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [0.075, -0.1],
                    spin_dir: SpinDir::Ccw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [-0.075, 0.1],
                    spin_dir: SpinDir::Ccw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [0.075, 0.1],
                    spin_dir: SpinDir::Cw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
            ],
            control: ControlGains {
                pos_kp: [4.0, 4.0, 5.0],
                pos_kd: [4.0, 4.0, 4.0],
                att_k_rate: [3.0, 3.0, 1.0],
                rate_kp: [0.1, 0.08, 0.05],
                rate_ki: [0.0, 0.0, 0.0],
                rate_kd: [0.0, 0.0, 0.0],
            },
            indi_effectiveness: IndiEffectivenessParams::default(),
            indi_controller: IndiControllerParams::default(),
            learner: LearnerParams::default(),
            mpc: MpcParams::default(),
            planner: PlannerParams::default(),
        }
    }

    #[test]
    fn round_trip() {
        let params = test_params();
        let bytes = params.to_bytes();
        let restored = VehicleParams::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.body.mass_kg, params.body.mass_kg);
        assert_eq!(restored.body.inertia_kg_m2, params.body.inertia_kg_m2);
        for (a, b) in restored.motors.iter().zip(params.motors.iter()) {
            assert_eq!(a.position_m, b.position_m);
            assert_eq!(a.spin_dir, b.spin_dir);
            assert_eq!(a.max_thrust_n, b.max_thrust_n);
            assert_eq!(a.torque_coeff_m, b.torque_coeff_m);
        }
    }

    #[test]
    fn round_trip_with_indi() {
        let mut params = test_params();
        params.indi_effectiveness.g1_force[0] = [1.0, 2.0, 3.0];
        params.indi_effectiveness.g1_torque[1] = [4.0, 5.0, 6.0];
        params.indi_effectiveness.g2[2] = [7.0, 8.0, 9.0];
        params.indi_effectiveness.max_omega = [2000.0, 1900.0, 2100.0, 1950.0];
        params.indi_effectiveness.time_const_s = [0.025, 0.030, 0.022, 0.028];
        params.indi_effectiveness.nonlinearity = [0.5, 0.6, 0.45, 0.55];

        let bytes = params.to_bytes();
        let restored = VehicleParams::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(
            restored.indi_effectiveness.g1_force,
            params.indi_effectiveness.g1_force
        );
        assert_eq!(
            restored.indi_effectiveness.g1_torque,
            params.indi_effectiveness.g1_torque
        );
        assert_eq!(restored.indi_effectiveness.g2, params.indi_effectiveness.g2);
        assert_eq!(
            restored.indi_effectiveness.max_omega,
            params.indi_effectiveness.max_omega
        );
        assert_eq!(
            restored.indi_effectiveness.time_const_s,
            params.indi_effectiveness.time_const_s
        );
        assert_eq!(
            restored.indi_effectiveness.nonlinearity,
            params.indi_effectiveness.nonlinearity
        );
    }

    #[test]
    fn bad_magic_returns_none() {
        let params = test_params();
        let mut bytes = params.to_bytes();
        bytes[0] = 0xFF;
        assert!(VehicleParams::from_bytes(&bytes).is_none());
    }

    #[test]
    fn bad_crc_returns_none() {
        let params = test_params();
        let mut bytes = params.to_bytes();
        bytes[HEADER_SIZE] ^= 0xFF; // corrupt payload
        assert!(VehicleParams::from_bytes(&bytes).is_none());
    }

    #[test]
    fn blank_flash_returns_none() {
        let bytes = [0xFF; PADDED_SIZE];
        assert!(VehicleParams::from_bytes(&bytes).is_none());
    }

    #[test]
    fn get_set_round_trip() {
        let mut params = test_params();
        params.set(ParamKey::Mass, 0.6);
        assert_eq!(params.get(ParamKey::Mass), 0.6);
        params.set(ParamKey::M2Thrust, 9.0);
        assert_eq!(params.get(ParamKey::M2Thrust), 9.0);
    }

    #[test]
    fn get_set_indi_round_trip() {
        let mut params = test_params();
        params.set(ParamKey::G1FxM0, 1.5);
        assert_eq!(params.get(ParamKey::G1FxM0), 1.5);
        params.set(ParamKey::G2RyM3, -0.001);
        assert_eq!(params.get(ParamKey::G2RyM3), -0.001);
        params.set(ParamKey::IndiOmegaM2, 2100.0);
        assert_eq!(params.get(ParamKey::IndiOmegaM2), 2100.0);
        params.set(ParamKey::IndiTauM1, 0.03);
        assert_eq!(params.get(ParamKey::IndiTauM1), 0.03);
        params.set(ParamKey::IndiNonlinM0, 0.7);
        assert_eq!(params.get(ParamKey::IndiNonlinM0), 0.7);
    }

    #[test]
    fn param_key_from_str() {
        assert_eq!(ParamKey::from_str("mass"), Some(ParamKey::Mass));
        assert_eq!(ParamKey::from_str("m3_torque"), Some(ParamKey::M3Torque));
        assert_eq!(ParamKey::from_str("g1_fx_m0"), Some(ParamKey::G1FxM0));
        assert_eq!(ParamKey::from_str("g2_ry_m3"), Some(ParamKey::G2RyM3));
        assert_eq!(
            ParamKey::from_str("indi_omega_m2"),
            Some(ParamKey::IndiOmegaM2)
        );
        assert_eq!(ParamKey::from_str("indi_tau_m1"), Some(ParamKey::IndiTauM1));
        assert_eq!(
            ParamKey::from_str("indi_nonlin_m0"),
            Some(ParamKey::IndiNonlinM0)
        );
        assert_eq!(ParamKey::from_str("indi_rate_r"), Some(ParamKey::IndiRateR));
        assert_eq!(
            ParamKey::from_str("indi_sync_hz"),
            Some(ParamKey::IndiSyncHz)
        );
        assert_eq!(ParamKey::from_str("wls_wv_fz"), Some(ParamKey::WlsWvFz));
        assert_eq!(ParamKey::from_str("wls_wu_m2"), Some(ParamKey::WlsWuM2));
        assert_eq!(
            ParamKey::from_str("motor_poles"),
            Some(ParamKey::MotorPoles)
        );
        assert_eq!(ParamKey::from_str("learn_fx_hz"), Some(ParamKey::LearnFxHz));
        assert_eq!(
            ParamKey::from_str("learn_gamma"),
            Some(ParamKey::LearnGamma)
        );
        assert_eq!(
            ParamKey::from_str("learn_zeta_att"),
            Some(ParamKey::LearnZetaAtt)
        );
        assert_eq!(ParamKey::from_str("invalid"), None);
    }

    #[test]
    fn round_trip_with_indi_controller_and_learner() {
        let mut params = test_params();
        params.indi_controller.rate_gains = [25.0, 30.0, 10.0];
        params.indi_controller.sync_filter_hz = 20.0;
        params.indi_controller.wls_wv = [2.0, 2.0, 100.0, 100.0, 100.0, 10.0];
        params.indi_controller.wls_wu = [1.5, 1.5, 1.5, 1.5];
        params.indi_controller.motor_pole_count = 12;
        params.learner.fx_filt_hz = 25.0;
        params.learner.motor_filt_hz = 50.0;
        params.learner.acc_offset_m = [0.01, -0.02, 0.03];
        params.learner.rls_gamma = 200.0;
        params.learner.rls_t_char_s = 0.5;
        params.learner.zeta_rate = 0.7;
        params.learner.zeta_attitude = 0.9;

        let bytes = params.to_bytes();
        let restored = VehicleParams::from_bytes(&bytes).expect("from_bytes failed");

        // INDI controller
        assert_eq!(
            restored.indi_controller.rate_gains,
            params.indi_controller.rate_gains
        );
        assert_eq!(
            restored.indi_controller.sync_filter_hz,
            params.indi_controller.sync_filter_hz
        );
        assert_eq!(
            restored.indi_controller.wls_wv,
            params.indi_controller.wls_wv
        );
        assert_eq!(
            restored.indi_controller.wls_wu,
            params.indi_controller.wls_wu
        );
        assert_eq!(
            restored.indi_controller.motor_pole_count,
            params.indi_controller.motor_pole_count
        );

        // Learner
        assert_eq!(restored.learner.fx_filt_hz, params.learner.fx_filt_hz);
        assert_eq!(restored.learner.motor_filt_hz, params.learner.motor_filt_hz);
        assert_eq!(restored.learner.acc_offset_m, params.learner.acc_offset_m);
        assert_eq!(restored.learner.rls_gamma, params.learner.rls_gamma);
        assert_eq!(restored.learner.rls_t_char_s, params.learner.rls_t_char_s);
        assert_eq!(restored.learner.zeta_rate, params.learner.zeta_rate);
        assert_eq!(restored.learner.zeta_attitude, params.learner.zeta_attitude);

        // Existing fields still intact
        assert_eq!(restored.body.mass_kg, params.body.mass_kg);
    }

    #[test]
    fn get_set_indi_controller_and_learner() {
        let mut params = test_params();
        params.set(ParamKey::IndiRateR, 25.0);
        assert_eq!(params.get(ParamKey::IndiRateR), 25.0);
        params.set(ParamKey::IndiSyncHz, 20.0);
        assert_eq!(params.get(ParamKey::IndiSyncHz), 20.0);
        params.set(ParamKey::WlsWvFz, 100.0);
        assert_eq!(params.get(ParamKey::WlsWvFz), 100.0);
        params.set(ParamKey::WlsWuM2, 2.0);
        assert_eq!(params.get(ParamKey::WlsWuM2), 2.0);
        params.set(ParamKey::MotorPoles, 12.0);
        assert_eq!(params.get(ParamKey::MotorPoles), 12.0);
        params.set(ParamKey::LearnFxHz, 25.0);
        assert_eq!(params.get(ParamKey::LearnFxHz), 25.0);
        params.set(ParamKey::LearnGamma, 200.0);
        assert_eq!(params.get(ParamKey::LearnGamma), 200.0);
        params.set(ParamKey::LearnTchar, 0.5);
        assert_eq!(params.get(ParamKey::LearnTchar), 0.5);
        params.set(ParamKey::LearnZetaRate, 0.7);
        assert_eq!(params.get(ParamKey::LearnZetaRate), 0.7);
        params.set(ParamKey::LearnZetaAtt, 0.9);
        assert_eq!(params.get(ParamKey::LearnZetaAtt), 0.9);
        params.set(ParamKey::LearnAccX, 0.01);
        assert_eq!(params.get(ParamKey::LearnAccX), 0.01);
    }
}
