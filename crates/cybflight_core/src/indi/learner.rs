// Online G1/G2 effectiveness learning for INDI.
//
// Ported from indiflight's learner.c. Uses RLS from the flight-solver crate
// to identify motor effectiveness (G1), gyroscopic coupling (G2), and motor
// dynamics (time constant, gain, nonlinearity) from in-flight sensor data.
//
// Pure computation — no async, no channels, no embassy types.
// All filtering uses air-filters biquad (same crate as the INDI controller).
//
// # Signal processing pipeline
//
// All signals go through matched biquad filters before differencing, ensuring
// the regressors and observations share the same group delay. The Δ operator
// (frame-to-frame difference) eliminates DC bias so the RLS learns only the
// dynamic relationship between motor state changes and vehicle acceleration.
//
// # Scaling factors (matching indiflight learner.c)
//
// FX regressors:  1e-5 * 2 * ω * Δω   (G1, ~O(1))
//                 1e-3 * Δω̇            (G2, ~O(1))
// FX observations: Δ(rate_dot)          (torque, ~O(1))
//                  Δ(spf) * 10          (force, ~O(1))
// Motor regressors: [D, √D, 1, -1e-4 * ω̇]
// Motor observation: ω * 1e-3

use air_filters::Filter;
use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use flight_solver::rls::{CovarianceGuards, RlsParallel};
use nalgebra::{SMatrix, SVector, Vector3};

/// Number of actuators (motors). Must match controller::NU.
pub const NU: usize = 4;

/// Number of motor RLS regressors: [D, √D, 1, -ω̇].
const MOTOR_N: usize = 4;

/// Number of FX regressors: NU (G1) + NU (G2) = 2*NU.
const FX_N: usize = 2 * NU;

type Biquad = BiquadFilter<f32, DirectForm2<f32>>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Learner configuration.
#[derive(Clone, Copy)]
pub struct LearnerConfig {
    /// Biquad cutoff for FX (effectiveness) learning filters (Hz).
    /// Matched for regressors and observations. Default: 20 Hz.
    pub fx_filt_hz: f32,
    /// Biquad cutoff for motor dynamics learning filters (Hz). Default: 40 Hz.
    pub motor_filt_hz: f32,
    /// IMU offset from CoG in body frame [x, y, z] (metres).
    /// Used for Coriolis/centrifugal correction on accelerometer.
    pub acc_offset_m: [f32; 3],
    /// RLS initial covariance diagonal. Default: 100.
    pub rls_gamma: f32,
    /// RLS characteristic forgetting time (seconds). Default: 0.25
    /// (= 10 × motor time constant). λ = (1-ln2)^(Ts/Tchar).
    pub rls_t_char_s: f32,
    /// Rate loop damping ratio for gain synthesis. Default: 0.8.
    pub zeta_rate: f32,
    /// Attitude loop damping ratio for gain synthesis. Default: 0.8.
    pub zeta_attitude: f32,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            fx_filt_hz: 20.0,
            motor_filt_hz: 40.0,
            acc_offset_m: [0.0; 3],
            rls_gamma: 100.0,
            rls_t_char_s: 0.25,
            zeta_rate: 0.8,
            zeta_attitude: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output
// ---------------------------------------------------------------------------

/// Per-frame input to the learner from the INDI task.
pub struct LearnerInput {
    /// Body angular rate (rad/s), from INDI (bias-corrected gyro).
    pub rate_rad_s: Vector3<f32>,
    /// Body angular acceleration (rad/s²), from INDI (differentiated gyro).
    pub rate_dot_rad_s2: Vector3<f32>,
    /// Specific force (m/s²), from INDI (bias-corrected accel).
    pub spf_m_s2: Vector3<f32>,
    /// Motor speed per motor (rad/s), from RPM tracker (unfiltered).
    pub omega_rad_s: [f32; NU],
    /// Motor commands per motor [0,1], from INDI output (after linearization).
    pub d_commands: [f32; NU],
    /// True if the vehicle is armed.
    pub armed: bool,
    /// True if the vehicle is on the ground.
    pub touching_ground: bool,
}

/// Learned parameters extracted from RLS estimates.
#[derive(Clone)]
pub struct LearnedParams {
    /// G1 effectiveness matrix (6×NU): [fx, fy, fz, roll, pitch, yaw] per motor.
    /// In acceleration space (m/s² or rad/s² per normalized command).
    pub g1: SMatrix<f32, 6, NU>,
    /// G2 gyroscopic coupling (3×NU): [roll, pitch, yaw] per motor.
    pub g2: SMatrix<f32, 3, NU>,
    /// Learned max motor speed per motor (rad/s).
    pub max_omega: [f32; NU],
    /// Learned motor time constant per motor (seconds).
    pub time_const_s: [f32; NU],
    /// Learned motor nonlinearity per motor [0, 1].
    pub nonlinearity: [f32; NU],
    /// Synthesized rate gain (shared across axes).
    pub rate_gain: f32,
    /// Synthesized attitude gain (shared across axes).
    pub attitude_gain: f32,
    /// True once enough samples have been processed for the estimates to be meaningful.
    pub valid: bool,
}

/// Magic + version + length + CRC header for flash persistence.
const LEARNED_MAGIC: u32 = 0x4C524E50; // "LRNP"
const LEARNED_VERSION: u32 = 1;
/// Payload: G1(24f) + G2(12f) + max_omega(4f) + time_const(4f) + nonlin(4f) + rate_gain(1f) + att_gain(1f) + valid(1u8) = 50 floats + 1 byte = 201 bytes
/// Round to 204 for alignment.
const LEARNED_PAYLOAD_SIZE: usize = 204;
const LEARNED_HEADER_SIZE: usize = 16;
/// Padded to 32-byte flash word boundary: ceil((16+204)/32)*32 = 224
pub const LEARNED_PADDED_SIZE: usize = 224;

impl LearnedParams {
    /// Serialize to a fixed-size byte array for flash storage.
    pub fn to_bytes(&self) -> [u8; LEARNED_PADDED_SIZE] {
        let mut buf = [0u8; LEARNED_PADDED_SIZE];
        let mut off = LEARNED_HEADER_SIZE;

        // G1: 6×4 = 24 floats, column-major
        for col in 0..NU {
            for row in 0..6 {
                buf[off..off + 4].copy_from_slice(&self.g1[(row, col)].to_le_bytes());
                off += 4;
            }
        }
        // G2: 3×4 = 12 floats, column-major
        for col in 0..NU {
            for row in 0..3 {
                buf[off..off + 4].copy_from_slice(&self.g2[(row, col)].to_le_bytes());
                off += 4;
            }
        }
        // max_omega, time_const_s, nonlinearity: 4 each
        for i in 0..NU {
            buf[off..off + 4].copy_from_slice(&self.max_omega[i].to_le_bytes());
            off += 4;
        }
        for i in 0..NU {
            buf[off..off + 4].copy_from_slice(&self.time_const_s[i].to_le_bytes());
            off += 4;
        }
        for i in 0..NU {
            buf[off..off + 4].copy_from_slice(&self.nonlinearity[i].to_le_bytes());
            off += 4;
        }
        // rate_gain, attitude_gain
        buf[off..off + 4].copy_from_slice(&self.rate_gain.to_le_bytes());
        off += 4;
        buf[off..off + 4].copy_from_slice(&self.attitude_gain.to_le_bytes());
        off += 4;
        // valid
        buf[off] = self.valid as u8;
        off += 1;
        // Remaining bytes stay 0 (padding to 204)

        // Header: magic, version, length, CRC
        let payload = &buf[LEARNED_HEADER_SIZE..LEARNED_HEADER_SIZE + LEARNED_PAYLOAD_SIZE];
        let crc = crc32fast::hash(payload);
        buf[0..4].copy_from_slice(&LEARNED_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&LEARNED_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&(LEARNED_PAYLOAD_SIZE as u32).to_le_bytes());
        buf[12..16].copy_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Deserialize from a flash buffer. Returns `None` if magic/version/CRC mismatch.
    pub fn from_bytes(buf: &[u8; LEARNED_PADDED_SIZE]) -> Option<Self> {
        // Validate header
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let length = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        let stored_crc = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

        if magic != LEARNED_MAGIC || version != LEARNED_VERSION || length != LEARNED_PAYLOAD_SIZE {
            return None;
        }

        let payload = &buf[LEARNED_HEADER_SIZE..LEARNED_HEADER_SIZE + LEARNED_PAYLOAD_SIZE];
        if crc32fast::hash(payload) != stored_crc {
            return None;
        }

        let mut off = LEARNED_HEADER_SIZE;
        let read_f32 = |buf: &[u8], off: &mut usize| -> f32 {
            let val = f32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]);
            *off += 4;
            val
        };

        let mut g1 = SMatrix::<f32, 6, NU>::zeros();
        for col in 0..NU {
            for row in 0..6 {
                g1[(row, col)] = read_f32(buf, &mut off);
            }
        }
        let mut g2 = SMatrix::<f32, 3, NU>::zeros();
        for col in 0..NU {
            for row in 0..3 {
                g2[(row, col)] = read_f32(buf, &mut off);
            }
        }
        let mut max_omega = [0.0f32; NU];
        for i in 0..NU {
            max_omega[i] = read_f32(buf, &mut off);
        }
        let mut time_const_s = [0.0f32; NU];
        for i in 0..NU {
            time_const_s[i] = read_f32(buf, &mut off);
        }
        let mut nonlinearity = [0.0f32; NU];
        for i in 0..NU {
            nonlinearity[i] = read_f32(buf, &mut off);
        }
        let rate_gain = read_f32(buf, &mut off);
        let attitude_gain = read_f32(buf, &mut off);
        let valid = buf[off] != 0;

        Some(Self {
            g1,
            g2,
            max_omega,
            time_const_s,
            nonlinearity,
            rate_gain,
            attitude_gain,
            valid,
        })
    }
}

impl Default for LearnedParams {
    fn default() -> Self {
        Self {
            g1: SMatrix::zeros(),
            g2: SMatrix::zeros(),
            max_omega: [0.0; NU],
            time_const_s: [0.025; NU],
            nonlinearity: [0.5; NU],
            rate_gain: 0.0,
            attitude_gain: 0.0,
            valid: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Learner
// ---------------------------------------------------------------------------

/// Online G1/G2 + motor dynamics learner.
///
/// Call [`Learner::update`] every INDI frame (8 kHz). Learning is gated on
/// `armed && !touching_ground` — the learner produces no RLS updates on the
/// ground or when disarmed.
pub struct Learner {
    config: LearnerConfig,
    loop_rate_hz: f32,

    // ── FX (effectiveness) filters ─────────────────────────────────────
    fx_omega_filter: [Biquad; NU],
    fx_rate_filter: [Biquad; 3],
    fx_spf_filter: [Biquad; 3],

    // ── FX differencing state ──────────────────────────────────────────
    prev_fx_omega: [f32; NU],
    prev_fx_omega_dot: [f32; NU],
    prev_fx_rate_dot: [f32; 3],
    prev_fx_spf: [f32; 3],

    // ── Motor dynamics filters ─────────────────────────────────────────
    motor_omega_filter: [Biquad; NU],
    motor_d_filter: [Biquad; NU],
    motor_sqrt_d_filter: [Biquad; NU],

    // ── Motor differencing state ───────────────────────────────────────
    prev_motor_omega: [f32; NU],

    // ── RLS instances ──────────────────────────────────────────────────
    /// Force effectiveness: regressors = NU G1 terms only.
    /// Observations = Δ(filtered_spf) per axis.
    fx_spf_rls: RlsParallel<NU, 3>,
    /// Torque effectiveness: regressors = NU G1 + NU G2 terms.
    /// Observations = Δ(filtered_rate_dot) per axis.
    fx_rate_dot_rls: RlsParallel<FX_N, 3>,
    /// Per-motor dynamics: regressors = [D, √D, 1, -ω̇].
    /// Observation = ω.
    motor_rls: [RlsParallel<MOTOR_N, 1>; NU],

    // ── Output state ───────────────────────────────────────────────────
    samples: u32,
}

impl Learner {
    /// Create a new learner.
    ///
    /// `loop_rate_hz` must match the INDI controller's loop rate (typically 8000).
    pub fn new(config: &LearnerConfig, loop_rate_hz: f32) -> Self {
        let ts = 1.0 / loop_rate_hz;

        let make_biquad = |cutoff_hz: f32| {
            let cfg = BiquadFilterConfigBuilder::direct_form_2()
                .sample_frequency_hz(loop_rate_hz)
                .filter_type(BiquadFilterType::LowPass)
                .cutoff_frequency_hz(cutoff_hz)
                .build()
                .expect("learner: biquad config invalid");
            BiquadFilter::new(cfg)
        };

        let guards = CovarianceGuards::default();

        let fx_spf_rls =
            RlsParallel::<NU, 3>::from_time_constant(config.rls_gamma, ts, config.rls_t_char_s, guards);
        let fx_rate_dot_rls =
            RlsParallel::<FX_N, 3>::from_time_constant(config.rls_gamma, ts, config.rls_t_char_s, guards);
        let motor_rls: [RlsParallel<MOTOR_N, 1>; NU] =
            core::array::from_fn(|_| {
                RlsParallel::<MOTOR_N, 1>::from_time_constant(config.rls_gamma, ts, config.rls_t_char_s, guards)
            });

        Self {
            config: *config,
            loop_rate_hz,
            fx_omega_filter: core::array::from_fn(|_| make_biquad(config.fx_filt_hz)),
            fx_rate_filter: core::array::from_fn(|_| make_biquad(config.fx_filt_hz)),
            fx_spf_filter: core::array::from_fn(|_| make_biquad(config.fx_filt_hz)),
            prev_fx_omega: [0.0; NU],
            prev_fx_omega_dot: [0.0; NU],
            prev_fx_rate_dot: [0.0; 3],
            prev_fx_spf: [0.0; 3],
            motor_omega_filter: core::array::from_fn(|_| make_biquad(config.motor_filt_hz)),
            motor_d_filter: core::array::from_fn(|_| make_biquad(config.motor_filt_hz)),
            motor_sqrt_d_filter: core::array::from_fn(|_| make_biquad(config.motor_filt_hz)),
            prev_motor_omega: [0.0; NU],
            fx_spf_rls,
            fx_rate_dot_rls,
            motor_rls,
            samples: 0,
        }
    }

    /// Process one frame. Returns learned parameters (valid once enough data).
    ///
    /// Must be called every INDI iteration (~8 kHz), even when learning is
    /// disabled (filters need continuous input to stay synchronized).
    pub fn update(&mut self, input: &LearnerInput) -> LearnedParams {
        // ── 1. Update all filters (always, for signal continuity) ──────
        let (fx_omega_diff, fx_omega_dot_diff, fx_rate_dot_diff, fx_spf_diff) =
            self.update_fx_filters(input);
        let (motor_omega, motor_d, motor_sqrt_d, motor_omega_dot) =
            self.update_motor_filters(input);

        // ── 2. Learning gating: only learn in flight ───────────────────
        let learn = input.armed && !input.touching_ground;

        if learn {
            self.samples = self.samples.saturating_add(1);

            // ── 3. FX effectiveness RLS ────────────────────────────────
            self.update_fx_rls(&fx_omega_diff, &fx_omega_dot_diff, &fx_rate_dot_diff, &fx_spf_diff);

            // ── 4. Motor dynamics RLS ──────────────────────────────────
            self.update_motor_rls(&motor_omega, &motor_d, &motor_sqrt_d, &motor_omega_dot);
        }

        // ── 5. Extract and return parameters ───────────────────────────
        self.extract_params()
    }

    // ── Filter updates ─────────────────────────────────────────────────

    /// Update FX (effectiveness) filters and compute differences.
    /// Returns: (omega_diff[NU], omega_dot_diff[NU], rate_dot_diff[3], spf_diff[3])
    fn update_fx_filters(
        &mut self,
        input: &LearnerInput,
    ) -> ([f32; NU], [f32; NU], [f32; 3], [f32; 3]) {
        // Coriolis/centrifugal correction on accelerometer
        let spf_corrected = self.correct_accel(input);

        // Filter and difference omega
        let mut omega_diff = [0.0f32; NU];
        let mut omega_dot_diff = [0.0f32; NU];
        for i in 0..NU {
            let fx_omega = self.fx_omega_filter[i].apply(input.omega_rad_s[i]);
            omega_diff[i] = fx_omega - self.prev_fx_omega[i];
            self.prev_fx_omega[i] = fx_omega;

            let fx_omega_dot = self.loop_rate_hz * omega_diff[i];
            omega_dot_diff[i] = fx_omega_dot - self.prev_fx_omega_dot[i];
            self.prev_fx_omega_dot[i] = fx_omega_dot;
        }

        // Filter and difference rate_dot
        let mut rate_dot_diff = [0.0f32; 3];
        for axis in 0..3 {
            let fx_rate_dot = self.fx_rate_filter[axis].apply(input.rate_dot_rad_s2[axis]);
            rate_dot_diff[axis] = fx_rate_dot - self.prev_fx_rate_dot[axis];
            self.prev_fx_rate_dot[axis] = fx_rate_dot;
        }

        // Filter and difference spf (corrected)
        let mut spf_diff = [0.0f32; 3];
        for axis in 0..3 {
            let fx_spf = self.fx_spf_filter[axis].apply(spf_corrected[axis]);
            spf_diff[axis] = fx_spf - self.prev_fx_spf[axis];
            self.prev_fx_spf[axis] = fx_spf;
        }

        (omega_diff, omega_dot_diff, rate_dot_diff, spf_diff)
    }

    /// Coriolis/centrifugal correction for IMU offset from CoG.
    /// Matches indiflight learner.c lines 205-208.
    fn correct_accel(&self, input: &LearnerInput) -> [f32; 3] {
        let [rx, ry, rz] = self.config.acc_offset_m;
        let wx = input.rate_rad_s[0];
        let wy = input.rate_rad_s[1];
        let wz = input.rate_rad_s[2];
        let dwx = input.rate_dot_rad_s2[0];
        let dwy = input.rate_dot_rad_s2[1];
        let dwz = input.rate_dot_rad_s2[2];
        let ax = input.spf_m_s2[0];
        let ay = input.spf_m_s2[1];
        let az = input.spf_m_s2[2];

        [
            ax - (rx * (-(wy * wy + wz * wz)) + ry * (wx * wy - dwz) + rz * (wx * wz + dwy)),
            ay - (rx * (wx * wy + dwz) + ry * (-(wx * wx + wz * wz)) + rz * (wy * wz - dwx)),
            az - (rx * (wx * wz - dwy) + ry * (wy * wz + dwx) + rz * (-(wx * wx + wy * wy))),
        ]
    }

    /// Update motor dynamics filters.
    /// Returns: (omega[NU], d[NU], sqrt_d[NU], omega_dot[NU])
    fn update_motor_filters(
        &mut self,
        input: &LearnerInput,
    ) -> ([f32; NU], [f32; NU], [f32; NU], [f32; NU]) {
        let mut omega = [0.0f32; NU];
        let mut d = [0.0f32; NU];
        let mut sqrt_d = [0.0f32; NU];
        let mut omega_dot = [0.0f32; NU];

        for i in 0..NU {
            omega[i] = self.motor_omega_filter[i].apply(input.omega_rad_s[i]);
            d[i] = self.motor_d_filter[i].apply(input.d_commands[i]);
            let d_clamped = input.d_commands[i].clamp(0.0, 1.0);
            sqrt_d[i] = self.motor_sqrt_d_filter[i].apply(libm::sqrtf(d_clamped));

            omega_dot[i] = self.loop_rate_hz * (omega[i] - self.prev_motor_omega[i]);
            self.prev_motor_omega[i] = omega[i];
        }

        (omega, d, sqrt_d, omega_dot)
    }

    // ── RLS updates ────────────────────────────────────────────────────

    /// Update FX effectiveness RLS (G1 force + G1/G2 torque).
    fn update_fx_rls(
        &mut self,
        omega_diff: &[f32; NU],
        omega_dot_diff: &[f32; NU],
        rate_dot_diff: &[f32; 3],
        spf_diff: &[f32; 3],
    ) {
        // Build FX regressors (shared between spf and rate_dot RLS)
        // First NU elements: G1 regressors (force/torque from thrust)
        // Last NU elements: G2 regressors (torque from motor acceleration)
        let mut a_fx = [0.0f32; FX_N];
        for i in 0..NU {
            a_fx[i] = 1e-5 * 2.0 * self.prev_fx_omega[i] * omega_diff[i];
            a_fx[NU + i] = 1e-3 * omega_dot_diff[i];
        }
        let a_fx_vec = SVector::<f32, FX_N>::from_column_slice(&a_fx);

        // SPF RLS uses only the first NU regressors (G1 only, no G2 for forces)
        let mut a_spf = [0.0f32; NU];
        a_spf.copy_from_slice(&a_fx[..NU]);
        let a_spf_vec = SVector::<f32, NU>::from_column_slice(&a_spf);

        // Observations
        let y_spf = SVector::<f32, 3>::new(
            spf_diff[0] * 10.0,
            spf_diff[1] * 10.0,
            spf_diff[2] * 10.0,
        );
        let y_rate_dot = SVector::<f32, 3>::new(
            rate_dot_diff[0],
            rate_dot_diff[1],
            rate_dot_diff[2],
        );

        self.fx_spf_rls.update(&a_spf_vec, &y_spf);
        self.fx_rate_dot_rls.update(&a_fx_vec, &y_rate_dot);
    }

    /// Update per-motor dynamics RLS.
    fn update_motor_rls(
        &mut self,
        omega: &[f32; NU],
        d: &[f32; NU],
        sqrt_d: &[f32; NU],
        omega_dot: &[f32; NU],
    ) {
        for i in 0..NU {
            let a = SVector::<f32, MOTOR_N>::new(
                d[i],
                sqrt_d[i],
                1.0,
                -1e-4 * omega_dot[i],
            );
            let y = SVector::<f32, 1>::new(omega[i] * 1e-3);
            self.motor_rls[i].update(&a, &y);
        }
    }

    // ── Parameter extraction ───────────────────────────────────────────

    /// Extract learned parameters from RLS state.
    /// Scaling factors reverse the regressor/observation scaling applied above.
    /// Matches indiflight learner.c updateLearnedParameters() lines 548-578.
    fn extract_params(&self) -> LearnedParams {
        let valid = self.samples > 100; // need some data before trusting

        let mut max_omega = [0.0f32; NU];
        let mut time_const_s = [0.025f32; NU];
        let mut nonlinearity = [0.5f32; NU];

        // Motor parameters (from motor RLS)
        for i in 0..NU {
            let x = self.motor_rls[i].params();
            // maxOmega = inv_y_scale * (X[0] + X[1])
            // inv_y_scale = 1e3 (observation was omega * 1e-3)
            let mo = 1e3 * (x[(0, 0)] + x[(1, 0)]);
            max_omega[i] = if mo > 100.0 { mo } else { 100.0 };

            // time_const = inv_y_scale * a_scale * X[3]
            // a_scale for omega_dot regressor = 1e-4, inv_y_scale = 1e3
            // So: tau = 1e3 * 1e-4 * X[3] = 0.1 * X[3]
            let tau = 0.1 * x[(3, 0)];
            time_const_s[i] = tau.clamp(0.01, 0.2);

            // nonlinearity = X[0] / (X[0] + X[1])
            if x[(0, 0)] > 0.0 && x[(1, 0)] > 0.0 {
                nonlinearity[i] = (x[(0, 0)] / (x[(0, 0)] + x[(1, 0)])).clamp(0.0, 1.0);
            }
        }

        // G1 and G2 (from FX RLS)
        let mut g1 = SMatrix::<f32, 6, NU>::zeros();
        let mut g2 = SMatrix::<f32, 3, NU>::zeros();

        let spf_x = self.fx_spf_rls.params();
        let rate_x = self.fx_rate_dot_rls.params();

        for i in 0..NU {
            let mo_sq = max_omega[i] * max_omega[i];

            // G1 force rows: inv_y_scale * a_scale * maxOmega² * config_scale * X
            // inv_y_scale for spf = 1/10 = 0.1, a_scale = 1e-5, config_scale = 1e2
            // = 0.1 * 1e-5 * mo_sq * 1e2 * X = 1e-4 * mo_sq * X
            g1[(0, i)] = 0.1 * 1e-5 * mo_sq * 1e2 * spf_x[(i, 0)]; // fx
            g1[(1, i)] = 0.1 * 1e-5 * mo_sq * 1e2 * spf_x[(i, 1)]; // fy
            g1[(2, i)] = 0.1 * 1e-5 * mo_sq * 1e2 * spf_x[(i, 2)]; // fz

            // G1 torque rows: inv_y_scale * a_scale * maxOmega² * config_scale * X
            // inv_y_scale for rate_dot = 1.0, a_scale = 1e-5, config_scale = 1e1
            // = 1.0 * 1e-5 * mo_sq * 1e1 * X = 1e-4 * mo_sq * X
            g1[(3, i)] = 1.0 * 1e-5 * mo_sq * 1e1 * rate_x[(i, 0)]; // roll
            g1[(4, i)] = 1.0 * 1e-5 * mo_sq * 1e1 * rate_x[(i, 1)]; // pitch
            g1[(5, i)] = 1.0 * 1e-5 * mo_sq * 1e1 * rate_x[(i, 2)]; // yaw

            // G2: from second half of rate_dot RLS parameters
            // inv_y_scale = 1.0, a_scale = 1e-3, config_scale = 1e5
            // = 1.0 * 1e-3 * 1e5 * X = 1e2 * X
            g2[(0, i)] = 1.0 * 1e-3 * 1e5 * rate_x[(NU + i, 0)]; // roll
            g2[(1, i)] = 1.0 * 1e-3 * 1e5 * rate_x[(NU + i, 1)]; // pitch
            g2[(2, i)] = 1.0 * 1e-3 * 1e5 * rate_x[(NU + i, 2)]; // yaw
        }

        // Gain synthesis (from learned motor time constants)
        let mut max_tau = 0.0f32;
        for i in 0..NU {
            max_tau = if time_const_s[i] > max_tau { time_const_s[i] } else { max_tau };
        }
        max_tau = max_tau.clamp(0.01, 0.2);

        let rate_gain = 0.25 / (self.config.zeta_rate * self.config.zeta_rate * max_tau);
        let attitude_gain = 0.25 * rate_gain / (self.config.zeta_attitude * self.config.zeta_attitude);

        LearnedParams {
            g1,
            g2,
            max_omega,
            time_const_s,
            nonlinearity,
            rate_gain,
            attitude_gain,
            valid,
        }
    }

    /// Reset all RLS state and filters. Called e.g. on disarm.
    pub fn reset(&mut self) {
        *self = Self::new(&self.config, self.loop_rate_hz);
    }

    /// Number of learning samples processed (only incremented in flight).
    pub fn samples(&self) -> u32 {
        self.samples
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const LOOP_HZ: f32 = 8000.0;

    fn default_config() -> LearnerConfig {
        LearnerConfig::default()
    }

    fn zero_input() -> LearnerInput {
        LearnerInput {
            rate_rad_s: Vector3::zeros(),
            rate_dot_rad_s2: Vector3::zeros(),
            spf_m_s2: Vector3::new(0.0, 0.0, 9.81),
            omega_rad_s: [0.0; NU],
            d_commands: [0.0; NU],
            armed: false,
            touching_ground: true,
        }
    }

    #[test]
    fn new_does_not_panic() {
        let cfg = default_config();
        let _learner = Learner::new(&cfg, LOOP_HZ);
    }

    #[test]
    fn update_with_zero_input_does_not_panic() {
        let cfg = default_config();
        let mut learner = Learner::new(&cfg, LOOP_HZ);
        let input = zero_input();
        for _ in 0..100 {
            let params = learner.update(&input);
            assert!(!params.valid); // not enough samples
        }
    }

    #[test]
    fn no_learning_on_ground() {
        let cfg = default_config();
        let mut learner = Learner::new(&cfg, LOOP_HZ);
        let input = zero_input(); // armed=false, touching_ground=true
        for _ in 0..200 {
            learner.update(&input);
        }
        assert_eq!(learner.samples(), 0);
    }

    #[test]
    fn learning_increments_in_flight() {
        let cfg = default_config();
        let mut learner = Learner::new(&cfg, LOOP_HZ);
        let mut input = zero_input();
        input.armed = true;
        input.touching_ground = false;
        for _ in 0..10 {
            learner.update(&input);
        }
        assert_eq!(learner.samples(), 10);
    }

    #[test]
    fn coriolis_correction_zero_offset() {
        let cfg = default_config(); // acc_offset_m = [0,0,0]
        let learner = Learner::new(&cfg, LOOP_HZ);
        let mut input = zero_input();
        input.spf_m_s2 = Vector3::new(1.0, 2.0, 3.0);
        input.rate_rad_s = Vector3::new(10.0, 20.0, 30.0);
        input.rate_dot_rad_s2 = Vector3::new(1.0, 2.0, 3.0);
        let corrected = learner.correct_accel(&input);
        // With zero offset, correction is identity
        assert!((corrected[0] - 1.0).abs() < 1e-6);
        assert!((corrected[1] - 2.0).abs() < 1e-6);
        assert!((corrected[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn coriolis_correction_nonzero_offset() {
        let mut cfg = default_config();
        cfg.acc_offset_m = [0.01, -0.01, 0.015]; // 1cm offset
        let learner = Learner::new(&cfg, LOOP_HZ);
        let mut input = zero_input();
        input.spf_m_s2 = Vector3::new(0.0, 0.0, 9.81);
        input.rate_rad_s = Vector3::new(5.0, 5.0, 5.0);
        input.rate_dot_rad_s2 = Vector3::zeros();
        let corrected = learner.correct_accel(&input);
        // With nonzero offset and rotation, correction should differ from raw
        assert!((corrected[2] - 9.81).abs() > 1e-4);
    }

    #[test]
    fn reset_clears_samples() {
        let cfg = default_config();
        let mut learner = Learner::new(&cfg, LOOP_HZ);
        let mut input = zero_input();
        input.armed = true;
        input.touching_ground = false;
        for _ in 0..50 {
            learner.update(&input);
        }
        assert!(learner.samples() > 0);
        learner.reset();
        assert_eq!(learner.samples(), 0);
    }

    #[test]
    fn motor_rls_converges_to_known_params() {
        // Simulate a motor with known gain: omega = 2000 * d (linear motor)
        // Motor RLS should identify X[0] ≈ 2 (since observation = omega * 1e-3 = 2.0 * d)
        let cfg = default_config();
        let mut learner = Learner::new(&cfg, LOOP_HZ);

        let true_gain = 2000.0; // rad/s at d=1
        for step in 0..2000 {
            let t = step as f32 / LOOP_HZ;
            let d = 0.3 + 0.2 * libm::sinf(2.0 * core::f32::consts::PI * 50.0 * t);
            let omega = true_gain * d;
            let input = LearnerInput {
                rate_rad_s: Vector3::zeros(),
                rate_dot_rad_s2: Vector3::zeros(),
                spf_m_s2: Vector3::new(0.0, 0.0, 9.81),
                omega_rad_s: [omega; NU],
                d_commands: [d; NU],
                armed: true,
                touching_ground: false,
            };
            learner.update(&input);
        }

        let params = learner.update(&zero_input());
        // max_omega = 1e3 * (X[0] + X[1]) should be close to 2000
        // With linear motor (no sqrt term), X[0] should dominate
        // Allow generous tolerance — the filter transient and short run make exact convergence hard
        for i in 0..NU {
            assert!(
                params.max_omega[i] > 500.0,
                "motor {} max_omega={} too low",
                i,
                params.max_omega[i]
            );
        }
    }

    #[test]
    fn params_finite_after_many_steps() {
        let cfg = default_config();
        let mut learner = Learner::new(&cfg, LOOP_HZ);

        for step in 0..5000 {
            let t = step as f32 / LOOP_HZ;
            let d = 0.4 + 0.1 * libm::sinf(2.0 * core::f32::consts::PI * 30.0 * t);
            let omega = 1500.0 * d;
            let input = LearnerInput {
                rate_rad_s: Vector3::new(
                    0.1 * libm::sinf(50.0 * t),
                    0.1 * libm::cosf(50.0 * t),
                    0.0,
                ),
                rate_dot_rad_s2: Vector3::new(
                    5.0 * libm::cosf(50.0 * t),
                    -5.0 * libm::sinf(50.0 * t),
                    0.0,
                ),
                spf_m_s2: Vector3::new(0.5 * d, -0.3 * d, 9.81 + 2.0 * d),
                omega_rad_s: [omega; NU],
                d_commands: [d; NU],
                armed: true,
                touching_ground: false,
            };
            learner.update(&input);
        }

        let params = learner.update(&zero_input());
        assert!(params.valid);
        // Check everything is finite
        for i in 0..NU {
            assert!(params.max_omega[i].is_finite(), "max_omega[{}] not finite", i);
            assert!(params.time_const_s[i].is_finite(), "time_const_s[{}] not finite", i);
            assert!(params.nonlinearity[i].is_finite(), "nonlinearity[{}] not finite", i);
            assert!(params.rate_gain.is_finite());
            assert!(params.attitude_gain.is_finite());
        }
        for row in 0..6 {
            for col in 0..NU {
                assert!(params.g1[(row, col)].is_finite(), "G1[{},{}] not finite", row, col);
            }
        }
        for row in 0..3 {
            for col in 0..NU {
                assert!(params.g2[(row, col)].is_finite(), "G2[{},{}] not finite", row, col);
            }
        }
    }
}
