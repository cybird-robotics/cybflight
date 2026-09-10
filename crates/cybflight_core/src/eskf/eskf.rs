use nalgebra::{Matrix3, SMatrix, UnitQuaternion, Vector3};

use crate::rotation::hat;

/// Gravity in ENU frame [m/s²].
const GRAVITY_VEC: Vector3<f32> = Vector3::new(0.0, 0.0, -9.81);

/// Outcome of a measurement update.
///
/// `Accepted { inflated }` — the update was applied. `inflated == true`
/// means the innovation exceeded the gate and `R` was scaled up so the
/// gain is reduced (Huber-style robust update); the measurement still
/// pulls the state, just with smaller weight.
///
/// `Rejected` variants do **not** modify state. They exist so the caller
/// can drive failsafe logic (consecutive rejections → `ESTIMATOR_READY`
/// drop) and surface counts to telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOutcome {
    Accepted {
        inflated: bool,
    },
    NotInitialized,
    InverseFailed,
    InflationCapExceeded,
    NaNAfterUpdate,
    /// Innovation exceeded an absolute (covariance-independent) threshold.
    /// Triggered by Vicon rigid-body flips, frame re-associations, and
    /// other catastrophic glitches that the Mahalanobis gate can soft-accept
    /// once `R` is inflated. Must NOT be absorbed: flips can rotate the
    /// nominal orientation tens of degrees per frame and crash the vehicle.
    JumpRejected,
}

impl UpdateOutcome {
    pub fn is_accepted(self) -> bool {
        matches!(self, UpdateOutcome::Accepted { .. })
    }

    pub fn is_jump(self) -> bool {
        matches!(self, UpdateOutcome::JumpRejected)
    }
}

/// Hard cap on `R`-inflation factor. Beyond this, the measurement is
/// treated as a true outlier and dropped (Rejected) so the failsafe
/// can fire. With `gate_sigma = 5`, cap=100 means residuals up to
/// `sqrt(100) · 5σ = 50σ` are still partially absorbed; anything
/// larger is rejected. Catches meter-scale glitches while letting
/// cm-scale latency residuals through.
pub const DEFAULT_INFLATION_CAP: f32 = 100.0;

/// Default magnetometer norm gate: reject a sample whose field magnitude
/// deviates from the reference by more than this fraction.
pub const DEFAULT_MAG_NORM_GATE: f32 = 0.3;

/// ESKF noise / measurement configuration.
pub struct EskfConfig {
    /// Accelerometer noise density [m/s²/√Hz].
    pub accel_noise_density: f32,
    /// Gyroscope noise density [rad/s/√Hz].
    pub gyro_noise_density: f32,
    /// Accelerometer bias random walk [m/s²/√s].
    pub accel_bias_random_walk: f32,
    /// Gyroscope bias random walk [rad/s/√s].
    pub gyro_bias_random_walk: f32,
    /// Barometer altitude noise standard deviation [m].
    pub baro_noise_std: f32,
    /// Magnetometer field noise standard deviation [body-frame µT equivalent].
    pub mag_noise_std: f32,
    /// Outlier gate threshold in sigma units (per dimension).
    /// Measurement is rejected if zᵀ S⁻¹ z / dof > gate_sigma².
    pub gate_sigma: f32,
    /// Absolute position-innovation threshold [m]. Independent of filter
    /// covariance. Catches measurement-source faults (Vicon rigid-body
    /// re-association, RTK carrier-ambiguity loss, multipath wraparound)
    /// that the Mahalanobis gate would soft-accept under R-inflation.
    ///
    /// Default 1.0 m sized for mocap: worst-case flight × worst-case latency
    /// = 10 m/s × 100 ms. GPS at 5–10 Hz needs a looser value (~3 m) so the
    /// inter-frame residual at speed isn't hard-rejected.
    pub max_pos_jump_m: f32,
    /// Absolute attitude-innovation threshold [rad]. Independent of filter
    /// covariance. Catches Vicon orientation flips (~180°) and rigid-body
    /// re-associations.
    ///
    /// Default 0.7 rad (~40°) is 3× the worst plausible inter-frame attitude
    /// change at 1000°/s × 10 ms mocap period, well below the smallest
    /// dangerous flip (90° axis swap, 180° quaternion flip).
    pub max_att_jump_rad: f32,

    // ── Initial covariance P₀ ────────────────────────────────────────
    //
    // P₀ is not merely a transient: the arming gate compares
    // `gyro_bias_cov_trace()` — literally the P[12..15] diagonal seeded
    // here — against the guards' `gyro_bias_cov_trace_thresh`, so these
    // values and that threshold jointly set time-to-arm. An
    // over-confident P₀ also shrinks the innovation covariance
    // `S = HPHᵀ + R`, inflating normalized innovations and feeding the
    // guards' reject cascade. Both are reasons they are configuration,
    // not literals.
    /// Initial position variance [m²] on all three axes.
    pub init_pos_var: f32,
    /// Initial velocity variance [(m/s)²] on all three axes.
    pub init_vel_var: f32,
    /// Initial accelerometer-bias variance [(m/s²)²] per axis.
    pub init_accel_bias_var: f32,
    /// Initial gyro-bias variance [(rad/s)²] per axis. Sets where the
    /// convergence trace starts its decay toward the guard threshold.
    pub init_gyro_bias_var: f32,
    /// Initial roll/pitch orientation variance [rad²]. Both axes are
    /// observable from the gravity vector at bootstrap regardless of
    /// position source, so this is source-independent — unlike yaw,
    /// whose initial variance is owned by the per-source guard
    /// (`init_yaw_cov`) because GPS-only cannot observe it at all.
    pub init_att_var_rp: f32,
    /// Hard cap on the `R`-inflation factor. A residual that would need
    /// more inflation than this is treated as a true outlier and
    /// Rejected, so the failsafe can count it, instead of being absorbed.
    ///
    /// Read together with [`Self::gate_sigma`]: the pair decides how big
    /// a residual is still "latency", since residuals up to
    /// `sqrt(cap) · gate_sigma` σ are partially absorbed. Changing one
    /// silently changes what the other means.
    pub inflation_cap: f32,
    /// Magnetometer norm gate: reject a sample whose field magnitude
    /// differs from the world reference by more than this fraction.
    ///
    /// The only magnetometer quality check in the filter. A site with
    /// local ferrous distortion is the reason to tighten it.
    pub mag_norm_gate: f32,
}

impl Default for EskfConfig {
    fn default() -> Self {
        Self {
            accel_noise_density: 0.01,
            gyro_noise_density: 0.0001,
            accel_bias_random_walk: 0.001,
            gyro_bias_random_walk: 0.00001,
            baro_noise_std: 0.5,
            mag_noise_std: 0.05,
            gate_sigma: 10.0,
            max_pos_jump_m: 1.0,
            max_att_jump_rad: 0.7,
            init_pos_var: 1.0,
            init_vel_var: 1.0,
            init_accel_bias_var: 0.01,
            init_gyro_bias_var: 0.01,
            init_att_var_rp: 0.1,
            inflation_cap: DEFAULT_INFLATION_CAP,
            mag_norm_gate: DEFAULT_MAG_NORM_GATE,
        }
    }
}

impl EskfConfig {
    /// Explicit constructor. Lets callers (notably the host
    /// simulation) freeze the tuning baseline against future
    /// `Default` retunes — the firmware path through `est_pos_gps`
    /// builds an `EskfConfig` literally too, where the
    /// `max_pos_jump_m` field is widened to GPS-appropriate values.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        accel_noise_density: f32,
        gyro_noise_density: f32,
        accel_bias_random_walk: f32,
        gyro_bias_random_walk: f32,
        baro_noise_std: f32,
        mag_noise_std: f32,
        gate_sigma: f32,
        max_pos_jump_m: f32,
        max_att_jump_rad: f32,
        init_pos_var: f32,
        init_vel_var: f32,
        init_accel_bias_var: f32,
        init_gyro_bias_var: f32,
        init_att_var_rp: f32,
    ) -> Self {
        Self {
            accel_noise_density,
            gyro_noise_density,
            accel_bias_random_walk,
            gyro_bias_random_walk,
            baro_noise_std,
            mag_noise_std,
            gate_sigma,
            max_pos_jump_m,
            max_att_jump_rad,
            init_pos_var,
            init_vel_var,
            init_accel_bias_var,
            init_gyro_bias_var,
            init_att_var_rp,
            // Robustness knobs keep their defaults; a caller freezing a
            // baseline sets the fields directly.
            inflation_cap: DEFAULT_INFLATION_CAP,
            mag_norm_gate: DEFAULT_MAG_NORM_GATE,
        }
    }
}

/// 15-state nominal state (gravity treated as constant).
#[derive(Clone, Copy)]
pub struct NominalState {
    pub position: Vector3<f32>,
    pub orientation: UnitQuaternion<f32>,
    pub velocity: Vector3<f32>,
    pub accel_bias: Vector3<f32>,
    pub gyro_bias: Vector3<f32>,
}

impl Default for NominalState {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            velocity: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
        }
    }
}

impl NominalState {
    /// Apply a 15-D error-state increment (boxplus).
    fn boxplus(&self, dx: &SMatrix<f32, 15, 1>) -> Self {
        Self {
            position: self.position + dx.fixed_rows::<3>(0),
            orientation: self.orientation * UnitQuaternion::from_scaled_axis(dx.fixed_rows::<3>(3)),
            velocity: self.velocity + dx.fixed_rows::<3>(6),
            accel_bias: self.accel_bias + dx.fixed_rows::<3>(9),
            gyro_bias: self.gyro_bias + dx.fixed_rows::<3>(12),
        }
    }
}

/// Cumulative ESKF counters and last-NIS per channel.
#[derive(Default, Clone, Copy, Debug)]
pub struct EskfHealth {
    pub nan_resets: u32,
    pub gate_rejects_pos: u32,
    pub gate_rejects_vel: u32,
    pub gate_rejects_att: u32,
    pub gate_rejects_baro: u32,
    pub gate_rejects_mag: u32,
    pub last_nis_pos: f32,
    pub last_nis_vel: f32,
    pub last_nis_att: f32,
}

/// 15-state Error-State Kalman Filter.
///
/// State layout: [position(3), orientation_error(3), velocity(3),
///                accel_bias(3), gyro_bias(3)].
/// Covariance update uses the Joseph form for numerical stability.
pub struct Eskf {
    config: EskfConfig,
    state: NominalState,
    cov: SMatrix<f32, 15, 15>,
    initialized: bool,
    health: EskfHealth,
}

impl Eskf {
    /// The filter's noise/initialisation configuration.
    ///
    /// Exposed so a guard driving a re-initialisation can seed the new
    /// covariance from the same configured values the bootstrap used,
    /// instead of repeating literals that then drift from the schema.
    pub fn config(&self) -> &EskfConfig {
        &self.config
    }

    pub fn new(config: EskfConfig) -> Self {
        Self {
            config,
            state: NominalState::default(),
            cov: SMatrix::zeros(),
            initialized: false,
            health: EskfHealth::default(),
        }
    }

    /// Initialise filter with a known pose and sensor biases; resets covariance.
    /// Orientation diagonal is `config.init_att_var_rp` on all three axes —
    /// appropriate when every axis is observable from the bootstrap
    /// measurement (e.g. mocap pose). For exteroceptive sources that don't
    /// observe all axes (GPS-only has no yaw measurement), use
    /// `init_with_cov` to set per-axis values.
    pub fn init(
        &mut self,
        position: Vector3<f32>,
        orientation: UnitQuaternion<f32>,
        gyro_bias: Vector3<f32>,
        accel_bias: Vector3<f32>,
    ) {
        let v = self.config.init_att_var_rp;
        self.init_with_cov(
            position,
            orientation,
            gyro_bias,
            accel_bias,
            Vector3::new(v, v, v),
        );
    }

    /// Initialise with an explicit per-axis orientation covariance diagonal.
    /// Use this when one or more orientation axes is unobservable from the
    /// bootstrap measurement: the corresponding diagonal entry should be set
    /// large (e.g. ~10 = ~π² for a fully unknown yaw) so the filter doesn't
    /// claim convergence on a state it has no evidence about.
    pub fn init_with_cov(
        &mut self,
        position: Vector3<f32>,
        orientation: UnitQuaternion<f32>,
        gyro_bias: Vector3<f32>,
        accel_bias: Vector3<f32>,
        orientation_cov_diag: Vector3<f32>,
    ) {
        self.state = NominalState {
            position,
            orientation,
            velocity: Vector3::zeros(),
            accel_bias,
            gyro_bias,
        };
        let mut cov = SMatrix::<f32, 15, 15>::zeros();
        cov.fixed_view_mut::<3, 3>(0, 0)
            .fill_diagonal(self.config.init_pos_var);
        cov[(3, 3)] = orientation_cov_diag.x;
        cov[(4, 4)] = orientation_cov_diag.y;
        cov[(5, 5)] = orientation_cov_diag.z;
        cov.fixed_view_mut::<3, 3>(6, 6)
            .fill_diagonal(self.config.init_vel_var);
        cov.fixed_view_mut::<3, 3>(9, 9)
            .fill_diagonal(self.config.init_accel_bias_var);
        cov.fixed_view_mut::<3, 3>(12, 12)
            .fill_diagonal(self.config.init_gyro_bias_var);
        self.cov = cov;
        self.initialized = true;
        // Health counters are intentionally not reset — re-init after NaN is itself a tracked event.
    }

    /// Propagate nominal state and covariance forward by `dt`.
    fn propagate_state(
        &self,
        accel: Vector3<f32>,
        gyro: Vector3<f32>,
        dt: f32,
    ) -> (NominalState, SMatrix<f32, 15, 15>) {
        let s = &self.state;
        let a_ub = accel - s.accel_bias;
        let w_ub = gyro - s.gyro_bias;
        let rmat = *s.orientation.to_rotation_matrix().matrix();

        let new_state = NominalState {
            position: s.position + s.velocity * dt,
            orientation: s.orientation * UnitQuaternion::from_scaled_axis(w_ub * dt),
            velocity: s.velocity + (rmat * a_ub + GRAVITY_VEC) * dt,
            accel_bias: s.accel_bias,
            gyro_bias: s.gyro_bias,
        };

        let rot_neg = *UnitQuaternion::from_scaled_axis(-w_ub * dt)
            .to_rotation_matrix()
            .matrix();
        let mut f = SMatrix::<f32, 15, 15>::zeros();
        f.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        f.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(Matrix3::identity() * dt));
        f.fixed_view_mut::<3, 3>(3, 3).copy_from(&rot_neg);
        f.fixed_view_mut::<3, 3>(3, 12)
            .copy_from(&(-Matrix3::identity() * dt));
        f.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(-rmat * hat(&a_ub) * dt));
        f.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&Matrix3::identity());
        f.fixed_view_mut::<3, 3>(6, 9).copy_from(&(-rmat * dt));
        f.fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&Matrix3::identity());
        f.fixed_view_mut::<3, 3>(12, 12)
            .copy_from(&Matrix3::identity());

        let cfg = &self.config;
        let mut q = SMatrix::<f32, 15, 15>::zeros();
        q.fixed_view_mut::<3, 3>(3, 3)
            .fill_diagonal(cfg.gyro_noise_density * cfg.gyro_noise_density * dt);
        q.fixed_view_mut::<3, 3>(6, 6)
            .fill_diagonal(cfg.accel_noise_density * cfg.accel_noise_density * dt);
        q.fixed_view_mut::<3, 3>(9, 9)
            .fill_diagonal(cfg.accel_bias_random_walk * cfg.accel_bias_random_walk * dt);
        q.fixed_view_mut::<3, 3>(12, 12)
            .fill_diagonal(cfg.gyro_bias_random_walk * cfg.gyro_bias_random_walk * dt);

        let new_cov = f * self.cov * f.transpose() + q;
        (new_state, new_cov)
    }

    /// IMU predict step. Propagates nominal state and covariance forward.
    pub fn predict(&mut self, accel: Vector3<f32>, gyro: Vector3<f32>, dt: f32) {
        if !self.initialized {
            return;
        }
        let (new_state, new_cov) = self.propagate_state(accel, gyro, dt);
        self.state = new_state;
        self.cov = new_cov;
        // Prevent unit-quaternion magnitude drift from accumulating over
        // millions of predict steps in float32.
        self.renormalize_orientation();
        // Clamp covariance diagonal so one bad step can't push P into an
        // indefinite state that would produce Inf via matrix inversion.
        self.clamp_covariance_diagonal();
        // NaN guard — forces re-init on the next mocap frame.
        if !self.state_is_finite() {
            self.initialized = false;
            self.health.nan_resets = self.health.nan_resets.saturating_add(1);
        }
    }

    /// Block-sparse equivalent of `propagate_state`, exploiting F's
    /// block structure to avoid the dense 15×15·15×15·15×15 product.
    ///
    /// F has only 9 non-zero 3×3 blocks (5 trivial: identity / scaled-identity;
    /// 3 non-trivial: `R(-ω·dt)`, `-R·[a]× ·dt`, `-R·dt`; 1 zero remaining).
    /// Computing `F·P·Fᵀ` block-by-block reduces the work from ~6750 FMAs
    /// to ~1100 FMAs (~6× speedup) while producing the same result to
    /// within float32 round-off.
    ///
    /// Symmetry is enforced by construction: only the upper triangle is
    /// computed; the lower is mirrored. This is mathematically equivalent
    /// to the dense version (which also produces a symmetric result up
    /// to round-off) but eliminates the asymmetric round-off the dense
    /// matmul leaves behind.
    fn propagate_state_sparse(
        &self,
        accel: Vector3<f32>,
        gyro: Vector3<f32>,
        dt: f32,
    ) -> (NominalState, SMatrix<f32, 15, 15>) {
        let s = &self.state;
        let a_ub = accel - s.accel_bias;
        let w_ub = gyro - s.gyro_bias;
        let rmat = *s.orientation.to_rotation_matrix().matrix();

        let new_state = NominalState {
            position: s.position + s.velocity * dt,
            orientation: s.orientation * UnitQuaternion::from_scaled_axis(w_ub * dt),
            velocity: s.velocity + (rmat * a_ub + GRAVITY_VEC) * dt,
            accel_bias: s.accel_bias,
            gyro_bias: s.gyro_bias,
        };

        // Non-trivial F blocks.
        let r_neg = *UnitQuaternion::from_scaled_axis(-w_ub * dt)
            .to_rotation_matrix()
            .matrix(); // R(-ω·dt)
        let m_a = -rmat * hat(&a_ub) * dt; // F's (vel, att) block
        let m_b = -rmat * dt; // F's (vel, ba) block
        let d = dt;

        // Extract P's upper-triangle blocks once (lower comes from transposes).
        let p_pp = self.cov.fixed_view::<3, 3>(0, 0).clone_owned();
        let p_pa = self.cov.fixed_view::<3, 3>(0, 3).clone_owned();
        let p_pv = self.cov.fixed_view::<3, 3>(0, 6).clone_owned();
        let p_pba = self.cov.fixed_view::<3, 3>(0, 9).clone_owned();
        let p_pbg = self.cov.fixed_view::<3, 3>(0, 12).clone_owned();
        let p_aa = self.cov.fixed_view::<3, 3>(3, 3).clone_owned();
        let p_av = self.cov.fixed_view::<3, 3>(3, 6).clone_owned();
        let p_aba = self.cov.fixed_view::<3, 3>(3, 9).clone_owned();
        let p_abg = self.cov.fixed_view::<3, 3>(3, 12).clone_owned();
        let p_vv = self.cov.fixed_view::<3, 3>(6, 6).clone_owned();
        let p_vba = self.cov.fixed_view::<3, 3>(6, 9).clone_owned();
        let p_vbg = self.cov.fixed_view::<3, 3>(6, 12).clone_owned();
        let p_baba = self.cov.fixed_view::<3, 3>(9, 9).clone_owned();
        let p_babg = self.cov.fixed_view::<3, 3>(9, 12).clone_owned();
        let p_bgbg = self.cov.fixed_view::<3, 3>(12, 12).clone_owned();

        // Reused intermediates from FP[i, l] = sum_k F[i,k] · P[k,l].
        // FP[0, l] = P[0,l] + d · P[2,l]
        let fp0_a = p_pa + d * p_av.transpose(); // FP[0, 1]
        let fp0_v = p_pv + d * p_vv; // FP[0, 2]
        let fp0_ba = p_pba + d * p_vba; // FP[0, 3]
        let fp0_bg = p_pbg + d * p_vbg; // FP[0, 4]

        // FP[1, l] = R⁻ · P[1,l] − d · P[4,l]
        let r_paa = r_neg * p_aa;
        let r_pav = r_neg * p_av;
        let r_paba = r_neg * p_aba;
        let r_pabg = r_neg * p_abg;
        // FP[1, 1] = R⁻·P_AA − d·P_ABGᵀ
        let fp1_a = r_paa - d * p_abg.transpose();
        let fp1_v = r_pav - d * p_vbg.transpose();
        let fp1_ba = r_paba - d * p_babg.transpose();
        let fp1_bg = r_pabg - d * p_bgbg;

        // FP[2, l] = M_a · P[1,l] + P[2,l] + M_b · P[3,l]
        let ma_paa = m_a * p_aa;
        let ma_pav = m_a * p_av;
        let ma_paba = m_a * p_aba;
        let ma_pabg = m_a * p_abg;
        // FP[2, 1] = M_a·P_AA + P_AVᵀ + M_b·P_ABAᵀ
        let fp2_a = ma_paa + p_av.transpose() + m_b * p_aba.transpose();
        let fp2_v = ma_pav + p_vv + m_b * p_vba.transpose();
        let fp2_ba = ma_paba + p_vba + m_b * p_baba;
        let fp2_bg = ma_pabg + p_vbg + m_b * p_babg;

        // FP[3, l] = P[3, l]; FP[4, l] = P[4, l] (identity F block).

        // P'[i, j] = sum_l FP[i,l] · F[j,l]ᵀ.
        // For column j, only F[j, l] non-zero contributions matter.

        // Upper-triangle output blocks.

        // P'[0, 0] = FP[0, 0] + d · FP[0, 2]
        let pp_pp = (p_pp + d * p_pv.transpose()) + d * fp0_v;

        // P'[0, 1] = FP[0, 1] · R⁻ᵀ − d · FP[0, 4]
        let pp_pa = fp0_a * r_neg.transpose() - d * fp0_bg;

        // P'[0, 2] = FP[0, 1] · M_aᵀ + FP[0, 2] + FP[0, 3] · M_bᵀ
        let pp_pv = fp0_a * m_a.transpose() + fp0_v + fp0_ba * m_b.transpose();

        // P'[0, 3] = FP[0, 3]
        let pp_pba = fp0_ba;

        // P'[0, 4] = FP[0, 4]
        let pp_pbg = fp0_bg;

        // P'[1, 1] = FP[1, 1] · R⁻ᵀ − d · FP[1, 4]
        let pp_aa = fp1_a * r_neg.transpose() - d * fp1_bg;

        // P'[1, 2] = FP[1, 1] · M_aᵀ + FP[1, 2] + FP[1, 3] · M_bᵀ
        let pp_av = fp1_a * m_a.transpose() + fp1_v + fp1_ba * m_b.transpose();

        // P'[1, 3] = FP[1, 3]
        let pp_aba = fp1_ba;

        // P'[1, 4] = FP[1, 4]
        let pp_abg = fp1_bg;

        // P'[2, 2] = FP[2, 1] · M_aᵀ + FP[2, 2] + FP[2, 3] · M_bᵀ
        let pp_vv = fp2_a * m_a.transpose() + fp2_v + fp2_ba * m_b.transpose();

        // P'[2, 3] = FP[2, 3]
        let pp_vba = fp2_ba;

        // P'[2, 4] = FP[2, 4]
        let pp_vbg = fp2_bg;

        // P'[3, 3] = P_BABA  (identity block in F leaves it unchanged)
        let pp_baba = p_baba;
        let pp_babg = p_babg;
        let pp_bgbg = p_bgbg;

        // Q is diagonal; add to (att, vel, ba, bg) diagonal blocks.
        let cfg = &self.config;
        let q_aa = cfg.gyro_noise_density * cfg.gyro_noise_density * dt;
        let q_vv = cfg.accel_noise_density * cfg.accel_noise_density * dt;
        let q_baba = cfg.accel_bias_random_walk * cfg.accel_bias_random_walk * dt;
        let q_bgbg = cfg.gyro_bias_random_walk * cfg.gyro_bias_random_walk * dt;
        let pp_aa = pp_aa + Matrix3::from_diagonal_element(q_aa);
        let pp_vv = pp_vv + Matrix3::from_diagonal_element(q_vv);
        let pp_baba = pp_baba + Matrix3::from_diagonal_element(q_baba);
        let pp_bgbg = pp_bgbg + Matrix3::from_diagonal_element(q_bgbg);

        // Reassemble. Lower triangle is the transpose of the upper.
        let mut new_cov = SMatrix::<f32, 15, 15>::zeros();
        // Row 0 (pos)
        new_cov.fixed_view_mut::<3, 3>(0, 0).copy_from(&pp_pp);
        new_cov.fixed_view_mut::<3, 3>(0, 3).copy_from(&pp_pa);
        new_cov.fixed_view_mut::<3, 3>(0, 6).copy_from(&pp_pv);
        new_cov.fixed_view_mut::<3, 3>(0, 9).copy_from(&pp_pba);
        new_cov.fixed_view_mut::<3, 3>(0, 12).copy_from(&pp_pbg);
        // Row 1 (att)
        new_cov.fixed_view_mut::<3, 3>(3, 0).copy_from(&pp_pa.transpose());
        new_cov.fixed_view_mut::<3, 3>(3, 3).copy_from(&pp_aa);
        new_cov.fixed_view_mut::<3, 3>(3, 6).copy_from(&pp_av);
        new_cov.fixed_view_mut::<3, 3>(3, 9).copy_from(&pp_aba);
        new_cov.fixed_view_mut::<3, 3>(3, 12).copy_from(&pp_abg);
        // Row 2 (vel)
        new_cov.fixed_view_mut::<3, 3>(6, 0).copy_from(&pp_pv.transpose());
        new_cov.fixed_view_mut::<3, 3>(6, 3).copy_from(&pp_av.transpose());
        new_cov.fixed_view_mut::<3, 3>(6, 6).copy_from(&pp_vv);
        new_cov.fixed_view_mut::<3, 3>(6, 9).copy_from(&pp_vba);
        new_cov.fixed_view_mut::<3, 3>(6, 12).copy_from(&pp_vbg);
        // Row 3 (ba)
        new_cov.fixed_view_mut::<3, 3>(9, 0).copy_from(&pp_pba.transpose());
        new_cov.fixed_view_mut::<3, 3>(9, 3).copy_from(&pp_aba.transpose());
        new_cov.fixed_view_mut::<3, 3>(9, 6).copy_from(&pp_vba.transpose());
        new_cov.fixed_view_mut::<3, 3>(9, 9).copy_from(&pp_baba);
        new_cov.fixed_view_mut::<3, 3>(9, 12).copy_from(&pp_babg);
        // Row 4 (bg)
        new_cov.fixed_view_mut::<3, 3>(12, 0).copy_from(&pp_pbg.transpose());
        new_cov.fixed_view_mut::<3, 3>(12, 3).copy_from(&pp_abg.transpose());
        new_cov.fixed_view_mut::<3, 3>(12, 6).copy_from(&pp_vbg.transpose());
        new_cov.fixed_view_mut::<3, 3>(12, 9).copy_from(&pp_babg.transpose());
        new_cov.fixed_view_mut::<3, 3>(12, 12).copy_from(&pp_bgbg);

        (new_state, new_cov)
    }

    /// Block-sparse predict step. Equivalent to `predict` to within
    /// float32 round-off; faster on the H743's M7 because F's structural
    /// zeros are skipped instead of multiplied through a dense matmul.
    pub fn predict_sparse(&mut self, accel: Vector3<f32>, gyro: Vector3<f32>, dt: f32) {
        if !self.initialized {
            return;
        }
        let (new_state, new_cov) = self.propagate_state_sparse(accel, gyro, dt);
        self.state = new_state;
        self.cov = new_cov;
        self.renormalize_orientation();
        self.clamp_covariance_diagonal();
        if !self.state_is_finite() {
            self.initialized = false;
        }
    }

    /// Joint pose measurement update (position + attitude in one step).
    ///
    /// Mocap delivers position and orientation at the same instant from the
    /// same rigid-body solve, so they share a measurement timestamp and
    /// their noise is independent. Applying them as a single 6-D update
    /// (rather than two sequential 3-D updates) preserves the cross-
    /// covariance pathway during gain computation: the position correction
    /// and the attitude correction both come from the *same* prior P,
    /// instead of the attitude update operating on a P that has already
    /// been collapsed by the position update.
    ///
    /// Numerically equivalent to `update_pos` followed by `update_att`
    /// only in the limit of infinite precision and zero inflation; in
    /// practice the joint update is sharper, especially during aggressive
    /// flight where the residuals are correlated.
    ///
    /// Absolute jump gates are checked per-component (position and attitude
    /// independently) — a flip on one channel rejects the whole frame.
    pub fn update_pose(
        &mut self,
        pos: Vector3<f32>,
        q: UnitQuaternion<f32>,
        pos_std: f32,
        att_std: f32,
    ) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }

        // Position residual + jump gate.
        let z_pos = pos - self.state.position;
        if z_pos.norm() > self.config.max_pos_jump_m {
            return UpdateOutcome::JumpRejected;
        }

        // Attitude residual: canonicalize the quaternion to the same
        // hemisphere as the estimate, then the scaled-axis of the error
        // quaternion gives a 3-D rotation vector innovation.
        let q = if self.state.orientation.coords.dot(&q.coords) < 0.0 {
            UnitQuaternion::from_quaternion(-q.into_inner())
        } else {
            q
        };
        let q_err = self.state.orientation.inverse() * q;
        let z_att = q_err.scaled_axis();
        if z_att.norm() > self.config.max_att_jump_rad {
            return UpdateOutcome::JumpRejected;
        }

        // Stack the 6-D innovation: [Δp; Δθ].
        let mut z = SMatrix::<f32, 6, 1>::zeros();
        z.fixed_rows_mut::<3>(0).copy_from(&z_pos);
        z.fixed_rows_mut::<3>(3).copy_from(&z_att);

        // H = [I_3  0  0  0  0;
        //       0  I_3 0  0  0]   (rows: pos, att; cols: pos, att, vel, ba, bg)
        let mut h = SMatrix::<f32, 6, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        h.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&Matrix3::identity());

        // R is block-diagonal: top-left 3×3 = pos_std² · I, bottom-right = att_std² · I.
        let pv = pos_std * pos_std;
        let av = att_std * att_std;
        let mut r = SMatrix::<f32, 6, 6>::zeros();
        r.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(pv);
        r.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(av);

        let hph_t = h * self.cov * h.transpose();
        let s_mat = hph_t + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };

        // Mahalanobis² normalised by 6 DoF.
        let gamma = (z.transpose() * s_inv * z)[(0, 0)] / 6.0;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = hph_t + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };

        let k = self.cov * h.transpose() * s_inv_eff;
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * r_eff * k.transpose();
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        // Boxplus rotated the orientation; renormalize so multiplicative
        // round-off doesn't leave a non-unit quaternion.
        self.renormalize_orientation();
        if !self.state_is_finite() {
            self.initialized = false;
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// Block-sparse equivalent of `update_pose`.
    ///
    /// H is sparse: only two 3×3 identity blocks at columns (0, 3) of the
    /// 15-state. Exploiting this lets us avoid materializing `K·H` and
    /// `(I − KH)` as full 15×15 matrices, replacing two 15·15·15 matmuls
    /// with structurally smaller 15×6·6×15 matmuls.
    ///
    /// Specifically:
    ///   - `H·P` is the first 6 rows of P (no matmul).
    ///   - `P·Hᵀ` is the first 6 columns of P (no matmul).
    ///   - Joseph form `(I−KH)·P·(I−KH)ᵀ` = `P − K·(HP) − (PHᵀ)·Kᵀ + K·(HPHᵀ)·Kᵀ`,
    ///     which equals `B − B·Hᵀ·Kᵀ + K·R·Kᵀ` where `B = P − K·HP`.
    ///
    /// FMA count drops from ~16500 to ~4800 — about 3.5× speedup. Result
    /// is equivalent to `update_pose` to within float32 round-off.
    pub fn update_pose_sparse(
        &mut self,
        pos: Vector3<f32>,
        q: UnitQuaternion<f32>,
        pos_std: f32,
        att_std: f32,
    ) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }

        let z_pos = pos - self.state.position;
        if z_pos.norm() > self.config.max_pos_jump_m {
            return UpdateOutcome::JumpRejected;
        }

        let q = if self.state.orientation.coords.dot(&q.coords) < 0.0 {
            UnitQuaternion::from_quaternion(-q.into_inner())
        } else {
            q
        };
        let q_err = self.state.orientation.inverse() * q;
        let z_att = q_err.scaled_axis();
        if z_att.norm() > self.config.max_att_jump_rad {
            return UpdateOutcome::JumpRejected;
        }

        let mut z = SMatrix::<f32, 6, 1>::zeros();
        z.fixed_rows_mut::<3>(0).copy_from(&z_pos);
        z.fixed_rows_mut::<3>(3).copy_from(&z_att);

        // HPHᵀ is the upper-left 6×6 of P (since H selects the pos and
        // att blocks of P). No matmul needed.
        let hph_t: SMatrix<f32, 6, 6> = self.cov.fixed_view::<6, 6>(0, 0).clone_owned();

        let pv = pos_std * pos_std;
        let av = att_std * att_std;
        let mut r = SMatrix::<f32, 6, 6>::zeros();
        r.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(pv);
        r.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(av);

        let s_mat = hph_t + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };

        let gamma = (z.transpose() * s_inv * z)[(0, 0)] / 6.0;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = hph_t + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };

        // K = P·Hᵀ·S⁻¹ = (first 6 cols of P) · S⁻¹ → 15×6.
        let p_ht: SMatrix<f32, 15, 6> = self.cov.fixed_view::<15, 6>(0, 0).clone_owned();
        let k: SMatrix<f32, 15, 6> = p_ht * s_inv_eff;

        // State update.
        self.state = self.state.boxplus(&(k * z));

        // Joseph form, sparse.
        // H·P is the first 6 rows of P (no matmul).
        let hp: SMatrix<f32, 6, 15> = self.cov.fixed_view::<6, 15>(0, 0).clone_owned();
        // B = P − K·(H·P)
        let b = self.cov - k * hp;
        // B·Hᵀ is the first 6 cols of B.
        let b_ht: SMatrix<f32, 15, 6> = b.fixed_view::<15, 6>(0, 0).clone_owned();
        // P' = B − B·Hᵀ·Kᵀ + K·R·Kᵀ
        let kt = k.transpose();
        self.cov = b - b_ht * kt + k * r_eff * kt;

        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        self.renormalize_orientation();
        if !self.state_is_finite() {
            self.initialized = false;
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// Position measurement update (ENU). `pos_std` is std-dev in metres.
    pub fn update_pos(&mut self, pos: Vector3<f32>, pos_std: f32) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }
        let z = pos - self.state.position;
        // Absolute jump gate — runs before the Mahalanobis/inflation logic
        // so a frame-swap can't be soft-accepted via R inflation.
        if z.norm() > self.config.max_pos_jump_m {
            return UpdateOutcome::JumpRejected;
        }
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        let pv = pos_std * pos_std;
        let r = Matrix3::identity() * pv;
        let s_mat = h * self.cov * h.transpose() + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            // Pathological covariance — force re-init from next measurement.
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };
        // Innovation gate (per-DOF Mahalanobis²) decides whether to apply
        // the update directly or down-weight by inflating R. A measurement
        // that would have been silently dropped is now partially absorbed,
        // preventing dead-reckoning during sustained large innovations.
        let gamma = z.dot(&(s_inv * z)) / 3.0;
        self.health.last_nis_pos = gamma;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                self.health.gate_rejects_pos =
                    self.health.gate_rejects_pos.saturating_add(1);
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = h * self.cov * h.transpose() + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };
        let k = self.cov * h.transpose() * s_inv_eff;
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * r_eff * k.transpose();
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        if !self.state_is_finite() {
            self.initialized = false;
            self.health.nan_resets = self.health.nan_resets.saturating_add(1);
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// Block-sparse equivalent of `update_pos`.
    ///
    /// H is sparse: a single 3×3 identity block at columns (0..3) of the
    /// 15-state. Exploiting this lets us avoid materializing `K·H` and
    /// `(I − KH)` as full 15×15 matrices, replacing two 15·15·15 matmuls
    /// with structurally smaller 15×3·3×15 matmuls.
    ///
    /// Specifically:
    ///   - `H·P` is the first 3 rows of P (no matmul).
    ///   - `P·Hᵀ` is the first 3 columns of P (no matmul).
    ///   - `H·P·Hᵀ` is the upper-left 3×3 of P (no matmul).
    ///   - Joseph form `(I−KH)·P·(I−KH)ᵀ` = `P − K·(HP) − (PHᵀ)·Kᵀ + K·(HPHᵀ)·Kᵀ`,
    ///     which equals `B − B·Hᵀ·Kᵀ + K·R·Kᵀ` where `B = P − K·HP`.
    ///
    /// Mirrors `update_pose_sparse`'s shape with the attitude block
    /// removed; the 6-DoF z and innovation gating shrink to 3-DoF.
    /// Result is equivalent to `update_pos` to within float32 round-off.
    pub fn update_pos_sparse(&mut self, pos: Vector3<f32>, pos_std: f32) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }

        let z = pos - self.state.position;
        // Absolute jump gate — runs before the Mahalanobis/inflation logic
        // so a frame-swap can't be soft-accepted via R inflation. Mirrors
        // the dense update_pos.
        if z.norm() > self.config.max_pos_jump_m {
            return UpdateOutcome::JumpRejected;
        }

        // HPHᵀ is the upper-left 3×3 of P (since H selects the pos block).
        // No matmul needed.
        let hph_t: Matrix3<f32> = self.cov.fixed_view::<3, 3>(0, 0).clone_owned();

        let pv = pos_std * pos_std;
        let r = Matrix3::identity() * pv;

        let s_mat = hph_t + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };

        // Mahalanobis² normalised by 3 DoF, evaluated on the un-inflated
        // S so the gate decision matches the dense update_pos exactly.
        let gamma = z.dot(&(s_inv * z)) / 3.0;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = hph_t + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };

        // K = P·Hᵀ·S⁻¹ = (first 3 cols of P) · S⁻¹ → 15×3.
        let p_ht: SMatrix<f32, 15, 3> = self.cov.fixed_view::<15, 3>(0, 0).clone_owned();
        let k: SMatrix<f32, 15, 3> = p_ht * s_inv_eff;

        // State update. Boxplus interprets the orientation rows of `k·z`
        // as a rotation vector and rotates the orientation accordingly,
        // so attitude can move via cross-covariance even though the
        // measurement is position-only — same behaviour as update_pos.
        self.state = self.state.boxplus(&(k * z));

        // Joseph form, sparse.
        // H·P is the first 3 rows of P (no matmul).
        let hp: SMatrix<f32, 3, 15> = self.cov.fixed_view::<3, 15>(0, 0).clone_owned();
        // B = P − K·(H·P)
        let b = self.cov - k * hp;
        // B·Hᵀ is the first 3 cols of B.
        let b_ht: SMatrix<f32, 15, 3> = b.fixed_view::<15, 3>(0, 0).clone_owned();
        // P' = B − B·Hᵀ·Kᵀ + K·R·Kᵀ
        let kt = k.transpose();
        self.cov = b - b_ht * kt + k * r_eff * kt;

        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        // Match dense `update_pos`: no `renormalize_orientation` call.
        // (`update_pose` and `update_pose_sparse` do renormalize because
        // the attitude measurement induces O(1) δθ; here δθ comes only
        // through cross-covariance and is small enough that the dense
        // form skips it. Keep the two paths bit-identical.)
        if !self.state_is_finite() {
            self.initialized = false;
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// Velocity measurement update (ENU). `vel_std` is std-dev in m/s.
    ///
    /// NOTE: GNSS velocity is fed raw, at the ANT1 phase centre — there is no
    /// lever-arm ω×r compensation to the IMU frame. The error is small for a
    /// short baseline / low body rates; TODO: add the ω×r term if a high-rate
    /// platform shows velocity fighting the IMU.
    pub fn update_vel(&mut self, vel: Vector3<f32>, vel_std: f32) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }
        let z = vel - self.state.velocity;
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&Matrix3::identity());
        let vv = vel_std * vel_std;
        let r = Matrix3::identity() * vv;
        let s_mat = h * self.cov * h.transpose() + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };
        let gamma = z.dot(&(s_inv * z)) / 3.0;
        self.health.last_nis_vel = gamma;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                self.health.gate_rejects_vel =
                    self.health.gate_rejects_vel.saturating_add(1);
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = h * self.cov * h.transpose() + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };
        let k = self.cov * h.transpose() * s_inv_eff;
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * r_eff * k.transpose();
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        if !self.state_is_finite() {
            self.initialized = false;
            self.health.nan_resets = self.health.nan_resets.saturating_add(1);
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// Attitude measurement update. `att_std` is std-dev in radians.
    pub fn update_att(&mut self, q: UnitQuaternion<f32>, att_std: f32) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }
        // Canonicalize q to the same hemisphere as the current estimate.
        let q = if self.state.orientation.coords.dot(&q.coords) < 0.0 {
            UnitQuaternion::from_quaternion(-q.into_inner())
        } else {
            q
        };
        let q_err = self.state.orientation.inverse() * q;
        let z = q_err.scaled_axis();
        // Absolute attitude-jump gate — catches Vicon rigid-body flips
        // (~180°) which the Mahalanobis gate soft-accepts under inflation.
        // Must run before any inflation logic so the threshold is independent
        // of filter overconfidence.
        if z.norm() > self.config.max_att_jump_rad {
            return UpdateOutcome::JumpRejected;
        }
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());
        let av = att_std * att_std;
        let r = Matrix3::identity() * av;
        let s_mat = h * self.cov * h.transpose() + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };
        let gamma = z.dot(&(s_inv * z)) / 3.0;
        self.health.last_nis_att = gamma;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                self.health.gate_rejects_att =
                    self.health.gate_rejects_att.saturating_add(1);
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = h * self.cov * h.transpose() + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };
        let k = self.cov * h.transpose() * s_inv_eff;
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * r_eff * k.transpose();
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        // The boxplus applied a rotation correction — renormalize so
        // round-off in the multiplication doesn't leave a non-unit q.
        self.renormalize_orientation();
        if !self.state_is_finite() {
            self.initialized = false;
            self.health.nan_resets = self.health.nan_resets.saturating_add(1);
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// 1D barometric altitude update: observes position.z only.
    pub fn update_altitude(&mut self, altitude_m: f32) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }
        let z = altitude_m - self.state.position.z;
        let mut h = SMatrix::<f32, 1, 15>::zeros();
        h[(0, 2)] = 1.0;
        let r = self.config.baro_noise_std * self.config.baro_noise_std;
        let s = (h * self.cov * h.transpose())[(0, 0)] + r;
        let gamma = z * z / s;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                self.health.gate_rejects_baro =
                    self.health.gate_rejects_baro.saturating_add(1);
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            (r_eff, (h * self.cov * h.transpose())[(0, 0)] + r_eff, true)
        } else {
            (r, s, false)
        };
        let k = self.cov * h.transpose() * (1.0 / s_eff);
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * k.transpose() * r_eff;
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        if !self.state_is_finite() {
            self.initialized = false;
            self.health.nan_resets = self.health.nan_resets.saturating_add(1);
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// 3D magnetometer body-frame update.
    ///
    /// `mag_body` is the measured field in body frame; `mag_world_ref` is the
    /// reference field vector in world frame (bootstrapped from first measurement).
    /// Rejects updates where the field norm deviates more than 30% from reference
    /// (hard-iron distortion gate).
    pub fn update_mag(
        &mut self,
        mag_body: Vector3<f32>,
        mag_world_ref: Vector3<f32>,
    ) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }
        let world_norm = mag_world_ref.norm();
        if (mag_body.norm() - world_norm).abs() > self.config.mag_norm_gate * world_norm {
            self.health.gate_rejects_mag =
                self.health.gate_rejects_mag.saturating_add(1);
            return UpdateOutcome::InflationCapExceeded;
        }
        let m_b = self.state.orientation.inverse() * mag_world_ref;
        let z = mag_body - m_b;
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 3).copy_from(&hat(&m_b));
        let r = Matrix3::identity() * (self.config.mag_noise_std * self.config.mag_noise_std);
        let s_mat = h * self.cov * h.transpose() + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };
        let gamma = z.dot(&(s_inv * z)) / 3.0;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                self.health.gate_rejects_mag =
                    self.health.gate_rejects_mag.saturating_add(1);
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = h * self.cov * h.transpose() + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };
        let k = self.cov * h.transpose() * s_inv_eff;
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * r_eff * k.transpose();
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        self.renormalize_orientation();
        if !self.state_is_finite() {
            self.initialized = false;
            self.health.nan_resets = self.health.nan_resets.saturating_add(1);
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// Dual-antenna heading update (GPS yaw aiding).
    ///
    /// A body-fixed unit baseline `b_body` (e.g. `(0,-1,0)` for the
    /// ANT1→ANT2 vector of a left/right-mounted pair) is measured as a
    /// **world-frame** unit direction `b_world_meas`, reconstructed from the
    /// receiver's heading + pitch. `sigma_rad` is the angular 1-σ of that
    /// direction.
    ///
    /// Unlike a scalar-yaw update, this predicts the baseline in the world
    /// frame, so `H_att = -R·[b_body]×` — rank 2 at **every** attitude (it is
    /// blind only to rotation *about* the baseline, never to yaw specifically).
    /// There is no `quaternion_to_yaw` singularity through vertical / inverted
    /// flight, which makes it the correct heading aid for acrobatic use. It
    /// makes yaw observable in GPS-only builds.
    ///
    /// The measured direction is renormalised so its (unobservable) radial
    /// component injects no spurious innovation.
    pub fn update_baseline(
        &mut self,
        b_world_meas: Vector3<f32>,
        b_body: Vector3<f32>,
        sigma_heading_rad: f32,
        sigma_pitch_rad: f32,
    ) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }
        let Some(b_meas) = b_world_meas.try_normalize(1e-6) else {
            return UpdateOutcome::InverseFailed;
        };
        let b_pred = self.state.orientation * b_body;
        let z = b_meas - b_pred;
        let rmat = *self.state.orientation.to_rotation_matrix().matrix();
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-rmat * hat(&b_body)));
        // Anisotropic 2-axis measurement covariance. A single 3-D baseline is a
        // rank-2 attitude measurement: heading noise acts along the azimuth
        // tangent `u × b` (magnitude cos(elevation), so it vanishes smoothly as
        // the baseline nears vertical, where azimuth is genuinely undetermined),
        // pitch noise along the elevation tangent. The radial axis carries no
        // information (H annihilates it: H^T·b_pred = hat(b_body)·b_body = 0), so
        // it gets a moderate isotropic regularizer — large enough the 2nd-order
        // radial innovation can't inflate the NIS, irrelevant to the state update.
        const SIGMA_FLOOR: f32 = 1e-3;
        const RADIAL_SIGMA: f32 = 0.1;
        const COS_ELEV_MIN_SQ: f32 = 0.02;
        let sh = sigma_heading_rad.max(SIGMA_FLOOR);
        let sp = sigma_pitch_rad.max(SIGMA_FLOOR);
        let up = Vector3::new(0.0_f32, 0.0, 1.0);
        let az = up.cross(&b_meas); // azimuth tangent; |az| = cos(elevation)
        let n2 = az.norm_squared();
        let reg2 = RADIAL_SIGMA * RADIAL_SIGMA;
        let r = if n2 > COS_ELEV_MIN_SQ {
            let el = b_meas.cross(&az).normalize(); // unit elevation tangent (no_std: avoid f32::sqrt)
            az * az.transpose() * (sh * sh)
                + el * el.transpose() * (sp * sp)
                + b_meas * b_meas.transpose() * reg2
        } else {
            // Baseline ~vertical: azimuth tangent degenerate -> isotropic.
            Matrix3::identity() * (sp * sp).max(reg2)
        };
        let s_mat = h * self.cov * h.transpose() + r;
        let Some(s_inv) = s_mat.try_inverse() else {
            self.initialized = false;
            return UpdateOutcome::InverseFailed;
        };
        let gamma = z.dot(&(s_inv * z)) / 3.0;
        self.health.last_nis_att = gamma;
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > self.config.inflation_cap {
                self.health.gate_rejects_att = self.health.gate_rejects_att.saturating_add(1);
                return UpdateOutcome::InflationCapExceeded;
            }
            let r_eff = r * inflate;
            let s_eff = h * self.cov * h.transpose() + r_eff;
            let Some(s_inv_eff) = s_eff.try_inverse() else {
                self.initialized = false;
                return UpdateOutcome::InverseFailed;
            };
            (r_eff, s_inv_eff, true)
        } else {
            (r, s_inv, false)
        };
        let k = self.cov * h.transpose() * s_inv_eff;
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * r_eff * k.transpose();
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
        self.clamp_covariance_diagonal();
        self.renormalize_orientation();
        if !self.state_is_finite() {
            self.initialized = false;
            self.health.nan_resets = self.health.nan_resets.saturating_add(1);
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    pub fn position(&self) -> Vector3<f32> {
        self.state.position
    }

    pub fn velocity(&self) -> Vector3<f32> {
        self.state.velocity
    }

    pub fn orientation(&self) -> UnitQuaternion<f32> {
        self.state.orientation
    }

    pub fn accel_bias(&self) -> Vector3<f32> {
        self.state.accel_bias
    }

    pub fn gyro_bias(&self) -> Vector3<f32> {
        self.state.gyro_bias
    }

    pub fn covariance(&self) -> &SMatrix<f32, 15, 15> {
        &self.cov
    }

    /// Sum of the three diagonal gyro-bias covariance entries (indices 12–14).
    pub fn gyro_bias_cov_trace(&self) -> f32 {
        self.cov[(12, 12)] + self.cov[(13, 13)] + self.cov[(14, 14)]
    }

    /// Sum of the x and y gyro-bias covariance entries (indices 12, 13).
    /// Use this for convergence checks when yaw is unobservable from the
    /// available measurements (e.g. GPS-only with no magnetometer or
    /// course-over-ground update): the z-axis bias variance never decreases
    /// and would otherwise prevent `gyro_bias_cov_trace` from ever crossing
    /// the threshold.
    pub fn gyro_bias_cov_trace_xy(&self) -> f32 {
        self.cov[(12, 12)] + self.cov[(13, 13)]
    }

    pub fn pos_cov_trace(&self) -> f32 {
        self.cov[(0, 0)] + self.cov[(1, 1)] + self.cov[(2, 2)]
    }

    pub fn vel_cov_trace(&self) -> f32 {
        self.cov[(6, 6)] + self.cov[(7, 7)] + self.cov[(8, 8)]
    }

    pub fn health(&self) -> EskfHealth {
        self.health
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Minimum variance floor for covariance diagonal.
    /// Prevents pathological overconfidence that leads to indefinite P.
    const MIN_VAR: f32 = 1e-10;

    /// Clamp each diagonal entry of `cov` to at least `MIN_VAR`.
    /// Float32 round-off in the Joseph update can drive diagonals negative;
    /// this floor keeps P positive-definite on the diagonal.
    fn clamp_covariance_diagonal(&mut self) {
        for i in 0..15 {
            if self.cov[(i, i)] < Self::MIN_VAR {
                self.cov[(i, i)] = Self::MIN_VAR;
            }
        }
    }

    /// Re-normalize the orientation quaternion.
    /// Protects against magnitude drift from many multiplications in float32.
    fn renormalize_orientation(&mut self) {
        self.state.orientation = UnitQuaternion::new_normalize(self.state.orientation.into_inner());
    }

    /// `true` iff every component of the nominal state is finite.
    pub fn state_is_finite(&self) -> bool {
        let s = &self.state;
        let q = s.orientation.as_vector();
        s.position.iter().all(|v| v.is_finite())
            && s.velocity.iter().all(|v| v.is_finite())
            && s.accel_bias.iter().all(|v| v.is_finite())
            && s.gyro_bias.iter().all(|v| v.is_finite())
            && q.x.is_finite()
            && q.y.is_finite()
            && q.z.is_finite()
            && q.w.is_finite()
    }
}

#[cfg(test)]
mod health_tests {
    use super::*;

    fn fresh_eskf() -> Eskf {
        let mut e = Eskf::new(EskfConfig::default());
        e.init(
            Vector3::zeros(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
        );
        e
    }

    #[test]
    fn fresh_filter_has_zero_counters() {
        let e = fresh_eskf();
        let h = e.health();
        assert_eq!(h.nan_resets, 0);
        assert_eq!(h.gate_rejects_pos, 0);
        assert_eq!(h.gate_rejects_vel, 0);
        assert_eq!(h.gate_rejects_att, 0);
        assert_eq!(h.gate_rejects_baro, 0);
        assert_eq!(h.gate_rejects_mag, 0);
    }

    #[test]
    fn outlier_position_increments_gate_reject_pos() {
        // 0.5m offset is below `MAX_POS_JUMP_M = 1.0` (so the absolute-jump
        // gate doesn't pre-empt this), yet far enough above the tightened
        // covariance that gamma >> gate²·INFLATION_CAP and the inflation
        // path rejects via `gate_rejects_pos`.
        let mut e = fresh_eskf();
        // Tighten cov[0,0] from 1.0 to ~σ² = 1e-6 so a sub-jump-gate offset
        // still clears the inflation cap.
        let _ = e.update_pos(Vector3::zeros(), 0.001);
        e.update_pos(Vector3::new(0.5, 0.0, 0.0), 0.001);
        let h = e.health();
        assert_eq!(h.gate_rejects_pos, 1);
        assert!(h.last_nis_pos > e.config.gate_sigma * e.config.gate_sigma);
    }

    #[test]
    fn inlier_position_does_not_increment_gate_reject() {
        // 1 cm offset against σ=0.5m measurement: well inside the gate.
        let mut e = fresh_eskf();
        e.update_pos(Vector3::new(0.01, 0.0, 0.0), 0.5);
        let h = e.health();
        assert_eq!(h.gate_rejects_pos, 0);
        assert!(h.last_nis_pos < e.config.gate_sigma * e.config.gate_sigma);
    }

    #[test]
    fn nan_input_predict_increments_nan_resets() {
        let mut e = fresh_eskf();
        // Feed a NaN-laden accel; predict should detect it post-propagation.
        e.predict(
            Vector3::new(f32::NAN, 0.0, 0.0),
            Vector3::zeros(),
            0.001,
        );
        assert_eq!(e.health().nan_resets, 1);
        assert!(!e.is_initialized());
    }

    #[test]
    fn pos_cov_trace_reflects_init_diag() {
        // init() sets P[0..3,0..3] diag = 1.0 each, so trace = 3.0.
        let e = fresh_eskf();
        assert!((e.pos_cov_trace() - 3.0).abs() < 1e-6);
        assert!((e.vel_cov_trace() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn baro_outlier_increments_baro_counter() {
        let mut e = fresh_eskf();
        e.update_altitude(10_000.0); // 10 km, way past gate.
        assert_eq!(e.health().gate_rejects_baro, 1);
    }

    // ---- dual-antenna heading (update_baseline) ----

    /// A baseline measurement consistent with a yawed truth pulls the filter's
    /// orientation toward that truth (yaw becomes observable).
    #[test]
    fn baseline_update_corrects_yaw_toward_truth() {
        let b_body = Vector3::new(0.0, -1.0, 0.0); // ANT1(L)->ANT2(R)
        let mut e = fresh_eskf(); // q_nom = identity
        let q_true = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5236); // +30° yaw
        let b_meas = q_true * b_body; // world direction the true attitude produces
        let err_before = e.orientation().angle_to(&q_true);
        let out = e.update_baseline(b_meas, b_body, 0.02, 0.02);
        let err_after = e.orientation().angle_to(&q_true);
        assert!(matches!(out, UpdateOutcome::Accepted { .. }));
        assert!(
            err_after < err_before,
            "attitude error should shrink toward truth: {err_before} -> {err_after}"
        );
    }

    /// The key acro property: at +90° pitch (nose vertical) — where a scalar
    /// `quaternion_to_yaw` update is singular — the vector update is still
    /// finite, accepted, and corrects toward truth.
    #[test]
    fn baseline_update_nonsingular_at_vertical_nose() {
        let b_body = Vector3::new(0.0, -1.0, 0.0);
        let q_pitch =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), core::f32::consts::FRAC_PI_2);
        let mut e = Eskf::new(EskfConfig::default());
        e.init(Vector3::zeros(), q_pitch, Vector3::zeros(), Vector3::zeros());
        // Truth: nose-up AND yawed 20° about world-up (a heading error).
        let q_true = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.349) * q_pitch;
        let b_meas = q_true * b_body;
        let err_before = e.orientation().angle_to(&q_true);
        let out = e.update_baseline(b_meas, b_body, 0.02, 0.02);
        let err_after = e.orientation().angle_to(&q_true);
        assert!(
            matches!(out, UpdateOutcome::Accepted { .. }),
            "must not blow up at vertical nose"
        );
        assert!(
            e.orientation().into_inner().coords.iter().all(|c| c.is_finite()),
            "orientation must stay finite at vertical nose"
        );
        assert!(
            err_after < err_before,
            "error should shrink even at 90° pitch: {err_before} -> {err_after}"
        );
    }

    /// A measurement equal to the prediction leaves the state untouched.
    #[test]
    fn baseline_update_perfect_measurement_is_noop() {
        let b_body = Vector3::new(0.0, -1.0, 0.0);
        let mut e = fresh_eskf();
        let q0 = e.orientation();
        let b_meas = q0 * b_body; // exactly the predicted world direction
        let out = e.update_baseline(b_meas, b_body, 0.02, 0.02);
        assert!(matches!(out, UpdateOutcome::Accepted { .. }));
        assert!(
            e.orientation().angle_to(&q0) < 1e-4,
            "perfect measurement should not move the state"
        );
    }

    /// 90° roll puts the left-right baseline vertical: the azimuth tangent
    /// collapses, so the 2-axis R takes its isotropic fallback. The update must
    /// stay finite and accepted (no divide-by-zero at the pole).
    #[test]
    fn baseline_update_vertical_baseline_fallback_is_finite() {
        let b_body = Vector3::new(0.0, -1.0, 0.0);
        let q_roll =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), core::f32::consts::FRAC_PI_2);
        let mut e = Eskf::new(EskfConfig::default());
        e.init(Vector3::zeros(), q_roll, Vector3::zeros(), Vector3::zeros());
        let q_true = UnitQuaternion::from_axis_angle(
            &Vector3::x_axis(),
            core::f32::consts::FRAC_PI_2 + 0.1,
        );
        let b_meas = q_true * b_body;
        let out = e.update_baseline(b_meas, b_body, 0.02, 0.02);
        assert!(
            matches!(out, UpdateOutcome::Accepted { .. }),
            "vertical-baseline fallback must be accepted"
        );
        assert!(
            e.orientation().into_inner().coords.iter().all(|c| c.is_finite()),
            "orientation must stay finite in the fallback path"
        );
    }
}
