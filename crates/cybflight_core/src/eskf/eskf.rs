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
const INFLATION_CAP: f32 = 100.0;

/// Absolute position-innovation threshold [m]. Independent of filter
/// covariance. Catches Vicon rigid-body re-association (frame swap to a
/// different body in the volume) which can produce meter-scale jumps
/// the Mahalanobis gate would soft-accept under inflation.
///
/// Sized for worst-case flight × worst-case latency: 10 m/s × 100 ms = 1.0 m.
/// Real flight residuals at typical envelope are cm-scale, so this gate
/// is comfortably above legitimate measurements and far below frame swaps.
const MAX_POS_JUMP_M: f32 = 1.0;

/// Absolute attitude-innovation threshold [rad]. Independent of filter
/// covariance. Catches Vicon orientation flips (~180°) and rigid-body
/// re-associations.
///
/// 30° is 3× the worst plausible inter-frame attitude change at 1000°/s
/// gyro rate × 10 ms mocap period (=10°), and far below the smallest
/// dangerous flip (90° axis swap, 180° quaternion flip).
const MAX_ATT_JUMP_RAD: f32 = 0.7;

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
            gate_sigma: 7.0,
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
}

impl Eskf {
    pub fn new(config: EskfConfig) -> Self {
        Self {
            config,
            state: NominalState::default(),
            cov: SMatrix::zeros(),
            initialized: false,
        }
    }

    /// Initialise filter with a known pose and sensor biases; resets covariance.
    pub fn init(
        &mut self,
        position: Vector3<f32>,
        orientation: UnitQuaternion<f32>,
        gyro_bias: Vector3<f32>,
        accel_bias: Vector3<f32>,
    ) {
        self.state = NominalState {
            position,
            orientation,
            velocity: Vector3::zeros(),
            accel_bias,
            gyro_bias,
        };
        let mut cov = SMatrix::<f32, 15, 15>::zeros();
        cov.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(1.0); // position
        cov.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(0.1); // orientation
        cov.fixed_view_mut::<3, 3>(6, 6).fill_diagonal(1.0); // velocity
        cov.fixed_view_mut::<3, 3>(9, 9).fill_diagonal(0.01); // accel bias
        cov.fixed_view_mut::<3, 3>(12, 12).fill_diagonal(0.01); // gyro bias
        self.cov = cov;
        self.initialized = true;
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
        if z_pos.norm() > MAX_POS_JUMP_M {
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
        if z_att.norm() > MAX_ATT_JUMP_RAD {
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
            if inflate > INFLATION_CAP {
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

    /// Position measurement update (ENU). `pos_std` is std-dev in metres.
    pub fn update_pos(&mut self, pos: Vector3<f32>, pos_std: f32) -> UpdateOutcome {
        if !self.initialized {
            return UpdateOutcome::NotInitialized;
        }
        let z = pos - self.state.position;
        // Absolute jump gate — runs before the Mahalanobis/inflation logic
        // so a frame-swap can't be soft-accepted via R inflation.
        if z.norm() > MAX_POS_JUMP_M {
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
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > INFLATION_CAP {
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
            return UpdateOutcome::NaNAfterUpdate;
        }
        UpdateOutcome::Accepted { inflated }
    }

    /// Velocity measurement update (ENU). `vel_std` is std-dev in m/s.
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
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > INFLATION_CAP {
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
        if z.norm() > MAX_ATT_JUMP_RAD {
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
        let gate_sq = self.config.gate_sigma * self.config.gate_sigma;
        let (r_eff, s_inv_eff, inflated) = if gamma > gate_sq {
            let inflate = gamma / gate_sq;
            if inflate > INFLATION_CAP {
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
            if inflate > INFLATION_CAP {
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
        if (mag_body.norm() - world_norm).abs() > 0.3 * world_norm {
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
            if inflate > INFLATION_CAP {
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
