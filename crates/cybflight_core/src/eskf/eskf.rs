use nalgebra::{Matrix3, SMatrix, UnitQuaternion, Vector3};

use crate::rotation::hat;

/// Gravity in ENU frame [m/s²].
const GRAVITY_VEC: Vector3<f32> = Vector3::new(0.0, 0.0, -9.81);

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
            gyro_noise_density: 0.001,
            accel_bias_random_walk: 0.0001,
            gyro_bias_random_walk: 0.00001,
            baro_noise_std: 0.5,
            mag_noise_std: 0.05,
            gate_sigma: 5.0,
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
        cov.fixed_view_mut::<3, 3>(12, 12).fill_diagonal(0.001); // gyro bias
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
    }

    /// Position measurement update (ENU). `pos_std` is std-dev in metres.
    pub fn update_pos(&mut self, pos: Vector3<f32>, pos_std: f32) {
        if !self.initialized {
            return;
        }
        let z = pos - self.state.position;
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        let pv = pos_std * pos_std;
        let r = Matrix3::identity() * pv;
        let s_mat = h * self.cov * h.transpose() + r;
        if let Some(s_inv) = s_mat.try_inverse() {
            let k = self.cov * h.transpose() * s_inv;
            self.state = self.state.boxplus(&(k * z));
            let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
            self.cov = i_kh * self.cov * i_kh.transpose() + k * r * k.transpose();
            self.cov = (self.cov + self.cov.transpose()) * 0.5;
        }
    }

    /// Velocity measurement update (ENU). `vel_std` is std-dev in m/s.
    pub fn update_vel(&mut self, vel: Vector3<f32>, vel_std: f32) {
        if !self.initialized {
            return;
        }
        let z = vel - self.state.velocity;
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&Matrix3::identity());
        let vv = vel_std * vel_std;
        let r = Matrix3::identity() * vv;
        let s_mat = h * self.cov * h.transpose() + r;
        if let Some(s_inv) = s_mat.try_inverse() {
            let k = self.cov * h.transpose() * s_inv;
            self.state = self.state.boxplus(&(k * z));
            let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
            self.cov = i_kh * self.cov * i_kh.transpose() + k * r * k.transpose();
            self.cov = (self.cov + self.cov.transpose()) * 0.5;
        }
    }

    /// Attitude measurement update. `att_std` is std-dev in radians.
    pub fn update_att(&mut self, q: UnitQuaternion<f32>, att_std: f32) {
        if !self.initialized {
            return;
        }
        let q_err = self.state.orientation.inverse() * q;
        let z = q_err.scaled_axis();
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());
        let av = att_std * att_std;
        let r = Matrix3::identity() * av;
        let s_mat = h * self.cov * h.transpose() + r;
        if let Some(s_inv) = s_mat.try_inverse() {
            let k = self.cov * h.transpose() * s_inv;
            self.state = self.state.boxplus(&(k * z));
            let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
            self.cov = i_kh * self.cov * i_kh.transpose() + k * r * k.transpose();
            self.cov = (self.cov + self.cov.transpose()) * 0.5;
        }
    }

    /// 1D barometric altitude update: observes position.z only.
    pub fn update_altitude(&mut self, altitude_m: f32) {
        if !self.initialized {
            return;
        }
        let z = altitude_m - self.state.position.z;
        let mut h = SMatrix::<f32, 1, 15>::zeros();
        h[(0, 2)] = 1.0;
        let r = self.config.baro_noise_std * self.config.baro_noise_std;
        let s = (h * self.cov * h.transpose())[(0, 0)] + r;
        if z * z / s > self.config.gate_sigma * self.config.gate_sigma {
            return;
        }
        let k = self.cov * h.transpose() * (1.0 / s);
        self.state = self.state.boxplus(&(k * z));
        let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
        self.cov = i_kh * self.cov * i_kh.transpose() + k * k.transpose() * r;
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
    }

    /// 3D magnetometer body-frame update.
    ///
    /// `mag_body` is the measured field in body frame; `mag_world_ref` is the
    /// reference field vector in world frame (bootstrapped from first measurement).
    /// Rejects updates where the field norm deviates more than 30% from reference
    /// (hard-iron distortion gate).
    pub fn update_mag(&mut self, mag_body: Vector3<f32>, mag_world_ref: Vector3<f32>) {
        if !self.initialized {
            return;
        }
        let world_norm = mag_world_ref.norm();
        if (mag_body.norm() - world_norm).abs() > 0.3 * world_norm {
            return;
        }
        let m_b = self.state.orientation.inverse() * mag_world_ref;
        let z = mag_body - m_b;
        let mut h = SMatrix::<f32, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 3).copy_from(&hat(&m_b));
        let r = Matrix3::identity() * (self.config.mag_noise_std * self.config.mag_noise_std);
        let s_mat = h * self.cov * h.transpose() + r;
        if let Some(s_inv) = s_mat.try_inverse() {
            let gamma = z.dot(&(s_inv * z)) / 3.0;
            if gamma > self.config.gate_sigma * self.config.gate_sigma {
                return;
            }
            let k = self.cov * h.transpose() * s_inv;
            self.state = self.state.boxplus(&(k * z));
            let i_kh = SMatrix::<f32, 15, 15>::identity() - k * h;
            self.cov = i_kh * self.cov * i_kh.transpose() + k * r * k.transpose();
            self.cov = (self.cov + self.cov.transpose()) * 0.5;
        }
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

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}
