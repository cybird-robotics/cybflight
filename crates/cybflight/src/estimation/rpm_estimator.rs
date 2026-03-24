use core::ops::Deref;

use heapless::Deque;
use nalgebra::{Matrix2, Vector2};

#[derive(Debug, Clone, Copy)]
pub enum RpmEstimatorError {
    NonPositiveNoiseCovariance,
    NonPositiveTimeConstant,
    NonPositiveNISGate,
    OutlierDetected(f32),
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizedThrottle(f32);

impl NormalizedThrottle {
    pub fn new(value: f32) -> Option<Self> {
        (0.0..=1.0).contains(&value).then_some(Self(value))
    }

    pub fn new_clamped(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Deref for NormalizedThrottle {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ThrottleStamped {
    pub timestamp: f32,
    pub u: NormalizedThrottle,
}

#[derive(Debug, Clone, Copy)]
pub struct StateAndCov {
    state: Vector2<f32>,
    covariance: Matrix2<f32>,
}

impl StateAndCov {
    pub fn new(omega: f32, c_m: f32, covariance: Matrix2<f32>) -> Self {
        Self {
            state: Vector2::new(omega, c_m),
            covariance,
        }
    }

    pub fn omega(&self) -> f32 {
        self.state[0]
    }

    pub fn c_m(&self) -> f32 {
        self.state[1]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RpmEstimatorConfig {
    tau_m_up: f32,
    tau_m_down: f32,
    tau_d: f32,
    throttle_noise_cov: f32,
    c_m_noise_cov: f32,
    omega_noise_cov: f32,
    /// NIS gate threshold (chi-squared, 1 DOF). Measurements with innov² / s_cov above this are
    /// rejected. Typical values: 3.84 (95th percentile), 9.0 (~3σ, 99.7th percentile).
    nis_gate: f32,
}

#[derive(Debug, Clone)]
pub struct RpmEstimatorConfigBuilder {
    tau_m_up: Option<f32>,
    tau_m_down: Option<f32>,
    tau_d: Option<f32>,
    throttle_noise_cov: Option<f32>,
    c_m_noise_cov: Option<f32>,
    omega_noise_cov: Option<f32>,
    nis_gate: Option<f32>,
}

impl RpmEstimatorConfigBuilder {
    pub fn new() -> Self {
        Self {
            tau_m_up: None,
            tau_m_down: None,
            tau_d: None,
            throttle_noise_cov: None,
            c_m_noise_cov: None,
            omega_noise_cov: None,
            nis_gate: None,
        }
    }

    pub fn tau_m_up(mut self, value: f32) -> Self {
        self.tau_m_up = Some(value);
        self
    }

    pub fn tau_m_down(mut self, value: f32) -> Self {
        self.tau_m_down = Some(value);
        self
    }

    pub fn tau_d(mut self, value: f32) -> Self {
        self.tau_d = Some(value);
        self
    }

    pub fn throttle_noise_cov(mut self, value: f32) -> Self {
        self.throttle_noise_cov = Some(value);
        self
    }

    pub fn c_m_noise_cov(mut self, value: f32) -> Self {
        self.c_m_noise_cov = Some(value);
        self
    }

    pub fn omega_noise_cov(mut self, value: f32) -> Self {
        self.omega_noise_cov = Some(value);
        self
    }

    pub fn nis_gate(mut self, value: f32) -> Self {
        self.nis_gate = Some(value);
        self
    }

    pub fn build(self) -> Result<RpmEstimatorConfig, RpmEstimatorError> {
        let tau_m_up = self.tau_m_up.unwrap_or(0.033);
        if tau_m_up <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveTimeConstant);
        }
        let tau_m_down = self.tau_m_down.unwrap_or(tau_m_up);
        if tau_m_down <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveTimeConstant);
        }
        let tau_d = self.tau_d.unwrap_or(0.0);
        if tau_d < 0.0 {
            return Err(RpmEstimatorError::NonPositiveTimeConstant);
        }
        let throttle_noise_cov = self.throttle_noise_cov.unwrap_or(0.01);
        if throttle_noise_cov <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNoiseCovariance);
        }
        let c_m_noise_cov = self.c_m_noise_cov.unwrap_or(0.1);
        if c_m_noise_cov <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNoiseCovariance);
        }
        let omega_noise_cov = self.omega_noise_cov.unwrap_or(100.0);
        if omega_noise_cov <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNoiseCovariance);
        }
        let nis_gate = self.nis_gate.unwrap_or(9.0);
        if nis_gate <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNISGate);
        }

        Ok(RpmEstimatorConfig {
            tau_m_up,
            tau_m_down,
            tau_d,
            throttle_noise_cov,
            c_m_noise_cov,
            omega_noise_cov,
            nis_gate,
        })
    }
}

impl Default for RpmEstimatorConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_HISTORY: usize = 10;

pub struct RpmEstimator {
    config: RpmEstimatorConfig,
    u_history: Deque<ThrottleStamped, MAX_HISTORY>,
    posterior: StateAndCov,
}

impl RpmEstimator {
    pub fn new(config: RpmEstimatorConfig, initial_state: StateAndCov) -> Self {
        Self {
            config,
            u_history: Deque::new(),
            posterior: initial_state,
        }
    }

    pub fn push_throttle(&mut self, timestamp: f32, u: NormalizedThrottle) {
        if self.u_history.len() == MAX_HISTORY {
            self.u_history.pop_front();
        }
        self.u_history
            .push_back(ThrottleStamped { timestamp, u })
            .ok();
    }

    pub fn state(&self) -> StateAndCov {
        self.posterior
    }

    fn inv_tau_m(&self, is_accelerating: bool) -> f32 {
        // Branchless selection of the appropriate time constant based on acceleration direction
        [1.0 / self.config.tau_m_down, 1.0 / self.config.tau_m_up][is_accelerating as usize]
    }

    fn predict(&self, post: StateAndCov, u: f32, dt: f32) -> StateAndCov {
        let StateAndCov { state, covariance } = post;
        let [omega, c_m] = state.into();
        let delta_omega = c_m * u - omega;
        let inv_tau_m = self.inv_tau_m(delta_omega > 0.0);
        // Discrete-time state transition Jacobian
        let f_mat = Matrix2::new(1.0 - dt * inv_tau_m, dt * u * inv_tau_m, 0.0, 1.0);
        let qcov = Matrix2::new(
            (dt * c_m * inv_tau_m) * (dt * c_m * inv_tau_m) * self.config.throttle_noise_cov,
            0.0,
            0.0,
            dt * self.config.c_m_noise_cov,
        );

        StateAndCov {
            state: [omega + delta_omega * inv_tau_m * dt, c_m].into(),
            covariance: f_mat * covariance * f_mat.transpose() + qcov,
        }
    }

    fn update(&self, prio: StateAndCov, y: f32) -> Result<StateAndCov, RpmEstimatorError> {
        let StateAndCov { state, covariance } = prio;

        let innov = y - state[0];
        // Optimization: H = [1, 0].
        // 1. H * P * H.T == H[0, 0],
        // 2. H * P == P[0, :]
        // 3. K * H.T = [K, zeros((2, 1))], I - K = [[1-K[0], 0], [-K[1], 1]]
        let s_cov = covariance[(0, 0)] + self.config.omega_noise_cov;
        if innov * innov > self.config.nis_gate * s_cov {
            return Err(RpmEstimatorError::OutlierDetected(innov));
        }
        let k_gain = covariance.column(0) / s_cov;
        let mut state = state + k_gain * innov;
        state[1] = state[1].max(0.0); // Enforce non-negative c_m

        let i_m_kh = Matrix2::new(1.0 - k_gain[0], 0.0, -k_gain[1], 1.0);
        let covariance = i_m_kh * covariance * i_m_kh.transpose()
            + k_gain * self.config.omega_noise_cov * k_gain.transpose();
        Ok(StateAndCov { state, covariance })
    }

    pub fn step(
        &mut self,
        current_timestamp: f32,
        dt: f32,
        y_meas: Option<f32>, // Some(rpm) if a valid DShot packet arrived, None otherwise
    ) {
        // 1. FOPDT: Find the command from tau_d seconds ago
        let target_time = current_timestamp - self.config.tau_d;

        // Iterate backwards. Find the newest command that is older than our target time.

        let delayed_u = if let Some(found) = self
            .u_history
            .iter()
            .rfind(|entry| entry.timestamp <= target_time)
        {
            found.u.value()
        } else {
            // Fallback if history is too short: use the oldest available
            self.u_history.front().map(|e| e.u.value()).unwrap_or(0.0)
        };

        // 2. Predict the state forward using the delayed input
        let prior = self.predict(self.posterior, delayed_u, dt);

        // 3. Update if we received valid telemetry on this tick
        self.posterior = if let Some(y) = y_meas {
            // Optional: Add innovation gating here to reject extreme outliers
            self.update(prior, y).unwrap_or(prior)
        } else {
            // No telemetry? Trust the model.
            prior
        };
    }
}
