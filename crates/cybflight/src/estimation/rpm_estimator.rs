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

/// What [`RpmEstimator::step`] did with this tick's measurement.
///
/// Returned so the caller can account for rejections instead of losing
/// them: a sustained run of [`StepOutcome::Rejected`] means telemetry *is*
/// arriving but the model and the measurements disagree — the filter is
/// coasting open-loop while looking healthy from the outside. That is a
/// different failure from [`StepOutcome::Coasted`], which just means no
/// measurement was offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// No measurement this tick; the posterior is a pure model prediction.
    Coasted,
    /// Measurement fused into the posterior.
    Updated,
    /// Measurement rejected by the NIS gate; the posterior is the prediction.
    Rejected,
    /// Measurement failed the hard validity checks (zero/implausible eRPM
    /// decode); the posterior is the prediction. Distinct from `Rejected`:
    /// validity failures are expected ESC behavior and never feed the
    /// rejection-escape counter.
    Invalid,
    /// The rejection escape fired: after `rpm_est_escape_rejects`
    /// consecutive NIS rejections the filter re-seeded ω from the
    /// measurement (wide ω covariance, `c_m` kept). Without this a filter
    /// whose model mispredicts can reject every measurement indefinitely —
    /// each rejection skips the very update that would fix the model, so
    /// the innovation only grows (observed in flight logs: 35–96 % of a
    /// time-optimal mission rejected, ω̂ coasting 2–4× above raw eRPM).
    Reseeded,
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
    /// Throttle-curve nonlinearity `k`: steady-state speed follows
    /// `ω_ss = c_m·√(k·u² + (1−k)·u)` (the identified actuator curve;
    /// same `k` as the `m*_nonlin` thrust-curve parameter, since
    /// thrust ∝ ω²). `1.0` degenerates to the previous linear `c_m·u`
    /// model. A linear model whose gain is adapted at cruise throttle
    /// overpredicts the top of a concave curve by ~50 % — the seed of the
    /// rejection spiral this filter used to fall into.
    curve_k: f32,
    /// Reference speed for the ω-scaled measurement variance. See
    /// [`DEFAULT_OMEGA_NOISE_REF_RAD_S`].
    omega_noise_ref: f32,
    /// Initial / re-seed variance on ω. See [`DEFAULT_WIDE_OMEGA_VAR`].
    init_omega_var: f32,
    /// Consecutive NIS rejections before the escape re-seeds ω.
    escape_consecutive_rejects: u16,
    /// Hard-invalid measurement bound, as a multiple of `c_m`.
    plausible_omega_frac: f32,
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
    curve_k: Option<f32>,
    omega_noise_ref: Option<f32>,
    init_omega_var: Option<f32>,
    escape_consecutive_rejects: Option<u16>,
    plausible_omega_frac: Option<f32>,
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
            curve_k: None,
            omega_noise_ref: None,
            init_omega_var: None,
            escape_consecutive_rejects: None,
            plausible_omega_frac: None,
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

    /// Intensity of the throttle-error process driving ω, in seconds
    /// (a noise *spectral density*, `Var(u) × correlation time`, not a
    /// per-step variance — see the `qcov` construction in `predict`).
    ///
    /// It is multiplied by `dt·(c_m/τ)²`, so a value of `v` corresponds to
    /// a per-step throttle-error variance of `v/dt`. The historical
    /// per-step variance of 0.01 at an 8 kHz loop is `0.01 × 125 µs =
    /// 1.25e-6` in these units — which is the default.
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

    /// Throttle-curve nonlinearity `k` in `ω_ss = c_m·√(k·u² + (1−k)·u)`.
    /// Values outside `(0, 1]` (including the `m*_nonlin` zero sentinel)
    /// select the linear legacy model (`k = 1`).
    pub fn curve_k(mut self, value: f32) -> Self {
        self.curve_k = Some(value);
        self
    }

    /// Reference speed for the ω-scaled measurement variance
    /// (`rpm_est_omega_ref`). Change it together with the variance
    /// itself — it is what gives that number a meaning.
    pub fn omega_noise_ref(mut self, value: f32) -> Self {
        self.omega_noise_ref = Some(value);
        self
    }

    /// Initial and re-seed variance on ω (`rpm_est_init_omega_var`).
    pub fn init_omega_var(mut self, value: f32) -> Self {
        self.init_omega_var = Some(value);
        self
    }

    /// Consecutive NIS rejections before the escape re-seeds ω
    /// (`rpm_est_escape_rejects`).
    pub fn escape_consecutive_rejects(mut self, value: u16) -> Self {
        self.escape_consecutive_rejects = Some(value);
        self
    }

    /// Hard-invalid measurement bound as a multiple of `c_m`
    /// (`rpm_est_plausible_frac`).
    pub fn plausible_omega_frac(mut self, value: f32) -> Self {
        self.plausible_omega_frac = Some(value);
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
        // Noise/gate fallbacks come from the parameter schema's defaults,
        // not local literals: these used to be hardcoded copies and had
        // silently drifted 4× on omega_noise_cov (100 vs 400) and 1000×
        // on c_m_noise_cov (0.1 vs 100). The firmware path always sets
        // every field from params, so the fallbacks only fire for direct
        // builder use (tests, bring-up) — exactly where a stale copy is
        // hardest to notice.
        let schema = cybflight_core::params::RpmEstimatorParams::default();
        let throttle_noise_cov = self.throttle_noise_cov.unwrap_or(schema.throttle_noise_cov);
        if throttle_noise_cov <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNoiseCovariance);
        }
        let c_m_noise_cov = self.c_m_noise_cov.unwrap_or(schema.c_m_noise_cov);
        if c_m_noise_cov <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNoiseCovariance);
        }
        let omega_noise_cov = self.omega_noise_cov.unwrap_or(schema.omega_noise_cov);
        if omega_noise_cov <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNoiseCovariance);
        }
        let nis_gate = self.nis_gate.unwrap_or(schema.nis_gate);
        if nis_gate <= 0.0 {
            return Err(RpmEstimatorError::NonPositiveNISGate);
        }

        let curve_k = match self.curve_k {
            Some(k) if k > 0.0 && k <= 1.0 => k,
            _ => 1.0,
        };

        // Each of these degrades to its default rather than erroring: a
        // bad robustness knob must not stop the estimator from being
        // built, per the "degrade, never panic" rule.
        let usable = |v: Option<f32>, d: f32| match v {
            Some(x) if x.is_finite() && x > 0.0 => x,
            _ => d,
        };
        Ok(RpmEstimatorConfig {
            tau_m_up,
            tau_m_down,
            tau_d,
            throttle_noise_cov,
            c_m_noise_cov,
            omega_noise_cov,
            nis_gate,
            curve_k,
            omega_noise_ref: usable(self.omega_noise_ref, DEFAULT_OMEGA_NOISE_REF_RAD_S),
            init_omega_var: usable(self.init_omega_var, DEFAULT_WIDE_OMEGA_VAR),
            escape_consecutive_rejects: self
                .escape_consecutive_rejects
                .filter(|n| *n > 0)
                .unwrap_or(DEFAULT_ESCAPE_CONSECUTIVE_REJECTS),
            plausible_omega_frac: usable(
                self.plausible_omega_frac,
                DEFAULT_PLAUSIBLE_OMEGA_FRAC,
            ),
        })
    }
}

impl Default for RpmEstimatorConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Throttle-history depth, in control ticks.
///
/// The history exists to answer "what was commanded `tau_d` ago", so it
/// must span the largest `rpm_est_tau_d` the schema allows (20 ms) at
/// the fastest rate it is fed. `push_throttle` runs once per *control*
/// tick — IMU ODR / `indi_ctrl_div` — and the fastest shipped vehicle
/// runs 2 kHz (8 kHz / 4), so 40 ticks covers the ceiling with one spare
/// entry for the strict `<=` search.
///
/// At 10 it spanned only 5 ms at 2 kHz, so the top three quarters of the
/// parameter's own documented range silently fell back to the oldest
/// entry — the caller got a delay it never asked for and nothing said
/// so. Configurations faster than 2 kHz are still possible on paper
/// (`indi_ctrl_div: 1` on an 8 kHz board); those are caught at
/// construction by [`max_representable_tau_d_s`], which clamps and warns
/// rather than truncating quietly.
pub const MAX_HISTORY: usize = 41;

/// The largest transport delay the history can actually represent at a
/// given control rate.
///
/// `MAX_HISTORY - 1` because the lookup wants at least one entry older
/// than the target time to find; with exactly `MAX_HISTORY` ticks of
/// span the oldest entry sits on the boundary.
pub fn max_representable_tau_d_s(control_rate_hz: f32) -> f32 {
    if control_rate_hz.is_finite() && control_rate_hz > 0.0 {
        (MAX_HISTORY - 1) as f32 / control_rate_hz
    } else {
        0.0
    }
}

/// Reference speed for the ω-scaled measurement variance:
/// `R(y) = omega_noise_cov · max(1, (y/omega_noise_ref)²)`.
///
/// Bidirectional-DShot eRPM is decoded from single 60° commutation steps,
/// so its noise scales with speed. Measured from the SAKURAH743 flight
/// logs (residual of raw eRPM against its own 30 Hz trend):
/// σ ≈ 0.087·ω rad/s — σ ≈ 177 at ω = 2000, i.e. R ≈ 3.1e4, roughly 80×
/// the old constant-R default. `omega_noise_cov` keeps its meaning as the
/// variance at (and below) this reference speed.
pub const DEFAULT_OMEGA_NOISE_REF_RAD_S: f32 = 2000.0;

/// Consecutive NIS rejections before the escape re-seeds ω from the
/// measurement. At the 500 Hz–1 kHz estimator rates this is 25–50 ms of
/// solid disagreement — far longer than any decode glitch burst, far
/// shorter than the multi-second open-loop coasts seen in the logs.
pub const DEFAULT_ESCAPE_CONSECUTIVE_REJECTS: u16 = 25;

/// ω variance seeded by `reset_state`, `reconfigure` and the rejection
/// escape.
pub const DEFAULT_WIDE_OMEGA_VAR: f32 = 1000.0;

/// A measurement above this multiple of the current `c_m` (≈ full-throttle
/// speed) is a decode error, not a rotor state — hard-invalid.
pub const DEFAULT_PLAUSIBLE_OMEGA_FRAC: f32 = 1.5;

pub struct RpmEstimator {
    config: RpmEstimatorConfig,
    u_history: Deque<ThrottleStamped, MAX_HISTORY>,
    posterior: StateAndCov,
    /// NIS rejections since the last accepted measurement (or escape).
    consecutive_rejects: u16,
}

impl RpmEstimator {
    pub fn new(config: RpmEstimatorConfig, initial_state: StateAndCov) -> Self {
        Self {
            config,
            u_history: Deque::new(),
            posterior: initial_state,
            consecutive_rejects: 0,
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

    /// Reconfigure motor dynamics from the configured params
    /// (`indi_tau_m*` / `indi_omega_m*`, applied on the disarmed
    /// param hot-reload).
    ///
    /// Updates `tau_m` (motor time constant) and `c_m` (max omega at full
    /// throttle). Resets covariance wide so the filter re-converges quickly
    /// on the next arm cycle.
    pub fn reconfigure(&mut self, tau_m: f32, c_m: f32, curve_k: f32) {
        self.config.tau_m_up = tau_m;
        self.config.tau_m_down = tau_m;
        self.config.curve_k = if curve_k > 0.0 && curve_k <= 1.0 {
            curve_k
        } else {
            1.0
        };
        let omega = self.posterior.omega();
        self.posterior = StateAndCov::new(
            omega,
            c_m,
            Matrix2::new(self.config.init_omega_var, 0.0, 0.0, c_m * c_m),
        );
        self.consecutive_rejects = 0;
    }

    /// Replace the whole configuration, preserving the state estimate and
    /// covariance.
    ///
    /// This is the bench-tuning path for the filter's *own* tuning (noise
    /// intensities, NIS gate, transport delay) as opposed to
    /// [`Self::reconfigure`], which changes the motor dynamics and
    /// therefore has to re-seed the covariance. Sweeping a gate value
    /// should not throw away a converged `c_m`.
    pub fn set_config(&mut self, config: RpmEstimatorConfig) {
        self.config = config;
    }

    /// Reset state for a new flight.
    ///
    /// Zeros omega and sets wide covariance so the filter converges from
    /// whatever telemetry arrives after arming.
    pub fn reset_state(&mut self) {
        let c_m = self.posterior.c_m();
        self.posterior = StateAndCov::new(
            0.0,
            c_m,
            Matrix2::new(self.config.init_omega_var, 0.0, 0.0, c_m * c_m),
        );
        self.u_history.clear();
        self.consecutive_rejects = 0;
    }

    fn inv_tau_m(&self, is_accelerating: bool) -> f32 {
        // Branchless selection of the appropriate time constant based on acceleration direction
        [1.0 / self.config.tau_m_down, 1.0 / self.config.tau_m_up][is_accelerating as usize]
    }

    /// Effective throttle: `√(k·u² + (1−k)·u)`, so that `c_m·u_eff` is the
    /// identified steady-state speed curve. `k = 1` is the legacy linear
    /// model.
    fn effective_throttle(&self, u: f32) -> f32 {
        let k = self.config.curve_k;
        libm::sqrtf((k * u * u + (1.0 - k) * u).max(0.0))
    }

    fn predict(&self, post: StateAndCov, u: f32, dt: f32) -> StateAndCov {
        let StateAndCov { state, covariance } = post;
        let [omega, c_m] = state.into();
        let u = self.effective_throttle(u);
        let delta_omega = c_m * u - omega;
        let inv_tau_m = self.inv_tau_m(delta_omega > 0.0);
        // Discrete-time state transition Jacobian
        let f_mat = Matrix2::new(1.0 - dt * inv_tau_m, dt * u * inv_tau_m, 0.0, 1.0);
        // Process noise on ω is a Wiener increment: its variance grows
        // linearly with elapsed time, so exactly ONE factor of `dt`.
        //
        // The previous form squared the whole input term — `(dt·c_m/τ)²·q` —
        // which made the per-step variance ∝ dt² and therefore the variance
        // accumulated over a fixed wall-clock interval ∝ dt. The filter's
        // confidence then depended on the IMU ODR rather than on physics.
        // Simulated with telemetry at ~6 kHz followed by a 5 ms blackout,
        // P₀₀ after the blackout was 232 (σ ≈ 15 rad/s) at 8 kHz but 1859
        // (σ ≈ 43 rad/s) on `imu_1khz` builds — same airframe, same ESC —
        // and the NIS gate, 9·(P₀₀+R), inherited it: ±55 vs ±133 rad/s.
        // With one factor of dt the 1 kHz figure becomes 256 / ±57 rad/s,
        // i.e. the two builds agree. The `c_m` entry below always used the
        // correct ∝ dt form.
        let input_gain = c_m * inv_tau_m;
        let qcov = Matrix2::new(
            dt * input_gain * input_gain * self.config.throttle_noise_cov,
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
        // eRPM decode noise scales with speed (single 60° commutation
        // step); measured σ ≈ 8.7 % of ω on the SAKURAH743 logs. Treat
        // `omega_noise_cov` as the variance at OMEGA_NOISE_REF and scale
        // quadratically above it — a constant R sized at low speed makes
        // the NIS gate ~an order of magnitude too tight at speed, which
        // is how the filter ended up rejecting most of a fast mission.
        let ratio = y / self.config.omega_noise_ref;
        let r_eff = self.config.omega_noise_cov * (ratio * ratio).max(1.0);
        // Optimization: H = [1, 0].
        // 1. H * P * H.T == H[0, 0],
        // 2. H * P == P[0, :]
        // 3. K * H.T = [K, zeros((2, 1))], I - K = [[1-K[0], 0], [-K[1], 1]]
        let s_cov = covariance[(0, 0)] + r_eff;
        if innov * innov > self.config.nis_gate * s_cov {
            return Err(RpmEstimatorError::OutlierDetected(innov));
        }
        let k_gain = covariance.column(0) / s_cov;
        let mut state = state + k_gain * innov;
        state[1] = state[1].max(0.0); // Enforce non-negative c_m

        let i_m_kh = Matrix2::new(1.0 - k_gain[0], 0.0, -k_gain[1], 1.0);
        let covariance =
            i_m_kh * covariance * i_m_kh.transpose() + k_gain * r_eff * k_gain.transpose();
        Ok(StateAndCov { state, covariance })
    }

    /// Advance one tick. Returns what happened to `y_meas` so the caller can
    /// count NIS rejections — they used to be swallowed by `unwrap_or`, which
    /// made a filter coasting through every measurement indistinguishable
    /// from one tracking perfectly.
    pub fn step(
        &mut self,
        current_timestamp: f32,
        dt: f32,
        y_meas: Option<f32>, // Some(rpm) if a valid DShot packet arrived, None otherwise
    ) -> StepOutcome {
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

        // 3. Update if we received valid telemetry on this tick.
        //
        // Hard validity first, statistics second: a zero eRPM while the
        // motor is driven and anything above ~1.5× the full-throttle speed
        // are decode failures, not rotor states — they must not reach the
        // NIS gate (a zero would be a huge "legitimate" innovation) and
        // must not count toward the rejection escape.
        let (posterior, outcome) = match y_meas {
            Some(y) if y <= 0.0 || y >= self.config.plausible_omega_frac * prior.c_m() => {
                (prior, StepOutcome::Invalid)
            }
            Some(y) => match self.update(prior, y) {
                Ok(post) => {
                    self.consecutive_rejects = 0;
                    (post, StepOutcome::Updated)
                }
                // NIS gate tripped — keep the prediction, but say so; and
                // if the gate has tripped for `rpm_est_escape_rejects`
                // ticks straight, the model (not the telemetry) is wrong:
                // re-seed ω from the measurement with wide covariance so
                // the filter re-acquires instead of coasting open-loop.
                Err(_) => {
                    self.consecutive_rejects = self.consecutive_rejects.saturating_add(1);
                    if self.consecutive_rejects >= self.config.escape_consecutive_rejects {
                        self.consecutive_rejects = 0;
                        let mut cov = prior.covariance;
                        cov[(0, 0)] = self.config.init_omega_var;
                        cov[(0, 1)] = 0.0;
                        cov[(1, 0)] = 0.0;
                        (
                            StateAndCov {
                                state: Vector2::new(y, prior.c_m()),
                                covariance: cov,
                            },
                            StepOutcome::Reseeded,
                        )
                    } else {
                        (prior, StepOutcome::Rejected)
                    }
                }
            },
            // No telemetry? Trust the model.
            None => (prior, StepOutcome::Coasted),
        };
        self.posterior = posterior;
        outcome
    }
}
