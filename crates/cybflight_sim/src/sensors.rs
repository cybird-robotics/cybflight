//! Sensor simulation for the host-side sim.
//!
//! An `ImuModel` consumes ground truth (body-frame specific force + body
//! rate) and produces an `ImuMeasurement`. The runner samples the
//! scenario's `ImuModel` every controller tick; controllers that care
//! (currently `MpcIndiController`, because INDI is the only consumer) read
//! the measurement from their `step()` call. Ground-truth controllers
//! (`CascadeController`, `MpcDirectController`) ignore it.
//!
//! Splitting sensor synthesis out of the plant + controllers means
//! scenarios can swap noise/bias characteristics without touching the
//! control stack — and sets up a clean seam for ESKF-in-the-loop in a
//! follow-up (it will also consume `ImuMeasurement`).

use core::f32::consts::TAU;

use nalgebra::{SVector, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use cybflight_core::mpc::NU;

use crate::plant::QuadPlant;

/// One IMU sample. Units: gyro [rad/s], accel [m/s²] (body-frame
/// specific force — what a strapdown accelerometer measures, i.e. proper
/// acceleration = (a_world − g_world) rotated into body).
#[derive(Clone, Copy, Debug)]
pub struct ImuMeasurement {
    pub gyro: Vector3<f32>,
    pub accel: Vector3<f32>,
}

/// Produce an IMU measurement from the current plant state. The runner
/// provides the most recently applied motor command vector so the sensor
/// model can derive specific force — for a drag-free quadrotor plant,
/// body-frame specific force is always `[0, 0, Σu / m]`.
pub trait ImuModel: Send {
    fn sample(&mut self, plant: &QuadPlant, u_last: &SVector<f32, NU>) -> ImuMeasurement;
}

/// Ground-truth IMU (no noise, no bias). Sets the numerical baseline for
/// `regression_snapshot.json`.
#[derive(Default)]
pub struct PerfectImu;

impl ImuModel for PerfectImu {
    fn sample(&mut self, plant: &QuadPlant, u_last: &SVector<f32, NU>) -> ImuMeasurement {
        let mass = plant.params.body.mass_kg;
        let total_thrust_n: f32 = u_last.iter().sum();
        ImuMeasurement {
            gyro: plant.body_rate(),
            accel: Vector3::new(0.0, 0.0, total_thrust_n / mass),
        }
    }
}

/// IMU with white Gaussian noise and constant bias, wrapping a
/// `PerfectImu` as the truth source. Deterministic across runs for a
/// given `seed` so snapshot regression stays stable.
pub struct NoisyImu {
    rng: ChaCha8Rng,
    truth: PerfectImu,
    /// Per-axis gyro noise stddev [rad/s].
    pub gyro_sigma_rad_s: Vector3<f32>,
    /// Per-axis accel noise stddev [m/s²].
    pub accel_sigma_m_s2: Vector3<f32>,
    /// Per-axis constant gyro bias [rad/s].
    pub gyro_bias_rad_s: Vector3<f32>,
    /// Per-axis constant accel bias [m/s²].
    pub accel_bias_m_s2: Vector3<f32>,
}

impl NoisyImu {
    /// Convenience: scalar (isotropic) sigmas and zero bias. Good enough
    /// for a first-pass "can the stack take noise" check.
    pub fn isotropic(seed: u64, gyro_sigma: f32, accel_sigma: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            truth: PerfectImu,
            gyro_sigma_rad_s: Vector3::repeat(gyro_sigma),
            accel_sigma_m_s2: Vector3::repeat(accel_sigma),
            gyro_bias_rad_s: Vector3::zeros(),
            accel_bias_m_s2: Vector3::zeros(),
        }
    }

    pub fn with_bias(
        mut self,
        gyro_bias_rad_s: Vector3<f32>,
        accel_bias_m_s2: Vector3<f32>,
    ) -> Self {
        self.gyro_bias_rad_s = gyro_bias_rad_s;
        self.accel_bias_m_s2 = accel_bias_m_s2;
        self
    }

    // Box–Muller: one call returns one standard-normal sample. Good enough
    // for sensor noise; we aren't running Monte Carlo inference off it.
    fn normal(&mut self) -> f32 {
        let u1 = self.rng.random::<f32>().max(1e-10);
        let u2 = self.rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    fn noise(&mut self, sigma: &Vector3<f32>) -> Vector3<f32> {
        Vector3::new(
            sigma.x * self.normal(),
            sigma.y * self.normal(),
            sigma.z * self.normal(),
        )
    }
}

impl ImuModel for NoisyImu {
    fn sample(&mut self, plant: &QuadPlant, u_last: &SVector<f32, NU>) -> ImuMeasurement {
        let truth = self.truth.sample(plant, u_last);
        let gyro_noise = self.noise(&self.gyro_sigma_rad_s.clone());
        let accel_noise = self.noise(&self.accel_sigma_m_s2.clone());
        ImuMeasurement {
            gyro: truth.gyro + self.gyro_bias_rad_s + gyro_noise,
            accel: truth.accel + self.accel_bias_m_s2 + accel_noise,
        }
    }
}

// ── GPS ─────────────────────────────────────────────────────────────────────

/// One GPS fix in the sim's ENU frame (the sim bypasses LLH — the
/// geodetic conversion is unit-tested separately in `cybflight_core`).
/// `sigma_*` values are the σ the firmware-side ESKF would derive from
/// u-blox accuracy estimates, plumbed through so the in-sim ESKF uses the
/// same calibration story as the real thing.
#[derive(Clone, Copy, Debug)]
pub struct GpsMeasurement {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub sigma_pos: f32,
    pub sigma_vel: f32,
}

/// Publish GPS measurements at a controller-independent rate (the runner
/// resamples based on `rate_hz()`; at 5 Hz, an 8 kHz substep loop fires
/// an update every 1600 substeps).
pub trait GpsModel: Send {
    fn rate_hz(&self) -> f32;
    fn sample(&mut self, plant: &QuadPlant) -> GpsMeasurement;
}

/// Truth GPS for pre-snapshot baselines — σ=0, 5 Hz.
pub struct PerfectGps {
    pub rate_hz: f32,
    pub sigma_pos: f32,
    pub sigma_vel: f32,
}

impl Default for PerfectGps {
    fn default() -> Self {
        Self {
            rate_hz: 5.0,
            sigma_pos: 0.0,
            sigma_vel: 0.0,
        }
    }
}

impl GpsModel for PerfectGps {
    fn rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn sample(&mut self, plant: &QuadPlant) -> GpsMeasurement {
        GpsMeasurement {
            position: plant.position(),
            velocity: plant.velocity(),
            // Report a small σ so the ESKF doesn't lock at near-zero
            // variance in perfect-truth mode (which would produce a
            // numerically brittle information matrix).
            sigma_pos: self.sigma_pos.max(0.05),
            sigma_vel: self.sigma_vel.max(0.10),
        }
    }
}

/// Noisy GPS: per-axis Gaussian noise + constant bias, deterministic for
/// a given seed. Matches the `NoisyImu` pattern so snapshot rows stay
/// reproducible.
pub struct NoisyGps {
    rng: ChaCha8Rng,
    rate_hz: f32,
    pub sigma_pos_m: Vector3<f32>,
    pub sigma_vel_m_s: Vector3<f32>,
    pub bias_pos_m: Vector3<f32>,
    /// σ reported to the ESKF (may differ from the actual noise σ to
    /// exercise mis-tuned filters).
    pub reported_sigma_pos: f32,
    pub reported_sigma_vel: f32,
}

impl NoisyGps {
    /// Isotropic horizontal+vertical σ, zero bias, configurable rate.
    /// `sigma_pos_m` ≈ 0.5 m and `sigma_vel_m_s` ≈ 0.2 m/s roughly mirror
    /// u-blox M10 open-sky SBAS-aided performance.
    pub fn isotropic(seed: u64, rate_hz: f32, sigma_pos_m: f32, sigma_vel_m_s: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            rate_hz,
            sigma_pos_m: Vector3::repeat(sigma_pos_m),
            sigma_vel_m_s: Vector3::repeat(sigma_vel_m_s),
            bias_pos_m: Vector3::zeros(),
            reported_sigma_pos: sigma_pos_m,
            reported_sigma_vel: sigma_vel_m_s,
        }
    }

    pub fn with_bias(mut self, bias_pos_m: Vector3<f32>) -> Self {
        self.bias_pos_m = bias_pos_m;
        self
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.rng.random::<f32>().max(1e-10);
        let u2 = self.rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    fn noise(&mut self, sigma: &Vector3<f32>) -> Vector3<f32> {
        Vector3::new(
            sigma.x * self.normal(),
            sigma.y * self.normal(),
            sigma.z * self.normal(),
        )
    }
}

impl GpsModel for NoisyGps {
    fn rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn sample(&mut self, plant: &QuadPlant) -> GpsMeasurement {
        let pos_noise = self.noise(&self.sigma_pos_m.clone());
        let vel_noise = self.noise(&self.sigma_vel_m_s.clone());
        GpsMeasurement {
            position: plant.position() + self.bias_pos_m + pos_noise,
            velocity: plant.velocity() + vel_noise,
            sigma_pos: self.reported_sigma_pos.max(0.05),
            sigma_vel: self.reported_sigma_vel.max(0.10),
        }
    }
}

/// Time-windowed fault decorator. Wraps any `GpsModel` and adds a
/// constant `bias_during_fault` to the reported position when sim time
/// is within `[t_start_s, t_end_s)`. Outside the window, samples pass
/// through unchanged.
///
/// Used by `tests/autotest_gps_failsafe.rs` to inject the position
/// excursion that triggers `EskfGpsGuard`'s consecutive-jump cascade.
pub struct FaultedGps<G: GpsModel> {
    inner: G,
    t_start_s: f32,
    t_end_s: f32,
    bias_during_fault: Vector3<f32>,
}

impl<G: GpsModel> FaultedGps<G> {
    pub fn new(inner: G, t_start_s: f32, t_end_s: f32, bias_during_fault: Vector3<f32>) -> Self {
        Self {
            inner,
            t_start_s,
            t_end_s,
            bias_during_fault,
        }
    }
}

impl<G: GpsModel> GpsModel for FaultedGps<G> {
    fn rate_hz(&self) -> f32 {
        self.inner.rate_hz()
    }
    fn sample(&mut self, plant: &QuadPlant) -> GpsMeasurement {
        let mut m = self.inner.sample(plant);
        let t = plant.time_s();
        if t >= self.t_start_s && t < self.t_end_s {
            m.position += self.bias_during_fault;
        }
        m
    }
}

/// Time-windowed *outage* decorator: during `[t_start_s, t_end_s)`,
/// reports a non-finite position. The guard's usability gate
/// (`UsabilityReason::NonFinitePosition`) drops the frame outright,
/// modelling a real GPS dropout where no NAV-PVT arrives at the
/// firmware's `Signal`.
///
/// Used by the GPS-outage regression test in
/// `tests/autotest_gps_failsafe.rs` to verify that a long outage
/// followed by a fresh fix doesn't permanently wedge the estimator —
/// the IMU drifts during the outage, post-outage frames look like
/// jumps relative to the drifted estimate, the cascade fires, and
/// the Phase-5 re-init heals it.
pub struct OutageGps<G: GpsModel> {
    inner: G,
    t_start_s: f32,
    t_end_s: f32,
}

impl<G: GpsModel> OutageGps<G> {
    pub fn new(inner: G, t_start_s: f32, t_end_s: f32) -> Self {
        Self {
            inner,
            t_start_s,
            t_end_s,
        }
    }
}

impl<G: GpsModel> GpsModel for OutageGps<G> {
    fn rate_hz(&self) -> f32 {
        self.inner.rate_hz()
    }
    fn sample(&mut self, plant: &QuadPlant) -> GpsMeasurement {
        let m = self.inner.sample(plant);
        let t = plant.time_s();
        if t >= self.t_start_s && t < self.t_end_s {
            // Non-finite position is what the guard's usability
            // predicate treats as "GPS not really delivering data" —
            // same effect as no Signal::set call on the firmware
            // side. The runner still ticks `next_gps_t`, so when
            // the window ends, real fixes resume immediately.
            GpsMeasurement {
                position: Vector3::new(f32::NAN, f32::NAN, f32::NAN),
                velocity: Vector3::zeros(),
                sigma_pos: 0.5,
                sigma_vel: 0.2,
            }
        } else {
            m
        }
    }
}
