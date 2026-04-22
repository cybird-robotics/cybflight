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
