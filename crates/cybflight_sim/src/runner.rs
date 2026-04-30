//! MissionRunner: drives plant/controller/setpoint loop and produces a
//! structured history + verdict.
//!
//! The controller declares its own `tick_rate_hz()`. The runner ticks the
//! controller at that rate and integrates the plant in substeps of
//! `cfg.dt_sim` per controller tick. This matches the firmware model where
//! the outer + inner loops run at different rates.
//!
//! History is decimated to ~100 Hz effective regardless of tick rate, so
//! high-rate stacks (INDI at 8 kHz) don't bloat reports.
//!
//! When the scenario attaches a [`GpsModel`](crate::sensors::GpsModel),
//! the runner spins up an in-loop ESKF (`cybflight_core::eskf::Eskf`) and
//! feeds the controller **estimator-derived state** instead of plant
//! ground truth — mirroring the firmware's `eskf_imu_gps` task. Clean
//! scenarios (no GPS) keep the truth-state code path bit-for-bit so
//! snapshot numbers don't drift.

use nalgebra::{SVector, UnitQuaternion, Vector3};

use cybflight_core::eskf::{Eskf, EskfConfig};
use cybflight_core::mpc::{NU, NX};

use crate::controller::Controller;
use crate::plant::QuadPlant;
use crate::scenario::{PassCriteria, Scenario, Verdict};
use crate::sensors::ImuMeasurement;
use crate::trajectory::Setpoint;

/// Match firmware: 8 IMU samples per predict. At 8 kHz IMU this is 1 kHz;
/// at 100 Hz controller ticks it reduces to ~12.5 Hz — still above GPS
/// rate, and still representative of IMU-aided filtering.
const PREDICT_DECIMATION: u32 = 8;

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    /// Plant integration timestep [s].
    pub dt_sim: f32,
    /// Hard cap on total simulation time [s].
    pub max_sim_time_s: f32,
    /// Target effective history sample rate [Hz]. Runner decimates tick-by-
    /// tick records to ≈ this rate.
    pub history_rate_hz: f32,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        // 8 kHz matches the INDI stack; higher-level stacks use fewer substeps.
        Self {
            dt_sim: 1.0 / 8000.0,
            max_sim_time_s: 60.0,
            history_rate_hz: 100.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepRecord {
    pub t: f32,
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub attitude: UnitQuaternion<f32>,
    pub body_rate: Vector3<f32>,
    pub tilt_rad: f32,
    pub setpoint: Setpoint,
    pub motor_forces: [f32; 4],
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SummaryMetrics {
    pub rms_pos_err_m: f32,
    pub terminal_pos_err_m: f32,
    pub peak_tilt_rad: f32,
    pub peak_pos_err_m: f32,
    pub peak_motor_saturation_pct: f32,
    pub geofence_violation: bool,
    pub total_sim_time_s: f32,
    pub trajectory_duration_s: f32,
    pub early_exit_reason: Option<String>,
}

pub struct RunOutput {
    pub history: Vec<StepRecord>,
    pub summary: SummaryMetrics,
    pub verdict: Verdict,
    pub failure_reasons: Vec<String>,
}

pub struct MissionRunner {
    pub cfg: RunnerConfig,
}

struct InSimEskf {
    eskf: Eskf,
    gps_rate_hz: f32,
    next_gps_t: f32,
    last_predict_t: f32,
    predict_counter: u32,
}

impl MissionRunner {
    pub fn new(cfg: RunnerConfig) -> Self {
        Self { cfg }
    }

    pub fn run<C: Controller + ?Sized>(
        &self,
        scenario: &mut Scenario,
        plant: &mut QuadPlant,
        controller: &mut C,
    ) -> RunOutput {
        let tick_dt = 1.0 / controller.tick_rate_hz();
        assert!(
            tick_dt >= self.cfg.dt_sim - 1e-9,
            "tick_dt ({}) must be >= dt_sim ({})",
            tick_dt,
            self.cfg.dt_sim
        );

        plant.reset(
            scenario.initial_position,
            scenario.initial_velocity,
            scenario.initial_attitude,
        );

        let substeps = (tick_dt / self.cfg.dt_sim).round().max(1.0) as usize;
        let history_stride = ((controller.tick_rate_hz() / self.cfg.history_rate_hz)
            .round()
            .max(1.0)) as u32;

        let trajectory_s = scenario.setpoints.duration_s();
        let terminal_time_s = if trajectory_s.is_finite() {
            trajectory_s + scenario.terminal_hold_s
        } else {
            scenario.terminal_hold_s.max(5.0)
        };
        let sim_deadline_s = terminal_time_s.min(self.cfg.max_sim_time_s);
        let max_ticks = ((sim_deadline_s * controller.tick_rate_hz()).ceil() as usize).max(1);

        let expected_records = (max_ticks / history_stride as usize) + 2;
        let mut history = Vec::with_capacity(expected_records);
        let mut early_exit: Option<String> = None;
        let mut geofence_violation = false;

        let per_motor_max = plant
            .params
            .motors
            .iter()
            .map(|m| m.max_thrust_n)
            .fold(0.0f32, f32::max)
            .max(1e-6);

        let horizon_len = controller.horizon_samples().max(1);
        let horizon_stride_s = controller.horizon_stride_s();
        let mut horizon: Vec<Setpoint> = Vec::with_capacity(horizon_len);

        // Seed `u_last` at per-motor hover so the IMU model's first sample
        // reports [0, 0, g] (matching a real accelerometer with the vehicle
        // about to take off) rather than zero, which would look like free
        // fall to INDI's takeoff detector.
        let hover_per_motor = plant.params.body.mass_kg * 9.81 / NU as f32;
        let mut u_last = SVector::<f32, NU>::from_element(hover_per_motor);

        // Spin up the in-sim ESKF if a GPS model is attached. Initialise
        // from plant truth — mirrors the firmware `init(pos, orient, zero
        // biases)` that `eskf_imu_mocap` does on the first mocap frame.
        let mut eskf_state: Option<InSimEskf> = scenario.gps_model.as_ref().map(|gps| {
            let mut eskf = Eskf::new(EskfConfig::default());
            eskf.init(
                plant.position(),
                plant.attitude(),
                Vector3::zeros(),
                Vector3::zeros(),
            );
            InSimEskf {
                eskf,
                gps_rate_hz: gps.rate_hz(),
                next_gps_t: plant.time_s(),
                last_predict_t: plant.time_s(),
                predict_counter: 0,
            }
        });

        for tick_idx in 0..max_ticks {
            let t = plant.time_s();
            horizon.clear();
            for k in 0..horizon_len {
                let tk = t + k as f32 * horizon_stride_s;
                horizon.push(scenario.setpoints.sample(tk));
            }
            let sp0 = horizon[0];

            let imu = scenario.imu_model.sample(plant, &u_last);

            // Decide the state vector the controller sees. Two branches:
            //   - GPS attached: run the ESKF in-loop, feed it IMU and GPS
            //     exactly as the firmware task does, hand the controller
            //     an estimator-derived state.
            //   - Otherwise: plant ground truth (unchanged code path, so
            //     existing snapshot rows remain bit-stable).
            let owned_state: SVector<f32, NX>;
            let controller_state: &SVector<f32, NX> = if let Some(es) = eskf_state.as_mut() {
                // ESKF predict — decimated to ~1 kHz (PREDICT_DECIMATION=8
                // matches the firmware decimation from 8 kHz IMU).
                es.predict_counter += 1;
                if es.predict_counter >= PREDICT_DECIMATION {
                    let dt = t - es.last_predict_t;
                    if dt > 0.0 && dt < 0.05 {
                        es.eskf.predict(imu.accel, imu.gyro, dt);
                    }
                    es.last_predict_t = t;
                    es.predict_counter = 0;
                }
                // GPS update at its configured rate.
                if t >= es.next_gps_t {
                    if let Some(gps_model) = scenario.gps_model.as_mut() {
                        let fix = gps_model.sample(plant);
                        let _ = es.eskf.update_pos(fix.position, fix.sigma_pos);
                        let _ = es.eskf.update_vel(fix.velocity, fix.sigma_vel);
                    }
                    es.next_gps_t += 1.0 / es.gps_rate_hz;
                }
                owned_state = build_state_from_eskf(&es.eskf, &imu);
                &owned_state
            } else {
                plant.raw_state()
            };

            let u = controller.step(controller_state, &imu, &horizon);
            let motor_forces = [u[0], u[1], u[2], u[3]];

            if (tick_idx as u32) % history_stride == 0 {
                history.push(StepRecord {
                    t,
                    position: plant.position(),
                    velocity: plant.velocity(),
                    attitude: plant.attitude(),
                    body_rate: plant.body_rate(),
                    tilt_rad: plant.tilt_rad(),
                    setpoint: sp0,
                    motor_forces,
                });
            }

            for _ in 0..substeps {
                plant.step(&u);
            }
            u_last = u;

            let pos = plant.position();
            if violates_geofence(
                pos,
                scenario.pass_criteria.geofence_min,
                scenario.pass_criteria.geofence_max,
            ) {
                geofence_violation = true;
                early_exit = Some(format!(
                    "geofence violation at t={:.2}s pos=({:.2},{:.2},{:.2})",
                    plant.time_s(),
                    pos.x,
                    pos.y,
                    pos.z
                ));
                break;
            }
            if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
                early_exit = Some(format!("non-finite state at t={:.2}s", plant.time_s()));
                break;
            }
            if plant.time_s() >= sim_deadline_s {
                break;
            }
        }

        let summary = summarize(
            &history,
            trajectory_s,
            plant.time_s(),
            geofence_violation,
            early_exit.clone(),
            per_motor_max,
        );
        let (verdict, failure_reasons) = evaluate(&summary, &scenario.pass_criteria);
        RunOutput {
            history,
            summary,
            verdict,
            failure_reasons,
        }
    }
}

/// Build the FullQuadModel state vector the controllers expect (layout
/// `[px py pz | qx qy qz qw | vx vy vz | wx wy wz]`) from ESKF outputs.
/// Body rate = raw gyro minus the ESKF's gyro-bias estimate — the same
/// bias-corrected signal the firmware's INDI loop sees.
fn build_state_from_eskf(eskf: &Eskf, imu: &ImuMeasurement) -> SVector<f32, NX> {
    let pos = eskf.position();
    let vel = eskf.velocity();
    let q = eskf.orientation();
    let w = imu.gyro - eskf.gyro_bias();
    SVector::<f32, NX>::from_column_slice(&[
        pos.x, pos.y, pos.z, q.i, q.j, q.k, q.w, vel.x, vel.y, vel.z, w.x, w.y, w.z,
    ])
}

fn violates_geofence(p: Vector3<f32>, lo: Vector3<f32>, hi: Vector3<f32>) -> bool {
    p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z
}

fn summarize(
    history: &[StepRecord],
    trajectory_s: f32,
    total_s: f32,
    geofence_violation: bool,
    early_exit_reason: Option<String>,
    per_motor_max_n: f32,
) -> SummaryMetrics {
    let mut sum_sq = 0.0f32;
    let mut peak_err = 0.0f32;
    let mut peak_tilt = 0.0f32;
    let mut peak_sat = 0.0f32;
    let mut terminal_err = 0.0f32;
    let mut terminal_count: u32 = 0;

    for rec in history {
        let err = (rec.position - rec.setpoint.position).norm();
        sum_sq += err * err;
        if err > peak_err {
            peak_err = err;
        }
        if rec.tilt_rad > peak_tilt {
            peak_tilt = rec.tilt_rad;
        }
        for f in rec.motor_forces {
            let sat = (f / per_motor_max_n).clamp(0.0, 1.0);
            if sat > peak_sat {
                peak_sat = sat;
            }
        }
        if rec.setpoint.terminal {
            terminal_err += err;
            terminal_count += 1;
        }
    }
    let rms_pos_err_m = if history.is_empty() {
        0.0
    } else {
        (sum_sq / history.len() as f32).sqrt()
    };
    let terminal_pos_err_m = if terminal_count > 0 {
        terminal_err / terminal_count as f32
    } else {
        peak_err
    };

    SummaryMetrics {
        rms_pos_err_m,
        terminal_pos_err_m,
        peak_tilt_rad: peak_tilt,
        peak_pos_err_m: peak_err,
        peak_motor_saturation_pct: peak_sat * 100.0,
        geofence_violation,
        total_sim_time_s: total_s,
        trajectory_duration_s: trajectory_s,
        early_exit_reason,
    }
}

fn evaluate(summary: &SummaryMetrics, criteria: &PassCriteria) -> (Verdict, Vec<String>) {
    let mut reasons = Vec::new();
    if summary.geofence_violation {
        reasons.push("geofence_violation".into());
    }
    if summary.terminal_pos_err_m > criteria.terminal_pos_err_m {
        reasons.push(format!(
            "terminal_pos_err_m {:.3} > {:.3}",
            summary.terminal_pos_err_m, criteria.terminal_pos_err_m
        ));
    }
    if summary.rms_pos_err_m > criteria.rms_pos_err_m {
        reasons.push(format!(
            "rms_pos_err_m {:.3} > {:.3}",
            summary.rms_pos_err_m, criteria.rms_pos_err_m
        ));
    }
    if summary.peak_tilt_rad > criteria.peak_tilt_rad {
        reasons.push(format!(
            "peak_tilt_deg {:.1} > {:.1}",
            summary.peak_tilt_rad.to_degrees(),
            criteria.peak_tilt_rad.to_degrees()
        ));
    }
    if summary.early_exit_reason.is_some() && !summary.geofence_violation {
        reasons.push(
            summary
                .early_exit_reason
                .clone()
                .unwrap_or_else(|| "early_exit".into()),
        );
    }
    let verdict = if reasons.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    (verdict, reasons)
}
