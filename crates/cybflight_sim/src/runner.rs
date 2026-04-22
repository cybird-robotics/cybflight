//! MissionRunner equivalent: drives the plant/controller/setpoint loop and
//! produces a structured history + verdict. Mirrors the C++
//! `autopilot::MissionRunner` in spirit (see
//! `/home/hs293go/src/autopilot/validation/src/mission_runner.cpp`) but
//! without the estimator stage — state is taken as perfect truth from the
//! plant for the first (mission-level) milestone.

use nalgebra::{UnitQuaternion, Vector3};

use crate::controller::Controller;
use crate::plant::QuadPlant;
use crate::scenario::{PassCriteria, Scenario, Verdict};
use crate::trajectory::Setpoint;

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    /// Simulation integration timestep [s].
    pub dt_sim: f32,
    /// Controller update timestep [s]. Must satisfy `dt_ctrl >= dt_sim`.
    pub dt_ctrl: f32,
    /// Hard cap on total simulation time [s] — prevents runaway tests.
    pub max_sim_time_s: f32,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            dt_sim: 0.002,
            dt_ctrl: 0.01,
            max_sim_time_s: 60.0,
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
        assert!(
            self.cfg.dt_ctrl >= self.cfg.dt_sim,
            "dt_ctrl ({}) must be >= dt_sim ({})",
            self.cfg.dt_ctrl,
            self.cfg.dt_sim
        );

        plant.reset(
            scenario.initial_position,
            scenario.initial_velocity,
            scenario.initial_attitude,
        );

        let substeps = (self.cfg.dt_ctrl / self.cfg.dt_sim).round().max(1.0) as usize;
        let trajectory_s = scenario.setpoints.duration_s();
        let terminal_time_s = if trajectory_s.is_finite() {
            trajectory_s + scenario.terminal_hold_s
        } else {
            scenario.terminal_hold_s.max(5.0)
        };
        let sim_deadline_s = terminal_time_s.min(self.cfg.max_sim_time_s);
        let max_ctrl_steps = ((sim_deadline_s / self.cfg.dt_ctrl).ceil() as usize).max(1);

        let mut history = Vec::with_capacity(max_ctrl_steps);
        let mut early_exit: Option<String> = None;
        let mut geofence_violation = false;

        let per_motor_max = plant
            .params
            .motors
            .iter()
            .map(|m| m.max_thrust_n)
            .fold(0.0f32, f32::max)
            .max(1e-6);

        // Pre-allocate a horizon buffer sized to the controller's request.
        let horizon_len = controller.horizon_samples().max(1);
        let horizon_stride = controller.horizon_stride_s();
        let mut horizon: Vec<Setpoint> = Vec::with_capacity(horizon_len);

        for _ in 0..max_ctrl_steps {
            let t = plant.time_s();
            horizon.clear();
            for k in 0..horizon_len {
                let tk = t + k as f32 * horizon_stride;
                horizon.push(scenario.setpoints.sample(tk));
            }
            let sp = horizon[0];

            let u = controller.compute(plant.raw_state(), &horizon, self.cfg.dt_ctrl);
            let motor_forces = [u[0], u[1], u[2], u[3]];

            history.push(StepRecord {
                t,
                position: plant.position(),
                velocity: plant.velocity(),
                attitude: plant.attitude(),
                body_rate: plant.body_rate(),
                tilt_rad: plant.tilt_rad(),
                setpoint: sp,
                motor_forces,
            });

            for _ in 0..substeps {
                plant.step(&u);
            }

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
                early_exit = Some(format!(
                    "non-finite state at t={:.2}s",
                    plant.time_s()
                ));
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
