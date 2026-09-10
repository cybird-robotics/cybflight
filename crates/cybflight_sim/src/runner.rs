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

use cybflight_core::eskf::{Eskf, EskfConfig, EskfGpsGuard, GpsFix, GpsGuardConfig, GuardSnapshot};
use cybflight_core::mpc::NX;

use crate::controller::Controller;
use crate::plant::QuadPlant;
use crate::scenario::{PassCriteria, Scenario, Verdict};
use crate::sensors::ImuMeasurement;
use crate::trajectory::Setpoint;

/// Sim-side `EskfConfig` baseline.
///
/// **Schema stability**: every field is set explicitly via
/// `EskfConfig::new` so future changes to `EskfConfig::default()` in
/// cybflight-core (noise density retunes, gate-sigma tweaks) do not
/// propagate into the sim snapshot. The values here are the sim's
/// frozen baseline; they match the firmware GPS path's tuning at the
/// time of writing (notably `max_pos_jump_m = 3.0`, which mirrors
/// the GPS vehicles' pinned `eskf_max_pos_jump_m`).
fn sim_eskf_config() -> EskfConfig {
    EskfConfig::new(
        0.01,    // accel_noise_density
        0.0001,  // gyro_noise_density
        0.001,   // accel_bias_random_walk
        0.00001, // gyro_bias_random_walk
        0.5,     // baro_noise_std
        0.05,    // mag_noise_std
        10.0,    // gate_sigma
        3.0,     // max_pos_jump_m (GPS-widened, vs mocap default of 1.0)
        0.7,     // max_att_jump_rad
        1.0,     // init_pos_var
        1.0,     // init_vel_var
        0.01,    // init_accel_bias_var
        0.01,    // init_gyro_bias_var
        0.1,     // init_att_var_rp
    )
}

/// Sim-side `GpsGuardConfig` baseline. Same schema-stability contract
/// as [`sim_eskf_config`] — all fields explicit via
/// `GpsGuardConfig::new` so the sim doesn't drift if firmware
/// engineers retune the failsafe defaults.
fn sim_gps_guard_config() -> GpsGuardConfig {
    GpsGuardConfig::new(
        2,      // max_consecutive_jumps
        5,      // max_consecutive_rejects
        2_000,  // gps_stale_ms
        2_000,  // rtk_fix_debounce_ms
        1_000,  // rtk_loss_debounce_ms
        0.002,  // gyro_bias_cov_trace_xy_thresh
        10.0,   // init_yaw_cov
        6,      // gps_min_sv
        50_000, // gps_h_acc_max_mm
        0.05,   // pos_sigma_floor_fix_m
        0.30,   // pos_sigma_floor_float_m
        2.0,    // pos_sigma_floor_none_m
        0.10,   // vel_sigma_floor_m_s
        true,   // fuse_velocity
        2,      // reinit_min_carr_soln
    )
}

/// Match firmware: 8 IMU samples per predict. At 8 kHz IMU this is 1 kHz;
/// at 100 Hz controller ticks it reduces to ~12.5 Hz — still above GPS
/// rate, and still representative of IMU-aided filtering.
const PREDICT_DECIMATION: u32 = 8;

/// Cap on the pre-mission wait for `EskfGpsGuard::is_ready()` [s].
///
/// The firmware gates flight on estimator readiness; the sim mirrors that
/// so a scenario is not scored on a transient the real vehicle would never
/// fly. The cap exists because some scenarios deliberately keep the guard
/// un-ready (GPS outage / fault injection) and still need to fly — there,
/// the mission starts anyway and the degraded estimate is the point.
///
/// 8 s clears the guard's own `rtk_fix_debounce_ms` (2 s) plus covariance
/// convergence with margin; the GPS baseline reports ready at ≈2.6 s.
const MAX_ESTIMATOR_WAIT_S: f32 = 8.0;

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
    /// Normalized ESC commands `d ∈ [0,1]` emitted by the controller.
    pub motor_commands: [f32; 4],
    /// Per-rotor thrust [N] the plant actually produced at these rotor
    /// speeds — lags `motor_commands` by the actuator time constant.
    pub motor_forces: [f32; 4],
    /// Rotor speeds [rad/s].
    pub rotor_omega: [f32; 4],
    /// False while the runner is holding station waiting for the
    /// estimator to converge. Scored metrics ignore these records — see
    /// [`summarize`] — but they stay in `history` so the hold is still
    /// visible to the reporter and the viz.
    pub in_mission: bool,
    /// In-sim ESKF guard state at this tick. `None` for non-GPS
    /// scenarios. Tests use this to assert on ready-flag, jump
    /// counters, RTK quality, etc. Skipped from `report.json` /
    /// regression snapshot — those serialize only `summary` fields.
    pub estimator: Option<GuardSnapshot>,
    /// In-sim ESKF position estimate (ENU). `None` for non-GPS
    /// scenarios. Tests assert on estimator-vs-truth divergence here.
    pub estimator_position: Option<Vector3<f32>>,
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
    /// Wall-clock time at which the in-sim ESKF guard first reported
    /// ready, and the mission clock therefore started. `None` when no
    /// estimator was in the loop (mission starts at t=0) or when the
    /// guard never converged and the wait cap released it instead.
    pub estimator_ready_s: Option<f32>,
    /// Peak tilt observed during the pre-mission estimator hold, if there
    /// was one. Reported rather than scored: the real vehicle is on the
    /// ground and disarmed until the guard reports ready, so this window
    /// is not a flight the pass criteria should judge — but it must stay
    /// visible, because a wild excursion here is still a signal.
    pub pre_mission_peak_tilt_rad: f32,
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
    /// The same failsafe state machine the firmware GPS task drives.
    /// Owns RTK debounce, jump/reject cascades, NaN re-init, staleness,
    /// convergence, ready flag.
    guard: EskfGpsGuard,
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
        // The mission clock does not start until the estimator is usable
        // (see `mission_start_s` below), so wall-clock has to allow for
        // that hold on top of the trajectory itself.
        let wall_deadline_s =
            (sim_deadline_s + MAX_ESTIMATOR_WAIT_S).min(self.cfg.max_sim_time_s.max(sim_deadline_s));
        let max_ticks = ((wall_deadline_s * controller.tick_rate_hz()).ceil() as usize).max(1);

        let expected_records = (max_ticks / history_stride as usize) + 2;
        let mut history = Vec::with_capacity(expected_records);
        let mut early_exit: Option<String> = None;
        let mut geofence_violation = false;
        // Sampled every tick, not only on history records — a brief clip
        // between two decimated samples is still a clip.
        let mut peak_saturation = 0.0f32;

        let horizon_len = controller.horizon_samples().max(1);
        let horizon_stride_s = controller.horizon_stride_s();
        let mut horizon: Vec<Setpoint> = Vec::with_capacity(horizon_len);

        // Pre-mission hold: station-keep at the start pose until the
        // estimator is usable. `Setpoint::hover` marks itself terminal;
        // clear that so the hold does not leak into terminal metrics.
        let mut hold_setpoint = Setpoint::hover(scenario.initial_position);
        hold_setpoint.terminal = false;

        // The plant resets with its rotors already at hover speed, so the
        // first IMU sample reads [0, 0, g] — what a real accelerometer
        // shows with the vehicle about to take off. Zero would look like
        // free fall to INDI's takeoff detector.

        // Spin up the in-sim ESKF if a GPS model is attached. The
        // `EskfGpsGuard` is the same one driving the firmware GPS task,
        // so failsafe behaviour (jump cascade, RTK debounce, staleness)
        // is exercised here too. Initialise the underlying filter from
        // plant truth: the vehicle really is there, and the firmware
        // likewise initialises from a fix taken while sitting still on
        // the ground. (Seeding from the first *noisy* fix instead was
        // tried and is strictly worse — it hands the controller that
        // fix's full error to chase from tick zero.)
        let mut eskf_state: Option<InSimEskf> = scenario.gps_model.as_ref().map(|gps| {
            let mut eskf = Eskf::new(sim_eskf_config());
            eskf.init(
                plant.position(),
                plant.attitude(),
                Vector3::zeros(),
                Vector3::zeros(),
            );
            let anchor_ms = (plant.time_s() * 1000.0) as u64;
            InSimEskf {
                eskf,
                guard: EskfGpsGuard::new(sim_gps_guard_config(), anchor_ms),
                gps_rate_hz: gps.rate_hz(),
                next_gps_t: plant.time_s(),
                last_predict_t: plant.time_s(),
                predict_counter: 0,
            }
        });

        // Mission clock. With no estimator in the loop the mission starts
        // immediately, which keeps every truth-state scenario (and its
        // snapshot rows) bit-identical. With an ESKF attached the mission
        // waits for `EskfGpsGuard::is_ready()`, mirroring the firmware,
        // which will not hand an unconverged estimate to the controller.
        let mut mission_start_s: Option<f32> = if eskf_state.is_some() {
            None
        } else {
            Some(0.0)
        };
        let mut estimator_ready_s: Option<f32> = None;

        for tick_idx in 0..max_ticks {
            let t = plant.time_s();

            let imu = scenario.imu_model.sample(plant);
            let rotor = scenario.rotor_model.sample(plant);

            // Decide the state vector the controller sees. Two branches:
            //   - GPS attached: run the ESKF in-loop, feed it IMU and GPS
            //     exactly as the firmware task does, hand the controller
            //     an estimator-derived state.
            //   - Otherwise: plant ground truth (unchanged code path, so
            //     existing snapshot rows remain bit-stable).
            let owned_state: SVector<f32, NX>;
            let truth_state: SVector<f32, NX>;
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
                // GPS update at its configured rate. Position AND
                // velocity updates flow through the guard — the
                // guard owns the full PVT pipeline (failsafe + both
                // measurement channels) so sim and firmware exercise
                // the same code path.
                if t >= es.next_gps_t {
                    if let Some(gps_model) = scenario.gps_model.as_mut() {
                        let m = gps_model.sample(plant);
                        let fix = gps_measurement_to_fix(&m, t);
                        let _ = es.guard.on_pvt(&mut es.eskf, &fix);
                    }
                    es.next_gps_t += 1.0 / es.gps_rate_hz;
                }
                let _ = es
                    .guard
                    .on_predict_tick(&es.eskf, (t * 1000.0) as u64);
                owned_state = build_state_from_eskf(&es.eskf, &imu);
                &owned_state
            } else {
                truth_state = plant.control_state();
                &truth_state
            };

            // Release the mission once the guard reports ready, or once the
            // wait cap expires — the cap keeps a scenario that deliberately
            // degrades GPS (outage / fault injection) from never flying.
            if mission_start_s.is_none() {
                let ready = eskf_state
                    .as_ref()
                    .map(|es| es.guard.is_ready())
                    .unwrap_or(true);
                if ready {
                    estimator_ready_s = Some(t);
                    mission_start_s = Some(t);
                } else if t >= MAX_ESTIMATOR_WAIT_S {
                    mission_start_s = Some(t);
                }
            }
            let mission_t = mission_start_s.map(|t0| t - t0);

            horizon.clear();
            match mission_t {
                Some(mt) => {
                    for k in 0..horizon_len {
                        horizon.push(scenario.setpoints.sample(mt + k as f32 * horizon_stride_s));
                    }
                }
                None => horizon.resize(horizon_len, hold_setpoint),
            }
            let sp0 = horizon[0];

            let u = controller.step(controller_state, &imu, &rotor, &horizon);
            let motor_commands = [u[0], u[1], u[2], u[3]];
            let thrusts = plant.motor_thrusts();
            let motor_forces = [thrusts[0], thrusts[1], thrusts[2], thrusts[3]];
            let omega = plant.rotor_omega();
            let rotor_omega = [omega[0], omega[1], omega[2], omega[3]];
            // Saturation is measured on the *command*: `d = 1` is where
            // the actuator clips, and it is the quantity the allocator
            // has to live within. Rotor-speed fraction lags it and never
            // reaches 1 during a transient, which would under-report a
            // real clip.
            let tick_saturation = u.iter().fold(0.0f32, |a, &b| a.max(b)).clamp(0.0, 1.0);

            if (tick_idx as u32) % history_stride == 0 {
                let (estimator, estimator_position) = match eskf_state.as_ref() {
                    Some(es) => (Some(es.guard.snapshot()), Some(es.eskf.position())),
                    None => (None, None),
                };
                history.push(StepRecord {
                    t,
                    position: plant.position(),
                    velocity: plant.velocity(),
                    attitude: plant.attitude(),
                    body_rate: plant.body_rate(),
                    tilt_rad: plant.tilt_rad(),
                    setpoint: sp0,
                    motor_commands,
                    motor_forces,
                    rotor_omega,
                    in_mission: mission_t.is_some(),
                    estimator,
                    estimator_position,
                });
            }

            for _ in 0..substeps {
                plant.step(&u);
            }
            if tick_saturation > peak_saturation {
                peak_saturation = tick_saturation;
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
                early_exit = Some(format!("non-finite state at t={:.2}s", plant.time_s()));
                break;
            }
            let elapsed_mission_s = mission_start_s.map(|t0| plant.time_s() - t0);
            if elapsed_mission_s.is_some_and(|m| m >= sim_deadline_s) {
                break;
            }
        }

        let summary = summarize(
            &history,
            trajectory_s,
            plant.time_s(),
            geofence_violation,
            early_exit.clone(),
            peak_saturation,
            estimator_ready_s,
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

/// Translate a sim `GpsMeasurement` (already in ENU, no carrier-solution
/// concept) into the guard's `GpsFix`. The synthetic fields
/// (`carr_soln=2`, `num_sv=12`, `fix_type=3`) say "this is RTK-fixed
/// open sky" — the sim isn't testing degraded-fix behaviour by default.
/// `h_acc_mm` mirrors the model's reported `sigma_pos`; the same value
/// is reused for `v_acc_mm` since `GpsMeasurement` doesn't track them
/// separately.
fn gps_measurement_to_fix(m: &crate::sensors::GpsMeasurement, t_s: f32) -> GpsFix {
    let acc_mm = ((m.sigma_pos.max(0.001)) * 1000.0).round().max(1.0) as u32;
    let s_acc_mm_s = ((m.sigma_vel.max(0.001)) * 1000.0).round().max(1.0) as u32;
    GpsFix {
        enu_pos: m.position,
        enu_vel: m.velocity,
        h_acc_mm: acc_mm,
        v_acc_mm: acc_mm,
        s_acc_mm_s,
        num_sv: 12,
        fix_type: 3,
        carr_soln: 2,
        timestamp_ms: (t_s * 1000.0) as u64,
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
    peak_saturation: f32,
    estimator_ready_s: Option<f32>,
) -> SummaryMetrics {
    let mut sum_sq = 0.0f32;
    let mut peak_err = 0.0f32;
    let mut peak_tilt = 0.0f32;
    let mut terminal_err = 0.0f32;
    let mut terminal_count: u32 = 0;
    let mut scored: u32 = 0;
    let mut pre_mission_peak_tilt = 0.0f32;

    for rec in history {
        // The pre-mission hold is not a flight: the firmware keeps the
        // vehicle disarmed on the ground until `EskfGpsGuard::is_ready()`,
        // so an excursion there is not something the controller would ever
        // be asked to survive. Record it, do not score it.
        if !rec.in_mission {
            if rec.tilt_rad > pre_mission_peak_tilt {
                pre_mission_peak_tilt = rec.tilt_rad;
            }
            continue;
        }
        scored += 1;
        let err = (rec.position - rec.setpoint.position).norm();
        sum_sq += err * err;
        if err > peak_err {
            peak_err = err;
        }
        if rec.tilt_rad > peak_tilt {
            peak_tilt = rec.tilt_rad;
        }
        if rec.setpoint.terminal {
            terminal_err += err;
            terminal_count += 1;
        }
    }
    let rms_pos_err_m = if scored == 0 {
        0.0
    } else {
        (sum_sq / scored as f32).sqrt()
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
        peak_motor_saturation_pct: peak_saturation * 100.0,
        geofence_violation,
        total_sim_time_s: total_s,
        trajectory_duration_s: trajectory_s,
        early_exit_reason,
        estimator_ready_s,
        pre_mission_peak_tilt_rad: pre_mission_peak_tilt,
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
