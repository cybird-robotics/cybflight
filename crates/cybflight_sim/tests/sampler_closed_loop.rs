//! Closed-loop sampler verification.
//!
//! Drives a `QuadPlant` with `MpcIndiController` while *we* — the test —
//! play the role the firmware's `outer_loop.rs` plays in production:
//! mission state machine (Idle ↔ Executing), active-setpoint cell that
//! RC writes to, and the per-tick `Sampler::sample(...)` call that fills
//! the controller's horizon during Executing.
//!
//! The test answers two specific questions the user wanted nailed down
//! before either sampler flies:
//!
//! 1. **Does the position sampler honour active-setpoint changes from
//!    "RC"?** During Idle, the controller must track whatever the active
//!    setpoint cell holds — even after a position-sampler mission has
//!    converged at some other point. The sampler is *not* allowed to
//!    pin the controller to its last τ when the mission is over.
//!
//! 2. **Does mission status correctly transition Executing → Idle once
//!    the destination is reached?** The sampler reports `mission_done`
//!    when its node 0 hits the trajectory's end (or the state lands
//!    inside `radius_of_acceptance`); the outer loop must flip the
//!    mission state, refresh the active-setpoint cell to the trajectory
//!    endpoint, and the very next tick must hold position there.
//!
//! Both invariants are tested for `Sampler::Time` and `Sampler::Position`
//! so a regression in either path is loud.

use cybflight_core::mpc::NU as MPC_NU;
use cybflight_core::params::VehicleParams;
use cybflight_core::trajectory_planning::bfgs_trust::BfgsWorkspace;
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::planner::{plan_with_workspace, PlannerInput};
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_core::trajectory_planning::sampler::{
    PositionSampler, Sampler, SamplerInputs, SamplerNode, TimeSampler,
};
use cybflight_core::trajectory_planning::types::ZERO3;
use cybflight_sim::{
    Controller, ImuModel, MpcIndiController, PerfectImu, QuadPlant, Setpoint, VEHICLE,
};
use nalgebra::{SVector, Vector3};

const MPC_HORIZON_NODES: usize = 21; // SIMPLE_N + 1 in the firmware
const MPC_HORIZON_DT: f32 = 0.05; // SIMPLE_MPC_DT in the firmware
const SIM_DT: f32 = 1.0 / 8000.0; // matches MissionRunner's dt_sim
const TICK_HZ: f32 = 8000.0; // INDI tick rate
const SUBSTEPS_PER_TICK: usize = 1; // tick_dt = 1/8000 = dt_sim

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissionState {
    Idle,
    Executing,
}

/// Mirrors `outer_loop.rs::ActiveSetpoint`: a single `(timestamp, position,
/// yaw)` cell. RC integration in the firmware drifts `position` during
/// Idle; mission completion in the firmware writes the trajectory
/// endpoint here. We just hold `position` (yaw=0 always for this sim).
#[derive(Clone, Copy, Debug)]
struct ActiveSetpoint {
    position: Vector3<f32>,
}

/// In-test mirror of `outer_loop.rs`'s mission orchestration. Owns the
/// sampler, the active-setpoint cell, and the mission state machine.
struct MissionDriver {
    sampler: Sampler,
    state: MissionState,
    prev_state: MissionState,
    active: ActiveSetpoint,
    /// Active trajectory. `None` outside Executing.
    traj: Option<PiecewisePolynomial>,
    traj_start_t: f32,
    traj_total_s: f32,
}

impl MissionDriver {
    fn new(sampler: Sampler, initial_position: Vector3<f32>) -> Self {
        Self {
            sampler,
            state: MissionState::Idle,
            prev_state: MissionState::Idle,
            active: ActiveSetpoint {
                position: initial_position,
            },
            traj: None,
            traj_start_t: 0.0,
            traj_total_s: 0.0,
        }
    }

    fn start_mission(&mut self, traj: PiecewisePolynomial, start_t: f32) {
        self.traj_total_s = traj.total_duration();
        self.traj = Some(traj);
        self.traj_start_t = start_t;
        self.state = MissionState::Executing;
        // Mirror outer_loop.rs: reset() fires on Idle→Executing edge.
    }

    fn rc_set_active(&mut self, position: Vector3<f32>) {
        self.active.position = position;
    }

    /// Per-tick: fill `out` with horizon nodes and update mission state.
    /// Returns whether the mission state transitioned this tick (for
    /// assertions). The horizon-fill semantics mirror outer_loop.rs:
    /// during Executing, sample the trajectory; during Idle, hover at
    /// the active setpoint.
    fn tick(
        &mut self,
        now_s: f32,
        state_pos: Vector3<f32>,
        out: &mut [SamplerNode],
    ) -> Vec<Setpoint> {
        // Fire reset() on the Idle→Executing edge — outer_loop's policy.
        if self.state == MissionState::Executing && self.prev_state != MissionState::Executing {
            self.sampler.reset();
        }

        let mut horizon = Vec::with_capacity(out.len());

        match self.state {
            MissionState::Executing => {
                let traj = self.traj.as_ref().expect("Executing without trajectory");
                let tau0_s = (now_s - self.traj_start_t).max(0.0);
                let inputs = SamplerInputs {
                    traj,
                    total_duration_s: self.traj_total_s,
                    tau0_s,
                    state_pos,
                    horizon_dt: MPC_HORIZON_DT,
                };
                let result = self.sampler.sample(&inputs, out);

                for n in out.iter() {
                    horizon.push(Setpoint {
                        position: n.pos,
                        velocity: n.vel,
                        acceleration: n.acc,
                        yaw: 0.0,
                        terminal: n.past_end,
                    });
                }

                // Mission-done handoff — exactly the outer_loop pattern:
                // refresh active setpoint to the τ₀ sample BEFORE flipping
                // state, then transition Executing → Idle.
                self.active.position = out[0].pos;
                if result.mission_done {
                    // Trajectory endpoint as the hover target.
                    self.active.position = out[out.len() - 1].pos;
                    self.traj = None;
                    self.state = MissionState::Idle;
                }
            }
            MissionState::Idle => {
                // Hover at the active setpoint. Every horizon node gets
                // (active_pos, 0, 0). past_end stays false because there
                // is no trajectory to be past the end of.
                let p = self.active.position;
                for n in out.iter_mut() {
                    *n = SamplerNode {
                        pos: p,
                        vel: Vector3::zeros(),
                        acc: Vector3::zeros(),
                        past_end: false,
                    };
                }
                for _ in 0..out.len() {
                    horizon.push(Setpoint {
                        position: p,
                        velocity: Vector3::zeros(),
                        acceleration: Vector3::zeros(),
                        yaw: 0.0,
                        terminal: true,
                    });
                }
            }
        }

        self.prev_state = self.state;
        horizon
    }
}

fn plan_short_mission(start: Vector3<f32>, target: Vector3<f32>) -> PiecewisePolynomial {
    let vp = VEHICLE.build();
    let cfg = QuadPlanningConfig::from_vehicle_params(&vp);
    let input = PlannerInput::goto(start, ZERO3, target);
    let mut ws = Box::new(BfgsWorkspace::new());
    let res = plan_with_workspace(&input, &cfg, &mut ws);
    res.trajectory
}

/// Build plant + controller with default vehicle params, plant initialised
/// at `start_pos` with zero velocity and identity attitude.
fn build_rig(vp: VehicleParams, start_pos: Vector3<f32>) -> (QuadPlant, MpcIndiController) {
    let mut plant = QuadPlant::new(vp.clone(), SIM_DT);
    plant.reset(
        start_pos,
        Vector3::zeros(),
        nalgebra::UnitQuaternion::identity(),
    );
    let ctl = MpcIndiController::from_params(&vp);
    (plant, ctl)
}

/// Run the closed loop for `duration_s` seconds. Calls `rc_callback` once
/// per tick with `(time_s, &mut driver)` so the test can simulate RC
/// active-setpoint changes mid-flight. Returns the plant's final pose
/// plus the tick at which (if ever) the mission first transitioned to
/// Idle.
fn run_loop(
    plant: &mut QuadPlant,
    ctl: &mut MpcIndiController,
    driver: &mut MissionDriver,
    duration_s: f32,
    mut rc_callback: impl FnMut(f32, &mut MissionDriver),
) -> (Vector3<f32>, Option<f32>) {
    let n_ticks = (duration_s * TICK_HZ).round() as u32;
    let mut sample_buf = vec![SamplerNode::default(); MPC_HORIZON_NODES];
    let mut idle_at: Option<f32> = None;
    // Drag-free quadrotor specific force = body-z * Σthrust / mass. Seed
    // the previous-tick u with hover so the *very first* IMU sample
    // reports 1g body-z (matching what a real accelerometer reads with
    // the vehicle about to take off) instead of zero — INDI's takeoff
    // detector treats zero specific force as free-fall.
    let hover_per_motor = plant.params.body.mass_kg * 9.81 / MPC_NU as f32;
    let mut u_last = SVector::<f32, MPC_NU>::from_element(hover_per_motor);
    let mut imu_model = PerfectImu;
    for tick_idx in 0..n_ticks {
        let now_s = plant.time_s();
        rc_callback(now_s, driver);

        let state_pos = plant.position();
        let prev_state = driver.state;
        let horizon = driver.tick(now_s, state_pos, &mut sample_buf);
        if prev_state == MissionState::Executing && driver.state == MissionState::Idle {
            idle_at = Some(now_s);
        }

        let imu = imu_model.sample(plant, &u_last);
        let u = ctl.step(plant.raw_state(), &imu, &horizon);
        for _ in 0..SUBSTEPS_PER_TICK {
            plant.step(&u);
        }
        u_last = u;

        if !plant.position().x.is_finite() {
            panic!("plant diverged at tick {tick_idx} t={now_s:.3}");
        }
    }
    (plant.position(), idle_at)
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Both samplers must drive a short mission to its endpoint and then
/// transition Executing → Idle, at which point the controller hovers
/// stably at the trajectory endpoint.
///
/// The `expects_early_idle` flag captures one real semantic difference
/// between the two samplers: PositionSampler intentionally fires
/// `mission_done` early — the moment the *plant* enters its
/// `radius_of_acceptance` ball around the endpoint — even if τ_curr is
/// still short of `total_duration_s`. This is the cyblib
/// "reach-hover-by-entering-radius-of-acceptance" path. TimeSampler has
/// no such shortcut and only flips at τ ≥ total_duration_s.
fn mission_completion_for_sampler(name: &str, sampler: Sampler, expects_early_idle: bool) {
    let start = Vector3::new(0.0, 0.0, 1.0);
    let target = Vector3::new(1.0, 0.0, 1.0); // 1 m forward, same altitude
    let traj = plan_short_mission(start, target);
    let mission_duration_s = traj.total_duration();
    assert!(
        mission_duration_s > 0.5 && mission_duration_s < 5.0,
        "{name}: planned mission duration {mission_duration_s} s out of expected range"
    );

    let vp = VEHICLE.build();
    let (mut plant, mut ctl) = build_rig(vp, start);
    let mut driver = MissionDriver::new(sampler, start);
    driver.start_mission(traj, plant.time_s());

    // Run for the trajectory length plus a 1 s hover-hold so the
    // Executing→Idle handoff has time to settle.
    let total_s = mission_duration_s + 1.0;
    let (final_pos, idle_at) = run_loop(&mut plant, &mut ctl, &mut driver, total_s, |_, _| {});

    // (a) The mission must have transitioned to Idle.
    let idle_at = idle_at.unwrap_or_else(|| {
        panic!("{name}: mission never transitioned Executing→Idle within {total_s}s")
    });
    if expects_early_idle {
        // PositionSampler may flip to Idle as soon as the plant enters
        // radius_of_acceptance around the endpoint — anywhere from
        // ~0.5·mission_duration onward is plausible for a 1 m mission.
        assert!(
            idle_at <= mission_duration_s + 0.1,
            "{name}: Idle handoff at {idle_at:.3}s should be ≤ {mission_duration_s:.3}s + 0.1",
        );
    } else {
        // TimeSampler has no early shortcut; Idle should fire near the
        // trajectory's nominal end.
        assert!(
            idle_at >= mission_duration_s - 0.1 && idle_at <= mission_duration_s + 0.1,
            "{name}: Idle handoff at {idle_at:.3}s, expected near {mission_duration_s:.3}s"
        );
    }
    assert_eq!(driver.state, MissionState::Idle, "{name}: state");

    // (b) Active setpoint refreshed near the trajectory endpoint. For
    //     TimeSampler this is exact (last horizon node = trajectory end).
    //     For PositionSampler with early Idle, the active setpoint is
    //     the τ_curr sample at the moment of transition, which can be
    //     slightly short of the actual endpoint. We allow 20 cm slack
    //     for that path.
    let active_pos = driver.active.position;
    let active_tol = if expects_early_idle { 0.20 } else { 0.05 };
    assert!(
        (active_pos - target).norm() < active_tol,
        "{name}: active setpoint after Idle handoff is {active_pos:?}, expected within {active_tol} of {target:?}"
    );

    // (c) Plant settled within terminal tolerance.
    assert!(
        (final_pos - target).norm() < 0.15,
        "{name}: plant final pos {final_pos:?} not within 15 cm of target {target:?}"
    );
}

#[test]
fn time_sampler_completes_mission_and_transitions_to_idle() {
    mission_completion_for_sampler("time", Sampler::Time(TimeSampler::new()), false);
}

#[test]
fn position_sampler_completes_mission_and_transitions_to_idle() {
    let params = VEHICLE.build().sampler.to_position_sampler_params();
    mission_completion_for_sampler(
        "position",
        Sampler::Position(PositionSampler::new(params)),
        true,
    );
}

/// Invariant: while in Idle (mission complete or never started), the
/// controller tracks the active setpoint — *not* whatever τ-position the
/// sampler converged to during the previous mission.
///
/// Sequence:
///   1. Plan + execute a mission from A=(0,0,1) to B=(1,0,1).
///   2. After mission completion, simulate an "RC stick movement" by
///      rewriting `active.position` to C=(0,0.5,1).
///   3. Hold for 1.5 s.
///   4. Assert plant is now hovering near C, not B.
///
/// For PositionSampler this is the critical guarantee: the sampler's
/// `prev_query_tau` is from the (now-finished) mission to B; if the
/// outer-loop logic accidentally kept calling `sampler.sample()` during
/// Idle, the controller would chase the trajectory's endpoint forever
/// and ignore the RC change. The driver's `match self.state` branches
/// guarantee otherwise — this test enforces that boundary.
fn rc_setpoint_change_for_sampler(name: &str, sampler: Sampler) {
    let start = Vector3::new(0.0, 0.0, 1.0);
    let mission_target = Vector3::new(1.0, 0.0, 1.0);
    let rc_target = Vector3::new(0.0, 0.5, 1.0);
    let traj = plan_short_mission(start, mission_target);
    let mission_duration_s = traj.total_duration();

    let vp = VEHICLE.build();
    let (mut plant, mut ctl) = build_rig(vp, start);
    let mut driver = MissionDriver::new(sampler, start);
    driver.start_mission(traj, plant.time_s());

    // Phase 1: run mission to completion + 0.5 s settle.
    let phase1 = mission_duration_s + 0.5;
    let (after_mission_pos, idle_at) =
        run_loop(&mut plant, &mut ctl, &mut driver, phase1, |_, _| {});
    let idle_at = idle_at.unwrap_or_else(|| {
        panic!("{name}: mission never reached Idle, can't test RC change")
    });
    assert!(idle_at < phase1, "{name}: idle should have happened in phase1");
    assert!(
        (after_mission_pos - mission_target).norm() < 0.10,
        "{name}: after mission, expected near {mission_target:?}, got {after_mission_pos:?}"
    );
    assert_eq!(driver.state, MissionState::Idle);

    // Phase 2: simulate RC writing a new active setpoint mid-Idle.
    // The first rc_callback invocation does the write; subsequent calls
    // are no-ops. Hold for 1.5 s — plenty of time at 4 m/s² for a 0.5 m
    // step.
    let mut applied = false;
    let phase2 = 1.5;
    let (after_rc_pos, _) = run_loop(
        &mut plant,
        &mut ctl,
        &mut driver,
        phase2,
        |_now, drv| {
            if !applied {
                drv.rc_set_active(rc_target);
                applied = true;
            }
        },
    );

    // Plant must now track the RC-set point, not the previous mission target.
    let dist_to_rc = (after_rc_pos - rc_target).norm();
    let dist_to_mission = (after_rc_pos - mission_target).norm();
    assert!(
        dist_to_rc < 0.15,
        "{name}: after RC setpoint change, plant at {after_rc_pos:?}, expected near {rc_target:?} (d={dist_to_rc:.3}m vs mission target d={dist_to_mission:.3}m)"
    );
    assert!(
        dist_to_rc < dist_to_mission,
        "{name}: plant closer to old mission target ({dist_to_mission:.3}m) than RC target ({dist_to_rc:.3}m) — RC change was ignored!"
    );
    // And state still Idle — RC writes do NOT re-arm the mission.
    assert_eq!(driver.state, MissionState::Idle);
}

#[test]
fn time_sampler_honours_rc_setpoint_change() {
    rc_setpoint_change_for_sampler("time", Sampler::Time(TimeSampler::new()));
}

#[test]
fn position_sampler_honours_rc_setpoint_change() {
    let params = VEHICLE.build().sampler.to_position_sampler_params();
    rc_setpoint_change_for_sampler(
        "position",
        Sampler::Position(PositionSampler::new(params)),
    );
}

/// Sanity: the inline horizon-fill we do in MissionDriver (Idle branch
/// uses active setpoint, Executing branch uses sampler) matches what the
/// firmware controller expects — `MpcIndiController::step` consumes the
/// horizon untouched. This test just exercises a hover-only run for both
/// samplers and asserts the plant doesn't drift, catching any wiring
/// mistake in the test harness itself.
fn hover_holds_for_sampler(name: &str, sampler: Sampler) {
    let start = Vector3::new(0.5, -0.3, 1.2);
    let vp = VEHICLE.build();
    let (mut plant, mut ctl) = build_rig(vp, start);
    let mut driver = MissionDriver::new(sampler, start);
    // Do NOT start a mission — straight Idle.
    let (final_pos, idle_at) = run_loop(&mut plant, &mut ctl, &mut driver, 2.0, |_, _| {});
    assert!(idle_at.is_none(), "{name}: idle_at should be None for hover-only run");
    let drift = (final_pos - start).norm();
    assert!(
        drift < 0.10,
        "{name}: hover drifted by {drift:.3}m from {start:?} to {final_pos:?}"
    );
}

#[test]
fn time_sampler_idle_hover_holds() {
    hover_holds_for_sampler("time", Sampler::Time(TimeSampler::new()));
}

#[test]
fn position_sampler_idle_hover_holds() {
    let params = VEHICLE.build().sampler.to_position_sampler_params();
    hover_holds_for_sampler(
        "position",
        Sampler::Position(PositionSampler::new(params)),
    );
}

