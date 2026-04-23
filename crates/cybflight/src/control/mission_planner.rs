//! Mission planner task (outer_mpc + est_eskf only).
//!
//! Blocks on `PLAN_REQUEST` (rising edge from `rc_interpreter_task` on the
//! mission-trigger AUX channel). On each request:
//!
//!   1. Reads `ACTIVE_POSITION_SETPOINT` — the shared single-source-of-truth
//!      cell that rc_interpreter writes during Idle. Using this as the
//!      start pose means the mission is planned from the exact reference
//!      the drone is currently flying to, so the first trajectory sample
//!      and the last Idle setpoint are structurally identical — no step
//!      when the outer loop switches to the Executing branch.
//!   2. Builds hardcoded circular waypoints at the current altitude, with
//!      the **terminal position equal to the start position** (return home).
//!   3. Solves a min-jerk MINCO trajectory using a **static** BFGS workspace
//!      (~35 KB in BSS, not on the task stack).
//!   4. Validates the result (converged status, finite duration > 0).
//!   5. On success: stores the trajectory in `MISSION_TRAJECTORY_SLOT`,
//!      transitions `MISSION_STATE` → `Executing`. The outer loop picks up
//!      the trajectory on its next 100 Hz tick and samples one reference
//!      state per MPC horizon node.
//!
//! Solver latency (up to ~100 ms) does not block the MPC or INDI loops:
//! they run on their own `Ticker`s on the same cooperative executor. This
//! task only holds the CPU while it is computing — which happens **at
//! most once per mission**.
//!
//! ## Safety
//! - Solver workspace is BSS-resident (`StaticCell`) → task stack stays tiny.
//! - Trajectory is validated before publication; a bad solve leaves the
//!   mission in `Idle` with sticks still in charge.
//! - A plan request is **rejected** if `ACTIVE_POSITION_SETPOINT` is
//!   `None`, non-finite, or its timestamp is older than
//!   `SETPOINT_STALE_TIMEOUT` — the active writer (rc_interpreter on
//!   Idle, outer_loop on Executing) refreshes the cell each tick, so a
//!   stale stamp means the controller chain is not live (disarmed,
//!   estimator unready, or tick stalled) and the solver would be seeded
//!   from a value the controller cannot smoothly pick up.
//! - On failsafe/disarm, `failsafe_task` clears `MISSION_STATE` and the
//!   trajectory slot, so a re-arm never resumes a stale mission.

#![cfg(feature = "outer_mpc")]

use core::sync::atomic::Ordering;

use cybflight_core::trajectory_planning::MAX_PIECES;
use embassy_time::{Duration, Instant};
use static_cell::StaticCell;

use cybflight_core::trajectory_planning::bfgs_trust::BfgsWorkspace;
use cybflight_core::trajectory_planning::planner::{
    plan_finalize, plan_init, plan_resume, PlannerInput, SolverStatus,
};
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_core::trajectory_planning::types::Vec3;

use crate::msgs;

use super::{
    read_active_setpoint, MissionState, MissionTrajectory, MISSION_ABORT_REQUESTED, MISSION_STATE,
    MISSION_STATUS, MISSION_TRAJECTORY_SLOT, PLAN_REQUEST,
};

/// BFGS scratch memory (~35 KB). BSS-resident; init-once on first plan.
static WORKSPACE: StaticCell<BfgsWorkspace> = StaticCell::new();

/// Heartbeat cadence for `MISSION_STATUS` during Planning. The outer loop
/// owns the heartbeat when it is the authoritative state-writer (Idle /
/// Executing), but while we hold state=Planning it may early-continue on
/// stale odometry — so the planner publishes its own status. ~100 ms
/// matches the outer loop's own publish decimation.
const PLANNING_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(100);

/// Number of intermediate waypoints on the test circle. Total trajectory
/// pieces = `NUM_CIRCLE_WAYPOINTS + 1` (≤ `MAX_PIECES` = 16).
const NUM_CIRCLE_WAYPOINTS: usize = 8;

/// Circle radius [m].
const CIRCLE_RADIUS_M: f32 = 1.0;

/// Reject `ACTIVE_POSITION_SETPOINT` older than this when snapshotting the
/// mission start pose. rc_interpreter refreshes the cell's timestamp on
/// every Idle-state RC frame (~150 Hz), so anything older than ~50 ms
/// means rc_interpreter is not running (disarmed, estimator not ready,
/// or the RC link is dead) — in which case the setpoint is not a
/// faithful representation of "where the drone will be when the
/// trajectory starts".
const SETPOINT_STALE_TIMEOUT: Duration = Duration::from_millis(50);

/// Hard ceiling on trajectory duration. Rejects pathological solves that
/// would leave the vehicle committed to a 10-minute mission.
const MAX_TRAJECTORY_DURATION_S: f32 = 30.0;

/// Hard floor on trajectory duration. A solve producing near-zero duration
/// means the solver collapsed all segment times — almost certainly invalid.
const MIN_TRAJECTORY_DURATION_S: f32 = 0.5;

/// Wall-clock budget for a single solve. If the BFGS solver has not
/// converged within this window, it is aborted gracefully and the
/// mission returns to Idle with the drone continuing to hover on its
/// prior setpoint — **no watchdog reset**.
///
/// With cooperative yielding (yield_now() between every BFGS burst), the
/// IWDG feed task runs freely between iterations, so the budget is no
/// longer constrained by the 500 ms IWDG timeout. The binding constraint
/// is the outer_loop MPC tick: at ~4 ms per solve and a 10 ms timer
/// period, the outer_loop occupies ~40% of the Thread executor, leaving
/// mission_planner ~6 ms per window → ~3 BFGS iterations per 10 ms.
/// With max_iterations = 500, worst-case wall time ≈ 500 / 3 × 10 ms
/// ≈ 1670 ms. 3000 ms gives comfortable headroom; the drone hovers
/// safely in Planning while the pilot waits.
const SOLVE_BUDGET: Duration = Duration::from_millis(5000);

/// Outer BFGS iterations per cooperative-yield burst. After this many
/// iterations the solver returns to the async context so peer thread-
/// executor tasks (MPC outer loop, ESKF, GPS, baro, mag) can run; we
/// then `yield_now().await` and resume the solve.
///
/// Set to 1: one outer iteration of MINCO + cost eval is ~1–3 ms on
/// STM32H743. Yielding every iteration limits the uninterrupted CPU
/// window to ~3 ms — the MPC outer loop (10 ms timer, ~4 ms tick body)
/// is therefore delayed by at most one burst duration, keeping its
/// effective rate close to the intended 100 Hz.
///
/// Note: the effective yield overhead is NOT the bare yield_now() cost
/// (~2–5 μs). When the outer_loop timer fires during a yield, the
/// ~4 ms MPC solve runs before mission_planner resumes, producing an
/// average ~1.6 ms overhead per yield. SOLVE_BUDGET accounts for this.
const BFGS_ITERS_PER_YIELD: usize = 1;

/// Build the hardcoded circular waypoint list returning to `start`.
///
/// The trajectory passes through `NUM_CIRCLE_WAYPOINTS` points on a circle
/// of radius `CIRCLE_RADIUS_M` centered one radius to +X of `start`, at
/// altitude `start.z`, and the final waypoint is `start` itself (return home).
///
/// Layout (phase 0 is at the start; phases evenly spaced over 2π):
///   center = start + [r, 0, 0]
///   wp_i   = center + [-r·cos(θ_i), r·sin(θ_i), 0], i = 1..N
///   tail   = start  (returned as the final element)
// fn circular_waypoints_return_home(start: [f32; 3]) -> [Vec3; NUM_CIRCLE_WAYPOINTS + 1] {
//     let mut out = [[0.0_f32; 3]; NUM_CIRCLE_WAYPOINTS + 1];
//     let cx = start[0] + CIRCLE_RADIUS_M;
//     let cy = start[1];
//     let cz = start[2];
//     for i in 0..NUM_CIRCLE_WAYPOINTS {
//         // Phase 0 is at the start point itself; we want the first waypoint
//         // to be 2π/(N+1) past phase 0 so we land on `start` again at the
//         // final (return-home) waypoint.
//         let theta =
//             (i + 1) as f32 * core::f32::consts::TAU * 2.0 / (NUM_CIRCLE_WAYPOINTS as f32 + 1.0);
//         out[i] = [
//             cx - CIRCLE_RADIUS_M * libm::cosf(theta),
//             cy + CIRCLE_RADIUS_M * libm::sinf(theta),
//             cz,
//         ];
//     }
//     // Final target: exact start position (return home).
//     out[NUM_CIRCLE_WAYPOINTS] = start;
//     out
// }

/// Atomically read `ACTIVE_POSITION_SETPOINT` and validate it as a plan seed.
///
/// Returns `Some(position)` iff the cell holds a finite value whose
/// timestamp is fresh (≤ `SETPOINT_STALE_TIMEOUT`) and not future-dated.
/// Guarantees the caller:
/// * **Finite**: no NaN/±∞ — planning on a degenerate seed would
///   propagate NaN through MINCO and BFGS.
/// * **Fresh**: rc_interpreter has written within the liveness window,
///   proving the RC link + ESKF + estimator-ready gate are all alive.
/// * **Monotonic**: defends against clock skew or a corrupt future stamp.
fn try_snapshot_setpoint() -> Option<nalgebra::Vector3<f32>> {
    let sp = read_active_setpoint()?;
    let now = Instant::now();
    let p = &sp.position;
    let finite = p.x.is_finite() && p.y.is_finite() && p.z.is_finite();
    if !finite {
        return None;
    }
    if sp.timestamp > now {
        return None; // future-dated; reject
    }
    if now.duration_since(sp.timestamp) > SETPOINT_STALE_TIMEOUT {
        return None;
    }
    Some(sp.position)
}

#[embassy_executor::task]
pub async fn mission_planner_task() {
    let workspace: &mut BfgsWorkspace = WORKSPACE.init(BfgsWorkspace::new());
    let mission_status_pub = MISSION_STATUS.immediate_publisher();
    // Build the planner config from the *live* vehicle params (mass,
    // inertia, motor thrust caps, planner tunables) rather than the
    // hardcoded `default()` — otherwise the trajectory is solved for a
    // 0.55 kg quad regardless of what the firmware is actually flying,
    // and the outer-loop's flatness map (which uses live mass via
    // `mpc_problem`) sees a physically inconsistent acceleration profile.
    // Refreshed per PLAN_REQUEST so a between-mission param edit takes
    // effect on the next solve.
    let mut config = QuadPlanningConfig::from_vehicle_params(&crate::params::get());
    let mut local_param_ver = crate::params::PARAM_VERSION.load(Ordering::Acquire);

    defmt::info!("mission_planner: ready (idle)");

    loop {
        // Block until the RC trigger fires.
        PLAN_REQUEST.wait().await;

        // Refresh planner config if vehicle params changed since the
        // last solve. Only safe between solves (no in-flight BFGS state
        // depends on `config`).
        let cur = crate::params::PARAM_VERSION.load(Ordering::Acquire);
        if cur != local_param_ver {
            local_param_ver = cur;
            config = QuadPlanningConfig::from_vehicle_params(&crate::params::get());
            defmt::info!("mission_planner: planner config reloaded (ver {})", cur);
        }

        // Clear any stale abort flag left over from a previous mission
        // (e.g. failsafe-on-disarm scenarios). A fresh plan request is
        // the user saying "yes, go" — the slate starts clean.
        MISSION_ABORT_REQUESTED.store(false, Ordering::Release);

        // Only honor requests while Idle. (The edge detector in
        // rc_interpreter already guards against this, but verify here too:
        // a stale signal could have been pending across a failsafe reset.)
        let state = MissionState::from_u8(MISSION_STATE.load(Ordering::Acquire));
        if state != MissionState::Idle {
            defmt::warn!(
                "mission_planner: PLAN_REQUEST ignored, state={}",
                state as u8
            );
            continue;
        }

        // Snapshot the controller's current reference (what it is actively
        // tracking right now). A `None` here means `ACTIVE_POSITION_SETPOINT`
        // is uninitialized, non-finite, or its timestamp is stale — i.e.
        // rc_interpreter is not refreshing the cell. Drop the request
        // rather than seeding BFGS with junk.
        //
        // Start velocity is zero: the mission trigger fires with the
        // drone hovering, and rc-integrated setpoints carry no
        // feedforward velocity.
        let start_position = match try_snapshot_setpoint() {
            Some(p) => p,
            None => {
                defmt::warn!(
                    "mission_planner: no fresh active setpoint (controller not tracking?) — request dropped"
                );
                continue;
            }
        };

        let start_pos: [f32; 3] = [start_position.x, start_position.y, start_position.z];
        let start_vel: [f32; 3] = [0.0, 0.0, 0.0];

        // Enter Planning — sticks are already gated by rc_interpreter from
        // the moment it sent the trigger, but make the state machine honest.
        MISSION_STATE.store(MissionState::Planning as u8, Ordering::Release);

        let targets = [
            [-0.3267, -2.231, 1.6],
            [-1.845, 1.942, 1.0],
            [2.292, 1.637, 1.0],
            [2.547, -2.108, 1.8],
            [2.547, -2.108, 0.8],
            [0.3099, 0.3554, 1.0],
            [-2.396, -2.214, 1.0],
            [-0.3267, -2.231, 1.6],
            [-1.845, 1.942, 1.0],
            [2.292, 1.637, 1.0],
            [2.547, -2.108, 1.8],
            [2.547, -2.108, 0.8],
            [0.3099, 0.3554, 1.0],
            [-2.396, -2.214, 1.0],
            [-0.3267, -2.231, 1.6],
        ];

        let input = PlannerInput::waypoints(start_pos, start_vel, &targets);

        defmt::info!(
            "mission_planner: solving (start=[{},{},{}], {} waypoints + return)",
            start_pos[0],
            start_pos[1],
            start_pos[2],
            NUM_CIRCLE_WAYPOINTS
        );

        // Wall-clock deadline for the solve. The inner BFGS polls
        // `keep_going` at each outer iteration; returning `false`
        // cleanly aborts with `SolverStatus::TimeExceeded`.
        //
        // The 250 ms budget is chosen to stay well under the 500 ms
        // IWDG timeout even when stacked with the worst-case 200 ms
        // feed interval (250 + 200 = 450 ms < 500 ms). The IWDG is the
        // sole safety net here; we intentionally do NOT extend it
        // around the solve — see design rationale in watchdog.rs.
        //
        // The callback ALSO returns false when `IS_ARMED` clears. If
        // the pilot emergency-disarms during the solve, the RC parser
        // (on ctrl_spawner P10) detects the arm-switch edge and
        // signals `ARM_STATE`; DShot (P6) consumes it and sets
        // `IS_ARMED = false`. The solver sees that on its next outer
        // iteration (≤20 ms) and exits — so no stale trajectory ever
        // lands in `MISSION_TRAJECTORY_SLOT` from a disarmed attempt.
        //
        // Cooperative yielding: the solve runs in bursts of
        // `BFGS_ITERS_PER_YIELD` iterations. Between bursts we
        // `yield_now().await` so peer thread-executor tasks (GPS,
        // baro, mag) get scheduled — a long solve no longer starves
        // them for the full 250 ms budget.
        let t0 = Instant::now();
        let deadline = t0 + SOLVE_BUDGET;
        let mut keep_going =
            || Instant::now() < deadline && crate::motors::IS_ARMED.load(Ordering::Acquire);
        let mut session = plan_init(&input, &config, workspace);
        let mut last_heartbeat = t0;
        let status = loop {
            match plan_resume(
                &mut session,
                &config,
                workspace,
                &mut keep_going,
                BFGS_ITERS_PER_YIELD,
            ) {
                Some(s) => break s,
                None => {
                    // Burst done but solver still running. Re-check the
                    // abort conditions eagerly (cheap) and then yield so
                    // GPS / baro / mag tasks can make progress before we
                    // start the next burst.
                    if !keep_going() {
                        break SolverStatus::TimeExceeded;
                    }
                    let now = Instant::now();
                    if now.duration_since(last_heartbeat) >= PLANNING_HEARTBEAT_INTERVAL {
                        last_heartbeat = now;
                        mission_status_pub.publish_immediate(msgs::MissionStatus {
                            timestamp: now,
                            state: MissionState::Planning as u8,
                            tau_s: 0.0,
                            total_duration_s: 0.0,
                            target_position: start_position,
                        });
                    }
                    embassy_futures::yield_now().await;
                }
            }
        };
        // Yield once before plan_finalize so the IWDG feed task and any
        // other pending thread-executor tasks (ESKF, GPS) can run. The
        // finalize call is synchronous and potentially several ms long;
        // a single yield ensures the last-fed timestamp stays fresh even
        // if the disarm path gets here with the IWDG nearly exhausted.
        embassy_futures::yield_now().await;
        let result = plan_finalize(session, workspace, status);
        let elapsed_ms = Instant::now().duration_since(t0).as_millis();

        defmt::info!(
            "mission_planner: solve done ({}ms, iters={}, cost={}, status={})",
            elapsed_ms,
            result.iterations,
            result.final_cost,
            result.status as u8
        );

        // Classify the outcome.
        //
        // `TimeExceeded` is the graceful-abort path: the solver hit
        // the wall-clock budget. Treat it distinctly from a numerical
        // failure so the pilot's defmt log shows the true cause. The
        // drone stays on its prior setpoint (the hover position from
        // before the mission trigger) and the pilot can either retry
        // or keep flying manually.
        if result.status == SolverStatus::TimeExceeded {
            // Either the wall-clock budget elapsed, or IS_ARMED cleared
            // (emergency disarm). Both are graceful exits; distinguish
            // in the log so the pilot/defmt reader knows why.
            if !crate::motors::IS_ARMED.load(Ordering::Acquire) {
                defmt::warn!(
                    "mission_planner: disarm detected mid-solve — aborting (no trajectory)"
                );
            } else {
                defmt::warn!(
                    "mission_planner: solve exceeded {}ms budget — aborting mission cleanly",
                    SOLVE_BUDGET.as_millis()
                );
            }
            MISSION_STATE.store(MissionState::Idle as u8, Ordering::Release);
            continue;
        }

        // Convergence / Stop / MaxIterations are all "solver returned a
        // trajectory" outcomes; validate the numerical result. Reject
        // InvalidValue outright.
        let converged = matches!(
            result.status,
            SolverStatus::Convergence | SolverStatus::Stop | SolverStatus::MaxIterations
        );
        let dur = result.trajectory.total_duration();
        let valid = converged
            && dur.is_finite()
            && dur >= MIN_TRAJECTORY_DURATION_S
            && dur <= MAX_TRAJECTORY_DURATION_S
            && result.final_cost.is_finite();

        if !valid {
            defmt::warn!(
                "mission_planner: rejecting trajectory (status={}, dur={})",
                result.status as u8,
                dur
            );
            MISSION_STATE.store(MissionState::Idle as u8, Ordering::Release);
            continue;
        }

        // If the pilot aborted during the ~100 ms solve, drop the result
        // on the floor — do NOT transition to Executing. The outer loop
        // will not pick up the abort flag because the state never leaves
        // Idle, so we clear the flag here ourselves.
        if MISSION_ABORT_REQUESTED.load(Ordering::Acquire) {
            defmt::info!("mission_planner: user aborted during solve, trajectory discarded");
            MISSION_ABORT_REQUESTED.store(false, Ordering::Release);
            MISSION_STATE.store(MissionState::Idle as u8, Ordering::Release);
            continue;
        }

        // Also handle the failsafe-during-solve case: if `enter_failsafe`
        // hard-reset `MISSION_STATE` to Idle (and cleared the abort flag
        // along with it), the planner must NOT re-enter Executing — that
        // would leak a stale trajectory anchored at a now-obsolete
        // `t_start` into the slot, which a future re-arm would then
        // resume from.
        let state_now = MissionState::from_u8(MISSION_STATE.load(Ordering::Acquire));
        if state_now != MissionState::Planning {
            defmt::warn!(
                "mission_planner: state left Planning during solve ({}), discarding",
                state_now as u8
            );
            continue;
        }

        // Publish & transition under the SAME critical section.
        //
        // Both the slot write and the `MISSION_STATE = Executing` flip
        // happen inside one `MISSION_TRAJECTORY_SLOT.lock` so that any
        // task which observes `Executing` and then takes the slot lock
        // is guaranteed to see the trajectory. Without this, the
        // sequence (planner unlock → failsafe takes lock, clears slot,
        // sets Idle → planner stores Executing → outer_loop sees
        // Executing + empty slot) was reachable. failsafe.rs writes its
        // state flip inside the same lock for the symmetric reason.
        // Re-check abort + state INSIDE the publish lock. The outer
        // checks above (lines 348-394) raced against any concurrent
        // state writer (outer_loop's abort path, failsafe). Both of
        // those writers also take this same lock to flip MISSION_STATE,
        // so re-reading the flag and the state here serializes the
        // publish against them — closing the window where:
        //   - planner sees abort=false, state=Planning;
        //   - outer_loop's tick consumes the abort flag (swap → false),
        //     locks slot, stores state=Idle, releases;
        //   - planner acquires lock and overwrites with state=Executing.
        // We `swap(false)` rather than `load` so this consumes the
        // intent symmetrically with the outer_loop abort branch.
        let t_start = Instant::now();
        let mut published = false;
        MISSION_TRAJECTORY_SLOT.lock(|slot| {
            if MISSION_ABORT_REQUESTED.swap(false, Ordering::AcqRel) {
                return;
            }
            if MissionState::from_u8(MISSION_STATE.load(Ordering::Acquire))
                != MissionState::Planning
            {
                return;
            }
            *slot.borrow_mut() = Some(MissionTrajectory {
                traj: result.trajectory,
                t_start,
                total_duration_s: dur,
            });
            MISSION_STATE.store(MissionState::Executing as u8, Ordering::Release);
            published = true;
        });
        if !published {
            defmt::warn!(
                "mission_planner: publish suppressed (abort/state changed during lock acquisition)"
            );
            continue;
        }
        defmt::info!("mission_planner: → Executing (duration={}s)", dur);
        // Publish the Executing state immediately so the ground station sees
        // state=2 without waiting up to 100 ms for the outer_loop's decimated
        // MissionStatus heartbeat.
        mission_status_pub.publish_immediate(msgs::MissionStatus {
            timestamp: t_start,
            state: MissionState::Executing as u8,
            tau_s: 0.0,
            total_duration_s: dur,
            target_position: start_position,
        });
    }
}
