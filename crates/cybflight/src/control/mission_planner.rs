//! Mission planner task (outer_mpc + est_eskf only).
//!
//! Blocks on `PLAN_REQUEST` (rising edge from `rc_interpreter_task` on the
//! mission-trigger AUX channel). On each request the task snapshots the
//! controller's current setpoint and dispatches to one of two planning
//! schemas (selected at compile time by [`USE_OFFLINE_PLAN`]):
//!
//! - **Online** ([`plan_online`]): builds hardcoded waypoints and solves
//!   a min-jerk MINCO trajectory with the on-device BFGS optimizer. The
//!   solver runs cooperatively in 1-iteration bursts so peer Embassy
//!   tasks keep their scheduling slots; convergence/validity rejects fall
//!   back to Idle without disarming.
//! - **Offline** ([`plan_offline`]): consumes a precomputed waypoint +
//!   timestamp schedule from [`super::offline_mission`] (60 pieces) and
//!   feeds it directly into the MINCO banded solver — no BFGS, no
//!   cooperative yielding, single-digit-ms wall time.
//!
//! Either path returns a [`PlanOutcome`]; the outer task body validates
//! it against an abort/state-change re-check inside the
//! `MISSION_TRAJECTORY_SLOT` lock (the same critical section that
//! `failsafe.rs` takes on disarm) and either publishes the trajectory
//! and flips `MISSION_STATE → Executing`, or emits a reject breadcrumb
//! `MissionStatus` so the GCS can diagnose without RTT/defmt.
//!
//! Reading `ACTIVE_POSITION_SETPOINT` as the start pose means the
//! mission is planned from the exact reference the drone is currently
//! flying to, so the first trajectory sample and the last Idle setpoint
//! are structurally identical — no step when the outer loop switches to
//! the Executing branch. The offline path can override this with
//! [`OFFLINE_USE_YAML_START`].
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

use embassy_sync::pubsub::publisher::ImmediatePublisher;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_time::{Duration, Instant, Timer};
use static_cell::StaticCell;

use cybflight_core::trajectory_planning::bfgs_trust::BfgsWorkspace;
use cybflight_core::trajectory_planning::minco_snap::MincoSnap;
use cybflight_core::trajectory_planning::piecewise_polynomial::PiecewisePolynomial;
use cybflight_core::trajectory_planning::planner::{
    plan_finalize, plan_init, plan_resume, PlannerInput, SolverStatus,
};
use cybflight_core::trajectory_planning::quad_planning_config::QuadPlanningConfig;
use cybflight_core::trajectory_planning::types::{Vec3, ZERO3};

use crate::msgs;

use super::offline_mission::{self, OFFLINE_MAX_PIECES};
use super::{
    read_active_setpoint, MissionState, MissionTrajectory, MISSION_ABORT_REQUESTED, MISSION_STATE,
    MISSION_STATUS, MISSION_TRAJECTORY_SLOT, PLAN_REQUEST,
};

// ─── Planning schema flags ────────────────────────────────────────────
//
// Both flags are compile-time `const bool`. Flip and rebuild to switch.
// Kept as flags (not Cargo features) so that toggling between schemas
// doesn't require a feature recompile of `cybflight_core` and the BSP.

/// Selects which planning function the task dispatches to per
/// `PLAN_REQUEST`. `false` runs [`plan_online`] (the on-device BFGS
/// trajectory optimizer); `true` runs [`plan_offline`] (load the
/// precomputed waypoints + timestamps from [`super::offline_mission`]
/// and feed them directly to a MINCO solver).
const USE_OFFLINE_PLAN: bool = true;

/// Only consulted when [`USE_OFFLINE_PLAN`] is `true`.
/// `false` (default): the offline trajectory's head pose is the live
///   `ACTIVE_POSITION_SETPOINT` — the same seamless-handoff semantics
///   the online path uses, so the first trajectory sample matches the
///   last Idle setpoint.
/// `true`: head pose is the YAML-recorded
///   [`OFFLINE_START_POS`]. Useful for replaying the offline solve
///   verbatim regardless of where the drone is hovering.
const OFFLINE_USE_YAML_START: bool = false;

// Local reject reason for the offline-path "non-monotonic / non-finite
// segment durations" guard. The added constant lives in the local
// `cybflight-msgs` working copy (`SOLVE_REJECT_INVALID_TIMES = 8`)
// but the firmware currently pins to the published 0.1.17 registry
// version, which doesn't yet include it. Defined here so the build
// works against the published crate; once a release containing the
// new constant lands, drop this and switch to `msgs::SOLVE_REJECT_INVALID_TIMES`.
const SOLVE_REJECT_INVALID_TIMES: u8 = 8;

// Local reject reason for the offline-path "active profile fails the
// length / bound invariants" guard (waypoint count != timestamp count,
// zero-length, or > OFFLINE_MAX_PIECES). Same temporary-shim story as
// SOLVE_REJECT_INVALID_TIMES — fold into `msgs` once the next msgs
// release lands.
const SOLVE_REJECT_PROFILE_INVALID: u8 = 9;

/// BFGS scratch memory (~35 KB). BSS-resident; init-once on first plan.
/// Only consumed by the online planner (`plan_online`); allocation is
/// unconditional because `StaticCell::new()` is `const` and free at
/// rest, so no need to gate it on `USE_OFFLINE_PLAN`.
static WORKSPACE: StaticCell<BfgsWorkspace> = StaticCell::new();

/// MINCO-snap scratch memory for the offline planner. BSS-resident;
/// init once on first plan. ~42 KB at `MAX_PIECES = 64` (8N×8N banded
/// LU + 8N coefficient rows + seven `[f32; MAX_PIECES]` time tables).
/// Like `WORKSPACE`, allocated unconditionally — only `plan_offline`
/// ever touches it.
static OFFLINE_MINCO: StaticCell<MincoSnap> = StaticCell::new();

/// Heartbeat cadence for `MISSION_STATUS` during Planning. The outer loop
/// owns the heartbeat when it is the authoritative state-writer (Idle /
/// Executing), but while we hold state=Planning it may early-continue on
/// stale odometry — so the planner publishes its own status. ~100 ms
/// matches the outer loop's own publish decimation.
const PLANNING_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(100);

/// Number of intermediate waypoints used by the legacy circle test —
/// retained as a defmt log breadcrumb in `plan_online`. The actual
/// hardcoded waypoint set there is the larger 19-target list, so this
/// is a conservative lower bound on what gets logged, not a contract.
const NUM_CIRCLE_WAYPOINTS: usize = 8;

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
///
/// Sized to admit legitimate long-circuit solves: a 19-piece mission with
/// the `d.max(1.0)` seed (init ≈ 60 s) converges to ~30 s in flight, which
/// tripped a previous 30 s cap. 120 s gives comfortable headroom for a
/// 20-piece circuit at ~3 m/s while still catching runaway solves; the
/// `compression_ok` guard (dur < 0.9 × init_dur) independently rejects
/// under-compressed trajectories, so this ceiling is a belt-and-braces
/// check rather than the primary gate.
const MAX_TRAJECTORY_DURATION_S: f32 = 120.0;

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
/// is the outer_loop MPC tick: at ~4 ms per solve and a 20 ms timer
/// period, the outer_loop occupies ~20% of the Thread executor, leaving
/// mission_planner ~16 ms per window → ~3 BFGS iterations per 20 ms.
/// With max_iterations = 500, worst-case wall time ≈ 500 / 3 × 20 ms
/// ≈ 3340 ms. 5000 ms gives comfortable headroom; the drone hovers
/// safely in Planning while the pilot waits.
const SOLVE_BUDGET: Duration = Duration::from_millis(5000);

/// Outer BFGS iterations per cooperative-yield burst. After this many
/// iterations the solver hands control back to the async runtime so peer
/// thread-executor tasks (MPC outer loop, ESKF, GPS, baro, mag) can run.
///
/// Set to 1: one outer iteration of MINCO + cost eval is ~15–22 ms on
/// STM32H743 at the current `num_check_per_piece`. Yielding every
/// iteration keeps the uninterrupted CPU window bounded by a single iter.
const BFGS_ITERS_PER_YIELD: usize = 1;

/// Minimum wall-clock gap between BFGS bursts.
///
/// A bare `yield_now().await` only marks the task ready again immediately,
/// so the scheduler will re-poll mission_planner as soon as no other task
/// is actively runnable — which on STM32 means it usually resumes before
/// the outer_loop's 20 ms `Ticker` fires even once. The result is that
/// during Planning the outer_loop runs at ~20–30 Hz (one tick per BFGS
/// iter) instead of its designed 50 Hz, which causes the MPC's 1-iter
/// SQP warm-start to go stale and the drone to stutter / jump on the Z
/// axis while the planner is active.
///
/// Using `Timer::after(INTER_BURST_DELAY)` instead of `yield_now` forces
/// mission_planner to be *unready* for a fixed window, guaranteeing the
/// outer_loop's pending tick gets scheduled. 1 ms is long enough to admit
/// a 20 ms-period tick that was queued during the BFGS iter, and short
/// enough that the extra cost per solve is ~`max_iterations × 1 ms`
/// (≤500 ms, well inside SOLVE_BUDGET).
const INTER_BURST_DELAY: Duration = Duration::from_millis(1);

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

/// Result of one planning attempt — either a candidate trajectory ready
/// for publication, or a rejection with a populated `SolveDiagnostics`
/// that the outer task body forwards to the GCS as a breadcrumb.
///
/// Both [`plan_online`] and [`plan_offline`] return this so the outer
/// task body has one merge point for the abort/state-change re-checks
/// and the `MISSION_TRAJECTORY_SLOT` publish dance.
enum PlanOutcome {
    Ok {
        trajectory: PiecewisePolynomial,
        dur: f32,
        solve: msgs::SolveDiagnostics,
    },
    Reject {
        dur: f32,
        solve: msgs::SolveDiagnostics,
    },
}

/// Sample peak velocity on a finished trajectory. 200 uniform samples
/// is plenty to catch the peak of a smooth quintic spline.
fn peak_vel_m_s(traj: &PiecewisePolynomial, dur: f32) -> f32 {
    let mut v_max = 0.0f32;
    let n_samples = 200;
    for i in 0..=n_samples {
        let t = dur * i as f32 / n_samples as f32;
        let v = traj.get_vel(t);
        let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if v2 > v_max {
            v_max = v2;
        }
    }
    libm::sqrtf(v_max)
}

/// Online (BFGS) planning schema.
///
/// Builds hardcoded waypoints, runs the on-device BFGS optimizer in
/// cooperative bursts (one outer iter per yield), classifies the
/// outcome, and returns either an optimised trajectory or a
/// fully-populated `Reject`. All defmt logging, heartbeat publishing
/// during Planning, IS_ARMED interlock, and BFGS-specific reject
/// reasons (`TimeExceeded` → DISARMED/BUDGET_EXCEEDED,
/// UNDER_COMPRESSED) are owned here — the outer task body only sees
/// the `PlanOutcome`.
async fn plan_online(
    start_position: nalgebra::Vector3<f32>,
    workspace: &mut BfgsWorkspace,
    config: &mut QuadPlanningConfig,
    local_param_ver: &mut u32,
    mission_status_pub: &ImmediatePublisher<
        '_,
        CriticalSectionRawMutex,
        msgs::MissionStatus,
        2,
        2,
        1,
    >,
) -> PlanOutcome {
    // Refresh planner config if vehicle params changed since the last
    // solve. Only safe between solves (no in-flight BFGS state depends
    // on `config`).
    let cur = crate::params::PARAM_VERSION.load(Ordering::Acquire);
    if cur != *local_param_ver {
        *local_param_ver = cur;
        *config = QuadPlanningConfig::from_vehicle_params(&crate::params::get());
        defmt::info!("mission_planner: planner config reloaded (ver {})", cur);
    }

    let start_pos: [f32; 3] = [start_position.x, start_position.y, start_position.z];
    let start_vel: [f32; 3] = [0.0, 0.0, 0.0];

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
        [-1.845, 1.942, 1.0],
        [2.292, 1.637, 1.0],
        [2.547, -2.108, 1.8],
        [2.547, -2.108, 0.8],
    ];

    let input = PlannerInput::waypoints(start_pos, start_vel, &targets.map(Vec3::from));

    // Snapshot the pre-BFGS time allocation so the ground station can
    // compare it against the optimized total duration — a ratio ≈ 1
    // means BFGS made no progress.
    let n_pieces_input = input.num_waypoints + 1;
    let init_duration_s: f32 = input.init_times[..n_pieces_input].iter().sum();

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
    // The callback ALSO returns false when `IS_ARMED` clears. If
    // the pilot emergency-disarms during the solve, the RC parser
    // (on ctrl_spawner P10) detects the arm-switch edge and
    // signals `ARM_STATE`; DShot (P6) consumes it and sets
    // `IS_ARMED = false`. The solver sees that on its next outer
    // iteration (≤20 ms) and exits — so no stale trajectory ever
    // lands in `MISSION_TRAJECTORY_SLOT` from a disarmed attempt.
    //
    // Cooperative yielding: the solve runs in bursts of
    // `BFGS_ITERS_PER_YIELD` iterations. Between bursts we sleep
    // for `INTER_BURST_DELAY` so peer thread-executor tasks (MPC
    // outer loop, ESKF, GPS, baro, mag) get a scheduling slot that
    // mission_planner cannot immediately reclaim — without this
    // delay, the outer_loop runs at ~20–30 Hz during Planning and
    // the drone stutters vertically while BFGS solves.
    let t0 = Instant::now();
    let deadline = t0 + SOLVE_BUDGET;
    let mut keep_going =
        || Instant::now() < deadline && crate::motors::IS_ARMED.load(Ordering::Acquire);
    let mut session = plan_init(&input, config, workspace);
    let mut last_heartbeat = t0;
    let status = loop {
        match plan_resume(
            &mut session,
            config,
            workspace,
            &mut keep_going,
            BFGS_ITERS_PER_YIELD,
        ) {
            Some(s) => break s,
            None => {
                // Burst done but solver still running. Re-check the
                // abort conditions eagerly (cheap) and then sleep
                // for INTER_BURST_DELAY — a bare yield_now would let
                // mission_planner reclaim the CPU before the
                // outer_loop's 20 ms tick fires, leaving MPC running
                // at only ~25 Hz and causing visible motor stutter
                // during Planning.
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
                        solve: msgs::SolveDiagnostics::NONE,
                    });
                }
                Timer::after(INTER_BURST_DELAY).await;
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

    let mk_solve = |reason: u8, peak: f32| msgs::SolveDiagnostics {
        status: result.status as u8,
        iterations: result.iterations.min(u16::MAX as usize) as u16,
        solve_time_ms: elapsed_ms.min(u16::MAX as u64) as u16,
        num_pieces: result.num_pieces.min(u8::MAX as usize) as u8,
        init_duration_s,
        final_cost: result.final_cost,
        peak_vel_m_s: peak,
        reject_reason: reason,
    };

    // `TimeExceeded` is the graceful-abort path: the solver hit the
    // wall-clock budget OR `IS_ARMED` cleared. Distinguish in the log
    // so the pilot/defmt reader knows why; both yield a `Reject`.
    if result.status == SolverStatus::TimeExceeded {
        let reject_reason = if !crate::motors::IS_ARMED.load(Ordering::Acquire) {
            defmt::warn!(
                "mission_planner: disarm detected mid-solve — aborting (no trajectory)"
            );
            msgs::SOLVE_REJECT_DISARMED
        } else {
            defmt::warn!(
                "mission_planner: solve exceeded {}ms budget — aborting mission cleanly",
                SOLVE_BUDGET.as_millis()
            );
            msgs::SOLVE_REJECT_BUDGET_EXCEEDED
        };
        return PlanOutcome::Reject {
            dur: 0.0,
            solve: mk_solve(reject_reason, 0.0),
        };
    }

    // Convergence / Stop / MaxIterations are all "solver returned a
    // trajectory" outcomes; validate the numerical result. Reject
    // InvalidValue outright.
    let converged = matches!(
        result.status,
        SolverStatus::Convergence | SolverStatus::Stop | SolverStatus::MaxIterations
    );
    let dur = result.trajectory.total_duration();
    // "Under-compressed" guard: if the optimizer claims convergence but
    // the total duration is ≥ 90% of the pre-BFGS init allocation, it
    // didn't actually compress the trajectory.
    let compression_ok = dur < 0.9 * init_duration_s;
    let basic_valid = converged
        && dur.is_finite()
        && dur >= MIN_TRAJECTORY_DURATION_S
        && dur <= MAX_TRAJECTORY_DURATION_S
        && result.final_cost.is_finite();
    let valid = basic_valid && compression_ok;

    if !valid {
        let reject_reason = if !basic_valid {
            msgs::SOLVE_REJECT_INVALID
        } else {
            msgs::SOLVE_REJECT_UNDER_COMPRESSED
        };
        defmt::warn!(
            "mission_planner: rejecting trajectory (status={}, dur={}, init_dur={}, reason={})",
            result.status as u8,
            dur,
            init_duration_s,
            reject_reason,
        );
        return PlanOutcome::Reject {
            dur,
            solve: mk_solve(reject_reason, 0.0),
        };
    }

    let peak_v = peak_vel_m_s(&result.trajectory, dur);
    PlanOutcome::Ok {
        trajectory: result.trajectory,
        dur,
        solve: mk_solve(msgs::SOLVE_REJECT_NONE, peak_v),
    }
}

/// Offline (precomputed) planning schema.
///
/// Reads `OFFLINE_WAYPOINTS` + `OFFLINE_TIMESTAMPS` (60 pieces,
/// solved offline by a heavier planner), recovers per-segment
/// durations from consecutive timestamp differences, and feeds
/// the schedule directly into a [`MincoSnap`] solver. No BFGS. No
/// cooperative yielding — the banded LU completes in single-digit
/// milliseconds, well inside any IWDG budget.
///
/// Honors [`OFFLINE_USE_YAML_START`] for the head pose; the rest of
/// the trajectory (intermediate waypoints, tail) is unconditionally
/// the YAML schedule.
fn plan_offline(
    start_position: nalgebra::Vector3<f32>,
    minco: &mut MincoSnap,
) -> PlanOutcome {
    // Pull the active profile and verify its shape before any indexing.
    // A profile with mismatched waypoint/timestamp lengths, zero entries,
    // or more than `OFFLINE_MAX_PIECES` would crash on slice access or
    // overflow the stack-sized scratch arrays — reject up front so the
    // mission never enters the planning stage. Per-profile compile-time
    // const_assert!s catch the same conditions for in-tree data; this
    // guard backstops any future runtime-loaded source.
    let profile = offline_mission::active();
    let n = profile.waypoints.len();
    if n == 0 || n != profile.timestamps.len() || n > OFFLINE_MAX_PIECES {
        defmt::warn!(
            "mission_planner: profile '{}' invalid (wp={}, ts={}, max={}) — request dropped",
            profile.name,
            n,
            profile.timestamps.len(),
            OFFLINE_MAX_PIECES,
        );
        return PlanOutcome::Reject {
            dur: 0.0,
            solve: msgs::SolveDiagnostics {
                status: 0,
                iterations: 0,
                solve_time_ms: 0,
                num_pieces: n.min(u8::MAX as usize) as u8,
                init_duration_s: 0.0,
                final_cost: 0.0,
                peak_vel_m_s: 0.0,
                reject_reason: SOLVE_REJECT_PROFILE_INVALID,
            },
        };
    }

    // Recover per-segment durations from the absolute timestamp
    // schedule. `dur[0] = timestamps[0]` (segment running from t=0);
    // subsequent durations are consecutive differences. Reject on
    // non-monotonic / non-positive / non-finite — the offline
    // generator should never emit such a schedule, but if the YAML
    // were ever hand-edited we don't want NaN to propagate through
    // MINCO.
    let mut durations = [0.0f32; OFFLINE_MAX_PIECES];
    let mut prev_t = 0.0f32;
    for i in 0..n {
        let ts = profile.timestamps[i];
        let d = ts - prev_t;
        if !d.is_finite() || d <= 0.0 {
            defmt::warn!(
                "mission_planner: offline timestamps invalid at i={} (d={}) — request dropped",
                i,
                d,
            );
            return PlanOutcome::Reject {
                dur: 0.0,
                solve: msgs::SolveDiagnostics {
                    status: 0,
                    iterations: 0,
                    solve_time_ms: 0,
                    num_pieces: n as u8,
                    init_duration_s: profile.timestamps[n - 1],
                    final_cost: 0.0,
                    peak_vel_m_s: 0.0,
                    reject_reason: SOLVE_REJECT_INVALID_TIMES,
                },
            };
        }
        durations[i] = d;
        prev_t = ts;
    }
    let init_duration_s = profile.timestamps[n - 1];

    // Head pose: live setpoint by default, YAML start if the flag
    // says so. The trajectory's first segment duration is unchanged
    // either way.
    let head_pos: Vec3 = if OFFLINE_USE_YAML_START {
        Vec3::from(profile.start_pos)
    } else {
        start_position
    };

    // MINCO-snap contract: for `n` pieces, the solver needs `n−1`
    // intermediate waypoints and a tail boundary. The schedule
    // provides `n` waypoints; `wp[0..n−1]` are the intermediates and
    // `wp[n−1]` is the tail.
    let mut intermediate = [ZERO3; OFFLINE_MAX_PIECES - 1];
    for i in 0..n - 1 {
        intermediate[i] = Vec3::from(profile.waypoints[i]);
    }
    let tail_pos = Vec3::from(profile.waypoints[n - 1]);

    // PVAJ boundaries: zero v, a, j at both head and tail (the drone
    // is hovering when the mission triggers, and the offline schedule
    // returns to rest).
    let head: [Vec3; 4] = [head_pos, ZERO3, ZERO3, ZERO3];
    let tail: [Vec3; 4] = [tail_pos, ZERO3, ZERO3, ZERO3];

    defmt::info!(
        "mission_planner: offline solve (profile={}, head=[{},{},{}], n={}, total={}s)",
        profile.name,
        head_pos.x,
        head_pos.y,
        head_pos.z,
        n,
        init_duration_s
    );

    let t0 = Instant::now();
    minco.set_piece_count(n);
    minco.set_boundary(&head, &tail);
    minco.solve(&intermediate[..n - 1], &durations[..n]);
    let trajectory = minco.get_trajectory();
    let final_cost = minco.get_energy();
    let elapsed_ms = Instant::now().duration_since(t0).as_millis();

    let dur = trajectory.total_duration();
    let basic_valid = dur.is_finite()
        && dur >= MIN_TRAJECTORY_DURATION_S
        && dur <= MAX_TRAJECTORY_DURATION_S
        && final_cost.is_finite();

    let mk_solve = |reason: u8, peak: f32| msgs::SolveDiagnostics {
        // `status` field is BFGS-shaped; on the offline path we have
        // no solver status to report, so the closest analogue is
        // `Convergence` (the schedule is taken verbatim, by definition
        // optimal under the offline cost).
        status: SolverStatus::Convergence as u8,
        iterations: 0,
        solve_time_ms: elapsed_ms.min(u16::MAX as u64) as u16,
        num_pieces: n as u8,
        init_duration_s,
        final_cost,
        peak_vel_m_s: peak,
        reject_reason: reason,
    };

    if !basic_valid {
        defmt::warn!(
            "mission_planner: offline trajectory rejected (dur={}, cost={})",
            dur,
            final_cost
        );
        return PlanOutcome::Reject {
            dur,
            solve: mk_solve(msgs::SOLVE_REJECT_INVALID, 0.0),
        };
    }

    let peak_v = peak_vel_m_s(&trajectory, dur);
    defmt::info!(
        "mission_planner: offline solve done ({}ms, dur={}s, peak_v={}m/s)",
        elapsed_ms,
        dur,
        peak_v,
    );
    PlanOutcome::Ok {
        trajectory,
        dur,
        solve: mk_solve(msgs::SOLVE_REJECT_NONE, peak_v),
    }
}

#[embassy_executor::task]
pub async fn mission_planner_task() {
    let workspace: &mut BfgsWorkspace = WORKSPACE.init(BfgsWorkspace::new());
    // Pre-allocate the offline MINCO-snap solver at the registry-wide
    // upper bound. The active profile may need fewer pieces; `plan_offline`
    // rebinds the active extent in place via `set_piece_count` on every
    // request, and `set_boundary` rewrites the placeholder PVAJ.
    let offline_minco: &mut MincoSnap = OFFLINE_MINCO.init(MincoSnap::new(
        &[ZERO3, ZERO3, ZERO3, ZERO3],
        &[ZERO3, ZERO3, ZERO3, ZERO3],
        OFFLINE_MAX_PIECES,
    ));
    let mission_status_pub = MISSION_STATUS.immediate_publisher();
    // Build the planner config from the *live* vehicle params (mass,
    // inertia, motor thrust caps, planner tunables) rather than the
    // hardcoded `default()` — otherwise the trajectory is solved for a
    // 0.55 kg quad regardless of what the firmware is actually flying,
    // and the outer-loop's flatness map (which uses live mass via
    // `mpc_problem`) sees a physically inconsistent acceleration profile.
    // Refreshed per PLAN_REQUEST inside `plan_online`. Unused by the
    // offline path; left in scope for symmetry.
    let mut config = QuadPlanningConfig::from_vehicle_params(&crate::params::get());
    let mut local_param_ver = crate::params::PARAM_VERSION.load(Ordering::Acquire);

    defmt::info!(
        "mission_planner: ready (idle, schema={})",
        if USE_OFFLINE_PLAN { "offline" } else { "online" }
    );

    loop {
        // Block until the RC trigger fires.
        PLAN_REQUEST.wait().await;

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
        // rather than seeding the planner with junk.
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

        // Enter Planning — sticks are already gated by rc_interpreter from
        // the moment it sent the trigger, but make the state machine honest.
        MISSION_STATE.store(MissionState::Planning as u8, Ordering::Release);

        // Dispatch to the chosen planning schema. Both functions own
        // their solver-specific reject reasons; the outer body only
        // sees a `PlanOutcome`.
        let outcome = if USE_OFFLINE_PLAN {
            plan_offline(start_position, offline_minco)
        } else {
            plan_online(
                start_position,
                workspace,
                &mut config,
                &mut local_param_ver,
                &mission_status_pub,
            )
            .await
        };

        let (trajectory, dur, solve) = match outcome {
            PlanOutcome::Ok {
                trajectory,
                dur,
                solve,
            } => (trajectory, dur, solve),
            PlanOutcome::Reject { dur, solve } => {
                MISSION_STATE.store(MissionState::Idle as u8, Ordering::Release);
                mission_status_pub.publish_immediate(msgs::MissionStatus {
                    timestamp: Instant::now(),
                    state: MissionState::Idle as u8,
                    tau_s: 0.0,
                    total_duration_s: dur,
                    target_position: start_position,
                    solve,
                });
                continue;
            }
        };

        // If the pilot aborted during the solve, drop the result on the
        // floor — do NOT transition to Executing. The outer loop will
        // not pick up the abort flag because the state never leaves
        // Idle, so we clear the flag here ourselves.
        if MISSION_ABORT_REQUESTED.load(Ordering::Acquire) {
            defmt::info!("mission_planner: user aborted during solve, trajectory discarded");
            MISSION_ABORT_REQUESTED.store(false, Ordering::Release);
            MISSION_STATE.store(MissionState::Idle as u8, Ordering::Release);
            let mut s = solve;
            s.reject_reason = msgs::SOLVE_REJECT_USER_ABORT;
            mission_status_pub.publish_immediate(msgs::MissionStatus {
                timestamp: Instant::now(),
                state: MissionState::Idle as u8,
                tau_s: 0.0,
                total_duration_s: dur,
                target_position: start_position,
                solve: s,
            });
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
            let mut s = solve;
            s.reject_reason = msgs::SOLVE_REJECT_STATE_CHANGED;
            mission_status_pub.publish_immediate(msgs::MissionStatus {
                timestamp: Instant::now(),
                state: state_now as u8,
                tau_s: 0.0,
                total_duration_s: dur,
                target_position: start_position,
                solve: s,
            });
            continue;
        }

        // Publish & transition under the SAME critical section.
        //
        // Both the slot write and the `MISSION_STATE = Executing` flip
        // happen inside one `MISSION_TRAJECTORY_SLOT.lock` so that any
        // task which observes `Executing` and then takes the slot lock
        // is guaranteed to see the trajectory. failsafe.rs writes its
        // state flip inside the same lock for the symmetric reason.
        // Re-check abort + state INSIDE the publish lock to serialize
        // against any concurrent state writer (outer_loop's abort path,
        // failsafe). We `swap(false)` rather than `load` so this
        // consumes the intent symmetrically with the outer_loop abort
        // branch.
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
                traj: trajectory,
                t_start,
                total_duration_s: dur,
                solve,
            });
            MISSION_STATE.store(MissionState::Executing as u8, Ordering::Release);
            published = true;
        });
        if !published {
            // Critical: the publish was suppressed because either (a) abort
            // raced inside the lock and we consumed the flag via swap, or
            // (b) state was no longer Planning (e.g. failsafe). In case (a)
            // outer_loop's abort path is gated on `MISSION_ABORT_REQUESTED.swap`
            // returning true — since *we* already consumed the flag, that
            // gate will never fire and the state would otherwise stay
            // Planning forever (rc_interpreter's stick gate is `state != Idle`,
            // so the pilot would lose stick control entirely).
            //
            // Force state to Idle here. Idempotent in case (b).
            MISSION_STATE.store(MissionState::Idle as u8, Ordering::Release);
            defmt::warn!(
                "mission_planner: publish suppressed (abort/state changed during lock acquisition) — forced Idle"
            );
            let mut race_solve = solve;
            race_solve.reject_reason = msgs::SOLVE_REJECT_PUBLISH_RACE;
            mission_status_pub.publish_immediate(msgs::MissionStatus {
                timestamp: Instant::now(),
                state: MissionState::Idle as u8,
                tau_s: 0.0,
                total_duration_s: dur,
                target_position: start_position,
                solve: race_solve,
            });
            continue;
        }
        defmt::info!(
            "mission_planner: → Executing (dur={}s, peak_v={}m/s)",
            dur,
            solve.peak_vel_m_s,
        );
        // Publish the Executing state immediately so the ground station sees
        // state=2 without waiting up to 100 ms for the outer_loop's decimated
        // MissionStatus heartbeat.
        mission_status_pub.publish_immediate(msgs::MissionStatus {
            timestamp: t_start,
            state: MissionState::Executing as u8,
            tau_s: 0.0,
            total_duration_s: dur,
            target_position: start_position,
            solve,
        });
    }
}
