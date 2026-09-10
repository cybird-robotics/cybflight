# Mission Planner Safety Gap Analysis — Cybflight vs. ArduPilot

> **STATUS: HISTORICAL (as of 2026-08-07).** This is a point-in-time
> research deliverable, kept for the record. It predates the
> configuration-plane refactor, the baked mission tables, and both ESKF
> guards, so **every file:line reference below is stale** and several
> listed gaps are closed. Do not use it as a current gap list; see
> `docs/safety_protocol.md` and `docs/parameters.md`.
>
> Specifically superseded:
> - **Gap 2 (no VICON link-quality gate)** — closed. `EskfMocapGuard`
>   (`crates/cybflight_core/src/eskf/mocap_guard.rs`) implements jump and
>   reject cascades plus a staleness window (`eskf_mocap_stale_s`,
>   default 0.1 s) that drops `ESTIMATOR_READY`, and a disarmed-only
>   re-anchor hatch. It no longer waits for the controller watchdog.
> - **Gap 3 (no geofence)** — partly closed for stick input:
>   `fence_enable` / `fence_x_m` / `fence_y_m` / `fence_z_max_m` /
>   `fence_z_min_m` are params. Mission waypoints are still unclamped, so
>   the mission-planner half of this gap stands.
> - **"Hardcoded circular waypoints"** — obsolete. Missions are
>   `missions/*.yaml`, baked by `build.rs`.
> - **"Stick Z clamped to 5 m"** — obsolete; now `fence_z_max_m`
>   (default 2.0).
> - **"100 Hz outer MPC"** — obsolete; `mpc_rate_hz` defaults to 50.
> - **Gap 13 (no ADC wired)** — obsolete; `BatteryParams` and
>   `sensors/power.rs` exist.
> - **Gap 14 ("cybflight intentionally mocap-only")** — scope has
>   expanded: `pos_source` is `mocap` or `gps` per vehicle, and most
>   vehicles in `vehicles/` are now GPS.

## Context

The user asked for a comparison between cybflight's current mission planner workflow
and ArduPilot's, aimed at surfacing **safety mechanisms ArduPilot treats as essential
that cybflight does not yet implement**. This is a research/analysis deliverable — no
code changes are being proposed yet. The output is a gap list that can later be used
to prioritize safety work.

Cybflight currently targets an indoor VICON-mocap environment. Many ArduPilot
safeguards are designed for outdoor autonomous flight and may not all apply, but the
exercise is still useful because several gaps (EKF divergence, motor saturation,
geofence, pre-arm health) matter even indoors.

## Current Cybflight Mission Workflow (baseline)

- **State machine** (`control/mission_planner.rs:159-359`, `control/mod.rs:94-110`):
  `Idle → Planning → Executing → Idle`, triggered by an RC AUX channel.
- **Trajectory**: hardcoded circular waypoints, min-jerk MINCO via BFGS trust-region;
  250 ms solve budget; duration must fall in [0.5, 30.0] s; validated for finite cost
  and solver status before publish.
- **Execution**: 100 Hz outer MPC samples the published polynomial; 8 kHz INDI inner
  loop runs off IMU + VEHICLE_ODOMETRY.
- **Existing safety** (cybflight already has these):
  1. IWDG watchdog (500 ms, thread executor only).
  2. Two-stage RC failsafe (`control/failsafe.rs`): 150 ms frame loss → 1.5 s guard → disarm.
  3. Controller watchdog (500 ms) detecting stale control publishes.
  4. Arming gates (`sensors/rc.rs:40-150`): throttle-low, RC link quality ≥50%, arm
     switch debounce, estimator-ready (one-shot ESKF convergence).
  5. Odometry staleness + NaN/future-timestamp rejection in INDI, MPC and planner.
  6. Trajectory validation after solve.
  7. DShot idle fallback if motor command stalls 10 ms while armed.
  8. Solver polls `IS_ARMED` during BFGS iterations → pilot disarm aborts solve cleanly.

## Safety Gaps vs. ArduPilot (ranked by relevance to cybflight's current mission)

### Tier 1 — Flight-critical even in the current indoor VICON use case

1. **No continuous EKF/ESKF health monitoring.**
   `ESTIMATOR_READY` latches true once at startup (`estimation/mod.rs:16-59`). ArduPilot
   runs EKF variance checks every tick (attitude, velocity, position, compass, height)
   and triggers EKF failsafe on divergence. Cybflight will only notice via the reactive
   100 ms odometry-stale / NaN gate. If ESKF diverges but still publishes finite values,
   the vehicle will fly the bad estimate straight into a wall.

2. **No VICON/mocap link-quality gate.**
   VICON dropout sends ESKF into prediction-only drift. ArduPilot's equivalent (GPS
   glitch / GPS failsafe) explicitly detects jumps and stalls. Cybflight only disarms
   after the 500 ms controller-watchdog fires — a long time at 5 m altitude.

3. **No geofence / altitude ceiling for trajectories.**
   Stick-commanded Z is clamped to 5 m (`rc_interpreter.rs:121`) but the **mission
   planner has no boundary or ceiling enforcement**. A hardcoded waypoint change or
   planner bug could plan a trajectory outside the capture volume. ArduPilot has
   cylindrical, polygon, and altitude fences with configurable actions (RTL, LAND, HOLD).

4. **No pre-arm sensor health checks beyond "first ESKF convergence".**
   ArduPilot blocks arming on: accel/gyro inconsistency between IMUs, baro sanity,
   compass field strength, mag variance, vibration level, EKF innovation gates,
   throttle-not-at-min-plus-deadband, parameter bounds. Cybflight has **dual IMUs on
   some boards but no cross-check** and no baro sanity gate pre-arm.

5. **No motor saturation / ESC-telemetry health check.**
   INDI does not explicitly detect persistent motor saturation (one motor pinned at
   max → loss of control authority). DShot telemetry is consumed but not bounds-checked
   for per-motor RPM sanity, temperature, or current. ArduPilot has "thrust loss
   detection", "yaw imbalance detection", and (optional) single-motor-failure mitigation.

6. **No crash / tumble detection.**
   ArduPilot's crash_check disarms on sustained large attitude error + low climb rate.
   Cybflight will keep commanding motors into the ground until the controller watchdog
   fires or the pilot disarms. Important even indoors.

7. **No vibration failsafe.**
   ArduPilot monitors accel clipping / variance and switches to alt-hold fallback.
   Cybflight has no such monitor; a loose mount could silently corrupt the ESKF.

### Tier 2 — Mission/planner-specific

8. **Trajectory feasibility is enforced by soft penalties only.**
   `MaxIterations` is accepted as a valid solver terminal status
   (`mission_planner.rs:265-313`). A converging-but-infeasible trajectory (tilt > 60°,
   thrust > 1.0, velocity > limit) could be published. Recommend a post-solve hard
   feasibility gate sampling the polynomial at fine resolution and checking kinematic
   bounds explicitly.

9. **No yaw-rate / jerk limits on emitted reference.**
   Planner comment notes yaw ≡ 0 assumption for differential flatness. If any future
   trajectory has non-trivial yaw, there is no rate cap. Jerk (3rd derivative of the
   MINCO polynomial) is unbounded at segment boundaries.

10. **No "home" concept and no return-to-home on abort.**
    Mission abort resets to `Idle` → pilot is expected to hand-fly out. ArduPilot's
    RTL is the default failsafe action. For cybflight a "hover-in-place" fallback
    trajectory (constant position at current state) when aborting mid-mission would be
    a strict improvement over abrupt INDI-timeout disarm.

11. **No graceful landing on disarm-during-mission.**
    Pilot disarm in `Executing` → INDI times out in 500 ms → motors cut. For indoor
    flight a 500 ms free-fall from 5 m is ~1.2 m — survivable but avoidable with a
    "commanded descent" abort path.

12. **No planner-time vs MPC-time anchor check.**
    Trajectory is published with an implicit t_start. If the outer loop's clock drifts
    or a resample occurs post-disarm cycle, no explicit consistency check exists.

### Tier 3 — Outdoor/long-flight gaps (lower priority for current scope)

13. No battery voltage / current monitoring or low-battery failsafe — no ADC wired.
14. No GPS fallback (cybflight intentionally mocap-only; flag if scope expands).
15. No GCS-link failsafe / telemetry timeout (no GCS present).
16. No logging / SD-card pre-arm check.
17. No parachute, no terrain-following, no deadreckoning mode.
18. No hardware safety switch (many H743 boards expose one; currently unused).

## Critical Files (for follow-up implementation work)

- `crates/cybflight/src/control/mission_planner.rs` — state machine, trajectory validation
- `crates/cybflight/src/control/failsafe.rs` — add EKF + vibration + crash monitors here
- `crates/cybflight/src/control/mod.rs` — shared state flags (extend with health bits)
- `crates/cybflight/src/sensors/rc.rs:40-150` — arming gate (extend pre-arm check list)
- `crates/cybflight/src/estimation/mod.rs` — add continuous covariance / innovation monitor
- `crates/cybflight/src/control/indi_task.rs` — motor saturation detection
- `crates/cybflight/src/control/outer_loop.rs` — reference clamp (add geofence clip)
- `crates/cybflight_core/src/trajectory_planning/planner.rs` — post-solve feasibility gate

## Recommended Next Steps (if the user wants to close gaps)

Prioritize in this order, gated by RC-testable indoor risk:

1. **Continuous ESKF health monitor** (covariance + innovation gate) → failsafe trigger.
2. **Pre-arm sensor cross-check** (dual-IMU gyro/accel agreement, baro sanity).
3. **Hard trajectory feasibility gate** post-solve (sample polynomial, enforce kinematic bounds).
4. **Altitude ceiling + cylindrical geofence** applied both to stick reference and planner.
5. **Crash / tumble detection** in `failsafe.rs`.
6. **Hover-in-place abort trajectory** for mission abort and controlled-descent landing
   for disarm-during-mission.
7. **Motor saturation monitor** in INDI, + DShot telemetry bounds.
8. **Vibration monitor** on raw IMU.

## Verification Approach (for when changes land)

- Unit-test each monitor's state machine with mocked inputs (stale, NaN, diverged).
- Hardware-in-loop bench test: disconnect VICON mid-flight, verify failsafe fires.
- Bench test: spoof one ESKF variance channel beyond threshold, verify EKF failsafe.
- Indoor flight test: arm with degraded IMU (loose mount) → pre-arm should block.
- Regression: verify existing RC failsafe, controller watchdog, IWDG still trigger.
