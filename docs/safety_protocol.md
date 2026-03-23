# Safety Protocol

This document defines the safety principles for the cybflight autopilot.
All tasks that produce or consume control data MUST follow these rules.
Update this document when the design changes.

## Core Principle: Silence Is the Universal Failure Signal

When a task cannot produce valid output — because its inputs are stale,
corrupt, non-finite, or missing — it **stops publishing**. It does not
crash, exit, or directly command a disarm. Downstream consumers detect
the silence and act accordingly.

This is the same pattern used by the RC radio: when the link drops, the
CRSF/GHST driver simply stops publishing `RC_INPUT`. The failsafe task
detects the silence and disarms. Every layer of the autopilot follows
this pattern.

```
Producer goes silent
  → Consumer detects stale input (recency check)
    → Consumer also goes silent
      → ... propagates to DShot + failsafe
        → DShot drops to idle (10 ms)
        → Failsafe disarms (500 ms)
```

## Rules

### 1. Never directly disarm from a controller

Controllers (INDI, position, attitude) must NOT write to `ARM_STATE`,
`FAILSAFE_ACTIVE`, or `IS_ARMED`. Arming and disarming is the sole
responsibility of:

- **RC receiver task** — user-initiated arm/disarm via switch.
- **Failsafe task** — automated disarm on RC loss or controller silence.

If a controller detects an unrecoverable condition, the correct response
is to stop publishing. The failsafe watchdog will detect the silence.

### 2. Never exit a task loop

Embassy tasks cannot be restarted after returning. A controller that
`break`s out of its loop is permanently dead — the failsafe will disarm,
but the system can never re-arm without a reboot. Instead:

- **Skip the frame** (`continue`) when output is invalid.
- The task stays alive and will resume publishing when inputs recover.
- The watchdog heartbeat naturally stops during silence, which is the
  intended trigger for the failsafe.

### 3. Every consumer must check input recency

Before using any input, check that it is recent enough to be useful.
If stale, skip publishing for that frame. Examples:

| Consumer | Input | Stale threshold | Action when stale |
|----------|-------|-----------------|-------------------|
| INDI task | `VEHICLE_ODOMETRY` | 100 ms | Stop publishing `ACTUATOR_MOTORS` |
| DShot task | `ACTUATOR_MOTORS` | 10 ms | Drop to idle throttle |
| Failsafe | `LAST_CONTROLLER_PUBLISH` | 500 ms | Disarm |
| Failsafe | `RC_INPUT` | 150 ms | Enter guard period |

The thresholds must be ordered so that upstream silence is always
detected before the downstream timeout expires:

```
INDI odom stale (100 ms)
  < failsafe controller watchdog (500 ms)

DShot motor cmd stale (10 ms)
  < failsafe controller watchdog (500 ms)

RC frame timeout (150 ms)
  < RC guard period (1500 ms)
```

### 4. Only update the watchdog heartbeat on valid publish

`LAST_CONTROLLER_PUBLISH` must only be written when a valid motor
command is actually published. Any code path that skips publishing
(NaN output, stale input, etc.) must NOT update the heartbeat. This
ensures the failsafe accurately detects controller silence.

### 5. DShot is the last line of defense before the motors

The DShot task enforces two safety properties:

- **Armed gate**: Motors only receive non-idle throttle when `IS_ARMED`
  is true. This is checked every frame (~8 kHz).
- **Stale command fallback**: If no `ACTUATOR_MOTORS` update arrives
  within `MOTOR_CMD_STALE` (10 ms) while armed, DShot drops to idle
  throttle. This prevents holding stale thrust during the gap between
  controller silence and failsafe disarm.

On the disarm transition, DShot sends `MOTOR_STOP` (command 0) for one
frame, then reverts to `DSHOT_MIN_THROTTLE`.

### 6. Startup must not create deadlocks or block arming

Tasks that block during startup (`Signal::wait`, channel subscribe,
convergence polling) must be ordered so that no circular dependency
exists. The current startup chain for `est_eskf` mode:

```
ESKF waits for first VICON_POSE         (hardware dependency, by design)
  → ESKF converges → ESTIMATOR_READY = true
    → RC interpreter unblocks, captures origin
      → publishes AUTO_SETPOINT
        → INDI unblocks, polls ESTIMATOR_READY (already true)
          → INDI enters main loop
```

Rules for startup:

- **No task may block on a channel that another blocked task produces.**
  This creates a deadlock. The startup chain must be a DAG.
- **Convergence/readiness flags are polled, not awaited.** Use
  `Timer::after_millis(100)` loops, not blocking `.wait()` on readiness
  signals, to avoid starving the executor.
- **The arming state machine must not require a running controller.**
  Arming gates check `ESTIMATOR_READY` (set by ESKF, independent of
  the controller). The controller starting is a consequence of arming
  prerequisites being met, not a prerequisite itself.
- **Startup waits must not consume inputs needed later.** For example,
  `odom_sub.next_message_pure().await` after ESTIMATOR_READY returns
  fresh (converged) data, not stale buffered messages from before
  convergence.

### 7. Arming gates

All of the following must be true to arm (checked by the RC receiver
task on every RC frame):

1. `FAILSAFE_ACTIVE` is false.
2. Throttle channel < 1050 µs (stick low).
3. RC link quality ≥ 50%, frame received within 500 ms.
4. `ESTIMATOR_READY` is true (`est_eskf` mode only).
5. Arm switch held for ≥ 100 ms (debounce).

Disarm is always immediate (no debounce, no gates).

No controller task, estimation task, or sensor task may add implicit
arming prerequisites by blocking or failing to publish. If a task is
not ready, it simply does not publish — the system remains armable as
long as the five explicit gates pass.

### 8. Failsafe state machine

Two independent monitors run in the failsafe task:

**RC link failsafe** (two-stage, Betaflight-inspired):

| Phase | Entry condition | Duration | Action |
|-------|-----------------|----------|--------|
| Idle | Normal operation | — | — |
| Guard | No RC frame for 150 ms | 1500 ms | Wait for recovery |
| Landed | Guard expired | Until 500 ms valid RC | Disarm, set `FAILSAFE_ACTIVE` |

**Controller watchdog** (`est_eskf` only):

- If `LAST_CONTROLLER_PUBLISH` is older than 500 ms, disarm immediately.
- If `LAST_CONTROLLER_PUBLISH` is `None` (controller hasn't started), do
  nothing — this is not a failure.

Recovery: 500 ms of continuous valid RC frames clears `FAILSAFE_ACTIVE`
and returns to Idle.

## Failure Propagation Examples

### VICON dropout

```
1. ESKF stops receiving mocap updates
2. ESKF continues predicting (IMU only), odometry drifts
3. INDI sees odom age > 100 ms → stops publishing ACTUATOR_MOTORS
4. DShot sees no motor cmd for 10 ms → drops to idle throttle
5. Failsafe sees LAST_CONTROLLER_PUBLISH age > 500 ms → disarm
```

### WLS produces NaN

```
1. INDI output contains NaN → skip publish (continue)
2. If transient: next frame is finite → resumes publishing, no effect
3. If sustained: watchdog heartbeat stops → DShot idle (10 ms) → failsafe disarm (500 ms)
```

### RC link lost

```
1. CRSF task stops publishing RC_INPUT
2. Failsafe detects 150 ms silence → guard period
3. No recovery within 1500 ms → disarm, FAILSAFE_ACTIVE = true
4. RC interpreter stops publishing AUTO_SETPOINT (no RC frames to map)
5. INDI continues with last setpoint (position hold) — this is safe,
   the failsafe disarm handles the situation
```

### ESKF divergence

```
1. ESKF publishes bad odometry (large covariance, NaN state)
2. INDI odom_is_valid() rejects NaN frames → odom_fresh stays false
3. INDI stops publishing → DShot idle → failsafe disarm
```
