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

Silence is reserved for output that **can no longer be trusted**. When a
task can still produce a valid but degraded output, it degrades instead
(loudly, via defmt) — going silent would trade a flyable degraded
controller for a guaranteed disarm-in-flight. Rule 9 lists the
sanctioned degradations and where the line sits.

## The Control Chain

```
VEHICLE_ODOMETRY (ESKF)
  → outer loop ──────────────→ RATE_COMMAND
       one of:                     │
       · outer_rate      (sticks → body rate + thrust)
       · outer_geometric (cascade PD + geometric, 100 Hz)
       · outer_mpc       (SQP MPC, 50 Hz; `outer_mpc_full` variant)
                                   ▼
                              INDI task (IMU rate)
                                   │
                                   ▼
                            ACTUATOR_MOTORS → DShot → motors
```

Every outer-loop family publishes the same `RATE_COMMAND` channel and is
covered by the same staleness gates below. `outer_mpc_full` additionally
carries a body-torque setpoint τ_d in the same message (INDI re-derives
the α pseudo-control from it with fresh gyro every IMU tick), and
`build: indi: no` degrades the inner loop to model-based static
inversion — **neither variant changes any threshold or silence rule in
this document**.

## Rules

### 1. Never directly disarm from a controller

Controllers (INDI, outer loops) must NOT write to `ARM_STATE`,
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

Corollary: code that can panic (filter builders, config constructors)
must not be reachable from a hot loop with parameter-derived input.
Structural guards degrade to schema defaults instead of panicking the
inner loop into a flash-persistent boot-panic cycle.

### 3. Every consumer must check input recency

Before using any input, check that it is recent enough to be useful.
If stale, skip publishing for that frame. Current gates:

| Consumer | Input | Stale threshold | Action when stale |
|----------|-------|-----------------|-------------------|
| Outer loop (`outer_mpc`) | `VEHICLE_ODOMETRY` | 50 ms | Skip tick (no `RATE_COMMAND`) |
| Outer loop (`outer_geometric`) | `VEHICLE_ODOMETRY` | 100 ms | Skip tick (no `RATE_COMMAND`) |
| INDI task (armed) | `RATE_COMMAND` | 100 ms | Stop publishing `ACTUATOR_MOTORS` |
| INDI task, `table` thrust model (armed) | `POWER_STATUS` voltage | hold 0.5 s; silent at 2 s | Stop publishing `ACTUATOR_MOTORS` |
| DShot task | `ACTUATOR_MOTORS` | 10 ms | Drop to idle throttle |
| Failsafe | `LAST_CONTROLLER_PUBLISH` | 500 ms | Disarm |
| Failsafe | `RC_INPUT` | 150 ms | Enter guard period |

The MPC's odometry gate is tighter than the cascade's because stale
`state_pos` also corrupts the position sampler's closest-point search
(see `outer_loop::ODOM_STALE_TIMEOUT` for the derivation). Validity is
part of recency: each consumer's `odom_is_valid` must reject non-finite
components of **every field that build actually consumes** — e.g. the
full-model MPC checks `twist.angular` because body rates are states in
its `x0`; the reduced model deliberately does not, so a frame it can fly
on is never rejected for a component it ignores.

The thresholds must be ordered so that upstream silence is always
detected before the downstream timeout expires:

```
outer odom stale (50 ms MPC / 100 ms cascade)
  < INDI RATE_COMMAND stale (100 ms)
    < failsafe controller watchdog (500 ms)

DShot motor cmd stale (10 ms)
  < failsafe controller watchdog (500 ms)

RC frame timeout (150 ms)
  < RC guard period (1500 ms)
```

And in the other direction: a **healthy producer must never look
stale**. The outer tick-rate floor (25 Hz, both loop families) keeps
≥ 2.5 outer periods inside INDI's 100 ms command gate, so one or two
missed solves cannot trip the failsafe in flight.

### 4. Only update the watchdog heartbeat on valid publish

`LAST_CONTROLLER_PUBLISH` must only be written when a valid motor
command is actually published. Any code path that skips publishing
(NaN output, stale input, sustained-NaN fallback, etc.) must NOT update
the heartbeat. This ensures the failsafe accurately detects controller
silence.

Pre-launch nuance (position-mode builds): while `armed && !LAUNCHED`,
INDI bypasses the controller stack and publishes a uniform idle
throttle. This **does** stamp the heartbeat (DShot's 10 ms watchdog
must not force-idle the pre-launch spin) but deliberately does NOT
refresh INDI's own `RATE_COMMAND` freshness clock: if the outer loop
was silent during pre-launch, the first launched tick sees a stale
command and INDI goes silent → watchdog disarm — instead of flying up
to 100 ms on a stale cached command at the most critical moment.

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
exists. The current startup chain for `est_eskf` position-mode builds:

```
ESKF waits for first VICON_POSE          (hardware dependency, by design)
  → ESKF converges → ESTIMATOR_READY = true
    → rc_interpreter unblocks (polls ESTIMATOR_READY), waits for a
      finite odometry sample, captures origin
      → seeds ACTIVE_POSITION_SETPOINT, fires ACTIVE_SETPOINT_READY
        → outer loop unblocks (awaits ACTIVE_SETPOINT_READY once, then
          polls ESTIMATOR_READY) → enters its tick loop
          → publishes RATE_COMMAND
```

INDI is deliberately **outside** this chain: it starts at boot, runs
from the first IMU sample, and blocks on nothing — the first fresh
`RATE_COMMAND` activates motor output.

Rules for startup:

- **No task may block on a channel that another blocked task produces.**
  This creates a deadlock. The startup chain must be a DAG.
- **Convergence/readiness flags are polled, not awaited.** Use
  `Timer::after_millis(100)` loops, not blocking `.wait()` on readiness
  signals, to avoid starving the executor. (One-shot handshake signals
  like `ACTIVE_SETPOINT_READY`, fired exactly once by a task that is
  itself unblocked, are fine to await.)
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

All thresholds below are **parameters**, not literals — the values given
are the current schema defaults. See `docs/parameters.md`.

1. `FAILSAFE_ACTIVE` is false.
2. Throttle channel below `rc_throttle_mincheck_us` (1050 µs).
3. RC link quality ≥ `arm_min_link_quality` (50%), frame received within
   the link-staleness window.
4. `ESTIMATOR_READY` is true (`est_eskf` mode only).
5. Arm switch held for ≥ `arm_switch_hold_s` (100 ms) debounce.
6. Tilt within `arm_max_tilt_deg`.
7. ESKF/Mahony attitude agreement within `arm_eskf_mahony_tol_deg`.
8. Accel magnitude within `arm_accel_tol_m_s2` of gravity, and gyro rates
   below `arm_gyro_limit_rad_s` (the `ATTITUDE_HEALTH` bits).
9. No asserted ESKF fault bits (`BlockReason::EskfDegraded`).

Disarm is always immediate (no debounce, no gates).

No controller task, estimation task, or sensor task may add implicit
arming prerequisites by blocking or failing to publish. If a task is
not ready, it simply does not publish — the system remains armable as
long as the explicit gates above pass. This is also why controller-side
silence decisions (stale command, sustained WLS NaN, Table-mode voltage
loss) are **armed-gated**: while disarmed, the heartbeat keeps running
so a bench vehicle stays armable, and the published output sits
harmlessly behind DShot's armed gate.

### 8. Failsafe state machine

Two independent monitors run in the failsafe task:

**RC link failsafe** (two-stage, Betaflight-inspired):

| Phase | Entry condition | Duration | Action |
|-------|-----------------|----------|--------|
| Idle | Normal operation | — | — |
| Guard | No RC frame for 150 ms | 1500 ms | Wait for recovery |
| Landed | Guard expired | Until 500 ms valid RC | Disarm, set `FAILSAFE_ACTIVE` |

Stage durations are the params `fs_rxloss_trigger_s`, `fs_guard_period_s`
and `fs_ctrl_timeout_s`; the table shows current defaults.

**Controller watchdog** (all builds):

- If `LAST_CONTROLLER_PUBLISH` is older than `fs_ctrl_timeout_s` (500 ms),
  disarm immediately.
- If `LAST_CONTROLLER_PUBLISH` is `None` (controller hasn't started), do
  nothing — this is not a failure.

Recovery: 500 ms of continuous valid RC frames clears `FAILSAFE_ACTIVE`
and returns to Idle.

### 9. Graceful degradation must be explicit

The counterpart to the core principle: silence is for output that can
no longer be trusted; a **valid but degraded** output keeps publishing.
Every sanctioned degradation is explicit in code and logged once per
episode (edge-triggered, never per-tick at 8 kHz):

- **All-motor RPM telemetry stale** → INDI switches to the du-based
  internal motor model (`MotorState::Internal`) and drops G2. Losing
  RPM feedback degrades disturbance rejection; it does not make the
  vehicle uncontrollable, so going silent here would be strictly worse.
- **Transient WLS NaN** (< `nan_limit` consecutive ticks) → a finite
  decayed hold of the last actuator state bridges the glitch. Past
  `nan_limit` the hold is no longer a controller decision → silence
  (see the failure example below).
- **Battery voltage stale** (`table` thrust model) → hold the last
  reading for up to 2 s (sag is slow against that window); beyond it
  the linearization is unreliable enough that flying on is more
  dangerous than the disarm → silence.
- **RPM notch banks** fade to passthrough when a motor frequency is
  invalid or below band — they never inject NaN or stale notches into
  the IMU path, and they leave the silence signals of the surrounding
  stages untouched.

The dividing line: **degrade** when the fallback is a model of the
missing *input* with bounded error; **go silent** when the *output*
itself is no longer a controller decision.

### 10. Airframe identity is applied atomically at boot

Reboot-flagged identity params — mass, the inertia tensor, motor
geometry/max-thrust/spin, site gravity — take effect **only at boot,
everywhere at once**. Multiple coupled consumers capture them at task
start: INDI's geometric G1 (thrust rows `t/m`, torque rows `I⁻¹τ`), the
`outer_mpc_full` α-path inertia, the RPM KF, and the MPC model.

The disarmed param hot-reload applies **live-tunable groups only**
(`mpc`, `trajectory`, INDI gains/filters, motor tau/omega/G2/nonlin —
exactly the fields the schema leaves un-flagged). Reload paths that
rebuild a model from a fresh `params::get()` must pin the boot snapshot
of the reboot-flagged groups (see `outer_loop`'s hot-reload). Applying
identity early in one consumer but not another puts the outer and inner
loop on different vehicles — e.g. an MPC commanding thrust scaled for a
mass INDI does not have. The shell already tells the operator
"(reboot required)"; the code must make that promise true.

## Failure Propagation Examples

### VICON dropout

The mocap guard absorbs short dropouts by design — mocap arrives at
100–360 Hz, so a gap of O(100 ms) is well inside the IMU's drift budget.

```
1. ESKF stops receiving mocap updates
2. ESKF continues predicting (IMU only); odometry and attitude keep
   publishing on the IMU-propagated state
3. eskf_mocap_stale_s (0.1 s) elapses → EskfMocapGuard clears `converged`
   → ESTIMATOR_READY=false (arming blocked) + annunciation.
   Publishing deliberately CONTINUES here: withholding odometry starves
   the outer loop's 50 ms freshness gate, which cascades to INDI's
   CMD_STALE_TIMEOUT and then the 10 ms DShot watchdog — i.e. motors to
   idle ~110 ms into a routine marker occlusion.
4. Stream recovers → poses accepted, converged re-earned, ESTIMATOR_READY
   returns. No control discontinuity.
```

If the outage is *sustained* rather than transient:

```
5. eskf_fault_pos_timeout_s elapses → POS_STALE asserts in ESKF_FAULTS
   (and ESKF_SEVERE_FAULT while armed) → annunciated to the GCS and the
   blackbox. This does NOT auto-disarm: attitude control is still good,
   and on a mocap vehicle there is no secondary position source, so the
   pilot retains authority.
6. Only if the controller itself goes silent for fs_ctrl_timeout_s does
   the watchdog disarm.
```

If the filter's position has meanwhile separated from truth by more than
`eskf_max_pos_jump_m`, every returning pose is jump-rejected. **Armed**
that persists (a jump usually means rigid-body re-association, where IMU
dead-reckoning is the more trustworthy source). **Disarmed**, the guard
re-anchors once the stream agrees with itself within
`eskf_mocap_reanchor_m` over `REANCHOR_FRAMES`, so the wedge is not a
power-cycle.

### WLS produces NaN

```
1. WLS exits with NaN → INDI substitutes a finite decayed hold
   (u_state · 0.95 per tick), bridging the glitch without a control
   gap; publishing and the heartbeat continue (rule 9)
2. If transient: solver recovers within nan_limit (20) consecutive
   ticks → counter resets — a ~2 ms blip, no failsafe involvement
3. If sustained: nan_failsafe asserts while armed → INDI stops
   publishing → DShot idle (10 ms) → failsafe disarm (500 ms)
```

A NaN that reaches the motor commands themselves (e.g. through the
thrust linearization) skips that frame's publish directly — same
terminal cascade if it persists.

### RC link lost

```
1. CRSF task stops publishing RC_INPUT
2. Failsafe detects 150 ms silence → guard period
3. No recovery within 1500 ms → disarm, FAILSAFE_ACTIVE = true
4. rc_interpreter blocks on its RC subscription —
   ACTIVE_POSITION_SETPOINT simply stops being updated (its last value
   stays valid)
5. The outer loop keeps tracking the last setpoint (position hold) —
   this is safe; the failsafe disarm handles the situation
```

### ESKF divergence

```
1. ESKF publishes bad odometry (NaN state, non-finite components)
2. The outer loop's odom_is_valid() + freshness gate reject the frames
   → it skips ticks and RATE_COMMAND goes stale
3. INDI's 100 ms command gate trips → stops publishing → DShot idle
   (10 ms) → failsafe disarm (500 ms)
```
