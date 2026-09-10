# Why mpc_full + INDI Crashes: the Zero-Thrust Bang-Bang Limit Cycle

*2026-08-13 · Instrumented diagnosis of the mpc_full+INDI mission
failures from the tilt-free campaign
([mission_campaign_notilt.md](mission_campaign_notilt.md)). Per-solve
solver telemetry (`FullSolveDiag`: cost / converged / diverged /
input-clamp count / held (T_d, τ_d)) + 100 Hz state timelines on the
failing missions, plus two causal experiments. Harness:
`crates/cybflight_sim/tests/mpc_full_failure_diag.rs`.*

## Headline

**The solver never diverges. Every mpc_full crash is a ground impact
caused by a collective-thrust bang-bang limit cycle: once position error
exceeds ~0.5 m under a high-tilt reference, the solver alternates the
held thrust setpoint between 0 N and 34 N (full span). Each zero-thrust
interval idles all four rotors, which physically removes INDI's torque
authority — attitude tumbles in free fall, thrust returns too late
(rotor spool-up lag), altitude ratchets down, ground.** The entry into
the error regime differs per mission class (S1 reference gap, thrust
ceiling, infeasible demand), but the crash mode itself is this cycle.

## What it is NOT

Ruled out by the telemetry across all failing runs:

- **Solver divergence / NaN**: `diverged` fired 0 times; no non-finite
  solutions were ever held (`held_nonfinite` = 0). The recent KKT
  robustness work is not implicated.
- **Convergence starvation as the trigger**: RTI mode reports
  ~100 % non-converged on the *healthy* baseline too (1 iteration never
  meets `kkt_tol`); raising `max_iters` 1 → 5 barely moves any outcome
  (large_mid 0.72 → 0.80 rms, still crashes; splits_mid 1.28 → 1.05,
  still crashes). Iteration depth is not the mechanism.
- **Post-solve input clamping**: the u₀ clamp moved 0 inputs at every
  failure point — the solver's own soft bounds were already active; the
  bang-bang values are the solver's *chosen* optimum, not an artifact.
- **INDI ground-mode false trigger**: the takeoff detector requires
  `accel_high` (‖f‖ > 0.8 g); in free fall the accelerometer reads ≈ 0,
  so INDI stays active throughout. The authority loss is physical.

## The crash sequence (outdoor_splits-large_mid, mpc_full_8k, timeline)

| t [s] | err [m] | z / sp_z | v_z | tilt / ref | thrust_d |
|---|---|---|---|---|---|
| 3.0 | 0.29 | 4.58 / 4.57 | +1.0 | 69° / 71° | 13.3 N |
| 3.2 | 0.33 | 4.81 / 4.62 | +1.4 | 50° / 64° | 6.5 N |
| 3.4 | 0.51 | 5.05 / 4.55 | +0.5 | 29° / 36° | **0.0 N** |
| 3.6 | 0.60 | 4.96 / 4.35 | −1.4 | 51° / 63° | **0.0 N** |
| 3.8 | 0.57 | 4.51 / 4.03 | −2.7 | 81° / 71° | 30.2 N |

1. **Lag → vertical overshoot.** Tracking a 2.8 g, ~70°-tilt segment,
   the stack lags the reference (see "entry causes"), then overshoots
   it vertically (z 0.2–0.5 m above sp_z, climbing).
2. **Zero thrust is the position-cost optimum.** With the reference
   below/behind, free fall is the fastest descent, and nothing in the
   cost prices the controllability it destroys: `w_pos = 500` vs
   `thrust_weight = 1` around `u_r = hover`. Rotor telemetry confirms
   commands [0,0,0,0], rotors spooled to idle (~185 rad/s) for ~0.4 s.
3. **Free fall ⇒ no attitude authority.** Torque is differential
   *thrust*; at idle there is none to differentiate. The held τ_d is
   near zero anyway (symmetric u₀ at the corner allocates no torque).
   Tilt drifts 29°→81° across one cycle; on splits_mid it reaches 162°.
4. **Recovery arrives late.** The next 27–34 N command meets rotors
   still at idle; thrust lags by the motor time constant, the vehicle
   keeps a −2.7 m/s descent, re-overshoots, and the cycle repeats with
   growing amplitude — z ratchets 5.0 → 3.0 m over two cycles — until
   z = 0.
5. Every recorded early exit is the geofence *floor* (`pos z = −1.0`),
   i.e. a ground impact, not a lateral escape.

**Dose–response** (fraction of solves at thrust < 1 N / > 33 N,
switch rate): healthy baseline 0.01 / 0.00 / 0.0 Hz →
large_mid 0.15 / 0.02 / 0.2 Hz → splits_mid 0.29 / 0.06 / 0.8 Hz →
indoor_fast 0.51 / 0.27 / 3.2 Hz. Severity of the cycle tracks severity
of the failure exactly.

## Entry causes (what creates the initial ~0.5 m error)

Per mission class, confirmed by the S1 experiment (reference
completion: Ω_r finite-differenced from the reference quaternions +
u_r = T_r/4 feedforward, `complete_references` flag):

- **S1 reference gap** (feasible aggressive missions): with Ω_r = 0 and
  u_r = hover, the cost actively penalizes the body rates and thrust
  the maneuver requires — the stack under-tilts and lags. S1 completion
  cut drag-super_mid rms 0.216 → 0.169 (−22 %) and **fixed
  outdoor_splits_mid outright** (1.28 rms + ground crash → 0.239 rms,
  clean finish, zero-thrust fraction 0.29 → 0.05). This is the paper's
  eq. 10 discipline; gap analysis S1 confirmed empirically.
- **Thrust ceiling (C3)** (large_mid, 3.2 g vs 6.3 g sim ceiling): S1
  *changes* the failure rather than fixing it — the vehicle stays
  airborne and drifts wide (lateral fence exit at z ≈ 2–4 m, no ground
  impact) instead of entering the bang-bang cycle. The error source is
  missing thrust, and honest attitude/rate references at least keep the
  aircraft controllable while it fails to keep up.
- **Infeasible references (C4)** (`_fast` missions, 10–13 rad/s
  demanded vs 10/10/6 limits): error appears immediately regardless;
  S1 crashes *earlier* because it chases the infeasible reference
  harder. Nothing recoverable here except rejecting the mission
  (bake-time feasibility check, as already recommended).

Contrast: the reduced `mpc` stack on the same missions dies
differently — it keeps thrust as a smoothly-tracked input and instead
tumbles through the 90–120° attitude-reference flip (rates 15 rad/s)
of the singular cross-product map. Two stacks, two failure grammars:
`mpc` fails through *attitude*, `mpc_full` fails through *thrust*.

## Implications / fixes, in order of leverage

1. **S1 reference completion** (planned; now verified): implement
   Ω_r + u_r completion in the firmware `fill_reference` and the sim.
   It removes the lag that opens the door on every feasible mission.
2. **Price controllability in the cost**: the bang-bang exists because
   zero thrust is free. A thrust floor (u_min ≈ 5–10 % hover, matching
   the real ESC idle) or an asymmetric input cost below hover would
   deny the solver the free-fall corner at trivial tracking cost. The
   firmware's `u_bounds` lower edge is 0.0 today.
3. **S2 reference governor** (gap analysis): saturate the position
   error fed to the cost so a 0.5 m lag cannot flip the optimum to the
   input-space corner; re-time the reference on gross lag.
4. The INDI side needs no change: it did exactly what was asked with
   the authority it had.

*Method note: `FullSolveDiag` logging and the `complete_references`
experiment flag are opt-in fields on the sim controller (default off,
default paths byte-identical; snapshot + suites verified green).
Single runs per cell; sim vehicle (8.5 N motors), perfect sensors.*
