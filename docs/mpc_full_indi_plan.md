# Plan: `outer_loop: mpc_full` — Full-Model NMPC + INDI Inner Loop

> **Status 2026-08-12** — Stages 1–5 implemented (Stage 0 turned out
> unnecessary: `AttitudeControlSetpoint.torque_n_m` was telemetry-dead on
> the consuming side, so `(T_d, τ_d = I∘α_d)` ships in the existing
> message with no external crate bump). Sim stability grid passes at both
> 50 Hz and 100 Hz solve rates, clean + noisy IMU
> (`cybflight_sim/tests/mpc_full_indi_stability.rs`). Firmware compiles in
> all three variants (`outer_mpc`, `outer_mpc_full`,
> `outer_mpc_full + indi_off`). Remaining: Stage 6 (on-target solve-time
> measurement + bench checklist).
>
> **Math review (delegated agent, vs INDIflight C reference + paper
> eq. 32–35): sound — cleared for bench.** Q1 pseudo-control space exact;
> Q2 incremental law = verified INDIflight formulation (valid fixed
> point, correct G2 sign); Q3a no gyroscopic double-counting with INDI
> on; Q4 inertia roundtrip exact; Q5 thrust channel consistent. Two
> review fixes applied: `torque_n_m` now carries the RAW allocation
> torque τ(u₀) and the α pseudo-control `α = I⁻¹·(τ_d − ω×Iω)` is
> derived at IMU rate with the fresh gyro (paper Fig. 3 placement of
> eq. 32) in both firmware `indi_task` and the sim controller.
> **2026-08-13**: the MPC model, the (T_d, τ_d) reduction, and the α
> inner loop now consume the **full 3×3 inertia tensor** (validated
> constructor `full_quad_model::inertia_from_array`: symmetrize, floor
> the diagonal, Sylvester SPD check, fall back to the floored diagonal
> on failure) — closing the review's diagonal-vs-full split with INDI's
> G1 at no measurable solve cost (RTI mean 34.1 µs, was 34.2). The
> full-model builder warns only if validation falls back.
> **Tilt fence adopted** (cos-space relaxed log barrier on
> `1 − 2(qx²+qy²)`, both MPC models, `mpc_tilt_max_deg` /
> `mpc_tilt_barrier_tau` / `mpc_tilt_barrier_delta`, schema v47, flight
> YAMLs pin 60° / 0.5 / 0.05): measured overhead ≈ +1 % of an RTI solve
> (noise-floor); in closed-loop sim it held a 1.0×-limit tumble arrest to
> 66° peak tilt (was 119°, diverged). Fence semantics only — it bounds
> what the controller *chooses*; it does not recover states pushed past
> it (the deep-tilt free-fall stationary point persists; a rate-threshold
> recovery gate remains the complement). Known accepted
> deviation: the `indi: no` static-inversion ablation omits the eq.-29
> gyroscopic re-add (realizes τ_mpc − ω×Iω; ≈ 9 rad/s² at ω = 5 rad/s) —
> flag before using that ablation in published comparisons.

*2026-08-12 · reference architecture: Sun et al., "A Comparative Study of
Nonlinear MPC and Differential-Flatness-Based Control for Quadrotor Agile
Flight," IEEE T-RO 2022 ([arXiv:2109.01365v6](https://arxiv.org/abs/2109.01365v6), Fig. 3, eq. 32–35)*

## Architecture

```
                 50–100 Hz                      IMU rate (1 kHz)
 ┌─────────────┐  (T_d, α_d)  ┌──────────────────────────────────┐  DShot
 │  FullQuad   │─────────────▶│ INDI: τ_d = τ_f + I(α_d − ω̇_f)  │────────▶ motors
 │  NMPC (SQP) │              │ + G1/G2 inversion + thrust model │
 └─────────────┘              └──────────────────────────────────┘
   rate-limit barrier              torque loop + disturbance
   constraints active              rejection at full IMU rate
```

- The NMPC solves for per-motor thrusts `u` (13-state `FullQuadModel`,
  body-rate barrier constraints active). We do **not** send `u` to the
  motors. Instead, per paper eq. (32), the first control is algebraically
  reduced to a *desired collective thrust* and *desired angular
  acceleration*:

  `T_d = Σ u₀ᵢ`, `α_d = I⁻¹·(τ(u₀) − ω×Iω)` — via `FullQuadModel::alloc`.

- INDI consumes `(T_d, α_d)` **in place of its rate-error stage**. The
  entire remaining pipeline — filtered gyro derivative, rotor-speed
  feedback (τ_f), G1/G2 incremental inversion, thrust linearization,
  motor dynamics compensation, failsafes — is unchanged and keeps running
  at IMU rate. Between NMPC ticks `α_d`/`T_d` are held; the gyroscopic
  term and all incremental corrections re-evaluate at 1 kHz.
- **No rate gains anywhere in this path.** `rate_gains` is only used by
  the legacy `rate_sp` entry point.

### Why 50–100 Hz NMPC is enough to stabilize (the concern, addressed)

Separate the three loops by what actually needs bandwidth:

1. **Attitude/position stabilization** (what the NMPC closes). A hover
   quad's unstable modes are *slow* — the pendulum-like
   attitude–translation coupling has time constants of hundreds of ms; the
   attitude loop needs a crossover of only ~5–15 Hz. A 100 Hz zero-order
   hold contributes ≈ 5 ms of effective delay ≈ 15° of phase at 8 Hz —
   comfortable margin. Discrete state feedback at 10–20× the crossover
   frequency is the classic rule, and 100 Hz satisfies it.
2. **Torque / angular-acceleration tracking** (what INDI closes). This is
   where model error and fast disturbances live: motor thrust curves,
   CoG offset, prop wash, aero torque. It genuinely benefits from rate —
   the incremental assumption needs Δt ≪ motor time constant (~30 ms) and
   matched filter group delays. The paper ran it at 300 Hz; **we run it at
   1 kHz, faster than the reference implementation.**
3. **Rotor-speed control** — in the paper delegated to a 500 Hz low-level
   board; for us the ESC + our motor-dynamics compensation covers this.

The kHz rates in FPV firmware exist for gyro filtering, D-term noise, and
prop-wash rejection with high PID gains — i.e. loop-2 concerns, which INDI
keeps at full rate here — not because attitude stabilization needs kHz.
Two more enablers in the paper that we replicate: the NMPC's body-rate
constraints keep the vehicle in a regime where 100 Hz replanning is
adequate (our barrier does this), and warm-started RTI makes each tick a
cheap state-feedback evaluation. Finally, our *current* stack already
stabilizes with the outer MPC at 50 Hz commanding rates — mpc_full changes
what the 50–100 Hz loop commands, not how much stabilization runs between
its ticks.

What a 100 Hz NMPC cannot do: correct *rate-tracking* error between ticks
(the α it commands may be stale for up to one tick). INDI guarantees the
commanded α is achieved despite disturbance torque; the residual staleness
is the price of deleting the rate gains, and the paper's 20 m/s / 5 g
flights bound how much that costs in practice.

### `mpc_full` with `indi: no` — recommendation

The INDI pipeline already carries a `do_indi` factor that zeroes the
incremental terms (`indi: no` today = "plain rate controller"). With the
α entry point, the same factor degrades to **model-based static
inversion**: `τ = I·α_d + ω×Iω` → `G1⁻¹` → thrust linearization, at IMU
rate with fresh gyro in the gyroscopic term. This is *exactly* the
paper's "NMPC without INDI" baseline (eq. 29–30), which they flew — with
~78 % worse tracking and materially worse robustness to model mismatch.

**Recommendation: allow it, wired as that static inversion, labeled
experimental.** Rationale:

- It is the scientifically meaningful A/B baseline (the whole point of
  this exercise is comparison), and it costs ~zero extra code — it is the
  α-mode pipeline with `do_indi = 0`.
- Rejecting it at bake would delete the paper's own ablation from our
  toolbox; resurrecting a P-rate loop instead would reintroduce the gains
  this architecture exists to remove.
- Guard rails: a loud boot warning (mirroring the existing
  "INDI DISABLED" warn), and treat it as bench/indoor-mocap-only. It has
  **no disturbance rejection** below the NMPC — expect steady-state
  attitude offset under any asymmetry and no prop-wash rejection. Not a
  flight default, ever.

## Stages

**Stage 0 — messages** *(small; external coordination)*
`cybflight-msgs` (registry crate) gains an angular-acceleration field:
extend `AttitudeControlSetpoint` (it already carries
`collective_thrust_n`, `body_rate_rad_s`, `torque_n_m`) with
`body_ang_accel_rad_s2: Vector3<f32>`, or add a dedicated
`InnerLoopSetpoint`. Version bump + publish to the utadr registry.
Decision at implementation time; extending the existing message keeps the
`RATE_COMMAND` signal reusable.

**Stage 1 — INDI α entry point** *(cybflight_core, small)*
Refactor `IndiController::step` so the post-`rate_dot_sp` pipeline is a
shared internal function; add `step_alpha(gyro, accel, alpha_sp, spf_sp_z,
…)` that feeds `alpha_sp` where `rate_gains ∘ rate_err` used to go
(`indi/controller.rs:545`). `rate_gains` untouched (legacy path only).
Tests: `step_alpha` ≡ `step` when fed the identical `rate_dot_sp`;
`do_indi = 0` reduces to static inversion.

**Stage 2 — setpoint extraction** *(cybflight_core, small)*
`FullQuadModel::inner_setpoint(u0, omega) -> (T_d, alpha_d)` using
`alloc()` + `inertia_inv`, matching paper eq. (32) with *measured* ω in
the gyroscopic term. Document the hold/recompute split explicitly:
held at NMPC rate = `T_d`, `α_d`; re-evaluated at IMU rate = ω×Iω, τ_f,
all filters. Test: `alpha_d` ≡ `dynamics(x0, u0)[10..13]`.

**Stage 3 — firmware outer-loop variant** *(cybflight, the big one)*
Feature `outer_mpc_full`: an `outer_loop` variant (shared mission/sampler
infrastructure) running `FullSqpSolver` over
`FullQuadModel::from_vehicle_params` — barrier params, `mpc_max_iters`,
`mpc_kkt_tol` all already in the param plane. `x0` takes ω from
`VehicleOdometry.twist` (verify frame + latency against FUSED_IMU during
bring-up). Publishes `(T_d, α_d)` on the extended setpoint.
**Health guards are part of this stage, not a follow-up**: non-finite or
cap-hit-with-huge-KKT solutions are discarded — fallback publishes zero-α
+ hover thrust and escalates to the existing staleness failsafe after N
consecutive bad ticks. (This structure promotes solver failure from
"degraded setpoint" to "commanded acceleration"; the guard is
non-negotiable.)

**Stage 4 — build plumbing** *(small)*
`tools/vehicle_features.py`: `OUTER_FEATURE["mpc_full"] =
"outer_mpc_full"`; `vehicle_yaml` `BuildYaml::validate`: `mpc_full` +
`indi: no` allowed with a stderr warning naming it experimental;
Cargo feature; mutual-exclusion gates in `control/mod.rs`; a `check-all`
matrix row; docs (`architecture.md` outer-loop section, CLAUDE.md crate
map untouched).

**Stage 5 — sim validation first** *(moderate — do before hardware)*
`MpcFullIndiController` in `cybflight_sim` mirroring the firmware split
(NMPC 50/100 Hz + INDI 1 kHz vs the plant), plus the `indi: no`
static-inversion variant as the ablation. This directly answers the
stability question empirically at both solve rates, with sensor noise,
before any hardware risk. Reuse the closed-loop stress scenarios
(`mpc_state_constraint_stress.rs`) against the sim plant. New regression
rows = deliberate `just sim-snapshot` in its own commit.

**Stage 6 — target bench** *(bench time)*
On-target solve-time measurement (expect ≈ 1.9× the simple solver per
iteration; budget check against `mpc_rate_hz`, warning already wired);
props-off arming checks; staleness-fallback drill (kill the outer task,
verify failsafe); tethered hover indoors on mocap before free flight.

## Risks and open questions

| Risk | Mitigation |
|---|---|
| Solver divergence commands garbage α | Stage-3 guards + fallback path; barrier keeps solver in-basin (stress suite) |
| ω source frame/latency mismatch | Bring-up check `twist.angular` vs FUSED_IMU; use gyro directly if needed |
| msgs crate coordination | Stage 0 is one additive field + version bump; do it first |
| INDI filter delay caps α-loop bandwidth | Already tuned for current INDI; revisit `sync_filter_hz` only if bench shows lag |
| CPU at `mpc_rate_hz` ≥ 100 | Measured before flight (Stage 6); `mpc_max_iters` stays 1 (RTI) |
| Outdoor/GPS estimate quality | First bring-up indoors on mocap (paper-equivalent conditions); outdoor later, `indi: yes` required |
