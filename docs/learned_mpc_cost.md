# Learned adaptive cost for the NMPC — discussion, PLAN A, PLAN B

Status: PLAN B implemented and trained in sim, 2026-08-29 (see §4). PLAN A is design only.

This file records (1) the review of the ACMPC paper against the
`cybflight_core::acmpc` port, (2) the discussion of whether an ACMPC-style
learned cost can be flown by the existing SQP-RTI NMPC, and the two plans
that came out of it:

* **PLAN A** — reduce tracking RMSE on the trajectories we train on, by
  learning a per-tick modulation of the NMPC cost (stage set + terminal
  set). Trained and evaluated against the actual Rust solver and plant.
* **PLAN B** — generalize to *new* reference trajectories without
  retraining. (Open; see the section at the end.)

---

## 1. Background: what was reviewed

### 1.1 ACMPC port (`crates/cybflight_core/src/acmpc/`) vs the paper

Romero et al., *Actor-Critic Model Predictive Control*, IEEE T-RO 2025.
The paper's actor is a cost network (observation → per-stage diagonal
`Q_t`, linear `p_t`) feeding a differentiable box-DDP (`mpc.pytorch`,
one iteration, hover cold start); the first control of the plan is the
CTBR action. PPO backpropagates through the solve into the cost network.

The Rust port's solver math is correct: the Riccati backward pass,
projected-Newton QP (`pnqp`: exact-equality active set, `1e-11` ridge,
Armijo γ = 0.1, warm start from the later stage, 20-iteration cap), the
forward line search on the true nonlinear cost, and the CTBR model
jacobians all match `mpc.pytorch` and were re-derived by hand.

It is, however, a port of **our own** training re-implementation
(`cybflight-isaac/policy_acmpc.py`, verified to 3e-3 by
`tests/acmpc_gate_race.rs`), not of `acmpc_public`, and it diverges from
the paper's text in these places:

| item | paper | ours |
|---|---|---|
| cost-net activation | ReLU | GELU (`mod.rs:149`) |
| observation | `[v, R]` + 4 corners × next 2 gates = 36, running-stat normalized | 20-dim gate-frame `p, v, rpy, ω, rotor speeds`, 1 gate, no normalization |
| horizon | N = 2 for all racing results | N = 5 |
| `p` bounds | "0.1 and 100000" | `(σ−0.5)·1e5`, thrust entry `−(σ·range_q·G+0.1)` — not in the paper |
| MPC actuation model | per-rotor thrust limits, CTBR derived from them | CTBR box directly |

One latent bug: `PolicyFrame::Enu` is accepted by `AcmpcPolicy` but
`DroneDx` is hard-coded NED (gravity `+G e₃`, thrust `−a_c z_b`). Either
reject `Enu` in `new()` or make the model frame-aware.

### 1.2 box-DDP vs our SQP-RTI (both at one iteration)

| | box-DDP (`acmpc/ddp.rs`) | SQP-RTI (`mpc/sqp_solver.rs`) |
|---|---|---|
| start each tick | hover cold start (stateless by design) | warm start from last `u_bar`; reset on divergence |
| input bounds | exact active set per stage; gains only on free controls; rides the bound | cubic penalty `rho` + clamp in forward sweep; gains ignore the bound; hugs, does not ride |
| step control | line search on true cost | full Newton α = 1 |
| regularization | `1e-11` ridge | 1e-4 → 1e-2 ladder, finiteness checks, `diverged` verdict |
| integration | exact quat-exp rollout, Euler linearization | RK4 rollout, Euler linearization |
| cost | diagonal `Q_t, p_t` on absolute `[x;u]`, per stage, no terminal | tracking error vs `x_ref/u_ref`, geodesic attitude, contouring, tilt barrier; one weight set + terminal |
| runtime | data-dependent (PNQP ≤ 20 × Armijo ≤ 10 per stage) | fixed per iteration, sparse-B |
| state constraints | none | tilt barrier |

The DDP is the better optimizer per iteration; the SQP is the better
real-time controller. The paper chose the DDP for differentiability and
statelessness (what PPO wants), not flight quality.

### 1.3 Can an ACMPC-trained cost be flown by the SQP?

Not as trained, with N = 20 required (N < 10 was found unstable on
aggressive maneuvers with the tracking cost):

1. The cost head emits exactly `2·T·14` numbers for its training horizon
   (T = 5); nothing exists for stages 5..19. Retraining at T = 20 is
   mandatory.
2. The network learned "one Newton step *from a hover rollout*". A
   warm-started RTI lands elsewhere; cold-starting the SQP at N = 20 is
   worse (a 0.4 s hover rollout from a steep bank is far from the real
   trajectory, so the linearization is bad).
3. Attitude cost (per-component `q` weights vs geodesic error) and bound
   handling (active set vs cubic penalty) have no exact mapping; they
   diverge exactly where racing lives (on the bounds).

Where the mapping *is* exact: diagonal `Q x² + p x = Q(x−x_ref)² + c`
with `x_ref = −p/(2Q)`, for position, velocity and inputs.

If the goal were to fly the learned cost through the SQP, the honest path
would be a differentiable torch port of `SqpSolver` (RTI, RK4, cubic
penalty, warm start as a detached observation) in the training loop.

### 1.4 One cost set for all stages?

Agreed direction: **two sets — a constant stage cost plus a separate
terminal cost**, not per-stage. Reasons: fits `SqpSolver::solve` with no
solver change; far fewer outputs (the paper's Fig. 4 shows learnability
drops with head size); interpretable; matches the paper's own finding
(§V-F) that the terminal `Q_N, p_N` track the critic's Hessian/gradient
while stage costs do something else. Lost: in-horizon mode switching
(Fig. 2). Partially recovered by position *and* velocity references, and
by a v2 stage→terminal blend exponent.

### 1.5 The objective is tracking, not racing

Our objective is faithful tracking of a given agile reference (position
sampler + contouring cost). That changes the problem: the reference is
fixed, the action is the *weight vector*, and the MPC is part of the
environment. So **no differentiable solver is needed** — plain PPO on a
black-box environment that can be the real Rust `SqpSolver` + `QuadPlant`.
Everything in §1.3 about solver-in-the-loop mismatch disappears.

---

## 2. PLAN A — learned adaptive cost on the training trajectory family


### Goal

Reduce the geometric (closest-point) position RMSE of the SQP-RTI tracking
NMPC on agile reference trajectories by letting a small network modulate
the MPC's cost weights every tick — **two weight sets: one constant stage
set and one terminal set** — while keeping the flight solver exactly as it
is: `QuadModel`, N = 20, one warm-started Newton step per tick, position
sampler + contouring cost.

This is *not* ACMPC. ACMPC (Romero et al., T-RO 2025) learns the cost of a
short-horizon MPC as the action of a racing policy and needs a
differentiable solver because the action is the solver's output. Here the
reference trajectory is given and must be tracked faithfully; the network's
action is the weight vector, and the MPC is part of the environment. Two
consequences:

* **No differentiable MPC.** Plain PPO on a black-box environment. The
  environment can be the real Rust solver and plant (`cybflight_core::mpc`
  + `cybflight_sim`), so the trained policy sees byte-identical solver
  behaviour on the bench.
* **The warm start is inside the environment.** Cold-vs-warm consistency,
  which is the main transfer risk for ACMPC-style costs, does not arise.

### What the network outputs

Twenty scalars, all **log-multiplicative modulations of the hand tune**:

```
w = w_nom · exp(β · tanh(z))          β = ln 10  → w ∈ [w_nom/10, 10·w_nom]
```

| set | weights | count |
|---|---|---|
| stage | `w_contour, w_lag, w_vel[3], w_att[3], w_input[4]` | 12 |
| terminal | `w_contour_N, w_lag_N, w_vel_N[3], w_att_N[3]` | 8 |

Why this parameterization and not raw `Q, p` à la ACMPC:

* `z = 0` reproduces the current controller exactly. PPO starts from a
  flying baseline instead of from crashes, and the "off" switch is trivial.
* Bounded on both sides, so no near-zero weight can make the Riccati
  sweep ill-conditioned (the failure the ACMPC paper's §VI warns about).
* The network never moves the *reference*. Faithful tracking is the
  objective, so the sampler's `x_ref` stays authoritative; the network
  only decides how the solver trades the error components against each
  other and against control effort over the next 1 s.

Two optional knobs for a v2, deliberately excluded from v1 so the first
result is attributable:

* a scalar interpolation exponent that blends stage → terminal weights
  along the horizon (recovers most of a time-varying `Q_t` for one number);
* a reference time offset `Δτ` fed to the position sampler's trust window.

### What the network sees

Everything must exist at the outer-loop tick on the H743 and be
yaw-invariant, so express it in the reference's path frame
`{t̂ = v_ref/‖v_ref‖, n̂ = ẑ × t̂, b̂ = t̂ × n̂}` (hover fallback to world
frame when `‖v_ref‖ < VEL_EPS`, matching the contouring cost).

| block | entries | dim |
|---|---|---|
| tracking error | position error in path frame; velocity error in path frame | 6 |
| attitude | attitude error vector vs `q_ref`; body rates | 6 |
| reference preview | at nodes k ∈ {0, 5, 10, 15, 20}: `‖v_ref‖`, `a_ref` in path frame (3), `‖jerk_ref‖` — the agility of what is coming | 25 |
| solver health | last `u0` (normalized), last SQP cost (log), KKT norm, `diverged` flag | 7 |
| saturation | current thrust fraction of ceiling; max rate fraction | 2 |

≈ 46 inputs, fixed normalization constants (no running statistics on the
firmware). Network: 46 → 128 → 128 → 20, tanh, ≈ 25 k MACs — negligible
next to the SQP.

### Reward

The sim's score is `rms_pos_err_m` (`cybflight_sim/src/runner.rs`,
`(position − setpoint.position)` RMS over the mission). RMSE is monotone in
the mean of `e²`, so the per-step reward optimizes it directly:

```
r_k = − ‖e_geom‖²                       closest-point distance to the path, squared
      − λ_lag · e_lag²                  λ_lag ≈ 0.1: keep time progress, do not let the
                                        policy trade all lag for contour
      − λ_u   · ‖u_k − u_{k−1}‖²_W      control smoothness (W: 1/u_max² per channel)
      − λ_z   · ‖z_k − z_{k−1}‖²        weight slew: the RTI assumes cost continuity
                                        between ticks, a jump breaks the one-step solve
      − λ_sat · saturation_k            fraction of channels on their bound
      − C     · 1[crash | geofence | tilt > limit]   episode terminates
```

`λ_u, λ_z, λ_sat` are small regularizers (start 1e-2, 1e-1, 1e-2); the
first term dominates by design. Add `− λ_T · ‖e_geom‖²` on the final
`terminal` samples if `terminal_pos_err_m` regresses.

### Environment

* **Plant**: `cybflight_sim` `QuadPlant` with the mpc_indi stack (the
  configuration the regression snapshot scores). Not `rl_reference`.
* **Controller**: the real `SqpSolver` at N = 20, `mpc_max_iters = 1`,
  position sampler, contouring cost, tilt barrier as flown. Weights are
  written into `QuadModel` before each solve.
* **Trajectories**: procedurally generated MINCO trajectories from random
  waypoint sets (the planner in `trajectory_planning`), with the speed /
  aggressiveness scaled so that the set spans "easy" to "at the actuator
  limit". Hold out the figure8 slow/mid/timeopt scenarios as the eval set.
* **Domain randomization**: mass ±15 %, thrust-map gain ±10 %, motor τ,
  outer-loop latency 0–1 tick, constant wind, IMU/estimator noise from the
  noisy autotest. Start states perturbed off the trajectory start.
* **Episode**: one trajectory; terminate on crash/geofence/tilt.
* **Binding**: `pyo3` module over `cybflight_sim` exposing
  `reset(seed) -> obs`, `step(z) -> (obs, r, done, info)`, batched over
  environments. Alternative if a Python dependency is unwelcome: run PPO
  in Rust; the env is the harder part and it is already Rust.

PPO settings: the ACMPC paper's finding that MPC-based actors want *low*
exploration transfers directly — init log-std ≈ −1.5 in `z` space,
γ = 0.98, GAE λ = 0.95, clip 0.2. Run the network every tick in training
(it is what the firmware will do).

### Controls (so the result means something)

1. **Constant-weight re-optimization first.** Before training an adaptive
   policy, optimize the same 20 constant weights (CMA-ES over `z`, same
   reward, same trajectories). Whatever RMSE that reaches becomes the
   baseline. The adaptive policy must beat *that*, not the hand tune;
   otherwise the gain is attributable to tuning, not adaptivity.
2. **Terminal-only ablation.** Freeze the stage set at the CMA-ES optimum,
   learn only the terminal 8. Tells you where the leverage is.
3. **Eval on held-out trajectories and on the frozen snapshot scenarios**,
   reported through `just sim-check`-style rows so the numbers are
   comparable with everything else in the repo.

### Firmware / core changes (small)

1. `QuadModel`: add terminal weight fields `w_pos_n, w_vel_n, w_att_n`
   and make `MpcProblem::terminal_cost_hess_grad` use them (today it
   reuses the stage state cost). Default = stage weights → snapshot
   unchanged.
2. `outer_loop`: a per-tick hook that writes the 20 weights into the model
   before `solve`. The hot-reload path already mutates the model, so this
   is a widening, not a new mechanism.
3. `nn::Mlp` already runs the network; add the observation assembler
   (path-frame errors + preview) next to the sampler, since the preview
   is exactly the sampler's `SamplerNode` array.
4. Safety: clamp `z` to `[−1, 1]`; on any non-finite output or a
   `diverged` solve, fall back to `z = 0` (the hand tune) for that tick and
   reset the warm start as today. A param `mpc_learned_cost` (0 = off)
   gates the whole thing.

### Phases

| phase | deliverable | done when |
|---|---|---|
| 0 | terminal weights in core; per-tick weight injection in sim controller | snapshot unchanged with defaults |
| 1 | pyo3 env + CMA-ES constant-weight baseline | new constant tune, RMSE row vs hand tune |
| 2 | PPO adaptive policy + ablations | RMSE beats phase-1 baseline on held-out set |
| 3 | firmware hook + bench flight on figure8 mid | logged RMSE vs phase-1 tune on the real vehicle |

---

## 3. PLAN B — situation-conditioned weight adaptation that transfers to unseen trajectories

### Goal

Learn a **situation → priority** map: given the vehicle's current
attitude, velocity, tracking error and the maneuver the reference is about
to demand, decide how the NMPC should weigh its objectives so the vehicle
returns to the desired flight behaviour. Examples of the knowledge to be
learned:

* about to enter a split-S (reference attitude goes through inverted,
  thrust demand collapses, pitch-rate demand peaks): raise attitude
  regulation so the maneuver stays stable;
* large contour error: prioritize shrinking it over lag and effort;
* large velocity error at low tilt: prioritize velocity before position;
* on a saturated straight sprint: relax lag, hold contour.

This is knowledge about *situations*, and situations recur across
trajectories. The plan therefore differs from PLAN A in what the network
sees, what it is trained on, and how it is verified — not in the solver,
the weight parameterization, or the reward's core term, which are shared.

### Design principle

**The network must never be able to tell which trajectory it is on.**
Everything it sees is a local, yaw-invariant, physically bounded
description of the current situation. If trajectory identity is not in
the input, it cannot be overfit.

Concretely excluded from the observation: absolute position, yaw,
mission time / progress, trajectory duration, and solver history (last
cost, KKT, `diverged`) — the last because solver history is the most
likely channel for trajectory-specific quirks to leak in.

### Observation (the "situation")

All vectors in the reference path frame `{t̂, n̂, b̂}` (hover fallback to
world frame as in the contouring cost).

| block | entries | dim |
|---|---|---|
| **rotation** | body `z` axis of the vehicle in path frame (3); body `z` of the reference attitude `q_ref` in path frame (3); attitude error vector (3); body rates (3) | 12 |
| **velocity** | `v` in path frame (3); `‖v_ref‖`; velocity error in path frame (3) | 7 |
| **tracking error** | contour error (`n̂`, `b̂` components), lag error, `‖e‖`; sign-preserving so direction is visible | 4 |
| **upcoming maneuver demand** (what "about to enter a split-S" looks like in numbers) at preview nodes k ∈ {0, 5, 10, 15, 20}: tilt of `q_ref` (cos θ), body `z` of `q_ref` in path frame (3), flatness body-rate demand `‖ω_ref‖`, thrust margin `1 − T_ref/T_max`, rate margin `1 − ‖ω_ref‖/ω_max` | 35 |
| **actuation margin now** | thrust fraction of ceiling; max rate fraction | 2 |

≈ 60 inputs, fixed normalization constants. The preview block is
computed from the sampler's `SamplerNode` array by the existing flatness
chain (`flatness_to_thrust_omega`, tilt-yaw reference quaternion), so
the firmware already has every quantity.

### Outputs, solver, reward

As PLAN A: 12 stage + 8 terminal log-multiplicative modulations of the
hand tune, written into `QuadModel` before each one-iteration SQP solve.
Reward core is `−‖e_geom‖²`; PLAN B adds explicit *stability* shaping
because recovery is the point:

```
r_k = − ‖e_geom‖² − λ_lag·e_lag²
      − λ_att · ‖attitude error‖²        keeps "stable" in the objective, not just "close"
      − λ_ω   · ‖ω‖²/ω_max²              discourages rate-limit riding during recovery
      − λ_u·‖Δu‖² − λ_z·‖Δz‖² − λ_sat·sat
      − C · 1[crash | tilt beyond limit | geofence]
      + B · 1[first entry into ‖e_geom‖ < e_band]   one-off recovery bonus (optional)
```

### Training distribution: situations, not trajectories

Episodes are short (2–3 s) **recovery episodes**: a random maneuver
primitive as the reference, and the vehicle initialized *off* it.

* **Maneuver primitives** (procedural, MINCO-feasible, parameterized by
  scale and speed so that thrust/rate margins sweep 0 → 1): straight
  sprint / brake, hairpin, chicane / slalom, climb / dive, split-S,
  Immelmann, power loop, hover-to-sprint step, sustained banked turn.
  Chain 1–3 primitives per episode so "the maneuver after this one" is
  also varied.
* **Initial disturbance**, sampled independently of the primitive:
  position offset (0–2 m, random direction incl. lag/lead along the
  path), velocity error (0–5 m/s), attitude offset (0–90° tilt, random
  axis), body-rate offset. Each sampled log-uniformly so small
  disturbances are not drowned out.
* **Domain randomization** as PLAN A (mass, thrust map, motor τ, latency,
  wind, noise).
* Full laps of figure8 / racetrack scenarios are used **only for
  evaluation**, never for training.

Because the primitives are generated to cover the vehicle's feasibility
envelope and the disturbances cover the error space, any state a real
trajectory can put the vehicle in is locally in-distribution. That is the
coverage argument that replaces "we trained on many trajectories".

### Function class

An MLP (60 → 128 → 128 → 20, tanh) is acceptable here because its input
is a low-dimensional physical state with no identity channel. Two
optional structural priors if the learned map turns out noisy:

* **monotone heads** for the obvious relations (attitude weight
  non-decreasing in reference tilt; contour weight non-decreasing in
  contour error), enforced by construction;
* a **gain-schedule fallback** over three margin features (thrust margin,
  rate margin, `κ·v²`) fit by CMA-ES — the 60-parameter model that
  cannot surprise, kept as a lower bound.

### Verification that the knowledge is situational

1. **Probe plots** (the analogue of the paper's Fig. 2 value maps): hold
   the observation at a nominal level-flight situation and sweep one
   input at a time — reference tilt preview 0 → 180°, contour error
   0 → 2 m, velocity error 0 → 5 m/s — and plot every weight. The claims
   "split-S ⇒ more attitude regulation" and "large error ⇒ contour
   priority" become checkable curves, and a curve that depends on nothing
   physical is a red flag for leakage.
2. **OOD evaluation**: figure8 slow/mid/timeopt (the frozen snapshot
   rows), plus shape classes never used as primitives, at 0.5× and 1.2×
   speed. Report RMSE vs the constant-weight CMA-ES tune (PLAN A phase 1)
   and the in-distribution number, so the generalization gap is one
   figure.
3. **Coverage check** on any new trajectory: histogram of its preview
   features against the training distribution — done offline before a
   flight, and cheap.
4. **Ablation**: same policy with the preview block zeroed. Quantifies how
   much of the gain is anticipation ("about to enter") vs reaction.

### Relation to PLAN A

Same core, solver, parameterization and firmware hooks (PLAN A phase 0
serves both). Differences: PLAN B removes the solver-history inputs,
replaces trajectory-family training with recovery-episode training on
maneuver primitives, adds stability shaping to the reward, and adds the
probe/OOD/coverage verification. Expect PLAN B to lose a little
in-distribution RMSE to PLAN A and to win on anything unseen; the
snapshot scenarios are the tie-breaker because they are the ones the
vehicle actually flies.

### Phases

| phase | deliverable | done when |
|---|---|---|
| 0 | shared with PLAN A phase 0 (terminal weights in core, per-tick weight injection) | snapshot unchanged with defaults |
| 1 | maneuver-primitive generator + recovery-episode env (pyo3 over `cybflight_sim`) | coverage histograms span margins 0 → 1 |
| 2 | CMA-ES constant tune on recovery episodes (baseline) | RMSE row |
| 3 | PPO situational policy + probe plots + preview ablation | beats phase-2 on figure8 rows and OOD shapes |
| 4 | firmware hook + bench flight | logged RMSE vs phase-2 tune on the real vehicle |

---

## 4. Implementation (PLAN B, sim) — 2026-08-29

### What was built

| piece | where | notes |
|---|---|---|
| terminal weights | `cybflight_core::mpc::quad_model::QuadModel::{w_pos_n, w_vel_n, w_att_n}`, `terminal_cost_hess_grad`; `QuadDynamicsModel::terminal_cost_hess_grad` (default = stage) | defaults equal the stage weights → `regression_snapshot` unchanged |
| weight parameterization + observation + policy wrapper | `cybflight_core::mpc::cost_adapt` (`no_std`) | `w = w_nom·10^z`, `z ∈ [−1,1]^20`; 59-entry path-frame situation observation (§3); `CostPolicy` over `nn::Mlp` (tanh hidden, new `Activation::Tanh`) |
| controller hooks | `cybflight_sim::controller::MpcIndiController::{set_cost_z, set_cost_policy, peek_situation, cost_nominal}` | observation assembled between reference fill and solve; `peek_situation` is side-effect free (sampler is `Copy`) |
| maneuver primitives | `cybflight_sim::primitives` | 10 kinds (sprint, hairpin, chicane, climb/dive, split-S, Immelmann, loop, circle, hover-sprint, random dense waypoints), optional chaining, lead-in/out straights, bidirectional time scaling to a sampled thrust budget (0.45–1.05 of ceiling), 1.2 m ground floor |
| recovery environment | `cybflight_sim::rl_env::RecoveryEnv` | one env step = one 100 Hz outer period (10 INDI ticks, 80 plant RK4 substeps); reward `−10·e_geom² − 1·e_time² − 0.2·‖ω/ω_max‖² − 0.1·mean(Δz²) − 20·crash` (crash = ground, >3 m off the path, or non-finite; no speed cap); disturbances log-uniform, 30 % zero per channel; plant mass ±10 % |
| Python bridge | `crates/cybflight_rl` (pyo3 + rayon), `maturin develop --release -m crates/cybflight_rl/Cargo.toml --target x86_64-unknown-linux-gnu` | `VecEnv(n, vehicle_yaml, …)`, ~100 k env-steps/s on 24 cores |
| training | `tools/train_cost_policy.py` (SB3 PPO, `[128,128]` tanh, log-std −1.5, γ 0.98, λ 0.95) | exports `policy.bin` (flat `nn::Mlp` layout) and `probe.json` (situation sweeps); `--blind` zeroes the observation → the constant-`z` control |
| evaluation | `crates/cybflight_sim/tests/learned_cost_eval.rs`, `tools/learned_cost_table.py` | 13 indoor missions × {nominal, learned, blind}; geometric (closest-point) and time-indexed RMSE; `indoor_splits_timeopt` is the held-out test |

Vehicle throughout: `vehicles/sakura_bench_leader_1khz.yaml` (the flight airframe and its flown tune), plant physics from `sim_baseline.yaml`, exactly as `figure8_tinympc_compare.rs`.

### Calibration facts worth keeping

* The planned time-optimal missions peak at **0.92 of the thrust ceiling with rates at 1.0× the limit** (`tests/mission_demand_probe.rs`); sparse-waypoint min-snap primitives were rate-bound instead, which is why the templates are dense, have lead-in straights, and scale head/tail velocity together with the durations.
* Under nominal weights the recovery episodes crash 0.6 % (all ground hits before the 1.2 m floor was added: 15 %, dominated by split-S).
* The sim's `rms_pos_err_m` is **time-indexed**; with the position sampler's 0.1 s lag allowance it is 3–5× the geometric RMSE on the fast missions. Both are reported.

### Results, v1 (20 M steps, reward `−10·e_geom² − 1·e_time²`)

Held-out **`indoor_splits_timeopt`** (never seen in training, nor any
mission file): geometric RMSE **0.1588 → 0.0963 m (−39 %)**, completed,
time-indexed RMSE 0.696 → 0.763 m (+9.5 %). Blind constant-`z` control:
0.1467 m (−8 %). Full table (`target/sim-out/learned_cost/table_v1.txt`):

| mission | nominal geom | learned geom | blind geom | Δgeom | Δtime |
|---|---|---|---|---|---|
| circle slow / mid / timeopt | 0.091 / 0.123 / 0.181 | 0.029 / 0.047 / 0.098 | 0.088 / 0.118 / 0.167 | −68 / −62 / −46 % | +8 / +13 / +45 % |
| figure8 slow / mid / timeopt | 0.065 / 0.087 / 0.147 | 0.028 / 0.042 / 0.077 | 0.061 / 0.081 / 0.134 | −56 / −52 / −48 % | −3 / −1 / +2 % |
| slalom slow / mid / **timeopt** | 0.043 / 0.056 / 0.102 | 0.021 / 0.032 / **0.208 (incomplete)** | 0.041 / 0.053 / 0.096 | −51 / −43 / **+104 %** | +8 / +4 / +91 % |
| splits slow / mid / fast / **timeopt (held out)** | 0.071 / 0.093 / 0.127 / 0.159 | 0.031 / 0.045 / 0.079 / **0.096** | 0.067 / 0.088 / 0.129 / 0.147 | −57 / −52 / −38 / **−39 %** | −1 / +3 / +3 / +10 % |

Reading:

* The gain is **adaptivity, not tuning**: the blind control (same
  optimizer, same reward, observation zeroed) settles on a mild constant
  retune (contour ×1.5, roll/pitch attitude ×1.7, terminal roll ×2.3) worth
  5–10 %; the situational policy is worth 40–70 %.
* The policy pays for contour with lag: time-indexed RMSE creeps up on
  most missions (the position sampler's 0.1 s lag allowance is where the
  freedom comes from), and on `slalom_timeopt` — the mission with the
  most sustained near-saturation speed — it falls 3.4 m behind late in the
  run and misses the 0.15 m terminal gate (no crash). Even the nominal
  stack lags 1.5 m there. v2 prices lag 4× higher (`--w-time 4`).
* Probes (`target/cost_policy/probe.json`, `tools/learned_cost_table.py`):
  as the previewed **thrust margin closes**, contour weight rises ×3→×8;
  as the **reference tilts toward inverted**, yaw-attitude and thrust
  effort weights drop (frees the thrust axis); a **positive along-path
  velocity error** raises contour priority; a **roll attitude error**
  swings contour vs lag priority with the error's sign. The hypothesized
  "inverted ⇒ more roll/pitch attitude weight" did *not* emerge — the
  policy found a different lever for the same situation.

### Robustness of v2b, and the change to v5 (2026-08-29)

`tests/learned_cost_robustness.rs` flies 4 missions × 9 conditions ×
{nominal, learned}: localization noise injected into the state the outer
loop sees (white σ_p/σ_v/σ_θ + a 1 s-correlated drifting position bias;
INDI keeps the true IMU), and plant-only mismatch (mass ±15 %, thrust
ceiling −15 %, motor τ ×2, rotor drag ×3, and a combined case).

**v2b (log-multiplicative, ×10 range, input weights learned) was not
robust**: it held on the held-out split-S under 3 cm noise, mass ±15 %,
thrust −15 % and drag ×3 (−38…−43 %), but figure8_timeopt collapsed under
3 cm noise alone (+241 %), mass +15 % and thrust −15 % (incomplete), and
the combined case failed where nominal completed. The policy lived at
the ×10 edge, and it had been trained on ground-truth state.

Decision (not domain randomization): **v5** fixes the input weights at
the YAML nominal `[mpc_w_thrust, mpc_w_rate_*] = [1, 20, 20, 20]` (never
learned) and learns a *residual* on the state weights,
`w = w_nom·(1 + 0.9·z)`, so every weight stays within `[0.1, 1.9]·w_nom`.
16 outputs (8 stage + 8 terminal). INDI at 1 kHz is part of the
environment in training and in every evaluation (it always was; the
`MpcIndiController` stack is what the env steps).

Note the baseline moved with the canonical input weights (the sim's
snapshot path overrides them to `[1,1,1,1]`): nominal geometric RMSE on
the held-out split-S is **0.284 m** with `[1,20,20,20]` (0.159 m before).

**v5 results (20 M steps, reward `−10·e_geom² − 4·e_time² − …`)**:

| mission | nominal geom | v5 geom | Δgeom | Δtime | blind Δgeom |
|---|---|---|---|---|---|
| splits_timeopt (held out) | 0.284 | **0.261** | **−8.3 %** | +2.7 % | +0.9 % |
| circle slow / mid / timeopt | 0.218 / 0.275 / 0.345 | 0.186 / 0.235 / 0.298 | −14.5 / −14.5 / −13.6 % | +2…3 % | ≈ +1 % |
| figure8 slow / mid / timeopt | 0.153 / 0.185 / 0.267 | 0.140 / 0.169 / 0.249 | −8.7 / −9.1 / −6.8 % | +2 / +1 / 0 % | ≈ +1 % |
| slalom slow / mid / timeopt | 0.094 / 0.117 / 0.217 | 0.088 / 0.108 / 0.197 | −5.8 / −8.1 / −9.5 % | +5 / +2 / +4 % | ≈ +1 % |
| splits slow / mid / fast | 0.158 / 0.202 / 0.262 | 0.142 / 0.179 / 0.244 | −9.9 / −11.5 / −6.7 % | +3 / +3 / +2 % | ≈ +1 % |

All 13 complete; no mission regresses. **Robustness (v5)**: the gain is
preserved under every perturbation — −8 % on the held-out split-S in all
nine conditions (−7.9 % even at drag ×3 and combined), −5…−15 % elsewhere
— and v5 never fails where nominal completes (the two incomplete cells,
8 cm-noise-with-10 cm-drift and figure8 combined, are incomplete for
nominal too). Time-indexed RMSE within +5 %.

Reading: bounding the residual to `[0.1, 1.9]·w_nom` traded most of the
v2b headline gain (−44 %) for uniform, perturbation-invariant improvement
(−6…−15 %) with the input weights the vehicle actually flies. The blind
control is ≈0 here, i.e. within this range there is no better *constant*
tune than the current one — the improvement is entirely situational.

### v6 — learned body-rate input weights (2026-08-29)

Changes from v5: the three body-rate input weights are learned on a log
scale, `w_rate = w_nom·4^z ∈ [w_nom/4, 4·w_nom]`, thrust weight fixed at 1
(19 outputs); the nominal rate weight is **10** (`EnvConfig::rate_weight_nominal`,
`MPC_W_RATE` in the evals; the YAML's flown 20 is kept as a reference
row). To make a low rate weight cost something in the sim, training runs
INDI on a `NoisyImu` (gyro σ 0.03 rad/s, accel 0.3 m/s²) and the reward
adds `−25·‖Δω_cmd/ω_max‖²` on the commanded body-rate change between
solves. Calibration at `z = 0`: geometric term 0.39 → 0.25 for
`w_rate` 20 → 2.5 while the chatter term rises 0.13 → 0.20, so the
constant optimum sits near `w_rate ≈ 5–10` rather than at the floor
(the blind control confirms: it chose `[9.9, 13.4, 12.2]`).

Nominal baselines (geometric RMSE, held-out split-S timeopt): 0.284 m at
`w_rate 20`, **0.239 m at `w_rate 10`**.

| mission | nominal (r=10) | v6 | Δgeom | Δtime | blind Δgeom |
|---|---|---|---|---|---|
| **splits_timeopt (held out)** | 0.239 | **0.207** | **−13.2 %** | −0.5 % | +4 % |
| circle slow / mid / timeopt | 0.174 / 0.223 / 0.290 | 0.091 / 0.134 / 0.261 | −48 / −40 / −10 % | −7 / −2 / +8 % | +16 / +15 / +8 % |
| figure8 slow / mid / timeopt | 0.123 / 0.155 / 0.229 | 0.074 / 0.109 / 0.191 | −40 / −30 / −17 % | −5 / −2 / +2 % | +12 / +9 / +6 % |
| slalom slow / mid / timeopt | 0.076 / 0.098 / 0.171 | 0.047 / 0.067 / 0.133 | −38 / −31 / −22 % | 0 / −4 / +6 % | +14 / +12 / +17 % |
| splits slow / mid / fast | 0.129 / 0.166 / 0.201 | 0.077 / 0.109 / 0.159 | −40 / −34 / −21 % | −3 / 0 / +7 % | +12 / +11 / +10 % |

All 13 complete. The blind constant control is *worse* than nominal
everywhere (+4…+17 %): within this weight space there is no better
constant tune, so the whole gain is situational.

**Robustness (4 missions × 11 conditions, now including gyro noise
σ = 0.03 / 0.06 into INDI)**: the gain is preserved in every cell —
held-out split-S −10…−14 % under all eleven conditions (gyro noise has
no measurable effect on either controller; loc 8 cm + 10 cm drift −9.7 %;
drag ×3 −10 %; combined −10.8 %), circle_mid −32…−42 %, slalom_mid
−12…−33 %, figure8_timeopt −14…−17 %. One cell is flagged incomplete
for v6 (figure8_timeopt, motor τ ×2): geometric RMSE still −16 % and no
crash, but the 3 s terminal hold ends 0.15+ m off the endpoint.

**What v6 learned about the rate weights** (per-solve `z` logs and
probes): on slow/mid missions it holds roll/pitch rate weights at ~4–6
(below the nominal 10) with yaw ~6–8; on the held-out split-S timeopt
the roll-rate weight *rises* with reference speed and with closing
thrust margin (corr +0.74 / −0.67, p90 = 21) while the yaw-rate weight
does the opposite (corr −0.64 / +0.71). Read: cheap, agile rates when
the maneuver is precise and unsaturated; stiffer rate commands when fast
and thrust-limited, where fighting saturation with body rates costs
tracking. State weights: stage contour ×1.7–1.9 and terminal pitch
×1.4–1.5 throughout, terminal contour ×0.7; with a large contour error
the stage roll-attitude weight sweeps from ×0.1 to ×1.3 with the error's
sign (bank into the correction).

### `mpc_w_rate` sweep on the held-out split-S timeopt (2026-08-29)

Nominal state weights, no policy, clean IMU
(`target/sim-out/learned_cost/w_rate_sweep_splits_timeopt.txt`):

| w_rate | 0.5 | 1 | 2 | 3 | 4 | 5 | 7.5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| geom RMSE [m] | 0.143 | 0.159 | 0.176 | 0.188 | 0.198 | 0.206 | 0.224 | 0.239 | 0.263 | 0.284 |

Monotone: in the clean sim the best rate weight is the smallest one, and
gyro noise (σ 0.03 / 0.06 into INDI) changes nothing — the 30 Hz sync
filter removes it before the rate loop sees it, and the commanded-rate
chatter only rises from 0.072 to 0.098 (rms Δω/ω_max per solve) between
w_rate 20 and 1. The sim therefore cannot price a low rate weight through
noise. What it *can* price is **motor lag**
(`w_rate_motor_tau_cliff.txt`, plant motor time constant scaled, controller
unchanged):

| w_rate | τ ×1 | τ ×1.5 (30 ms) | τ ×2 (40 ms) |
|---|---|---|---|
| 1 | 0.159 | — | **1.08 (diverged)** |
| 2 | 0.176 | 0.177 | **1.26 (diverged)** |
| 3 | 0.188 | 0.190 | **1.03 (diverged)** |
| 4 | 0.198 | 0.200 | 0.216 |
| 5 | 0.206 | 0.208 | 0.222 |
| 7.5 | 0.224 | 0.226 | 0.241 |
| 10 | 0.239 | — | 0.255 |

A hard cliff between 3 and 4 at 40 ms motor lag; nothing at 30 ms. The
rate weight is the MPC's only guard against commanding rate changes the
motors cannot deliver, and `m*_tau: 0.02` in the vehicle YAML is an
un-identified default. Recommendation: **`mpc_w_rate = 5`** as the flown
nominal (≈ 1 decade of margin on motor lag, −28 % geometric RMSE vs the
current 20); `4` if the bench identifies τ ≤ 25 ms; `1–2` only after τ is
measured and a rate-command spectrum on the real vehicle confirms no
chatter. The v6 policy on top of a nominal of 4–5 adds a further
−11…−12 % on this mission.

### v7 — log-symmetric state range, nominal `w_rate = 5` (2026-08-29)

`w = w_nom·4^z` for the 16 state weights (`[0.25, 4]·w_nom`), rate
weights `5·4^z`, thrust fixed; reward as v6 (`w_time 4`, `λ_du 25`,
noisy gyro). Nominal at `w_rate 5`: split-S timeopt **0.206 m**.

Result: held-out split-S **0.2055 m (−0.3 %)** — no gain; other missions
−5 … −35 % (slow/mid) and −5 … −15 % (timeopt); robustness intact
(no new failures, gain preserved where present). Training stagnated
(return −364 → −358 over 20 M steps); the policy stays near nominal
(2 % of solves at the range edge, contour ×1.7–2.4), i.e. the wider
range was not used. Blind controls (24 envs) drifted *worse* during
training and are not a reliable comparison at this size.

Reading: with `λ_du = 25` the reward charges exactly the rate-command
changes that a stiff contour cost produces, so the sim-optimal policy
under this reward is "stay near nominal". v2b's situational gain was
partly funded by rate-command chatter this reward now prices. v8 tests
the hypothesis properly: same parameterization, `λ_du = 5`,
`log_std_init −1.0`, 48-env blind control.

### v8 — v7 parameterization with `λ_du = 5`, `log_std_init −1.0` (2026-08-29)

Held-out split-S timeopt **0.206 → 0.170 m (−17.7 %)**; all 13 missions
−18 … −60 % geometric RMSE (blind 48-env control: −5 … −10 %); time-indexed
−11 … +13 %. Training actually progressed (−320 → −291). Robustness:
gain preserved in every condition *except* motor τ ×2, where v8 diverges
on split-S (1.03 m) and figure8_timeopt (0.58 m) — τ ×1.5 is fine.
Per-solve logs: contour ×2.6–3.7 (the v2b lever, now fenced at ×4) **and**
rate weights at a median of 2–4 with 72–100 % of solves below 3 — the
motor-lag cliff zone the `mpc_w_rate` sweep located. So v8 = the
situational contour gain + an unsafe rate floor. v9 keeps everything and
puts a hard floor of 4 under the rate weights (piecewise-log map,
nominal 5 at `z = 0`, `[4, 20]`), to test whether the stiff-contour gain
survives motor lag on its own.

### v9 — v8 + rate-weight floor of 4 (2026-08-29) — **deliverable**

Parameterization: state weights `w_nom·4^z` (`[0.25, 4]·w_nom`), rate
weights piecewise-log `[4, 20]` with nominal 5 at `z = 0`
(`cost_adapt::{STATE_LOG_BASE, RATE_LOG_BASE, RATE_FLOOR_FRAC}`), thrust
weight fixed at 1; reward `−10·e_geom² − 4·e_time² − 0.2·‖ω/ω_max‖² −
0.1·mean(Δz²) − 5·‖Δω_cmd/ω_max‖² − 20·crash`; INDI on a σ 0.03 rad/s
gyro; PPO `log_std_init −1.0`, 20 M steps, 48 envs.

| mission | nominal (r=5) | v9 | Δgeom | Δtime | blind Δgeom |
|---|---|---|---|---|---|
| **splits_timeopt (held out)** | 0.206 | **0.179** | **−13.2 %** | +6.0 % | −1.2 % |
| circle slow / mid / timeopt | 0.140 / 0.183 / 0.245 | 0.071 / 0.098 / 0.161 | −49 / −47 / −34 % | +2 / +9 / +22 % | −1 / −2 / −2 % |
| figure8 slow / mid / timeopt | 0.100 / 0.129 / 0.199 | 0.057 / 0.087 / 0.147 | −42 / −32 / −26 % | −1 / +1 / +3 % | −1 / 0 / −1 % |
| slalom slow / mid / timeopt | 0.063 / 0.081 / 0.140 | 0.038 / 0.052 / 0.098 | −40 / −36 / −30 % | +8 / +3 / +8 % | −3 / −2 / −3 % |
| splits slow / mid / fast | 0.106 / 0.137 / 0.166 | 0.062 / 0.096 / 0.139 | −41 / −30 / −16 % | +1 / +5 / +3 % | 0 / −2 / −2 % |

All 13 complete. **Robustness (4 × 12 conditions, incl. gyro noise and
motor τ ×1.5 / ×2)**: the gain is preserved in every cell — held-out
split-S −7.6 … −13.7 % under all twelve, **including motor τ ×2
(−13.7 %)** where v8 diverged; figure8_timeopt −19 … −29 %, slalom_mid
−7 … −37 %, circle_mid −30 … −52 %. No cell fails that nominal completes.
Time-indexed RMSE rises on the fast circle (+22 %) — the policy still
trades some lag for contour there.

What v9 does (per-solve logs, held-out split-S): stage contour ×3.0
(relaxed toward ×2 as the thrust margin closes, corr +0.47 — it does
*not* fight saturation with position stiffness), lag ×1.4, lateral
velocity ×1.2, stage roll/pitch attitude ×0.7, terminal pitch ×2.0 and
terminal roll ×1.5; rate weights average 6–7 with p10 ≈ 4.4 (on the
floor) and p90 ≈ 10–13, the roll-rate weight rising with reference
speed (corr +0.54). The contour stiffening is the v2b lever, now fenced
at ×4 and paired with a rate floor the motor-lag sweep justifies —
v2b's "situational −28 %" delivered robustly.

### Conclusion of the study

| | held-out ΔRMSE | robust to motor τ ×2 | robust to 3 cm loc noise | blind control |
|---|---|---|---|---|
| v2b (×10, rates learned, `w_rate` base 1) | −44 % | no | no | −8 % |
| v5 (±0.9 residual, `[1,20,20,20]` fixed) | −8 % | yes | yes | +1 % |
| v6 (rates `10·4^z`, chatter cost 25) | −13 % | yes | yes | +4 % |
| v7 (state `4^z`, rates `5·4^z`, chatter 25) | −0.3 % | yes | yes | −8 % (noisy) |
| v8 (v7, chatter 5) | −18 % | **no** | yes | −5 % |
| **v9 (v8 + rate floor 4)** | **−13 %** | **yes** | **yes** | −1 % |

Plus the constant-tune finding that dominates everything: `mpc_w_rate`
20 → 5 is worth 0.284 → 0.206 m (−28 %) on its own, with the motor-lag
cliff between 3 and 4 setting the floor. The flight recommendation is
therefore **`mpc_w_rate = 5` + the v9 policy** (0.284 → 0.179 m on the
held-out mission, −37 % total), pending a bench identification of the
motor time constants and a rate-command spectrum check at `w_rate 5`.

### Thorough evaluation of the flight recommendation (2026-08-29)

`tests/learned_cost_eval.rs` with `MISSION_SET=all MPC_W_RATE_CURRENT=20
MPC_W_RATE=5`: every mission file (13 indoor + 11 outdoor), three
configurations — **current** (the flown tune, `w_rate 20`, no policy),
**r5** (`w_rate 5`, no policy), **rec** (`w_rate 5` + v9). Full table:
`target/sim-out/learned_cost/final_table.txt`.

* Geometric RMSE, rec vs current: **mean −54 %** (−33 … −67 %) on all
  24 missions; time-indexed RMSE mean −8 %; peak geometric error roughly
  halved everywhere (e.g. splits_fast 0.81 → 0.41 m, outdoor
  splits-large_fast 0.80 → 0.35 m); terminal error 3–5× smaller; peak
  motor saturation unchanged (the fast missions sit at 100 % under every
  configuration).
* Held-out split-S timeopt: 0.284 → 0.179 m (−37 %).
* The three `outdoor_*-super*` missions leave the harness's ±15 m
  geofence under all three configurations (a harness limit, not a
  controller event); their numbers are reported up to the exit.

`tests/learned_cost_robustness.rs` on the six fastest missions with
`MPC_W_RATE_NOMINAL=20` (current) vs `MPC_W_RATE=5` + v9 (rec), 12
conditions each (`robustness_final.txt`):

* Gyro noise, localization noise/drift, mass ±15 %, thrust −15 %, motor
  τ ×1.5, drag ×3: rec keeps −29 … −58 % on every mission.
* **Motor τ ×2 (40 ms)**: rec diverges on slalom_timeopt, circle_timeopt,
  splits_fast and outdoor splits-large_fast, where current completes.
  Attribution (`w_rate_cliff_all_fast.txt`): the *constant* `w_rate 5`
  already diverges there (1.06 / 1.01 / 5.3 m); raising the policy floor
  to 5 changes nothing. The motor-lag cliff is mission-dependent — on
  the split-S it lies between 3 and 4, on outdoor splits-large_fast
  between 7.5 and 10:

| constant `w_rate` at τ ×2 | slalom_to | circle_to | splits_fast | large_fast | splits_to | figure8_to |
|---|---|---|---|---|---|---|
| 5 | **diverged** | **diverged** | 0.44 | **diverged** | 0.22 | 0.21 |
| 7.5 | 0.21 | 0.31 | 0.32 | **diverged** | 0.24 | 0.22 |
| 10 | 0.19 | 0.33 | 0.30 | 0.44 | 0.26 | 0.24 |
| 15 | 0.21 | 0.35 | 0.34 | 0.56 | 0.28 | 0.26 |

  At τ ×1.5 (30 ms) every value from 5 up is fine on every mission.

**Revised recommendation.** The motor time constant decides it, and it is
an unidentified default (`m*_tau: 0.02`):

* τ ≤ 30 ms (identify on the bench): `mpc_w_rate = 5` + v9 — mean
  −54 % geometric RMSE across all missions, robust to everything tested
  at that lag.
* τ unknown / up to 40 ms: `mpc_w_rate = 10` + a policy with its rate
  floor at 10 (v10, range `[10, 40]`, in training) — expected in the
  −25 … −45 % band from the constant-`w_rate` and v6 numbers; robustness
  matrix to follow.
* Either way the bench step that unlocks the aggressive setting is a
  motor step-response identification; the sim cannot substitute for it.

### v10 — conservative variant: nominal `w_rate 10`, policy rate floor 10 (2026-08-29)

Same as v9 otherwise (state `4^z`, rates `[10, 40]`, `λ_du 5`). All 24
missions vs the current tune (`final_table_v10.txt`): **mean −41.6 %**
geometric RMSE (−24 … −54 %), time-indexed −7 %, peak error roughly
halved, terminal error 2–4× smaller; held-out split-S 0.284 → 0.203 m
(−29 %). Six-mission matrix (`robustness_final_v10.txt`): the gain holds
under gyro noise, localization noise/drift, mass ±15 %, thrust −15 %,
motor τ ×1.5, drag ×3 and the combined case (−24 … −54 %). At **motor
τ ×2** it still diverges on circle_timeopt and outdoor splits-large_fast
(the constant `w_rate 10` survives there at 0.33 / 0.44 m), i.e. with the
rate weights floored, the policy's ×3–4 contour stiffening is itself
what exceeds the 40 ms motor-lag margin on the two most demanding
missions. splits_timeopt, figure8_timeopt, slalom_timeopt and
splits_fast keep −27 … −39 % at τ ×2.

Summary of the recommendation space (all vs current tune, all missions):

| configuration | mean Δgeom | τ ×1.5 | τ ×2 |
|---|---|---|---|
| `w_rate 5` constant | ≈ −25 % | ok | fails on 3 fast missions |
| `w_rate 10` constant | ≈ −17 % | ok | ok |
| `w_rate 5` + v9 | **−54 %** | ok | fails on 4 fast missions |
| `w_rate 10` + v10 | **−42 %** | ok | fails on 2 (circle_timeopt, large_fast) |

The remaining exposure is a motor time constant of 40 ms, twice the
YAML's un-identified default. Closing it inside the policy would need
motor-lag randomization in training (plant `m*_tau` ∈ [15, 40] ms), a
plant-uncertainty measure distinct from the localization-noise DR that
was ruled out; not run here.

### Unmodelled quadratic body drag (2026-08-29)

Added to the plant only: `SimYaml::body_drag` (`½ρC_dA` per body axis,
`F = −k·|v|v`; default 0 so the snapshot is unchanged; `plant.rs`). The
controllers keep their drag-free model. Conditions in
`learned_cost_robustness.rs`: 0.006 / 0.012 / 0.024 N·s²/m² in xy
(C_dA ≈ 0.01 / 0.02 / 0.04 m²; 0.9 / 1.7 / 3.5 N at 12 m/s on the 0.6 kg
frame), 2× on z. Eight missions, current tune vs each policy
(`bodydrag_v9.txt`, `bodydrag_v10.txt`):

| body drag | current (r20) split-S | v9 | v10 | v9 Δ | v10 Δ |
|---|---|---|---|---|---|
| 0 | 0.284 | 0.179 | 0.203 | −37.0 % | −28.7 % |
| 0.006 | 0.302 | 0.191 | 0.216 | −36.7 % | −28.5 % |
| 0.012 | 0.319 | 0.203 | 0.229 | −36.4 % | −28.3 % |
| 0.024 | 0.351 | 0.226 | 0.253 | −35.7 % | −28.0 % |

Same picture on all eight missions: every controller degrades by the
same ≈ +24 % from zero to the heaviest drag (the position loop absorbs
a slowly varying disturbance), the relative gain of both policies moves
by ≤ 3 points (v9 −36 … −59 %, v10 −28 … −55 % at 0.024), no failures,
peak errors stay ≈ half of current. Neither policy is more sensitive to
unmodelled drag than the hand tune; drag is not where their risk is —
motor lag is.

### The `*-super` missions (fence widened to ±80 m, 2026-08-29)

`outdoor_splits-super_fast` is the fastest mission in the set: 32 × 36 m,
peak segment speed 30.7 m/s, 21.5 s, thrust-saturated throughout. With
the harness fence widened all three complete. Geometric RMSE [m]:

| mission | current (r20) | r5 | r5+v9 | r10 | r10+v10 |
|---|---|---|---|---|---|
| splits-super_fast | 0.700 | 0.510 | **0.324 (−54 %)** | 0.594 | **0.351 (−50 %)** |
| splits-super_slow | 0.363 | 0.245 | 0.140 (−61 %) | 0.296 | 0.165 (−54 %) |
| drag-super_mid | 0.216 | 0.161 | 0.111 (−49 %) | 0.186 | 0.116 (−46 %) |

Peak error halved (1.12 → 0.49 / 0.54 m on super_fast), terminal error
2–4× smaller, saturation unchanged (100 % on super_fast for all).
Time-indexed RMSE on the splits-super missions is 2–2.6 m for every
controller — at 30 m/s the sampler's 0.1 s lag allowance is 3 m, so the
time metric is dominated by allowed lag, not by tracking.

Matrix on splits-super_fast (`super_robust_{v9,v10}.txt`): both keep
−43 … −56 % under gyro/localization noise, mass ±15 %, thrust −15 %,
τ ×1.5, rotor drag ×3, body drag 0.006 / 0.012 and the combined case.
Differences: at **τ ×2** v10 holds (−49.5 %) while v9 diverges (4.7 m) —
the same motor-lag split as before, at the highest speed; at **body drag
0.024** (≈ 22 N ≈ 3.6 g at 30 m/s, above the 30 N thrust ceiling —
physically infeasible for the reference) v9 diverges and v10 degrades to
−21 % with a 3.5 m peak; the current tune survives both at 1.1 m RMSE.

### Geometric-only objective: can RMSE go below 0.2 m everywhere? (2026-08-29)

Premise from the user: time-indexed error is irrelevant as long as the
geometric error is small and there is no crash. Changes:

* **Metric**: `rms_geom` in both harnesses is now the distance to the
  closest point on the *whole* path (`rl_env::closest_point_dist_global`),
  lag-independent. The earlier ±0.5 s-window numbers overstated error for
  lagging controllers (e.g. super_fast constant `w_rate 10`: 0.594 → 0.503;
  v10 0.351 → 0.300).
* **Sampler lag** (`sampler_max_lag_s`, `SAMPLER_MAX_LAG` in the
  harnesses, `EnvConfig::sampler_max_lag_s`): the lever that lets the MPC
  slow into corners. Constant `w_rate 10`: 0.1 → 0.5 s takes split-S
  timeopt 0.213 → 0.185, figure8_timeopt 0.199 → 0.173, super_fast
  0.503 → 0.469; ≥ 0.5 s stops paying and the mission ends behind the
  3 s terminal gate (a harness criterion, not a crash).
* **Controller thrust ceiling** `mpc_thrust_frac` 0.75 → 1.0: no effect on
  any fast mission — the MPC is not collective-limited; INDI's 100 % is
  torque allocation at the corners.
* **v11**: v10's map (nominal `w_rate` 10, floor 10, state `4^z`) trained
  with `w_time 0.5`, sampler lag 0.5 s (20 M steps; the 12 M best
  checkpoint is evaluated below, `v11b` finishes the run).

All 24 missions, whole-path geometric RMSE [m]
(`summary_v10_lag01_global.json`, `summary_v11_lag05.json`):

| mission | current r20 (lag 0.1) | **v10 (lag 0.1)** | current (lag 0.5) | **v11 (lag 0.5)** |
|---|---|---|---|---|
| splits_timeopt (held out) | 0.256 | 0.180 | 0.229 | **0.133** |
| figure8 / slalom / circle timeopt | 0.232 / 0.210 / 0.284 | 0.142 / 0.118 / 0.169 | 0.175 / 0.213 / 0.275 | 0.105 / 0.101 / 0.130 |
| splits_fast | 0.219 | 0.118 | 0.252 | 0.101 |
| outdoor splits-large fast / mid / slow | 0.462 / 0.344 / 0.277 | 0.211 / 0.162 / 0.128 | 0.428 / 0.320 / 0.269 | 0.191 / 0.183 / 0.161 |
| outdoor splits-super fast / slow | 0.592 / 0.315 | 0.300 / 0.142 | 0.553 / 0.313 | 0.296 / 0.202 |
| all slow/mid indoor + outdoor drag | 0.045–0.217 | 0.035–0.119 | — | 0.037–0.156 |
| **missions < 0.2 m** | 6 / 24 | **22 / 24** | 7 / 24 | **22 / 24** |

Answer: **yes for 22 of 24 missions** with either configuration; the
two above 0.2 m are the 30 m/s `outdoor_splits-super_fast` (0.30) and
`splits-super_slow` (0.20, borderline for v11; 0.14 for v10). v11's
geometric-only objective buys the most on the held-out split-S
(0.180 → 0.133) and on large_fast (0.211 → 0.191, now under 0.2), and
costs a little on the slow missions (its slack is spent on progress it
no longer values).

Robustness of v11 (seven fast missions × 15 conditions, `robustness_v11.txt`):
gain preserved (−30 … −60 %) under gyro noise, localization noise/drift,
mass ±15 %, thrust −15 %, τ ×1.5, rotor drag ×3, body drag up to 0.024
and the combined case; no crashes. At **motor τ ×2** it diverges on four
missions (slalom_timeopt, circle_timeopt, splits_fast, large_fast) versus
v10's two — the geometric-only objective is a little less lag-tolerant.
Many cells carry a terminal-gate "incomplete" flag because the vehicle
arrives late at the endpoint; none of those is a crash.

Remaining limit: super_fast at 0.30 m is set by the reference itself
(30 m/s at 100 % allocation), not by the cost; neither more sampler lag
(the closest-point search loses the track beyond 0.5 s at that speed) nor
more thrust authority moves it.

v11b (training completed to 20 M steps, `summary_v11b_lag05.json`): still
22 / 24 below 0.2 m, but the split moved — slow/mid missions improve
(super_slow 0.202 → 0.149, drag/slow rows −5…−25 %) while the fastest
regress (large_fast 0.191 → 0.226, figure8_timeopt 0.105 → 0.137,
super_fast 0.296 → 0.306); no crashes. The 12 M checkpoint
(`cost_policy_v11/policy_best.bin`) is the better one for the fast
missions. super_fast stays at ≈ 0.30 m in every variant.

### Model improvements in the controller (2026-08-29)

Two physics changes to the outer-loop model, both trajectory-independent:

**Rotor drag in the prediction model** (`QuadModel::{drag_coeff, thrust_coeff}`,
`model_utils::rotor_drag_accel_jac`): the system-identification model of
`analysis/sysid_mcap.py`, `a_b = k·v_b·Σω` (`c = −m·k` = the YAML
`sim: aero_drag`), with `Σω` recovered from the commanded collective,
`Σω(T) = 2√(T/c_T)`, `c_T = max_thrust_n/ω_max²`. Analytic Jacobians
w.r.t. v, q (polynomial-R derivatives, consistent with the solver's
linearization) and T, verified by central differences. The
`aero_drag` coefficients are the identified sim-baseline ones; the plant's
quadratic body drag stays *unmodelled*.

Result with the **constant tune** (`w_rate 10`, no policy), whole-path
geometric RMSE, all 24 missions (`summary_dragmodel_const.json`):

| | current tune | `w_rate 20` + drag | **`w_rate 10` + drag** |
|---|---|---|---|
| splits_timeopt (held out) | 0.256 | 0.093 | **0.063** |
| figure8 / slalom / circle timeopt | 0.23 / 0.21 / 0.28 | 0.08 / 0.10 / 0.06 | 0.067 / 0.075 / 0.047 |
| splits_fast | 0.219 | 0.085 | 0.066 |
| outdoor splits-large fast / mid / slow | 0.46 / 0.34 / 0.28 | 0.05 / 0.04 / 0.02 | 0.044 / 0.034 / 0.019 |
| outdoor splits-super fast / slow | 0.59 / 0.32 | 0.05 / 0.02 | **0.041** / 0.020 |
| slow/mid indoor, outdoor drag | 0.05–0.21 | 0.015–0.04 | 0.012–0.03 |
| **missions < 0.2 m** | 6 / 24 | 24 / 24 | **24 / 24** (mean −84 %) |

Time-indexed RMSE falls with it (split-S 0.72 → 0.18 m): most of the
"tracking error" of the hand tune was a systematic drag-induced lag the
feedback never fully cancelled. Robustness (`robustness_dragmodel.txt`,
constant `w_rate 10`): holds under localization noise/drift, motor τ ×1.5,
rotor drag ×3 (0.24–0.67, i.e. mis-identified drag degrades gracefully),
**unmodelled quadratic body drag** up to 0.024 (≤ 0.27 m), and the
combined case; at motor τ ×2 it survives on all fast missions except
super_fast. `w_rate 5` under the drag model is *not* τ ×2-safe (the
better model makes the solve more aggressive) — 10 stays the nominal.

**Rate-loop lag in the prediction model** (`mpc::lag_quad_model::LagQuadModel`,
13 states `[p, q, v, ω]`, `ω̇ = (ω_cmd − ω)/τ`, lag rows integrated exactly
so a 50 ms step can represent a 20 ms constant; `MpcIndiController::enable_rate_lag`):
implemented, Jacobians verified, and **counter-productive at every τ
tried** (5 → 40 ms: split-S 0.125 → 0.372 vs 0.063 without it, with the
drag model). INDI's incremental law tracks a rate step faster than any
first-order lag the outer loop can assume at its 50 ms step, so leading
the command only overshoots. Kept, disabled. The earlier motor-lag
cliff is therefore not a missing-lag-model problem; it is the plant's
motor τ against INDI's own assumptions.

The learned policies (v9–v11) were trained without the drag model and
mismatch it (−34 … +17 %, fail τ ×2 more often); **v12** retrains the
policy with the drag model in the loop (`--model-drag`, nominal `w_rate`
10, `w_time 1`).

### v12 — policy retrained with the drag model in the loop (2026-08-29)

All 24 missions vs the drag-modelled constant tune (`summary_v12.json`):
**mean −7 %** (−23 … +27 %); better on the slow/mid missions (−10 … −23 %),
*worse* on the fastest ones (circle_timeopt +24 %, super_fast +19 %,
splits_timeopt +7 %). Matrix (`robustness_v12.txt`): equal to the
constant tune under noise, mass, thrust, τ ×1.5, body drag and rotor drag
×3; at motor τ ×2 it **loses** the margin the constant tune has
(circle_timeopt 0.10 → 0.97, large_fast 0.07 → 2.8, both incomplete).

Reading: the situational cost adaptation had been compensating mostly for
model error. With the drag term in the model there is little left for
it to buy on the demanding missions, and its remaining behaviour
(stiffening) still costs motor-lag margin. **The deliverable is the
constant tune `mpc_w_rate = 10` with the rotor-drag model**: 24 / 24
missions below 0.1 m geometric RMSE (0.012–0.075), −84 % vs the current
tune, robust to every perturbation tested except a 40 ms motor time
constant on the 30 m/s track. The learned layer stays as an optional
−10…−20 % on the slow/mid missions, not recommended for the time-optimal
ones.

### Final re-evaluation with the drag model only (rate-lag model removed) — 2026-08-29

`LagQuadModel` and every switch for it are deleted; the only model change
kept is the rotor-drag term. All 24 missions, whole-path geometric RMSE,
controller with drag model (`summary_drag_{v9,v10,v11}.json`):

| mission | cur r20 | r5 | **v9** (r5) | r10 | **v10** (r10) | r10, lag 0.5 | **v11** (r10, lag 0.5) |
|---|---|---|---|---|---|---|---|
| splits_timeopt (held out) | 0.093 | 0.049 | **0.046** | 0.063 | 0.054 | 0.063 | 0.049 |
| figure8 / slalom / circle timeopt | 0.083 / 0.104 / 0.060 | 0.052 / 0.054 / 0.038 | 0.039 / 0.043 / 0.038 | 0.067 / 0.075 / 0.047 | 0.049 / 0.050 / 0.044 | 0.067 / 0.075 / 0.047 | 0.042 / 0.043 / 0.031 |
| splits_fast | 0.085 | 0.058 | 0.063 | 0.066 | 0.065 | 0.070 | 0.063 |
| outdoor splits-large / -super fast | 0.051 / 0.052 | 0.042 / 0.046 | 0.035 / 0.032 | 0.044 / 0.041 | 0.036 / 0.043 | 0.044 / 0.041 | 0.038 / 0.047 |
| slow / mid rows | 0.015–0.040 | 0.010–0.028 | 0.006–0.022 | 0.012–0.034 | 0.009–0.025 | 0.012–0.034 | 0.010–0.038 |
| mean Δ vs own nominal | — | — | **−24.6 %** | — | **−20.3 %** | — | **−14.3 %** |
| missions < 0.1 m | 23/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |

Fast-mission matrix (`robustness_drag_{v9,v10,v11}.txt`): all three
policies keep or improve the nominal under localization drift, body drag
0.024 (except super_fast, where the reference is infeasible at that
drag and everything degrades) and the combined case (−15 … −45 %). At
**motor τ ×2** the policies lose the constant tune's margin on
circle_timeopt, large_fast and (v10/v11) super_fast (0.06–0.10 → 1.3–3.2),
while v9 *rescues* slalom_timeopt and splits_fast there (0.94 → 0.08,
1.01 → 0.30) where the constant `w_rate 5` fails. With sampler lag 0.5 the
constant `w_rate 10` + drag survives τ ×2 on all six fast missions.

Standing recommendation: **`mpc_w_rate = 10` + rotor-drag model** (0.012–0.075 m
everywhere, τ ×2-safe on all but super_fast; with `sampler_max_lag_s 0.5`
also on super_fast). The learned layer (v10 on this nominal) is worth a
further −20 % on average and is neutral-to-positive under every
perturbation except a 40 ms motor time constant on the three most
demanding missions; enable it once `m*_tau` is identified ≤ 30 ms.

### Unified indoor/outdoor framework (2026-08-30)

One fixed envelope for every vehicle and mission, frozen with the
checkpoint (`cost_adapt::{V_SCALE, V_ERR_SCALE, RATE_SCALE, OBS_CLAMP}`):
velocity ÷ 60 m/s, velocity error ÷ 15 m/s, position error in metres with
the ±3 clamp equal to the 3 m training crash bound, rates ÷ 10 rad/s.
Nothing is scaled per vehicle; a policy is valid wherever these units
hold. The clamp only bounds what the network reads — the MPC always sees
the true state and references.

Training family to match: primitive speeds log-uniform 1.5–60 m/s with
template geometry scaled by `(v/12)²` (speed-independent centripetal
demand), speed-up capped at 65 m/s, 10× compression and a 1.2 s minimum;
velocity disturbances to 15 m/s. Sanity at `z = 0` with the drag model:
median speed 12.6 m/s, p90 51 m/s, 0 crashes / 60 episodes, |obs| ≤ 3.

Quadratic body drag `a_b = −(k_q/m)·|v_b|v_b` (`½ρC_dA`, dominant above
~30 m/s) added to `QuadModel` (`body_drag_coeff`, `set_body_drag`; shared
Jacobian helper `model_utils::drag_accel_jac` with the rotor term, FD
verified) and to `analysis/sysid_mcap.py` as a second regressor column
`v_b|v_b|` in the drag fit, printed as `mpc_bodydrag_*` and
`sim: body_drag` next to the rotor coefficients. `v13` trains on the
unified units with both drag terms modelled (rotor from `sim_baseline`,
body 0.012 N·s²/m² as the outdoor stand-in until identified).

**v13 results** (unified units, 1.5–60 m/s primitives, both drag terms
in the training controller; evaluated on the indoor/outdoor missions with
the rotor-drag model, `w_rate 10`; `summary_drag_v13.json`,
`robustness_drag_v13.txt`): **mean −25.1 %** vs the drag-modelled nominal
(v10 on the same runs: −21.7 %), 24 / 24 missions below 0.1 m; held-out
split-S timeopt 0.064 → **0.047** (v10 0.054); super_fast 0.047 → 0.041.
One regression: `indoor_splits_fast` +14 % (0.082 → 0.093, peak 0.43 m).
Matrix: gain preserved under gyro/localization noise, mass, thrust,
τ ×1.5, body drag to 0.012 and (mostly) 0.024; at motor τ ×2 it fails on
splits_fast / large_fast / super_fast where the nominal survives two of
them (circle_timeopt now holds); on the 30 m/s track with rotor drag
×3 or body drag 0.024 it degrades more than the nominal. Same class of
exposure as v10 — the unified envelope did not cost indoor performance,
and the checkpoint is now valid to 60 m/s by construction.

### `outdoor_splits-super_timeopt` (2026-08-30)

New mission from the planner export `tmp/tmp_missions/outdoor_missions/outdoor_splits-super_timeopt.yaml`
(105 waypoints, 21.7 s, peak segment speed 29 m/s). MINCO's `MAX_PIECES`
(64) allows 63 waypoints, so the 42 straight-segment points with the
smallest chord deviation were dropped (`missions/outdoor_splits-super_timeopt.yaml`,
header documents the decimation). Fidelity of the 63-point fit against all
105 export points: max 0.66 m, mean 0.054 m (`tests/mission_demand_probe.rs`).
The fitted trajectory demands 1.16× the thrust ceiling and 1.24× the rate
limit — beyond the envelope everywhere, i.e. infeasible by construction.

| | geom RMSE | time RMSE | peak |
|---|---|---|---|
| current tune (`w_rate 20`, no drag) | 0.612 | 2.66 | 1.14 |
| `w_rate 10` + drag | 0.044 | 1.41 | 0.23 |
| + v10 | 0.036 (−17 %) | 1.25 | 0.23 |
| + v13 | 0.040 (−9 %) | 0.86 | 0.23 |

Matrix (15 conditions): both policies keep −10…−27 % under gyro noise,
3 cm localization noise, mass ±15 %, thrust −15 %, τ ×1.5, body drag
0.006/0.012; at 8 cm drift v10 +19 %, v13 ±0; at motor τ ×2, rotor drag
×3, body drag 0.024 and the combined case the *nominal itself* diverges or
degrades (0.67–3.9 m) — the reference is beyond the actuators — and the
policies do not rescue it (v13 worse at drag ×3 and body drag 0.024,
better in the combined case).

### `MAX_PIECES` 64 → 128 (2026-08-30)

`cybflight_core::trajectory_planning::MAX_PIECES = 128` so planner exports
with up to 127 waypoints bake unchanged; `outdoor_splits-super_timeopt`
now carries all 105 (63-point fit vs export: max 0.004 m). Firmware RAM
impact (static `.data + .bss`, 512 KB part), `sakura_bench_leader_1khz`:

| | static RAM | headroom |
|---|---|---|
| `MAX_PIECES 64` | 322 KB | 190 KB |
| 128, naïve | 451 KB | 74 KB |
| 128 + right-sized banded storage (`BandedSystem<S>`: yaw 4N×9, jerk 6N×13 instead of the snap 8N×17) | **370 KB** | **154 KB** |

Largest scaled statics at 128: `OFFLINE_MINCO` 86 KB, mission-planner task
state 56 KB (trajectory copies in the async future), `OFFLINE_MINCO_YAW`
22 KB, `MISSION_TRAJECTORY_SLOT` 16 KB. **`sakura_bench_leader_8khz`: 394 KB
static, 130 KB headroom** — below the ~145 KB at which that vehicle once
boot-looped on a 32 KB `.bss` addition (mechanism unresolved, see project
memory). Boot-test the 8 kHz vehicle before flying this build; the next
RAM reduction if needed is to stop the planner task holding trajectories
by value (write `get_trajectory()` into the slot directly, ≈ −30 KB).

#### Update 2026-09-08: online planner gated out, MINCO scratch right-sized

The 56 KB of "mission-planner task state" above was not trajectory
copies — it was the online BFGS `PlanSession` (its embedded `MincoJerk`
was sized at `MAX_PIECES` = 128 while the planner is capped at 20), held
across the solver's cooperative yields and therefore laid out in the task
future even though `USE_OFFLINE_PLAN = true` made that arm unreachable.
The 55 KB `WORKSPACE` static was allocated unconditionally for the same
dead arm. Both are now behind the `plan_online` cargo feature (off), and
`MincoSnap`/`MincoAcc`/`MincoJerk` are const-generic so each instance is
sized to its consumer's bound (`OFFLINE_MAX_PIECES` = 105 for the offline
solvers, `MAX_PLANNED_PIECES` = 20 for the planner). Measured with
`just size` (decimal kB, same convention as the table above):

| build | static RAM | headroom |
|---|---|---|
| `sakura_bench_leader_1khz`, before | 370.6 kB | 153.7 kB |
| `sakura_bench_leader_1khz`, after | **239.6 kB** | **284.7 kB** |
| `sakura_bench_leader_8khz`, after | **263.8 kB** | **260.5 kB** |
| `sakura_bench_leader_1khz` + `plan_online` | 219.3 kB | 305.0 kB |

Largest statics now: `OFFLINE_MINCO` 70 kB (was 86), `MPC_SOLVER` 35 kB,
`BATCH_BUF` 33 kB, `OFFLINE_MINCO_YAW` 18 kB (was 22),
`MISSION_TRAJECTORY_SLOT` 16 kB. In the `plan_online` build the planner
task future is 12 kB (was 56). The 8 kHz vehicle is back above the ~145 kB
threshold by a wide margin; the boot test before flying still applies
because the mechanism was never isolated. See `docs/architecture.md`
"Compile-Time Optimization Rules" #7 for the rule this established.

Full 105-waypoint mission (peak demand 1.24× thrust, 1.43× rate limit —
infeasible reference):

| | geom | time | peak |
|---|---|---|---|
| current tune | 0.598 | 2.68 | 1.15 |
| `w_rate 10` + drag | 0.043 | 0.71 | 0.29 |
| + v10 | **0.035** (−18 %) | 0.78 | 0.23 |
| + v13 | 0.038 (−13 %) | 0.51 | 0.30 |

v13 matrix: −2 … −20 % under noise, mass, thrust, τ ×1.5, body drag to
0.012, drag ×3 (1.01 → 0.50) and combined (0.49 → 0.35); worse at τ ×2 and
body drag 0.024 where the nominal already diverges.

### Sim-only extreme reference `simulation_splits-ultra_timeopt` (2026-08-30)

Source: `tmp/tmp_missions/outdoor_missions/simulation_splits-ultra_timeopt.csv`
— a full 100 Hz state/input trajectory: 70 s, **peak 99.7 m/s** (median
64 m/s), ±190 m footprint, altitude 8.5–100 m, thrust demand 0.91 of the
30 N ceiling, yaw rate at the 6 rad/s limit. Not a MINCO-through-waypoints
mission: the harnesses take it as 126 degree-7 pieces fitted by least
squares (`tests/fixtures/ultra_timeopt_pieces.json`, max residual 6.75 mm;
`PIECES_JSON=<path>`), fence widened to ±300 × 150 m. The plant's linear
rotor drag is 13–21 N at 60–100 m/s, so the vehicle cannot hold the
reference's timing: every controller runs ~16–18 m behind (time-indexed),
and the question is only whether it holds the *line*.

| | no drag model: geom / peak | drag model: geom / peak |
|---|---|---|
| current tune (`w_rate 20`) | 1.53 / 2.43 (incomplete) | — |
| `w_rate 10` | 1.29 / 2.28 | 0.383 / 3.41 |
| + v10 (indoor envelope, ÷10 units) | **6.37 / 36.3 — lost the track** | **4.02 / 24.8 — lost the track** |
| + v13 (unified envelope, ÷60 units) | 0.777 / 1.65 (terminal gate missed) | **0.215 / 1.35, completed (−44 %)** |

This is the case the unified framework was built for: v10's velocity
entries saturate at 30 m/s (`v/10`, clamp 3), so above that it operates
blind and its stiffened cost diverges twice; v13 sees the speed, was
trained to 60 m/s, and keeps the line at 100 m/s within 0.2 m with the
drag model — better than the constant tune in every configuration.

v13 matrix on the ultra reference (`ultra_matrix_v13.txt`, drag model
on): −47 … −73 % under gyro noise, 3 cm localization noise, mass ±15 %,
thrust −15 % and motor τ ×1.5; in two cells (gyro 0.03, loc 3 cm) the
constant tune itself loses the track (4.6–4.9 m) while v13 holds 0.12–
0.15 m — at 100 m/s the nominal is on a knife edge and the adaptation is
what keeps it on the line. It loses at 8 cm drift (+15 %). Everything
fails at motor τ ×2 (19 → 24 m) and under any quadratic body drag
(0.006 N·s²/m² is 60 N at 100 m/s, twice the thrust ceiling) — the
reference is then physically unreachable, not a controller matter; under
rotor drag ×3 v13 still salvages 0.73 m from the nominal's 9.4 m.

### v14 — v13 + domain randomization (2026-08-30)

Per user direction, training now randomizes, per episode, the **plant**
(controller model stays nominal): inertia ×[0.8, 1.25] per axis, per-motor
thrust ceiling ±15 % common ±3 % per motor, motor τ ×[0.75, 2.0]
log-uniform (spans the τ×2 robustness condition), rotor drag ×[0.5, 3],
body drag ×[0.5, 2]; plus per-episode **localization noise** on the state
the outer loop sees (white σ_p ≤ 3 cm, σ_v ≤ 10 cm/s, σ_θ ≤ 1°, drifting
bias ≤ 5 cm; INDI keeps the IMU). `EnvConfig::domain_rand`,
`--domain-rand 1.0`. Otherwise identical to v13.

Results (drag model, `summary_drag_v14.json`, `robustness_drag_v14.txt`,
`ultra_matrix_v14.txt`):

* All 25 missions: mean **−20.7 %** vs the drag-modelled nominal (v13:
  −25.1 %), 25/25 below 0.1 m. v14 gives back some clean-run gain on the
  indoor timeopt missions (splits_timeopt 0.047 → 0.066, slalom_timeopt
  0.055 → 0.081 vs v13) and fixes v13's one regression (splits_fast 0.093
  → 0.087); slightly better outdoors (super_timeopt 0.034).
* Matrix: at **motor τ ×2**, v14 now *holds or improves* on splits_timeopt
  (0.071), slalom_timeopt (0.098) and splits_fast (0.467 → **0.196**) —
  the DR working as intended — but still diverges on circle_timeopt,
  large_fast and super_fast (1.1–3.8) where the constant tune survives
  two of them. Under rotor drag ×3 and combined it is broadly better than
  the nominal (super_fast combined 3.47 → 1.11).
* Ultra reference (100 m/s): 0.383 → **0.152 (−60 %)**, completed — better
  than v13's 0.215 on the same run.

Net: DR traded ~4 points of mean clean-run gain for materially better
behaviour under mismatch (three of the six τ×2 fast-mission cells now
held, the extreme-speed case improved), without fixing all of them. v13
remains the best clean-run checkpoint; v14 is the robustness-oriented one.

v14 ultra matrix (drag model, `ultra_matrix_v14.txt`): mixed at 100 m/s.
It rescues the nominal's loc-3 cm divergence (4.61 → 0.19) and improves
gyro 0.06, mass ±15 %, thrust −15 % and τ ×1.5 (−13 … −73 %), but on this
knife-edge reference it is noisier than v13: clean 0.24 (v13 0.11 on the
matrix run; 0.152 on the single-run eval), and at gyro 0.03 it diverges
(5.6) where v13 held 0.12. The physically-unreachable cells (τ ×2, body
drag, drag ×3, combined) fail for every controller as before. At the
extreme edge of the envelope v13 remains the stronger checkpoint; v14's
advantage is on the flyable missions under dynamics mismatch.

### Definitive all-mission table (2026-08-30)

Whole-path geometric RMSE [m], every mission plus the sim-only 100 m/s
reference; all rows completed, no crashes. "current" = the flown tune
(`w_rate 20`, no drag model); every other column uses `w_rate 10`;
"+drag" adds the rotor-drag prediction model; v13/v14 add the learned
cost on top of it. Data: `summary_nodrag_const.json`,
`summary_drag_{v13,v13_fill,v14}.json`.

Mean vs current tune over the 25 missions: `w_rate 10` alone −17.7 %,
+drag −83.6 %, +v13 −87.5 %, +v14 −86.3 %. Ranges: current 0.045–0.60
(1.53 ultra); +drag 0.012–0.082 (0.38 ultra); v13 0.008–0.093 (0.215);
v14 0.008–0.087 (0.152). Time-indexed RMSE falls alongside (e.g.
super_timeopt 2.68 → 0.46 with v14). Per-mission values are in the
summaries; the fastest rows:

| mission | current | r10 | +drag | v13 | v14 |
|---|---|---|---|---|---|
| splits_timeopt (held out) | 0.256 | 0.213 | 0.064 | **0.047** | 0.066 |
| splits_fast | 0.219 | 0.162 | 0.082 | 0.093 | **0.087** |
| outdoor splits-large_fast | 0.462 | 0.374 | 0.044 | 0.037 | **0.035** |
| outdoor splits-super_fast | 0.592 | 0.503 | 0.047 | 0.041 | **0.040** |
| outdoor splits-super_timeopt (105 wp) | 0.598 | 0.511 | 0.043 | 0.038 | **0.034** |
| ultra 100 m/s (sim-only) | 1.528 | 1.287 | 0.383 | 0.215 | **0.152** |

### Plant rotor drag ×2 (2026-08-30)

`SIM_ROTOR_DRAG_SCALE=2` in `learned_cost_eval.rs` doubles the *plant's*
rotor-drag coefficients while every controller keeps the nominal
identified model — a clean 2× drag-identification-error study over all
25 missions (`summary_dragx2_{nodrag,v13,v14}.json`). All rows complete,
no crashes.

Mean vs the current tune at the same ×2 plant: `w_rate 10` −16 %, +drag
model −52 %, +v13 −63 %, **+v14 −68 %**. Errors grow roughly 4–10× over
the 1× plant everywhere (v14 0.02–0.41 vs 0.008–0.087): half the drag is
now unmodelled, and the residual behaves like the no-model study scaled
down. Ordering by regime: v13 leads on the slow/mid indoor rows, **v14
leads decisively on the fast outdoor rows** (super_timeopt 0.28 vs v13's
0.69, super_fast 0.26 vs 0.45, large_slow 0.13 vs 0.39) — the DR
training (rotor drag ×[0.5, 3] among its draws) paying off exactly in
its own regime. One anomaly: `outdoor_splits-large_mid/slow` where the
drag-modelled *constant* tune (0.74 / 0.48) is worse than the unmodelled
one (0.54 / 0.44) — a mismatched model can be worse than none when the
error is systematic and large; both policies recover from it.

Ultra reference at plant drag ×2 (drag ≈ 42 N at 100 m/s — above the
30 N ceiling, so the reference timing is unreachable): the drag-modelled
constant tune loses the track entirely (rms 4.97, peak 21.7 m, 33 m from
the endpoint), while **v14 keeps the line at 0.49 m rms / 1.49 m peak**
and only arrives late (terminal 3.5 m). The learned layer is the
difference between "off the track" and "on the line, behind schedule".

### v19 — wider contour ceiling + yaw-nominal retune (2026-09-08): negative result

Motivated by v18's per-solve traces (contour saturating its ×3 ceiling
on 19 % of fast-mission solves; yaw z averaging −0.39 against the
soften-only fence): `CONTOUR_UP` 3 → 4 ([125, 2000]), nominal
`mpc_w_att_y` 200 → 50 with a symmetric ×[0.25, 4] yaw fence
([12.5, 200]). Same v18 recipe otherwise (identified plant, widened DR,
wind), two seeds.

All-mission means (geometric RMSE, identified plant, drag-modelled
`w_rate 10`): nominal-yaw200 0.0280, **nominal-yaw50 0.0263** (−6.4 % —
the retune alone), v18 0.0250 (−15.6 % vs its yaw200 nominal),
v19 seed-0 0.0288 (−0.5 % vs yaw50 nominal), v19 seed-1 0.0260
(mean-of-ratios +9 %: large regressions on the small-RMSE slow rows).
Robustness matrices mixed-to-worse than v18 for both seeds.

Readings:
1. **The yaw finding is real but belongs to the constant tune**: yaw 50
   captures most of what v18's policy was doing dynamically, for free.
   Worth a flight-tested constant-tune change later.
2. **On the improved nominal the learned layer stops paying** (echoes
   v12: the residual mostly prices model/tune error; fix the tune and
   the residual thins). Two seeds failed to beat their own nominal.
3. v18's traces sit *inside* the v19 fences (contour sat 19 % → 9 %,
   yaw centred), yet neither seed converted the wider space into
   performance — the binding constraint is the training signal, not the
   fence.

Decision: **keep the v18 contract deployed** (map ×3/soften-only yaw,
nominal yaw 200, absolute best at 0.0250); v19's map constants and the
yaw-50 nominal are reverted. Checkpoints archived under
`target/cost_policy_v19{,b}`; if the yaw-50 tune is ever adopted for
flight, the policy must be retrained on that nominal (more seeds/steps)
or disabled.

### v20 — contour ceiling ×3 → ×4 on the v18 map (2026-09-08) — **deliverable**

Only change from v18: `CONTOUR_UP` 4 ([100, 2000] at the nominal 500);
yaw nominal and fence untouched. Two seeds, same recipe (identified
plant, widened DR, wind):

| seed | mean Δgeom (25 missions) | fast-only | absolute mean |
|---|---|---|---|
| v20 (seed 0) | −12.6 % | **+12.9 %** | 0.0281 |
| **v20b (seed 1)** | **−19.3 %** | **−9.8 %** | **0.0239** |
| v18 (reference) | −15.6 % | −5.5 % | 0.0250 |

Seed variance dominates config differences at 20 M steps — v20 seed 0
over-stiffens contour in the saturated regime and loses every fast row,
while seed 1 with identical fences posts the best checkpoint of the
study on this plant: **no regressing mission** (held-out
splits_timeopt −4.3 %, figure8_timeopt −20.6 %, super_fast −11.4 %),
and τ×2 / combined robustness comparable-to-better than v18
(circle_mid τ×2 0.32 vs v18's 0.84 divergence; residual weak cells
splits_timeopt combined +162 %, slalom_mid τ×2 +21 %). Contour rides
the ×4 ceiling on 24.5 % of solves.

**v20b is the baked deliverable** (`data/cost_policies/v20b.bin`,
`mpc_cost_policy: v20b`), superseding v18 whose ×3 map no longer
matches the code.
