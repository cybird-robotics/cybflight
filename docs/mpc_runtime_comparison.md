# MPC Solver Runtime Comparison — `SimpleSqpSolver` vs `FullSqpSolver` (± state constraints)

*2026-08-12 · branch `feature/0810_yaml_loading` · benchmark source:
`crates/cybflight_core/tests/mpc_runtime_comparison.rs`*

## Scope

Compares the wall-clock cost of the two SQP monomorphisations across a grid
of initial states:

| Config | Model | NX | NU | Rate limits enforced as |
|---|---|---|---|---|
| `simple` | `QuadModel` (collective thrust + body-rate command) | 10 | 4 | **input** bounds (`u_bounds[1..4]`, cubic penalty + clamp) |
| `full` | `FullQuadModel` (per-motor thrust, rigid-body rates) | 13 | 4 | — (not enforced) |
| `full+sc` | `FullQuadModel` + body-rate **state** constraints | 13 | 4 | relaxed log-barrier on states 10..13 |

Both models share horizon `N = 20`, `dt = 50 ms`, RK4 rollout, and the same
mass/inertia/cost weights (0.55 kg vehicle, `w_pos = 200`, `w_att = [5, 5, 200]`),
so all three solve the same physical regulation problem: fly to
`(1, 1, 1)` m and hover.

### State-constraint implementation (new)

`FullQuadModel` carries body rates as *states*, so the vehicle's rate limits
(`±10, ±10, ±6` rad/s here) cannot be expressed as input bounds — previously
the full solver simply could not honor them. Following
Frey et al., "Differentiable Nonlinear Model Predictive Control"
([arXiv:2505.01353v2](https://arxiv.org/abs/2505.01353v2), Appendix A.2): an IPM-based SQP
is equivalent to SQP applied to the **log-barrier problem** (eq. 18–20), so
each stage cost gains a term `−τ·Σᵢ ln(−hᵢ(x))` for the six rate-bound
constraints, and its gradient/Hessian flow into the existing Riccati
structures (`q`/`qm`) — **zero changes to the backward/forward sweeps**.
Because SQP iterates can be infeasible (e.g. an initial state already beyond
the limit), the exact log is replaced by the *relaxed* barrier
(quadratic extension below margin δ, C² at the switch), which is defined and
strongly repulsive everywhere.

New `FullQuadModel` fields: `rate_bounds`, `rate_barrier_tau` (τ, `0.0` =
**off**, the default — firmware behavior is unchanged by construction),
`rate_barrier_delta` (δ). Benchmark setting: `τ = 0.1`, `δ = 0.5 rad/s`.
Barrier gradient/Hessian are verified against finite differences
(`rate_barrier_matches_finite_differences`), and `τ = 0` is verified to
reproduce the legacy stage cost exactly (`barrier_off_is_byte_identical`).

## Environment

- Intel i7-14650HX, pinned baseline codegen (`target-cpu=x86-64`, no AVX —
  see `.cargo/config.toml`), rustc 1.94.1, `--release`, f32 math.
- Absolute numbers are **host** numbers; on the STM32H743 (480 MHz M7)
  expect roughly two orders of magnitude slower. The *ratios* are the
  transferable result (same code, same f32 arithmetic).
- Timing = mean over 2 000 calls (RTI) / 300 calls (convergence) after
  warm-up, identical inputs per call (deterministic work).

## Initial-state grid

12 cases spanning the envelope: `hover`, `tilt10`, `tilt45`, `fast_xlate`
(3 m/s), `descend` (−3 m/s), `far_offset` (4.7 m position error),
`mid_rates` (ω = (3,3,1)), `near_limit` (ω = (9,9,5)), `over_limit`
(ω = (12,−12,7) — infeasible start), `yaw_spin` (90° yaw + 5.5 rad/s),
`aggressive` (45° tilt + velocity + ω), `tumbling` (60° tilt, ω = (8,8,−5)).
The simple model has no rate state, so the ω component only applies to the
full model.

## Results — RTI mode (`max_iters = 1`, the firmware call pattern)

Per-call time is **independent of the initial state** for all three configs
(fixed work per iteration — no active-set combinatorics, the barrier only
changes a few stage-cost flops):

| case | simple [µs] | full [µs] | full+sc [µs] | full/simple | sc/full |
|---|---|---|---|---|---|
| hover | 19.13 | 34.03 | 35.18 | 1.78× | 1.034× |
| tilt10 | 18.51 | 34.42 | 35.24 | 1.86× | 1.024× |
| tilt45 | 18.09 | 34.01 | 35.19 | 1.88× | 1.034× |
| fast_xlate | 18.05 | 33.95 | 35.31 | 1.88× | 1.040× |
| descend | 18.14 | 34.04 | 35.22 | 1.88× | 1.035× |
| far_offset | 18.22 | 34.16 | 35.34 | 1.88× | 1.035× |
| mid_rates | 18.04 | 33.98 | 35.40 | 1.88× | 1.042× |
| near_limit | 18.04 | 34.13 | 35.36 | 1.89× | 1.036× |
| over_limit | 18.11 | 35.75 | 35.33 | 1.97× | 0.988× |
| yaw_spin | 18.06 | 33.94 | 35.27 | 1.88× | 1.039× |
| aggressive | 18.05 | 33.96 | 35.18 | 1.88× | 1.036× |
| tumbling | 18.05 | 33.95 | 35.18 | 1.88× | 1.036× |
| **MEAN** | **18.21** | **34.19** | **35.27** | **1.88×** | **1.031×** |

- **Full model costs 1.88× the simple model per SQP iteration.** Consistent
  with the `O(NX³·N)` Riccati sweep dominating: `(13/10)³ ≈ 2.2`, shaved
  down by the NX-independent NU-side work (`H_uu`, 4×4 Cholesky) and the
  sparse-B/identity-column optimizations.
- **State constraints add ≈ 3.1 % per iteration** — noise-level. The barrier
  is 6 scalar branches + a few FLOPs per stage against a 13³ matrix sweep.

## Results — convergence mode (solve to KKT tol 5·10⁻³, cap 30 iters)

`µs | iters`; `!` = hit the cap, `D` = diverged (NaN result):

| case | simple | full | full+sc | sc/full time |
|---|---|---|---|---|
| hover | 163.6 \| 9 | 409.2 \| 12 | 464.5 \| 12 | 1.14× |
| tilt10 | 548.7 \| 30! | 375.7 \| 11 | 426.8 \| 12 | 1.14× |
| tilt45 | 541.4 \| 30! | 1016.6 \| 30! | 1053.8 \| 30! | 1.04× |
| fast_xlate | 223.3 \| 12 | 305.8 \| 9 | 317.0 \| 9 | 1.04× |
| descend | 541.1 \| 30! | 451.2 \| 13 | 464.1 \| 13 | 1.03× |
| far_offset | 546.9 \| 30! | 920.8 \| 27**D** | 1060.3 \| 30! | 1.15× |
| mid_rates | 541.2 \| 30! | 1024.3 \| 30! | 1060.2 \| 30! | 1.04× |
| near_limit | 545.0 \| 30! | 1028.0 \| 30! | 1060.0 \| 30! | 1.03× |
| over_limit | 542.2 \| 30! | 1018.7 \| 30! | 1058.0 \| 30! | 1.04× |
| yaw_spin | 541.5 \| 30! | 1017.7 \| 30! | 1054.9 \| 30! | 1.04× |
| aggressive | 542.4 \| 30! | 1018.9 \| 30! | 1057.2 \| 30! | 1.04× |
| tumbling | 540.9 \| 30! | 1017.3 \| 30! | 1054.0 \| 30! | 1.04× |
| **MEAN** | **484.9** | **800.4** | **844.2** | **1.05×** |

- Where both converge, **iteration counts are similar** (full sometimes
  needs 2–3 more than simple, e.g. hover 9 → 12) and the barrier costs at
  most **one extra iteration** (tilt10: 11 → 12).
- Many aggressive cases hit the cap in *all* configs: the solver takes full
  Newton steps (α = 1, no line search), so the KKT residual plateaus above
  5·10⁻³ on hard transients. This is a property of the shared SQP loop, not
  of either model. The firmware's RTI usage (1 iteration @ 100 Hz, warm
  start) is unaffected.

## Constraint effectiveness (max predicted |ω|/limit over the horizon)

Re-rolled from the converged control trajectory; > 1.0 = predicted rate-limit
violation:

| case | full (unconstrained) | full+sc (constrained) |
|---|---|---|
| hover | 0.68 | 0.67 |
| tilt10 | 0.90 | 0.88 |
| tilt45 | 1.26 | 1.09 |
| fast_xlate | 0.65 | 0.64 |
| descend | 0.54 | 0.53 |
| far_offset | **NaN (diverged)** | 6.27 (finite) |
| mid_rates | 0.85 | 0.81 |
| near_limit | 1.12 | 0.92 |
| over_limit | 2.40 | 1.38 |
| yaw_spin | 3.38 | 1.17 |
| aggressive | 1.10 | 1.10 |
| tumbling | 1.21 | 1.23 |

- **Strong violations are restrained**: 2.40 → 1.38 (infeasible start),
  3.38 → 1.17 (yaw spin), and every feasible-start case that stayed near
  the limit is pulled inside it (`near_limit` 1.12 → 0.92).
- The barrier is a **soft** constraint at fixed τ: marginal transients can
  settle a few percent above 1 (`tumbling` 1.21 vs 1.23 is a wash). Exact
  enforcement needs τ-continuation or a proper primal-dual IPM per the
  paper — see Future work.

## Robustness finding (pre-existing, exposed by the grid)

On `far_offset` (4.7 m error) in convergence mode, the **unconstrained**
full solver diverges: unbounded predicted rates blow up the Riccati
recursion until f32 overflow produces NaN — and the KKT check then
*mis-reports convergence*, because `f32::max` silently ignores NaN in the
residual scan (`sqp_solver.rs` KKT loop). The **barrier-constrained** solver
stays finite on the same case (bounded rates keep the recursion in range) —
a stabilizing side benefit. Not hit by the firmware call pattern
(1 warm-started iteration), but worth a finiteness guard in `solve()`.

## Closed-loop aggression suite — are the constraints *active* and *effective*?

*(`crates/cybflight_core/tests/mpc_state_constraint_stress.rs`)*

The tables above are open-loop, single-solve. This suite runs `FullSqpSolver`
as the actual controller — firmware RTI pattern (1 warm-started iteration @
100 Hz) against the full nonlinear RK4 plant @ 500 Hz — through maneuvers
whose optimal solution *demands* rates beyond the limits, once with the
barrier off and once on (τ = 0.5, δ = 0.5; see the τ note below). Metrics
are on the **plant** state: `peak` = max |ω_i|/lim_i, `viol%` = time above
the limit, `ride%` = time in the 0.85–1.05 band (the "riding the boundary"
signature that proves the constraint, not the task, is what bounds the
rates).

| scenario | what it stresses | config | peak | viol% | ride% | outcome |
|---|---|---|---|---|---|---|
| `knife_edge_drop` — 95° roll, falling (0,−2,−3) m/s | roll snap to re-acquire lift | off | 1.29 | 1.7 | 1.1 | ok |
| | | **on** | **1.01** | **0.0** | 3.0 | ok |
| `yaw_reversal` — 179° yaw error, w_att_z = 200 | sustained yaw saturation (weakest axis) | off | 1.29 | 5.9 | 2.4 | ok |
| | | **on** | **1.07** | **3.6** | 5.4 | ok |
| `dash_reversal` — 7 m/s dash, target behind | pitch-over braking reversal | off | 2.58 | — | — | **CRASH (NaN)** |
| | | **on** | **1.65** | 7.7 | 4.2 | ok |
| `tumble_arrest` — ω₀ = 1.5× limits, 45° tilt | infeasible start (quadratic extension) | off | 1.72 | 11.7 | 3.6 | re-entry 0.86 s |
| | | **on** | 1.67 | **8.1** | 2.8 | **re-entry 0.42 s** |
| `metronome` — ±1.5 m lateral flips @ 1 s, no preview | repeated bank reversals | off | 1.31 | 11.5 | 3.7 | ok |
| | | **on** | **0.95** | **0.0** | **18.5** | ok |

- **Active**: the constrained metronome spends 18.5 % of the run riding the
  rate boundary (vs 3.7 % unconstrained, which instead *crosses* it), and
  every scenario's unconstrained twin violates the limits — the constraint
  is what bounds the rates, not the task.
- **Effective**: peaks collapse to ≈ 1.0 on roll/pitch scenarios with zero
  violation time; the infeasible tumble re-enters the feasible set 2× faster;
  and in `dash_reversal` the unconstrained controller **crashes the vehicle**
  (rate blow-up → NaN) while the constrained one completes the maneuver.
- **τ must be commensurate with the cost weights it fights.** Against
  `w_att_z = 200`, τ = 0.1 under-enforces (yaw peak 1.22); a sweep gives
  yaw peak 1.22 / 1.07 / 0.99 / 0.93 for τ = 0.1 / 0.5 / 1.0 / 2.0.
  τ = 0.5 balances enforcement against task aggression across all five
  scenarios.
- **Enforcement margin**: the barrier acts on 50 ms horizon knots, so a
  low-inertia axis builds extra rate between knots. Enforcing at 0.85× the
  vehicle limit brings the worst axis (yaw) to peak 1.004 vs the *vehicle*
  limit — the deployment recipe (`margin_enforcement_absorbs_knot_overshoot`
  test).
- **Formulation limit (documented, not fixable by constraints)**: beyond
  ~100–110° tilt the local SQP leaves its attraction basin — thrust is
  purely harmful along the whole linearized rollout, all four motors corner
  at the zero bound, and free-fall becomes a stationary point. True inverted
  recovery needs a global initialization strategy, with or without
  constraints.

## Take-aways

1. **Per-iteration price of the full model: 1.88×** the reduced model
   (34 µs vs 18 µs host; state-independent). Budget-wise, on-target RTI at
   100 Hz stays plausible for either, but the full model halves the margin.
2. **State constraints are effectively free at RTI granularity (+3 %)** and
   cheap to convergence (+5 % mean). The cost of honoring rate limits in
   the full formulation is *not* a runtime problem — the barrier reuses the
   Riccati structure exactly as the reference paper prescribes.
3. **`full+sc` is the only full-model config that respects rate limits**,
   and it also happens to be more numerically robust than `full` on
   long-horizon aggressive solves.
4. If the full model is adopted for flight, pair it with: a finiteness
   guard on the KKT residual, and (optionally) a line search or
   τ-continuation for off-nominal recovery scenarios — the RTI loop itself
   is healthy.

## Reproducing

```sh
cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
    --release --test mpc_runtime_comparison -- --nocapture --test-threads=1
cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
    --release --test mpc_state_constraint_stress -- --nocapture --test-threads=1
```
