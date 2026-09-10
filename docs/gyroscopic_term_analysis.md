# Is the Gyroscopic Term ω×Iω Worth Keeping? — Numerical Verification

*2026-08-13 · Question: how much runtime/efficiency/performance could we
gain by ignoring the gyroscopic effect, as INDIflight and
`optimal_quad_control_RL` reportedly do — and is incorporating it truly
worth it?*

**Verdict: keep it. Removing it saves nothing measurable (≤0.3 %
solve time, inside run-to-run noise) and demonstrably degrades exactly
the regime the full-model NMPC exists for — multi-axis rates near the
body-rate limits, where the unmodeled term reaches 26–41 % of the rate
limit per 50 ms horizon knot and flipped one stress scenario from clean
completion to divergence.**

## 1. What the reference projects actually do

The premise "these projects ignore the gyroscopic effect" needs
sharpening — neither ignores it where *we* use it:

| Project | INDI incremental law | NMPC prediction model | (T,τ)→α decode |
|---|---|---|---|
| INDIflight (`src/main/flight/indi.c:419–427`) | **omits ω×Iω** — `dv = α_sp − doIndi·ω̇_f + doIndi·G2·ω̇_m` | has no NMPC | n/a |
| `optimal_quad_control_RL` (`nmpc/full_quad_model.py:130–132`, `simulate_nmpc_indi.py`) | omits it (same INDI) | **keeps ω×Iω** (Euler cross terms) | **keeps it**, "tracks the gyroscopic coupling between NMPC updates" |
| cybflight | omits it (INDIflight-derived — known accepted deviation for `indi: no`) | keeps it | keeps it (fresh gyro, IMU rate) |

The legitimate, well-supported claim is narrow: **the INDI *increment*
doesn't need an explicit gyroscopic feedforward**, because the measured
angular-acceleration feedback `ω̇_f` already contains the effect of
every torque acting on the body, gyroscopic included — INDI cancels it
as an unmodeled disturbance within its bandwidth. Our INDI inherits
this omission. (The RL project's sys-ID additionally absorbs *rotor*
gyroscopic residuals into fitted coefficients — a different, much
smaller term.) Neither project removes ω×Iω from a predictive model,
and the RL project — whose NMPC is the direct prototype of our
`FullQuadModel` — deliberately kept it in both remaining places.

## 2. Where it lives in cybflight, and its analytic magnitude

Three sites: (P1) `FullQuadModel::dynamics`/`dynamics_jac` — the NMPC's
prediction and the barrier's rate-row accuracy; (P2) the
`α = I⁻¹(τ_d − ω×Iω)` decode at IMU rate in `indi_task`/sim; (P3) the
INDI increment — already omitted, nothing to remove.

The term is quadratic in ω and vanishes for single-axis rotation.
Default model (I = diag(0.0025, 0.0021, 0.0043), limits (10, 10, 6) rad/s,
α authority ≈ (680, 607, 87) rad/s²):

| body rate [rad/s] | α-equivalent [rad/s²] | unmodeled Δω per 50 ms knot | % of torque authority |
|---|---|---|---|
| pure roll (10, 0, 0) | 0 | 0 | 0 % |
| moderate (2, 2, 1) | ≈ 1.8 | 0.09 rad/s (0.9 % of limit) | 0.4 % |
| aggressive (5, 5, 3) | ≈ 13 | 0.66 rad/s (6.6 %) | 2.7 % |
| at limits (10, 10, 6) | ≈ 53 | 2.6 rad/s (26 %) | 10.7 % |
| tumble start 1.25× | ≈ 82 | 4.1 rad/s (41 %) | 16.7 % |

Below ~3 rad/s it is invisible; at the rate limits it is a first-order
effect on the 50 ms knot prediction the barrier enforces against.

## 3. Experiment: runtime (E2)

Temporary `gyro_scale` patch: `0.0` skips the cross product, the matvec,
and the entire ∂α/∂ω Jacobian block (true dead-code removal, branch
overhead identical in both arms). Host bench, RTI mode, 2000 solves:

| run | full WITH [µs] | full WITHOUT [µs] | simple (no term at all) [µs] |
|---|---|---|---|
| 1 | 34.17 | 34.03 | 18.08 / 18.27 |
| 2 | 34.24 | 34.05 | 18.44 / 18.18 |
| 3 | 34.18 | 34.26 | 18.12 / 18.48 |

Distributions fully overlap; the simple solver — which contains no
gyroscopic term — moved more between identical runs than the with/without
delta. **Runtime gain ≤ 0.3 %, statistically indistinguishable from
zero.** The 8 kHz decode saves ~24 flops/tick ≈ 0.04 % of a 480 MHz
H743. This matches the cost structure: the term is ~30 flops inside a
solve dominated by the 13×13 Riccati sweep.

## 4. Experiment: closed loop with INDI, moderate rates (E3)

Controller-side term removed (model + decode); `QuadPlant` keeps true
physics. `mpc_full_indi` and `mpc_full_noindi`, four scenarios, clean +
MEMS-noise IMU: **every metric identical to print precision**
(e.g. p2p rms 0.053 m / terminal 0.005 m / peak tilt 14.5° both ways).
Peak rates here are ~2–4 rad/s — the ≈1 rad/s² model error is far below
what the 100 Hz outer loop corrects anyway. *This reproduces and
confirms the reference projects' "negligible influence" finding in the
regime they fly.*

## 5. Experiment: closed loop at high rates (E4)

Stress suite (raw NMPC, no INDI; plant keeps physics; the `gyro_scale=1`
control run reproduces the committed baseline bit-identically):

| scenario | config | baseline peak/viol% | gyroless peak/viol% | reading |
|---|---|---|---|---|
| tumble_arrest | off | 1.409 / 4.5 %, completes | **24.3 / 56.7 %, CRASH (51 m runaway)** | arrest torques mispredicted by ~80 rad/s² at start |
| knife_edge | ON | 0.960 / 0.0 %, ride 3.4 % | 1.071 / 0.5 %, ride 14.5 % | enforcement degraded: overshoots a boundary it previously held |
| dash_reversal | ON | 1.857 / 7.0 % | 1.538 / 10.6 % | shifted (this scenario is ulp-chaotic — weak evidence either way) |
| yaw_reversal | both | unchanged | unchanged | near-single-axis ⇒ ω×Iω ≈ 0, as predicted |
| metronome | both | unchanged | unchanged | moderate rates, as predicted |

Caveat honestly stated: this scenario family is chaotically sensitive to
model perturbations (see the stress-suite ulp finding), so single-run
deltas on dash are noise-grade. The tumble divergence and knife-edge
enforcement loss, however, are large, one-directional, and match the
analytic mechanism (multi-axis high rates ⇒ large unmodeled α).
tumble ON survives because the barrier collapses the rates — shrinking
the (quadratic) mismatch — before it can compound.

## 6. Conclusion

- **Runtime/efficiency gain from ignoring ω×Iω: ~zero.** ≤0.3 % of a
  solve (noise-bounded), 0.04 % CPU at the decode. There is nothing to
  win — the solve budget lives in the Riccati sweep, not the model.
- **Performance cost: zero at moderate rates, real at high rates.**
  The reference projects' claim is correct *for the INDI inner loop*
  (measured-ω̇ feedback absorbs the term — we already do this) and
  *for flight below ~5 rad/s* (our E3 confirms bit-level-negligible
  differences). It does not transfer to the NMPC prediction model at
  rate-limit utilization: there the term is 26–41 % of the limit per
  knot, degrades barrier enforcement, and removed an entire recovery
  in our experiment.
- **Keeping it is the free option.** The asymmetry decides: keep the
  term in the model and decode (status quo), keep omitting it in the
  INDI increment (status quo, per INDIflight). No change recommended.
