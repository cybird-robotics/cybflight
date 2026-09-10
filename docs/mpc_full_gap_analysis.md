# Why the Paper's NMPC+INDI Works and Ours "Fails" — Gap Analysis & Principled Fixes

*2026-08-13 · vs Sun et al., T-RO 2022 ([arXiv:2109.01365v6](https://arxiv.org/abs/2109.01365v6)), following the
stress campaign (`docs/mpc_stack_stress_campaign.md`).*

**Thesis: there is no magic in the paper we are missing. There is a
*discipline* they maintain (the NMPC only ever sees smooth, full-state +
input references from a planner) and a *task class* they avoid (upsets,
steps) — and their own data shows their system crashing exactly the way
ours does when either is violated. `mpc_full` is a high-performance
trajectory tracker that our stress campaign evaluated as an
upset-recovering planner. The fix is architectural discipline plus a
supervisor, not a different solver.**

## 1. The paper's system fails the same way — documented by the authors

The comparative study *includes* the failure regime we found:

- **Fig. 8**: under a 10 N lateral disturbance, "NMPC failed in
  converging and crashed the drone, while the DFBC method succeeds in
  recovering" — a memoryless cascaded controller recovers where the
  warm-started optimizer diverges. This is precisely our stack-A vs
  stack-B tumble result, in their lab.
- **Table V (with INDI)**: crash rates 9.3 % (+30 % mass), 18.7 %
  (−30 % thrust coefficient), 26.7 % (15 N force), **68 % at 50 ms
  estimation latency** (vs 6.7 % for DFBC).
- **Fig. 7**: ~30 % crash rate tracking infeasible references at
  a_max = 60 m/s², with the crash criterion (eq. 39) existing because
  crashes are expected.
- p. 10: "NMPC fails to converge in over 10 % of flights and crashes
  the drone… hybridizing INDI outer-loop with NMPC seems challenging
  owing to its non-cascaded nature." Fig. 10: raising Q_ξ beyond
  baseline *destabilizes* NMPC through actuator delay.

Their headline results (+48 % position accuracy over DFBC on
aggressive/infeasible references, 20 m/s flight) are earned **inside** a
regime: smooth references, latency < 30 ms, modest model error. Outside
it, their NMPC dies like ours. The architecture's fragility class is a
property of warm-started local NMPC, not of our port.

## 2. The regime match, verified on our stack

Our campaign already reproduces the paper's success in the paper's
regime: `mpc_full` + INDI (at 1 kHz *and* 8 kHz) passes every
smooth-trajectory scenario — hover, tilt-30°, p2p, mission square, with
MEMS/aggressive IMU noise. Every failure involved a step or upset the
paper never poses: tilt ≥ 60° starts, tumbles at ≥ 1.0× rate limits,
7 m/s step reversals. Two systems, same envelope boundary.

## 3. Real implementation gaps on our side (fixable, verified in code)

The paper's cost (eq. 10) tracks `‖x_k − x_{k,r}‖²_Q + ‖u_k − u_{k,r}‖²`
with **full** reference states and inputs from the planner. We differ:

| paper | cybflight today | consequence |
|---|---|---|
| Ω_r along horizon from differential flatness (eq. 18–20) | `fill_reference` writes **Ω_r = 0** | Q_Ω actively *penalizes* the body rates an aggressive trajectory requires |
| u_r feedforward thrusts from the planner | **u_r = hover**, constant | input cost biases toward hover thrust on aggressive segments |
| planner supplies jerk/snap for the flatness chain | `Setpoint` carries pos/vel/acc/yaw only | no data path for Ω_r without extending the sampler |

These caps our aggressive-tracking accuracy (their 20 m/s regime); they
are irrelevant to the upset failures (there the *reference itself* is
the problem).

## 4. Principled solutions

**S1 — Reference completion** (match eq. 10; unlocks the paper's
aggressive-tracking payoff): extend `Setpoint` with jerk (or finite-
difference acceleration along the horizon); populate Ω_r via the
flatness chain (we already own `trajectory_planning/flatness`); set
u_r = G1⁻¹·[T_r; τ_r] feedforward (minimally T_r/4 per motor with
T_r = m·‖ξ̈_r + g‖). Firmware + sim `fill_reference` both.

**S2 — Reference governor** (the solver is a tracker; guarantee it only
ever sees tracker-shaped problems): saturate the position error fed
into the cost (also closes the known f32 far-offset Riccati overflow);
blend the attitude reference from the *current* attitude toward the
trajectory attitude when the discrepancy is large; on gross lag,
re-time the reference rather than pulling harder. Steps never reach the
solver — the existing planner/sampler already provides the smooth path
for missions; the governor covers everything else.

**S3 — Supervisor / recovery gate** (the missing system layer; the
honest answer to upsets, which the paper scopes out): envelope monitor
on tilt/rates plus solver-health signature (non-finite, or
all-motors-at-lower-bound with tilt > 90°) → fall back to INDI rate
damping + reduced-attitude upright command — exactly the mechanism that
recovered every tumble for the reduced stack in our campaign and for
DFBC in the paper's Fig. 8 — → re-engage the NMPC with a fresh warm
start once inside the envelope. This converts every campaign divergence
into a recovery, by construction rather than by tuning.

**S4 — Operating budgets** (quantified by their Tables IV/V, Fig. 9–10):
end-to-end estimation latency **< 30 ms** — add an explicit measurement
to the Stage-6 bench checklist (their 50 ms → 68 % crashes is the
single scariest number in the paper for our mocap → ESKF → 50–100 Hz
pipeline); do not raise Q_ξ much beyond the baseline we inherited;
keep INDI on (their with/without-INDI tables halve the rotational crash
axes); keep the Ω constraints (they add them too, "beneficial for
stability" — our barrier is the differentiable version).

## 5. Verdict

`mpc_full` is not useless — inside the paper's regime our
implementation already matches its qualitative behavior at 1 kHz INDI,
and S1 is what stands between us and its quantitative aggressive-
tracking advantage. What the paper never solved — and what our campaign
measured — is life outside the reference tube. S2 shrinks how often you
leave the tube; S3 makes leaving it survivable. Order of work:
S3 (safety net first, enables edge testing), S2 (cheap, includes an
already-recommended fix), S1 (performance payoff), S4 (bench
discipline).
