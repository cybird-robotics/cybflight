# Mission-Completion Campaign: All 14 Missions × 4 Stacks, Tilt Constraints Off

*2026-08-13 · Goal: every controller completes every `missions/*.yaml`
without crash/failure, tilt constraint disabled. Follow-up to
[mission_campaign_notilt.md](mission_campaign_notilt.md) and
[mpc_full_crash_diagnosis.md](mpc_full_crash_diagnosis.md).
Harness: `crates/cybflight_sim/tests/mission_campaign_completion.rs`.
"Completed" = no geofence/ground/NaN exit AND terminal error < 0.15 m.*

## Result: 46/56 cells complete — 11/14 missions on all four stacks

Starting point (tilt off only): 9/14 missions all-stacks, with ground
crashes across the aggressive tier. Three principled fixes, applied
uniformly to all four stacks, raised it to 11/14 all-stacks and 12/14
for the mpc stacks — the three residual failures are characterized
below (one architectural, two beyond the sim vehicle's physics).

### The fixes (all opt-in fields on the sim controllers, defaults frozen)

1. **Honor the mission's `flatness_map: tilt_yaw`** (`use_tilt_map`) —
   all 14 missions declare it; the sim's hardcoded cross-product map is
   singular at 90° tilt (campaign cause C2). Fixed outdoor_splits_mid
   for the mpc stacks outright (3.04 → 0.070 rms).
2. **S1 reference completion via the flatness chain**
   (`complete_references`): Ω_r and u_r computed analytically from
   (acc, jerk) via `flatness_to_thrust_omega` (paper eq. 18–20;
   `Setpoint` gained a `jerk` field from the degree-7 trajectory).
   *Both* stacks: the reduced model's u_ref becomes `[T_r, Ω_r]`, the
   full model's x_ref rates + per-motor u_ref get filled. This fixed
   indoor_splits_fast for mpc (crash → **0.112 rms**) and
   outdoor_splits-large_mid for mpc_full (crash → **0.345 rms**), and
   cut rms 15–55 % on every already-passing aggressive mission.
   ⚠ An earlier finite-difference Ω_r (differencing consecutive
   q_refs) *destabilized* the 92°-tilt mission — the 50 ms stride
   aliases fast attitude flips into strong-but-wrong rate references.
   The analytic chain is the correct construction.
3. **Controller-side envelope matched to the missions**: rate limits
   [10,10,6] → [16,16,8] rad/s (`max_rate_rad_s` is not read by the
   plant; the fast tier demands 10.1–13.7 rad/s) and `thrust_frac`
   0.75 → 1.0 (0.75 capped the reduced stack at 4.7 g vs the 5.4 g
   demand). Tilt fence off (τ=0); mpc_full keeps the body-rate barrier.

Also built and validated, needed only in the hardest regime (kept as
knobs): **S2 error-saturation governor** (`err_sat_m`) and a
**thrust floor** (`set_thrust_floor`) that denies the solver the
zero-thrust free-fall corner from the crash diagnosis.

### Final grid (8.5 N sim vehicle; rms / terminal [m])

| mission | mpc_8k | mpc_1k | mpc_full_8k | mpc_full_1k |
|---|---|---|---|---|
| indoor_splits_slow | ✓ 0.052/0.001 | ✓ 0.052/0.000 | ✓ 0.131/0.002 | ✓ 0.131/0.002 |
| indoor_splits_mid | ✓ 0.070/0.002 | ✓ 0.070/0.002 | ✓ 0.167/0.008 | ✓ 0.167/0.008 |
| indoor_splits_fast | ✓ 0.112/0.013 | ✓ 0.113/0.013 | ✗ | ✗ |
| outdoor_drag_slow | ✓ 0.039/0.001 | ✓ 0.039/0.001 | ✓ 0.091/0.007 | ✓ 0.091/0.007 |
| outdoor_drag_mid | ✓ 0.037/0.003 | ✓ 0.037/0.002 | ✓ 0.072/0.010 | ✓ 0.072/0.010 |
| outdoor_drag-large_mid | ✓ 0.058/0.003 | ✓ 0.058/0.003 | ✓ 0.112/0.012 | ✓ 0.112/0.011 |
| outdoor_drag-super_mid | ✓ 0.088/0.004 | ✓ 0.088/0.004 | ✓ 0.168/0.012 | ✓ 0.168/0.011 |
| outdoor_splits_slow | ✓ 0.061/0.001 | ✓ 0.061/0.001 | ✓ 0.146/0.005 | ✓ 0.146/0.005 |
| outdoor_splits_mid | ✓ 0.070/0.002 | ✓ 0.070/0.002 | ✓ 0.173/0.008 | ✓ 0.173/0.008 |
| outdoor_splits-large_slow | ✓ 0.137/0.001 | ✓ 0.137/0.001 | ✓ 0.320/0.005 | ✓ 0.320/0.005 |
| outdoor_splits-large_mid | ✓ 0.156/0.003 | ✓ 0.156/0.002 | ✓ 0.345/0.008 | ✓ 0.345/0.008 |
| outdoor_splits-large_fast | ✗¹ | ✗¹ | ✗¹ | ✗¹ |
| outdoor_splits-super_slow | ✓ 0.190/0.001 | ✓ 0.190/0.001 | ✓ 0.464/0.002 | ✓ 0.464/0.002 |
| outdoor_splits-super_fast | ✗¹ | ✗¹ | ✗¹ | ✗¹ |

¹ Demand exceeds the vehicle: see below.

## The three residual failures, characterized

**outdoor_splits-large_fast / super_fast — physically infeasible on the
sim vehicle.** The demand probe (trajectory thrust demand incl. rotor
drag vs the 34 N ceiling) reads **102 % and 107 %**. No controller can
complete them on 8.5 N motors. On the flight vehicles' **12 N** motors
(demand 73–75 %, plant+controller changed consistently):
**large_fast completes on both mpc_full stacks** (0.39–0.51 rms, with
S2 + thrust floor) — while mpc still fails it (sustained 100 %
saturation runaway at 13.3 rad/s + near-ceiling thrust; the paper's
NMPC-advantage regime, again). super_fast (31 m/s, 122° ref tilt,
5.1 g) completes on nothing — the paper's own crash-rate-30 % class.

**indoor_splits_fast × mpc_full — architectural.** The mission is
feasible (81 % ceiling, 13.7 rad/s < limits) and mpc completes it at
0.112 rms. mpc_full fails under *every* configuration tried: solve rate
100→400 Hz, horizon dt 50→15 ms, S1, S2, thrust floor, yaw weight ÷10,
rate barrier off. Traces show why: the mission head whips to ~10 rad/s
within 40 ms of a standing start (min-snap through a steep speed ramp);
the reduced stack's INDI closes a **rate feedback** loop at 8 kHz and
brakes the whip reactively (its trace: ±14 rad/s swings, error held
≤ 0.19 m), while mpc_full's inner loop is feedforward torque —
`(T_d, τ_d)` held between solves, **no rate error term by design**
(paper eq. 32 topology). Implicit rate feedback exists only at the
solve rate, and even 400 Hz is not 8 kHz. This is the same
step/upset-class fragility the stress campaign measured and the paper
itself documents (Fig. 8: NMPC diverges under disturbance where DFBC
recovers). Fixing it means adding a rate-feedback term to the inner
loop — a topology change to evaluate deliberately, not a tuning knob.

## Takeaways

1. **The two stacks fail in complementary regimes**: mpc dies in
   sustained saturation (large_fast @12 N) where mpc_full plans within
   the limits; mpc_full dies on step-class attitude transients
   (indoor_fast) where mpc's 8 kHz rate loop absorbs them. This is the
   paper's comparative result, reproduced end-to-end in our sim.
2. **Reference completion (S1) is the single highest-value fix** and is
   now validated for *both* stacks — implement it in the firmware
   `fill_reference` (needs jerk in the mission sampler output; the
   flatness function already exists in `cybflight_core`).
3. The sim controllers should keep honoring the mission `flatness_map`
   (this campaign's `use_tilt_map`) — promote from experiment flag to
   the mission harness default.
4. INDI rate (8 kHz vs 1 kHz) remained a non-factor on every completed
   mission (identical rms to 3 decimals; perfect-sensor sim).
5. A bake-time feasibility check stays essential: the demand probe
   flags large_fast/super_fast as over-ceiling for a given vehicle
   before anyone flies them.

*Method: all controller changes are opt-in fields defaulting to frozen
behavior — regression snapshot + all sim suites verified green.
Experiments preserved in the harness (`splits_mid_full_matrix`,
`fast_mission_solve_rate`, `thrust_infeasible_on_12n`, `demand_probe`,
`focus_experiments`). Sim-only, perfect sensors, single runs per cell.*
