# Mission Campaign, Tilt-Constraint-Free: 4 Stacks × All 14 Missions

*2026-08-13 · Follow-up to [mission_campaign_report.md](mission_campaign_report.md):
the same fixed-time MincoSnap mission grid, re-flown with the MPC tilt
fence **disabled** (`mpc_tilt_max_deg = 180` **and** `mpc_tilt_barrier_tau
= 0` — τ=0 is the exact off switch; the fence is a relaxed log-barrier
with a smooth gradient even away from the bound, so both knobs matter),
across four stacks:*

| case | outer loop | inner loop |
|---|---|---|
| `mpc_8k` | 10-state `mpc` @ 100 Hz | INDI @ 8 kHz |
| `mpc_1k` | 10-state `mpc` @ 100 Hz | INDI @ 1 kHz |
| `mpc_full_8k` | 13-state `mpc_full` @ 100 Hz | INDI @ 8 kHz |
| `mpc_full_1k` | 13-state `mpc_full` @ 100 Hz | INDI @ 1 kHz |

*`mpc_full` keeps the adopted body-rate barrier (τ=0.5/δ=0.5); the
reduced model keeps its input-space rate bounds — those are the vehicle's
own limits, not the tilt fence. Everything else matches the original
campaign: frozen `sim_baseline` vehicle (8.5 N motors), perfect IMU/rotor
telemetry, cross-product reference map, yaw = 0. Because the original
campaign's harness was a temporary patch, a **control grid** with the
adopted fence (60°/τ=0.5/δ=0.05) was re-run through this harness first —
it reproduces the report's Stack A / Stack B columns row-for-row (0.055,
0.112, 2.82, 3.04, 4.82, 0.168, 0.398, 1.30 …), isolating the fence
effect from harness-reconstruction noise. Two rows (super_slow,
drag-super_mid) differ from the report in both grids; treat this
harness's control column as the baseline there.*

## Headline

**Removing the tilt fence is free-to-positive for tracking on every
feasible mission, and decisively better on the two missions that
legitimately need >60° tilt — but it buys nothing on the infeasible
`_fast` class, and it removes the safety envelope, so the recommendation
is "raise the limit for aggressive missions", not "turn it off".
INDI rate (8 kHz vs 1 kHz) is irrelevant on the feasible set in this
perfect-sensor sim — except right at the fence, where 1 kHz + barrier
interacted badly.**

## Grid: no tilt constraint (rms / terminal [m], peak tilt)

| mission | mpc_8k | mpc_1k | mpc_full_8k | mpc_full_1k |
|---|---|---|---|---|
| indoor_splits_slow | **0.052 / 0.001** (31°) | **0.052 / 0.000** (31°) | **0.168 / 0.003** (29°) | **0.168 / 0.003** (29°) |
| indoor_splits_mid | **0.083 / 0.002** (70°)¹ | **0.083 / 0.001** (70°)¹ | 0.270 / 0.012 (61°)¹ | 0.270 / 0.012 (61°)¹ |
| indoor_splits_fast | 2.26 / 4.33 ✗ | 2.49 / 5.16 ✗ | 2.56 / 7.47 ✗ | 1.39 / 3.60 ✗ |
| outdoor_drag_slow | **0.040 / 0.001** (36°) | **0.040 / 0.001** (36°) | **0.125 / 0.004** (31°) | **0.125 / 0.005** (31°) |
| outdoor_drag_mid | **0.044 / 0.002** (52°) | **0.044 / 0.002** (52°) | **0.135 / 0.008** (43°) | **0.135 / 0.008** (43°) |
| outdoor_drag-large_mid | **0.064 / 0.002** (63°)¹ | **0.064 / 0.002** (63°)¹ | **0.170 / 0.009** (57°) | **0.170 / 0.008** (57°) |
| outdoor_drag-super_mid | **0.085 / 0.003** (71°)¹ | **0.085 / 0.003** (71°)¹ | 0.216 / 0.009 (69°)¹ | 0.216 / 0.009 (69°)¹ |
| outdoor_splits_slow | **0.065 / 0.001** (51°) | **0.065 / 0.001** (51°) | **0.216 / 0.008** (43°) | **0.216 / 0.008** (43°) |
| outdoor_splits_mid² | 2.70 / 0.002 ✗ | 2.14 / 11.0 ✗ | 1.28 / 6.2 ✗ | 0.76 / 4.1 ✗ |
| outdoor_splits-large_slow | **0.142 / 0.001** (66°)¹ | **0.142 / 0.001** (66°)¹ | 0.385 / 0.009 (62°)¹ | 0.385 / 0.009 (62°)¹ |
| outdoor_splits-large_mid³ | 2.39 / 10.6 ✗ | 2.58 / 10.1 ✗ | 0.72 / 4.5 ✗ | 0.71 / 4.5 ✗ |
| outdoor_splits-large_fast | 9.28 / 14.9 ✗ | 9.49 / 18.0 ✗ | 1.91 / 4.7 ✗ | 2.02 / 4.8 ✗ |
| outdoor_splits-super_slow | **0.194 / 0.001** (60°) | **0.194 / 0.001** (60°) | 0.507 / 0.004 (57°)¹ | 0.507 / 0.004 (57°)¹ |
| outdoor_splits-super_fast | 5.96 / 12.8 ✗ | 7.92 / 16.2 ✗ | 2.51 / 6.0 ✗ | 6.30 / 14.1 ✗ |

¹ *only* failure is the legacy 60° peak-tilt (or 0.5 m rms for full on
super_slow) scenario criterion — tracking itself is clean (report's C1
class, now tripped by design since tilt is unconstrained).
² 92° reference tilt → sim-harness cross-product-map singularity (C2);
not a fence effect, unchanged by this experiment.
³ 3.2 g vs the sim vehicle's 6.3 g ceiling (C3): 100 % motor saturation;
flyable only with the 12 N flight motors per the original campaign.

## Fence-off vs fence-on (same harness, control grid)

Rows where the fence was actually binding, `mpc_8k` / `mpc_full_1k`
(rms / terminal):

| mission | fence 60°/τ=0.5 | no fence | Δ |
|---|---|---|---|
| indoor_splits_mid (A) | 0.112 / 0.001 | **0.083 / 0.002** | −26 % rms |
| outdoor_drag-super_mid (A) | 0.227 / 0.011, sat 100 % | **0.085 / 0.003**, sat 61 % | −63 % rms, no clipping |
| outdoor_drag-super_mid (B) | 0.240 / 0.008, sat 89 % | **0.216 / 0.009**, sat 70 % | −10 % rms |
| outdoor_splits-large_slow (A) | 0.168 / 0.001 | **0.142 / 0.001** | −15 % rms |
| outdoor_splits-super_slow (A) | 0.208 / 0.001 | **0.194 / 0.001** | −7 % rms |
| outdoor_splits-large_mid (B, still ✗) | 1.30 / 6.0 | 0.71 / 4.5 | −45 % rms |

On missions demanding ≤ ~50° tilt the fence never binds and the numbers
are identical to the third decimal. Nowhere in the grid did removing the
fence make tracking worse.

## Findings

1. **The fence costs tracking exactly where the missions are aggressive
   but feasible.** These trajectories demand 61–71° of tilt; a 60° fence
   forces the solver to under-tilt, so it substitutes thrust-vector error
   with extra collective — visible as motor saturation (drag-super_mid A:
   100 % clipped with the fence, 61 % without) and 2–3× the rms. The
   worst case was fence + 1 kHz INDI on drag-super_mid: the
   barrier-induced clipping transient grew to a 96° excursion and
   1.13 m rms — the only place in either grid where 1 kHz INDI visibly
   underperformed 8 kHz.
2. **INDI rate is a non-factor on the feasible set** — every feasible
   mission's rms matches to ≤ 0.001 m between 8 kHz and 1 kHz, for both
   outer loops. Expected: the sim's IMU and rotor telemetry are perfect
   and the plant is disturbance-free, so the inner loop is never
   stressed; the original campaign's A-vs-B gap was the outer model and
   reference-completion gap, not the INDI rate. (Divergent missions
   differ chaotically between rates — those deltas carry no signal.)
   Hardware noise/disturbance rejection is where 8 kHz should matter;
   this sim cannot rule on that.
3. **The mpc vs mpc_full story is unchanged by the fence.** `mpc` stays
   ~2.5× more precise on the feasible set (0.04–0.19 vs 0.13–0.51 rms —
   the known Ω_r = 0 / u_r = hover reference-completion gap); `mpc_full`
   still degrades far more gracefully on the hardest missions
   (large_mid 0.71 vs 2.39; large_fast 2.0 vs 9.3 rms).
4. **Fence removal rescues nothing in the infeasible class** (C4: all
   `_fast`, super_fast) — those missions violate the vehicle's own
   10/10/6 rad/s rate limits and thrust ceiling, which both stacks still
   enforce (as they must). It also does not touch the
   outdoor_splits_mid C2 singularity, which is the sim's cross-product
   reference map, not a constraint.
5. **But the fence was never a tracking device.** Per the stress
   campaign, `mpc_full`'s SQP has a free-fall stationary point past
   ~100° tilt and diverges from deep-tilt *states*; the fence bounds
   what the controller *chooses* so disturbances start from a shallower
   worst case. Unconstrained peak tilts in this grid reached 71° on
   feasible missions (fine) and 103–179° on infeasible ones (exactly
   the divergence region).

## Recommendation

Set `mpc_tilt_max_deg` per mission class rather than globally 60°:
**~80° for the aggressive-but-feasible missions** (covers the 71°
demand with margin, keeps the fence between the vehicle and the >100°
divergence region), 60° for gentle/indoor work. Do not fly τ=0/180° —
it buys nothing beyond ~80° on feasible missions and gives up the
envelope. Also (again): the 60° scenario pass criterion needs
mission-scaled bounds — it is now the *only* failure on five
clean-tracking rows.

*Method: harness rebuilt as `crates/cybflight_sim/tests/mission_campaign_notilt.rs`
(mission YAML → firmware `plan_offline` MincoSnap recipe → horizon
setpoints; geofence = waypoint bbox ± 10 m; 120 s cap; 3 s terminal
hold). Sim controllers gained explicit INDI-rate constructors
(`from_params_at_indi_rate` / `with_options_at_rate`); default paths are
byte-identical — regression snapshot, autotest, stability and noisy
suites verified green. `CAMPAIGN_TILT=on` re-runs the control grid.
Caveats: sim-only, perfect sensors, single runs per cell, frozen 8.5 N
sim vehicle.*
