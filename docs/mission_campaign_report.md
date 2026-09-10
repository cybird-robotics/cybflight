# Mission Campaign: Both MPC Stacks Across All 14 `missions/` Trajectories

*2026-08-13 · Every mission YAML flown in closed-loop sim through the
firmware's fixed-time MincoSnap path (waypoints + timestamps →
degree-7 trajectory, sampled as horizon setpoints). Stack A =
`mpc` + INDI @ 8 kHz; Stack B = `mpc_full` + INDI @ 1 kHz; both with the
adopted constraints (tilt fence, rate barrier for B). Plant =
`QuadPlant` (rotor lag, thrust curve, drag) on the frozen `sim_baseline`
vehicle (8.5 N motors ⇒ 6.3 g ceiling; the flight vehicles carry 12 N ⇒
8.9 g).*

## Headline results

**Every mission both stacks "failed" decomposes into four identified
causes — none of which is a stack deficiency in the paper's operating
regime.** On the genuinely flyable set, both stacks track cleanly; on
the aggressive-but-feasible missions, Stack B (mpc_full) *outperforms*
Stack A for the first time in this project — reproducing the paper's
claimed NMPC advantage exactly where it predicts it.

### Raw grid (adopted constraints, sim vehicle)

| mission | demand (peak g / v / ref-rate) | A rms/terminal [m] | B rms/terminal [m] |
|---|---|---|---|
| indoor_splits_slow | 1.3 g / 4.0 / 1.2 | **0.055 / 0.001** ✓ | **0.172 / 0.003** ✓ |
| indoor_splits_mid | 2.2 g / 7.2 / 8.4 | **0.112 / 0.001** ✓ | 0.280 / 0.011 (60° criterion) |
| indoor_splits_fast | 4.7 g / 11.3 / 11.8 | 2.82 / 5.23 ✗ | 6.85 / 14.2 ✗ |
| outdoor_drag_slow | 1.2 g / 4.9 | **0.043 / 0.001** ✓ | **0.128 / 0.004** ✓ |
| outdoor_drag_mid | 1.5 g / 6.6 | **0.050 / 0.001** ✓ | **0.140 / 0.007** ✓ |
| outdoor_drag-large_mid | 1.9 g / 11.0 | **0.080 / 0.002** ✓ | **0.175 / 0.007** ✓ |
| outdoor_drag-super_mid | 2.5 g / 18.6 | 0.166 / 0.368 (criterion) | 0.372 / 0.814 (criterion) |
| outdoor_splits_slow | 1.7 g / 5.5 | **0.072 / 0.001** ✓ | **0.221 / 0.007** ✓ |
| outdoor_splits_mid | 2.2 g / 7.3 / **92° ref tilt** | 3.04 ✗ → **0.088** with tilt-yaw map | 0.97 ✗ → 0.30 with tilt-yaw map |
| outdoor_splits-large_slow | 2.0 g / 12.2 | **0.168 / 0.001** ✓ | **0.398 / 0.009** ✓ |
| outdoor_splits-large_mid | 3.2 g / 16.7 / 12.8 rad/s | 4.82 ✗ → 4.27 (12 N, transient loss) | 1.30 ✗ → **0.399 / 0.016** (12 N) |
| outdoor_splits-large_fast | 5.4 g / 20.4 / 11.6 | ✗ (all configs) | ✗ (all configs) |
| outdoor_splits-super_slow | 1.8 g / 15.6 | 0.109 / 0.173 (criterion) | 0.300 / 0.468 (criterion) |
| outdoor_splits-super_fast | 5.1 g / 31.4 / **122° ref tilt** | ✗ | ✗ |

## The four causes, isolated experimentally

**C1 — Verdict-criteria artifacts** (indoor_splits_mid, super_slow,
drag-super_mid): tracking is fine (rms 0.08–0.37 m); the default
scenario criteria (60° peak tilt, 0.15 m terminal) predate aggressive
missions. Calibration, not control.

**C2 — Sim-harness flatness singularity** (outdoor_splits_mid 92°,
large_mid 90°, super_fast 122° reference tilt): the sim controllers
hardcode the cross-product attitude-reference map
(`USE_TILT_REFERENCE_QUATERNION = false`), which is singular at 90°
tilt. Switching to the tilt-yaw map — **the firmware mission path's
default** (`flatness_map: tilt_yaw`, robust through 90°) — fixed
outdoor_splits_mid outright (A: 3.04 → 0.088 m rms). **The deployed
firmware is not exposed to this**; the sim's fidelity gap is: sim
controllers ignore the mission's `flatness_map`. Worth fixing in the
sim when missions are wired in permanently.

**C3 — Sim vehicle ≠ flight vehicle** (outdoor_splits-large_mid at
3.2 g): the frozen `sim_baseline` has 8.5 N motors; the missions were
planned against flight vehicles with 12 N. With 12 N (and the tilt-yaw
map), Stack B tracks large_mid at **0.399 m rms over 22 s at
16.7 m/s** — while Stack A suffers a transient trajectory loss (4.3 m
rms, recovering to 0.002 terminal). This is the paper's regime where
NMPC beats the cascaded stack, reproduced: the reduced stack's
rate-command clamp saturates in sustained 3 g cornering and position
error runs away; the full model plans within the constraint instead.

**C4 — Genuinely infeasible references** (all `_fast` missions,
`super_fast`): reference tilt-rate demand 10.1–12.8 rad/s **at or
beyond the vehicle's own 10/10/6 rad/s limits** (which both stacks
enforce, by design), combined with 4.7–5.4 g against a 6.3 g sim
ceiling (thin margin even at 8.9 g), and for super_fast a 122°
reference tilt at 31 m/s. No controller that honors the vehicle limits
can track these on this vehicle — this is the paper's "dynamically
infeasible" class, where their own crash rates reach 30 %. Both stacks
degrade rather than track; neither NaN'd or hit the geofence runner
abort in a way the other avoided.

## Analysis

1. **On feasible missions both stacks work; A is ~2.5× more precise**
   (0.04–0.17 m vs 0.13–0.40 m) — consistent with every prior grid.
   The mpc_full accuracy deficit is the known reference-completion gap
   (Ω_r = 0, u_r = hover; gap analysis S1) plus its 100 Hz replanning
   against INDI's 8 kHz tracking in Stack A.
2. **On the hardest feasible mission, B wins decisively** — the first
   empirical payoff of mpc_full in this project, and it appears exactly
   where the paper predicts (sustained high-g maneuvering at actuator
   limits). This is the mission class that justifies keeping mpc_full.
3. **The `_fast` missions are not flyable on the sim vehicle by any
   controller respecting the rate limits.** Before flying them on
   hardware: re-check them against the flight vehicle's actual limits —
   a reference that demands 12.8 rad/s against a 10 rad/s limit is
   infeasible for the *real* vehicle too. Recommend a bake-time
   feasibility check: sample the MincoSnap trajectory's demanded
   tilt-rate and thrust against the target vehicle's
   `max_rate_rad_s` / thrust ceiling and warn (the demand probe built
   for this campaign is exactly that check).
4. **Sim-fidelity debts surfaced**: (a) the sim controllers should
   honor the mission `flatness_map` (tilt-yaw default) instead of the
   singular cross-product map; (b) scenario pass criteria need
   mission-scaled bounds; (c) a permanent `--mission` harness (this
   campaign's temporary one) would make all of this regression-testable.

*Caveats: sim-only; the 12 N runs change plant and controller
consistently but are not the frozen baseline; single runs per cell.
The mission harness, tilt-yaw switch, and thrust override were
temporary experiment patches, reverted after the campaign; suites
verified green.*
