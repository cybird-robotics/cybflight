# Stress Campaign: `mpc`+INDI vs `mpc_full`+INDI — Where Does Each Fail?

*2026-08-13 · Closed-loop sim campaign (real `QuadPlant`: rotor lag, thrust
curve, drag, rotor gyroscopics; ESKF in loop for GPS cases). Both stacks
with the adopted state constraints active: tilt fence 60°/τ=0.5, and for
the full model the body-rate barrier τ=0.5. Stack A = reduced 10-state
`mpc` + INDI @ 8 kHz. Stack B = 13-state `mpc_full` + INDI @ 1 kHz (the
stated firmware architecture; 8 kHz control runs included).*

**Headline: Stack A never diverged in any of 18 scenarios — worst-case
terminal error 9.3 cm. Stack B fails the entire aggressive family (tilt
≥ 60° starts, tumbles ≥ 1.0× rate limits, 5–7 m/s dash reversals),
flipping past 100–180° tilt with meters of error — and the failures are
architectural: robust to INDI rate (1 vs 8 kHz), tilt-fence strength
(τ 0.5–5.0), and constraints on/off (stock fails identically).**

## Nominal envelope — both stacks pass everything

| scenario | A: rms / terminal | B: rms / terminal |
|---|---|---|
| hover_level | 0.008 / 0.003 m | 0.005 / 0.002 m |
| hover_tilt30 | 0.019 / 0.008 m | 0.065 / 0.028 m |
| p2p_x3 | 0.016 / 0.001 m | 0.054 / 0.004 m |
| mission_square | 0.034 / 0.001 m | 0.110 / 0.003 m |
| + MEMS / aggressive IMU noise | unchanged | unchanged |

A tracks ~3× tighter. IMU noise (σ_gyro up to 0.1 rad/s) moves neither —
INDI's filtering absorbs it. `mpc_full` + **1 kHz** INDI is fully stable
in this envelope: the paper's rate claim holds where it was made.

## Stress family — the stacks diverge (literally)

| case | A (mpc + INDI 8k) | B (mpc_full + INDI 1k) |
|---|---|---|
| tilt-60° start | recovers, 1.7 cm terminal | **diverges**: 138° tilt, 4.1 m |
| tilt-80° start | recovers, 2.5 cm terminal | **diverges**: 139°, 1.1 m |
| tumble 1.0× limits, level | recovers, 5.6 cm, peak 54° | **diverges**: 149°, 2.5 m |
| tumble 1.25× limits, level | recovers, 7.2 cm, peak 70° | fails: 105°, 1.2 m |
| tumble 1.25×, 30° tilt | recovers, 4.7 cm | **diverges**: 161°, 0.9 m |
| tumble 1.5× limits | recovers, 9.3 cm, peak 78° | fails: 95°, 0.17 m (basin luck) |
| dash reversal −5 m/s | task completed, 0.1 cm terminal (transient tilt 75°) | **flips**: 176°, 2.3 m |
| dash overspeed +7 m/s | task completed, 0.1 cm terminal (transient tilt 84°) | **flips**: 177°, 2.1 m |
| tumble 1.25× + MEMS noise | recovers, 4.7 cm | **diverges**: 133°, 0.9 m |

Stack A's "Fail" verdicts in raw output are the hover scenario's 60°
peak-tilt criterion — frequently tripped by the *initial condition*
itself (tilt-80° start) or by dash-braking transients; terminal errors
show every task completed. Note the fence is soft in stack A too:
dash reversals exceed 60° by 15–24° transiently.

## Isolation experiments — why Stack B fails

1. **Not the INDI rate.** 8 kHz reruns of every failing case fail the
   same way (tumble 1.0×: 1.48 m @ 8 kHz vs 2.53 m @ 1 kHz — marginally
   better, same divergence).
2. **Not the constraints.** Tilt-fence τ swept 0.5 → 5.0: outcomes
   shuffle chaotically (tumble improves at τ=2, dash worsens), none
   pass. Stock (no constraints): fails identically. The constraints
   neither cause nor cure this family.
3. **Root cause is the established one**: aggressive transients carry
   attitude past the ~100° free-fall stationary point of the local SQP
   (all motors corner at zero; converged, not iteration-starved), and
   INDI then faithfully executes the divergent α. The earlier single-run
   result where a τ=1.0 fence "rescued" the 1.0× tumble does **not**
   reproduce across this campaign's perturbations — it was
   attraction-basin luck, consistent with the family's documented
   ulp-level sensitivity. That claim is retracted.

## Estimator-in-loop findings (both stacks, pre-existing)

- GPS sbas (σ = 0.5 m): both stacks "fail" with ~0.32 m terminal error —
  good estimation under that noise, but over the clean-scenario 0.15 m
  criterion. Calibration artifact, not divergence.
- GPS degraded (σ = 2 m): both runs abort on `geofence_violation` — the
  estimate wanders a 3 m mission out of the fence. A real operational
  fact: degraded GPS + tight geofence = mission abort, by design.

## Conclusions

1. **Safe envelope of `mpc_full`+INDI**: tilt ≲ 45°, rates inside
   limits, moderate speeds. Inside it, at either INDI rate, it is stable
   and clean (and 1 kHz costs nothing there). Outside it, it fails
   robustly — no tuning of the adopted constraints changes that.
2. **`mpc`+INDI@8k is the robust configuration**: nothing in this
   campaign made it diverge — 1.5× tumbles, 80° starts, 7 m/s
   reversals, noise, estimator in loop.
3. The constraints are **envelope hygiene, not crash protection**: they
   bound what the controller chooses within its valid region; they do
   not enlarge that region.
4. The **rate/attitude-threshold recovery gate** (fall back to INDI
   rate damping until re-entry) is now the single highest-value missing
   piece before `mpc_full` flies anything aggressive — it is exactly the
   mechanism stack A demonstrates in every recovered row above.

*Caveats: sim-only; single runs per cell from a chaotically sensitive
family (the multi-config sweeps, not individual cells, carry the
conclusions); stack A's 60°-criterion "Fail"s were re-read as recoveries
by terminal error, stated explicitly above.*
