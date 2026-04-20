Bench-only trial-and-error INDI tuning (no logs)

Work from the known-stable point outward. Change one knob  
 at a time, always return to a known-good baseline between  
 experiments, and use a test stand or tethered hover so a  
 bad setting doesn't cost props.

Phase 0 — Baseline

1. Lock in your known-stable config: sync_filter_hz = 5,  
   rate_gains = 20, SG window=7 order=2, motor_time_const_s at whatever it
   currently is. Confirm a clean tethered hover. This is your "revert point."
2. Hover, give small roll/pitch stick steps (~10°). Note: stick response
   latency, bounce-back, any visible wobble.  
   This is the reference feel.

Phase 1 — Find the real actuator bandwidth

Goal: figure out whether 5 Hz is hardware-limited or  
 pipeline-limited.

3.  With baseline config, drop rate_gains to ~10 (halve).  
    Retest. If it still tracks and feels similar but softer, your gains are not
    the limit. If it goes sluggish and the  
    oscillation threshold in step 4 moves, you were  
    gain-limited.
4.  Keeping rate_gains = 10, sweep sync_filter_hz upward: 5 → 7 → 9 → 12 → 15.
    Find the new oscillation knee.
    - If the knee moves up significantly (e.g. from 7.5 to 15 Hz), the original
      limit was loop gain / noise amplification, not actuator physics. Stop
      chasing the  
       actuator and go to Phase 2.
    - If the knee stays near 5–7 Hz regardless of rate_gains, the actuator or
      measurement pipeline really is that slow. Go to Phase 3.

Phase 2 — Gain-limited case

5. Return to stable sync_filter_hz (say 10 Hz, one step below the new knee). Now
   sweep rate_gains upward: 10 → 15 → 20 → 25 until response sharpens without
   bounce-back on stick release. Stop one step before oscillation returns.  
   This is your authority limit.
6. Fine-tune sync_filter_hz down by 1–2 Hz from the oscillation knee for margin.
   Done.

Phase 3 — Pipeline/actuator-limited case

Attack the suspects in order of cost/risk:

7. SG window. Widen to 11, then 13, keeping order=2. Each step does more
   in-filter smoothing, so the post-biquad has less work and you can retest the
   sync_filter_hz knee. If the knee moves up as you widen SG, noise
   amplification was the limit — pick the window one step before the knee stops
   improving.
8. SG decimated rate. Raise sg_target_rate_hz from 1000 →
9. Shrinks SG group delay in absolute time. Retest the knee. Improvement = delay
   mismatch was a contributor.
10. Motor time constant. Try motor_time_const_s at 0.015, 0.010, 0.008 (bracket
    of "faster than you think"). For  
    each, retest the knee. The correct value is the one where the knee is
    highest and stick response feels crisp without overshoot. Too-low τ will
    feel twitchy and overshoot on  
    stops; too-high τ will feel mushy.
11. Gyro pre-filtering. If you have an upstream LPF/notch  
    before the INDI task, temporarily lower its cutoff/widen notches and retest.
    If the knee changes, upstream lag is part of your sync-filter budget.

Phase 4 — Stick-feel validation

11. At your new settings, do this sequence on each axis: 30° snap roll, full
    stop; small continuous stirs; throttle punch-out at an angle. You're
    watching for:
    - Bounce-back after stop → sync_filter_hz still too high

or motor_time_const_s too low

- Mushy, delayed stop → sync_filter_hz too low, or  
  rate_gains too low
- Low-freq wobble during stir → sync matching is off, not a gain issue
- HF buzz under throttle punch → noise amplification,  
  widen SG or lower cutoff

12. Back off sync_filter_hz by ~20% from the oscillation  
    knee as your safety margin for battery sag, flight-time  
    warming, and prop wear.

Rules of thumb while tuning

- One knob per flight. If two changed, you don't know which helped.
- Always test at hover weight with the flight battery you'll use — current sag
  changes effective τ.
- Props matter more than you'd guess. A worn or chipped prop can single-handedly
  move the knee by 30%.
- If nothing you do lifts the knee above ~7 Hz, the  
  airframe (prop/motor/ESC combo) is genuinely that slow. Accept 5–6 Hz and tune
  rate_gains around it.
