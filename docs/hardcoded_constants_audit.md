# Hard-coded constant audit

A sweep of every numeric literal and `const` in the workspace, sorting
them into *should be a parameter*, *cannot be a parameter*, and *must
not be a parameter*. The first group was largely acted on in schema
`VERSION` 54; the rest is recorded here so the reasoning does not have
to be rediscovered.

The question this audit asks of each constant is the same one the
configuration plane asks of each parameter: **what would have to change
for this value to be wrong?** If the answer is "a different tune, site,
pack or transmitter", it belongs in the registry. If it is "a different
build of the firmware", it is sizing. If it is "nothing — it is
arithmetic, a datasheet, or a wire format", it stays a constant.

## Classification

| Class | Meaning | Action |
|---|---|---|
| **Runtime** | A tune, threshold or policy. No memory-layout or invariant coupling. | Make it a parameter. |
| **Sizing** | Sizes an array, a `static`, a const generic, or a channel. | Must stay `const` — architecture rule 7. |
| **Physical / protocol** | A datasheet value, unit conversion, register, wire format, or a numerical-stability epsilon. | Must stay `const`. |
| **Coupled** | Runtime-shaped, but tied to another constant or to a safety invariant. | Only with its partner, and only with a bound. |

## What changed in v54

Every value below now defaults to exactly what it was, which is why the
sim regression snapshot is unchanged. Names are registry keys.

### Parameters that existed but reached nothing

These are the sharpest findings: three keys were baked into every
vehicle YAML, documented, and range-validated, yet no code read them.

- **`rc_min_us` / `rc_mid_us` / `rc_max_us`.** The flown stick
  calibration was `988 / 1500 / 2012`, written as constructor literals
  in `ChannelCalibration`. A pilot who changed the endpoints moved the
  parameter and not the aircraft. Both the rate-mode and position-mode
  paths in `rc_interpreter` now build their calibration from the `rc`
  group through the new `StickEndpoints`, which degrades to the standard
  travel if the triple is not strictly ordered — the three keys validate
  independently, so each can be in range while the set is not.
- **`eskf_init_att_var_rp` on the GPS re-init path.** `Eskf::init`
  honoured it at bootstrap; `EskfGpsGuard::reinit_filter` passed a
  literal `0.1`. A vehicle that pinned the key got it on the first fix
  and never again after a glitch. The guard now reads the filter's own
  configuration.
- **`max_rate_r` / `max_rate_p` / `max_rate_y` in the cascade.**
  `GeometricAttitudeController::new` clamped the commanded body rate at
  a hardcoded 360/360/180 deg/s while the airframe declared its own rate
  envelope elsewhere. The cascade now clamps to the airframe's numbers.

  **This is the one change in v54 that alters behaviour rather than
  preserving it.** Every vehicle pins `[10, 10, 6]` rad/s, so on the
  cascade path the clamp moves from 6.28/6.28/3.14 to 10/10/6 rad/s —
  looser, particularly in yaw. It affects no vehicle today: all twelve
  build `outer_loop: mpc`, so `cascade_task` is not compiled into any
  shipped image. Re-check it before flying `outer_geometric` again. The
  alternative was a second `cascade`-group rate limit, which would have
  recreated exactly the two-sources-of-truth problem this audit exists
  to remove.

### New parameters

| Group | Keys | Was |
|---|---|---|
| `mahony` (new) | `mahony_kp`, `mahony_ki`, `mahony_min_accel_g`, `mahony_min_mag_ut` | `Mahony::new()` literals — the entire tuning surface of the independent attitude reference was unreachable. The task is spawned on every build and feeds the blackbox `/attitude` topic; see loose end 1 for why its arming cross-check does not yet consume it. |
| `cascade` | `pos_err_max_*`, `vel_err_max_*` | `Default`-impl literals that `PositionController::new` copied; nothing could call the builder that overrode them. |
| `cascade` | `cascade_odom_stale_s` | 100 ms constant. |
| `mpc` | `mpc_odom_stale_s` | 50 ms constant, tightened for the position sampler. |
| `mpc` | `mpc_u_ref_ff` | A `const bool` whose own comment asked to be runtime-settable "without a firmware reflash". |
| `eskf.filter` | `eskf_inflation_cap` | `INFLATION_CAP = 100`, the partner of the already-exposed `eskf_gate_sigma`. |
| `eskf.filter` | `eskf_mag_norm_gate` | A literal `0.3` inside `update_mag` — the filter's only magnetometer quality check. |
| `eskf.filter` | `eskf_max_predict_dt_s` | 50 ms constant bounding the integration gap. |
| `eskf.mocap_guard` | `eskf_mocap_reanchor_frames` | The radius was a parameter, its frame count was not. |
| `indi.controller` | `indi_nan_rampdown` | A bare `0.95` in the NaN failsafe path. |
| `indi.rpm_estimator` | `rpm_est_omega_ref`, `rpm_est_init_omega_var`, `rpm_est_escape_rejects`, `rpm_est_plausible_frac` | The estimator's robustness knobs, and the one piece of its configuration still written as a literal at the construction site. |
| `trajectory.planner` | `plan_dur_min_s`, `plan_dur_max_s`, `plan_waypoint_radius` | Acceptance window and the waypoint ball radius. |
| `trajectory.planner.bfgs` | `plan_bfgs_delta_collapse` | Trust-radius collapse threshold. |
| `battery` | `batt_settle_ticks`, `batt_max_cells` | Cell-detection settling window and its ceiling. |

Every one of these degrades rather than trusting a bad value: a
non-finite or non-positive staleness window, error clamp or collapse
radius falls back to the shipped constant and logs, per the "degrade,
never panic" rule.

## Deliberately not made tunable

### Sizing

These set static memory and cannot be runtime at all. The linker
removes code, not `.bss`, so a `const` flag cannot shrink them.

`MAX_PIECES` (128) and `MAX_PLANNED_PIECES` (20) in trajectory planning,
with the MINCO and BFGS workspaces derived from them; the MPC horizon
capacity `N = 20` and every state dimension; `TABLE_N` (50) for the
thrust table; `MAX_PARAMS` (512) bounding the flash key-value store;
every PubSub `CAP`/`SUBS`/`PUBS`; `BATCH_BLOCKS` (64) in the SD writer,
which is explicitly documented as un-raisable after a 32 KB static
boot-looped every 8 kHz build; the WS2812 chain length and frame buffer;
`MAX_HISTORY` (10) in the RPM estimator; `EVENT_RING_LEN` and
`RECORD_SIZE` in the post-mortem record, each pinned by a `size_of`
assertion.

`MAX_HISTORY` in the RPM estimator was the exception that proved the
rule and has since been fixed: at 10 entries it spanned 5 ms at the
fastest shipped control rate while `rpm_est_tau_d` advertised 20 ms, so
most of the parameter's own range silently fell back to the oldest
entry. It is now sized to the schema ceiling, and a rate faster than any
shipped vehicle is clamped with a warning instead of truncated.

### Physical, protocol and numerical

Sensor register values and datasheet timings; DShot bit timings and the
GCR line coding; the WGS-84 ellipsoid; CBOR and MCAP encodings; the
flash key-value record format and its FNV-1a seed; unit conversions;
u-blox `fix_type` and `carr_soln` enumerations; and the numerical floors
that keep divisions finite — the LU pivot floor, the covariance-diagonal
floor, the quaternion and vector normalisation guards, and the flatness
singularity floors.

The flatness floors deserve a specific note: `Q_REF_MIN_ALPHA_M_S2` in
the outer loop is matched deliberately to
`ALPHA_NORM_SQR_FLOOR_POLE_SAFE`, so that the band where one path
declines to produce an attitude direction is exactly the band where the
other declines. Tuning either alone opens a gap in which one builds a
quaternion out of noise.

### Coupled — runtime-shaped, but not safely alone

Listed with what they are tied to, since these are the candidates most
likely to look harmless later.

- **`CMD_STALE_TIMEOUT` (100 ms, INDI).** The most coupled constant in
  the tree. Both `MPC_RATE_HZ_RANGE` and `CASCADE_RATE_HZ_RANGE` derive
  their 25 Hz floors from it, so that a healthy outer loop cannot read
  as stale in flight. Making it tunable without deriving those two
  ranges from it admits a configuration that fails the moment it flies.
- **`MOTOR_CMD_STALE` (10 ms, DShot).** Documented invariant: it must be
  far shorter than the failsafe control timeout. That timeout is the
  parameter `fs_ctrl_timeout_s`, so the invariant spanned a constant and
  an operator-settable value while being stated only in prose. It is now
  checked at both ends — a compile-time assertion against the schema
  minimum, and a boot warning against the configured value — rather than
  turned into a second parameter.
- **The learned-cost policy constants** in `mpc::cost_adapt` — the
  observation scales, clamps and weight ranges. These are properties of
  the trained checkpoint, not of the vehicle. A policy trained under one
  set is only valid under the same set, so exposing them would let a
  `param set` silently take the policy off-distribution.
- **`PLL_SHIFT` (IMU sample stamping).** Derived jointly with the
  loss-detection threshold; the module documents that the neighbouring
  value settles at twice the residual and false-trips. It is also a
  shift, so only powers of two are expressible.
- **The G2 magnitude bound (`1e4`)** and the thrust-curve clamp
  (`0.025..1.0`), which mirror the schema ranges of `g1_*`, `m*_g2_*`
  and `m*_nonlin`. If the constant and the range diverge, a value
  validates at the write and is rejected at the apply, and the
  controller keeps the previous matrix without saying why.
- **`RATE_DOT_SG_MAX_GROUP_DELAY_S`.** Its only output is an integer
  filter window that a compile-time assertion also pins, and the window
  must stay delay-matched against `indi_sync_hz`.

### Duplicated defaults worth removing rather than exposing

Several structs carry a second, divergent copy of values the registry
already owns. These are drift hazards, not tuning surfaces.

- `FullQuadModel::default()` is a third copy of the MPC and airframe
  defaults and disagrees with the schema on several weights. It is
  reachable only from tests; deleting it is better than reconciling it.
- `ArmConfig::FALLBACK` in the RC task hand-maintains eight arming
  values that are all parameters, with nothing enforcing agreement.
  `StickConfig::FALLBACK` and `FsConfig::FALLBACK` do the same for the
  stick envelope and the failsafe timing. Each applies only in the
  window before its `init_*` function runs, so the fix is a compile-time
  assertion against the schema defaults.
- `is_touching_ground` in `rpm_tracker` hard-codes the ground-contact
  test that the live controller path reads from
  `indi_ground_gyro_dps` and `indi_ground_accel_g`. It has no
  non-test callers.

## Loose ends found on the way

Not parameter questions, but they surfaced during the sweep and are
recorded so they are not lost.

1. **The Mahony arming gates are unreachable.** `health.rs` hard-codes
   `mahony_ready = true` and the reported disagreement to `None`, so
   both `MahonyNotReady` and `EskfMahonyTiltDisagreement` are dead
   branches and `arm_eskf_mahony_tol_deg` has no effect today.
2. ~~Two disagreeing altitude floors.~~ Fixed, in two steps. The
   landing integrator first moved off its hard zero onto
   `fence_z_min_m`, then off absolute altitude entirely: it is now
   bounded relative to the vehicle's own measured altitude
   (`rc_land_lead_m`). z = 0 is the ENU origin, the mocap anchor or the
   first RTK fix, and not the ground, so any absolute floor stops the
   descent in mid-air over terrain below the takeoff point. The relative
   bound is terrain- and origin-agnostic and needs no per-site number.
   The fence keeps its absolute envelope, so the two are now different
   concepts rather than competing floors.
3. **The RPM-estimator builder fallbacks disagree with the schema
   defaults** by 4× on the measurement variance and 1000× on the `c_m`
   random walk. Only a path that skips `set_config` sees them, but the
   divergence is silent.
4. **Board inconsistencies.** One board uses a 3 s GPS init timeout
   where the others use 30 s — shorter than one full receiver retry
   cycle — and sizes its GPS receive buffer at 256 B against 1024 B
   elsewhere.
5. ~~Three clock-derived constants unguarded against an RCC change.~~
   Fixed: each BSP now declares `SYSCLK_HZ` and its timer kernel clocks
   beside the configuration that determines them, the DShot prescaler,
   WS2812 pulse widths and the DWT conversion derive from those, and
   `clocks::verify_clocks()` compares them against the running RCC at
   boot. The WS2812 tail wait now derives from the reset pad that
   governs it. Note the original report of that one was wrong in an
   instructive way: the wait covers the post-DMA tail and the strip's
   latch, not the whole burst, so it never did scale with chain length.
   The code's own comment misstated the burst as "~181 µs" against an
   actual ~1.02 ms, which is what made a chain-length dependency look
   real.
6. **The online planner's waypoint table is compiled in**, while the
   offline path bakes missions from YAML. The planner itself is now
   selectable per vehicle (`build: plan_online`), so this is the
   remaining half of that gap.
7. ~~The ESP bridge batch buffer panicked on overflow.~~ Fixed:
   `encode_frame` refuses a frame that will not fit instead of indexing
   past the destination, and the bridge counts the drops and reports
   them on its ping cadence.
