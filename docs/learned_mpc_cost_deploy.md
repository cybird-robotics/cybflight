# Deploying the drag model + v10 cost policy on the SAKURAH743 boards

Status: **implemented 2026-09-08** (plan written 2026-08-29). Companion to
[learned_mpc_cost.md](learned_mpc_cost.md) (methodology, results).
Landed: the drag model (v51 params, `from_vehicle_params` applies it),
`mpc_learned_cost`/`mpc_learned_gain` (v52), the `build.rs` policy bake
(`mpc_cost_policy:` YAML key → `data/cost_policies/<stem>.bin` →
flash statics, dims checked against `cost_adapt`), the outer-loop hook
(observation → policy → `CostNominal::apply` before every solve), and
the `/mpc_cost` blackbox topic (Mid tier, channel 16). The airframe was
identified 2026-09 (τ = 30 ms committed; the pooled re-fit reads
0.032–0.042 s) and the deployed checkpoint is **v20b**
(`data/cost_policies/v20b.bin` + sidecar), retrained on the identified
plant with user-widened domain randomization (τ ×[0.5, 2], inertia
×[0.7, 1.5], mass ±20 %, wider loc noise, unmodeled wind force) and the
**v20 asymmetric weight map** (contour [100, 2000], lag [50, 500],
att-yaw [50, 200] soften-only at the flight nominal — `cost_adapt`
constants; each map revision invalidates all earlier checkpoints). On
the identified plant v20b is **−19.3 %** mean geometric RMSE vs the
drag-modelled nominal over 25 missions with no regressing mission
(v18 was −15.6 %; the v19 yaw-retune experiment and per-seed history
are in learned_mpc_cost.md — judge configs on ≥ 2 seeds).

## What is being deployed

Two changes to the outer MPC, in the order they should reach the vehicle:

1. **Rotor-drag term in the prediction model** — the `sysid_mcap.py` model
   `a_b = k·v_b·Σω`, coefficients identified per airframe, `Σω` recovered
   from the commanded collective. In sim: −84 % geometric RMSE on every
   mission with the constant tune, robust to everything tested. This is
   the part that must fly first and on its own.
2. **v10 cost policy** — 59→128→128→19 tanh MLP, residual on the state
   weights `w = w_nom·4^z`, rate weights `10·4^{max(z,0)}` (floor at the
   nominal), thrust weight fixed. In sim: a further −20 % on top of (1);
   neutral-to-positive under every perturbation except a 40 ms motor time
   constant on the three most demanding missions. ~0.2 ms per solve,
   107 KB of flash for the weights.

The policy is a *residual on a specific nominal*. It was trained and
evaluated with: `mpc_w_pos [500,500,200]`, `mpc_w_vel 10`,
`mpc_w_att [50,50,200]`, **`mpc_w_rate 10`** (not the YAML's 20),
`mpc_w_thrust 1`, contouring mode, `mpc_dt 0.05`, `mpc_horizon_n 20`,
RTI (`mpc_max_iters 1`), tilt barrier off, position sampler with
`sampler_max_lag_s 0.1`, tilt-yaw `q_ref`, flatness `u_ref` feed-forward,
INDI rate gains 80 / sync 30 Hz. Every one of those is a vehicle-YAML
value the policy silently assumes; the checklist below pins them.

## Changes, by crate

### `cybflight_core` (`params.rs`, schema `VERSION` bump)

New `MpcParams` fields, all in the live-tunable `mpc` group:

| key | type | default | meaning |
|---|---|---|---|
| `mpc_drag_x`, `mpc_drag_y`, `mpc_drag_z` | f32 [N·s²/(m·rad)] | 0 | rotor-drag coefficients `c = −m·k` from `sysid_mcap.py`; 0 = term off |
| `mpc_bodydrag_x`, `mpc_bodydrag_y`, `mpc_bodydrag_z` | f32 [N·s²/m²] | 0 | quadratic body drag `½ρC_dA` (the `v_b|v_b|` regressor of the sysid fit); needed above ~30 m/s |
| `mpc_learned_cost` | u8 0/1 | 0 | run the baked cost policy every tick — including disarmed and off-trajectory, where its output is logged but withheld (see below) |
| `mpc_learned_gain` | f32 [0, 1] | 0 | scale on the policy output, `z_eff = gain·z`; the ramp-in knob (0 = nominal weights even with the policy running, 1 = as trained) |

`mpc_learned_cost` gates whether the policy **runs**; the gain and the
mission state gate whether its output is **applied**. That split is what
makes the path checkable on the bench: with `mpc_learned_cost 1` set
disarmed, `/mpc_cost` and `OcpSolverOutput` report a non-zero
`policy_time_us` and a live `z` immediately, instead of leaving the whole
chain (policy baked? shapes valid? param reload landed?) unverifiable
until a mission is already executing. The cost is ~0.2 ms per outer-loop
tick (~2 % of CPU at 100 Hz), spent after the command is published, so it
never touches command latency.

The output is applied only when the tick fanned out a trajectory **and**
the mission is `Executing`: the observation's preview is meaningless
without a trajectory and the policy was never trained on hover, so an
off-trajectory `z` is an out-of-distribution number. On those ticks the
record carries that `z` with `gain = 0` — the weights stay exactly
nominal, and `gain·z` remains the applied residual on every tick.

The thrust coefficient the drag term needs, `c_T = max_thrust_n/ω_max²`,
is not a new parameter: `airframe.motors[i].max_thrust_n` and
`m*_omega_max` are already pinned per vehicle; `build_outer_quad_model`
computes it (mean over the four motors).

Then: `VERSION += 1`, `just params-doc`, commit `docs/parameters.md`.

### `cybflight` firmware

**Baking the policy** (`build.rs`, mirroring `bake_thrust_tables`):
`crates/cybflight/data/cost_policies/<name>.bin` (the flat
`nn::Mlp` export `tools/train_cost_policy.py` writes) → `$OUT_DIR/cost_policies.rs`
with `pub static <NAME>_WEIGHTS: [f32; N]` and `<NAME>_SHAPES:
[LayerShape; L]` in `.rodata`. Vehicle selection through the YAML, like
the thrust table:

```yaml
airframe:
  ...
mpc_cost_policy: v10        # file stem under data/cost_policies/; omit = none baked
```

A vehicle without the key bakes no weights (no flash cost). The bake
checks the file's input/output widths against `cost_adapt::{OBS_DIM, NZ}`
so a stale export fails the build, not the flight. It also checks the
**cost-map revision** the export header carries against
`cost_adapt::COST_MAP_VERSION`: the width checks cannot see a map change
that leaves the dimensions alone — v18 → v20 moved one fence constant —
so without it an archived checkpoint from an earlier map would build
clean and fly the same `z` with different semantics. Bump
`COST_MAP_VERSION` whenever a `cost_adapt` map constant changes and
re-export the checkpoints still in service; the sim's loader
(`OwnedCostPolicy::from_file`) rejects the same mismatch, so a stale
checkpoint cannot be evaluated either. Put the checkpoint's
provenance (training config, sim numbers, and the observation units
`V_SCALE / V_ERR_SCALE / RATE_SCALE / OBS_CLAMP` it was trained under) in
a sidecar `data/cost_policies/<name>.md`; the bake asserts the units
against `cost_adapt`'s constants.

**Outer loop** (`control/outer_loop.rs`), all inside the existing tick:

1. `build_outer_quad_model`: after `from_vehicle_params`, call
   `model.set_rotor_drag([mpc_drag_x, _y, _z], c_T)` when any coefficient
   is non-zero; capture `CostNominal::from_model(&model)` next to the
   problem. Both are rebuilt on the disarmed hot-reload, so `param set
   mpc_drag_x …` and a `mpc_w_*` retune take effect the same way they do
   today — and the nominal the policy modulates follows the retune.
2. **After step 9 (`RATE_COMMAND.signal`)** — the latency-first ordering
   (2026-09-08): when `mpc_learned_cost == 1` and a policy is baked,
   assemble `SituationInputs { x0: &mpc_x0, body_rate:
   odom.twist.angular, x_refs, nodes: &sample_buf, last_u: &last_u0 (this
   tick's clamped u0), u_bounds, mass, grav }` → `situation_obs` →
   `CostPolicy::act` → `z *= gain_eff` → `nominal.apply`, where
   `gain_eff` is `mpc_learned_gain` on a tick that fanned out a trajectory
   with the mission `Executing` and 0 on every other tick. `gain_eff`
   travels with the residual (not re-read at publish time) so the
   `/mpc_cost` record reports the gain that was in force for the solve it
   describes. The residual is
   consumed by the NEXT tick's solve — one tick (10 ms) of weight
   staleness, so the ~0.2 ms policy never delays the inner-loop command.
   Measured cost of the delay (sim, v20b, 25 missions +
   robustness matrix, `COST_POLICY_DELAY=1` in the eval harnesses):
   −14.0 % vs −14.2 % mean geometric RMSE — noise. On a non-finite
   output `act` already returns `z = 0`; the diverged / non-finite solve
   guards restore the exact nominal for the retry. In hover /
   no-mission ticks, apply `z = 0` (the observation's preview is
   meaningless without a trajectory; the policy was never trained on
   hover). When re-evaluating checkpoints in sim, pass
   `COST_POLICY_DELAY=1` to mirror the flight ordering.
3. Keep `last_u0` (the clamped `u0` published to INDI) for the
   observation's last two entries.
4. Budget: the policy runs *after* the publish, off the critical path;
   the solve-time warning covers the solve alone and the policy's wall
   time rides in every `/mpc_cost` record.
   Expected ≈ 0.2 ms at 480 MHz (26 k MAC + 256 `tanhf`).
5. Blackbox: a `/mpc_cost` record at the outer-loop rate — `z[19]`,
   the effective weights (contour, lag, rate ×3), `gain`, policy time —
   in the `mid` tier. Without it the flight cannot be compared against the
   sim probes.

**Memory**: weights are a `static` in flash (not `.bss`; see the 8 kHz
boot-loop history in the project memory — do not make this a
zero-initialised static). Runtime stack: two 128-float buffers + 59-float
observation, inside the existing outer-loop task stack.

### `cybflight_sim`

Nothing new to fly, but keep parity honest: `MpcIndiController` must
build its model through the *same* `set_rotor_drag(coeffs, c_T)` the
firmware uses (today the sim test harness reads `sim: aero_drag`; the
firmware will read `mpc_drag_*`). The intended relationship is
`mpc_drag_* == sim: aero_drag` for a vehicle whose drag has been
identified — the bake should warn when both are set and differ.

### `analysis/sysid_mcap.py`

`print_yaml_block` already emits `sim: aero_drag: [−m·k_x, −m·k_y, 0]`.
Add the same numbers under `tuning:` as `mpc_drag_x/y` (and `mpc_drag_z:
0` explicitly — vertical drag is not identified; the sim baseline's note
applies), so one run of the script yields the paste-ready block for both
the plant and the controller.

### `vehicles/sakura_bench_leader_1khz.yaml` (and the other flying YAMLs)

```yaml
mpc_cost_policy: v10                # bake the policy (flash cost 107 KB)
tuning:
  # ── mpc — cost weights (the v10 nominal; the policy is a residual on these) ──
  mpc_w_pos_x: 500
  mpc_w_pos_y: 500
  mpc_w_pos_z: 200
  mpc_w_vel_x: 10
  mpc_w_vel_y: 10
  mpc_w_vel_z: 10
  mpc_w_att_r: 50
  mpc_w_att_p: 50
  mpc_w_att_y: 200
  mpc_w_rate_r: 10                  # was 20 — the constant-tune change (−28 % on its own)
  mpc_w_rate_p: 10
  mpc_w_rate_y: 10
  mpc_w_thrust: 1
  mpc_dt: 0.05
  mpc_horizon_n: 20
  mpc_max_iters: 1
  mpc_pos_cost_mode: 1
  mpc_tilt_barrier_tau: 0.0
  # ── mpc — rotor drag in the prediction model (identified, sysid_mcap.py) ──
  mpc_drag_x: 2.6675e-5             # placeholder = sim_baseline; REPLACE with this airframe's fit
  mpc_drag_y: 4.004e-5
  mpc_drag_z: 0.0
  # ── mpc — learned cost adaptation ──
  mpc_learned_cost: 0               # flipped to 1 at step 3 of the workflow
  mpc_learned_gain: 0.0             # ramped 0.25 → 0.5 → 1.0
  # ── trajectory.sampler — as trained ──
  sampler_kind: 1
  sampler_max_lag_s: 0.1
  sampler_max_lead_s: 0.1
```

`just print-features sakura_bench_leader_1khz` is the dry run; the bake
warns on any unpinned key the policy assumes.

## Workflow

Each step is a gate; do not skip one because the sim said it would be
fine — the sim's two known blind spots (motor time constant, rate-command
noise) are exactly what steps 0 and 2 measure.

**0. Identify the airframe (one flight, `blackbox set sysid`).**
Fly the current tune on `indoor_figure8_mid`; run
`python3 analysis/sysid_mcap.py <log> --vehicle vehicles/sakura_bench_leader_1khz.yaml`.
Take from it: `mpc_drag_x/y` (and update `sim: aero_drag` for the sim
twin), `m*_tau`, `m*_omega_max`, `max_thrust_n` cross-check. **Decision
point**: if τ ≤ 30 ms the whole plan applies; if τ is 40 ms or unknown,
`mpc_w_rate 10` is still the right constant (it is τ×2-safe on all but the
30 m/s track) but the policy stays at `mpc_learned_gain 0` until τ is
brought down or the policy is retrained with the measured τ in the plant.

**1. Constant-tune change, policy off.** `mpc_w_rate_* 20 → 10`,
`mpc_drag_*` set, `mpc_learned_cost 0`. Hover → `figure8_slow` → `figure8_mid`.
Compare the blackbox tracking error against the previous flights of the
same missions (the sim predicts roughly a halving from the rate weight
and a further ×3–4 from the drag term). Look at the rate-command
spectrum (`/mpc` u0 in the log): `w_rate 10` must not add chatter. If it
does, this is the bench-measured floor the sim could not price — stay
at whatever `w_rate` is clean, and note that v10's floor assumes 10.

**2. Drag term validation.** Same missions, `mpc_drag_*` on vs off
(hot-reload between flights, disarmed). Expect the time-indexed lag to
drop visibly on the fast segments; if it *increases*, the sign or the
`Σω(T)` scale is wrong for this airframe (check `max_thrust_n` and
`ω_max` — `c_T` is what converts thrust to rotor speed).

**3. Policy ramp-in.** `mpc_learned_cost 1`, `mpc_learned_gain` 0.25 →
0.5 → 1.0, one flight each on `figure8_mid`, then `figure8_timeopt`.
At each gain: solve+policy time from the log (< 1 ms total budget is the
line), `z` and effective-weight traces sane (`|z|` mostly < 0.7, no
tick-to-tick flapping), tracking error not worse than step 2. The sim
probes (`target/cost_policy_v10/probe.json`) say what to expect
qualitatively: contour weight rising as the thrust margin closes, rate
weights rising with speed.

**4. Time-optimal missions.** `splits_timeopt` (the held-out one), then
outdoor. Same three checks. Only now compare absolute numbers with
`learned_mpc_cost.md`'s table; hardware will not reproduce 0.05 m, but
the *ratios* between steps 1/2/3 should hold.

**5. Sync the tune back.** `param save` → `just param-sync` → commit the
vehicle YAML with the identified drag numbers, the rate weight and the
gain that flew, and add the flight's summary numbers to
`learned_mpc_cost.md`.

## Safety and fallbacks

* The policy can only move weights within `[0.25, 4]·w_nom` (state) and
  `[10, 40]` (rates); it cannot move references, bounds, the horizon or
  the rate floor. The worst case it can produce is a stiffer, more damped
  cost — the sim's failure mode for that is the 40 ms motor-lag divergence
  on the two hardest missions, which step 0 rules in or out.
* `mpc_learned_gain 0` and `mpc_learned_cost 0` are both live params;
  either returns bit-identical nominal *weights*. They differ in cost:
  `mpc_learned_cost 0` also stops the forward pass, `mpc_learned_gain 0`
  keeps running it (and logging its `z`) at `gain = 0`. So does any tick
  that is not executing a trajectory. The diverged / non-finite paths
  force `z = 0` for the tick and drop the warm start exactly as today.
* The drag term is an unconditional model change: `mpc_drag_* 0` turns it
  off. Its own failure mode is a wrong sign (drag becomes thrust along the
  velocity) — the bake should reject negative coefficients.
* Nothing touches INDI, the sampler, the planner or the mission logic.

## Not in scope, and why

* Retraining on hardware data: the policy is a residual on the model; if
  step 2 shows the drag model fits, the residual transfers. If the
  identified τ is large, the right fix is retraining with that τ in the
  *plant* of the sim, not a hardware-in-the-loop RL setup.
* `sampler_max_lag_s 0.5` (the geometric-only setting): worth its own
  flight later; it changes what "on time" means for the mission logic and
  the terminal hold, and the policy was trained at 0.1.
