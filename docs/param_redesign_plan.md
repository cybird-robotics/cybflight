# Parameter System Redesign — Master Plan

Status: **approved, not started** (2026-08-01, branch `feature/0801_param_config`).
Decisions below were settled in an architecture interview; do not re-litigate them
piecemeal — amend this document instead.

## Motivation (what the audit found)

- Adding one parameter today touches ~9 hand-maintained sites in
  `crates/cybflight_core/src/params.rs` (struct field, `Default`, `new()`,
  `to_bytes`, `from_bytes`, `ParamKey` variant, `from_str`, `as_str`,
  `get`/`set` arms, plus `VERSION`, `PAYLOAD_SIZE`, and the doc-header layout
  table). The file's own header is stale (documents v20/624 bytes vs actual
  v29/636) — evidence of the maintenance burden.
- Every schema `VERSION` bump rejects the stored blob and silently wipes all
  saved parameters. 29 versions of wipe history.
- Four divergent "default vehicle" definitions: `vehicle.rs` (0.6 kg / 12 N),
  sim `plant.rs` (0.55 / 8.5), `QuadModel::default()` (0.58, with a doc comment
  wrong about both), `QuadPlanningConfig::default()` (a fourth inertia set).
- **Split brain:** MPC/planner read `body`/`motors` from flash params, while
  INDI and cascade read the compile-time consts `QUADROTOR_BODY` /
  `QUADROTOR_MOTORS` (`indi_task.rs:245-246`). `param set mass` changes the
  outer loop but not the inner loop.
- `ControlGains` is dead data under the default `outer_mpc` build (sole
  consumer `cascade_task.rs` is `outer_geometric`-gated); `att_k_rate` is
  flash-persisted but has no `ParamKey` in any build.
- The INDI RLS learner module is **deprecated** (maintainer decision).
- `param set`/`param save` have no armed-state guard; a `param save` in flight
  stalls the CPU 1–2 s (same-bank flash erase) mid-control-loop.
- A default mass exists (0.6 kg). Physical identity parameters (mass, inertia,
  motor geometry/thrust) are facts about a specific airframe and must not have
  fallback defaults.

## Agreed decisions

| Axis | Decision |
|---|---|
| Schema source of truth | Rust structs + `#[derive(Params)]` proc-macro (generates key registry, name↔key, get/set, metadata, YAML serde, KV ser/de) |
| Values source of truth | Per-airframe `vehicles/<VEHICLE>.yaml`; `VEHICLE=` selected via `.env`/Justfile |
| Binding time | **Compile-time bake** — `build.rs` validates YAML against schema; missing *identity* param = `compile_error!`. One binary per airframe (fits existing per-role `.env` builds). Provisioning-tool design deferred until fleet scale; schema work is shared, so the door stays open |
| Identity vs tuning | Identity params (mass, inertia, motor geometry/thrust, extrinsics) **required, no defaults**; tuning params keep defaults |
| Persistence | **Key-value flash records** (overrides only) replacing the fixed blob; unknown keys ignored, missing keys default → schema changes stop wiping calibration |
| Precedence | Flash override wins per-key (PX4/Betaflight semantics); **auto-prune** a record when set equal to the baked value |
| Write-back | Firmware never edits YAML. `param diff --yaml` emits schema-shaped YAML; host-side `just param-sync` merges into the vehicle file for explicit git commit |
| Tuning I/O | USB CDC shell only (no MSP, no configurator, no telemetry param uplink) |
| Scope of const→param migration | IMU filter cutoffs, GPS antenna extrinsics, mag hard-iron, ESKF noise densities. **Not** migrated: safety thresholds (arming/failsafe/tilt) and the RC surface — those stay code-reviewed constants |
| Learner | Delete entirely; `IndiEffectivenessParams` stays, re-documented as "manually configured; all-zero = geometric fallback" |
| Container decomposition | `VehicleParams` → **`FirmwareConfig`** with per-subsystem groups (below). No group is ever `cfg`-gated out of the schema — dead in a build is fine, absent is not (keeps flash portable across feature builds) |
| Sim | Shares the schema/loader; values frozen in a committed `sim_baseline.yaml`. Divergent `::default()` impls deleted; snapshot re-pinned in the same commit |

### Target decomposition

```
FirmwareConfig
├── airframe:  AirframeParams   body, motors            (identity; required)
├── sensors:   SensorParams     IMU cutoffs, GPS antenna extrinsics,
│                               mag hard-iron, ESKF noise densities   (new)
├── indi:      IndiParams       effectiveness + controller tuning     (always hot)
├── cascade:   CascadeParams    pos_kp/kd, att_k_rate   (rename of ControlGains;
│                                                        live under outer_geometric)
├── mpc:       MpcParams        as-is                   (outer_mpc)
├── trajectory: TrajectoryParams planner, sampler, mission_profile    (outer_mpc)
└── system:    SystemSettings   arm_led_enabled, blackbox_record_set
```

### End-state dataflow

```
Rust structs + #[derive(Params)]  ── single source of SCHEMA
 ├── vehicles/<VEHICLE>.yaml ──build.rs──► baked defaults (identity required)
 ├── flash KV store ──► per-key overrides only (override wins; auto-prune)
 ├── shell: param list/get/set/diff/reset/save + range checks + arm guard
 ├── host: just param-sync  (param diff --yaml → merge → git commit)
 └── sim: sim_baseline.yaml through the same loader
```

## Phases

Each phase lands independently, compiles, and passes
`just check` / `just test` / `just sim-check` before the next begins.

### Phase 0 — Safety & hygiene quick wins (S)

1. Arm-guard `param set`/`param save` in `dispatch_param`
   (`usb_serial.rs:1563`), mirroring the existing `blackbox`/`mission` guard.
2. Fix mag alignment hardcode: pass `board.sensors.mag_align` instead of
   literal `Cw180Deg` (`board_init/sakurah743.rs:377,481`).
3. Name the IMU cutoff literals (`80.0, 200.0` at
   `board_init/sakurah743.rs:195,536`) — placeholder until Phase 4.

### Phase 1 — Learner removal (M)

Delete: `cybflight_core/src/indi/learner.rs` (~960 lines); `LearnerParams` +
its 9 `learn_*` keys and serialization slots; the prearm latch machine,
disarm auto-save, and `write_learned_to_params` in `indi_task.rs`;
`LEARNING_ENABLED`/`LEARNER_PREARM` statics in `control/mod.rs`; learn-switch
decode in `rc_interpreter.rs` (frees RC aux channels 6–7); sim `plant.rs`
learner config.

Keep: `learned_from_indi_params` / `apply_learned_params` (flash→controller
load path). Reword `IndiEffectivenessParams` docs. Bump `VERSION`. Verify the
sim snapshot does not drift; investigate before proceeding if it does.

> **Superseded (2026-08-02):** the retained load path was itself learner
> plumbing and has since been removed. `LearnedParams` /
> `learned_from_indi_params` / `apply_learned_params` /
> `update_from_learned` are replaced by
> `IndiController::apply_effectiveness_params` with per-block semantics
> (G1 zero-sentinel → geometric, invalid → degrade-to-geometric; G2
> verbatim from params; tau/omega carry real schema defaults).

### Phase 2 — Struct decomposition + derive macro (L, core)

1. New host-side proc-macro crate (e.g. `crates/params-derive/`):
   `#[derive(Params)]` with field attributes
   `#[param(key = "…", unit = "…", range(min, max), required)]`.
   Generates key registry, per-key typed ser/de, metadata table, YAML
   `Deserialize`. Doc comments become descriptions. Replaces ~1,100 lines of
   hand-written registry code.
2. Decompose into `FirmwareConfig` groups; `CascadeParams` rename;
   `att_k_rate` gains keys automatically.
3. Resolve the INDI `nonlinearity` dual-source (const `THRUST_NONLINEARITY`
   vs flash `indi_nonlin_m*`) to one param, one consumer path.
4. Values become properly typed (bool/u8/u16/enum) — no more all-f32 encoding.
5. Storage remains the flash blob this phase, but macro-generated. Not
   byte-compatible with v29 (learner slots gone; a wipe was due anyway) —
   export a `param diff` on the bench **before** flashing the upgrade.
6. Tests: per-group round-trip, key-coverage completeness.

### Phase 3 — KV flash store (M)

1. Replace the blob with a key-value map (e.g. `sequential-storage`).
   Hardware note: wear-leveled KV needs ≥2 erase units; H743 sectors are
   128 KB → reserve sectors 6+7 (256 KB), shrink app flash in `memory.x`
   1920K→1792K.
2. Flash holds overrides only; boot = baked defaults + overrides.
3. Auto-prune on equality. `PARAM_VERSION` hot-reload unchanged. Keep
   explicit `param save`; watchdog extension retained only for the
   GC/compaction path.
4. Bench flash-cycle test before trusting the new geometry.

### Phase 4 — YAML bake, required identity, split-brain fix (M)

1. `vehicles/<name>.yaml`; `VEHICLE=` in `.env`/Justfile; `build.rs`
   validates + emits baked consts; missing identity param = compile error.
   Delete `default_params()`, `QUADROTOR_BODY`, `QUADROTOR_MOTORS`,
   `DEFAULT_CONTROL_GAINS`.
2. Fix the split-brain: `indi_task` and `cascade_task` switch from consts to
   `params::get()` + disarmed hot-reload.
3. Migrate scoped constants into `SensorParams`.
4. Expose the hardcoded MPC `thrust_percentage = 0.75`
   (`quad_model.rs:183`); make planner and MPC agree on available thrust.

### Phase 5 — Shell + sync tooling (S–M)

1. `param diff [--yaml]`, `param reset <key>|all`, metadata-driven range
   validation on `set`, units in `param list`.
2. Host script + `just param-sync` (pull diff over USB CDC, merge into the
   vehicle YAML, show `git diff`). Explicit commit only — no automatic
   write-back.
3. Auto-generate `docs/parameters.md` from the metadata table (test or
   xtask).

### Phase 6 — Sim schema unification (S–M)

Sim loads committed `sim_baseline.yaml` via the shared loader. Delete
`QuadModel::default()` and `QuadPlanningConfig::default()`;
`from_params(...)` becomes the only constructor. Re-pin the regression
snapshot in the same commit per the `sim-snapshot` workflow.

## Net effect

~3,500 lines of hand-maintained code deleted; one schema everywhere; no
default mass; calibration survives schema changes; `param set mass` reaches
every consumer; adding a parameter = one struct field with an attribute + a
YAML value.

## Risks

- **Phase 2 proc-macro** is the biggest single step. Mitigation: storage and
  schema changes never land together (blob retained until Phase 3).
- **Phase 3 sector geometry** change needs a bench flash-cycle test.
- One-time parameter wipe at the Phase 2 upgrade — export overrides first.
