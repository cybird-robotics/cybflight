# Hacking on cybflight

Notes for contributors. Architecture and design live in
[architecture.md](architecture.md); this file is about the workflows you'll
actually touch while changing code.

## Development tools

Firmware builds and the default test suite use Rust, Bash, and `just`. Vehicle
feature selection uses the shared Rust YAML parser. Rust still needs its normal
host linker and platform libraries for build scripts and host executables.

Python 3 is needed only for optional utilities such as `just size`,
`just param-sync`, `just blackbox-pull`, and analysis plots; individual scripts
list any additional Python dependencies. Flashing over DFU requires `dfu-util`.

The optional Rerun visualization (`VIZ=1 just sim-run`) enables the `viz` Cargo
feature, whose dependencies compile C code and need a C compiler. Headless
regression tests do not enable it.

## Parameter & vehicle-config workflow

The full design is in
[architecture.md → The Configuration Plane](architecture.md#the-configuration-plane);
the generated per-parameter reference is [parameters.md](parameters.md).
Day-to-day:

- **Build for a vehicle**: `just vehicles` lists them; `just build <name>`
  builds one and `just flash <name>` builds + flashes the same one. With
  no argument both fall back to `VEHICLE=` in `.env`. Either way the
  chosen `vehicles/<name>.yaml`'s `build:` section drives the cargo
  feature list — `just print-features [<name>]` shows the resolution and
  per-knob provenance, and is the dry run for a build (`just --dry-run
  build` will not show features; it does not evaluate backticks). A name
  with no matching YAML fails immediately with the available names. Env
  vars (`BOARD=`, `OUTER_LOOP=`, `POS_SOURCE=`, …) remain dev overrides;
  `build.rs` hard-fails a mismatched board↔vehicle pairing.
- **Tune at the bench**: `param list|get|set|diff|save` over USB CDC.
  `param list` prints the whole config tree grouped by subsystem —
  `[eskf.mocap_guard]`, `[indi.controller]`, `[trajectory.sampler]`, … —
  led by a `[build]` block showing the compile-time selections
  (`board`, `pos_source`, `outer_loop`, `imu_rate`, the baked vehicle and
  `airframe.name`). Those are cargo features, so they are read-only, but
  they are the first thing to check when a board behaves like a vehicle
  it isn't. Narrow with `param list <group>` or `param <group> list`;
  a group matches its whole subtree, and `.`/`-`/`_` are
  interchangeable (`param eskf-mocap_guard list`). An unknown group
  prints the available ones.
  `param set` applies on the disarmed hot-reload (reboot-flagged params
  are marked `(reboot)` in the listing); `param save` persists to the KV
  flash log. Then
  `just param-sync` merges `param diff --yaml` into the vehicle file —
  review the git diff and commit. The firmware never writes YAML.
- **A YAML edit didn't take effect**: a saved override shadows it. Flash
  wins per key, and re-flashing does not clear the KV sectors (`memory.x`
  reserves them; DFU writes the app region only). `param diff` lists
  every key whose live value differs from the bake — that list *is* the
  answer. Fix it in whichever direction is right: `just param-sync` to
  adopt the flash tune into the YAML, or `param reset <name>` +
  **`param save --prune`** to adopt the YAML. Plain `param save` there
  only papers over it — with no tombstone record it stores today's baked
  value, so the same key is shadowed again at your next YAML edit.
  `param reset all` + `param save --prune` is also what you want after
  flashing a *different vehicle* onto a board: records carry no vehicle
  identity, so the previous airframe's mass/inertia replay onto the new
  one and pass range validation.
- **Add a parameter**: one attributed struct field in
  `cybflight_core/src/params.rs` + `VERSION` bump; see the checklist in
  architecture.md. Regenerate the reference with `just params-doc` and
  commit it.
- **Add a mission**: drop the offline planner's output YAML into
  `missions/<name>.yaml` (native `start`/`waypoints`/`durations` form is
  accepted) and rebuild — validation happens at bake. Select with
  `mission set <name>`; the vehicle YAML's `default_mission:` names the
  boot default. Re-run `mission set` + `param save` after adding or
  removing mission files (the persisted selection is an index into the
  name-sorted table).
- **Changing schema defaults**: flying vehicles pin their
  hardware-coupled values explicitly in YAML (the bake warns about
  unpinned keys), so a `Default` retune must not silently change them.
  Keep it that way — pin what you fly.

## Test fixtures and data

The INDI regression tests use the committed `tests/indi_golden/golden.csv`
fixture. Running them needs neither an external reference checkout nor a
fixture generator. Keep these expected results independent of the Rust
implementation under test; review numerical changes rather than regenerating
expected values to match them.

Commit new bench datasets under `crates/cybflight/data/` and mission inputs
under `missions/`, so contributors can reproduce checks from a fresh clone.

## Sim regression snapshot

`crates/cybflight_sim/tests/regression_snapshot.json` is a **committed**
JSON file with the expected summary metrics for the sim comparison
(16 rows: the 4 scenarios × 3 controllers grid plus contouring, GPS-in-
the-loop, and noisy variants). `cargo test` runs the sim and
compares the actual numbers against this file with a tight tolerance
(~1 % relative plus a small absolute floor so hover-level's ~0 values
don't trip). Drift fails the test.

### Workflow

You touched `cybflight-core` or `cybflight-sim` and want to know if
behavior moved:

```
just sim-check
```

Three outcomes:

1. **Green.** Nothing moved beyond tolerance. Ship it.

2. **Red, and you did not intend to change behavior.** You have a real
   regression. Look at which rows drifted — the failure message lists
   them with `expected → actual (Δ)` per metric — and fix the code.

3. **Red, and you did intend to change behavior** (retuned a gain,
   refactored the plant, picked a different solver warm-start, etc.).
   Regenerate the snapshot:
   ```
   just sim-snapshot
   git diff crates/cybflight_sim/tests/regression_snapshot.json
   ```
   Review the diff yourself — **this is the important step**. The diff
   is the change log for how the controllers now behave. Commit the
   code change and the updated snapshot **together** in one commit, so
   the reviewer sees the cause (code) and the effect (numbers) in the
   same diff.

### Why the snapshot file is committed

Same pattern as [insta]. A committed snapshot makes behavior change
**reviewable** at PR time: someone reading the diff sees
`mpc_indi/mission_square rms 0.029 → 0.034` and can ask "why?" instead
of discovering the drift in flight. The alternative — storing expected
values only in local scratch files, or computing them fresh each run —
hides the change from the reviewer.

### Why this feels scary if it's new to you

Committed snapshot files trip an instinct: "tests shouldn't depend on a
file I might 'accidentally fix' by regenerating." That instinct is
right *in general* and wrong *for this pattern specifically*, for two
reasons:

- **You cannot regenerate it silently.** `just sim-snapshot` writes the
  file, but the change only lands when you `git add` and commit it. The
  PR reviewer sees the numeric diff next to the code diff. A reviewer
  who doesn't understand why numbers moved should reject the PR — the
  whole point of committing the snapshot is to force that conversation.

- **Tolerances are tight but not bit-exact.** Unrelated rustc /
  nalgebra / LLVM-version churn won't touch the file, because 1 %
  relative is way looser than float rounding noise. Only real changes
  that meaningfully shift controller behavior trip it.

If you're genuinely unsure whether a diff is intentional, that's what
the PR reviewer is for. Don't guess — ask.

### What the file does **not** contain

- No per-step trajectory history. Too big, too churny, and the summary
  metrics already catch meaningful regressions. If you need to
  bit-exact-diff a refactor locally, export `history` to CSV before and
  after — don't commit it.
- No internal solver diagnostics (iteration counts, cost, etc.). Those
  are useful for perf work but not stable under environmental changes.

### Tuning the tolerances

In `crates/cybflight_sim/tests/regression_snapshot.rs`:

```rust
const TOL_POS:  Tol = Tol { rel: 0.01, abs: 1e-4 };
const TOL_TILT: Tol = Tol { rel: 0.01, abs: 1e-3 };
const TOL_SAT:  Tol = Tol { rel: 0.01, abs: 0.5  };
```

If the snapshot churns on rustc updates, loosen `rel` (probably to
0.02). If you want tighter guardrails on a specific number, tighten
`abs`. Don't commit a tolerance change without a specific justification
in the PR description.

### What happens on the first run ever

If you delete the snapshot file and run `just sim-check`, the test
panics with a clear message telling you to run `just sim-snapshot`.
This is the intended failure mode — it's explicitly not silent.

[insta]: https://insta.rs/

## Sensor simulation

`crates/cybflight_sim/src/sensors.rs` provides an `ImuModel` trait that
converts plant ground truth into an `ImuMeasurement`. Two impls today:

- `PerfectImu` — default. Returns `(body_rate, Σu/mass · ẑ)` with zero
  noise or bias. Used by every scenario unless overridden.
- `NoisyImu` — seeded ChaCha8 Gaussian noise on both channels, plus
  optional constant biases. Deterministic across runs for a fixed seed
  so assertions stay stable.

### Which controllers see the IMU

`MpcIndiController` consumes the `ImuMeasurement` (INDI is the only
sensor-consuming block in the stack). `CascadeController` and
`MpcDirectController` ignore it — they read perfect state from the plant
directly. That means **noise only affects the firmware-match topology**,
which is the interesting question. Ground-truth baselines stay as an
upper bound.

### Running the noisy autotest

```
cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
  --profile release-host --test autotest_noisy -- --nocapture
```

The test runs `mission_square` through `MpcIndiController` with a
`NoisyImu` configured for representative MEMS-IMU noise (σ=0.03 rad/s
gyro, σ=0.3 m/s² accel). Asserts only the pass criteria hold —
tight-tolerance assertions on noisy numbers would mostly catch noise-
seed changes, not control regressions, which isn't what you want. Use
it to confirm INDI still rides through realistic sensor noise after
changes to the inner loop.

### Noisy scenarios and the regression snapshot

Noisy rows *are* in `regression_snapshot.json` — labeled with a
`_noisy` suffix so you can see at a glance which rows are noise-
sensitive. The reasons it works:

- `ChaCha8Rng` commits to bit-exact output across `rand_chacha`
  versions for a fixed seed, and its algorithm is pure integer so no
  platform-FP difference enters the stream.
- `Cargo.lock` pins `rand`, `rand_chacha`, and everything else. This is
  a binary workspace, so the lockfile is committed.
- The noise → measurement transform (`ln`, `cos` in Box-Muller) is the
  same kind of IEEE-deterministic `f32` that the control math already
  relies on. Noisy rows inherit the control rows' fragility level; they
  don't add a new one.
- RMS averaged over 80 000 ticks is a central-limit statistic — it
  moves less than the clean `peak_pos_err_m` column does under the same
  f32 wobble.

What *would* invalidate a noisy row without touching control code:
- Replacing the Box-Muller sampler with a different algorithm.
- Changing the `NoisyImu` draw order (e.g. alternating gyro/accel axes).
- A rare `libm` / `rustc` intrinsic change that hits `ln`/`cos` on
  `f32` specifically.

Treat the first two as "you changed `NoisyImu`, snapshot will move —
regenerate and document in the commit". The third is the same category
as any tolerance-tripping environmental drift.

### Adding a new sensor model

1. Implement `ImuModel` (see `NoisyImu` for the Box-Muller pattern).
2. Wire it with `Scenario::...(...).with_imu(Box::new(MyModel::new()))`.
3. Add an autotest if the model captures a specific failure mode (e.g.
   large bias, latency spike, dropout).
4. Do **not** add noisy rows to the regression snapshot; see above.

`GpsModel` shipped (`PerfectGps`, `NoisyGps`, `OutageGps`, `FaultedGps`
in `sensors.rs`, driving the `mission_square_gps` scenario /
`autotest_gps.rs` / `just sim-run GPS=…`); a `ViconModel` would follow
the same pattern.

## Vehicle parameters in the sim

`Scenario` owns `vehicle_params: FirmwareConfig`. That's the **single
source of truth** for everything downstream:

- `QuadPlant::new(scenario.vehicle_params.clone(), ...)` — simulated
  rigid body
- `MpcIndiController::from_params(&scenario.vehicle_params)`, etc. —
  controller gains / limits
- `Scenario::mission(...)` internally derives a `QuadPlanningConfig`
  from those same params, so the planned trajectory respects the
  vehicle it will fly on

Tests grab `&scenario.vehicle_params` into plant + controller factories,
so all three stay consistent by construction. You cannot plan a
trajectory against one set of limits and fly it on a plant with
different mass.

### Defaults + overrides

`Scenario::hover`, `point_to_point`, `mission` fill
`vehicle_params` with `default_vehicle()` — the FROZEN sim baseline
loaded from `vehicles/sim_baseline.yaml` through the same
`vehicle-yaml` loader the firmware bake uses. The baseline
deliberately lags the flight tune (`vehicles/sakura_vicon.yaml`);
compare them with a plain file diff. To adopt a new tune, edit
`sim_baseline.yaml` and regenerate the regression snapshot in the
same commit.

For tuning sweeps or "what if" tests, the `_with_params` variants take
an explicit `FirmwareConfig`:

```rust
let vp = tweaked_vehicle(|p| p.airframe.body.mass_kg *= 1.5);
let scenario = Scenario::mission_with_params("heavy_square", vp, start, &wps);
let controller = MpcIndiController::from_params(&scenario.vehicle_params);
```

`tweaked_vehicle(|p| ...)` is the recommended one-field override
helper; it starts from `default_vehicle()` and applies your closure.
Both `default_vehicle()` and `tweaked_vehicle()` are re-exported from
the crate root.

### Why not mutable after construction?

`point_to_point` and `mission` bake the trajectory into the scenario at
construction time by running the MINCO/BFGS planner against the
supplied params. Letting callers mutate `vehicle_params` after the fact
would leave the trajectory stale (planned under old limits, flown under
new ones). If you need a different vehicle, rebuild the scenario with
the new params.

`hover` has no trajectory, so in principle it could allow post-hoc
mutation — but we keep the API uniform across all three constructors
to avoid a "works sometimes" trap.

## Simulation runner (`just sim-compare`, `just sim-run`)

`just sim-compare` runs all four sim scenarios through all three
controllers, plus the sensors-in-the-loop noisy and GPS variants, and
prints a 16-row comparison table. This is the
exploratory tool — use it when you want to *look* at how behavior
changed, not gate a PR on it.

`just sim-run SCENARIO=... CONTROLLER=... VIZ=1` runs one scenario
through one controller and (optionally) streams to a local rerun
viewer.

Both recipes use the `release-host` profile, which inherits from
`release` but disables LTO and loosens `codegen-units` so incremental
rebuilds don't pay the cross-crate LTO tax that embedded firmware
needs. Embedded builds (`just build`, `just flash`) are unaffected.

## Bench rate checks (`imurate`, `indistat`)

Two shell verbs measure what the inner loop actually does, as opposed to
what it was compiled or configured to do. Run both after any change to
the IMU driver, the SPI/bus setup in `board_init`, the IMU reader task,
or the INDI step — and before trusting a blackbox log for sysid.

```
> imurate
measuring for 1000 ms (build expects 8000 Hz; INDI at /4 = 2000 Hz)...
  imu1: 7922 samples / 1.000 s = 7921.8 Hz (-0.98 %)
> indistat          # first call discards what accumulated since boot
> indistat          # wait a second, then this is the number
indi: 2001 iters, step avg 61.3 us, max 118.0 us, loop period max 512.0 us
```

- `imurate` counts publishes on `IMU_1` (`Lagged(n)` counts as `n`) over
  one second against the MCU clock — a real ODR measurement, ±1 % is the
  chip's oscillator tolerance. Anything well below the build's ODR means
  the *reader* is losing samples: bus time (`bytes × 8 / f_spi` vs the
  period), a second transaction per sample, or a task on the same
  executor (INDI) running long enough to miss the DRDY pulse.
- `indistat` is a DWT cycle count around one INDI iteration (after the
  `indi_ctrl_div` gate, up to the motor publish). `step max` must stay
  well under `1 / (IMU_ODR_HZ / indi_ctrl_div)`; `loop period max` shows
  the worst gap between iterations.

History: until 2026-08-24 every build ran the IMU at ~2.5 kHz because
the SPI bus was 1 MHz (136 µs of bus time per sample against a 125 µs
period), and INDI — designed for 8 kHz — silently ran at that rate too.
Raising the clock exposed the INDI step (> 125 µs) as the next limit,
which is why the control rate is now a separate param. When a P10 task
saturates its executor the symptom is an IWDG reset loop (slow LED, no
USB), indistinguishable from a panic without the `postmortem` feature.

The per-sample reader is the interim design; the intended end state is
the ICM FIFO drained at the control rate — see [imu_fifo.md](imu_fifo.md).
