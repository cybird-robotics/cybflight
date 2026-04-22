# Hacking on cybflight

Notes for contributors. Architecture and design live in
[architecture.md](architecture.md); this file is about the workflows you'll
actually touch while changing code.

## Sim regression snapshot

`crates/cybflight_sim/tests/regression_snapshot.json` is a **committed**
JSON file with the expected summary metrics for the 12-row sim comparison
(4 scenarios × 3 controllers × 5 metrics). `cargo test` runs the sim and
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

A `GpsModel` / `ViconModel` follows the same pattern once ESKF-in-the-
loop is wired — that's the next milestone and it will consume
`PositionMeasurement` at ~10 Hz.

## Simulation runner (`just sim-compare`, `just sim-run`)

`just sim-compare` runs all four sim scenarios through all three
controllers and prints a 12-row comparison table. This is the
exploratory tool — use it when you want to *look* at how behavior
changed, not gate a PR on it.

`just sim-run SCENARIO=... CONTROLLER=... VIZ=1` runs one scenario
through one controller and (optionally) streams to a local rerun
viewer.

Both recipes use the `release-host` profile, which inherits from
`release` but disables LTO and loosens `codegen-units` so incremental
rebuilds don't pay the cross-crate LTO tax that embedded firmware
needs. Embedded builds (`just build`, `just flash`) are unaffected.
