# Cybflight - Embedded Flight Controller Firmware

## Project Overview

Rust no_std flight controller firmware targeting STM32H743 boards.
Uses Embassy async runtime, defmt logging, and embassy-stm32 HAL.

## Architecture

**Read [docs/architecture.md](docs/architecture.md) before making any changes.**

Key rules:

- **Channels are the abstraction boundary** between board-specific and board-agnostic code.
  Control logic reads from static channels (`FUSED_IMU`, `BARO`, etc.) and never
  touches hardware directly.
- **BSP crates are pin-mapping only.** They export a `Board` struct, `init()`, const
  capability flags, and interrupt bindings. No driver or control logic dependencies.
- **`board_init/` is the ONLY place boards diverge.** One module per board handles all
  wiring: SPI/I2C bus construction, driver init, task spawning, sensor topology.
- **`#[cfg(feature = "board_xxx")]` is confined to BSP imports and board_init dispatch.**
  Control logic uses `const` flags (`bsp::HAS_BARO`, `bsp::IMU_COUNT`) for zero-cost
  compile-time dead code elimination instead of cfg.
- **No trait objects or dyn dispatch for sensors.** Channels carry concrete types.
  All sensor tasks are monomorphized.

## Building

Prefer `just` first

```
just vehicles                              # what can I build?
just build                                 # uses VEHICLE= from .env
just build sakura_bench_hunter_indoor      # or name one explicitly
just flash sakura_bench_hunter_indoor      # build + DFU the same one
just check-all
just test
just size sakura_bench_leader_8khz         # static RAM + headroom (see architecture.md rule 7)
```

`build`, `flash`, `print-features` and `param-sync` all take the same
optional positional vehicle, defaulting to `VEHICLE=` in `.env`. Naming a
vehicle that does not exist fails immediately, listing the valid names —
it never composes a feature list for a vehicle that isn't there. (Because
`build` takes an argument, `just build test` means "build the vehicle
named test", not "build then test".)

The cargo feature list is derived from the **vehicle YAML's `build:`
section** (`vehicles/<vehicle>.yaml`) by `tools/vehicle_features.sh` —
`just print-features [vehicle]` shows the resolution and is the dry run
for a build. Env vars (`BOARD=`, `OUTER_LOOP=`, `POS_SOURCE=`, …) remain
dev overrides; `build.rs` cross-checks features against the YAML and
fails a mismatched board↔vehicle pairing.

```sh
# Bare cargo check works (bakes the fallback vehicle with a warning):
cargo check -p cybflight

# Hand-composed feature sets are for the check-all matrix only:
cargo check -p cybflight --no-default-features \
    --features board_foxeerh743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc
```

## Crate Map

| Crate | Path | Purpose |
|---|---|---|
| `cybflight` | `crates/cybflight/` | Main firmware binary + library |
| `cybflight-core` | `crates/cybflight_core/` | no_std control/estimation/planning + param schema (`params.rs`) |
| `cybflight-sim` | `crates/cybflight_sim/` | Host sim, regression snapshot, param_doc generator |
| `vehicle-yaml` | `crates/vehicle_yaml/` | Shared YAML loaders: vehicle config + missions (bake + sim) |
| `cybflight-params-derive` | `crates/params_derive/` | `#[derive(Params)]` registry proc-macro |
| `bsp-sakurah743` | `crates/bsp/sakurah743/` | SAKURAH743 pin mapping |
| `bsp-foxeerh743` | `crates/bsp/foxeerh743/` | FOXEERH743 pin mapping |
| `bsp-types` | `crates/bsp/types/` | Shared BSP types (SensorAlign, MotorMeta, etc.) |
| `cybflight-drivers` | `crates/drivers/` | Hardware-agnostic drivers (Icm426xx, Led, Beeper) |

## Configuration & Parameters

**Read [docs/architecture.md](docs/architecture.md) "The Configuration
Plane".** One YAML per vehicle (`vehicles/*.yaml`: `build:` features,
required `airframe` identity incl. `thrust_model`, `default_mission`,
`tuning:` overrides) + one YAML per mission (`missions/*.yaml`), both
baked at compile time. Runtime: `param set/save/diff` over USB,
`just param-sync` merges the tune back into the YAML. Adding a param =
one attributed field in `cybflight_core/src/params.rs` + `VERSION` bump +
`just params-doc`. Pin hardware-coupled values per vehicle (the bake
warns about unpinned keys).

## Adding a New Board

See [docs/architecture.md](docs/architecture.md) "How to Add a New Board" section.

## Adding a New Sensor

See [docs/architecture.md](docs/architecture.md) "How to Add a New Sensor Type" section.

## Conventions

- All BSPs are derived from Betaflight target config.h files (see each BSP crate's comments for the upstream target)
- Pin assignments must be verified against both the Betaflight config and the
  STM32H743 datasheet (AF table, DMAMUX request IDs)
- Drivers in `crates/drivers/` are generic over `embedded-hal-async` traits,
  never import `embassy-stm32` directly
- Use `defmt` for all logging, never `println` or `log` crate
- Prefer `nalgebra` for vector math; language arrays (and nested arrays) for
  vector math is acceptable only in the initial commit porting code from C/C++.
  In this case, claude must clearly annotate TODO comments to replace with
  nalgebra in a future refactor.

## Sim regression snapshot

`crates/cybflight_sim/tests/regression_snapshot.json` is a committed
golden file (insta-style) of the sim comparison metrics (16 rows: the 4×3 scenario/controller grid plus contouring/GPS/noisy variants).

- `just sim-check` — assert current numbers match the snapshot within a
  tight tolerance. Run this after any change to `cybflight-core` or
  `cybflight-sim` to catch unintended behavior drift.
- `just sim-snapshot` — regenerate the file. Only run this when you
  **intend** to change behavior (retuned a gain, swapped a solver,
  edited `vehicles/sim_baseline.yaml`, etc.). Then `git diff` the
  snapshot, review the numeric change, and commit it **in the same
  commit** as the code change so the reviewer sees cause (code) and
  effect (numbers) together. The sim's vehicle/tuning baseline is the
  frozen `vehicles/sim_baseline.yaml` (shared `vehicle-yaml` loader
  with the firmware bake) — it deliberately lags the flight tune.

Full workflow and rationale in [docs/HACKING.md](docs/HACKING.md#sim-regression-snapshot).

## Tooling

In environments where the username is `hs293go`, `rg` and `fd` are available as modern alternatives to `grep` and `find`. `fzf` is also available.
