# Cybflight Architecture

This document defines the firmware architecture for cybflight. All agents working on
this project MUST follow these patterns. Update this document when the design changes.

## Layer Diagram

```
+-----------------------------------------------------------+
|                 Control Layer (board-agnostic)             |
|  Flight controller, PID, motor mixing, state estimation   |
|  Reads ONLY from sensor channels. Never touches hardware. |
+-----------------------------+-----------------------------+
                              | static channels (the boundary)
+-----------------------------+-----------------------------+
|                 Sensor Channels                            |
|  IMU_1, IMU_2, BARO_1, BARO_2, MAG_EXT, MAG_INT, etc.     |
|  Defined in sensors/mod.rs. Always the same type/shape.   |
+-----------------------------+-----------------------------+
                              |
+-----------------------------+-----------------------------+
|               Board Init (cfg-gated, per-board)           |
|  One module per board in board_init/                       |
|  Constructs buses, inits drivers, spawns sensor tasks,    |
|  decides topology (fusion vs passthrough).                 |
|  THIS IS THE ONLY PLACE cfg(feature="board_xxx") DIVERGES |
+------+-----------+------------------+---------------------+
       |           |                  |
+------+--+ +------+------+ +--------+--------+
|   BSP   | |   Drivers   | |  Sensor Tasks   |
| (pins)  | | (Icm426xx,  | | (icm_reader,    |
|         | |  Dps310,    | |  baro_reader,   |
|         | |  Icp20100,  | |  mag_reader,    |
|         | |  Ist8310,..)| |  imu_fusion,..) |
+---------+ +-------------+ +-----------------+
```

## Core Principle: Channels Are the Abstraction Boundary

Static embassy channels separate board-specific code from board-agnostic code.

- **Below channels**: BSP pin wiring, SPI/I2C bus construction, driver init, sensor
  reader tasks. This code varies per board and lives in `board_init/`.
- **Above channels**: Flight controller, state estimation, motor output, telemetry.
  This code is written once and never knows what hardware exists.

The flight controller always sees ONE fused IMU stream, regardless of whether the
board has 1 or 2 physical IMUs. Optional sensors (baro, mag) use channels that
remain empty on boards without that sensor.

## Crate & Module Structure

```
crates/
  bsp/
    types/                # Shared types: SensorAlign, MotorMeta, DmaHint, etc.
    sakurah743/           # Board struct, pin mapping, init(), const capabilities
    foxeerh743/           # Board struct, pin mapping, init(), const capabilities
  drivers/                # Hardware-agnostic: Icm426xx, Dps310, Icp20100, Ist8310, Qmc5883l, Led, Beeper
  cybflight/
    src/
      main.rs             # Entry: bsp::init() -> board_init::init() -> spawn controller
      lib.rs              # cfg-gated BSP re-export, shared types, apply_alignment
      board_init/
        mod.rs            # cfg-dispatch: pub use {board}::init
        sakurah743.rs     # SPI4->IMU1, SPI1->DPS310, I2C1->ICP20100+IST8310, I2C2->QMC5883L
        foxeerh743.rs     # SPI2->IMU1 (probe: ICM or MPU), I2C1->DPS310+QMC5883L
      sensors/
        mod.rs            # Channel defs: IMU_1, IMU_2, BARO_1, BARO_2, MAG_EXT, MAG_INT
        imu.rs            # imu_reader_task, imu_fusion_task (reusable)
        baro.rs           # baro_reader_task (DPS310 SPI/I2C, ICP20100)
        mag.rs            # mag_reader_task (QMC5883L, IST8310)
      control/
        mod.rs            # Flight controller reads channels, board-agnostic
      status.rs           # Status LED task
      usb_serial.rs       # USB CDC task
```

## BSP Crate Conventions

Every BSP crate MUST export:

### Required Exports

```rust
pub use embassy_stm32 as hal;            // HAL re-export

pub const BOARD_NAME: &str = "...";       // Betaflight target name
pub const MANUFACTURER_ID: &str = "...";  // Betaflight manufacturer ID
pub const BEEPER_INVERTED: bool = ...;

pub struct Board { ... }                  // All board peripherals
pub fn init() -> Board;                   // HAL init + peripheral construction

// Interrupt bindings
pub struct ExtiIrqs;                      // EXTI bindings for gyro DRDY
pub struct UsbIrqs;                       // USB OTG interrupt binding
```

### Const Capability Flags

Every BSP MUST export these constants. They enable compile-time dead code
elimination in firmware — `if bsp::HAS_BARO { ... }` where HAS_BARO is
`const false` compiles to zero instructions.

```rust
pub const IMU_COUNT: usize = 1;     // 1 or 2. Controls fusion vs passthrough.
pub const HAS_BARO: bool = false;   // DPS310 or similar
pub const HAS_MAG: bool = false;    // IST8310 or similar
pub const HAS_OSD: bool = false;    // MAX7456
pub const HAS_FLASH: bool = false;  // W25Q128FV dataflash
pub const HAS_SDCARD: bool = false; // SDMMC
pub const LED_COUNT: usize = 1;     // Number of status LEDs
```

### BSP Dependencies

BSP crates depend ONLY on:
- `embassy-stm32` (HAL)
- `bsp-types` (shared enums/structs)
- `cybflight-drivers` (only for types like `Beeper` that BSP constructs)

BSPs MUST NOT depend on embassy-executor, embassy-sync, sensor drivers, or
control logic. They are pin-mapping crates only.

## Board Init Pattern

Each board has one module in `board_init/` with a single async `init` function:

```rust
// board_init/foxeerh743.rs
pub async fn init(spawner: &Spawner, board: bsp::Board) {
    // 1. Construct SPI/I2C buses from BSP pin structs
    // 2. Init drivers (Icm426xx, Dps310, etc.)
    // 3. Spawn sensor reader tasks -> feed channels
    // 4. Spawn status LED task
    // 5. Spawn USB serial task
}
```

### How board_init handles hardware differences

| Scenario | Board init does |
|---|---|
| 1 IMU | Spawns 1 `imu_reader_task` -> `IMU_1` channel. `IMU_2` stays empty. |
| 2 IMUs | Spawns 2 `imu_reader_task` -> `IMU_1` + `IMU_2` channels |
| Has baro | Inits baro driver, spawns `baro_reader_task` -> `BARO_1`/`BARO_2` channel |
| No baro | Does nothing. `BARO_1`/`BARO_2` channels stay empty. |
| Has mag | Inits mag driver, spawns `mag_reader_task` -> `MAG_EXT`/`MAG_INT` channel |
| No mag | Does nothing. `MAG_EXT`/`MAG_INT` channels stay empty. |
| Shared I2C bus | Inits ALL devices before spawning ANY tasks. See [sensor_bus_sharing.md](sensor_bus_sharing.md) |
| 1 LED | Spawns status task with 1 LED |
| 3 LEDs | Spawns status task with 1 primary LED, turns off extras |

### cfg dispatch

```rust
// board_init/mod.rs
#[cfg(feature = "board_sakurah743")]
mod sakurah743;
#[cfg(feature = "board_sakurah743")]
pub use sakurah743::init;

#[cfg(feature = "board_foxeerh743")]
mod foxeerh743;
#[cfg(feature = "board_foxeerh743")]
pub use foxeerh743::init;
```

Only one board module compiles per build. The others are excluded entirely.

## BSP Selection via Cargo Features

```toml
# crates/cybflight/Cargo.toml
[features]
board_sakurah743  = ["dep:bsp-sakurah743"]
board_foxeerh743  = ["dep:bsp-foxeerh743"]

[dependencies]
bsp-sakurah743  = { path = "../bsp/sakurah743",  optional = true }
bsp-foxeerh743  = { path = "../bsp/foxeerh743",  optional = true }
```

In practice you never hand-compose the feature list: the **vehicle YAML's
`build:` section selects it** (see "The Configuration Plane" below), and
`just build [<vehicle>]` derives `--features` via
`tools/vehicle_features.py`. The vehicle is the recipe's optional
positional argument, defaulting to `VEHICLE=` in `.env`; `just vehicles`
lists them, and an unknown name is a hard error rather than a fallback.

The `cybflight` lib.rs re-exports the selected BSP so other modules use `crate::bsp`:

```rust
// lib.rs
#[cfg(feature = "board_sakurah743")]
pub use bsp_sakurah743 as bsp;
#[cfg(feature = "board_foxeerh743")]
pub use bsp_foxeerh743 as bsp;

pub use bsp::hal;
```

Build for a specific board:
```sh
cargo build -p cybflight --no-default-features --features board_foxeerh743
```

## Sensor Channels

Defined in `sensors/mod.rs`. These are the ONLY interface between hardware and
control logic.

```rust
pub static IMU_1: PubSubChannel<CriticalSectionRawMutex, Imu, ...>;
pub static IMU_2: PubSubChannel<CriticalSectionRawMutex, Imu, ...>;
pub static VEHICLE_ATTITUDE: PubSubChannel<CriticalSectionRawMutex, VehicleAttitude, ...>;
pub static BARO_1: PubSubChannel<CriticalSectionRawMutex, BaroSample, ...>;
pub static BARO_2: PubSubChannel<CriticalSectionRawMutex, BaroSample, ...>;
pub static MAG_EXT: PubSubChannel<CriticalSectionRawMutex, MagSample, ...>;
pub static MAG_INT: PubSubChannel<CriticalSectionRawMutex, MagSample, ...>;
```

### Consuming optional sensors

Downstream code uses const flags for zero-cost optional sensor handling:

```rust
// In flight controller:
// Attitude estimator subscribes to IMU_1 and publishes VEHICLE_ATTITUDE.
// Control logic reads VEHICLE_ATTITUDE — never touches IMU directly.
let att = sensors::VEHICLE_ATTITUDE.receive().await;

if bsp::HAS_BARO {
    // Entire block eliminated at compile time when HAS_BARO=false.
    // No runtime branch, no dead code in flash.
    if let Ok(baro) = sensors::BARO_1.try_receive() {
        // use altitude data
    }
}
```

## Sensor Tasks (Reusable)

Sensor tasks are generic, board-agnostic async functions. Board init decides
which to spawn and how to wire them.

```rust
// sensors/imu.rs

// pool_size=2 supports up to dual-gyro. Single-gyro boards spawn 1 instance;
// the unused slot costs one task-struct of static RAM (~200 bytes), acceptable.
// Each reader publishes to its own channel (IMU_1 or IMU_2).
#[embassy_executor::task(pool_size = 2)]
async fn icm_reader_task(reader: ImuReader<IcmDev>, channel: &'static ImuChannel) { }

// Attitude estimator (Mahony) subscribes to IMU_1.
#[embassy_executor::task]
async fn mahony_task() { }
```

## Compile-Time Optimization Rules

These rules ensure the final binary contains ONLY the code for the target board:

1. **`const` flags in `if` expressions** — LLVM eliminates dead branches entirely.
   `if bsp::HAS_BARO { ... }` with `const HAS_BARO: bool = false` produces
   zero instructions. Prefer this over `#[cfg]` in control logic for readability.

2. **`#[cfg(feature = "board_xxx")]` in board_init/ and BSP imports only** —
   Keeps cfg confined to the wiring layer. Control code never uses cfg.

3. **Unreferenced task functions are stripped** — If `imu_fusion_task` is never
   spawned (single-gyro board), the linker removes it. No manual cfg needed.

4. **No trait objects or dyn dispatch for sensor interfaces** — Channels carry
   concrete types. All sensor task types are monomorphized at compile time.

5. **No `Option` wrapping for compile-time-known presence** — Don't wrap baro
   data in `Option<BaroSample>` if `HAS_BARO` is const. The const flag already
   eliminates the code path.

6. **Task pool_size can use max across all boards** — `pool_size = 2` for
   imu_reader_task is fine even on single-gyro boards. The cost is one unused
   static task struct.

7. **Rules 1–3 remove code, not RAM.** What the linker can drop depends on
   what kind of cost it is:

   | Cost | Eliminated by | Correct gate |
   |---|---|---|
   | `.text` / `.rodata` | LTO + `--gc-sections`, reliably | nothing — a `const` flag for readability |
   | `static` (`.bss` / `.data`) | only if **no surviving code names it** | **cargo feature** |
   | an `async` task's future (`POOL`) | never by const-folding — it is the union of every arm's live state, laid out before LLVM sees the flag | **cargo feature, one task body per variant** |

   A `StaticCell` is "free at rest" in flash, not in RAM: a `static` that
   any surviving code names occupies `.bss` whether or not `init()` ever
   runs, and `if CONST { small().await } else { big().await }` sizes the
   task future for `big` even when `CONST` is `true`. The mission planner
   paid 109 KiB of `.bss` this way for an online solver that a
   `const bool` had disabled (`mission_planner.rs`, `plan_online`). So:
   `cybflight-core` owns **no** `static` storage — every solver takes a
   `&mut Workspace` from its caller, which is why it can stay
   unconditionally compiled and host-testable — and every `static` and
   every subsystem gate lives in `crates/cybflight`, where a cargo feature
   can actually leave the state out. Size inline storage by the
   consumer's bound (`MincoSnapN<OFFLINE_MAX_PIECES, …>`), not by a
   global cap. `just size` reports the result.

## How to Add a New Board

1. **Create BSP crate**: `crates/bsp/{boardname}/`
   - `Cargo.toml` with `embassy-stm32`, `bsp-types`, `cybflight-drivers`
   - `src/lib.rs` with Board struct, init(), pin mappings, const capability flags
   - Verify all pins against Betaflight config.h reference

2. **Add to workspace**: Add path to root `Cargo.toml` members

3. **Add feature + dep to cybflight**: In `crates/cybflight/Cargo.toml`:
   ```toml
   board_{name} = ["dep:bsp-{name}"]
   bsp-{name} = { path = "../bsp/{name}", optional = true }
   ```

4. **Add cfg import**: In `crates/cybflight/src/lib.rs`:
   ```rust
   #[cfg(feature = "board_{name}")]
   pub use bsp_{name} as bsp;
   ```

5. **Create board init module**: `crates/cybflight/src/board_init/{name}.rs`
   - Wire SPI/I2C buses, init drivers, spawn tasks
   - Decide topology: fusion vs passthrough, which sensors present

6. **Register in board_init/mod.rs**:
   ```rust
   #[cfg(feature = "board_{name}")]
   mod {name};
   #[cfg(feature = "board_{name}")]
   pub use {name}::init;
   ```

7. **Test**: `cargo build -p cybflight --no-default-features --features board_{name}`

## How to Add a New Sensor Type

1. **Add driver** in `crates/drivers/src/` — generic over `embedded-hal-async` traits
2. **Add channel** in `sensors/mod.rs` — e.g. `pub static GPS: PubSubChannel<...>`
3. **Add reader task** in `sensors/{sensor}.rs` — generic, reusable
4. **Add const flag** to `bsp-types` or BSP — e.g. `pub const HAS_GPS: bool`
5. **Wire in board_init/** — only for boards that have the sensor
6. **Consume in control/** — guarded by `if bsp::HAS_GPS { ... }`

If the sensor shares a bus with other sensors, read
[sensor_bus_sharing.md](sensor_bus_sharing.md) for Timer-based pacing and
deferred task spawning patterns.

## Runtime Sensor Detection

Some boards ship with different sensor variants across production runs. For example,
FOXEERH743 may have an ICM42688P, MPU6000, or MPU6500 on the same SPI bus and pins.

### Probe-before-construct pattern

When a board has multiple possible sensors, `board_init` probes the WHO_AM_I register
on the **raw SPI bus** before wrapping it in a `Mutex`/`SpiDevice`:

```rust
let mut spi = Spi::new(...);
let mut cs = board.sensors.gyro1_cs;
let detected = imu::probe_imu_raw(&mut spi, &mut cs).await;

// Now wrap the bus and construct the correct driver
let spi_bus = SPI_BUS.init(Mutex::new(spi));
let dev = SpiDevice::new(spi_bus, cs);

match detected {
    Ok(DetectedImu::Icm42688P) => { /* Icm426xx::new(dev, ...) */ }
    Ok(DetectedImu::Mpu6000)   => { /* Mpu6x00::new(dev, ...) */ }
    // ...
}
```

Probing on the raw bus avoids ownership issues — drivers consume the `SpiDevice`
and DRDY pin in `new()`, so probing must happen before driver construction.

### Separate tasks per driver type

Embassy tasks cannot be generic (they need concrete types for static allocation).
Each driver gets its own reader task: `icm_reader_task` for `Icm426xx` and
`mpu_reader_task` for `Mpu6x00`. Both tasks have identical read-loop bodies but
operate on different concrete types. This follows the architecture rule: no trait
objects or dyn dispatch for sensors.

### Which boards need probing

| Board | Probing | Reason |
|---|---|---|
| SAKURAH743 | No | Fixed sensors: ICM42688P + IIM42652 |
| FOXEERH743 | Yes | Production variants: ICM42688P, MPU6000, or MPU6500 |

### Extending with new sensors

1. Add the new WHO_AM_I value to `DetectedImu` enum in `crates/drivers/src/imu/mod.rs`
2. Add the WHO_AM_I match arm to `probe_imu_raw()`
3. Create a new driver in `crates/drivers/src/imu/`
4. Add a new concrete task in `crates/cybflight/src/sensors/imu.rs`
5. Add the dispatch arm in the relevant `board_init/` module

## The Configuration Plane

Everything tunable or vehicle-specific lives in **one YAML file per
vehicle** (`vehicles/<name>.yaml`, selected by `VEHICLE=` in `.env`),
plus one YAML file per offline mission (`missions/<name>.yaml`). The
firmware never parses YAML at runtime — both are validated and **baked at
compile time** by `crates/cybflight/build.rs`, through the shared
host-side loader crate `crates/vehicle_yaml` (also used by the sim, so
firmware and sim can never drift on format).

### Vehicle YAML anatomy

```yaml
build:                      # compile-time hardware selections → cargo features
  board: sakurah743         #   a vehicle IS a PCB + wiring; build.rs
  rc_protocol: crsf         #   cross-checks these against the ACTIVE
  outer_loop: mpc           #   features and fails a mismatched pairing.
                            #   mpc = 10-state NMPC → (thrust, rate sp) → INDI;
                            #   mpc_full = 13-state NMPC → (T_d, τ_d) → INDI α
                            #   inner loop, no rate gains (docs/mpc_full_indi_plan.md);
                            #   cascade | rate = legacy outer loops
  pos_source: gps
  gps_model: ublox
  role: chaser
  imu_rate: 8khz            #   or 1khz: ICM low-noise mode, 1 kHz inner loop
  indi: yes                 #   or no: inner loop degrades to a rate controller
  plan_online: no           #   or yes: compile the online BFGS planner
                            #   (~110 KiB of solver .bss + task state)

default_mission: outdoor_splits_slow   # boot default, by mission name

airframe:                   # REQUIRED physical identity — no defaults.
  thrust_model: { type: table, table: a2rl_0114 }  # REQUIRED, baked
  mass_kg: 0.6              #   missing mass/inertia/motors/thrust model
  inertia_kg_m2: [...]      #   = BUILD ERROR ("no default mass" rule)
  max_rate_rad_s: [...]
  motors: [...]

tuning:                     # optional flat map of registry param names.
  pos_kp_x: 4.0             #   Unknown name or out-of-range value =
  m0_tau: 0.02              #   build error. Hardware-coupled keys should
  batt_nominal_v: 23.0      #   be pinned explicitly (bake warns if not).
```

### Parameter registry (schema in Rust, values in YAML)

The schema is the `#[derive(Params)]` structs in
`cybflight_core/src/params.rs` — per-subsystem groups (`airframe`,
`sensors`, `eskf`, `mahony`, `battery`, `rc`, `site`, `safety`, `indi`,
`rpm_notch`, `cascade`, `mpc`, `trajectory`, `system`) composed into
`FirmwareConfig`.

A constant that shapes flight behaviour but lives in source is a
parameter that nobody can reach. `docs/hardcoded_constants_audit.md`
sweeps the workspace for them and records, per constant, whether it
should be in the registry, is sizing that cannot be, or is a physical
fact that must not be. Consult it before adding a new tunable literal —
and before assuming an existing one is deliberate.

Group membership follows one test: *what would have to change for this
value to be wrong?* `airframe` holds what a different physical vehicle
would invalidate — rigid body, motor geometry, the identified actuator
dynamics (`m*_tau`, `m*_omega_max`, `m*_g2_*`, `m*_nonlin`) and the
sensor install extrinsics (`airframe.install`: antenna lever arm,
baseline, magnetometer hard iron). Consumer-named prefixes are avoided
deliberately: a motor time constant reads the same under any controller,
so filing it under `indi_` would have described who happens to read it
rather than what it is. `site` holds what a different *place* would
invalidate (gravity, the stick-integrator envelope); `battery` what a
different pack would. The derive
generates a flat, name-addressed registry of typed scalars with
`ParamMeta` (unit, min/max range, doc line, reboot flag). Adding a
parameter is **one struct field with an attribute** — the registry, shell
surface, YAML bake, flash persistence, and `docs/parameters.md` all
follow from it (`just params-doc` regenerates the reference).

Value layering, lowest to highest precedence:

1. **Schema defaults** (`Default` impls) — fleet-wide starting points.
2. **Vehicle YAML bake** — per-vehicle values, applied at compile time
   into `BAKED_PARAMS`. Groups are never cfg-gated out of the schema, so
   a flash image ports across feature builds.
3. **KV flash overrides** — `param set` + `param save` appends name-keyed
   records to the log in bank-2 flash sectors 6+7 (`cybflight_core::param_store`;
   replayed over the baked defaults at boot, range-validated).

The layering is per-key and flash wins, so layer 3 shadows a layer-2
edit: change a value in the vehicle YAML, re-flash, and a param you
once `param save`d keeps its stored value. That is by design, but the
log has no tombstone record — it can only say "override this key to X",
never "this key has no override" — so `param reset` + plain `param save`
records the *current* baked value rather than removing the key, and the
key stays pinned against the **next** YAML edit. `param save --prune`
rewrites the store as exactly the set differing from baked, which is the
only way a key stops being overridden. Reach for it in two places:

- A YAML edit that isn't taking effect (`param diff` names the culprits):
  `param reset <name>`, then `param save --prune`.
- After flashing a **different vehicle** onto a board. Records carry a
  name hash and a value, never a vehicle identity, so the previous
  airframe's saved mass/inertia/thrust replay onto the new one's
  defaults and pass range validation. `param reset all` + `param save
  --prune` is the clean slate.

Pruning costs a ~1-2 s blocking erase (watchdog-extended, refused while
armed) and reuses the compaction rewrite path, so plain `param save`
stays erase-free for routine tuning.

Range validation runs at all three entry points (shell `param set`, YAML
bake, flash replay); consumers additionally guard structurally
(finite, > 0) and **degrade, never panic** — a config problem must not
boot-loop the flight controller (see `docs/safety_protocol.md`).

The bench→git loop: tune over USB (`param set`, hot-reload applies
disarmed), `param save`, then `just param-sync` merges `param diff
--yaml` back into the vehicle file for review + commit. The firmware
never writes YAML. Syncing is also the *preferred* fix for a shadowed
YAML value — the flash tune is usually the value you actually want, and
once the two layers agree the shadowing is moot.

### Runtime params vs compile-time features

The boundary, decided per knob and recorded here so it doesn't drift:

- **Runtime params** (both variants always compiled): controller gains
  and weights, loop rates (`mpc_rate_hz`, `cascade_rate_hz`), sampler
  selection (`sampler_kind`), GPS velocity fusion (`gps_fuse_vel` — a
  *policy*, not a hardware fact: it works with any receiver and a
  multipath site is reason enough to turn it off), `peer_pose_en`,
  battery facts, RPM-notch config.
- **Compile-time features, selected by the YAML `build:` section**.
  Two things qualify a knob for this list.

  *It selects code*: board (linker/peripheral singletons), RC protocol
  (physically different wiring), outer-loop controller (~85 KB dead BSS
  if both compiled + safety-critical arming branches), mission-planning
  schema (`plan_online` — ~110 KiB of solver `.bss` + task-future state
  that a `const bool` cannot remove; see optimization rule 7), position source,
  GPS driver (monomorphized, no-dyn rule), gimbal role (USART2
  ownership), IMU rate (`imu_rate: 8khz|1khz` → `imu_1khz` — programs
  hardware registers at init and re-times the IMU path; every
  IMU-rate const derives from `rates::IMU_ODR_HZ` at compile time,
  keeping the default 8 kHz build bit-identical. The *control* rate is
  a separate, runtime choice: `indi_ctrl_div` (vehicle param, reboot)
  makes INDI step on every Nth IMU sample — 8 kHz / 4 = 2 kHz on the
  8 kHz vehicles, because an INDI step does not fit the 125 µs an
  8 kHz tick allows on the control executor, and 2 kHz is far above the
  motor dynamics it closes the loop around. Sensor rate buys filter
  margin and sysid data; control rate buys loop bandwidth — they are
  not the same number), INDI on/off
  (`indi: yes|no` → `indi_off` — selects which control law flies the
  airframe, which also decides how `indi_rate_*` must be tuned; a
  runtime toggle would let a `param set` swap the law under an armed
  vehicle, and the two laws want different gains).

  *Or it is an immutable hardware fact whose consistency with another
  `build:` knob is worth enforcing at compile time*: `gps_dual_antenna`
  (is ANT2 fitted?). It gates almost no code — `GPS_HAS_HEADING` already
  dead-codes the fusion path — but it must agree with `gps_model`, and
  only here can that be a build error instead of a boot warning on a
  link nobody watches. The pairing is checked in `BuildYaml::validate`.
  Note the split it implies, which mirrors `gps_model` vs `gps_ant_*`:
  the *capability* is compile-time, the *calibration* that goes with it
  (`gps_base_*`, the baseline direction) stays a runtime param.
- **Env-only dev knobs** (not vehicle facts): `DEFMT_UART`, `ESTIMATOR`
  (link-time singletons — cannot be runtime by construction).

Env vars of the `build:` knob names (`BOARD=`, `OUTER_LOOP=`, …) remain
as deliberate dev overrides; `vehicle_features.py` warns on divergence
and the build.rs guard downgrades to a warning for overridden knobs —
except `board`, which never downgrades (flashing one board's pin mapping
with another vehicle's airframe params is the exact hazard the guard
exists to kill).

### Missions

Offline-planned trajectories are data files: one `missions/<name>.yaml`
per profile, in the planner's native format (`start` / `waypoints` /
per-segment `durations`, or absolute `timestamps`). The bake converts,
validates (finite, matching lengths, strictly increasing timing, piece
cap), and generates the `PROFILES` table `include!`d by
`control/offline_mission.rs`. Adding a mission = dropping the planner's
output file in `missions/` and rebuilding. Select at runtime with
`mission set <name>` (persisted via `mission_profile`; note the persisted
value is an index into the *name-sorted* table — re-select after changing
the mission set).

### How to add a parameter

1. Add one field to the right `*Params` struct in
   `cybflight_core/src/params.rs` with a `#[param(key = "...", unit,
   min, max[, reboot])]` attribute and a doc comment (first line becomes
   the generated description).
2. Bump `VERSION` (blob layout changes with `PARAM_COUNT`).
3. Update `scaffold()` / `tests_support::test_config` (compile-enforced)
   and the name-presence test.
4. Consume it; if hardware-coupled, pin it in the vehicle YAMLs and add
   it to `HW_COUPLED_KEYS` in `build.rs`.
5. `just params-doc` and commit the regenerated `docs/parameters.md`.

## Future Improvements

- **D-Cache**: The Cortex-M7 data cache is currently disabled to avoid DMA
  coherency issues (DMA buffers must be cache-line aligned and manually
  invalidated/cleaned around transfers). Enabling it requires placing DMA
  buffers in a non-cacheable MPU region or using cache maintenance operations
  in every DMA path. This is worth revisiting once the DMA usage patterns
  stabilize.
- **ITCM/DTCM linker sections**: Hot-path code and data could be placed in
  tightly-coupled memory for deterministic zero-wait-state access.

## Design Decisions & Rationale

### Why channels, not traits?

Traits require associated types for every peripheral variant (SPI1 vs SPI4,
different DMA channels). On STM32H7 with embassy, the concrete SPI type is
actually the same (`Spi<'static, Async, Master>`) regardless of peripheral
instance, but this isn't guaranteed across MCU families. Channels provide a
clean cut without type gymnastics and are natural for async embedded (tasks
already communicate via channels/signals).

### Why cfg-gated board_init, not a BSP trait?

`#[embassy_executor::main]` doesn't support generics. A trait-based approach
would require either a macro-generated entry point or the BSP to own the main
function. The cfg approach is simpler, explicit, and the board_init module is
the only place per-board code exists — it's a small, bounded cost per board.

### Why const flags, not cfg features for sensor presence?

`const` flags are:
- Defined in one place (the BSP crate)
- Checked by the compiler identically to cfg (dead code elimination)
- Readable in normal Rust (no `#[cfg(...)]` attributes in control logic)
- Don't require threading features through Cargo dependency chains

`#[cfg]` is reserved for the BSP crate selection itself (which crate to compile).
