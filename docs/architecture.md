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
|  FUSED_IMU, BARO, MAG, etc.                               |
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
|         | |  Mpu6x00,   | |  mpu_reader,    |
|         | |  Dps310,..) | |  imu_fusion,..) |
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
  drivers/                # Hardware-agnostic: Icm426xx, Dps310, Led, Beeper
  cybflight/
    src/
      main.rs             # Entry: bsp::init() -> board_init::init() -> spawn controller
      lib.rs              # cfg-gated BSP re-export, shared types, apply_alignment
      board_init/
        mod.rs            # cfg-dispatch: pub use {board}::init
        sakurah743.rs     # SPI4->IMU1, SPI1->IMU2, spawn fusion, no baro
        foxeerh743.rs     # SPI2->IMU1 (probe: ICM or MPU), I2C1->baro, no fusion
      sensors/
        mod.rs            # Channel defs: FUSED_IMU, BARO, MAG
        imu.rs            # imu_reader_task, imu_fusion_task (reusable)
        baro.rs           # baro_reader_task (reusable)
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
| 1 IMU | Spawns 1 `imu_reader_task` -> writes directly to `FUSED_IMU` |
| 2 IMUs | Spawns 2 `imu_reader_task` -> `RAW_IMU` -> `imu_fusion_task` -> `FUSED_IMU` |
| Has baro | Inits baro driver, spawns `baro_reader_task` -> `BARO` channel |
| No baro | Does nothing. `BARO` channel stays empty. |
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
default = ["board_sakurah743"]
board_sakurah743  = ["dep:bsp-sakurah743"]
board_foxeerh743  = ["dep:bsp-foxeerh743"]

[dependencies]
bsp-sakurah743  = { path = "../bsp/sakurah743",  optional = true }
bsp-foxeerh743  = { path = "../bsp/foxeerh743",  optional = true }
```

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
pub static FUSED_IMU: Channel<CriticalSectionRawMutex, ImuSample, 4> = Channel::new();
pub static BARO: Channel<CriticalSectionRawMutex, BaroSample, 4> = Channel::new();
pub static MAG: Channel<CriticalSectionRawMutex, MagSample, 4> = Channel::new();
```

### Consuming optional sensors

Downstream code uses const flags for zero-cost optional sensor handling:

```rust
// In flight controller:
let imu = sensors::FUSED_IMU.receive().await;  // always present

if bsp::HAS_BARO {
    // Entire block eliminated at compile time when HAS_BARO=false.
    // No runtime branch, no dead code in flash.
    if let Ok(baro) = sensors::BARO.try_receive() {
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
#[embassy_executor::task(pool_size = 2)]
async fn imu_reader_task(source: u8, mut imu: ImuDev, align: SensorAlign, ...) { }

// Only spawned on dual-gyro boards. On single-gyro boards, this function is
// never referenced and stripped by the linker.
#[embassy_executor::task]
async fn imu_fusion_task(...) { }
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
2. **Add channel** in `sensors/mod.rs` — e.g. `pub static GPS: Channel<...>`
3. **Add reader task** in `sensors/{sensor}.rs` — generic, reusable
4. **Add const flag** to `bsp-types` or BSP — e.g. `pub const HAS_GPS: bool`
5. **Wire in board_init/** — only for boards that have the sensor
6. **Consume in control/** — guarded by `if bsp::HAS_GPS { ... }`

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
