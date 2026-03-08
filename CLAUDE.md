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

```sh
# Default board (sakurah743) + default RC protocol (CRSF):
cargo build -p cybflight

# Specific board + RC protocol:
cargo build -p cybflight --no-default-features --features board_foxeerh743,rx_crsf
cargo build -p cybflight --no-default-features --features board_sakurah743,rx_ghst
```

## Crate Map

| Crate | Path | Purpose |
|---|---|---|
| `cybflight` | `crates/cybflight/` | Main firmware binary + library |
| `bsp-sakurah743` | `crates/bsp/sakurah743/` | SAKURAH743 pin mapping |
| `bsp-foxeerh743` | `crates/bsp/foxeerh743/` | FOXEERH743 pin mapping |
| `bsp-types` | `crates/bsp/types/` | Shared BSP types (SensorAlign, MotorMeta, etc.) |
| `cybflight-drivers` | `crates/drivers/` | Hardware-agnostic drivers (Icm426xx, Led, Beeper) |

## Adding a New Board

See [docs/architecture.md](docs/architecture.md) "How to Add a New Board" section.

## Adding a New Sensor

See [docs/architecture.md](docs/architecture.md) "How to Add a New Sensor Type" section.

## Conventions

- All BSPs are derived from Betaflight target config.h files stored in `reference/`
- Pin assignments must be verified against both the Betaflight config and the
  STM32H743 datasheet (AF table, DMAMUX request IDs)
- Drivers in `crates/drivers/` are generic over `embedded-hal-async` traits,
  never import `embassy-stm32` directly
- Use `defmt` for all logging, never `println` or `log` crate
