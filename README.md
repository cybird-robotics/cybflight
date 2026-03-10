# CYBFLIGHT

Embedded flight controller firmware for Cybird autopilot.

## Tooling

The latest `rust-analyzer` will fail to analyzer nalgebra code.

You must install versions earlier than 1.94.0. In vscode, go to the `rust-analyzer` extension page and select install specific version; `0.3.2449` was tested to work.

## Getting Started

### Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Build

```bash
# Default board (sakurah743) + default RC protocol (CRSF) + UART logging:
cargo run --release -p cybflight

# Specific board + RC protocol:
cargo run --release -p cybflight --no-default-features --features board_foxeerh743,rx_ghst,defmt_uart
cargo run --release -p cybflight --no-default-features --features board_sakurah743,rx_ghst,defmt_uart

# Production build (no UART logging overhead):
# Set `DEFMT_LOG=off` in '.cargo/config.toml`
cargo run --release -p cybflight --release --no-default-features --features board_sakurah743,rx_crsf
```

### Cargo Features

| Feature | Default | Description |
|---|---|---|
| `board_sakurah743` | yes | Select the SAKURAH743 BSP |
| `board_foxeerh743` | no | Select the FOXEERH743 BSP |
| `rx_crsf` | yes | CRSF (ELRS/TBS) RC protocol |
| `rx_ghst` | no | GHST (ImmersionRC) RC protocol |
| `defmt_uart` | yes | UART-based defmt logging on USART3 (SAKURAH743: PD8, FOXEERH743: PB10) |

### Flash via USB DFU

```bash
dfu-util -l
# Expected: "Found DFU: [0483:df11] ..."

dfu-util -a 0 -s 0x08000000:leave -D target/thumbv7em-none-eabihf/release/cybflight.bin
```

### Connect to USB Shell
```bash
minicom -D /dev/ttyACM0 -b 115200
```

### View `defmt` Serial Log
```bash
cargo install defmt-print

stty -F /dev/ttyTHS0 921600 raw -echo
defmt-print -e target/thumbv7em-none-eabihf/release/cybflight < /dev/ttyTHS0
```

## References
* [Embassy Book](https://embassy.dev/book)