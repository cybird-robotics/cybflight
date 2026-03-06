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
cargo run --release
```

This compiles the firmware and runs `tools/build.sh` to produce a raw binary at
`target/thumbv7em-none-eabihf/release/cybflight.bin`.

## Flash via USB DFU

```bash
dfu-util -l
# Expected: "Found DFU: [0483:df11] ..."

dfu-util -a 0 -s 0x08000000:leave -D target/thumbv7em-none-eabihf/release/cybflight.bin
```

## Debug via USB Serial
```bash
minicom -D /dev/ttyACM0 -b 115200
```

## References
* [Embassy Book](https://embassy.dev/book)