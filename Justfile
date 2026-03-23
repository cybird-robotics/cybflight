set dotenv-load := true

BIN := "target/thumbv7em-none-eabihf/release/cybflight.bin"

HOST := `rustc -vV | sed -n 's/^host: //p'`

BOARD := env_var_or_default("BOARD", "sakurah743")

RC_PROTOCOL := env_var_or_default("RC_PROTOCOL", "crsf")

ESTIMATOR := env_var_or_default("ESTIMATOR", "eskf")

build:
    cargo run --release --no-default-features --features board_{{BOARD}},rx_{{RC_PROTOCOL}},est_{{ESTIMATOR}}

test:
    cargo test -p cybflight-core --target {{HOST}}

test-drivers:
    cargo test -p cybflight-drivers --target {{HOST}}

flash: build
    #!/usr/bin/env bash
    set -euo pipefail
    if ! dfu-util -l 2>&1 | grep -q "0483:df11"; then
        echo "No DFU device found. Boot the board into DFU mode first."
        exit 1
    fi
    dfu-util -d "0483:df11" -a 0 -s 0x08000000:leave -D {{BIN}}
