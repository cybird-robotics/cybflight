set dotenv-load := true

BIN := "target/thumbv7em-none-eabihf/release/cybflight.bin"

HOST := `rustc -vV | sed -n 's/^host: //p'`

BOARD := env_var_or_default("BOARD", "sakurah743")

RC_PROTOCOL := env_var_or_default("RC_PROTOCOL", "crsf")

ESTIMATOR := env_var_or_default("ESTIMATOR", "eskf")

# Outer-loop controller selection: "cascade" (default, verified in flight) or
# "mpc" (the SQP/MPC running in `outer_loop::control_loop_task` at 100 Hz).
# Only meaningful with ESTIMATOR=eskf; the cascade controller for est_mahony
# lives in inner_loop.rs and is selected automatically.
OUTER_LOOP := env_var_or_default("OUTER_LOOP", "cascade")

# Compose the feature list for `cargo build`. The `outer_mpc` feature is
# appended only when OUTER_LOOP=mpc; otherwise the cascade is used (the
# `outer_mpc` feature is gated on est_eskf via a compile_error guard, so
# misuse with est_mahony fails fast at build time).
FEATURES := if OUTER_LOOP == "mpc" {
    "board_" + BOARD + ",rx_" + RC_PROTOCOL + ",est_" + ESTIMATOR + ",outer_mpc"
} else {
    "board_" + BOARD + ",rx_" + RC_PROTOCOL + ",est_" + ESTIMATOR
}

build:
    cargo run --release --no-default-features --features {{FEATURES}}

# Convenience: build the MPC variant without having to set OUTER_LOOP=mpc.
build-mpc:
    OUTER_LOOP=mpc just build

# Convenience: build the cascade variant explicitly.
build-cascade:
    OUTER_LOOP=cascade just build

# Print the resolved feature list (useful for debugging the build matrix).
print-features:
    @echo "BOARD={{BOARD}} RC_PROTOCOL={{RC_PROTOCOL}} ESTIMATOR={{ESTIMATOR}} OUTER_LOOP={{OUTER_LOOP}}"
    @echo "FEATURES={{FEATURES}}"

# Run all host-side tests (cybflight-core convergence + benchmark suite).
test:
    cargo test -p cybflight-core --target {{HOST}}

# Run only the MPC convergence + benchmark tests (faster iteration).
test-mpc:
    cargo test -p cybflight-core --target {{HOST}} --release \
        --test control_convergence -- --nocapture

test-drivers:
    cargo test -p cybflight-drivers --target {{HOST}}

# Compile-check every supported build configuration without flashing.
# Useful as a CI smoke test for the cfg-gates around `outer_mpc`.
check-all:
    cargo check -p cybflight
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_eskf,rx_crsf,defmt_uart
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_eskf,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_foxeerh743,est_eskf,rx_crsf,defmt_uart,outer_mpc

flash: build
    #!/usr/bin/env bash
    set -euo pipefail
    if ! dfu-util -l 2>&1 | grep -q "0483:df11"; then
        echo "No DFU device found. Boot the board into DFU mode first."
        exit 1
    fi
    dfu-util -d "0483:df11" -a 0 -s 0x08000000:leave -D {{BIN}}

# Convenience flash targets that pre-set OUTER_LOOP.
flash-mpc:
    OUTER_LOOP=mpc just flash

flash-cascade:
    OUTER_LOOP=cascade just flash
