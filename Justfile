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
    cargo test -p cybflight-core --target {{HOST}} --release

# Run only the MPC convergence + benchmark tests (faster iteration).
test-mpc:
    cargo test -p cybflight-core --target {{HOST}} --release \
        --test control_convergence -- --nocapture

test-drivers:
    cargo test -p cybflight-drivers --target {{HOST}}

# Run the sim autotest: hover + p2p + mission through MPC+INDI (authoritative)
# plus MpcDirect and cascade as diagnostic baselines. Prints a comparison
# table and leaves JSON reports in
# target/{{HOST}}/tmp/<scenario>_<controller>/report.json.
sim-compare:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo test -p cybflight-sim --target {{HOST}} --profile release-host \
        --test autotest_mission -- --nocapture --test-threads=1 2>&1 | \
        tee /tmp/cybflight-sim-compare.log
    TMP_DIR="target/{{HOST}}/tmp"
    printf '\n%-16s %-11s %10s %10s %10s %8s %7s\n' scenario controller rms_err_m peak_err_m term_err_m tilt_deg sat_pct
    printf '%-16s %-11s %10s %10s %10s %8s %7s\n' ---------------- ----------- ---------- ---------- ---------- -------- -------
    for scenario in hover_level hover_tilt30 p2p_x3 mission_square; do
        for controller in cascade mpc_direct mpc_indi; do
            report="${TMP_DIR}/${scenario}_${controller}/report.json"
            [ -f "$report" ] || continue
            SCEN="$scenario" CTRL="$controller" REPORT="$report" python3 tools/sim_summary_row.py
        done
    done

# Run a single sim scenario end-to-end through the CLI binary. Defaults to
# mission_square / mpc_indi. Override with SCENARIO / CONTROLLER; set VIZ=1
# to stream to a rerun viewer.
SCENARIO := env_var_or_default("SCENARIO", "mission_square")
CONTROLLER := env_var_or_default("CONTROLLER", "mpc-indi")
VIZ := env_var_or_default("VIZ", "0")

sim-run:
    cargo run -p cybflight-sim --target {{HOST}} --profile release-host --bin sim-run -- \
        --scenario {{SCENARIO}} --controller {{CONTROLLER}} \
        {{ if VIZ == "1" { "--viz" } else { "" } }}

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
