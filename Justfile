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
OUTER_LOOP := env_var_or_default("OUTER_LOOP", "mpc")

# ESKF position source: "mocap" (default, ESP bridge + VICON_POSE) or
# "gps" (u-blox M10 NAV-PVT → LLH→ENU). Mutually exclusive via compile_error.
POS_SOURCE := env_var_or_default("POS_SOURCE", "mocap")

# Reference sampler selection (only meaningful with OUTER_LOOP=mpc):
#   "time"     — TimeSampler (default; firmware-verified path).
#   "position" — PositionSampler (closest-point search). Compiles in
#                cybflight's `position_sampler` feature.
# Misuse with OUTER_LOOP=cascade is silently ignored — the cascade path
# does not consume the sampler abstraction.
SAMPLER := env_var_or_default("SAMPLER", "position")

# Compose the feature list for `cargo build`. The `outer_mpc` feature is
# appended only when OUTER_LOOP=mpc; otherwise the cascade is used (the
# `outer_mpc` feature is gated on est_eskf via a compile_error guard, so
# misuse with est_mahony fails fast at build time). The `position_sampler`
# feature is appended only when SAMPLER=position AND OUTER_LOOP=mpc — the
# feature is a no-op outside the MPC outer loop.
FEATURES := if OUTER_LOOP == "mpc" {
    if SAMPLER == "position" {
        "board_" + BOARD + ",rx_" + RC_PROTOCOL + ",est_" + ESTIMATOR + ",est_pos_" + POS_SOURCE + ",outer_mpc,position_sampler"
    } else {
        "board_" + BOARD + ",rx_" + RC_PROTOCOL + ",est_" + ESTIMATOR + ",est_pos_" + POS_SOURCE + ",outer_mpc"
    }
} else {
    "board_" + BOARD + ",rx_" + RC_PROTOCOL + ",est_" + ESTIMATOR + ",est_pos_" + POS_SOURCE
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
    @echo "BOARD={{BOARD}} RC_PROTOCOL={{RC_PROTOCOL}} ESTIMATOR={{ESTIMATOR}} OUTER_LOOP={{OUTER_LOOP}} POS_SOURCE={{POS_SOURCE}} SAMPLER={{SAMPLER}}"
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
# plus MpcDirect and cascade as diagnostic baselines, and the noisy variant
# through MPC+INDI. Prints a comparison table and leaves JSON reports in
# target/{{HOST}}/tmp/<scenario>_<controller>/report.json.
sim-compare:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo test -p cybflight-sim --target {{HOST}} --profile release-host \
        --test autotest_mission --test autotest_noisy --test autotest_gps \
        -- --nocapture --test-threads=1 2>&1 | \
        tee /tmp/cybflight-sim-compare.log
    TMP_DIR="target/{{HOST}}/tmp"
    printf '\n%-22s %-11s %10s %10s %10s %8s %7s\n' scenario controller rms_err_m peak_err_m term_err_m tilt_deg sat_pct
    printf '%-22s %-11s %10s %10s %10s %8s %7s\n' ---------------------- ----------- ---------- ---------- ---------- -------- -------
    # Clean rows: all scenarios × all controllers (rows missing on disk skipped).
    for scenario in hover_level hover_tilt30 p2p_x3 mission_square; do
        for controller in cascade mpc_direct mpc_indi; do
            report="${TMP_DIR}/${scenario}_${controller}/report.json"
            [ -f "$report" ] || continue
            SCEN="$scenario" CTRL="$controller" REPORT="$report" python3 tools/sim_summary_row.py
        done
    done
    # Sensors-in-the-loop rows: only MpcIndi is firmware-representative,
    # so only that column is emitted.
    for scenario in mission_square_noisy mission_square_gps; do
        for controller in mpc_indi; do
            report="${TMP_DIR}/${scenario}_${controller}/report.json"
            [ -f "$report" ] || continue
            SCEN="$scenario" CTRL="$controller" REPORT="$report" python3 tools/sim_summary_row.py
        done
    done

# Verify the sim comparison table matches the committed snapshot at
# crates/cybflight_sim/tests/regression_snapshot.json. Tight-tolerance
# regression gate for algebraic changes to controllers / plant / planner.
# See docs/HACKING.md for the review workflow.
sim-check:
    cargo test -p cybflight-sim --target {{HOST}} --profile release-host \
        --test regression_snapshot -- --nocapture

# Regenerate the sim regression snapshot from the current code. Review with
# `git diff` before committing — the snapshot IS the change log for how the
# controllers behave, and any drift should be reviewed by a human.
sim-snapshot:
    UPDATE_SNAPSHOTS=1 cargo test -p cybflight-sim --target {{HOST}} \
        --profile release-host --test regression_snapshot -- --nocapture

# Run a single sim scenario end-to-end through the CLI binary. Defaults to
# mission_square / mpc_indi. Override with SCENARIO / CONTROLLER; set VIZ=1
# to stream to a rerun viewer.
SCENARIO := env_var_or_default("SCENARIO", "mission_square")
CONTROLLER := env_var_or_default("CONTROLLER", "mpc-indi")
NOISE := env_var_or_default("NOISE", "none")
GPS := env_var_or_default("GPS", "none")
VIZ := env_var_or_default("VIZ", "0")

sim-run:
    cargo run -p cybflight-sim --target {{HOST}} --profile release-host --bin sim-run -- \
        --scenario {{SCENARIO}} --controller {{CONTROLLER}} --noise {{NOISE}} --gps {{GPS}} \
        {{ if VIZ == "1" { "--viz" } else { "" } }}

# Compile-check every supported build configuration without flashing.
# Useful as a CI smoke test for the cfg-gates around `outer_mpc`.
check-all:
    cargo check -p cybflight
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_foxeerh743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_gps,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_foxeerh743,est_pos_gps,rx_crsf,defmt_uart,outer_mpc

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
