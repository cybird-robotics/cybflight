set dotenv-load := true

BIN := "target/thumbv7em-none-eabihf/release/cybflight.bin"

HOST := `rustc -vV | sed -n 's/^host: //p'`

# ── Feature composition ────────────────────────────────────────────────
# The vehicle YAML's `build:` section (vehicles/$VEHICLE.yaml) is the
# source of truth for per-vehicle compile-time selections: board,
# rc_protocol, outer_loop (mpc|mpc_full|cascade|rate), pos_source
# (gps|mocap), gps_model (ublox|unicore), gps_dual_antenna (yes|no),
# imu_rate (8khz|1khz), indi (yes|no) and
# plan_online (yes|no). Env vars of the same names remain as dev
# overrides (the script warns on divergence). `just print-features`
# lists every knob with its resolved value and provenance.
#
# Dev/bench-only knobs stay env-only (they are not vehicle facts):
#   DEFMT_UART=true   link the UART defmt logger
#   ESTIMATOR=eskf    attitude/position estimator (only est_eskf exists)
#
# Historical env vars that no longer exist: SAMPLER (`sampler_kind`
# runtime param), GPS_FUSE_VEL / GPS_FUSE_HEADING (`gps_fuse_*` params),
#
# Feature resolution runs per-recipe rather than as a top-level backtick,
# so the vehicle can come from a positional argument. That also stops
# `just test` / `just --list` / tab-completion from running the feature helper.
# The script HARD FAILS when vehicles/<vehicle>.yaml does not exist —
# `just vehicles` lists the legal names. build.rs stays the feature↔YAML
# consistency guard.
DEFAULT_VEHICLE := env_var_or_default("VEHICLE", "sakura_vicon")

# NOTE: this recipe takes an argument, so `just build test` means "build the
# vehicle named test", not "build, then test" — that fails immediately with
# the list of available names rather than building anything.
# Build firmware for a vehicle (default: $VEHICLE from .env; see `just vehicles`).
build $VEHICLE=DEFAULT_VEHICLE:
    cargo run --release --no-default-features --features {{ `bash tools/vehicle_features.sh` }},postmortem

# Convenience: build the MPC variant without editing the vehicle YAML.
build-mpc $VEHICLE=DEFAULT_VEHICLE:
    OUTER_LOOP=mpc just build {{VEHICLE}}

# Convenience: build the cascade variant explicitly.
build-cascade $VEHICLE=DEFAULT_VEHICLE:
    OUTER_LOOP=cascade just build {{VEHICLE}}

# `*` marks the current default; any name listed is a valid `just build` arg.
# List every vehicle YAML with the build facts that distinguish them.
vehicles:
    @bash tools/vehicle_features.sh --list

# Takes the same positional vehicle as `just build`, so it is the dry run
# for a build you are about to do. (`just --dry-run build` does NOT evaluate
# backticks — it shows the literal command, not the resolved features.)
# Print the resolved feature list + per-knob provenance for a vehicle.
print-features $VEHICLE=DEFAULT_VEHICLE:
    @echo "VEHICLE={{VEHICLE}}"
    @echo "FEATURES={{ `bash tools/vehicle_features.sh --explain` }}"

# Static RAM / flash report for a vehicle's firmware (builds it first):
# section sizes, the 20 largest .bss/.data symbols, and the headroom
# between the end of the statics and the initial stack pointer. Everything
# lives in the one 512 KiB AXI SRAM region, so headroom is the whole budget
# for stack growth. The 8 kHz vehicles once boot-looped at ~145 KB of it
# (crates/cybflight/src/blackbox/sdmmc_block.rs) — check before flying a
# build that adds statics.
size $VEHICLE=DEFAULT_VEHICLE: (build VEHICLE)
    @python3 tools/size_report.py target/thumbv7em-none-eabihf/release/cybflight \
        --llvm-bin "$(rustc --print target-libdir)/../bin"

# Run the cybflight-core host tests (convergence + benchmark suite).
# NOT the whole workspace: `test-drivers` and the `sim-*` recipes are
# separate, and `sim-check` is the regression gate for controller
# behaviour. `test-all` runs the three together.
test:
    cargo test -p cybflight-core --target {{HOST}} --release

# Core, build-tool and driver tests plus the sim regression snapshot —
# the set worth passing before a commit that touches control or
# estimation code.
test-all: test test-build-tools test-drivers sim-check

# Validate vehicle parsing and build feature selection.
test-build-tools:
    cargo test -p vehicle-yaml --target {{HOST}} --release

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

# TinyMPC (ADMM, hover-linearised, paper setup 500 Hz / N = 15) vs the
# SQP-NMPC on every indoor mission (circle / figure8 / slalom / splits ×
# speed profiles). Writes target/sim-out/figure8_tinympc/ and renders the
# per-family figures + docs table with analysis/plot_figure8_compare.py.
sim-figure8-compare:
    cargo test -p cybflight-sim --target {{HOST}} --profile release-host \
        --test figure8_tinympc_compare -- --nocapture
    python3 analysis/plot_figure8_compare.py

# Neural gate-racing policy: validates the ported MLP against PyTorch, the
# ENU observation adapter against the trainer's own observations, and flies
# the 8-gate track on both the RL reference plant and cybflight's plant.
# Fixtures are regenerated with tools/export_rl_policy.py — see its header.
sim-nn:
    cargo test -p cybflight-sim --target {{HOST}} --profile release-host \
        --test nn_gate_race -- --nocapture --test-threads=1

# ACMPC gate racing: validates the ported cost network against PyTorch, the
# box-DDP solve against the trainer's own, and flies the 8-gate track on
# both the reference plant's identified parameters and cybflight's own
# vehicle. Fixtures are regenerated with tools/export_acmpc_policy.py.
sim-acmpc:
    cargo test -p cybflight-sim --target {{HOST}} --profile release-host \
        --test acmpc_gate_race -- --nocapture --test-threads=1

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
    cargo run -p cybflight-sim --target {{HOST}} --profile release-host --bin sim-run \
        {{ if VIZ == "1" { "--features viz" } else { "" } }} -- \
        --scenario {{SCENARIO}} --controller {{CONTROLLER}} --noise {{NOISE}} --gps {{GPS}} \
        {{ if VIZ == "1" { "--viz" } else { "" } }}

# Compile-check every supported build configuration without flashing.
# Useful as a CI smoke test for the cfg-gates around `outer_mpc`.
# The geometric cascade and rate-mode variants are maintained alternatives
# to MPC+INDI — keep them here so they cannot silently rot.
check-all:
    #!/usr/bin/env bash
    set -euo pipefail
    # The matrix deliberately builds board/source combinations that do NOT
    # match the .env vehicle's build: section — disable the feature↔YAML
    # guard for these compile-only smoke checks.
    export VEHICLE_GUARD=off
    cargo check -p cybflight
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_geometric
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_rate
    # Rate-only, no position source at all — legal for outer_rate; the
    # Mahony task is then the sole attitude estimator (blackbox /attitude).
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,rx_crsf,defmt_uart,outer_rate
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_foxeerh743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_gps,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_foxeerh743,est_pos_gps,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_micoair743v2,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc
    cargo check -p cybflight --no-default-features \
        --features board_micoair743v2,est_pos_gps,rx_crsf,defmt_uart,outer_mpc
    # 1 kHz low-noise IMU mode (`imu_rate: 1khz`) on the ICM boards.
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_gps,rx_crsf,defmt_uart,outer_mpc,imu_1khz
    cargo check -p cybflight --no-default-features \
        --features board_foxeerh743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc,imu_1khz
    # INDI off (`indi: no`) — inner loop degrades to a rate controller.
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc,indi_off
    # Full-model outer loop (`outer_loop: mpc_full`) — NMPC → (T_d, τ_d) →
    # INDI α inner loop; plus its static-inversion ablation (indi: no).
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc_full
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc_full,indi_off
    # Dual-antenna heading. The `gps_unicore` fusion path had NO coverage
    # here at all, so `update_baseline` and the init yaw-seed were only
    # ever compiled by a real vehicle build.
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_gps,rx_crsf,defmt_uart,outer_mpc,gps_unicore,gps_dual_antenna
    # Online BFGS planning (`plan_online: yes`). Costs ~110 KiB of solver
    # .bss + task-future state, so no shipped vehicle enables it — which
    # is exactly why it needs a compile check of its own.
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc,plan_online
    # GHST receivers. A whole parser and telemetry path that no vehicle
    # YAML currently selects.
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_ghst,defmt_uart,outer_mpc
    # Raw-sensor downlink branches in the ESP bridge.
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_mocap,rx_crsf,defmt_uart,outer_mpc,dev_telem
    # ANT2 declared without a heading-capable driver. `BuildYaml::validate`
    # makes this illegal for a real vehicle, but a hand-composed set can
    # still reach it — and `param list build` has a branch that reports it.
    # Compiled so that diagnostic cannot rot.
    cargo check -p cybflight --no-default-features \
        --features board_sakurah743,est_pos_gps,rx_crsf,defmt_uart,outer_mpc,gps_dual_antenna

# The dependency is passed the argument, so the binary flashed is the one
# just built for THIS vehicle. (BIN is a single fixed path shared by every
# vehicle; running dfu-util by hand outside just would flash whatever was
# built last.)
# Build then DFU-flash a vehicle. Same positional argument as `just build`.
# Build + DFU-flash a vehicle, retrying the download and reporting the boot.
flash $VEHICLE=DEFAULT_VEHICLE: (build VEHICLE)
    #!/usr/bin/env bash
    set -euo pipefail
    # The download is retried because a DFU transfer that dies partway
    # ("Error during special command SET_ADDRESS get_status" is the one
    # seen in practice) leaves a half-written image, and an STM32 that
    # faults before USB init does not enumerate in ANY mode — no CDC
    # shell, no `0483:df11`. That is indistinguishable, from the outside,
    # from "the firmware I just built bricks the board", so this recipe
    # says which of the two it was instead of exiting quietly on a
    # transfer that never finished. Recovery from a partial write always
    # needs BOOT0 held at power-up, so the failure path says so.
    if ! dfu-util -l 2>&1 | grep -q "0483:df11"; then
        echo "No DFU device found. Boot the board into DFU mode first."
        exit 1
    fi
    for attempt in 1 2 3; do
        if dfu-util -d "0483:df11" -a 0 -s 0x08000000:leave -D {{BIN}}; then
            break
        fi
        echo "--- DFU download failed (attempt ${attempt}/3) ---"
        if [ "$attempt" = 3 ]; then
            echo "FLASH FAILED: the image on flash is now PARTIAL — the board" >&2
            echo "will not boot and may not enumerate at all. Hold BOOT while" >&2
            echo "replugging USB to re-enter DFU, then retry." >&2
            exit 1
        fi
        # A failed transfer can leave the device wedged rather than gone.
        sleep 2
        if ! dfu-util -l 2>&1 | grep -q "0483:df11"; then
            echo "DFU device left the bus. Hold BOOT while replugging USB," >&2
            echo "then retry — the image on flash is partial." >&2
            exit 1
        fi
    done
    # `:leave` jumps to the application; the shell should enumerate shortly.
    # Not finding it is not proof of a bad image (a board with no USB shell
    # build, or a slow host, both look like this), so this reports rather
    # than fails.
    for _ in $(seq 1 20); do
        sleep 0.5
        if ls /dev/serial/by-id/ 2>/dev/null | grep -q cybflight; then
            echo "flashed and booted: $(ls /dev/serial/by-id/ | grep cybflight)"
            exit 0
        fi
    done
    echo "NOTE: download completed but no cybflight shell appeared within 10 s."
    echo "      Power-cycle and check; 'just flash-verify' compares the image"
    echo "      on flash against {{BIN}} if you suspect a bad write."

# Flash, then read the image back off the chip and byte-compare it.
flash-verify $VEHICLE=DEFAULT_VEHICLE: (build VEHICLE)
    #!/usr/bin/env bash
    set -euo pipefail
    # Leaves the board in DFU on success (nothing here jumps to the app),
    # so follow with `just flash` or a power-cycle. Use when a flash is
    # suspect: this is the only check that separates "the image on the
    # chip is wrong" from "the firmware is misbehaving".
    if ! dfu-util -l 2>&1 | grep -q "0483:df11"; then
        echo "No DFU device found. Boot the board into DFU mode first."
        exit 1
    fi
    size=$(stat -c %s {{BIN}})
    dfu-util -d "0483:df11" -a 0 -s 0x08000000 -D {{BIN}}
    # mktemp -u: dfu-util refuses to upload onto an existing file.
    readback=$(mktemp -u /tmp/cybflight_readback_XXXXXX.bin)
    dfu-util -d "0483:df11" -a 0 -s "0x08000000:${size}" -U "$readback"
    if cmp -s {{BIN}} "$readback"; then
        rm -f "$readback"
        echo "VERIFY OK: ${size} bytes on flash match {{BIN}}"
        echo "  (board is still in DFU — 'just flash' or a power-cycle boots it)"
    else
        rm -f "$readback"
        echo "VERIFY MISMATCH: flash does not match {{BIN}} — do not fly this." >&2
        exit 1
    fi

# Convenience flash targets that pre-set OUTER_LOOP.
flash-mpc $VEHICLE=DEFAULT_VEHICLE:
    OUTER_LOOP=mpc just flash {{VEHICLE}}

flash-cascade $VEHICLE=DEFAULT_VEHICLE:
    OUTER_LOOP=cascade just flash {{VEHICLE}}

# Merges into vehicles/<vehicle>.yaml for review + commit; the firmware never
# writes YAML. See tools/param_sync.py.
# Pull `param diff --yaml` off the FC over USB CDC into a vehicle's YAML.
param-sync $VEHICLE=DEFAULT_VEHICLE:
    python3 tools/param_sync.py --vehicle {{VEHICLE}}

# Pull blackbox MCAP logs off the FC over USB CDC into ./logs/.
# No args = every *.mcap not already local; or name files, or pass
# flags (--list, --delete, --force ...). See tools/blackbox_pull.py.
blackbox-pull *ARGS:
    python3 tools/blackbox_pull.py {{ARGS}}

# Regenerate docs/parameters.md from the parameter registry. Commit the
# result whenever the schema changes.
params-doc:
    cargo run -p cybflight-sim --target {{HOST}} --bin param_doc > docs/parameters.md
    @echo "wrote docs/parameters.md"
