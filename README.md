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

### Configure Private Registry

Add the following to `~/.cargo/config.toml` to proxy crates.io through the self-hosted registry:

```toml
[registry]
global-credential-providers = ["cargo:token"]

[registries]
utadr-cratesio = { index = "sparse+https://crates.yifanl.com/api/v1/cratesio/" }

[source.utadr-cratesio]
registry = "sparse+https://crates.yifanl.com/api/v1/cratesio/"

[source.crates-io]
replace-with = "utadr-cratesio"
```

Then authenticate with both registries:

```bash
cargo login --registry utadr-cratesio
cargo login --registry utadr
```

## Build

One firmware build targets one **vehicle** — a `vehicles/*.yaml` whose
`build:` section selects the cargo features and whose `tuning:` section is
baked in as the parameter defaults.

```bash
just vehicles                            # list vehicles + their build facts
just build                               # build the default (VEHICLE= in .env)
just build sakura_bench_hunter_indoor    # …or name one explicitly
just flash sakura_bench_hunter_indoor    # build + DFU the same one

just print-features [<vehicle>]  # resolved feature list + per-knob provenance
just check-all                   # compile-check every supported configuration

# Bare cargo also works (bakes the fallback vehicle with a warning):
cargo check -p cybflight
```

`build`, `flash`, `print-features` and `param-sync` all take the same
optional positional vehicle, defaulting to `VEHICLE=` in `.env`. Naming a
vehicle that does not exist fails immediately with the valid names — it
never composes a feature list for a vehicle that isn't there.

`just print-features <vehicle>` is the dry run for a build. (`just --dry-run
build` will *not* show you the features: `just` does not evaluate backticks
in dry-run mode.)

> Because `build` takes an argument, **`just build test` means "build the
> vehicle named `test`"**, not "build, then test". To chain, either reverse
> it (`just test build`), name the vehicle (`just build sakura_bench test`),
> or use `just build && just test`.

Env vars (`BOARD=`, `OUTER_LOOP=`, `POS_SOURCE=`, `ROLE=`, …) remain
per-invocation dev overrides on top of the YAML; `build.rs` cross-checks the
feature list against the vehicle and hard-fails a mismatched pairing.

### Cargo Features

Features are normally selected by the vehicle YAML (`build:` section);
the table lists the main axes. Cargo's *default* set targets
FOXEERH743 + mocap (the bare-`cargo check` / rust-analyzer path).

| Feature | Default | `build:` knob | Description |
|---|---|---|---|
| `board_sakurah743` | no | `board` | Select the SAKURAH743 BSP |
| `board_foxeerh743` | yes | `board` | Select the FOXEERH743 BSP |
| `board_micoair743v2` | no | `board` | Select the MICOAIR743V2 BSP |
| `est_pos_mocap` | yes | `pos_source` | Position from motion capture (indoor; VICON_POSE over the ESP bridge) |
| `est_pos_gps` | no | `pos_source` | Position from GNSS (outdoor; NAV-PVT). Mutually exclusive with the above |
| `outer_mpc` | yes | `outer_loop` | MPC outer loop (`cascade` → `outer_geometric`, `rate` → `outer_rate`) |
| `rx_crsf` | yes | `rc_protocol` | CRSF (ELRS/TBS) RC protocol |
| `rx_ghst` | no | `rc_protocol` | GHST (ImmersionRC) RC protocol |
| `imu_1khz` | no | `imu_rate` | ICM low-noise 1 kHz mode (default is 8 kHz) |
| `indi_off` | no | `indi` | Disable the INDI inner loop — degrades to a proportional rate controller |
| `role_leader` / `role_chaser` | no | `role` | Multi-drone role (see below) |
| `gps_unicore` | no | `gps_model` | UM982 dual-antenna driver (ublox is the driver default — no feature) |
| `defmt_uart` | yes | — | UART defmt logging on USART3 (SAKURAH743: PD8, FOXEERH743: PB10). Env-only: `DEFMT_UART=true` |

`pos_source` is the axis that most changes behaviour: it selects which
estimation task is spawned, which failsafe guard is constructed, and whether
the vehicle can fly indoors at all. A board flashed with the outdoor build
never subscribes to the pose channel.

### Flash via USB DFU

```bash
dfu-util -l
# Expected: "Found DFU: [0483:df11] ..."

dfu-util -a 0 -s 0x08000000:leave -D target/thumbv7em-none-eabihf/release/cybflight.bin
```

### Multi-drone: leader / chaser (gimbal)

Two variants, each a **vehicle YAML** carrying `build.role` (plus a matching
`gps_model`). The **chaser** carries the Z-1Mini gimbal on **USART2 (PD5 TX /
PD6 RX, 115200 baud)** + a UM982 and aims the camera at the **leader**; the
leader is the unchanged airframe that broadcasts its pose. Both run on
**sakurah743**.

| Role | `build.role` | `build.gps_model` | Features added |
|---|---|---|---|
| **Chaser** | `chaser` | `unicore` (UM982) | `role_chaser`, `gps_unicore` |
| **Leader** | `leader` | `ublox` | `role_leader` (ublox is the driver default — no feature) |

Put the FC in **DFU mode**, then flash the vehicle you want:

```bash
just flash sakura_chaser    # chaser: gimbal + UM982, aims at leader
just flash sakura_leader    # leader:  broadcasts its pose
```

`.env.chaser` / `.env.leader` remain as templates if you would rather pin a
default (`cp .env.chaser .env && just flash`), but naming the vehicle on the
command line is usually less error-prone — nothing persists between builds.

Notes:
- `just flash` ends with `dfu-util … :leave`; a final `Error during download
  get_status` (exit 74) is **benign** — the board reboots out of DFU and drops
  USB before the status read. The image is written + verified ("File downloaded
  successfully").
- No `ROLE` set → the standard firmware (no gimbal), unchanged.
- The chaser receives the leader's pose over a **direct ESP→ESP link** — see the
  cybesp-bridge README for the matching ESP flash (the chaser ESP must be at
  `192.168.50.6`).

### Vehicle identity (indoor / mocap)

A vehicle YAML declares `airframe.name` — the **physical machine**, not the
file. One drone has one name, so the same airframe flown indoors and outdoors
is two vehicle files sharing a name, and their `airframe:` blocks (mass,
inertia, motor geometry) must match. A test enforces that.

That name is the join key across three places that otherwise agree only by
convention:

| where | what it identifies |
|---|---|
| `vehicles/*.yaml` `airframe.name` | the airframe this firmware is tuned for |
| the mocap rigid body | whose poses get streamed |
| cybgcs `fleet.toml` `airframe` | which drone is at which ESP32 IP |

Poses are routed to a drone **by IP**, and nothing used to check that the
board at that address was running the firmware for the airframe whose poses
were being sent to it. A mis-flash silently pairs one machine's mass and gains
with another machine's pose stream. The firmware now announces its baked
identity (~0.2 Hz), and cybgcs **stops sending poses** to a drone whose
reported airframe disagrees with the roster — no poses means
`ESTIMATOR_READY` never goes true, so it cannot arm. Name the Vicon object
after the airframe and `fleet.toml`'s `vicon_subject` can be omitted.

### Connect to USB Shell
```bash
minicom -D /dev/ttyACM0 -b 115200
minicom -D /dev/ttyACM0 -b 230400 # GPS
```

Bench tuning happens here. `param list` prints the whole config tree grouped
by subsystem, led by a `[build]` block showing the compile-time selections
(board, `pos_source`, `outer_loop`, `imu_rate`, the baked vehicle and
`airframe.name`) — read-only, but the first thing to check when a board is
behaving like a vehicle it isn't:

```
param list                       # everything, grouped
param eskf-mocap_guard list      # one subtree ('.', '-', '_' interchangeable)
param list mpc                   # same thing, verb-first
param set <name> <value>         # applies on the disarmed hot-reload
param save                       # persist to the KV flash log
```

Tab completes commands, stream topics, and param / mission names
(`param set eskf_mo<Tab>`); an ambiguous Tab lists the candidates.

A group name matches its whole subtree, params needing a reboot are marked
`(reboot)`, and an unknown group prints the available ones. Then
`just param-sync [<vehicle>]` merges `param diff --yaml` back into the vehicle
file for review and commit — the firmware never writes YAML.

### View `defmt` Serial Log
```bash
cargo install defmt-print

stty -F /dev/ttyUSB0 921600 raw -echo
# stty -F /dev/ttyACM0 921600 raw -echo
defmt-print -e target/thumbv7em-none-eabihf/release/cybflight < /dev/ttyUSB0
# defmt-print -e target/thumbv7em-none-eabihf/release/cybflight < /dev/ttyACM0
```

## Further reading

| Doc | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Channels, BSPs, `board_init/`, the configuration plane. **Read before changing anything.** |
| [docs/HACKING.md](docs/HACKING.md) | Day-to-day workflow: build for a vehicle, tune at the bench, add a parameter, the sim regression snapshot |
| [docs/parameters.md](docs/parameters.md) | Generated reference for every parameter (`just params-doc`) |
| [docs/safety_protocol.md](docs/safety_protocol.md) | Arming gates, failsafe state machine, failure-propagation walkthroughs |
| [docs/control_overview.md](docs/control_overview.md) | Outer loop, INDI inner loop, trajectory planning |

## References
* [Embassy Book](https://embassy.dev/book)
