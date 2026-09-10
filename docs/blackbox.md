# SD-card blackbox: design + staged plan

## What this document is

A handoff doc for the next agent picking up the cybflight blackbox
work. Reading order:

1. **Goal** — why we're rebuilding this.
2. **Architecture** — the layers, where they live, what's the seam between them.
3. **Staged plan** — what's done, what's next, why the staging matters.
4. **Current state** — exact line numbers + file map.
5. **Operational guide** — the full user instruction: enabling, recording, tiers, downloading, analysis, troubleshooting.
6. **Future work** — the parking lot from the architectural review.

The repo also has a [staffing-quality review](#post-mortem-of-the-original-wip)
of the *original* WIP blackbox at the bottom — read that to understand
why we deleted ~2800 lines and started over.

---

## Goal

A flight data recorder that produces **structurally-valid MCAP files
on a microSD card**, written through a FAT filesystem so users can
pop the card into a desktop OS and see the file directly. The
recorder triggers on arm-edge for production use, and has a manual
override for bench testing.

Hard constraints (from the [original review](#post-mortem-of-the-original-wip)):

- **Every file must pass `mcap doctor`** — strict tooling
  (`mcap cat`, Foxglove, Python `mcap` library) must accept it.
- **Architecture rule respect** — `BlockStore` trait (storage seam)
  in `cybflight-drivers`; the only `embassy_stm32::sdmmc` consumer
  is the SDMMC adapter; capability flag `bsp::HAS_BLACKBOX_STORAGE`
  gates the whole subsystem (no Cargo feature flag).
- **Bench-testable from the USB shell** — every stage of the rebuild
  is verifiable from a desk with `mcap doctor` / `analysis/read_mcap.py`,
  no flying required.

---

## Architecture

```
                ┌─────────────────────────────────────────────────────┐
shell ─Request─▶│                                                     │
                │            blackbox_task                            │
IS_ARMED edge ──┤  (in mod.rs — orchestrates session lifecycle,      │─▶ recorder
                │   debounces edges, dispatches LS_REQUEST)           │           │
                └─────────────────────────────────────────────────────┘           │
                                                                                  ▼
                                                  fat::write_file<P: FileBody>
                                                                │
                                                  embedded-fatfs / partitions
                                                                │
                                                  block_device_adapters::BufStream<_, 512>
                                                                │
                                                  block_device_driver::BlockDevice<512>
                                                                │
                                                  SdmmcBlockStore::open_session
                                                                │
                                                  embassy-stm32 SDMMC peripheral
```

### Layers (top → bottom)

| Layer | Crate / file | What it does |
|---|---|---|
| **Op** | `cybflight/blackbox/recorder.rs` | Owns subscribers, runs capture loop, produces MCAP records. Implements `FileBody`. |
| **MCAP framer** | `cybflight/blackbox/mcap.rs` | Wire-format writer for `Magic`/`Header`/`Schema`/`Channel`/`Message`/`DataEnd`/`Footer`. Hand-rolled, ~200 LOC. |
| **CBOR encoder** | `cybflight/blackbox/cbor.rs` | Hand-rolled CBOR writer over a stack buffer. Used by per-topic encoders. |
| **Topics** | `cybflight/blackbox/topics/*.rs` + `topics::ALL` | Per-topic schemas + CBOR encoders, and the registry that keeps channel ids unique. Single source of truth for each `msgs::*` type's wire format. |
| **FAT pipeline** | `cybflight/blackbox/fat.rs` | `write_file<P: FileBody>` mounts FAT, finds the FAT partition (handles MBR vs superfloppy), creates the file, hands the writer to `body.write(...)`, flushes + unmounts. |
| **Partition** | `embedded-partitions` (git-pinned) | Reads sector 0; classifies MBR vs superfloppy; returns a `StreamSlice` over the FAT partition. |
| **Filesystem** | `embedded-fatfs` (git-pinned) | FAT16/FAT32 read+write. |
| **Byte ↔ block** | `cybflight/blackbox/lazy_stream.rs` (`LazyBufStream`) | Drop-in for `block-device-adapters::BufStream` (same API, fast paths, error type) that does **not** read a block before overwriting it — `BufStream`'s read-modify-write cost one CMD17 (~1 ms) per 512 B of recorder output. Loads lazily only when old contents are actually needed (read outside the written prefix, a write leaving a hole, or a partial-block flush). |
| **Write combiner** | `cybflight/blackbox/sdmmc_block.rs` (`WriteCombiner`) | Coalesces sequential block writes into one CMD25 multi-block transfer (64 × 512 B static batch). Flushes on non-sequential write, full buffer, overlapping read, and explicitly after unmount. A failed batch degrades to per-block CMD24 retries, so fault semantics match the old path. |
| **Block trait** | `cybflight-drivers/blackbox_storage.rs` (`BlockStore`) + `block-device-driver::BlockDevice<512>` | The architectural seam. `BlockStore` is no-`embassy-stm32`; future SPI-NOR / FRAM backends slot in here. |
| **SDMMC adapter** | `cybflight/blackbox/sdmmc_block.rs` | The **only** code in the firmware that imports `embassy_stm32::sdmmc`. Implements `BlockDevice<512>` via embassy-stm32's existing impl on `StorageDevice`. |
| **Driver** | `embassy-stm32::sdmmc` | The HAL's async SDMMC driver. |

### The two architectural rules

1. **`BlockStore` trait in `cybflight-drivers`** — no `embassy-stm32`
   dependency. Adding a new storage backend (SPI NOR, FRAM, MRAM)
   means writing one new adapter that implements `BlockDevice<512>`;
   nothing in `blackbox/` changes.

2. **`bsp::HAS_BLACKBOX_STORAGE` capability flag** in each BSP. The
   entire pipeline compiles out at this `const bool`. There is no
   Cargo feature; profile selection is by board.

These rules existed in the [WIP code's review](#post-mortem-of-the-original-wip)
as W4 and W5 — both were violated. Don't re-introduce them.

### Recorder triggers — two paths, one engine

Both end at `recorder::run_session`:

1. **Production** (real flight): RC switch / failsafe writes
   `motors::ARM_STATE` Signal → DShot consumes it and toggles the
   `motors::IS_ARMED` atomic → `blackbox_task` sees the rising edge.

2. **Bench test**: shell command `blackbox record on` sets
   `blackbox::RECORDER_HOLD` atomic directly (without touching
   `ARM_STATE`, so DShot output stays idle).

Within `blackbox_task`, the rising-edge predicate is symmetric:

```rust
fn should_record() -> bool {
    IS_ARMED.load(Acquire) || RECORDER_HOLD.load(Acquire)
}
```

But inside the capture loop, the keep-going predicate is **asymmetric**:

```rust
// recorder.rs::keep_recording
fn keep_recording(was_armed: &mut bool) -> bool {
    let now_armed = IS_ARMED.load(Acquire);
    let now_hold  = RECORDER_HOLD.load(Acquire);
    if now_armed { *was_armed = true; }            // latch
    if *was_armed { now_armed } else { now_hold }
}
```

This supports the "start logging early" pattern: `record on` → arm
→ fly → disarm closes the file (despite `RECORDER_HOLD` still set,
because `was_armed` latched and disarm now governs). After the
session, `blackbox_task` auto-clears `RECORDER_HOLD` so the next
cycle requires a deliberate fresh start.

---

## Staged plan

Each stage is **independently flash-and-verify testable** from the
USB shell. Don't skip stages — the staging is the testing strategy.

### Stage 0 — `touch sd` (smoke test) — DELETED

Was: `touch sd` shell command writes a canned text file via FAT.
Validated the SDMMC + FAT + partition stack end-to-end before any
MCAP code existed. Deleted once the recorder proved the stack at
higher fidelity.

### Stages 1–4 — Incremental MCAP buildup — DELETED

Each stage produced a structurally-valid MCAP file with progressively
more content:

- **Stage 1** — `blackbox skeleton`: empty file, just `Magic /
  Header / DataEnd / Footer / Magic`. Locked in the W1 fix
  (structurally valid from line 1).
- **Stage 2** — `blackbox synth`: one synthetic CBOR `BootInfo`
  message. Validated `Schema`/`Channel`/`Message` framing.
- **Stage 3** — `blackbox snap imu1`: one captured IMU sample.
  First touch of the existing pubsub plumbing.
- **Stage 4** — `blackbox capture imu1 <ms>`: time-bounded IMU
  capture loop. First throughput test (W3 from the review).

All four were deleted in the trim — they were development scaffolding,
verified at the time, no longer carried. The infrastructure they
exercised (BlockStore trait, FAT pipeline, MCAP framer, CBOR
encoder, topic encoders) is what stayed.

### Stage 5 — Multi-topic capture — DELETED post-merge

A bench-only `blackbox capture all <ms>` op existed in earlier
revisions for SD-throughput probing. It produced a separate
`cap_all_NNNN.mcap` lineage, had its own shell command, and
mirrored ~250 LOC of recorder.rs's drain machinery. With Stage 6
now consistent (RECORDER_HOLD-driven bench sessions live alongside
real arm sessions in the same file lineage), the bench-only op
became dead weight — removed in this branch's clean-up commit.
SD-throughput probes are now done by setting tier=Large via
`blackbox set large`, then `blackbox record on` for a few seconds.

### Stage 6 — Arm-triggered recorder — DONE (current tip)

`recorder::run_session` opens `flight_<NNNN>.mcap` on the rising
edge of `should_record()`, runs the multi-topic capture loop until
`keep_recording` returns false, emits ARM/DISARM/LOG_END events on
`/events`, closes cleanly. `KIND_DISARM` carries **why** in its `data`
field (0 = commanded, otherwise the live `FailsafeReason`), so a normal
landing and a watchdog disarm are distinguishable from the disarm record
alone — `msgs::ArmDisarm` carries only `armed`, so that cause exists
nowhere else in the file.

Three events name causes the rest of the log can only imply:

- `INNER_SILENT` (0x0C) — INDI stopped publishing on purpose; `data` is
  1 = voltage stale past the 2 s Table-mode gate, 2 = WLS NaN past
  `nan_limit`. Both trips let the controller watchdog disarm, which
  lands as `FAILSAFE(ControllerTimeout)` — a true symptom with the wrong
  cause attached. Latched per boot, so at most one appears.
- `POWER_STALE` / `POWER_OK` (0x0D/0x0E) — the *soft* 500 ms battery
  telemetry gap that precedes the trip above and also fires on episodes
  that recover. `data` is the episode index on entry (a jump in it means
  the recorder's poll aliased an episode away) and the episode length in
  ms on recovery.
- `RECORDER_OVERRUN` (0x0F) — the recorder is losing records; `data` is
  the session's cumulative drop count. Rate-limited to 1 Hz after the
  first edge, because it fires exactly when the write path is already
  backed up. Before this, loss was only visible by reconstructing
  sequence holes from the file afterwards. Which topics are logged is set by the
record set (`record_set.rs`); Small is `/events` + `/rc`, Mid adds
the IMU / attitude / motor / health stream, Large adds the estimator
and outer-loop chain, Sysid is the identification set (see §2).

Two ways to trigger:
- Real arm via RC / failsafe path (production)
- `blackbox record on/off` shell (bench)

### Stage 7 — Persistent params + status — DONE

The user-facing knobs, as they landed (names differ from the original
WIP plan where the tier system made an item redundant):

- `blackbox_device` / `blackbox_mode` — SUPERSEDED: board storage is
  the BSP capability flag, on/off is the `none` tier, and "always" is
  `blackbox record on`.
- `blackbox_rate_div` — DONE (param, v48): the recorder emits every
  Nth received message on the high-rate topics (`/imu1`, `/imu1_raw`,
  `/odometry`); 1 = full rate. Snapshotted at session start; the
  effective value is written into the session's `Metadata` record so
  readers never mis-assume nominal rates.
- `blackbox_mute_mask` — DONE (param, v48; the plan's
  `blackbox_fields_disabled`): bitmask by MCAP channel id (bit N =
  id N). A muted topic is absent from the session entirely — no
  Schema/Channel records, no subscription. `/events` is not maskable.
  Also recorded in `Metadata`. The mask must be at least as wide as the
  highest channel id — a topic above the mask's width is silently
  unmutable — so its `max` moves with the id space (it widened past 16
  bits for `/mpc_cost`, id 16).
- **record set tier** — DONE as a runtime atomic
  ([`crate::blackbox::record_set`]). Five variants (`none` /
  `small` / `mid` / `large` / `sysid`); `none`→`large` strictly
  monotonic. Mid = small + `/imu1`, `/attitude`, `/motors`,
  `/motor_state`, `/health`, `/gps_health`; Large = mid +
  `/odometry`, `/mpc`, `/tracking_error`, `/control_setpoint`,
  `/estimator_state`. `sysid` is a separate set (small + `/imu1_raw`,
  `/motors`, `/motor_state`, `/odometry`, `/mpc`, `/power`) with the INDI telemetry
  mirrors raised from 100 Hz to ≥500 Hz — actuator sysid needs the
  command and the response sampled on the timescale of the motor
  dynamics (10–30 ms), which 100 Hz undersamples — and nothing the
  identification script does not read (2026-08-24 restructure: the
  old all-topics sysid set did not fit the card at a real 8 kHz IMU).
  See [`record_set.rs`] module docs for the per-tier topic lists +
  byte-rate estimates. The value lives in
  an in-RAM `AtomicU8` and persists via the `blackbox_record_set`
  param; out-of-range values (e.g. a downgraded firmware reading a
  flash slot from a build with more variants) fall back to
  `RecordSet::DEFAULT` (Mid).

Plus runtime status:
- `blackbox status` shell command — DONE, including the last-session
  line: `blackbox_task` populates the `LAST_SESSION` static after each
  close (file, bytes, msgs, drops, encoder overflows, fault reason if
  any); the shell reads it synchronously.
- `param set blackbox_*` rejected during an active session — DONE
  (mirrors Betaflight's `blackboxMayEditConfig`; the armed case was
  already covered by the global param armed-guard, this closes the
  bench `RECORDER_HOLD` path).

Persistent flight counter — DONE. The blackbox task lazily scans
the FAT root for the highest existing `flight_NNNN.mcap` on first
use after boot, then increments in-memory. Reboots no longer
overwrite `flight_0001.mcap`. See `fat::scan_highest_seq` and the
`next_seq` helper in `blackbox::mod`.

`blackbox ls` shell command — DONE. Prints a directory listing of
the FAT root (capped at `LS_MAX_ENTRIES = 16`, with a "showing N of
M" header when truncated). Rejected with `Busy` while a recorder
session is active so the held SDMMC peripheral isn't contended.

### Stage 8 — Crash-survival flush

The single biggest gap in the original review (W6). On panic /
HardFault / brown-out, the in-flight `BufStream` cluster cache and
the most recent ~1 second of samples evaporate.

Plan:
- `panic_handler` that synchronously writes the current ring tail
  + a `PANIC` event to the still-open file before halting.
- PVD interrupt at ~2.7 V → same emergency-flush path.
- (Optional) BKPSRAM mirror of the last 16 events for cross-reset
  recovery.

Bench-testable via a debug shell command that triggers `panic!()`
mid-session.

### Binary-size review (post-Stage 8 work)

Measured cost vs the no-storage `foxeerh743` build:

| Stage                                   | blackbox `.text` | net delta |
|-----------------------------------------|------------------|-----------|
| Initial multi-tier subsystem            | ~210 KB          | -         |
| Drop superfloppy FAT layout (MBR-only)  | ~165 KB          | **−45 KB** |
| Drop `Debug2Format` in error logs       | ~121 KB          | **−85 KB net (vs initial)** |

**Why dropping superfloppy was the big win.** With both layouts
supported, every `embedded_fatfs::*` operation got monomorphised
twice: once for the MBR partition slice (`StreamSlice<BufStream<…>>`)
and once for the bare BufStream (superfloppy). `FileSystem::new`,
`Dir::create_file`, `File::flush`, `DirIter::next`,
`DirEntryData::deserialize`, `find_free_cluster`, `write_fat` —
each appeared twice. SD cards from any modern host OS ship with
MBR; superfloppy is 1990s-era legacy. The MBR-only check refuses
the superfloppy branch with a clear "reformat as MBR + FAT32"
message and halves the FAT-layer instantiations.

**Remaining candidates (not yet acted on):**

1. **Drop `lfn` feature in `embedded-fatfs`** — saves an estimated
   ~10–15 KB by removing the UCS-2 long-filename path. Cost: every
   on-card filename must fit 8.3, so we'd rename `flight_NNNN.mcap`
   to e.g. `FLT0001.MCA`. Breaks the docs / `analysis/read_mcap.py`
   examples — UX-disruptive but reversible if size pressure ever
   returns.

2. **Fold the `next_seq` scan into the same FS mount as
   `write_file`** — currently we mount, scan, unmount, then mount
   again to write. One combined op would shave the runtime cost of
   the second CMD0/ACMD41 cycle but **not** binary size (both
   paths share the same `FileSystem<IO>::new` monomorphisation
   already; that's why dropping superfloppy mattered and this
   doesn't).

3. **Move `blackbox ls` behind a debug feature** — if shell
   inspection isn't needed in the production firmware, gating
   `dispatch_blackbox_ls` + the scan path saves the 1.6 KB shell
   handler plus part of `DirIter::next`. Modest win (~3–5 KB).

4. **Compress JSON schemas** — each topic carries a ~300-byte
   pretty-printed schema in `.rodata`. Minified versions (no
   whitespace, single-line) would save ~1 KB total. Tiny win,
   trivial change.

The biggest single remaining symbol is the
`__blackbox_task_task` async state machine itself (~25 KB), which
holds the futures for `recorder::run_session` and `run_ls`.
That's intrinsic to the multi-handler design; reducing it would
require splitting the task into sub-tasks, which costs RAM (each
task gets its own `TaskStorage` / future arena).

### Stages 9+ (parking lot)

Deferred until we've actually flown a few hundred sessions:

- ~~**CMD25 multi-block writes**~~ — DONE: `WriteCombiner` in
  `sdmmc_block.rs`; the HAL already routed multi-block slices to
  CMD25, only `BufStream`'s one-block cache stood in the way.
- **CMD38 TRIM** on `blackbox erase` to reset the freed cluster region.
- **GPS-derived wall-clock anchor** in the header `Metadata` record (W8).
- **MCAP `Statistics` + `Chunk`** records — enables `mcap cat`
  without flags, indexed seek on long sessions.
- **Telemetry-streamed mirror** (CRSF telem-frame or MAVLink) —
  log-without-card mode.
- **Move recorder to its own interrupt executor** below the control
  loop priority (W13) — currently shares the thread executor with
  USB CDC; a SD page-erase pause could stall the shell.

---

## Current state — file map

```
crates/cybflight/src/blackbox/
├── mod.rs            Task + RECORDER_HOLD + should_record() + dispatch
├── recorder.rs       Stage 6 op: arm-triggered FlightRecorder body
├── record_set.rs     RecordSet enum + per-tier topic-set slices + global atomic
├── fat.rs            FAT mount/unmount + FileBody trait + scan helpers
├── mcap.rs           MCAP record framer
├── cbor.rs           CBOR writer
├── sdmmc_block.rs    SDMMC adapter (only embassy-stm32::sdmmc consumer)
└── topics/
    ├── mod.rs        TopicDef struct (channel_id + topic + schema) + plan note
    ├── imu.rs        /imu1      channel id 1: Imu.v2 positional array (wire fn + golden
    │                            tests in cybflight_core::blackbox_wire; temp_c rides in /health)
    ├── attitude.rs   /attitude  channel id 2: Mahony IMU-only attitude (~100 Hz from
    │                            estimation::mahony_task — the only attitude record on
    │                            no-mocap/no-GPS builds; independent of the ESKF's
    │                            /odometry.pose.orientation on est_pos_* builds)
    ├── rc.rs         /rc        channel id 3: schema + CBOR encoder
    ├── events.rs     /events    channel id 4: ARM/DISARM/FAILSAFE/MISSION_*/LOG_END encoder
    ├── odometry.rs   /odometry  channel id 5: VehicleOdometry.v2 positional array (~1 kHz;
    │                            wire fn + golden tests in cybflight_core::blackbox_wire)
    ├── mpc.rs        /mpc       channel id 6: OCP solver command + telemetry
    ├── motors.rs     /motors    channel id 7: INDI per-motor normalized output (100 Hz, commanded)
    ├── motor_state.rs /motor_state channel id 8: KF-fused per-motor ω + ω̇ + raw eRPM (100 Hz, achieved)
    ├── tracking_error.rs /tracking_error channel id 9: controller-reported `reference - actual` (cascade/MPC: 50–100 Hz, INDI: 100 Hz)
    ├── imu_raw.rs    /imu1_raw  channel id 13: ImuRaw.v2 positional array — pre-biquad-LP
    │                            mirror of /imu1 (IMU rate; Sysid only)
    ├── control_setpoint.rs /control_setpoint channel id 14: outer-loop RATE_COMMAND mirror (50–100 Hz; Large)
    ├── power.rs      /power     channel id 15: pack voltage raw + `batt_lpf_hz`-filtered, current, mAh
    │                            (100 Hz ADC tick; Sysid only)
    ├── mpc_cost.rs   /mpc_cost  channel id 16: cost residual in force for each solve + the
    │                            effective weights (every commanding outer-loop tick; Mid)
    ├── estimator_state.rs /estimator_state channel id 12: ESKF gyro/accel bias snapshot (10 Hz; Large, est_eskf-gated)
    ├── health.rs     /health    channel id 10: `.v2` positional array — `health` shell mirror + INDI timing + RC link (20 Hz)
    └── gps_health.rs /gps_health channel id 11: flat snapshot of GPS_HEALTH + LATEST_NAV_PVT
                                 (polled 20 Hz, emitted on change + 1 Hz heartbeat; stays at
                                  NotConfigured on non-est_pos_gps builds)

    Channel ids must be globally unique — MCAP keys both Channel and
    Schema records by id, so two topics sharing one produces a malformed
    file whose messages are mislabelled. `topics::ALL` is the registry and
    `record_set.rs` asserts uniqueness at compile time; take the next free
    id from `ALL`, never from this list. (Ids 10/11 were each issued twice
    precisely because this list said "reserved" instead of a number.)

crates/cybflight-drivers/src/blackbox_storage.rs    BlockStore trait + BlockStoreError

crates/bsp/{sakurah743,foxeerh743}/src/lib.rs       HAS_BLACKBOX_STORAGE: bool
```

External git deps (workspace `Cargo.toml` `[patch.crates-io]` for
`block-device-driver`; per-crate `[dependencies]` for `embedded-fatfs`,
`block-device-adapters`, `embedded-partitions`):

```toml
embedded-fatfs        git=MabezDev/embedded-fatfs rev=518528cc default-features=false features=["lfn"]
block-device-adapters git=MabezDev/embedded-fatfs rev=518528cc
embedded-partitions   git=MabezDev/embedded-fatfs rev=518528cc
block-device-driver   git=MabezDev/embedded-fatfs rev=518528cc  (via [patch.crates-io])
aligned               0.4
```

The `[patch.crates-io]` for `block-device-driver` is **load-bearing**
— `embassy-stm32` depends on the registry version, the
`embedded-fatfs` ecosystem on its git counterpart. Without the
patch they're treated as different traits and `BufStream<StorageDevice>:
ReadWriteSeek` fails to resolve.

The `cybflight-msgs` `[patch.utadr]` is **currently active** for
local development on the `EstimatorBias` type added in
cybflight-msgs 0.1.20. Re-comment after the new revision is
published.

---

## Operational guide

The complete user instruction for the blackbox, desk to desk: enable →
record → download → analyse → clean up.

### 0. Prerequisites

- **A board with storage.** The subsystem is gated on the BSP const
  `HAS_BLACKBOX_STORAGE`: `sakurah743` and `micoair743v2` have SDMMC
  and record; `foxeerh743` does not (every `blackbox *` shell command
  answers "no storage backend on this board").
- **A microSD card formatted MBR + FAT32 with 32 KB clusters**:
  `sudo mkfs.fat -F 32 -s 64 -v /dev/sdX1` (the SD Association
  formatter also picks 32 KB on cards ≥ 32 GB). Superfloppy (FAT BPB
  at LBA 0), GPT, exFAT and ext4 are all refused with a "reformat as
  MBR + FAT32" style error. Cluster size matters for throughput:
  every cluster allocation costs ~8 FAT/directory round trips on the
  card (bench: 16 KB clusters → 9 card commands per 16 KB of log,
  `blackbox status` shows `max 32 per cmd`); 32 KB halves that and
  lets the CMD25 batches run 64 blocks. **Do not use `-s 128`
  (64 KB) on cards ≤ 4 GB**: the volume ends up with fewer than
  65,525 clusters, which the spec — and this firmware's fatfs —
  classifies as FAT16, so the FAT32 BPB is rejected at mount with
  "FAT boot sector unreadable". `mkfs.fat -v` prints the cluster
  count; it must exceed 65,525.
- **Host tools** (for download + analysis):
  `pip install pyserial mcap cbor2` (+ `numpy matplotlib` for
  `analysis/plot_mcap.py`), plus the
  [`mcap` CLI](https://mcap.dev/guides/cli) if you want `mcap doctor` /
  `mcap info`. Optional: Foxglove Studio for interactive plots.

### 1. What gets recorded

One MCAP file per session, `flight_NNNN.mcap` in the FAT root. The
sequence number continues from the highest existing file across
reboots. Messages are CBOR; the two hot topics use compact positional
arrays (`.v2` schemas, self-describing via `prefixItems`).

| Topic | id | Rate | Tier | Content (source) |
|---|---|---|---|---|
| `/events` | 4 | on change | small+ | ARM / DISARM / FAILSAFE / RC_LOSS / MISSION_* / INNER_SILENT / POWER_STALE / RECORDER_OVERRUN / PANIC / LOG_END … brackets |
| `/rc` | 3 | ~150 Hz | small+ | stick + aux channels (RX task) |
| `/attitude` | 2 | ~100 Hz | mid, large | **Mahony IMU-only attitude** (`estimation::mahony_task`) — present in *every* build, the only attitude on no-mocap/no-GPS vehicles; yaw drifts, tilt is trustworthy |
| `/odometry` | 5 | ~1 kHz | large, sysid | ESKF fused pose + twist; `pose.orientation` is the fused attitude (empty until the estimator initialises — e.g. mocap off) |
| `/mpc` | 6 | ~50–100 Hz | large, sysid | OCP solver command + iterations/convergence/solve-time |
| `/motors` | 7 | 100 Hz (≥500 Hz on sysid) | mid, large, sysid | INDI per-motor normalized command (INDI runs at IMU rate / `indi_ctrl_div`, 2 kHz on 8 kHz vehicles; mirrors are decimated from that) |
| `/motor_state` | 8 | 100 Hz (≥500 Hz on sysid) | mid, large, sysid | KF-fused per-motor ω, ω̇, raw eRPM (achieved) |
| `/tracking_error` | 9 | 50–100 Hz | large | controller `reference − actual` |
| `/imu1` | 1 | IMU rate (1/8 kHz) | mid, large | post-biquad-LP accel + gyro. **Really** 8 kHz on 8 kHz vehicles since the SPI-clock fix (2026-08-24; earlier logs show ~2.5 kHz — that was the bus, not the chip) |
| `/control_setpoint` | 14 | 50–100 Hz | large | outer-loop RATE_COMMAND mirror |
| `/estimator_state` | 12 | ~10 Hz | large (est_eskf builds) | ESKF gyro/accel bias snapshot |
| `/health` | 10 | 20 Hz | mid, large, sysid | `SystemHealth.v2` positional array: the live `health` shell line + IMU die temp, plus INDI worst iteration / worst inter-iteration gap per window (`step_stats`, otherwise shell-only) and RC link quality / RSSI / frame age (logged nowhere else — `/rc` carries channels only). ~100 B/record |
| `/gps_health` | 11 | on change + 1 Hz | mid, large | GPS fix/RTK state (NotConfigured on non-GPS builds) |
| `/imu1_raw` | 13 | IMU rate | sysid | pre-filter IMU mirror (filter tuning / RPM-notch fits) |
| `/power` | 15 | 100 Hz | sysid | pack voltage **raw and filtered** (`batt_lpf_hz` PT1 — the filtered value is what the INDI thrust table linearizes at), unfiltered current, mAh, cell count. Raw/filtered pair is for fitting the filter lag against throttle-punch IR drop |

**Attitude, specifically:** `/attitude` and `/odometry.pose.orientation`
are *independent estimates* (Mahony vs ESKF). On a mocap/GPS build both
record — disagreement is diagnostic (external reference vs IMU fault).
On an `outer_rate` build with no position source, `/attitude` is the
only attitude in the log.

### 2. Record-set tiers

`blackbox set <none|small|mid|large|sysid>` — persists via the
`blackbox_record_set` param (`param save`), takes effect next session.
| Tier | Topics | ~KB/s @1 kHz / @8 kHz IMU |
|---|---|---|
| `none` | — | 0 |
| `small` | `/events`, `/rc` | 19 |
| `mid` (default) | small + `/imu1`, `/attitude`, `/motors`, `/motor_state`, `/health`, `/gps_health` | ~110 / ~540 |
| `large` | mid + `/odometry`, `/mpc`, `/tracking_error`, `/control_setpoint`, `/estimator_state` | ~230 / ~660 |
| `sysid` | small + `/imu1_raw`, `/motors`, `/motor_state`, `/odometry`, `/mpc`, `/power`; INDI mirrors at ≥500 Hz | ~240 / ~610 |

`none`→`large` are strict supersets: Mid is *what the vehicle did*,
Large adds *why* (estimator + outer-loop chain). `sysid` is a separate
set — exactly what `analysis/sysid_mcap.py` reads (pre-filter IMU in
place of the filtered one, the commanded/achieved motor pair at frame
rate, odometry for the drag fit) and nothing more, because at a real
8 kHz IMU the old all-topics sysid set (~1.2 MB/s) was twice what the
card path sustains. Rates use the measured 62 B/msg for IMU topics;
`/odometry` is ~1 kHz only with an estimator running. See
`record_set.rs` module docs for the derivation.

Two finer-grained shaping params layer on top of the tier (both
snapshotted at session start; both echoed into the file's `Metadata`
record; `param save` to persist):

- `param set blackbox_rate_div <N>` — emit every Nth message on the
  high-rate topics only (`/imu1`, `/imu1_raw`, `/odometry`). E.g.
  `4` on an 8 kHz sysid flight logs 2 kHz IMU instead of 8 kHz.
  Decimation does not advance the MCAP sequence counters, so real
  drops remain distinguishable from configured skipping.
- `param set blackbox_mute_mask <bits>` — mute topics by channel id
  (bit N = id N, see the table above; `/events` is not maskable).
  E.g. `512` (bit 9) drops `/tracking_error` from a Large session.

The card path sustained ~600 KB/s (~545 KB/s of message bytes) on the
bench (2026-08-24), so at a real 8 kHz IMU `large` and `sysid` both sit
at that ceiling — `blackbox_rate_div 2` gives margin. A
throughput-bound recording is unmistakable in `read_mcap.py
--summary`: periodic 28–35 ms holes on every topic (the pre-restructure
all-topics sysid log at 8 kHz lost 56 % of `/imu1_raw` that way) and
`/health` below 20 Hz.

Editing `blackbox_*` params is refused while a session is recording.

### 3. Shell commands

| Command | What it does | Output |
|---|---|---|
| `blackbox record on` | Set `RECORDER_HOLD = true` — recorder opens session on next 40 ms debounce-confirmed edge | `blackbox record: ON ...` |
| `blackbox record off` | Clear `RECORDER_HOLD` — recorder closes after next 50 ms keep-going poll | `blackbox record: OFF ...` |
| `blackbox status` | Read `IS_ARMED` + `RECORDER_HOLD` + active record set + shaping params + last-session outcome + card I/O counters (write cmds / blocks / max batch, reads, CMD25 fallbacks) synchronously | state line + `IS_ARMED=…, RECORDER_HOLD=…` + `record_set=<tier>, rate_div=…, mute_mask=…` + `last session: /flight_NNNN.mcap …  [ok\|FAULT: reason]` |
| `blackbox set <tier>` | Change active record set (`none\|small\|mid\|large\|sysid`); takes effect on next session | `blackbox set: record_set = <tier>` |
| `blackbox ls` | List files in the FAT root; rejected with `Busy` while recorder is active | `N file(s)` header + one `/<name>  <size> bytes` line per entry (capped at `LS_MAX_ENTRIES = 16`) |
| `blackbox get <f> [<off> [<len>]]` | Stream a FAT-root file to the host as **raw binary** (use `just blackbox-pull`, not a bare terminal); rejected while recording | `OK <size>` + `<size>` raw bytes + `CRC <hex8>` trailer (see "Downloading logs") |
| `blackbox rm <f>` | Delete a FAT-root file to free card space; rejected while recording | `blackbox rm: removed /<name>` |
| `blackbox clean` | Delete every `flight_NNNN.mcap` in the FAT root in one card session (other files untouched); resets the in-memory sequence cache so numbering restarts at `flight_0001`; rejected while recording | `blackbox clean: removed N flight log(s); numbering restarts at flight_0001` (or `ABORTED after removing N …` if a delete failed) |

### 4. Recording workflows

**Production (no shell involved)**: arming via RC switch starts a
session; disarm (or failsafe disarm) closes it. Nothing to configure
beyond the tier.

**Pure bench logging**:
```
> blackbox record on
[ wait a few seconds / excite the vehicle ]
> blackbox record off
[ defmt: blackbox: closed /flight_0001.mcap (XX bytes, NN msgs, 0 drops, 0 enc-overflows) ]
```

**Pre-arm logging then real flight**:
```
> blackbox record on            # recorder opens flight_NNNN.mcap immediately
[ pre-flight checks happen here, all logged ]
[ pilot arms via RC switch ]    # arm latched; record_on no longer the keep-going condition
[ flight ]
[ pilot disarms ]               # disarm-edge stops the session, flushes, unmounts
```
`RECORDER_HOLD` auto-clears after the session so the next cycle needs
a deliberate fresh start.

**SD throughput probe**: `blackbox set large` → `blackbox record on`
for a few seconds → `off` → check `bytes` / `drops` in the
closed-session defmt line.

**INDI sysid recording**:
```
> blackbox set sysid
> blackbox record on
[ fly a chirp / step / aggressive maneuver ]
> blackbox record off
```
Use the `sysid` tier: it carries exactly what the actuator /
thrust+drag / moment fits read (`/imu1_raw`, `/motors`,
`/motor_state`, `/odometry`) with the motor mirrors at ≥500 Hz so the
10–30 ms motor time constants are actually sampled, plus `/mpc` (the
outer-loop command and solver iterations / solve time) so the same
flight can be used to check the controller. Mid also works for
the thrust/drag/moment fits at 100 Hz motor data (the script tightens
its ω̇ low-pass on 100 Hz logs). Manual stick excitation is easiest in
`outer_rate` / `outer_geometric`.

### 5. Downloading logs (`just blackbox-pull`)

Logs pull over the USB CDC shell — no card popping:

```sh
just blackbox-pull                      # every *.mcap not already in ./logs/
just blackbox-pull flight_0003.mcap     # one file
just blackbox-pull --list               # show card contents
just blackbox-pull --all --delete       # pull everything, then blackbox rm each
just blackbox-pull --force f.mcap       # re-pull over an existing local copy
```

Files land in `logs/`, written atomically and only after the CRC-32
verifies. Wire protocol (`blackbox get <file> [<offset> [<len>]]`):

```
OK <size>\r\n          <- or an error line, and no binary follows
<size raw bytes>
\r\nCRC <hex8>\r\n     <- CRC-32 (crc32fast / zlib.crc32) of the raw bytes
```

A mid-stream fault truncates the binary and emits `\r\nERR <reason>\r\n`
in place of the CRC trailer; `<offset>`/`<len>` clamp to the file size,
so a stalled pull can resume from where it stopped.

Plumbing: the shell handler signals `GET_REQUEST`; `blackbox_task`
(which owns the SDMMC peripheral) streams the file through
`fat::read_file<S: FileSink>` — the read-side mirror of
`FileBody`/`write_file`, reusing the same `FileSystem`
monomorphization — into the static `GET_STREAM` channel (2 × 512 B =
double buffering), which the handler drains into `write_all`. Binary
is safe on the wire because `write_all` is byte-transparent and the
shell loop does not service `SHELL_OUT` stream lines while a dispatch
handler runs. Every channel send/receive is timeout-bounded so a dead
host, unplugged cable, or hung card can't wedge either side, and the
sink polls `IS_ARMED` per chunk — arming mid-transfer aborts the pull
within one chunk and the recorder session starts normally.

Throughput is USB-FS-CDC-bound: expect ~0.3–0.8 MB/s, i.e. a 10 MB
flight log in ~15–35 s. Anything faster would need a dedicated bulk
endpoint — out of scope for the shell path.

### 6. Analysis on host

**Quick look / verification — `analysis/read_mcap.py`**
(`pip install mcap cbor2`). Default mode is a pure greppable dump,
one line per message; decoding is schema-driven, so it stays correct
as wire formats evolve:

```sh
python3 analysis/read_mcap.py logs/flight_0001.mcap --summary
#   ch 1 /imu1        14496 msgs   926.8 Hz  seq-gaps(drops): 87
#   ch 2 /attitude     1488 msgs    95.1 Hz
#   ...
#   events (3):  +0.000s ARM   +12.648s DISARM   +15.648s LOG_END

python3 analysis/read_mcap.py f.mcap | head              # ARM event up top
python3 analysis/read_mcap.py f.mcap | tail              # ends at LOG_END
python3 analysis/read_mcap.py f.mcap | grep events       # all events in order
python3 analysis/read_mcap.py f.mcap --topic /attitude   # one topic only
python3 analysis/read_mcap.py f.mcap | grep '/odom'      # ESKF pose+twist (large / sysid)
python3 analysis/read_mcap.py f.mcap | grep '/motors'    # INDI per-motor output
```

`seq-gaps(drops)` in the summary = messages the recorder shed under
SD backpressure (drop-aware sequence numbers make them visible); a
missing-footer warning means the session crashed / lost power before
close (the periodic flush still preserves everything up to the last
~1 s).

**Standard plots — `analysis/plot_mcap.py`**
(`pip install mcap cbor2 numpy matplotlib`). Three figures, each
produced only when its topic has data:

1. *XY position* — top-down 2D trajectory from `/odometry`
   (`pos_x`/`pos_y`; skipped with a note when the estimator never
   initialised).
2. *Attitude* — roll / pitch / yaw in degrees vs time from the logged
   quaternions ([w,i,j,k] → intrinsic Z-Y-X euler). Overlays **both**
   estimates when present: `/attitude` (Mahony, solid) and
   `/odometry.pose.orientation` (ESKF, dashed) — divergence between
   the two is itself diagnostic.
3. *IMU* — accel [m/s²] + gyro [rad/s] xyz vs time from `/imu1`
   (`--raw` switches to `/imu1_raw`; sysid-tier logs carry only the raw stream and are picked up automatically).

ARM / DISARM / FAILSAFE events overlay as vertical lines on the time
axes.

```sh
python3 analysis/plot_mcap.py logs/flight_0001.mcap              # interactive
python3 analysis/plot_mcap.py f.mcap --no-show --save out        # out_xy.png,
                                                                 # out_attitude.png,
                                                                 # out_imu.png
python3 analysis/plot_mcap.py f.mcap --raw                       # pre-filter IMU
```

Sanity anchor when reading the attitude + IMU figures together: a
static tilt of θ shows up as `accel_x ≈ g·sin θ` and
`accel_z ≈ g·cos θ` — if the two figures disagree on θ, suspect the
estimate, not the sensor.

**System identification — `analysis/sysid_mcap.py`**: thrust/drag,
actuator, and moment coefficient fits from a `sysid`-tier flight.
Operating guide + derivations in [system_id.md](system_id.md).

**Structural check — `mcap` CLI**:

```sh
mcap doctor flight_0001.mcap     # must print no issues
mcap info   flight_0001.mcap     # channel count depends on tier:
                                 #   small = 2  (events, rc),
                                 #   mid   = 8  (+ imu1, attitude, motors, motor_state,
                                 #               health, gps_health),
                                 #   large = 13 (+ odometry, mpc, tracking_error,
                                 #               control_setpoint, estimator_state),
                                 #   sysid = 6  (events, rc, imu1_raw, motors,
                                 #               motor_state, odometry)
                                 # every channel id must be distinct — a repeat means
                                 # the topics::ALL uniqueness assert was bypassed
```

Note the firmware writes *streamed* MCAP (no summary section — legal,
`mcap doctor` passes); use linear readers (`mcap cat`,
`mcap.stream_reader` in Python) rather than index-requiring ones.

**Interactive plots — Foxglove Studio**: drag-and-drop the file.
`/imu1` plots accel/gyro, `/odometry` fused pose+twist
(`pose.orientation` = ESKF attitude), `/attitude` the independent
Mahony estimate, `/rc` stick channels, `/events` discrete markers on
the timeline.

**Full flight report**: the sibling `cybflight-review` repo renders a
self-contained Bokeh HTML report (`uv run cybflight-review <path>`).
(Its bundled `read_mcap.py` plotter predates the `.v2` array formats —
prefer this repo's `analysis/read_mcap.py` for anything recent.)

### 7. Card housekeeping

- `blackbox ls` to see what's on the card, `blackbox rm <f>` to free
  space (or `just blackbox-pull --all --delete` to archive-then-free).
- Deleting the highest-numbered flight file never causes a name
  collision: the in-RAM sequence cache only increments within a boot,
  and the on-card scan after reboot picks the highest *remaining* + 1.
- The card stays desktop-mountable at all times — popping it into a
  reader and copying files off is always a valid fallback.

### 8. Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `blackbox <op>: no storage backend on this board` | Build is for a board with `HAS_BLACKBOX_STORAGE = false` (e.g. foxeerh743) |
| `REJECTED - recorder is mid-flight (disarm first)` | ls/get/rm are refused while a session is recording — disarm or `blackbox record off` first |
| `card acquire failed (CMD0/ACMD41 …)` | Card missing, unseated, or electrically unhappy |
| `sector 0 has neither MBR signature nor FAT BPB` / `no FAT partition entry` | Wrong card format — reformat MBR + FAT32 |
| `blackbox <op>: TIMEOUT — task didn't respond` | blackbox task wedged or SD hung; the shell stays usable — retry, then power-cycle |
| Pull stalls / CRC mismatch | Re-run the pull (the tool prints a resume hint `blackbox get <f> <offset>`); a short read + `ERR` tail names the firmware-side fault |
| `drops` / `seq-gaps` in a session | SD backpressure — the recorder sheds IMU first by design; sustained drops at mid tier suggest a slow card |
| Sequence gaps at the very start of a file | Known cost of mount-at-arm-edge (~100–250 ms card acquire); see "What done looks like" §5 |
| `/odometry` empty but `/attitude` present | Estimator never initialised (mocap/GPS absent) — expected; Mahony is the fallback attitude record |

## Adding things (recipes)

### A new logged topic

1. Create `crates/cybflight/src/blackbox/topics/<name>.rs` with:
   ```rust
   pub const TOPIC: &str = "/...";
   pub const SCHEMA_NAME: &str = "...";
   pub const SCHEMA: &[u8] = br#"{ json schema }"#;
   pub const DEF: TopicDef = TopicDef { topic: TOPIC, schema_name: SCHEMA_NAME, schema_data: SCHEMA };
   pub fn encode(scratch: &mut [u8], msg: &MsgType) -> cbor::Result<usize> { ... }
   ```
2. Add `pub mod <name>;` to `topics/mod.rs`.
3. Add `topics::<name>::DEF` to whichever `TOPICS_*` arrays in
   `record_set.rs` should include the new topic.
4. In `recorder.rs`:
   - Add a `Subscriber` field on `FlightRecorder` plus a per-topic
     `<name>_seq: u32` counter.
   - Add a per-topic `emit_<name>` async method (mirror the existing
     ones — they all share the same shape).
   - Add a `select6` arm and a tier-ordered drain block calling
     `emit_<name>` (use `DRAIN_BUDGET_NORMAL` unless this topic
     should be deprioritised under SD backpressure like `/imu1`).
   - Subscribe to the source pubsub in `run_session` and populate
     the field.

### A new storage backend (SPI NOR, FRAM, ...)

1. New file `crates/cybflight/src/blackbox/<backend>_block.rs`.
2. Implement `block_device_driver::BlockDevice<512>` for the new
   adapter type.
3. Wire it into `board_init/<board>.rs` instead of `SdmmcBlockStore::new`.
4. Add `bsp::HAS_BLACKBOX_STORAGE = true` for the new board.

The `BlockStore` trait in `cybflight-drivers` and the rest of the
pipeline see only the trait; nothing in `blackbox/` changes.

### A new shell command

1. Add a `request: Signal<_, ArgType>` and `result: Signal<_, ReportType>`
   pair in `mod.rs`.
2. Add a dispatch arm in `blackbox_task`'s `select`. The current
   select is `select4(timer, LS_REQUEST, GET_REQUEST, RM_REQUEST)`;
   grow it as you add more.
3. Add a `dispatch_blackbox_<op>` shell handler in `usb_serial.rs`
   that does the trigger-then-await dance.
4. Add the line to `HELP_TEXT`.

Worked examples, simple → involved: `blackbox rm` (unit-ish payloads
both ways), `blackbox ls` (bounded `heapless::Vec` result), and
`blackbox get` (result too big for a Signal → a static `GET_STREAM`
chunk `Channel` alongside the request/result pair, with
timeout-bounded sends so neither side can wedge the other).

---

## Post-mortem of the original WIP

When this work started, the branch carried 3 commits (~2800 LOC) of
prior blackbox work — call it the "WIP recorder". A
`blackbox-architect` agent reviewed it (full report in agent memory)
and found 15 categorized issues. The biggest:

- **W1**: Files were structurally invalid MCAP — never emitted
  `DataEnd`/`Footer`/trailing `Magic`. Strict tools would reject
  every session.
- **W2**: `end_session` swallowed flush errors then committed
  metadata pointing past the failure → corrupted next-session state.
- **W3**: Documented throughput claim was 1.3 MB/s; actual peak
  was 2.0–2.3 MB/s. 4-slot ring + single-block CMD24 → silent drops
  on commodity cards.
- **W4**: SDMMC backend lived in `cybflight/blackbox/sink.rs` and
  imported `embassy_stm32::sdmmc` directly — violating the rule
  that drivers be generic over `embedded-hal-async`.
- **W5**: Pipeline gated on `#[cfg(feature = "blackbox")]` instead
  of the BSP capability flag pattern used by every other peripheral.
- **W6**: No crash-survival path. Any panic / HardFault evaporated
  the in-flight ring.

The decision was to **delete all of it** and rebuild stage-by-stage
on top of the architectural rules main already follows. That's why
this codebase has a `recorder.rs` instead of a `sink.rs`, why
`SdmmcBlockStore` is the only `embassy_stm32::sdmmc` consumer, and
why every stage's verification recipe explicitly runs `mcap doctor`.

The full review is in `.claude/agent-memory/blackbox-architect/`.

---

## What "done" looks like

Ship-blocking, in priority order:

1. **Stage 7 status command + persistent flight counter.**
   Without these, the user can't tell whether logging is on, and
   `flight_0001.mcap` overwrites itself across reboots.
2. ~~Stage 8 crash flush.~~ Done: the capture loop commits the FAT
   directory entry every second (`FLUSH_INTERVAL`), and a mid-session
   write fault salvages the partial file (`OpError::PartialWrite`)
   instead of leaving a 0-byte entry.
3. **Profile selection** — superseded by the runtime record-set tiers
   (`none|small|mid|large|sysid`, `blackbox set <tier>`); no cargo
   features needed.
4. ~~**CMD25 multi-block writes**~~ Done (`WriteCombiner`, 32 KB
   batches). Re-measure the `blackbox set large` throughput probe
   against the old ~2.0–2.3 MB/s CMD24 ceiling and check `drops`.
5. **Mount at arm-edge costs the takeoff transient.** `run_session`
   pays card acquisition (CMD0/ACMD41, ~100–250 ms on cheap cards) +
   FAT mount + ~13 KB of schema/metadata before the first sample can
   be written; subscribers are taken first, so that window shows up
   honestly as sequence gaps at the start of every file — but the
   samples are still lost. The fix is keeping the card acquired and
   the FS mounted across sessions so the arm edge only pays
   `create_file`. Deferred, not forgotten: `embedded_fatfs`'s
   ownership design makes it a real refactor —
   `FileSystem<StreamSlice<BufStream<StorageDevice<'_>>>>` borrows
   `&mut Sdmmc` through four layers, so a long-lived mount held in
   `blackbox_task`'s loop while also supporting drop-and-remount on
   card error runs into the loop-carried-borrow pattern rustc
   rejects. Doing it well probably means owning the Sdmmc inside a
   state enum that is *moved* through mount/unmount transitions
   rather than borrowed.

After those four, this is production-comparable with Betaflight's
blackbox. Points where it should *exceed* Betaflight: typed-Rust
encoders (no field-table drift), MCAP (Foxglove integration without
log-converter scripts), partition-aware FAT (cards stay
desktop-mountable).
