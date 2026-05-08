# SD-card blackbox: design + staged plan

## What this document is

A handoff doc for the next agent picking up the cybflight blackbox
work. Reading order:

1. **Goal** — why we're rebuilding this.
2. **Architecture** — the layers, where they live, what's the seam between them.
3. **Staged plan** — what's done, what's next, why the staging matters.
4. **Current state** — exact line numbers + file map.
5. **Operational guide** — how to verify a change end-to-end.
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
  is verifiable from a desk with `mcap doctor` / `python read_mcap.py`,
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
| **Topics** | `cybflight/blackbox/topics/{imu,attitude,events}.rs` | Per-topic schemas + CBOR encoders. Single source of truth for each `msgs::*` type's wire format. |
| **FAT pipeline** | `cybflight/blackbox/fat.rs` | `write_file<P: FileBody>` mounts FAT, finds the FAT partition (handles MBR vs superfloppy), creates the file, hands the writer to `body.write(...)`, flushes + unmounts. |
| **Partition** | `embedded-partitions` (git-pinned) | Reads sector 0; classifies MBR vs superfloppy; returns a `StreamSlice` over the FAT partition. |
| **Filesystem** | `embedded-fatfs` (git-pinned) | FAT16/FAT32 read+write. |
| **Byte ↔ block** | `block-device-adapters::BufStream` (git-pinned) | RMW-cached 512-byte block buffer; bridges async byte-level Read/Write/Seek to `BlockDevice<512>`. |
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
`/events`, closes cleanly. Logged topics: `/imu1`, `/attitude`,
`/rc`, `/events`.

Two ways to trigger:
- Real arm via RC / failsafe path (production)
- `blackbox record on/off` shell (bench)

### Stage 7 — Persistent params + status (NEXT)

Bring back the user-facing knobs that the original WIP had:

- `blackbox_device` (off / sdcard / future-spinor)
- `blackbox_mode` (off / normal / always)
- `blackbox_rate_div` — per-topic rate divider; default 1 (full rate)
- `blackbox_fields_disabled` — bitmask to mute high-rate topics
- **record set tier** — DONE as a runtime atomic
  ([`crate::blackbox::record_set`]). Four variants (`none` /
  `small` / `mid` / `large`); strictly monotonic — each tier is a
  superset of the previous. The cheap controller / sysid topics
  (`/control_setpoint`, `/estimator_state`) live in Mid; only
  `/imu1_raw` is gated on Large (the only topic with a meaningful
  bandwidth cost). See [`record_set.rs`] module docs for the
  per-tier topic lists + byte-rate estimates. The value lives in
  an in-RAM `AtomicU8` and persists via the `blackbox_record_set`
  param; out-of-range values (e.g. a downgraded firmware reading a
  flash slot from a build with more variants) fall back to
  `RecordSet::DEFAULT` (Mid).

Plus runtime status:
- `blackbox status` shell command — DONE for the recording-state +
  trigger-source view (reads `IS_ARMED` and `RECORDER_HOLD`
  synchronously, no task round-trip). **Still TODO**: last-session
  counters, drop counts, last error — needs a `LAST_SESSION` static
  populated by `blackbox_task` after each session closes.
- `param set blackbox_*` rejected during active session (mirrors
  Betaflight's `blackboxMayEditConfig`)

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
   to e.g. `FLT0001.MCA`. Breaks the docs / `read_mcap.py`
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

- **CMD25 multi-block writes** for ≥4-slot batches (W3 throughput).
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
    ├── imu.rs        /imu1      channel id 1: schema + CBOR encoder
    ├── attitude.rs   /attitude  channel id 2: schema + CBOR encoder
    ├── rc.rs         /rc        channel id 3: schema + CBOR encoder
    ├── events.rs     /events    channel id 4: ARM/DISARM/FAILSAFE/MISSION_*/LOG_END encoder
    ├── odometry.rs   /odometry  channel id 5: ESKF fused pose+twist (~1 kHz)
    ├── mpc.rs        /mpc       channel id 6: OCP solver command + telemetry
    ├── motors.rs     /motors    channel id 7: INDI per-motor normalized output (100 Hz, commanded)
    ├── motor_state.rs /motor_state channel id 8: KF-fused per-motor ω + ω̇ + raw eRPM (100 Hz, achieved)
    ├── tracking_error.rs /tracking_error channel id 9: controller-reported `reference - actual` (cascade/MPC: 50–100 Hz, INDI: 100 Hz)
    ├── imu_raw.rs    /imu1_raw  channel id 10: pre-biquad-LP IMU mirror of /imu1 (8 kHz; Large only)
    ├── control_setpoint.rs /control_setpoint channel id 11: outer-loop RATE_COMMAND mirror (50–100 Hz; Mid+Large)
    ├── estimator_state.rs /estimator_state channel id 12: ESKF gyro/accel bias snapshot (10 Hz; Mid+Large, est_eskf-gated)
    ├── health.rs     /health    (channel id reserved): post-flight mirror of the live `health` shell line (20 Hz)
    └── gps_health.rs /gps_health (channel id reserved): flat snapshot of GPS_HEALTH + LATEST_NAV_PVT (20 Hz; stays at NotConfigured on non-est_pos_gps builds)

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

### Shell commands

| Command | What it does | Output |
|---|---|---|
| `blackbox record on` | Set `RECORDER_HOLD = true` — recorder opens session on next 40 ms debounce-confirmed edge | `blackbox record: ON ...` |
| `blackbox record off` | Clear `RECORDER_HOLD` — recorder closes after next 50 ms keep-going poll | `blackbox record: OFF ...` |
| `blackbox status` | Read `IS_ARMED` + `RECORDER_HOLD` + active record set synchronously | state line + `IS_ARMED=…, RECORDER_HOLD=…` + `record_set=<tier>` |
| `blackbox set <tier>` | Change active record set (`none\|small\|mid\|large\|sysid`); takes effect on next session | `blackbox set: record_set = <tier>` |
| `blackbox ls` | List files in the FAT root; rejected with `Busy` while recorder is active | `N file(s)` header + one `/<name>  <size> bytes` line per entry (capped at `LS_MAX_ENTRIES = 16`) |

### Bench-test recipes

**Pure bench logging**:
```
> blackbox record on
[ wait a few seconds ]
> blackbox record off
[ defmt: blackbox: closed /flight_0001.mcap (XX bytes, NN msgs, 0 drops) ]
```

**Pre-arm logging then real flight**:
```
> blackbox record on            # recorder opens flight_0001.mcap immediately
[ pre-flight checks happen here, all logged ]
[ pilot arms via RC switch ]    # arm latched in `was_armed`; record_on no longer the keep-going condition
[ flight ]
[ pilot disarms ]               # disarm-edge stops the session, flushes, unmounts
[ defmt: blackbox: closed /flight_0001.mcap ]
```

**SD throughput probe**: `blackbox set large` → `blackbox record on`
for a few seconds → `blackbox record off` → check the `bytes` /
`drops` numbers in the closed-session defmt log.

**INDI sysid recording**:
```
> blackbox set large
> blackbox record on
[ fly a chirp / step / aggressive maneuver — see analyze.py for fits ]
> blackbox record off
```
Mid already carries everything `analyze.py` needs for the
actuator / thrust+drag / moment fits (`/motors`, `/motor_state`,
`/odometry`, `/control_setpoint`, `/tracking_error`,
`/estimator_state`, post-LP IMU). Step up to Large when you also
want pre-biquad-LP IMU for RPM-notch fits — that's the only
sysid-relevant signal that's not in Mid. INDI sysid is easiest in
`outer_rate` / `outer_geometric` (manual stick excitation); on
`outer_mpc` builds the topic mix still works but the input
excitation has to come from somewhere else.

### Verification on host

After popping the card:

```sh
mcap doctor flight_0001.mcap     # must print no issues
mcap info   flight_0001.mcap     # channel count depends on tier:
                                 #   small = 2  (events, rc),
                                 #   mid   = 13 (+ attitude, odometry, mpc, motors, motor_state,
                                 #               tracking_error, imu1, control_setpoint,
                                 #               estimator_state, health, gps_health),
                                 #   large = 14 (+ imu1_raw)
python3 read_mcap.py flight_0001.mcap | head            # see ARM event up top
python3 read_mcap.py flight_0001.mcap | tail            # see DISARM + LOG_END
python3 read_mcap.py flight_0001.mcap | grep events     # all events in order
python3 read_mcap.py flight_0001.mcap | grep '/rc'      # stick / aux samples
python3 read_mcap.py flight_0001.mcap | grep '/odom'    # ESKF pose+twist (mid+)
python3 read_mcap.py flight_0001.mcap | grep '/mpc'     # OCP commands     (mid+)
python3 read_mcap.py flight_0001.mcap | grep '/motors'  # INDI per-motor output (mid+)
```

The included `read_mcap.py` decodes CBOR via the `cbor2` Python
package (`pip install mcap cbor2`).

For Foxglove Studio: drag-and-drop the file. All four topics
should appear in the sidebar: `/imu1` plots accel/gyro, `/attitude`
plots quaternion, `/rc` plots stick channels (when an RX is
connected), `/events` shows discrete event markers on the timeline.

### Build

```sh
cargo build -p cybflight                                                # foxeerh743 (default; HAS_BLACKBOX_STORAGE=false)
cargo build -p cybflight --no-default-features \
  --features board_sakurah743,rx_crsf,outer_mpc                         # sakurah743 (HAS_BLACKBOX_STORAGE=true)
```

Both must compile clean. The foxeerh743 build verifies the entire
blackbox subsystem dead-code-eliminates correctly when
`HAS_BLACKBOX_STORAGE=false`.

---

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
   select is `select(timer, LS_REQUEST)`; bump to `select3`/`select4`
   etc. as you add more.
3. Add a `dispatch_blackbox_<op>` shell handler in `usb_serial.rs`
   that does the trigger-then-await dance.
4. Add the line to `HELP_TEXT`.

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
2. **Stage 8 crash flush.** Without this, the most valuable seconds
   of any actual crash evaporate.
3. **Profile selection** (Stage 7's `blackbox_profile` cargo
   features). Until staff defines real log profiles, the placeholder
   IMU+attitude set is bench-only.
4. **CMD25 multi-block writes** (Stage 9 parking lot). Required
   for sustained 8 kHz IMU on commodity microSDs.

After those four, this is production-comparable with Betaflight's
blackbox. Points where it should *exceed* Betaflight: typed-Rust
encoders (no field-table drift), MCAP (Foxglove integration without
log-converter scripts), partition-aware FAT (cards stay
desktop-mountable).
