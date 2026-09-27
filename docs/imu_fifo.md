# IMU FIFO: motivation and plan

Historical experiment notes; hardware support and current build instructions are in [hardware-support.md](hardware-support.md) and the repository README.

Status: **proposal** (2026-08-24). Nothing here is implemented. The
per-sample data-ready reader described in [imu_filtering.md](imu_filtering.md)
is what flies today; this document records why the ICM FIFO is the
right end state for the IMU path and how to get there without a
flag-day change.

Register and packet facts below are from the TDK InvenSense ICM-42688-P
datasheet (DS-000347 rev 1.6,
[product page](https://invensense.tdk.com/products/motion-tracking/6-axis/icm-42688-p/));
section numbers refer to rev 1.6 — the product page serves the newest revision
(1.9 as of 2026-09), whose numbering may differ.

## 1. Where we are

The IMU path went through three states on 2026-08-24, all measured with
the `imurate` / `indistat` shell verbs (see [HACKING.md](HACKING.md#bench-rate-checks-imurate-indistat)):

| reader configuration | `imurate` | note |
|---|---|---|
| 1 MHz SPI, 2 transactions/sample, pulsed DRDY (since bring-up) | **2,565 Hz** | 17 B @ 1 MHz = 136 µs of bus time per 125 µs period |
| 8 MHz SPI, 2 transactions, pulsed DRDY, INDI ÷4 | 6,336 Hz | one sample lost per INDI step (reader still inside its read when the 8 µs pulse arrives) |
| 8 MHz SPI, 1 transaction, **latched** DRDY, INDI ÷4 | **7,918 Hz**, lost ≈ 0 | |
| 24 MHz SPI, 1 transaction, latched DRDY, INDI ÷4 | re-bench with `imurate` | current tree (24 MHz boot-looped only with the old two-transaction driver) |

So the sensor side now delivers what the chip produces. It does so by
managing a structural weakness rather than removing it: the consumer of
the data-ready line is an **async task** on the P10 executor, which can
be late by whatever else is running there (INDI's step is 105 µs avg /
126 µs max — `indistat`). Latched INT1 turns that lateness into latency
instead of loss, and `cybflight_core::imu_stamp::SampleStamper` puts the
timestamps back on the ODR grid and counts the samples that really went
missing. It works, and it costs the P10 executor ~28 % of its time
(8,000 wakes/s × ~35 µs).

Three things remain unsatisfying about it:

1. **Register coherency is unspecified.** A late reader's 15-byte burst
   can start at any phase of the 125 µs period. §4.11 says only that the
   sensor data registers "may be read anytime"; it does not say they are
   held stable during a burst. A read that straddles the ODR update can
   mix sample n and n+1, worst case tearing one 16-bit word. There is a
   `TODO(verify)` on this in `icm426xx.rs`. Betaflight never hits it
   because its ISR starts the read µs after the edge; we can.
2. **Loss is inferred, not known.** The stamper's lost-sample rule is a
   timing heuristic (two consecutive reads ≥ T/2 behind). It is right in
   the cases it was tested for, and the 32.768 kHz embassy timebase
   (30.5 µs ticks) leaves it less margin than the µs arithmetic suggests.
3. **8,000 wakes a second for a 2 kHz consumer.** INDI steps on every
   4th sample (`indi_ctrl_div`); the ESKF predicts at ~1 kHz; the other
   samples exist for filter margin and the sysid log. Waking a task per
   sample to feed a consumer that wants every 4th is the expensive way to
   get there, and it is the reason the P10 budget is a thing we have to
   measure.

## 2. What the FIFO gives

The ICM-42688-P (and the IIM-42652, same register map) has a 2 kB
hardware FIFO (§6) that the sensor writes at ODR, independent of when
the host reads.

- **Atomic packets.** "A number of bytes equal to the packet size
  selected is reserved to prevent reading a packet during write
  operation" (§6.3). A packet is either fully in the FIFO or not; there
  is no straddling read. Finding 1 above disappears by construction.
- **Exact loss accounting.** `FIFO_COUNT` (0x2E/0x2F) says how many
  packets are waiting; the FIFO-full interrupt / `INT_STATUS.FIFO_FULL`
  says if any were dropped. Loss becomes a hardware fact, not a timing
  inference. With 16-byte packets the FIFO holds 128 samples = **16 ms
  at 8 kHz** before anything is lost — two orders of magnitude more
  tolerance than the reader needs.
- **Per-sample hardware timestamps.** Packet 3 carries a 16-bit ODR
  timestamp (§6.1, `HEADER_TIMESTAMP_FSYNC = 10`) from the chip's own
  counter, 1 µs resolution (`TMST_CONFIG.TMST_RES = 0`, §14.42). The
  timestamps come from the clock that actually sampled the data. The
  stamper's job reduces to mapping the chip clock onto the MCU timebase
  once per batch instead of guessing each sample's time from a wake.
- **One wake per batch.** Set `FIFO_WM = 4` records and route
  `FIFO_THS` to INT1: the host wakes at the control rate (2 kHz), reads
  `FIFO_COUNT`, then drains N packets in one burst. Per-sample host cost
  drops ~4× and the reader's executor share goes from ~28 % to ~8 %.

What it does **not** give: the freshest possible sample at the lowest
possible latency. Betaflight rejects the FIFO for exactly that reason —
a racing PID loop wants the gyro sample that is 20 µs old, not 145 µs.
That trade is the wrong one for this project (§4).

## 3. Design

### 3.1 Chip configuration (bank 0)

| register | value | why |
|---|---|---|
| `FIFO_CONFIG` (0x16) | `FIFO_MODE = 01` stream-to-FIFO | overwrite oldest on overflow; STOP-on-FULL would freeze the stream on a host stall |
| `FIFO_CONFIG1` (0x5F) | `FIFO_ACCEL_EN \| FIFO_GYRO_EN \| FIFO_TEMP_EN \| FIFO_WM_GT_TH`; `FIFO_HIRES_EN = 0` | packet 3 (16 B: header, accel, gyro, temp, timestamp). `FIFO_WM_GT_TH` re-fires the watermark interrupt on every ODR while `FIFO_COUNT ≥ WM`, so a late host is re-armed rather than stranded (§14.45) |
| `FIFO_CONFIG2/3` (0x60/0x61) | `FIFO_WM = 4` | one batch per control tick at `indi_ctrl_div = 4`; must be non-zero (§14.47 note) |
| `INTF_CONFIG0` (0x4C) | `FIFO_COUNT_REC = 1`, `FIFO_HOLD_LAST_DATA_EN = 0` | count in records not bytes; invalid samples marked (-32768) rather than repeated (§14.34) |
| `TMST_CONFIG` (0x54) | `TMST_EN = 1`, `TMST_RES = 0`, `TMST_DELTA_EN = 0` | absolute 1 µs timestamp in each packet |
| `INT_SOURCE0` (0x65) | `FIFO_THS_INT1_EN` (bit 2) instead of `UI_DRDY_INT1_EN` (bit 3) | wake on watermark, not on every sample |
| `INT_CONFIG0` (0x63) | `FIFO_THS_INT_CLEAR = 10` (clear on FIFO data read) | keep INT1 latched (`INT_CONFIG.INT1_MODE = 1`, as today) so a late host still sees it; the drain itself clears it |
| `SIGNAL_PATH_RESET` (0x4B) | `FIFO_FLUSH` once after configuration | start from an empty FIFO |

`FIFO_COUNT_REC = 1` matters: with 16-byte packets the byte count would
need dividing anyway, and the record count is what the host loops on.
Reading `FIFO_COUNTH` latches both count bytes (§14.22) — always read
0x2E then 0x2F in one 2-byte burst.

### 3.2 Host read sequence (per wake)

```
wait_for_high(INT1)                       // latched FIFO_THS
count = burst_read(FIFO_COUNTH, 2)        // records waiting, latched pair
n     = min(count, MAX_BATCH)             // MAX_BATCH = 8 (2× WM: absorbs one late wake)
buf   = burst_read(FIFO_DATA, 16·n)       // one transaction, 0x30 auto-repeats
for pkt in buf.chunks(16):
    header = pkt[0]
    reject if HEADER_MSG (bit 7) or !(HEADER_ACCEL && HEADER_GYRO)
    accel = i16 BE ×3, gyro = i16 BE ×3, temp = i8, tmst = u16 BE
    skip sample if any field == -32768 (invalid marker, INTF_CONFIG0)
    t_mcu = stamper.map(tmst)             // see 3.3
    publish raw / filtered as today
if count > MAX_BATCH:   IMU_LOST += count - MAX_BATCH   (drain the rest next wake; the FIFO holds them)
if INT_STATUS.FIFO_FULL: IMU_LOST += unknown (report as overflow event, not a count)
```

Bus cost at 8 MHz: 2 + 2 + 65 bytes ≈ 70 µs of bus time per 2 kHz wake
(vs. 4 × 15 µs = 60 µs today) — the same bytes, but **one DMA setup, one
completion IRQ and one executor wake instead of four**. That is where the
CPU saving comes from. At the 1 kHz vehicle (`imu_1khz`, `indi_ctrl_div
= 1`) `FIFO_WM = 1` and the sequence degenerates to today's behaviour
with atomic packets and hardware timestamps — the FIFO is not an
8 kHz-only feature.

`MAX_BATCH = 8` is a deliberate cap, not the FIFO depth: it bounds the
burst (128 B ≈ 130 µs at 8 MHz) and the per-wake publish work so one
wake cannot itself become a P10 stall. A stall longer than 8 samples is
drained over the following wakes; the FIFO's 128-sample depth is what
makes that safe.

### 3.3 Timestamps

Packet timestamps are 16-bit at 1 µs → wrap every 65.5 ms (524 samples
at 8 kHz), which is far longer than any batch, so unwrapping across
consecutive packets is trivial (`Δ = (t − prev) mod 65536`). The chip's
counter runs on the chip's crystal — the same ±1 % as the ODR — so
consecutive packets are exactly 125 µs apart in chip time by
definition, not by reconstruction.

Mapping chip time onto the MCU `Instant` timebase reuses the existing
`SampleStamper` idea one level up: the observation is the wake time of
the *batch* and the model is "the newest packet in this batch was
sampled ≈ (read latency) before the wake; earlier packets are
`k × 125 µs` earlier in chip time". The PLL tracks the batch-level
offset (crystal-vs-MCU drift, 1.25 µs per sample at 1 %); within a
batch the spacing is the chip's, so no per-sample jitter is ever
introduced. Stamps remain monotonic and never later than the wake,
which keeps every downstream `saturating_duration_since` and `dt ≤ 0`
guard exactly as safe as the safety review found them (2026-08-24).

The 32.768 kHz embassy tick quantises the *published* `Instant` to
30.5 µs regardless of source. If the sysid log needs better than that,
carry the raw chip timestamp (u16) as an extra field in `msgs::Imu`
and write it to the blackbox; consumers that want µs spacing then have
it without touching the MCU timebase. Out of scope for the first step.

### 3.4 Interaction with consumers

- **INDI** subscribes to `IMU_1` and steps on every `indi_ctrl_div`-th
  message. With batches of 4 published back-to-back, that is exactly
  once per batch, on the newest sample — the decimation logic is
  unchanged and INDI's input latency is `read time + wake latency`
  (same order as today: ~50–150 µs), not a full batch.
- **ESKF / Mahony / recorder / ESP bridge** see the same message stream
  at the same rate; back-to-back publishing of 4 messages is within
  every subscriber's CAP (`IMU_PUBSUB_CAP = 192` at 8 kHz; mirrors 16).
- **Blackbox sysid tier** finally gets a genuinely complete 8 kHz raw
  stream with exact per-sample chip timestamps — the reason the tier
  exists. It still has to fit the card (~1.4 MB/s for two IMU topics vs
  ~400 KB/s sustained; see [blackbox.md](blackbox.md) §3), which is a
  recorder-side problem the FIFO does not change.
- **`imurate`** keeps working unchanged (it counts publishes). Its
  `lost` column becomes the hardware figure (`count − MAX_BATCH`
  carry-overs and `FIFO_FULL` events) instead of the timing heuristic.

### 3.5 What is retired

- `SampleStamper::stamp()` per-sample path → batch-level offset PLL
  (same module; the tests carry over with a batch input).
- The `TODO(verify)` on register coherency in `icm426xx.rs` — moot.
- Latched UI DRDY (`UI_DRDY_INT1_EN`) — replaced by the FIFO watermark
  interrupt; still latched, still level-waited.

## 4. Why this is the right trade for cybflight

| stack | IMU read model | reason |
|---|---|---|
| Betaflight / indiflight | per-sample, EXTI ISR kicks the DMA, pulsed DRDY | 8 kHz PID racer; every 100 µs of gyro latency is felt |
| PX4 (`icm42688p`) | FIFO drained by a data-ready/timer at 1–2 kHz, batches of samples | lossless stream for the estimator, bounded CPU |
| ArduPilot | FIFO polled at 1 kHz, 8 samples per read, backend filtering | same |
| cybflight today | per-sample, async task, latched DRDY | Betaflight's model without Betaflight's ISR |
| cybflight proposed | FIFO at the control rate | PX4/ArduPilot's model, which matches what this project asks of the IMU |

What this project asks of the IMU: a **complete** 8 kHz stream for
system identification and filter design, and a **fresh** 2 kHz sample
for an INDI loop whose synchronising filter (`indi_sync_hz`, 12–15 Hz)
already has ~10–13 ms of group delay. An extra 100 µs of sample age on
the control path is 1 % of that; a torn or silently dropped sample in
the sysid log is a fitting artefact we cannot remove afterwards. The
FIFO optimises for the thing we actually need.

## 5. Plan

Each step is independently flashable and measurable; none changes the
`msgs::Imu` wire format until step 4.

1. **Driver: FIFO mode behind a constructor option.**
   `Icm426xx::new(.., ReadMode::Fifo { watermark })` next to the
   existing per-sample mode; new `read_batch(&mut self, out: &mut
   [FifoPacket; MAX_BATCH]) -> Result<usize>` implementing §3.2. Keep
   `read()` for the per-sample mode so the BMI270/MPU6000 `ReadImu`
   trait is untouched. `cargo check` on all boards; no vehicle uses it
   yet.
2. **Reader: batch path in `ImuReader::run`.** Publish each packet as
   today (raw then filtered), stamp via the batch-level PLL, count
   `count − MAX_BATCH` carry-overs and `FIFO_FULL` into
   `IMU1_LOST_SAMPLES`. Gate with a `const IMU_FIFO: bool` in
   `board_init/sakurah743.rs` (same pattern as `ENABLE_*`), default
   `false`.
3. **Bench, `sakura_bench_hunter_8khz`, `IMU_FIFO = true`:**
   `imurate` ≈ 8,000 Hz with `lost = 0` over minutes, including with
   motors spinning; `indistat` step avg unchanged (~105 µs) and P10 idle
   share up; one 60 s sysid log with `read_mcap.py --summary` showing
   `/imu1` seq-gaps = 0 and a flat 125 µs dt histogram; `imu1`/`imu2`
   one-shot values sane; 1 kHz vehicle regression (`FIFO_WM = 1`).
4. **Optional: raw chip timestamp in `msgs::Imu` + blackbox schema
   bump** for µs-exact sysid spacing (§3.3). Separate change; touches
   `cybflight-msgs`, `blackbox_wire`, `read_mcap.py`.
5. **Flip the default** (`IMU_FIFO = true`), retire the per-sample
   ICM path and the per-sample stamper, update
   [imu_filtering.md](imu_filtering.md) (its "no decimation anywhere"
   paragraph is already out of date since `indi_ctrl_div`) and
   [architecture.md](architecture.md).

## 6. Open questions

- **IIM-42652 (IMU2)** shares the register map; confirm the FIFO
  timestamp feature bits are identical before enabling it there
  (it is disabled by `ENABLE_IMU2` today, so not blocking).
- **Invalid-sample policy.** `FIFO_HOLD_LAST_DATA_EN = 0` marks invalid
  samples with -32768; the reader should skip them and count them
  separately from lost samples. Alternative: hold-last (`= 1`) and
  never see them — simpler, but hides a fault.
- **Watermark vs. timer.** The watermark interrupt is the natural wake.
  If INT1 ever proves unreliable, a 2 kHz `Ticker` polling
  `FIFO_COUNT` is the PX4 fallback and needs no interrupt at all; the
  FIFO depth makes both equally lossless.
- **Batch size at other control rates.** `FIFO_WM` should follow
  `indi_ctrl_div` (a runtime param today), so the driver needs the
  watermark at construction, i.e. `board_init` reads the param before
  building the IMU. That ordering already exists for the LPF cutoffs.
