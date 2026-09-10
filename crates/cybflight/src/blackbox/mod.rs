//! Blackbox / flight-data-recorder subsystem.
//!
//! Single capture path:
//!
//! - **`recorder`** — opens `flight_<NNNN>.mcap` on the rising edge
//!   of [`should_record`], runs a multi-topic capture loop until the
//!   predicate falls, closes cleanly. Triggered by either the real
//!   arm path (`motors::IS_ARMED`, set via RC arm switch / failsafe)
//!   or the bench path (`RECORDER_HOLD`, set by `blackbox record on`
//!   in the shell).
//!
//! ## Architecture
//!
//! ```text
//!   shell `blackbox record on` ─┐
//!                               ├─▶ blackbox_task ─▶ recorder
//!   IS_ARMED edge ──────────────┘                          │
//!                                                          ▼
//!                                  fat::write_file<P: FileBody>
//!                                                          │
//!                                  embedded-fatfs / partitions
//!                                                          │
//!                                  SdmmcBlockStore::open_session
//!                                                          │
//!                                  embassy-stm32 SDMMC
//! ```
//!
//! - `mcap`, `cbor`, `fat`, `sdmmc_block` are the layered building blocks.
//! - `topics::*` holds per-topic schema + CBOR encoder.
//! - `record_set` is the small/mid/large/none tier classifier.
//! - Shell file ops (`blackbox ls/get/rm`) reach the card through
//!   request/result Signals serviced by `blackbox_task` between
//!   recording sessions; `get` streams file bytes back through
//!   [`GET_STREAM`].
//! - The whole subsystem is gated on `bsp::HAS_BLACKBOX_STORAGE` (a
//!   compile-time `const bool`) — no Cargo feature flag.

pub mod cbor;
pub mod fat;
pub mod lazy_stream;
pub mod mcap;
pub mod record_set;
pub mod recorder;
pub mod sdmmc_block;
pub mod topics;

pub use record_set::{current as record_set, set as set_record_set, RecordSet};

use core::sync::atomic::{AtomicBool, Ordering};

use embassy_futures::select::{select, select4, Either, Either4};
use core::cell::Cell;

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::blocking_mutex::Mutex;
use embassy_sync::channel::Channel;
use embassy_sync::signal::Signal;
use embassy_time::{with_timeout, Duration, Timer};

pub use fat::{CleanSummary, EntrySummary, MAX_FILENAME};
pub use sdmmc_block::SdmmcBlockStore;

/// Manual record-toggle for bench testing. When `true`, the recorder
/// behaves as if `IS_ARMED` were `true` — opens a `flight_NNNN.mcap`
/// file and logs until either this flag clears or `IS_ARMED` is also
/// false. Set by the `blackbox record on/off` shell command.
///
/// This bypass exists so the recorder can be exercised on the bench
/// without touching `ARM_STATE` (which would also spin DShot output).
pub static RECORDER_HOLD: AtomicBool = AtomicBool::new(false);

/// True if the recorder should be capturing right now — `IS_ARMED`
/// from real flight OR the manual hold flag.
#[inline]
pub fn should_record() -> bool {
    crate::motors::IS_ARMED.load(Ordering::Acquire) || RECORDER_HOLD.load(Ordering::Acquire)
}

// ─── Request / result signals ───────────────────────────────────────────

/// Raised by `blackbox ls`. Payload is unit; the request itself
/// carries no parameters (always lists the FAT root).
pub static LS_REQUEST: Signal<CriticalSectionRawMutex, ()> = Signal::new();
/// Outcome of one `blackbox ls` op.
pub static LS_RESULT: Signal<CriticalSectionRawMutex, LsReport> = Signal::new();

/// Maximum number of entries surfaced through one `LsReport::Ok`.
/// Beyond this we stop populating but keep counting, and set
/// `truncated = true` so the shell can warn the user.
pub const LS_MAX_ENTRIES: usize = 16;

/// Result of one `blackbox ls` op. `Busy` covers the recorder-active
/// case — concurrent SDMMC access from the shell while the recorder
/// owns the peripheral would race.
#[derive(Clone)]
pub enum LsReport {
    Ok {
        entries: heapless::Vec<EntrySummary, LS_MAX_ENTRIES>,
        /// True if the FAT root held more than `LS_MAX_ENTRIES`
        /// files; the surplus is silently dropped.
        truncated: bool,
        /// Total file count seen during the scan (not capped).
        total_files: u32,
    },
    Failed(fat::OpError),
    Busy,
}

/// Raised by `blackbox get`. Names a root-directory file plus the
/// byte window to stream (`len = None` means "to EOF").
pub struct GetRequest {
    pub name: fat::EntryName,
    pub offset: u32,
    pub len: Option<u32>,
}
pub static GET_REQUEST: Signal<CriticalSectionRawMutex, GetRequest> = Signal::new();

/// Header outcome of one `blackbox get` op — everything knowable
/// before the first data byte. `Ok` means the size header may be
/// sent and [`GET_STREAM`] chunks will follow; every other variant
/// terminates the op with no binary payload.
#[derive(Clone)]
pub enum GetReport {
    Ok { size: u32 },
    NotFound,
    Failed(fat::OpError),
    Busy,
}
pub static GET_RESULT: Signal<CriticalSectionRawMutex, GetReport> = Signal::new();

/// Chunk payload size for [`GET_STREAM`]. Matches the FAT read
/// granularity (one BufStream block per chunk).
pub const GET_CHUNK_LEN: usize = 512;

/// One unit of `blackbox get` file flow, task → shell.
pub enum GetChunk {
    Data { buf: [u8; GET_CHUNK_LEN], len: u16 },
    /// Clean end of stream: CRC-32 of every streamed byte.
    End { crc32: u32 },
    /// Mid-stream fault; the transfer is dead and the shell should
    /// print the reason in-band (the size header is already out).
    Err(fat::OpError),
}

/// File-content conduit for `blackbox get`. Depth 2 = double
/// buffering (the task reads block N+1 from the card while the shell
/// drains block N over USB); USB FS is the throughput bottleneck, so
/// deeper buys nothing. RAM cost ≈ 1.1 KB static.
pub static GET_STREAM: Channel<CriticalSectionRawMutex, GetChunk, 2> = Channel::new();

/// Bound on every `GET_STREAM.send` so a dead shell (host stopped
/// reading, USB unplugged) can never wedge `blackbox_task` — it also
/// services arm-edges and must return to its loop.
const GET_SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// Raised by `blackbox rm`. Payload is the root-directory file name.
pub static RM_REQUEST: Signal<CriticalSectionRawMutex, fat::EntryName> = Signal::new();

/// Outcome of one `blackbox rm` op.
#[derive(Clone)]
pub enum RmReport {
    Ok,
    NotFound,
    Failed(fat::OpError),
    Busy,
}
pub static RM_RESULT: Signal<CriticalSectionRawMutex, RmReport> = Signal::new();

/// Raised by `blackbox clean`: delete every `flight_NNNN.mcap` in the
/// FAT root (nothing else on the card is touched).
pub static CLEAN_REQUEST: Signal<CriticalSectionRawMutex, ()> = Signal::new();

/// Outcome of one `blackbox clean` op.
#[derive(Clone)]
pub enum CleanReport {
    Ok(CleanSummary),
    Failed(fat::OpError),
    Busy,
}
pub static CLEAN_RESULT: Signal<CriticalSectionRawMutex, CleanReport> = Signal::new();

/// Outcome of the most recent recording session this boot, for
/// `blackbox status` (Stage 7: the user can see what the last session
/// did without a defmt console). `None` until the first session
/// closes. Written by `blackbox_task` right after `run_session`
/// returns; read synchronously by the shell.
#[derive(Clone, Copy)]
pub struct LastSession {
    /// Sequence number NNNN of `flight_NNNN.mcap`.
    pub seq: u32,
    pub bytes: u32,
    pub messages: u32,
    pub drops: u32,
    pub encode_overflows: u32,
    /// `None` = clean close. `Some(PartialWrite)` = mid-session fault
    /// but a truncated log was salvaged under this name. Any other
    /// `Some(e)` = session failed; counters above are 0 (the summary
    /// died with the session).
    pub error: Option<fat::OpError>,
    /// Card-level I/O counters for the session (CMD25 batches, reads,
    /// fallbacks) — see [`sdmmc_block::IoStats`].
    pub io: sdmmc_block::IoStats,
}

pub static LAST_SESSION: Mutex<CriticalSectionRawMutex, Cell<Option<LastSession>>> =
    Mutex::new(Cell::new(None));

// ─── Arm-edge detection ─────────────────────────────────────────────────
//
// 20 ms poll cadence + 40 ms confirmation window matches the
// Betaflight blackbox debounce. Cheap, no extra channel needed:
// IS_ARMED is the established source of truth for the rest of the
// firmware.

const ARM_POLL_INTERVAL: Duration = Duration::from_millis(20);
const ARM_DEBOUNCE_POLLS: u8 = 2; // 2 × 20 ms = 40 ms

// ─── Task ───────────────────────────────────────────────────────────────

/// Owns the SDMMC peripheral for the duration of the firmware. Four
/// triggers wake it up:
///
/// 1. Arm-edge — runs `recorder::run_session` until disarm (or the
///    user clears `RECORDER_HOLD`).
/// 2. `LS_REQUEST` — scans the FAT root and returns an `LsReport`.
/// 3. `GET_REQUEST` — streams one file through `GET_STREAM`.
/// 4. `RM_REQUEST` — deletes one file.
/// 5. `CLEAN_REQUEST` — deletes every `flight_NNNN.mcap`.
///
/// All file ops are rejected with `Busy` if the recorder is
/// currently active; while a recording session runs the task is
/// blocked inside `run_session` and requests queue (then get dropped
/// as stale when the session ends).
#[embassy_executor::task]
pub async fn blackbox_task(mut store: SdmmcBlockStore) -> ! {
    defmt::info!("blackbox: ready");
    let mut prev_record = should_record();
    let mut debounce: u8 = 0;
    // Sequence-number cache: highest `flight_NNNN.mcap` index handed
    // out so far. `None` = "not yet probed; scan the FAT root before
    // first use". Filled lazily from disk so reboots pick up where
    // the previous session left off, then incremented in-memory.
    let mut flight_seq: Option<u32> = None;

    loop {
        match select4(
            Timer::after(ARM_POLL_INTERVAL),
            LS_REQUEST.wait(),
            GET_REQUEST.wait(),
            select(RM_REQUEST.wait(), CLEAN_REQUEST.wait()),
        )
        .await
        {
            Either4::First(()) => {
                let now_record = should_record();
                if now_record && !prev_record {
                    debounce = debounce.saturating_add(1);
                    if debounce >= ARM_DEBOUNCE_POLLS {
                        // Confirmed rising edge.
                        debounce = 0;
                        let rs = record_set::current();
                        if !rs.enabled() {
                            // Tier `none` — record-edge observed but recorder
                            // is muted. Don't open a file, don't bump the
                            // sequence counter, just resync edge state and
                            // wait for the next disarm/rearm cycle (or a
                            // tier change).
                            defmt::info!(
                                "blackbox: record-edge (record_set=none, session skipped)"
                            );
                        } else {
                            match next_seq(&mut store, &mut flight_seq).await {
                                Ok(seq) => {
                                    defmt::info!(
                                        "blackbox: record-edge → session {} (record_set={})",
                                        seq,
                                        rs.name(),
                                    );
                                    let outcome = recorder::run_session(&mut store, seq, rs).await;
                                    let io = sdmmc_block::LAST_IO_STATS.lock(|c| c.get());
                                    defmt::info!(
                                        "blackbox: card io: {} writes / {} blocks (max batch {}), \
                                         {} reads / {} blocks, {} CMD25 fallbacks",
                                        io.writes,
                                        io.blocks_written,
                                        io.max_batch,
                                        io.reads,
                                        io.blocks_read,
                                        io.fallbacks,
                                    );
                                    match outcome {
                                        Ok(s) => {
                                            defmt::info!(
                                                "blackbox: closed /{} ({} bytes, {} msgs, \
                                                 {} drops, {} enc-overflows)",
                                                s.file_name.as_str(),
                                                s.bytes,
                                                s.messages,
                                                s.drops,
                                                s.encode_overflows,
                                            );
                                            LAST_SESSION.lock(|c| {
                                                c.set(Some(LastSession {
                                                    seq,
                                                    bytes: s.bytes,
                                                    messages: s.messages,
                                                    drops: s.drops,
                                                    encode_overflows: s.encode_overflows,
                                                    error: None,
                                                    io,
                                                }))
                                            });
                                        }
                                        Err(fat::OpError::PartialWrite) => {
                                            // A truncated log *was* committed
                                            // under this number. Keep the seq
                                            // advanced: recycling it would make
                                            // the next arm `create_file` the same
                                            // name and `truncate()` the salvage
                                            // away — turning a partial log into
                                            // no log at all.
                                            defmt::warn!(
                                                "blackbox: session {} faulted; \
                                                 partial log kept (seq not recycled)",
                                                seq,
                                            );
                                            LAST_SESSION.lock(|c| {
                                                c.set(Some(LastSession {
                                                    seq,
                                                    bytes: 0,
                                                    messages: 0,
                                                    drops: 0,
                                                    encode_overflows: 0,
                                                    error: Some(fat::OpError::PartialWrite),
                                                    io,
                                                }))
                                            });
                                        }
                                        Err(e) => {
                                            defmt::warn!(
                                                "blackbox: session failed: {:?}",
                                                e,
                                            );
                                            LAST_SESSION.lock(|c| {
                                                c.set(Some(LastSession {
                                                    seq,
                                                    bytes: 0,
                                                    messages: 0,
                                                    drops: 0,
                                                    encode_overflows: 0,
                                                    error: Some(e),
                                                    io,
                                                }))
                                            });
                                            // Roll back the cached seq so we retry
                                            // this number next time. If the file
                                            // did make it onto the card despite the
                                            // error, the next-rising-edge scan will
                                            // see it and skip past.
                                            if let Some(s) = flight_seq.as_mut() {
                                                *s = s.saturating_sub(1);
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    defmt::warn!("blackbox: seq scan failed: {:?}", e)
                                }
                            }
                        }
                        // Resync edge state with whatever's true after the session.
                        prev_record = should_record();
                        // Drain any file-op request queued during the
                        // session — stale by definition.
                        let _ = LS_REQUEST.try_take();
                        let _ = GET_REQUEST.try_take();
                        let _ = RM_REQUEST.try_take();
                        let _ = CLEAN_REQUEST.try_take();
                    }
                } else {
                    debounce = 0;
                    prev_record = now_record;
                }
            }
            Either4::Second(()) => {
                if should_record() {
                    defmt::warn!("blackbox: ls rejected (recorder active)");
                    LS_RESULT.signal(LsReport::Busy);
                    continue;
                }
                LS_RESULT.signal(run_ls(&mut store).await);
            }
            Either4::Third(req) => {
                if should_record() {
                    defmt::warn!("blackbox: get rejected (recorder active)");
                    GET_RESULT.signal(GetReport::Busy);
                    continue;
                }
                run_get(&mut store, &req).await;
            }
            Either4::Fourth(Either::First(name)) => {
                if should_record() {
                    defmt::warn!("blackbox: rm rejected (recorder active)");
                    RM_RESULT.signal(RmReport::Busy);
                    continue;
                }
                RM_RESULT.signal(run_rm(&mut store, name.as_str()).await);
            }
            Either4::Fourth(Either::Second(())) => {
                if should_record() {
                    defmt::warn!("blackbox: clean rejected (recorder active)");
                    CLEAN_RESULT.signal(CleanReport::Busy);
                    continue;
                }
                let report = match fat::remove_flight_logs(&mut store).await {
                    Ok(s) => CleanReport::Ok(s),
                    Err(e) => CleanReport::Failed(e),
                };
                // The card no longer holds the files the cached
                // sequence was derived from: forget it so the next
                // session rescans and numbering restarts at 0001
                // (or after whatever survived an aborted clean).
                if matches!(report, CleanReport::Ok(_)) {
                    flight_seq = None;
                }
                CLEAN_RESULT.signal(report);
            }
        }
    }
}

/// Get the next `flight_NNNN.mcap` sequence number. First call after
/// boot scans the FAT root once and caches the highest existing NNNN;
/// subsequent calls just increment the cache, so steady-state arming
/// never re-mounts for sequencing.
///
/// The cache is populated **after** a successful scan, so a boot
/// with no card / unmounted card defers the scan until the card is
/// inserted instead of poisoning the cache with 0.
async fn next_seq(
    store: &mut SdmmcBlockStore,
    cache: &mut Option<u32>,
) -> Result<u32, fat::OpError> {
    let highest = match *cache {
        Some(h) => h,
        None => fat::scan_highest_seq(store, "flight_", ".mcap").await?,
    };
    let next = highest.saturating_add(1);
    *cache = Some(next);
    Ok(next)
}

/// Scan the FAT root and assemble an `LsReport`. Caps the per-entry
/// list at `LS_MAX_ENTRIES`; counts every file regardless so the
/// shell can show "showing N of M".
async fn run_ls(store: &mut SdmmcBlockStore) -> LsReport {
    let mut entries: heapless::Vec<EntrySummary, LS_MAX_ENTRIES> = heapless::Vec::new();
    let mut total: u32 = 0;
    let mut truncated = false;
    let mut visit = |name: &str, size: u32| {
        total = total.saturating_add(1);
        if entries.is_full() {
            truncated = true;
            return;
        }
        let mut n: fat::EntryName = heapless::String::new();
        // push char-by-char so an over-long name is silently truncated
        // rather than dropped entirely.
        for c in name.chars() {
            if n.push(c).is_err() {
                break;
            }
        }
        // is_full() above guarantees this push won't fail.
        let _ = entries.push(EntrySummary { name: n, size });
    };
    match fat::scan_root_files(store, &mut visit).await {
        Ok(()) => LsReport::Ok {
            entries,
            truncated,
            total_files: total,
        },
        Err(e) => LsReport::Failed(e),
    }
}

/// [`fat::FileSink`] that forwards a file to the shell through
/// [`GET_RESULT`] (size header) and [`GET_STREAM`] (content), CRC-ing
/// every byte on the way past.
struct ShellGetSink {
    crc: crc32fast::Hasher,
    /// True once `begin` has fired — from then on errors must go
    /// in-band via `GetChunk::Err`, not `GET_RESULT`.
    started: bool,
}

impl fat::FileSink for ShellGetSink {
    async fn begin(&mut self, total: u32) -> Result<(), fat::SinkAbort> {
        self.started = true;
        GET_RESULT.signal(GetReport::Ok { size: total });
        Ok(())
    }

    async fn data(&mut self, chunk: &[u8]) -> Result<(), fat::SinkAbort> {
        // A real arm preempts a download. The main loop can't service
        // the arm-edge until this op returns, so poll here — bounds
        // the abort latency to one chunk. (`RECORDER_HOLD` can't flip
        // mid-op: the shell that would set it is busy running us.)
        if crate::motors::IS_ARMED.load(Ordering::Acquire) {
            defmt::warn!("blackbox: get aborted (armed mid-transfer)");
            return Err(fat::SinkAbort);
        }
        self.crc.update(chunk);
        let mut buf = [0u8; GET_CHUNK_LEN];
        let len = chunk.len().min(GET_CHUNK_LEN);
        buf[..len].copy_from_slice(&chunk[..len]);
        with_timeout(
            GET_SEND_TIMEOUT,
            GET_STREAM.send(GetChunk::Data {
                buf,
                len: len as u16,
            }),
        )
        .await
        .map_err(|_| fat::SinkAbort)
    }
}

/// Service one `blackbox get`: stream the requested window through
/// `GET_STREAM`, terminated by `End { crc32 }` on success or an
/// in-band `Err` chunk if the fault happened after the size header
/// went out.
async fn run_get(store: &mut SdmmcBlockStore, req: &GetRequest) {
    // Chunks abandoned by a previous aborted transfer would corrupt
    // this one's framing.
    while GET_STREAM.try_receive().is_ok() {}
    let mut sink = ShellGetSink {
        crc: crc32fast::Hasher::new(),
        started: false,
    };
    match fat::read_file(store, req.name.as_str(), req.offset, req.len, &mut sink).await {
        Ok(_) => {
            let crc32 = sink.crc.finalize();
            let _ = with_timeout(GET_SEND_TIMEOUT, GET_STREAM.send(GetChunk::End { crc32 })).await;
        }
        Err(e) if sink.started => {
            // Header already out — the error must travel in-band.
            // Best-effort: if the channel is full the shell side's
            // chunk timeout reports "task stalled" instead.
            let _ = GET_STREAM.try_send(GetChunk::Err(e));
        }
        Err(fat::OpError::NotFound) => GET_RESULT.signal(GetReport::NotFound),
        Err(e) => GET_RESULT.signal(GetReport::Failed(e)),
    }
}

/// Service one `blackbox rm`. Deleting the highest-numbered flight
/// file is safe w.r.t. `next_seq` — the cached sequence only ever
/// increments, so the name is never reused within this boot.
async fn run_rm(store: &mut SdmmcBlockStore, path: &str) -> RmReport {
    match fat::remove_file(store, path).await {
        Ok(()) => RmReport::Ok,
        Err(fat::OpError::NotFound) => RmReport::NotFound,
        Err(e) => RmReport::Failed(e),
    }
}
