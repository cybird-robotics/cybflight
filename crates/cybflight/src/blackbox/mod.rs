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
//! - The whole subsystem is gated on `bsp::HAS_BLACKBOX_STORAGE` (a
//!   compile-time `const bool`) — no Cargo feature flag.

pub mod cbor;
pub mod fat;
pub mod mcap;
pub mod record_set;
pub mod recorder;
pub mod sdmmc_block;
pub mod topics;

pub use record_set::{current as record_set, set as set_record_set, RecordSet};

use core::sync::atomic::{AtomicBool, Ordering};

use embassy_futures::select::{select, Either};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Timer};

pub use fat::{EntrySummary, MAX_FILENAME};
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

// ─── Arm-edge detection ─────────────────────────────────────────────────
//
// 20 ms poll cadence + 40 ms confirmation window matches the
// Betaflight blackbox debounce. Cheap, no extra channel needed:
// IS_ARMED is the established source of truth for the rest of the
// firmware.

const ARM_POLL_INTERVAL: Duration = Duration::from_millis(20);
const ARM_DEBOUNCE_POLLS: u8 = 2; // 2 × 20 ms = 40 ms

// ─── Task ───────────────────────────────────────────────────────────────

/// Owns the SDMMC peripheral for the duration of the firmware. Two
/// triggers wake it up:
///
/// 1. Arm-edge — runs `recorder::run_session` until disarm (or the
///    user clears `RECORDER_HOLD`).
/// 2. `LS_REQUEST` — scans the FAT root and returns an `LsReport`,
///    rejected with `Busy` if the recorder is currently active.
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
        match select(Timer::after(ARM_POLL_INTERVAL), LS_REQUEST.wait()).await {
            Either::First(()) => {
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
                                    match recorder::run_session(&mut store, seq, rs).await {
                                        Ok(s) => defmt::info!(
                                            "blackbox: closed /{} ({} bytes, {} msgs, {} drops)",
                                            s.file_name.as_str(),
                                            s.bytes,
                                            s.messages,
                                            s.drops,
                                        ),
                                        Err(e) => {
                                            defmt::warn!(
                                                "blackbox: session failed: {:?}",
                                                e,
                                            );
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
                        // Drain any LS request queued during the session
                        // — stale by definition.
                        let _ = LS_REQUEST.try_take();
                    }
                } else {
                    debounce = 0;
                    prev_record = now_record;
                }
            }
            Either::Second(()) => {
                if should_record() {
                    defmt::warn!("blackbox: ls rejected (recorder active)");
                    LS_RESULT.signal(LsReport::Busy);
                    continue;
                }
                LS_RESULT.signal(run_ls(&mut store).await);
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
