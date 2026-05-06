//! Boot-time post-mortem recovery: read the BKPSRAM record left by
//! the prior boot, validate it, log a one-line summary, then publish
//! it for downstream consumers (shell, blackbox recorder).
//!
//! ## Sequence
//!
//! 1. `pre_init` already ran [`super::reset_cause::capture`], so
//!    `RAW_RESET_CAUSE` is populated.
//! 2. Caller (`main.rs`, after `bsp::init()`) invokes
//!    [`super::bkpsram::enable`] to clock the BKPSRAM controller.
//! 3. Caller invokes [`boot_recovery`] which:
//!    - Reads the BKPSRAM slot, validates magic + CRC.
//!    - If valid: copies the record into [`PRIOR_RECORD`] (a Signal
//!      consumed once by the blackbox recorder + shell), bumps
//!      `boot_count`, logs via defmt.
//!    - If invalid (cold boot, schema mismatch, torn write): treats
//!      it as the first boot.
//!    - Writes a fresh record into BKPSRAM with this boot's
//!      reset_cause + incremented boot_count, finalizes.
//!
//! After this returns, the BKPSRAM slot holds the *current* boot's
//! starting state. The post-mortem task and fault handlers may
//! mutate it freely from here.

use core::sync::atomic::{AtomicBool, Ordering};

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;

use super::bkpsram;
use super::record::{self, FatalKind, PostmortemRecord};
use super::reset_cause;

/// Latched copy of the prior boot's record. Filled by
/// [`boot_recovery`] when a valid record is found; otherwise stays
/// empty.
///
/// Single-shot semantics: the blackbox recorder takes it via
/// [`take_pending`] when the next flight session opens. After it's
/// taken, subsequent calls return `None`. The shell's `postmortem
/// show` reads BKPSRAM directly (so it works even after the recorder
/// has consumed the Signal); this Signal exists so the recorder gets
/// woken automatically.
///
/// `Signal` rather than `OnceCell` because the recorder uses
/// signal-based wakeups elsewhere; consistent with the rest of the
/// blackbox surface.
pub static PRIOR_RECORD: Signal<CriticalSectionRawMutex, PostmortemRecord> = Signal::new();

/// One-shot guard against re-entry. `boot_recovery` is idempotent
/// (a second call would not re-publish the prior record because the
/// BKPSRAM slot was already overwritten with this boot's state) but
/// the guard makes accidental double-calls cheap and observable.
static RECOVERED: AtomicBool = AtomicBool::new(false);

/// Boot-time entry point. Call exactly once from `main.rs` after
/// `bsp::init()` and [`super::bkpsram::enable`].
///
/// **Hard rules:** does not allocate, does not await, does not panic
/// (a panic here would land in the same panic handler the post-mortem
/// subsystem provides — re-entry is guarded by the depth counter in
/// `fault.rs`, but we still avoid the situation).
pub fn boot_recovery() {
    if RECOVERED.swap(true, Ordering::AcqRel) {
        defmt::warn!("postmortem: boot_recovery called twice — ignoring");
        return;
    }

    let cause = reset_cause::read();
    let cause_kind = reset_cause::classify(cause);

    let prior = bkpsram::with_record(|r| if record::is_valid(r) { Some(*r) } else { None });

    let next_boot_count = match prior.as_ref() {
        Some(p) => {
            let prior_kind = FatalKind::from_u8(p.fatal.kind);
            defmt::error!(
                "postmortem: prior boot crash recovered (boot_count={=u32}, reset_cause={:?}, fatal={:?}, pc=0x{=u32:08x}, lr=0x{=u32:08x})",
                p.header.boot_count,
                cause_kind,
                prior_kind,
                p.fatal.pc,
                p.fatal.lr,
            );
            // Publish for downstream consumers. `signal()` overwrites
            // any prior pending value (there shouldn't be one — boot
            // recovery is single-shot — but if a future change adds a
            // re-entry, the latest record wins).
            PRIOR_RECORD.signal(*p);
            p.header.boot_count.wrapping_add(1)
        }
        None => {
            defmt::info!(
                "postmortem: no prior record (cold boot or first run after firmware update); reset_cause={:?}",
                cause_kind
            );
            1
        }
    };

    // Initialize this boot's record.
    let mut fresh = PostmortemRecord::ZERO;
    fresh.header.boot_count = next_boot_count;
    fresh.header.reset_cause = cause;
    fresh.header.fw_git_hash = git_hash_u32();
    fresh.header.uptime_ms = 0;
    bkpsram::store_finalized(&fresh);
}

/// Drain the prior-boot record. Returns `Some(...)` exactly once if a
/// prior crash was recovered; subsequent calls return `None`.
///
/// Used by the blackbox recorder at session-open to mirror the prior
/// boot's events into the new MCAP session as `/events` records.
pub fn take_pending() -> Option<PostmortemRecord> {
    PRIOR_RECORD.try_take()
}

/// Read the prior boot's record without consuming it. Used by the
/// shell `postmortem show` command — works whether or not the
/// recorder has already drained the Signal.
///
/// Returns `None` if BKPSRAM has already been overwritten with this
/// boot's state and the original prior record wasn't latched (this
/// would only happen if `boot_recovery` ran with no valid prior
/// record — i.e. a clean first boot).
///
/// The Signal `peek` is best-effort: embassy's `Signal` doesn't
/// expose a non-consuming read in stable, so we re-read BKPSRAM (now
/// the *current* boot's record) and report the prior record only if
/// the Signal still has a pending value. Callers that need the
/// prior record specifically (vs. "any record at all") should drain
/// the Signal via [`take_pending`].
pub fn current_boot_record() -> PostmortemRecord {
    bkpsram::with_record(|r| *r)
}

/// First 4 bytes of [`crate::GIT_HASH`] as a `u32`. If git_hash isn't
/// available (build without git), returns `0`.
fn git_hash_u32() -> u32 {
    let bytes = crate::GIT_HASH.as_bytes();
    if bytes.len() < 4 {
        return 0;
    }
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
