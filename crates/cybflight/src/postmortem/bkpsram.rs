//! D3-domain backup SRAM access for the post-mortem subsystem.
//!
//! ## What lives here
//!
//! - `enable()`: turns on the BKPSRAM clock + opens the backup-domain
//!   write protection so writes to the slot succeed.
//! - `with_record()` / `with_record_mut()`: safe wrappers around the
//!   raw-pointer access to the `POSTMORTEM_SLOT` static. The static
//!   itself lives in the `.bkpsram` linker section (declared in
//!   `memory.x`); the section is `(NOLOAD)`, so no startup code zeroes
//!   it — that's the whole point: BKPSRAM contents survive across
//!   reboots so the post-mortem record from the *prior* boot is
//!   readable on the *next* boot.
//!
//! ## Concurrency / atomicity
//!
//! BKPSRAM is byte-addressable via a normal AHB4 access. There is no
//! erase cycle, no page boundary, no flash-style commit phase — a
//! `*mut u32` write is observable to the next reader as soon as the
//! AHB write posts. The post-mortem record's atomicity comes entirely
//! from the magic-last + CRC-over-body protocol implemented in
//! [`crate::postmortem::record`], not from any property of BKPSRAM
//! itself.
//!
//! ## D3 domain caveat
//!
//! BKPSRAM lives in the D3 domain. If the firmware ever puts D3 into
//! DSTOP / DSTANDBY for power saving, BKPSRAM access will hang the
//! AHB4 bus until the domain wakes. **This subsystem assumes D3 stays
//! in Run mode.** embassy-stm32's default `Config::default()` does not
//! enter D3 stop modes; if a future change adds explicit low-power
//! sleep, audit this module first.
//!
//! ## Hard rule for fault paths
//!
//! `with_record_mut` is callable from a `#[panic_handler]`, HardFault
//! exception, or PVD IRQ — it does no allocation, takes no locks, and
//! never awaits. The closure passed to it must respect the same
//! constraints.

use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::hal::pac;

use super::record::PostmortemRecord;

/// The post-mortem record itself, pinned at a fixed address in the
/// BKPSRAM region declared in `memory.x`.
///
/// `MaybeUninit` is load-bearing: the linker section is `(NOLOAD)`, so
/// the runtime never initializes this static from program data. On
/// cold boot the contents are unspecified hardware state; on warm
/// boot they're whatever the prior session wrote. Either way, callers
/// must validate via [`super::record::is_valid`] (magic + CRC) before
/// trusting any field.
#[unsafe(link_section = ".bkpsram")]
#[unsafe(no_mangle)]
static mut POSTMORTEM_SLOT: MaybeUninit<PostmortemRecord> = MaybeUninit::uninit();

/// Idempotency latch: `enable()` is safe to call more than once, but
/// the underlying RCC/PWR writes are best done once during init.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Bring BKPSRAM online. Call once during boot, before any
/// post-mortem reader/writer touches the static.
///
/// Steps:
/// 1. `RCC.AHB4ENR.BKPRAMEN = 1` — clocks the BKPSRAM controller so
///    AHB4 reads/writes complete instead of stalling.
/// 2. `PWR.CR1.DBP = 1` — disables backup-domain write protection.
///    Without this, writes to RTC backup registers + BKPSRAM are
///    silently dropped (the existing DFU magic at BKP0 set this
///    indirectly via embassy-stm32, but we make it explicit here).
/// 3. `PWR.CR2.BREN = 1` — enables the backup regulator. Without
///    this, BKPSRAM is **not retained across VBAT-only operation**;
///    cold-power-down survival depends on this bit. Setting it costs
///    a few µA of standby current — negligible on a flight controller.
///
/// Idempotent — repeated calls just observe `ENABLED == true` and
/// return.
pub fn enable() {
    if ENABLED.swap(true, Ordering::AcqRel) {
        return;
    }
    let rcc = pac::RCC;
    let pwr = pac::PWR;
    rcc.ahb4enr().modify(|w| w.set_bkpsramen(true));
    pwr.cr1().modify(|w| w.set_dbp(true));
    pwr.cr2().modify(|w| w.set_bren(true));
    // No explicit poll on BRRDY — the ready bit only matters for
    // VBAT switchover; in normal operation the regulator is already
    // running. Skip the busy-wait so `enable()` is callable from a
    // path where stalling is undesirable (e.g., re-entry from a
    // recovery handler).
}

/// Read-only access to the post-mortem record. Caller's closure runs
/// against a borrow of the static — do not copy out raw pointers.
///
/// **Safety contract for the caller:** [`enable`] must have been
/// called already. Reading from BKPSRAM before clocking it produces a
/// bus fault.
pub fn with_record<R>(f: impl FnOnce(&PostmortemRecord) -> R) -> R {
    // SAFETY: `POSTMORTEM_SLOT` is `static mut` so we synthesize a
    // shared reference for the duration of `f`. No async, no other
    // accessor can be concurrent because every call site is either
    // (a) running on the post-mortem task with no other writers, or
    // (b) running in a fault handler with interrupts disabled.
    // `MaybeUninit::assume_init_ref` is sound because every public
    // accessor either initializes the slot (writers via finalize) or
    // validates it (readers via `is_valid`) before trusting fields.
    let r: &PostmortemRecord = unsafe { (&*(&raw const POSTMORTEM_SLOT)).assume_init_ref() };
    f(r)
}

/// Mutable access to the post-mortem record. Caller's closure runs
/// against an exclusive borrow.
///
/// **Hard rule:** the closure must not await, allocate, take a lock
/// that any post-mortem-emitting path also takes, or call back into
/// other post-mortem APIs. It is callable from a fault handler, but
/// only because it strictly avoids those things.
///
/// **Safety contract for the caller:** [`enable`] must have been
/// called already.
pub fn with_record_mut<R>(f: impl FnOnce(&mut PostmortemRecord) -> R) -> R {
    // SAFETY: same reasoning as `with_record`. `static mut` exclusive
    // access is a deliberate choice: the post-mortem task is the
    // single async writer and serializes all mutations through itself
    // during normal operation; fault handlers run with IRQs disabled.
    let r: &mut PostmortemRecord = unsafe { (&mut *(&raw mut POSTMORTEM_SLOT)).assume_init_mut() };
    f(r)
}

/// Raw pointer to the slot. Used only by the boot-recovery reader
/// before `enable()` has been called: in `pre_init` we want to read
/// BKPSRAM if a prior boot wrote something, but PWR/RCC may not be
/// initialized yet. The returned pointer is valid for the lifetime
/// of the firmware.
///
/// **Hard rule:** the returned pointer must not be dereferenced
/// before clocks/PWR are sufficient. STM32H7's RCC default at reset
/// already has BKPRAMEN clear; reads will stall until clock-gating
/// and DBP are configured (typically by the time `main` runs).
pub fn raw_ptr() -> *mut PostmortemRecord {
    &raw mut POSTMORTEM_SLOT as *mut PostmortemRecord
}

/// Convenience: write the entire record from a freshly-built value.
/// Equivalent to `with_record_mut(|r| { *r = src; finalize(r); })`
/// but inlines the finalize step so the magic-last write order is
/// always honored. Used by writers that build the full record on the
/// stack and copy it in atomically.
pub fn store_finalized(src: &PostmortemRecord) {
    with_record_mut(|r| {
        *r = *src;
        super::record::finalize(r);
    });
}
