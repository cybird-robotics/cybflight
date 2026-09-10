//! Reset-cause capture: read `RCC.RSR` + `PWR.CSR1` once, very early
//! in boot, then clear the latch.
//!
//! ## Why this needs to live in `pre_init`
//!
//! `RCC.RSR` is sticky: every boot's reset cause stays in the register
//! until software clears `RMVF`. If main code (or even `bsp::init`)
//! reads RCC for any other reason and then writes `RMVF` before we
//! capture, we lose the cause forever — there's no other path to
//! distinguish "IWDG reset" from "soft reset" after the flags are
//! gone. So we read it in `pre_init`, before any HAL code runs, and
//! stash the value into a `.uninit` static.
//!
//! ## Why `.uninit` (no init data)
//!
//! `RAW_RESET_CAUSE` is a regular RAM static, but we deliberately put
//! it in the `.uninit` section so its value is *not* zeroed by the
//! cortex-m-rt startup. `pre_init` runs **before** the .bss zeroing
//! loop, so the value we write there would otherwise be overwritten
//! immediately. Same trick `panic-persist` uses for its panic message.
//!
//! ## Hard rules
//!
//! - This module is callable from `pre_init` — no allocator, no async,
//!   no peripherals other than raw RCC/PWR reads. It runs before
//!   `bsp::init` configures clocks, but RCC.RSR / PWR.CSR1 are
//!   accessible without any clock setup (they're in the always-on
//!   domain).
//! - The capture must happen **before** the DFU magic check in
//!   `platform.rs::pre_init`, because the DFU jump path bypasses the
//!   rest of boot and the user might want to read RCC.RSR after DFU
//!   anyway. Capture is cheap (~10 cycles), DFU is rare — it's safe to
//!   capture unconditionally.

use core::mem::MaybeUninit;

use crate::hal::pac;

// (The CAPTURED sentinel bit is exposed as `flags::CAPTURED` below;
// it doubles as both "valid marker" and the bit-31 slot in the
// packed value, so callers and the capture/read fns share one
// constant.)

/// Raw register values stashed by `pre_init`. Layout:
///
/// ```text
///   bit 31    = CAPTURED_BIT (set by `capture`, cleared by hardware reset)
///   bit 30    = was_iwdg_reset (RCC.RSR.IWDG1RSTF, sticky across init)
///   bit 29    = was_brownout (PWR.CSR1.PVDO at the moment of capture)
///   bit 28    = was_software_reset (RCC.RSR.SFTRSTF)
///   bit 27    = was_pin_reset (RCC.RSR.PINRSTF)
///   bit 26    = was_por (RCC.RSR.PORRSTF)
///   bit 25    = was_bor (RCC.RSR.BORRSTF)
///   bit 24    = was_lpwr_reset (RCC.RSR.LPWRRSTF)
///   bits 0..16 = reserved for future fault flags
/// ```
///
/// Stored in `.uninit` so the cortex-m-rt zero-init loop does **not**
/// clobber the value `pre_init` writes.
#[unsafe(link_section = ".uninit")]
#[unsafe(no_mangle)]
static mut RAW_RESET_CAUSE: MaybeUninit<u32> = MaybeUninit::uninit();

/// Bit positions in the stashed value. Public so the post-mortem
/// record schema can reference them, and so test code on the host
/// can construct synthetic values without invoking the embedded
/// registers.
pub mod flags {
    pub const CAPTURED: u32 = 1 << 31;
    pub const IWDG_RESET: u32 = 1 << 30;
    pub const BROWNOUT: u32 = 1 << 29;
    pub const SOFTWARE_RESET: u32 = 1 << 28;
    pub const PIN_RESET: u32 = 1 << 27;
    pub const POR: u32 = 1 << 26;
    pub const BOR: u32 = 1 << 25;
    pub const LPWR_RESET: u32 = 1 << 24;
}

/// Read `RCC.RSR` + `PWR.CSR1`, pack the relevant flags into a single
/// `u32`, stash into `RAW_RESET_CAUSE`, then clear `RCC.RSR.RMVF` so
/// the *next* boot starts with a fresh latch.
///
/// **Hard rules:** no allocator, no Mutex, no defmt — this runs
/// before any clocks are up beyond HSI. defmt would be silent anyway
/// because UART isn't configured yet; calling it can be done by
/// later code reading [`take`].
///
/// Idempotency: re-running this fn after first capture would re-read
/// (now-cleared) RCC.RSR and overwrite the stashed value with zeros.
/// `pre_init` is only called once by cortex-m-rt so this is fine; if
/// a future change adds a second caller, gate on the CAPTURED bit
/// here.
pub fn capture() {
    let rcc = pac::RCC;
    let pwr = pac::PWR;

    let rsr = rcc.rsr().read();
    let csr1 = pwr.csr1().read();

    let mut packed = flags::CAPTURED;
    if rsr.iwdg1rstf() {
        packed |= flags::IWDG_RESET;
    }
    if csr1.pvdo() {
        // PVDO = 1 means VDD has dropped below the PVD threshold.
        packed |= flags::BROWNOUT;
    }
    if rsr.sftrstf() {
        packed |= flags::SOFTWARE_RESET;
    }
    if rsr.pinrstf() {
        packed |= flags::PIN_RESET;
    }
    if rsr.porrstf() {
        packed |= flags::POR;
    }
    if rsr.borrstf() {
        packed |= flags::BOR;
    }
    if rsr.lpwrrstf() {
        packed |= flags::LPWR_RESET;
    }

    // SAFETY: `pre_init` is single-threaded and runs before any other
    // code touches `RAW_RESET_CAUSE`. Writing through a raw pointer
    // avoids the `&mut static mut` lint.
    unsafe {
        (&raw mut RAW_RESET_CAUSE).write(MaybeUninit::new(packed));
    }

    // Clear the latch so the next reset starts with a clean slate.
    // RMVF is bit 16 of RSR — write 1 to clear (rs1cw semantics on
    // STM32H7).
    rcc.rsr().modify(|w| w.set_rmvf(true));
}

/// Read the stashed value. Idempotent — does not clear the static, so
/// `boot_recovery` can read it once and the shell `postmortem show`
/// command can also read it later in the same boot.
///
/// Returns `0` if [`capture`] was never called (test builds, or a
/// build that compiles out the post-mortem subsystem and somehow
/// still reaches this fn). The `CAPTURED` sentinel bit distinguishes
/// "captured nothing" (returns `flags::CAPTURED` only) from "never
/// ran" (returns `0`).
pub fn read() -> u32 {
    let raw = unsafe { (&raw const RAW_RESET_CAUSE).read().assume_init() };
    if raw & flags::CAPTURED == 0 { 0 } else { raw }
}

/// Decode a stashed value into a human-readable kind. Used by
/// `defmt` boot-recovery logging and the shell `postmortem show`
/// command.
#[derive(Clone, Copy, Debug, defmt::Format)]
pub enum ResetKind {
    Unknown,
    PowerOn,
    Pin,
    Software,
    IndependentWatchdog,
    LowPower,
    Brownout,
    /// The captured value reports multiple sticky flags simultaneously
    /// (e.g. POR + Pin) — common on the very first boot after flashing
    /// because both POR and PIN can latch.
    Multiple,
}

pub fn classify(raw: u32) -> ResetKind {
    if raw & flags::CAPTURED == 0 {
        return ResetKind::Unknown;
    }
    let bits = raw
        & (flags::IWDG_RESET
            | flags::BROWNOUT
            | flags::SOFTWARE_RESET
            | flags::PIN_RESET
            | flags::POR
            | flags::BOR
            | flags::LPWR_RESET);
    // Priority: the most-specific cause wins. IWDG > Brownout >
    // Software > LowPower > Pin > BOR > POR > Multiple.
    if bits & flags::IWDG_RESET != 0 {
        return ResetKind::IndependentWatchdog;
    }
    if bits & flags::BROWNOUT != 0 {
        return ResetKind::Brownout;
    }
    if bits & flags::SOFTWARE_RESET != 0 {
        return ResetKind::Software;
    }
    if bits & flags::LPWR_RESET != 0 {
        return ResetKind::LowPower;
    }
    if bits & flags::PIN_RESET != 0 {
        return ResetKind::Pin;
    }
    if bits & (flags::POR | flags::BOR) != 0 {
        return ResetKind::PowerOn;
    }
    ResetKind::Unknown
}
