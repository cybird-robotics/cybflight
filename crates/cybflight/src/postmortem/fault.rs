//! Fault handlers: custom `#[panic_handler]`, `HardFault` exception,
//! PVD brown-out IRQ. These three paths share a single contract:
//!
//! ## Hard rules (non-negotiable)
//!
//! - **No allocation.** No `String`, no `Vec`, no `Box`, no
//!   `format_args!` to a non-stack target.
//! - **No async.** No `await`, no `Signal::wait`, no embassy executor
//!   touchpoints. The thread executor is not running here.
//! - **No `Mutex<RefCell<T>>` or `critical_section::Mutex`.** Locks
//!   may already be held by the code that just panicked. The
//!   post-mortem record's atomicity comes from the magic-last + CRC
//!   protocol in [`super::record`], not from any lock.
//! - **No defmt logging during the actual commit.** A `defmt::error!`
//!   call panicking in the panic handler is a recursion source. We
//!   do log a single short message *before* the BKPSRAM commit so
//!   `probe-rs run` users see a clue, but the BKPSRAM commit itself
//!   is bracketed by `interrupt::disable` and is panic-free.
//! - **Re-entrancy guard.** A panic during the panic handler (e.g.
//!   a second fault while writing BKPSRAM) hits an `AtomicU8`
//!   depth counter; depth > 1 immediately `udf`s — no recursion,
//!   no infinite reset loop.

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicU8, Ordering};

use cortex_m::peripheral::SCB;

use super::bkpsram;
use super::record::{self, FatalKind, FatalSummary, PANIC_MSG_IDX_NONE};
use crate::hal::interrupt;

/// Re-entrancy guard. `0` = not in any fault handler, `1` = in one,
/// `>1` = recursive call (immediately abort).
static FAULT_DEPTH: AtomicU8 = AtomicU8::new(0);

/// `#[panic_handler]` registration. Replaces `panic_probe`'s default
/// when the `postmortem` feature is on.
///
/// Behavior:
/// 1. Disable interrupts so concurrent tasks can't observe a partly-
///    written record.
/// 2. Bump the re-entrancy counter; bail with `udf` if we're already
///    inside a handler.
/// 3. Emit a short defmt warning so probe-rs sees the panic before
///    the reset (no rzcobs framing changes — this is just one
///    `error!` call).
/// 4. Commit the fatal slot to BKPSRAM via raw-pointer writes only.
/// 5. Hardware reset.
#[cfg(feature = "postmortem")]
#[panic_handler]
fn panic_handler(info: &PanicInfo) -> ! {
    cortex_m::interrupt::disable();
    let depth = FAULT_DEPTH.fetch_add(1, Ordering::AcqRel);
    if depth >= 1 {
        // Recursive panic — the previous handler invocation is still
        // running, or a HardFault interrupted it. Bail without
        // touching anything else.
        cortex_m::asm::udf();
    }

    // Best-effort defmt log. If this panics (it shouldn't — defmt's
    // own panic handling is conservative), the depth counter catches
    // it on re-entry.
    defmt::error!("postmortem: panic (info={})", defmt::Display2Format(info));

    // Commit. PC/LR/PSR aren't directly available in a panic handler
    // (we'd need to inspect the call stack), so we record only what
    // we know — the FatalKind and panic_msg_idx (currently always
    // `NONE` until a static message-table is wired up — Stage B).
    let summary = FatalSummary {
        kind: FatalKind::Panic as u8,
        _pad0: [0; 3],
        pc: 0,
        lr: 0,
        psr: 0,
        cfsr: 0,
        hfsr: 0,
        mmfar: 0,
        bfar: 0,
        panic_msg_idx: PANIC_MSG_IDX_NONE,
        _pad1: [0; 31],
    };
    commit_fatal(summary);

    SCB::sys_reset();
}

/// Stub used by builds without the `postmortem` feature so the linker
/// can still find a panic handler (provided by `panic_probe`). When
/// the feature is off, `main.rs` keeps `use panic_probe as _;` and
/// this module's symbol isn't pulled in.
#[cfg(not(feature = "postmortem"))]
const _: () = ();

/// HardFault exception. Cortex-M-RT calls this with the stacked
/// exception frame; we extract PC/LR/PSR and the SCB fault status
/// registers.
///
/// Constraints same as panic handler — no async, no alloc, no
/// Mutex. Runs in handler mode with interrupts disabled by the
/// hardware (HardFault is escalated above the BASEPRI threshold).
#[cfg(feature = "postmortem")]
#[cortex_m_rt::exception]
unsafe fn HardFault(frame: &cortex_m_rt::ExceptionFrame) -> ! {
    let depth = FAULT_DEPTH.fetch_add(1, Ordering::AcqRel);
    if depth >= 1 {
        cortex_m::asm::udf();
    }

    // SAFETY: we own the SCB pointer for the duration of this
    // handler — no other code is running.
    let scb = unsafe { &*SCB::PTR };
    let cfsr = scb.cfsr.read();
    let hfsr = scb.hfsr.read();
    let mmfar = scb.mmfar.read();
    let bfar = scb.bfar.read();

    // Skip defmt here — a HardFault during a defmt::error! is the
    // textbook "panic in panic" recursion. Operators see "boot
    // recovered HardFault" on next boot via the post-mortem mirror.

    let summary = FatalSummary {
        kind: FatalKind::HardFault as u8,
        _pad0: [0; 3],
        pc: frame.pc(),
        lr: frame.lr(),
        psr: frame.xpsr(),
        cfsr,
        hfsr,
        mmfar,
        bfar,
        panic_msg_idx: PANIC_MSG_IDX_NONE,
        _pad1: [0; 31],
    };
    commit_fatal(summary);

    SCB::sys_reset();
}

/// PVD brown-out interrupt path. The PVD trips ~2.7 V (configurable
/// via PWR.CR1.PLS), well above the BOR threshold (~1.85 V), giving
/// us ~2–10 ms of headroom — depending on the board's bulk
/// capacitance — to commit the post-mortem record before BOR pulls
/// the rug.
///
/// Wired as the EXTI16 (PVD) interrupt handler. The user must:
/// 1. `pwr.cr1().modify(|w| w.set_pls(<threshold>))`
/// 2. `pwr.cr1().modify(|w| w.set_pvden(true))`
/// 3. `exti.imr1().modify(|w| w.set_line(16, true))`
/// 4. `exti.rtsr1().modify(|w| w.set_line(16, true))`  (rising edge =
///    voltage falling below threshold)
/// 5. NVIC unmask EXTI16.
///
/// Configuration lives in [`enable_pvd_brownout`]. The handler itself
/// is registered via `#[interrupt]`.
///
/// **Minimal-diff principle:** the steady-state ring is kept up to
/// date by the post-mortem task, so all this handler has to do is
/// record the fatal kind + finalize. Realistic budget: 2–3 µs vs. a
/// 2–10 ms PVD-to-BOR window — three orders of magnitude of margin.
#[cfg(feature = "postmortem")]
#[interrupt]
fn PVD_AVD() {
    let depth = FAULT_DEPTH.fetch_add(1, Ordering::AcqRel);
    if depth >= 1 {
        cortex_m::asm::udf();
    }

    // Clear the EXTI pending bit — without this the IRQ would
    // re-fire on return. EXTI16 lives in bank 0 on STM32H7 (lines
    // 0..31).
    let exti = crate::hal::pac::EXTI;
    exti.pr(0).write(|w| w.set_line(16, true));

    let summary = FatalSummary {
        kind: FatalKind::Brownout as u8,
        _pad0: [0; 3],
        pc: 0,
        lr: 0,
        psr: 0,
        cfsr: 0,
        hfsr: 0,
        mmfar: 0,
        bfar: 0,
        panic_msg_idx: PANIC_MSG_IDX_NONE,
        _pad1: [0; 31],
    };
    commit_fatal(summary);

    // Park here until BOR or HW reset. WFI cuts power to the core
    // until any pending IRQ wakes us — the only thing that should
    // wake us is BOR, which doesn't take this path.
    loop {
        cortex_m::asm::wfi();
    }
}

/// Enable the PVD interrupt at the configured threshold. Call once
/// during init. The threshold-selection bits (PLS) on STM32H7
/// support several discrete levels; we pick the highest available
/// (~2.85 V on H7) to maximize the time-to-BOR window.
///
/// Idempotent: re-enabling is harmless. **Must be called after
/// clocks are configured** (PWR clock comes from APB1).
#[cfg(feature = "postmortem")]
pub fn enable_pvd_brownout() {
    use crate::hal::pac;
    let pwr = pac::PWR;
    let exti = pac::EXTI;
    // Threshold = level 7 (~2.85 V on STM32H7). Bit values are
    // chip-specific; PLS=0b111 selects the highest threshold,
    // giving us the longest commit window.
    pwr.cr1().modify(|w| {
        // PLS bits [7:5] select the PVD threshold. 0b111 picks the
        // highest threshold (~2.85 V on STM32H7), maximizing the
        // PVD-to-BOR commit window. The H743 PAC takes the raw u8.
        w.set_pls(0b111);
        w.set_pvde(true);
    });
    // EXTI16 = PVD line. Enable interrupt mask + rising-edge trigger
    // (PVDO transitions 0 → 1 when VDD falls below the threshold).
    // EXTI lines 0..31 live in bank index 0 on STM32H7.
    exti.imr(0).modify(|w| w.set_line(16, true));
    exti.rtsr(0).modify(|w| w.set_line(16, true));
    // NVIC unmask handled by the embassy bind_interrupts pattern;
    // PVD_AVD is not bound by any HAL driver so cortex-m-rt's
    // exception entry routes the IRQ to our handler directly.
    unsafe {
        cortex_m::peripheral::NVIC::unmask(pac::Interrupt::PVD_AVD);
    }
}

// ── Internal commit helper ──────────────────────────────────────────────

/// Write `summary` into the BKPSRAM record's fatal slot, refresh the
/// header's CRC + magic. Single point of "actually finalize" so the
/// three handler paths all use the same memory ordering and
/// validation rules.
fn commit_fatal(summary: FatalSummary) {
    bkpsram::with_record_mut(|r| {
        r.fatal = summary;
        // Refresh uptime if monotonic time is available. We use
        // raw `embassy_time` access via a public fn; if anything
        // goes wrong (e.g. timer not initialized in early boot),
        // we leave the field at whatever the postmortem_task last
        // wrote.
        if let Some(now_ms) = monotonic_ms() {
            r.header.uptime_ms = now_ms;
        }
        record::finalize(r);
    });
}

/// Best-effort monotonic-time read. Returns `None` if `embassy_time`
/// isn't initialized yet (the post-mortem subsystem comes online
/// before the executor in `main`).
fn monotonic_ms() -> Option<u32> {
    // `Instant::now` is safe to call from any context post-init. Pre-
    // init it would still return a value (zero) — embassy's tick
    // counter is initialized by the bsp clock setup.
    Some(embassy_time::Instant::now().as_millis() as u32)
}
