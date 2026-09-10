//! Boot-time verification of the BSP's declared clock constants.
//!
//! Several drivers compute hardware timings from a clock they cannot
//! read: the DShot bit period from the timer kernel clock, the WS2812
//! pulse widths from the same, and the INDI loop-timing histogram from
//! SYSCLK via the DWT cycle counter. Each used to carry its own literal
//! (`240_000_000`, `480_000_000`, a bare `delay(200)`), so a change to a
//! BSP's `board_config` would silently retime DShot and WS2812 rather
//! than fail. The comment in `ws2812.rs` said as much: "the build will
//! not catch a clock mismatch."
//!
//! The constants now live in the BSP next to the RCC configuration that
//! determines them, and the drivers assert against those at compile
//! time. That closes the cross-board hazard — one driver constant shared
//! by three boards — but not the same-board one, because the constant
//! and the configuration still sit in the same file and can still be
//! edited apart.
//!
//! This module closes the rest: it asks the HAL what the timer kernel
//! clocks actually came out as, measures the CPU clock the DWT counter
//! runs on, and compares both against the BSP. A mismatch is reported,
//! not fatal — the failure mode is mistimed LEDs, DShot and loop-timing
//! telemetry, which the operator needs to be *told* about, whereas a
//! panic here would boot-loop the aircraft over what may be a benign
//! retune (docs/safety_protocol.md).

use embassy_time::{Duration, Instant};

use crate::hal::peripherals::{TIM1, TIM3};
use crate::hal::rcc;

/// Measurement window for the core-clock check.
///
/// SYSCLK is the one declared clock the RCC will not report back: it is
/// not any peripheral's kernel clock, and embassy's `rcc::clocks()` needs
/// the `RCC` peripheral, which `hal::init` has already consumed by the
/// time anything can ask. What the constant is actually *used* for is
/// converting DWT cycle counts to microseconds, so that is what gets
/// measured — the DWT counter is clocked by the CPU, and the embassy time
/// driver runs off a timer whose kernel clock is verified above, so the
/// ratio of the two over a fixed window is the core clock.
///
/// 10 ms is 328 ticks of the 32_768 Hz time base (±0.3 %) and ~4.8 M DWT
/// cycles at 480 MHz, far inside the counter's 32-bit wrap. Every error
/// this can catch is a prescaler change — a factor of two at minimum — so
/// the window buys about two orders of magnitude more resolution than the
/// check needs, at 10 ms of one-time boot cost before any task is
/// spawned.
const CORE_CLOCK_WINDOW: Duration = Duration::from_millis(10);

/// Tolerance on the measured core clock, as a divisor of the declared
/// value (1/50 = 2 %). Comfortably above the measurement's own ±0.3 % and
/// far below the smallest real error.
const CORE_CLOCK_TOL_DIV: u32 = 50;

/// Measure the CPU clock by counting DWT cycles over a fixed window of
/// the embassy time base. Returns 0 if the window came back empty.
fn measure_core_clock_hz() -> u32 {
    // Idempotent: `indi_task` enables the same counter later for its
    // step-cost probe.
    let mut cp = unsafe { cortex_m::Peripherals::steal() };
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    let t0 = Instant::now();
    let c0 = cortex_m::peripheral::DWT::cycle_count();
    let deadline = t0 + CORE_CLOCK_WINDOW;
    // Busy-wait rather than `Timer::after`: this runs before any task is
    // spawned, so there is nothing to yield to, and keeping the function
    // synchronous keeps the whole clock check at one call site in `main`.
    while Instant::now() < deadline {}
    let c1 = cortex_m::peripheral::DWT::cycle_count();
    let elapsed_us = Instant::now().saturating_duration_since(t0).as_micros();
    if elapsed_us == 0 {
        return 0;
    }
    let cycles = u64::from(c1.wrapping_sub(c0));
    (cycles * 1_000_000 / elapsed_us) as u32
}

/// Compare the BSP's declared clocks against the running configuration.
///
/// Returns `true` when everything matches. Call once, early in `main`,
/// after `hal::init`.
pub fn verify_clocks() -> bool {
    let mut ok = true;

    // The timer kernel clocks are what DShot and WS2812 derive their bit
    // timings from. `frequency::<T>()` reports the kernel clock, so on
    // H7 it already includes the ×2 the APB prescaler triggers.
    let apb2_tim = rcc::frequency::<TIM1>().0;
    if apb2_tim != crate::bsp::APB2_TIMER_HZ {
        defmt::error!(
            "clocks: APB2 timer kernel clock is {} Hz, BSP declares {} Hz — DShot and WS2812 bit timings are wrong",
            apb2_tim,
            crate::bsp::APB2_TIMER_HZ,
        );
        ok = false;
    }

    let apb1_tim = rcc::frequency::<TIM3>().0;
    if apb1_tim != crate::bsp::APB1_TIMER_HZ {
        defmt::error!(
            "clocks: APB1 timer kernel clock is {} Hz, BSP declares {} Hz",
            apb1_tim,
            crate::bsp::APB1_TIMER_HZ,
        );
        ok = false;
    }

    // The DWT-derived core clock, against what the BSP declares. This is
    // the constant `indi_task`'s loop-timing histogram and DShot's
    // `PIN_SETTLE_CYCLES` are computed from, so a mismatch silently
    // scales every reported loop time and shortens the DShot pin-settle
    // wait — the same class of failure as a mistimed bit period, and
    // previously the one thing this function printed without checking.
    let core_hz = measure_core_clock_hz();
    let declared = crate::bsp::SYSCLK_HZ;
    if core_hz == 0 || core_hz.abs_diff(declared) > declared / CORE_CLOCK_TOL_DIV {
        defmt::error!(
            "clocks: CPU clock measures {} Hz, BSP declares SYSCLK {} Hz — DWT loop timings and \
             DShot pin-settle delays are wrong",
            core_hz,
            declared,
        );
        ok = false;
    }

    if ok {
        defmt::info!(
            "clocks: verified — SYSCLK {} Hz (measured {}), timer kernel {} Hz",
            declared,
            core_hz,
            apb2_tim,
        );
    }
    ok
}
