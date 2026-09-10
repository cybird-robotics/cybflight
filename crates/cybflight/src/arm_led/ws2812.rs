//! Minimal WS2812B driver for an LED-strip pin, board-agnostic.
//!
//! `new()` takes the timer's general-purpose register block (`TimGp16`), the
//! data pin + its AF number, the timer channel, and a DMA channel + its CC
//! DMAMUX request id — the same `(TimGp16, channel, request)` shape the DShot
//! driver uses, so the strip can live on any timer/channel/pin/DMA the board
//! wires (e.g. SAKURAH743 PA8/TIM1_CH1/AF1/req11, MICOAIR743V2 PD14/TIM4_CH3/
//! AF2/req31). The board is responsible for enabling the timer's RCC clock
//! (`LLTimer::new`) and, for an **advanced** timer (TIM1/TIM8), setting
//! `BDTR.MOE` before constructing — this GP16-only driver never touches BDTR.
//!
//! Kernel clock: on our H7 RCC config both APB1 and APB2 timers run at
//! 240 MHz (prescaler ≠ 1 ⇒ ×2), so `TIMER_HZ` holds for TIM1 and TIM4 alike.
//!
//! Encoding: 800 kHz carrier, "1" = ~0.8 µs high, "0" = ~0.4 µs high; reset
//! latch ≥ 50 µs low between frames. WS2812B wire byte order is GRB.
//!
//! `write()` performs a single one-shot transmission (one DMA burst of
//! `FRAME_LEN` half-words), then stops the timer and parks CCR1 = 0 so the
//! line idles low. The first DMA-loaded compare value reaches the output on
//! the cycle *after* the first update event, so the buffer is prefixed by
//! one zero (low) and suffixed by enough zeros to exceed the 50 µs reset.

use crate::hal::dma::{AnyChannel, Channel, Transfer, TransferOptions};
use crate::hal::gpio::{AfType, Flex, OutputType, Pin, Speed};
use crate::hal::pac::timer::vals;
use crate::hal::pac::timer::TimGp16;
use crate::hal::Peri;

/// Number of LEDs in the chained strip. Each WS2812 consumes the first 24
/// bits it sees and forwards the remainder, so this MUST match the physical
/// chain length — too low and the tail LEDs stay dark; too high adds DMA
/// time but no functional harm. Currently 32 to match the wired Speedybee
/// 2812 Arm LED chain (4 modules × 8 LEDs each, daisy-chained).
pub const NUM_LEDS: usize = 32;

const BITS_PER_LED: usize = 24;

/// Reset pad: ≥ 50 µs of low. At 800 kHz a single PWM period is 1.25 µs, so
/// 48 zero entries give 60 µs — comfortably above the WS2812B latch threshold.
const RESET_PAD: usize = 48;

/// One leading zero absorbs the one-cycle DMA-load delay so the first real
/// bit lands cleanly on cycle 2 of the burst.
const LEADING_PAD: usize = 1;

/// Total frame buffer length (half-words).
pub const FRAME_LEN: usize = LEADING_PAD + NUM_LEDS * BITS_PER_LED + RESET_PAD;

// ---- timer constants (compile-time, asserted against the BSP clock config) -

/// Timer kernel clock in Hz, taken from the selected BSP rather than
/// assumed.
///
/// It was a literal `240_000_000` with a comment noting that "the build
/// will not catch a clock mismatch" — which was true, and also meant one
/// literal was serving three boards. It now tracks whichever board is
/// built, and `clocks::verify_clocks()` re-checks it against the running
/// RCC configuration at boot. The strip may sit on an APB1 or an APB2
/// timer (SAKURAH743 uses TIM1, MICOAIR743V2 TIM4); both run at the same
/// kernel clock under our RCC configuration, which the assertion below
/// pins so a board that breaks that symmetry fails the build.
const TIMER_HZ: u32 = crate::bsp::APB2_TIMER_HZ;
const _: () = assert!(
    crate::bsp::APB1_TIMER_HZ == crate::bsp::APB2_TIMER_HZ,
    "WS2812 may be wired to an APB1 or APB2 timer; this board runs them \
     at different kernel clocks, so the driver must be told which one."
);
const BIT_HZ: u32 = 800_000;

/// PWM period in timer ticks (ARR value). 240e6 / 800e3 = 300 ticks/period.
const PERIOD_TICKS: u16 = (TIMER_HZ / BIT_HZ - 1) as u16; // 299

/// "1" pulse width: 0.8 µs high. 0.8 µs × 240 MHz = 192 ticks.
const ONE_TICKS: u16 = ((TIMER_HZ as u64 * 8) / 10_000_000) as u16;

/// "0" pulse width: 0.4 µs high. 0.4 µs × 240 MHz = 96 ticks.
const ZERO_TICKS: u16 = ((TIMER_HZ as u64 * 4) / 10_000_000) as u16;

/// Tail wait after the DMA completes, in microseconds.
///
/// `xfer.await` fires when the last half-word reaches CCR1, by which
/// point the timer has already clocked out all but the final period. So
/// what this must cover is the tail and the strip's latch window, not
/// the whole burst — it is governed by [`RESET_PAD`], not by
/// [`NUM_LEDS`].
///
/// That distinction is why it stays a small fixed time rather than
/// growing with the chain: a previous comment here put the burst at
/// "~181 µs", which does not match `FRAME_LEN` periods at 800 kHz
/// (~1.02 ms) and made the 250 µs literal look like a frame-length
/// figure that a longer chain would invalidate. Deriving it from the
/// reset pad instead keeps the same ~0.24 ms and makes it correct for
/// any chain length.
const FRAME_DRAIN_US: u64 = {
    let latch_us = (RESET_PAD as u64 * 1_000_000).div_ceil(BIT_HZ as u64);
    latch_us * 4
};

/// The pad must satisfy the WS2812B's ≥ 50 µs latch on its own, since it
/// is what holds the line low at the end of the stream.
const _: () = assert!((RESET_PAD as u64 * 1_000_000) / (BIT_HZ as u64) >= 50);

// Sanity: pulses must fit inside the period.
const _: () = assert!(ONE_TICKS < PERIOD_TICKS);
const _: () = assert!(ZERO_TICKS < ONE_TICKS);

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const OFF: Self = Self { r: 0, g: 0, b: 0 };
    /// Full-brightness red (armed top arms).
    pub const RED: Self = Self { r: 255, g: 0, b: 0 };
    /// Full-brightness blue (armed bottom arms). Single-channel like red,
    /// so per-LED current matches `RED` (~20 mA); no power cap needed.
    pub const BLUE: Self = Self { r: 0, g: 0, b: 255 };
    /// Disarmed dim red (~12% of full). Still visible at a glance but pulls
    /// strip current down by ~6× vs. armed bright, saving battery on the
    /// bench / pre-flight. Arming snaps to `RED` for a clear intensity jump.
    pub const RED_DIM: Self = Self { r: 30, g: 0, b: 0 };
    /// Disarmed dim blue, matched in perceived intensity to `RED_DIM`.
    pub const BLUE_DIM: Self = Self { r: 0, g: 0, b: 30 };
}

pub struct Ws2812 {
    /// Timer register block (GP16 view; valid for any GP or advanced timer).
    /// The board keeps the RCC clock alive by `core::mem::forget`-ing its
    /// `LLTimer` after construction (same pattern as DShot).
    tim_regs: TimGp16,
    /// Output-compare channel index (0=CH1 .. 3=CH4).
    channel: u8,
    /// CC DMAMUX request id for `channel` (e.g. 11 = TIM1_CH1, 31 = TIM4_CH3).
    dma_request: u8,
    /// DMA channel (type-erased so `Ws2812` stays a single concrete type and
    /// `arm_led::task` needs no generic).
    dma: Peri<'static, AnyChannel>,
    buf: [u16; FRAME_LEN],
}

impl Ws2812 {
    /// Configure `channel` of `tim_regs` for 800 kHz PWM on `pin` (alternate
    /// function `af`) and prepare the buffer. Leaves the timer stopped and the
    /// channel's CCR = 0; `write` starts and stops it per call.
    ///
    /// The board must have already enabled the timer's RCC clock (via
    /// `LLTimer::new`, then `core::mem::forget` to keep it alive) and, for an
    /// advanced timer (TIM1/TIM8), set `BDTR.MOE` — this driver only touches
    /// the GP16 register subset.
    pub fn new(
        tim_regs: TimGp16,
        pin: Peri<'static, impl Pin>,
        af: u8,
        channel: u8,
        dma: Peri<'static, impl Channel>,
        dma_request: u8,
    ) -> Self {
        // Data pin → timer AF, push-pull, low slew rate.
        let mut flex = Flex::new(pin);
        flex.set_as_af_unchecked(af, AfType::output(OutputType::PushPull, Speed::Low));
        core::mem::forget(flex);

        let ch = channel as usize;
        let reg = ch / 2; // CH1/CH2 → ccmr_output(0); CH3/CH4 → ccmr_output(1)
        let sub = ch % 2; // channel within the CCMR register

        tim_regs.psc().write_value(0);
        tim_regs.arr().write(|w| w.set_arr(PERIOD_TICKS));
        tim_regs.cr1().modify(|w| {
            w.set_arpe(true);
            w.set_urs(vals::Urs::COUNTER_ONLY);
        });

        // Channel → PWM mode 1 with output preload, normal polarity (idle low).
        tim_regs.ccmr_output(reg).modify(|w| {
            w.set_ocm(sub, vals::Ocm::PWM_MODE1);
            w.set_ocpe(sub, true);
        });
        tim_regs.ccer().modify(|w| {
            w.set_cce(ch, true);
            w.set_ccp(ch, false);
        });
        tim_regs.ccr(ch).write(|w| w.set_ccr(0));

        // Software UEV to load PSC/ARR/CCR/CCMR shadow → active.
        tim_regs.egr().write(|w| w.set_ug(true));

        Self {
            tim_regs,
            channel,
            dma_request,
            dma: dma.into(),
            buf: [0u16; FRAME_LEN],
        }
    }

    /// Drive `frame` onto the strip. Blocks until the DMA burst is loaded
    /// AND the timer has clocked all `FRAME_LEN` PWM periods out of the pin.
    /// `xfer.await` only signals "DMA copied the last word into CCR1"; the
    /// final period (and the reset pad) still needs ~1.25 µs/cycle to play
    /// out. Stopping the timer at `xfer.await` truncates the tail and the
    /// strip latches partial data → no visible output.
    pub async fn write(&mut self, frame: &[Rgb; NUM_LEDS]) {
        // Build the half-word stream: leading pad, then 24 bits/LED in GRB
        // order MSB-first, then the reset pad. Bits already encoded as PWM
        // compare values; DMA will copy them straight into CCR1.
        let mut idx = LEADING_PAD;
        for led in frame {
            let grb = ((led.g as u32) << 16) | ((led.r as u32) << 8) | (led.b as u32);
            for i in 0..BITS_PER_LED {
                let bit = (grb >> (BITS_PER_LED - 1 - i)) & 1;
                self.buf[idx] = if bit == 1 { ONE_TICKS } else { ZERO_TICKS };
                idx += 1;
            }
        }
        // Both the leading and reset slots are already zero — the buffer is
        // re-initialised on every call but the pad bytes never change.

        let regs = self.tim_regs;
        let ch = self.channel as usize;
        // Reset counter, enable the channel's CC DMA (matches `dma_request`),
        // start timer. CC DMA fires when the counter matches CCR mid-period,
        // the DMA controller writes the next compare value into CCR's shadow,
        // and ARPE/OCPE causes it to take effect at the next overflow.
        regs.cnt().write(|w| w.set_cnt(0));
        regs.dier().modify(|w| w.set_ccde(ch, true));
        regs.cr1().modify(|w| w.set_cen(true));

        let opts = TransferOptions::default();
        let xfer = unsafe {
            Transfer::new_write(
                self.dma.reborrow(),
                self.dma_request,
                &self.buf,
                regs.ccr(ch).as_ptr() as *mut u16,
                opts,
            )
        };
        xfer.await;

        // Wait for the timer to finish clocking the bit stream onto the
        // pin. Without this we stop the timer mid-burst and the strip
        // latches partial garbage (or nothing).
        //
        // Derived from FRAME_LEN rather than the former fixed 250 µs,
        // which was sized for a 32-LED chain and silently became too
        // short for a longer one — and NUM_LEDS reads as a freely
        // editable chain length.
        embassy_time::Timer::after_micros(FRAME_DRAIN_US).await;

        // Park: stop timer, disable CC DMA, drive CCR to 0 so the line idles
        // low between calls (which doubles as the next-frame reset).
        regs.cr1().modify(|w| w.set_cen(false));
        regs.dier().modify(|w| w.set_ccde(ch, false));
        regs.ccr(ch).write(|w| w.set_ccr(0));
    }
}
