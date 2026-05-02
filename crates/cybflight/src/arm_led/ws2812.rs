//! Minimal WS2812B driver for the SAKURAH743 LED_STRIP pin.
//!
//! Hardware wiring (from `bsp::LED_STRIP_META`): PA8 → TIM1_CH1 (AF1),
//! data fed by DMA1_CH7 with DMAMUX request 11 (TIM1_UP). TIM1 is an
//! advanced-control timer; BDTR.MOE must be set for the output stage to
//! drive the pin. Kernel clock derives from APB2: with the SAKURAH743 RCC
//! config (APB2=120 MHz, prescaler ≠ 1), the timer clock is 240 MHz.
//!
//! Encoding: 800 kHz carrier, "1" = ~0.8 µs high, "0" = ~0.4 µs high; reset
//! latch ≥ 50 µs low between frames. WS2812B wire byte order is GRB.
//!
//! `write()` performs a single one-shot transmission (one DMA burst of
//! `FRAME_LEN` half-words), then stops the timer and parks CCR1 = 0 so the
//! line idles low. The first DMA-loaded compare value reaches the output on
//! the cycle *after* the first update event, so the buffer is prefixed by
//! one zero (low) and suffixed by enough zeros to exceed the 50 µs reset.

use core::mem::ManuallyDrop;

use crate::hal::dma::{Transfer, TransferOptions};
use crate::hal::gpio::{AfType, Flex, OutputType, Speed};
use crate::hal::pac;
use crate::hal::pac::timer::vals;
use crate::hal::peripherals::{DMA1_CH7, PA8, TIM1};
use crate::hal::timer::low_level::Timer as LLTimer;
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

/// TIM1 kernel clock in Hz. APB2 = 120 MHz, prescaler ≠ 1 ⇒ timer clock is
/// APB2 × 2 = 240 MHz on STM32H7. If the BSP RCC config changes, update this
/// and the assertions below; the build will not catch a clock mismatch.
const TIMER_HZ: u32 = 240_000_000;
const BIT_HZ: u32 = 800_000;

/// PWM period in timer ticks (ARR value). 240e6 / 800e3 = 300 ticks/period.
const PERIOD_TICKS: u16 = (TIMER_HZ / BIT_HZ - 1) as u16; // 299

/// "1" pulse width: 0.8 µs high. 0.8 µs × 240 MHz = 192 ticks.
const ONE_TICKS: u16 = ((TIMER_HZ as u64 * 8) / 10_000_000) as u16;

/// "0" pulse width: 0.4 µs high. 0.4 µs × 240 MHz = 96 ticks.
const ZERO_TICKS: u16 = ((TIMER_HZ as u64 * 4) / 10_000_000) as u16;

// Sanity: pulses must fit inside the period.
const _: () = assert!(ONE_TICKS < PERIOD_TICKS);
const _: () = assert!(ZERO_TICKS < ONE_TICKS);

/// DMAMUX request ID for **TIM1_CH1** on STM32H743 (RM0433 Table 121).
/// Note: Betaflight's `LED_STRIP_META` lists this as request 11 too — that
/// metadata is the CH1 capture-compare request, not TIM1_UP (which is 15).
/// We pair it with `CCDE` (capture-compare DMA enable) below; using `UDE`
/// here would be a silent mismatch and the timer would never fire DMA.
const DMA_REQ_TIM1_CH1: u8 = 11;

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
    /// Held to keep TIM1's RCC clock enabled for the lifetime of the driver.
    /// Dropping `LLTimer` would gate the clock.
    _tim_keepalive: ManuallyDrop<LLTimer<'static, TIM1>>,
    dma: Peri<'static, DMA1_CH7>,
    buf: [u16; FRAME_LEN],
}

impl Ws2812 {
    /// Configure TIM1_CH1 for 800 kHz PWM on PA8 (AF1) and prepare the buffer.
    /// Leaves the timer stopped and CCR1 = 0; `write` starts and stops it
    /// per call.
    pub fn new(
        tim1: Peri<'static, TIM1>,
        pa8: Peri<'static, PA8>,
        dma: Peri<'static, DMA1_CH7>,
    ) -> Self {
        // Enable RCC clock + reset for TIM1.
        let timer = LLTimer::new(tim1);

        // PA8 → AF1 (TIM1_CH1), push-pull, low slew rate.
        let mut flex = Flex::new(pa8);
        flex.set_as_af_unchecked(1, AfType::output(OutputType::PushPull, Speed::Low));
        core::mem::forget(flex);

        let regs = timer.regs_advanced();
        regs.psc().write_value(0);
        regs.arr().write(|w| w.set_arr(PERIOD_TICKS));
        regs.cr1().modify(|w| {
            w.set_arpe(true);
            w.set_urs(vals::Urs::COUNTER_ONLY);
        });

        // CH1 → PWM mode 1 with output preload, normal polarity (idle low).
        regs.ccmr_output(0).modify(|w| {
            w.set_ocm(0, vals::Ocm::PWM_MODE1);
            w.set_ocpe(0, true);
        });
        regs.ccer().modify(|w| {
            w.set_cce(0, true);
            w.set_ccp(0, false);
        });
        regs.ccr(0).write(|w| w.set_ccr(0));

        // Advanced-timer-only: Main Output Enable. Without this the GPIO
        // stays Hi-Z regardless of CCR.
        regs.bdtr().modify(|w| w.set_moe(true));

        // Software UEV to load PSC/ARR/CCR/CCMR shadow → active.
        regs.egr().write(|w| w.set_ug(true));

        Self {
            _tim_keepalive: ManuallyDrop::new(timer),
            dma,
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

        let regs = pac::TIM1;
        // Reset counter, enable CC1 DMA (matches DMAMUX request 11), start timer.
        // CC1 DMA fires when the counter matches CCR1 mid-period, the DMA
        // controller writes the next compare value into CCR1's shadow, and
        // ARPE/OCPE causes it to take effect at the next overflow. Functionally
        // equivalent to update DMA, but matches the request ID we hold.
        regs.cnt().write(|w| w.set_cnt(0));
        regs.dier().modify(|w| w.set_ccde(0, true));
        regs.cr1().modify(|w| w.set_cen(true));

        let opts = TransferOptions::default();
        let xfer = unsafe {
            Transfer::new_write(
                self.dma.reborrow(),
                DMA_REQ_TIM1_CH1,
                &self.buf,
                regs.ccr(0).as_ptr() as *mut u16,
                opts,
            )
        };
        xfer.await;

        // Wait for the timer to finish clocking the bit stream onto the pin.
        // FRAME_LEN cycles × 1.25 µs/cycle = ~181 µs at 800 kHz; round up to
        // 250 µs for margin. Without this, we stop the timer mid-burst and
        // the strip latches partial garbage (or nothing).
        embassy_time::Timer::after_micros(250).await;

        // Park: stop timer, disable CC1 DMA, drive CCR1 to 0 so the line
        // idles low between calls (which doubles as the next-frame reset).
        regs.cr1().modify(|w| w.set_cen(false));
        regs.dier().modify(|w| w.set_ccde(0, false));
        regs.ccr(0).write(|w| w.set_ccr(0));
    }
}
