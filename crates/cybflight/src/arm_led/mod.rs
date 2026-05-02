//! External arm-LED control for the SAKURAH743 (Speedybee 2812 chain on PA8).
//!
//! Status indicator only — outside the safety chain. The task reads
//! `ARM_LED_ENABLED` and `motors::IS_ARMED`, never writes anything in the
//! safety chain; never touches `ACTUATOR_MOTORS`, `ARM_STATE`, or
//! `LAST_CONTROLLER_PUBLISH`. See `docs/safety_protocol.md`.
//!
//! ## Physical layout
//!
//! 32 LEDs in 8 groups of 4. The chain alternates bottom/top facing in the
//! pattern below — wired this way because the strip wraps from the bottom of
//! one arm onto the top of the next without breaking the daisy-chain. The
//! `GROUP_FACING` table is the single source of truth; reorder it if the
//! strip wiring changes.
//!
//! ```text
//!   chain index:  [0..4]  [4..8]  [8..12]  [12..16]  [16..20]  [20..24]  [24..28]  [28..32]
//!   facing:        bot    top     top      bot       bot       top       top       bot
//! ```
//!
//! ## Colour scheme
//!
//! Top arms = red, bottom arms = blue. Arming bumps both intensities to
//! "full" so the transition reads as a perceived menacing intensity jump
//! rather than an on/off blink — pilot can always see the airframe.
//!
//! | `ARM_LED_ENABLED` | `IS_ARMED` | top         | bottom       |
//! |-------------------|-----------|-------------|--------------|
//! | false             | ×         | off         | off          |
//! | true              | false     | `RED_DIM`   | `BLUE_DIM`   |
//! | true              | true      | `RED`       | `BLUE`       |
//!
//! ## Power
//!
//! Red and blue are both single-channel WS2812B colours, so per-LED current
//! matches between them (~20 mA at full). Worst case (armed full) ≈ 32 × 20
//! mA ≈ 0.64 A at 5 V; disarmed dim ≈ 32 × 20 mA × (30/255) ≈ 0.075 A — low
//! enough to leave the strip on indefinitely on the bench without loading
//! the BEC.

use core::sync::atomic::{AtomicBool, Ordering};

use embassy_futures::select::select;
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, signal::Signal};
use embassy_time::{with_timeout, Duration, Timer};

pub mod ws2812;

use ws2812::{Rgb, Ws2812, NUM_LEDS};

const LEDS_PER_GROUP: usize = 4;

/// Per-group facing along the daisy-chain. `true` = top of arm (red),
/// `false` = bottom of arm (blue). See the layout diagram in the module
/// docstring; reorder this if the strip wiring changes.
const GROUP_FACING_TOP: [bool; 8] = [
    false, // [0..4]   bottom of arm 0
    true,  // [4..8]   top    of arm 0
    true,  // [8..12]  top    of arm 1
    false, // [12..16] bottom of arm 1
    false, // [16..20] bottom of arm 2
    true,  // [20..24] top    of arm 2
    true,  // [24..28] top    of arm 3
    false, // [28..32] bottom of arm 3
];

// Compile-time guard: chain length must equal groups × leds-per-group.
const _: () = assert!(NUM_LEDS == GROUP_FACING_TOP.len() * LEDS_PER_GROUP);

/// Latest user-requested enable state. Mirrors `VehicleParams.arm_led_enabled`
/// for the LED task's hot path; written by board init at boot and by the
/// `led on` / `led off` shell verbs.
pub static ARM_LED_ENABLED: AtomicBool = AtomicBool::new(false);

/// Pulsed by the shell on toggle so the task wakes immediately instead of
/// waiting up to `POLL_INTERVAL` for the next periodic tick.
pub static ARM_LED_REFRESH: Signal<CriticalSectionRawMutex, ()> = Signal::new();

/// Poll cadence for `IS_ARMED` and the periodic re-emit. 100 ms gives a
/// snappy visual response to arming and lets the strip re-latch once per
/// tick as defensive glitch recovery. DMA cost ≈ 1 ms per emit at 32 LEDs,
/// so ~1% of the thread executor — negligible.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Bound on the very first DMA write. If something is mis-configured the
/// transfer never completes and the task would block forever, breaking
/// safety_protocol rule 6 (startup must not block). On timeout we drop the
/// frame and continue — silence-on-failure.
const FIRST_WRITE_TIMEOUT: Duration = Duration::from_millis(5);

/// Bound on every subsequent write in the main loop. Defensive cap so a
/// stalled DMA (e.g. transfer error never resolved) cannot wedge the task
/// indefinitely. Real transmission is ~1.25 ms (32-LED burst + tail wait);
/// 50 ms is two orders of magnitude over and still well below the 100 ms
/// poll cadence so it never gates the loop in the happy path. The LED
/// task hanging is not safety-relevant (it runs on the thread executor,
/// separate from the control chain), but freezing the indicator until
/// reboot is bad UX.
const WRITE_TIMEOUT: Duration = Duration::from_millis(50);

/// Build the strip frame: each group of 4 takes the colour for its facing
/// (red on top, blue on bottom); armed bumps both to full intensity.
fn compose_frame(enabled: bool, armed: bool) -> [Rgb; NUM_LEDS] {
    if !enabled {
        return [Rgb::OFF; NUM_LEDS];
    }
    let (top, bottom) = if armed {
        (Rgb::RED, Rgb::BLUE)
    } else {
        (Rgb::RED_DIM, Rgb::BLUE_DIM)
    };
    let mut out = [Rgb::OFF; NUM_LEDS];
    for (group, &is_top) in GROUP_FACING_TOP.iter().enumerate() {
        let base = group * LEDS_PER_GROUP;
        let colour = if is_top { top } else { bottom };
        for i in 0..LEDS_PER_GROUP {
            out[base + i] = colour;
        }
    }
    out
}

#[embassy_executor::task]
pub async fn task(mut strip: Ws2812) {
    // Push the initial state. Bound the first write so a mis-configured
    // DMA cannot wedge the task at startup.
    let enabled = ARM_LED_ENABLED.load(Ordering::Relaxed);
    let armed = crate::motors::IS_ARMED.load(Ordering::Acquire);
    let frame = compose_frame(enabled, armed);
    let _ = with_timeout(FIRST_WRITE_TIMEOUT, strip.write(&frame)).await;

    loop {
        // Wake on shell toggle or on the periodic poll. We re-emit every
        // tick regardless of state-change so a single corrupted frame
        // (signal glitch on the long chain) self-heals within 100 ms.
        let _ = select(ARM_LED_REFRESH.wait(), Timer::after(POLL_INTERVAL)).await;

        let enabled = ARM_LED_ENABLED.load(Ordering::Relaxed);
        let armed = crate::motors::IS_ARMED.load(Ordering::Acquire);
        let frame = compose_frame(enabled, armed);
        // Time-bounded so a stalled DMA cannot freeze the task. On timeout
        // we drop the frame; the next tick re-attempts.
        let _ = with_timeout(WRITE_TIMEOUT, strip.write(&frame)).await;
    }
}
