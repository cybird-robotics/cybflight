pub mod dshot;

use crate::hal::pac::gpio::Gpio;
use crate::hal::pac::timer::TimGp16;
use crate::msgs;
use core::sync::atomic::AtomicBool;

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;

/// Per-motor throttle commands. Written by the shell, read by dshot_task every loop.
/// Default: DSHOT_MIN_THROTTLE (arm/idle). 0 = MOTOR_STOP command.
pub static ACTUATOR_MOTORS: Signal<CriticalSectionRawMutex, msgs::ActuatorMotors> = Signal::new();

/// Arm/disarm command signal. Written by RC input + failsafe, read by DShot task.
/// Signal semantics: latest value wins (no queueing).
pub static ARM_STATE: Signal<CriticalSectionRawMutex, msgs::ArmDisarm> = Signal::new();

/// Current armed state as a simple atomic bool.
/// Set by DShot task (the primary ARM_STATE consumer) after processing each
/// arm/disarm command. Read by INDI task and any other consumer that needs
/// the current arming state without consuming the Signal.
pub static IS_ARMED: AtomicBool = AtomicBool::new(false);

// DShot600 bit timing. DShot600 is 600 kbit/s and the driver clocks each
// bit as 20 timer ticks, so the timer must run at 12 MHz.
//
// The prescaler was a bare `19` with `240 MHz / 20` in a comment, which
// is right only for the RCC configuration all three current boards
// happen to share. It is now derived from the BSP's declared timer
// clock, so a board with a different one gets a correct prescaler or a
// build error rather than silently wrong bit timing.
pub const DSHOT600_BIT_0: u32 = 7; // 35% duty
pub const DSHOT600_BIT_1: u32 = 14; // 70% duty

/// Bit rate of DShot600, in Hz.
pub const DSHOT600_BIT_HZ: u32 = 600_000;
/// Timer ticks per DShot bit; sets the duty resolution above.
pub const DSHOT600_TICKS_PER_BIT: u32 = 20;
/// Timer clock the bit timings need, in Hz.
const DSHOT600_TIMER_HZ: u32 = DSHOT600_BIT_HZ * DSHOT600_TICKS_PER_BIT;

pub const DSHOT600_PSC: u16 = (crate::bsp::APB2_TIMER_HZ / DSHOT600_TIMER_HZ - 1) as u16;
pub const DSHOT600_ARR: u16 = (DSHOT600_TICKS_PER_BIT - 1) as u16;

// The prescaler is integer, so the division must be exact or the bit
// period is off and no ESC decodes the frame.
const _: () = assert!(
    crate::bsp::APB2_TIMER_HZ % DSHOT600_TIMER_HZ == 0,
    "this board's timer kernel clock is not an integer multiple of the \
     12 MHz DShot600 needs; the prescaler cannot express the bit period."
);
const _: () = assert!(DSHOT600_BIT_0 < DSHOT600_BIT_1);
const _: () = assert!(DSHOT600_BIT_1 < DSHOT600_TICKS_PER_BIT);

/// Per-motor hardware configuration (constructed by board_init).
/// All fields are Copy — just register pointers and small integers.
#[derive(Clone, Copy)]
pub struct MotorTimerConfig {
    /// Timer GP16 register block (type-erased via regs_gp16)
    pub timer_regs: TimGp16,
    /// Channel index (0=Ch1, 1=Ch2, 2=Ch3, 3=Ch4)
    pub channel_index: u8,
    /// DMAMUX request ID for this channel's CC DMA event
    pub dma_request: u8,
    /// GPIO port register block (for Phase 4 pin direction switching)
    pub gpio_port: Gpio,
    /// GPIO pin number within port (0-15)
    pub gpio_pin: u8,
    /// Alternate function number for this pin's timer channel
    pub af_number: u8,
}

/// Quad motor DShot configuration — board-agnostic.
pub struct DshotQuadConfig {
    /// Per-motor config [M1, M2, M3, M4]
    pub motors: [MotorTimerConfig; 4],
    /// Unique timer register blocks that need start/stop/reset.
    /// Only timers[0..timer_count] are used.
    pub timers: [TimGp16; 2],
    /// Number of unique timers (1 or 2)
    pub timer_count: u8,
}
