pub mod dshot;

use crate::hal::pac::gpio::Gpio;
use crate::hal::pac::timer::TimGp16;
use crate::msgs;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;

/// Per-motor throttle commands. Written by the shell, read by dshot_task every loop.
/// Default: DSHOT_MIN_THROTTLE (arm/idle). 0 = MOTOR_STOP command.
pub static ACTUATOR_MOTORS: Signal<CriticalSectionRawMutex, msgs::ActuatorMotors> = Signal::new();

/// Arm/disarm state from RC input. DShot task sends MOTOR_STOP when disarmed.
pub static ARM_STATE: Signal<CriticalSectionRawMutex, msgs::ArmDisarm> = Signal::new();

// DShot600 bit timing (at 12 MHz effective timer clock, 20 ticks/bit)
pub const DSHOT600_BIT_0: u32 = 7; // 35% duty
pub const DSHOT600_BIT_1: u32 = 14; // 70% duty
pub const DSHOT600_PSC: u16 = 19; // 240 MHz / 20 = 12 MHz
pub const DSHOT600_ARR: u16 = 19; // 20 ticks per bit

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
