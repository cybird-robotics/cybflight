//! DShot600 bidirectional protocol layer.
//!
//! Pure protocol logic: frame encoding, GCR telemetry decode, and eRPM/EDT
//! interpretation. Hardware-agnostic and fully host-testable.

pub mod frame;
pub mod gcr;
pub mod telemetry;

// ---------------------------------------------------------------------------
// DShot command values (from dshot_command.h)
// ---------------------------------------------------------------------------

pub const DSHOT_CMD_MOTOR_STOP: u16 = 0;
pub const DSHOT_CMD_BEACON1: u16 = 1;
pub const DSHOT_CMD_BEACON2: u16 = 2;
pub const DSHOT_CMD_BEACON3: u16 = 3;
pub const DSHOT_CMD_BEACON4: u16 = 4;
pub const DSHOT_CMD_BEACON5: u16 = 5;
pub const DSHOT_CMD_ESC_INFO: u16 = 6;
pub const DSHOT_CMD_SPIN_DIRECTION_1: u16 = 7;
pub const DSHOT_CMD_SPIN_DIRECTION_2: u16 = 8;
pub const DSHOT_CMD_3D_MODE_OFF: u16 = 9;
pub const DSHOT_CMD_3D_MODE_ON: u16 = 10;
pub const DSHOT_CMD_SETTINGS_REQUEST: u16 = 11;
pub const DSHOT_CMD_SAVE_SETTINGS: u16 = 12;
pub const DSHOT_CMD_EXTENDED_TELEMETRY_ENABLE: u16 = 13;
pub const DSHOT_CMD_EXTENDED_TELEMETRY_DISABLE: u16 = 14;
pub const DSHOT_CMD_SPIN_DIRECTION_NORMAL: u16 = 20;
pub const DSHOT_CMD_SPIN_DIRECTION_REVERSED: u16 = 21;
pub const DSHOT_CMD_LED0_ON: u16 = 22;
pub const DSHOT_CMD_LED1_ON: u16 = 23;
pub const DSHOT_CMD_LED2_ON: u16 = 24;
pub const DSHOT_CMD_LED3_ON: u16 = 25;
pub const DSHOT_CMD_LED0_OFF: u16 = 26;
pub const DSHOT_CMD_LED1_OFF: u16 = 27;
pub const DSHOT_CMD_LED2_OFF: u16 = 28;
pub const DSHOT_CMD_LED3_OFF: u16 = 29;
pub const DSHOT_CMD_AUDIO_STREAM_MODE_ON_OFF: u16 = 30;
pub const DSHOT_CMD_SILENT_MODE_ON_OFF: u16 = 31;
pub const DSHOT_CMD_MAX: u16 = 47;

// ---------------------------------------------------------------------------
// Throttle range
// ---------------------------------------------------------------------------

pub const DSHOT_MIN_THROTTLE: u16 = 48;
pub const DSHOT_MAX_THROTTLE: u16 = 2047;

// ---------------------------------------------------------------------------
// Buffer sizes
// ---------------------------------------------------------------------------

/// DMA buffer: 16 data bits + 2 trailing reset words.
pub const DSHOT_DMA_BUFFER_SIZE: usize = 18;

/// Minimum number of edges in a valid GCR telemetry packet.
pub const MIN_GCR_EDGES: usize = 7;

/// Maximum number of edges in a valid GCR telemetry packet.
pub const MAX_GCR_EDGES: usize = 22;

/// Sentinel for invalid telemetry.
pub const DSHOT_TELEMETRY_INVALID: u32 = 0xFFFF;

/// GCR ticks per bit at DShot600 (12 MHz clock, 750 kbit/s = 5/4 × 600 kbit/s).
/// 12_000_000 / 750_000 = 16 ticks.
pub const DSHOT600_GCR_TICKS_PER_BIT: u32 = 16;
