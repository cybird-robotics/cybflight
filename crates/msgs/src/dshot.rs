//! DShot telemetry value types.
//!
//! Pure data types shared between `cybflight-drivers` (which decodes them)
//! and `cybflight` (which publishes them as messages). No HAL dependencies.

/// Decoded telemetry value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), derive(defmt::Format))]
pub enum TelemetryValue {
    /// Motor eRPM × 100.
    Erpm(u32),
    /// Motor stopped (raw 0x0FFF).
    Stopped,
    /// Extended DShot Telemetry value.
    Edt(EdtValue),
    /// Invalid telemetry (period == 0).
    Invalid,
}

/// Extended DShot Telemetry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), derive(defmt::Format))]
pub enum EdtType {
    Temperature,
    Voltage,
    Current,
    Debug1,
    Debug2,
    Debug3,
    StateEvents,
}

/// Extended DShot Telemetry value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), derive(defmt::Format))]
pub struct EdtValue {
    pub edt_type: EdtType,
    pub data: u8,
}
