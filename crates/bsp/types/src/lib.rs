#![no_std]

/// Sensor orientation tags from Betaflight — full set of alignment variants.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(defmt::Format)]
pub enum SensorAlign {
    Default,
    Cw0Deg,
    Cw90Deg,
    Cw180Deg,
    Cw270Deg,
    Cw0DegFlip,
    Cw90DegFlip,
    Cw180DegFlip,
    Cw270DegFlip,
}

/// Default serial role mapping (Betaflight target header).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(defmt::Format)]
pub enum SerialRole {
    Msp,
    DisplayPort,
    Gps,
    EscSensor,
    SerialRx,
}

/// A concrete serial port ID — full STM32H7 superset.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(defmt::Format)]
pub enum SerialPortId {
    Usart1,
    Usart2,
    Usart3,
    Uart4,
    Uart5,
    Usart6,
    Uart7,
    Uart8,
    Lpuart1,
}

/// Timer ID — full STM32H7 timer set.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(defmt::Format)]
pub enum TimerId {
    Tim1,
    Tim2,
    Tim3,
    Tim4,
    Tim5,
    Tim8,
    Tim12,
    Tim13,
    Tim14,
    Tim15,
    Tim16,
    Tim17,
}

/// Timer channel identifier.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(defmt::Format)]
pub enum TimerChannel {
    Ch1,
    Ch2,
    Ch3,
    Ch4,
}

/// DMA stream/request hint derived from Betaflight timer tables.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(defmt::Format)]
pub struct DmaHint {
    /// Betaflight "DMA stream" number (0..7). Embassy typically exposes these as DMAx_CH0..CH7.
    pub stream: u8,
    /// Betaflight "request" number (DMAMUX request).
    pub request: u16,
}

/// Motor timer grouping metadata derived from `timer` output.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[derive(defmt::Format)]
pub struct MotorMeta {
    pub timer: TimerId,
    pub channel: TimerChannel,
    pub af: u8,
    pub dma: Option<DmaHint>,
}
