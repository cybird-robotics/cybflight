//! GNSS receiver drivers. The active driver is chosen at build time: u-blox by
//! default, or the Unicore UM982 with `--features gps_unicore`. Both expose
//! `new(rw, delay)` and `read_fix() -> ublox::NavPvt`, so the GPS task,
//! channels, health state machine, and ESKF are identical for either — see
//! [`GpsDriver`].

pub mod ublox;
pub use ublox::Ublox;

/// What a single GPS read produced. The u-blox driver only ever yields `Fix`;
/// the Unicore UM982 driver additionally yields `Heading` (dual antenna). The
/// GPS task is written against this so both receivers share one read loop.
pub enum GpsEvent {
    Fix(ublox::NavPvt),
    Heading(HeadingFields),
    /// A valid frame we don't act on (skip).
    Other,
}

/// Dual-antenna heading, in radians, ready for the ESKF vector/direction update.
#[derive(Clone, Copy, Debug)]
pub struct HeadingFields {
    /// Baseline (ANT1→ANT2) azimuth, clockwise from True North [rad].
    /// Raw baseline azimuth — set the receiver's `CONFIG HEADING OFFSET 0`.
    pub heading_rad: f32,
    /// Baseline elevation [rad].
    pub pitch_rad: f32,
    /// Heading (azimuth) 1-σ [rad].
    pub heading_sigma_rad: f32,
    /// Pitch (elevation) 1-σ [rad].
    pub pitch_sigma_rad: f32,
    /// Moving-baseline carrier solution: 0 none, 1 float, 2 integer-FIXED.
    pub carr_soln: u8,
}

#[cfg(feature = "gps_unicore")]
pub mod unicore;
#[cfg(feature = "gps_unicore")]
pub use unicore::Unicore;

/// The GPS driver selected at build time. Defaults to [`ublox::Ublox`];
/// `--features gps_unicore` swaps in [`unicore::Unicore`] (UM982). The GPS task
/// is written against this alias so it is identical for either receiver.
#[cfg(not(feature = "gps_unicore"))]
pub use ublox::Ublox as GpsDriver;
#[cfg(feature = "gps_unicore")]
pub use unicore::Unicore as GpsDriver;
