//! Per-tick controller tracking-error message.
//!
//! Lives in the firmware crate (rather than `cybflight-msgs`) because
//! every consumer is in-process: the three publishers (cascade /
//! MPC / INDI) and the single subscriber (blackbox recorder). No
//! off-board consumer reads it — esp_bridge doesn't surface it,
//! shell doesn't query it. If that ever changes, promote the type
//! into `cybflight-msgs` and add a wire-format encoder; the channel
//! shape doesn't need to change.
//!
//! Each publisher fills only the fields it computes; the rest stay
//! zero. Distinguished by [`source`](TrackingError::source) — see
//! the constants below.

use embassy_time::Instant;
use nalgebra::Vector3;

#[derive(Clone, defmt::Format)]
pub struct TrackingError {
    pub timestamp: Instant,
    /// World-frame `reference - actual` position [m]. Raw (cascade)
    /// or raw against τ₀ reference (MPC). Zero from INDI.
    pub pos_err: Vector3<f32>,
    /// World-frame `reference - actual` velocity [m/s]. Raw. Zero
    /// from INDI.
    pub vel_err: Vector3<f32>,
    /// Body-frame tilt-prio 3-vec from
    /// `cybflight_core::mpc::model_utils::attitude_error`. Populated
    /// only by the MPC source — the cascade attitude error lands
    /// downstream as INDI's `body_rate_err`, and INDI itself sees no
    /// attitude reference.
    pub attitude_err: Vector3<f32>,
    /// Body-frame `rate_ref - gyro_corrected` [rad/s]. Populated
    /// only by the INDI source (`gyro_corrected` here is the
    /// post-RPM-notch shadow).
    pub body_rate_err: Vector3<f32>,
    /// One of [`TRACKING_ERROR_SOURCE_*`](TRACKING_ERROR_SOURCE_CASCADE).
    /// Tells the consumer which fields are meaningful; others are
    /// zero by construction.
    pub source: u8,
}

pub const TRACKING_ERROR_SOURCE_CASCADE: u8 = 0;
pub const TRACKING_ERROR_SOURCE_MPC: u8 = 1;
pub const TRACKING_ERROR_SOURCE_INDI: u8 = 2;
