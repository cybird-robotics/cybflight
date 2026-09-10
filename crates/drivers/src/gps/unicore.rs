//! Unicore UM982 GNSS driver — a thin adapter over the `um982` core crate that
//! produces the **same** [`NavPvt`] as the u-blox driver, so the GPS task,
//! health state machine, channels, and ESKF are all shared. Selected by
//! `--features gps_unicore`.
//!
//! The receiver's configuration (mode, signal group, BESTNAV/UNIHEADING rates,
//! heading geometry) lives in flash — set once in UPrecise — so this driver
//! only reads; there is no command handshake. `new` waits for the first valid
//! frame, which confirms the link is up at the expected baud.
//!
//! Heading (`UNIHEADING`) is decoded by the core but dropped here; it is wired
//! into a dedicated channel + ESKF yaw update in a later phase.

use embedded_io_async::Read;
use um982::{Message, Um982};

use super::ublox::{Error, NavPvt};

/// Unicore UM982 driver. Reads `BESTNAV` and yields the shared [`NavPvt`].
pub struct Unicore<RW> {
    inner: Um982<RW>,
}

impl<RW> Unicore<RW>
where
    RW: Read,
{
    /// Wrap the UART and confirm the link by waiting for one valid (CRC-checked)
    /// frame. The `delay` argument is accepted for signature parity with the
    /// u-blox driver but unused (the UM982 needs no init handshake). A handful
    /// of leading CRC errors (mid-frame startup garbage) are tolerated.
    pub async fn new(
        rw: RW,
        _delay: &mut impl embedded_hal_async::delay::DelayNs,
    ) -> Result<Self, Error<RW::Error>> {
        let mut inner = Um982::new(rw);
        let mut crc_retries = 0u8;
        loop {
            match inner.read_message().await {
                Ok(_) => {
                    defmt::info!("UM982: link up (first frame received)");
                    return Ok(Self { inner });
                }
                Err(um982::Error::BadCrc) if crc_retries < 16 => crc_retries += 1,
                Err(e) => return Err(map_err(e)),
            }
        }
    }

    /// Read the next position fix. Blocks until a `BESTNAV` frame arrives,
    /// skipping `UNIHEADING`/other logs and silently dropping corrupt frames
    /// (the wired bus is trusted, matching the u-blox driver's behaviour).
    pub async fn read_fix(&mut self) -> Result<NavPvt, Error<RW::Error>> {
        loop {
            match self.inner.read_message().await {
                Ok((_hdr, Message::BestNav(b))) => return Ok(bestnav_to_navpvt(&b)),
                Ok(_) => {}                       // heading / other — skipped in phase 1
                Err(um982::Error::BadCrc) => {}   // drop corrupt frame, keep reading
                Err(e) => return Err(map_err(e)), // real I/O error — surface it
            }
        }
    }

    /// Read the next driver event — a `BESTNAV` fix or a `UNIHEADING` heading.
    /// Corrupt frames are skipped (`Other`); only real I/O errors surface.
    pub async fn read_event(&mut self) -> Result<super::GpsEvent, Error<RW::Error>> {
        match self.inner.read_message().await {
            Ok((_, Message::BestNav(b))) => Ok(super::GpsEvent::Fix(bestnav_to_navpvt(&b))),
            Ok((_, Message::Heading(h))) => Ok(super::GpsEvent::Heading(heading_to_fields(&h))),
            Ok((_, Message::Other { .. })) => Ok(super::GpsEvent::Other),
            Err(um982::Error::BadCrc) => Ok(super::GpsEvent::Other),
            Err(e) => Err(map_err(e)),
        }
    }
}

/// Map a `um982::Error` onto the shared u-blox `Error` variants, so the existing
/// `err_kind` / `GpsHealth` machinery is reused unchanged.
fn map_err<E>(e: um982::Error<E>) -> Error<E> {
    match e {
        um982::Error::Io(e) => Error::Io(e),
        um982::Error::BadCrc => Error::BadChecksum,
        um982::Error::UnexpectedEof => Error::Timeout,
    }
}

/// Convert a UM982 `BESTNAV` into the receiver-agnostic [`NavPvt`] (same units
/// and flag semantics as the u-blox NAV-PVT the GPS task already consumes).
fn bestnav_to_navpvt(b: &um982::BestNav) -> NavPvt {
    let carr_soln = if b.pos_type.is_rtk_fixed() {
        2
    } else if b.pos_type.is_rtk_float() {
        1
    } else {
        0
    };
    // BESTNAV is a geodetic 3D solution; 2D-only is not distinguished.
    let fix_type = if b.pos_sol_status.is_computed() && b.pos_type.has_position() {
        3
    } else {
        0
    };
    let gnss_fix_ok = fix_type >= 3;
    let diff_soln = b.differential_applied() || b.pos_type.is_differential() || carr_soln > 0;

    let (vn, ve, vd) = b.velocity_ned();

    NavPvt {
        fix_type,
        gnss_fix_ok,
        diff_soln,
        carr_soln,
        num_sv: b.num_sv_used,
        lon_1e7: (b.lon_deg * 1e7) as i32,
        lat_1e7: (b.lat_deg * 1e7) as i32,
        alt_msl_mm: (b.height_msl_m * 1e3) as i32,
        h_acc_mm: (b.horizontal_accuracy_m() * 1e3) as u32,
        v_acc_mm: (b.height_sigma_m * 1e3) as u32,
        vel_north_mm_s: (vn * 1e3) as i32,
        vel_east_mm_s: (ve * 1e3) as i32,
        vel_down_mm_s: (vd * 1e3) as i32,
        ground_speed_mm_s: (b.horizontal_speed_mps * 1e3) as u32,
        heading_mot_1e5: (b.track_deg * 1e5) as i32,
        s_acc_mm_s: (b.horizontal_speed_sigma_mps * 1e3) as u32,
        pdop: 0, // BESTNAV carries no DOP; add a DOP log later if telemetry needs it
    }
}

const DEG2RAD: f32 = core::f32::consts::PI / 180.0;

/// Convert a UM982 `UNIHEADING` into the receiver-agnostic [`super::HeadingFields`]
/// (radians). Assumes `CONFIG HEADING OFFSET 0`, so `heading_deg` is the raw
/// ANT1→ANT2 baseline azimuth (CW from True North) and `pitch_deg` its elevation.
fn heading_to_fields(h: &um982::Heading) -> super::HeadingFields {
    super::HeadingFields {
        heading_rad: h.heading_deg * DEG2RAD,
        pitch_rad: h.pitch_deg * DEG2RAD,
        heading_sigma_rad: h.heading_sigma_deg * DEG2RAD,
        pitch_sigma_rad: h.pitch_sigma_deg * DEG2RAD,
        carr_soln: if h.pos_type.is_rtk_fixed() {
            2
        } else if h.pos_type.is_rtk_float() {
            1
        } else {
            0
        },
    }
}
