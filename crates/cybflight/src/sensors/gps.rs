//! GPS sensor task — reads NAV-PVT frames from a u-blox receiver
//! (M8/M9/F9P) and fans them out to both `GPS_FIX` (trimmed, for
//! telemetry) and `GPS_NAV_PVT` (full NAV-PVT signal used by the ESKF
//! GPS path).

use core::cell::Cell;

use cybflight_drivers::gps::Ublox;
use cybflight_drivers::gps::ublox::Error as UbxError;
use embassy_sync::blocking_mutex::{Mutex, raw::CriticalSectionRawMutex};
use embassy_time::Instant;

use crate::hal;
use cybflight_msgs as msgs;

pub type GpsUart = hal::usart::BufferedUart<'static>;

/// Tag for the four error variants of `cybflight_drivers::gps::ublox::Error`.
/// Stripped of the `Io(E)` payload so the snapshot stays `Copy` and lock-free.
#[derive(Clone, Copy)]
pub enum GpsErrKind {
    Io,
    BadChecksum,
    Nak,
    Timeout,
}

/// Operator-facing GPS health snapshot. Updated by `board_init` during the
/// init handshake and by `GpsRunner::run` thereafter. Read by the shell's
/// `gpshealth` command.
///
/// All variants are `Copy` so the value lives in a `Cell` and can be set
/// from any task without allocation.
#[derive(Clone, Copy)]
pub enum GpsHealth {
    /// Initial state — UART not yet opened, or init in progress.
    Initializing,
    /// `BufferedUart::new` failed (pin/peripheral conflict, bad config).
    UartInitFailed,
    /// Outer 3-second `with_timeout` tripped — no bytes from the module.
    /// Most common cause is a baud-rate mismatch.
    InitTimedOut,
    /// `Ublox::new` returned an error after receiving some bytes.
    InitFailed(GpsErrKind),
    /// Module is responding with NAV-PVT frames but no valid fix yet.
    Waiting {
        fix_type: u8,
        num_sv: u8,
        h_acc_mm: u32,
    },
    /// Module has a 3D fix; `last_fix_at` lets the shell display age.
    Locked {
        fix_type: u8,
        num_sv: u8,
        h_acc_mm: u32,
        last_fix_at: Instant,
    },
    /// `read_fix` returned an error after init succeeded.
    ReadError(GpsErrKind),
}

pub static GPS_HEALTH: Mutex<CriticalSectionRawMutex, Cell<GpsHealth>> =
    Mutex::new(Cell::new(GpsHealth::Initializing));

/// Convert a u-blox driver error into the variant tag stored in `GpsHealth`.
pub fn err_kind<E>(e: &UbxError<E>) -> GpsErrKind {
    match e {
        UbxError::Io(_) => GpsErrKind::Io,
        UbxError::BadChecksum => GpsErrKind::BadChecksum,
        UbxError::Nak => GpsErrKind::Nak,
        UbxError::Timeout => GpsErrKind::Timeout,
    }
}

fn err_kind_str(k: GpsErrKind) -> &'static str {
    match k {
        GpsErrKind::Io => "io",
        GpsErrKind::BadChecksum => "bad checksum",
        GpsErrKind::Nak => "NAK",
        GpsErrKind::Timeout => "timeout (no ACK)",
    }
}

impl core::fmt::Display for GpsHealth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpsHealth::Initializing => write!(f, "GPS: initializing"),
            GpsHealth::UartInitFailed => write!(f, "GPS: UART init failed"),
            GpsHealth::InitTimedOut => write!(
                f,
                "GPS: init timed out (no response from module — check baud/wiring/power)"
            ),
            GpsHealth::InitFailed(k) => write!(f, "GPS: init failed ({})", err_kind_str(*k)),
            GpsHealth::Waiting {
                fix_type,
                num_sv,
                h_acc_mm,
            } => write!(
                f,
                "GPS: waiting for lock (fix_type={}, num_sv={}, h_acc={}mm)",
                fix_type, num_sv, h_acc_mm
            ),
            GpsHealth::Locked {
                fix_type,
                num_sv,
                h_acc_mm,
                last_fix_at,
            } => {
                let age_ms = Instant::now().duration_since(*last_fix_at).as_millis();
                write!(
                    f,
                    "GPS: locked (fix_type={}, num_sv={}, h_acc={}mm, age={}ms)",
                    fix_type, num_sv, h_acc_mm, age_ms
                )
            }
            GpsHealth::ReadError(k) => write!(f, "GPS: read error ({})", err_kind_str(*k)),
        }
    }
}

/// Full NAV-PVT fix used by `eskf_imu_gps` for position + velocity updates.
/// Kept local until `cybflight-msgs::GpsFix` grows NED velocity and
/// speed-accuracy fields; published to `super::GPS_NAV_PVT` as a `Signal`
/// (latest-wins, 5 Hz cadence — no queuing needed).
#[derive(Clone, Copy)]
pub struct GpsNavPvt {
    pub timestamp: Instant,
    pub fix_type: u8,
    pub num_sv: u8,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_msl_mm: i32,
    pub vel_north_mm_s: i32,
    pub vel_east_mm_s: i32,
    pub vel_down_mm_s: i32,
    pub h_acc_mm: u32,
    pub v_acc_mm: u32,
    pub s_acc_mm_s: u32,
}

pub struct GpsRunner {
    gps: Ublox<GpsUart>,
}

impl GpsRunner {
    pub fn new(gps: Ublox<GpsUart>) -> Self {
        Self { gps }
    }

    pub async fn run(&mut self) -> ! {
        let publisher = super::GPS_FIX.immediate_publisher();
        defmt::info!("GPS task running — waiting for NAV-PVT frames");
        loop {
            match self.gps.read_fix().await {
                Ok(pvt) => {
                    let now = Instant::now();
                    let lat_deg = pvt.lat_1e7 as f64 * 1e-7;
                    let lon_deg = pvt.lon_1e7 as f64 * 1e-7;
                    let health = if pvt.fix_type >= 3 {
                        GpsHealth::Locked {
                            fix_type: pvt.fix_type,
                            num_sv: pvt.num_sv,
                            h_acc_mm: pvt.h_acc_mm,
                            last_fix_at: now,
                        }
                    } else {
                        GpsHealth::Waiting {
                            fix_type: pvt.fix_type,
                            num_sv: pvt.num_sv,
                            h_acc_mm: pvt.h_acc_mm,
                        }
                    };
                    GPS_HEALTH.lock(|c| c.set(health));
                    publisher.publish_immediate(msgs::GpsFix {
                        timestamp: now,
                        lat_deg,
                        lon_deg,
                        alt_msl_mm: pvt.alt_msl_mm,
                        ground_speed_mm_s: pvt.ground_speed_mm_s,
                        heading_mot_1e5: pvt.heading_mot_1e5,
                        fix_type: pvt.fix_type,
                        num_sv: pvt.num_sv,
                        h_acc_mm: pvt.h_acc_mm,
                        v_acc_mm: pvt.v_acc_mm,
                        pdop: pvt.pdop,
                    });
                    super::GPS_NAV_PVT.signal(GpsNavPvt {
                        timestamp: now,
                        fix_type: pvt.fix_type,
                        num_sv: pvt.num_sv,
                        lat_deg,
                        lon_deg,
                        alt_msl_mm: pvt.alt_msl_mm,
                        vel_north_mm_s: pvt.vel_north_mm_s,
                        vel_east_mm_s: pvt.vel_east_mm_s,
                        vel_down_mm_s: pvt.vel_down_mm_s,
                        h_acc_mm: pvt.h_acc_mm,
                        v_acc_mm: pvt.v_acc_mm,
                        s_acc_mm_s: pvt.s_acc_mm_s,
                    });
                }
                Err(e) => {
                    GPS_HEALTH.lock(|c| c.set(GpsHealth::ReadError(err_kind(&e))));
                    defmt::warn!("GPS read error: {}", e);
                }
            }
        }
    }
}

#[embassy_executor::task]
pub async fn ublox_gps_task(mut runner: GpsRunner) {
    runner.run().await;
}
