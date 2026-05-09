//! GPS sensor task — reads NAV-PVT frames from a u-blox receiver
//! (M8/M9/F9P) and fans them out to both `GPS_FIX` (trimmed, for
//! telemetry) and `GPS_NAV_PVT` (full NAV-PVT signal used by the ESKF
//! GPS path).

use core::cell::Cell;
use core::sync::atomic::{AtomicU32, Ordering};

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
    /// Build does not include `est_pos_gps`; the GPS task is not spawned
    /// and no UART has been opened. Reported by `gpshealth` so a mocap-only
    /// build doesn't look like the receiver is dead — it isn't there at all.
    NotConfigured,
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
        /// NAV-PVT flags bit 1: differential corrections applied.
        diff_soln: bool,
        /// NAV-PVT flags bits 6-7: 0=none, 1=RTK float, 2=RTK fixed.
        /// Always 0 on receivers without RTK hardware (M8/M9 non-P).
        carr_soln: u8,
        last_fix_at: Instant,
    },
    /// `read_fix` returned an error after init succeeded.
    ReadError(GpsErrKind),
}

/// Human-readable label for the carrier-phase solution status reported in
/// NAV-PVT flags bits 6-7.
pub fn carr_soln_str(c: u8) -> &'static str {
    match c {
        0 => "none",
        1 => "float",
        2 => "FIXED",
        _ => "?",
    }
}

pub static GPS_HEALTH: Mutex<CriticalSectionRawMutex, Cell<GpsHealth>> =
    Mutex::new(Cell::new(GpsHealth::NotConfigured));

/// Latest NAV-PVT snapshot, writeable from the GPS task and readable from
/// any shell context. Held as `Option` so the shell can distinguish "no
/// fix yet" from "stale fix"; pair the snapshot's `timestamp` with
/// `Instant::now()` for age. Post-flight inter-PVT-interval analysis
/// reads the `timestamp` field of successive `/gps_health` records and
/// diffs the unique values — no firmware-side jitter computation is
/// needed, the raw arrival time is the source-of-truth.
pub static LATEST_NAV_PVT: Mutex<CriticalSectionRawMutex, Cell<Option<GpsNavPvt>>> =
    Mutex::new(Cell::new(None));

/// Sliding-window count for the GPS jitter health metric. At 2 Hz this
/// is ~8 s of history; at 5 Hz, ~3 s. Pick a duration in the same
/// ballpark as the failsafe response time you'd want — large enough
/// that one bad PVT doesn't dominate, small enough that the metric
/// recovers when the receiver does.
pub const NAV_PVT_JITTER_WINDOW: usize = 16;

/// Worst-case (largest) inter-arrival interval among the most recent
/// `NAV_PVT_JITTER_WINDOW` NAV-PVTs, in microseconds.
///
/// Sensitive to dropouts: a single 1 s gap among fifteen 200 ms gaps
/// jumps this from 200 000 to 1 000 000. Use as the threshold input
/// for "did the receiver miss a frame?" alarms.
///
/// `u32::MAX` until at least two PVTs have been parsed since boot.
/// Updated atomically by the GPS task on each successful PVT.
pub static NAV_PVT_MAX_INTERVAL_RECENT_US: AtomicU32 = AtomicU32::new(u32::MAX);

/// Mean inter-arrival interval over the most recent
/// `NAV_PVT_JITTER_WINDOW` NAV-PVTs, in microseconds. Equivalent to
/// `(newest - oldest) / (n - 1)` over the window.
///
/// Sibling to [`NAV_PVT_MAX_INTERVAL_RECENT_US`]. Where the max
/// captures dropouts, this captures sustained rate — a receiver
/// that drops from 5 Hz → 2 Hz steadily moves this from 200 000 →
/// 500 000, while the max only spikes during the transition. ROS
/// `topic hz`-style "typical rate" view: a TUI computes
/// `Hz = 1_000_000 / value` for display.
///
/// `u32::MAX` until at least two PVTs have been parsed since boot.
pub static NAV_PVT_MEAN_INTERVAL_RECENT_US: AtomicU32 = AtomicU32::new(u32::MAX);

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
            GpsHealth::NotConfigured => {
                write!(f, "GPS: not configured (build has no `est_pos_gps` feature)")
            }
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
                diff_soln,
                carr_soln,
                last_fix_at,
            } => {
                let age_ms = Instant::now().duration_since(*last_fix_at).as_millis();
                write!(
                    f,
                    "GPS: locked (fix_type={}, num_sv={}, h_acc={}mm, dgps={}, rtk={}, age={}ms)",
                    fix_type,
                    num_sv,
                    h_acc_mm,
                    if *diff_soln { "yes" } else { "no" },
                    carr_soln_str(*carr_soln),
                    age_ms
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
    /// NAV-PVT flags bit 0: receiver believes the fix is valid.
    pub gnss_fix_ok: bool,
    /// NAV-PVT flags bit 1: differential corrections (e.g. RTCM3) were applied.
    pub diff_soln: bool,
    /// NAV-PVT flags bits 6-7: RTK carrier-phase solution status.
    /// 0=none, 1=float, 2=fixed.
    pub carr_soln: u8,
}

pub struct GpsRunner {
    gps: Ublox<GpsUart>,
    /// Ring of the most recent NAV-PVT arrival timestamps in
    /// microseconds since boot. `0` marks an empty slot. Used to
    /// compute the [`NAV_PVT_MAX_INTERVAL_RECENT_US`] sliding-window
    /// metric — order doesn't matter for that, only the set of
    /// values, so we don't track a head index.
    arrivals_us: [u64; NAV_PVT_JITTER_WINDOW],
    /// Next slot to overwrite. Wraps over the ring.
    arrivals_idx: usize,
    /// How many slots have been written. Saturates at the window
    /// size; once saturated the ring is full and every slot is valid.
    arrivals_count: usize,
}

impl GpsRunner {
    pub fn new(gps: Ublox<GpsUart>) -> Self {
        Self {
            gps,
            arrivals_us: [0; NAV_PVT_JITTER_WINDOW],
            arrivals_idx: 0,
            arrivals_count: 0,
        }
    }

    /// Push the current PVT arrival into the ring and republish the
    /// max-recent-interval atomic. Order of arrivals is preserved by
    /// monotonic [`Instant`]; sorting the ring before differencing
    /// gives us the gap distribution regardless of write position.
    fn update_jitter_window(&mut self, now_us: u64) {
        self.arrivals_us[self.arrivals_idx] = now_us;
        self.arrivals_idx = (self.arrivals_idx + 1) % NAV_PVT_JITTER_WINDOW;
        if self.arrivals_count < NAV_PVT_JITTER_WINDOW {
            self.arrivals_count += 1;
        }

        if self.arrivals_count < 2 {
            return;
        }

        // Copy out the valid prefix, sort ascending, and reduce the
        // max gap. N ≤ 16 so an in-place sort is cheaper than the
        // bookkeeping of a sliding-max deque.
        let mut sorted = [0u64; NAV_PVT_JITTER_WINDOW];
        sorted[..self.arrivals_count].copy_from_slice(&self.arrivals_us[..self.arrivals_count]);
        sorted[..self.arrivals_count].sort_unstable();

        let mut max_gap_us: u64 = 0;
        for w in sorted[..self.arrivals_count].windows(2) {
            let gap = w[1].saturating_sub(w[0]);
            if gap > max_gap_us {
                max_gap_us = gap;
            }
        }
        let max_clamped = max_gap_us.min(u32::MAX as u64) as u32;
        NAV_PVT_MAX_INTERVAL_RECENT_US.store(max_clamped, Ordering::Relaxed);

        // Mean = total span / (n - 1). The sort doesn't change the
        // span (oldest, newest are extremes either way), but using
        // the sorted view keeps the indices unambiguous.
        let span_us = sorted[self.arrivals_count - 1].saturating_sub(sorted[0]);
        let mean_period_us = span_us / (self.arrivals_count as u64 - 1);
        let mean_clamped = mean_period_us.min(u32::MAX as u64) as u32;
        NAV_PVT_MEAN_INTERVAL_RECENT_US.store(mean_clamped, Ordering::Relaxed);
    }

    pub async fn run(&mut self) -> ! {
        let publisher = super::GPS_FIX.immediate_publisher();
        defmt::info!("GPS task running — waiting for NAV-PVT frames");
        loop {
            match self.gps.read_fix().await {
                Ok(pvt) => {
                    let now = Instant::now();
                    self.update_jitter_window(now.as_micros());
                    let lat_deg = pvt.lat_1e7 as f64 * 1e-7;
                    let lon_deg = pvt.lon_1e7 as f64 * 1e-7;
                    let health = if pvt.fix_type >= 3 {
                        GpsHealth::Locked {
                            fix_type: pvt.fix_type,
                            num_sv: pvt.num_sv,
                            h_acc_mm: pvt.h_acc_mm,
                            diff_soln: pvt.diff_soln,
                            carr_soln: pvt.carr_soln,
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
                    let nav_pvt = GpsNavPvt {
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
                        gnss_fix_ok: pvt.gnss_fix_ok,
                        diff_soln: pvt.diff_soln,
                        carr_soln: pvt.carr_soln,
                    };
                    super::GPS_NAV_PVT.signal(nav_pvt);
                    LATEST_NAV_PVT.lock(|c| c.set(Some(nav_pvt)));
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
