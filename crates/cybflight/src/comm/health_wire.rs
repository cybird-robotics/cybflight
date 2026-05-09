//! Snapshot helpers for the GCS health-dashboard wire types.
//!
//! [`cybflight_msgs::wire::WireSystemHealth`] and
//! [`cybflight_msgs::wire::WireGpsHealth`] live in cybflight-msgs (the
//! cross-process schema) and have no `from_msg` constructor — there's
//! no in-process channel for them; they're populated directly from the
//! same atomics the live `health` / `gpshealth` shell commands and the
//! blackbox `/health` / `/gps_health` topics read.
//!
//! Cadence is set by the caller (`comm::esp_bridge`) at 1 Hz off the
//! existing 100 Hz tx tick.

use core::sync::atomic::Ordering;

use cybflight_msgs::wire::{self, WireGpsHealth, WireSystemHealth};
use embassy_time::Instant;

use crate::estimation::{
    ATTITUDE_HEALTH, ESKF_DEGRADED, ESKF_FAULTS, ESKF_HEALTH, ESKF_LAST_ATT_UPDATE,
    ESKF_LAST_POS_UPDATE, ESKF_LAST_VEL_UPDATE, ESKF_SEVERE_FAULT, ESTIMATOR_READY,
};
use crate::sensors::gps::{
    GpsErrKind, GpsHealth, GPS_HEALTH, LATEST_NAV_PVT, NAV_PVT_MAX_INTERVAL_RECENT_US,
    NAV_PVT_MEAN_INTERVAL_RECENT_US,
};

fn encode_err(k: GpsErrKind) -> u8 {
    match k {
        GpsErrKind::Io => wire::GPS_ERR_IO,
        GpsErrKind::BadChecksum => wire::GPS_ERR_BAD_CHECKSUM,
        GpsErrKind::Nak => wire::GPS_ERR_NAK,
        GpsErrKind::Timeout => wire::GPS_ERR_TIMEOUT,
    }
}

/// Sample every health atomic and pack into the wire frame.
///
/// `now` is the wall-clock the record carries — pass the same
/// `Instant` for both `snapshot_system` and `snapshot_gps` if the
/// caller wants the two records to share a timestamp.
pub fn snapshot_system(now: Instant) -> WireSystemHealth {
    let h = ESKF_HEALTH.lock(|c| c.get());
    let last_pos = ESKF_LAST_POS_UPDATE.lock(|c| c.get());
    let last_vel = ESKF_LAST_VEL_UPDATE.lock(|c| c.get());
    let last_att = ESKF_LAST_ATT_UPDATE.lock(|c| c.get());
    let age = |t: Option<Instant>| -> u32 {
        t.map(|t| {
            now.saturating_duration_since(t)
                .as_millis()
                .min(u32::MAX as u64) as u32
        })
        .unwrap_or(u32::MAX)
    };
    WireSystemHealth {
        timestamp_us: now.as_micros(),
        estimator_faults: ESKF_FAULTS.load(Ordering::Relaxed),
        nan_resets: h.nan_resets as u32,
        gate_rejects_pos: h.gate_rejects_pos as u32,
        gate_rejects_vel: h.gate_rejects_vel as u32,
        gate_rejects_att: h.gate_rejects_att as u32,
        last_pos_update_age_ms: age(last_pos),
        last_vel_update_age_ms: age(last_vel),
        last_att_update_age_ms: age(last_att),
        last_nis_pos: h.last_nis_pos,
        last_nis_vel: h.last_nis_vel,
        last_nis_att: h.last_nis_att,
        failsafe_active: u8::from(
            crate::control::failsafe::FAILSAFE_ACTIVE.load(Ordering::Acquire),
        ),
        estimator_ready: u8::from(ESTIMATOR_READY.load(Ordering::Acquire)),
        eskf_degraded: u8::from(ESKF_DEGRADED.load(Ordering::Acquire)),
        eskf_severe_fault: u8::from(ESKF_SEVERE_FAULT.load(Ordering::Acquire)),
        attitude_health: ATTITUDE_HEALTH.load(Ordering::Relaxed),
        _pad: [0; 3],
    }
}

pub fn snapshot_gps(now: Instant) -> WireGpsHealth {
    let mut s = WireGpsHealth {
        timestamp_us: now.as_micros(),
        nav_pvt_arrival_us: 0,
        h_acc_mm: 0,
        v_acc_mm: 0,
        s_acc_mm_s: 0,
        nav_pvt_max_interval_recent_us: u32::MAX,
        nav_pvt_mean_interval_recent_us: u32::MAX,
        state: wire::GPS_STATE_NOT_CONFIGURED,
        err_kind: wire::GPS_ERR_NONE,
        fix_type: 0,
        num_sv: 0,
        diff_soln: 0,
        carr_soln: 0,
        gnss_fix_ok: 0,
        _pad: 0,
    };

    let h = GPS_HEALTH.lock(|c| c.get());
    match h {
        GpsHealth::NotConfigured => s.state = wire::GPS_STATE_NOT_CONFIGURED,
        GpsHealth::Initializing => s.state = wire::GPS_STATE_INITIALIZING,
        GpsHealth::UartInitFailed => s.state = wire::GPS_STATE_UART_INIT_FAILED,
        GpsHealth::InitTimedOut => s.state = wire::GPS_STATE_INIT_TIMED_OUT,
        GpsHealth::InitFailed(k) => {
            s.state = wire::GPS_STATE_INIT_FAILED;
            s.err_kind = encode_err(k);
        }
        GpsHealth::Waiting {
            fix_type,
            num_sv,
            h_acc_mm,
        } => {
            s.state = wire::GPS_STATE_WAITING;
            s.fix_type = fix_type;
            s.num_sv = num_sv;
            s.h_acc_mm = h_acc_mm;
        }
        GpsHealth::Locked {
            fix_type,
            num_sv,
            h_acc_mm,
            diff_soln,
            carr_soln,
            last_fix_at: _,
        } => {
            s.state = wire::GPS_STATE_LOCKED;
            s.fix_type = fix_type;
            s.num_sv = num_sv;
            s.h_acc_mm = h_acc_mm;
            s.diff_soln = u8::from(diff_soln);
            s.carr_soln = carr_soln;
        }
        GpsHealth::ReadError(k) => {
            s.state = wire::GPS_STATE_READ_ERROR;
            s.err_kind = encode_err(k);
        }
    }

    if let Some(p) = LATEST_NAV_PVT.lock(|c| c.get()) {
        s.v_acc_mm = p.v_acc_mm;
        s.s_acc_mm_s = p.s_acc_mm_s;
        s.gnss_fix_ok = u8::from(p.gnss_fix_ok);
        s.nav_pvt_arrival_us = p.timestamp.as_micros();
    }

    s.nav_pvt_max_interval_recent_us = NAV_PVT_MAX_INTERVAL_RECENT_US.load(Ordering::Relaxed);
    s.nav_pvt_mean_interval_recent_us = NAV_PVT_MEAN_INTERVAL_RECENT_US.load(Ordering::Relaxed);

    s
}
