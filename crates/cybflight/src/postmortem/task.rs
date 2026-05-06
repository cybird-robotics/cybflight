//! Steady-state post-mortem task: edge-detects health atomics,
//! appends events to the BKPSRAM ring, refreshes the
//! attitude/setpoint/RC snapshot.
//!
//! ## Why this is a separate task
//!
//! Two reasons:
//!
//! 1. The blackbox recorder already edge-detects these atomics
//!    inside its capture loop, but only when a session is active.
//!    Post-mortem cares about events that fire *between* sessions —
//!    a brownout pre-arm, a panic during ESKF warm-up, an IWDG
//!    timeout that kicks in before the first arm event. So we
//!    duplicate the edge-detection here, intentionally.
//!
//! 2. The PVD brown-out path (`fault::PVD_AVD`) has only ~5 µs to
//!    commit the fatal slot before BOR. We can't gather snapshot
//!    data during that window — so the steady-state task does it
//!    100ms-tick-by-100ms-tick, and the PVD path only flips the
//!    fatal kind + finalize.
//!
//! ## Cadence
//!
//! 100 ms tick. Edge transitions on the watched atomics are observed
//! within at most one tick. The 100 Hz topic channels (VEHICLE_ATTITUDE,
//! VEHICLE_ODOMETRY, RC_INPUT) accumulate ~10 messages per tick and
//! get Lagged on overflow — which is fine: we only need the latest
//! sample for the snapshot, not history.
//!
//! ## Hard rule (post-mortem subsystem)
//!
//! This task does take a `Mutex` indirectly via `bkpsram::with_record_mut`'s
//! `static mut` access, but only against itself — fault handlers
//! disable interrupts and run handler-context-only writes. There is
//! no contention path. The contract: this task is the **only**
//! async-context writer.

use core::sync::atomic::Ordering;

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pubsub::Subscriber;
use embassy_sync::pubsub::WaitResult;
use embassy_time::{Instant, Ticker};

use crate::control::failsafe::{FAILSAFE_ACTIVE, FAILSAFE_REASON, RC_LINK_HEALTHY};
use crate::estimation::ESTIMATOR_READY;
use crate::sensors::{RC_INPUT, VEHICLE_ATTITUDE, VEHICLE_ODOMETRY};

#[cfg(feature = "outer_mpc")]
use crate::control::MISSION_STATE;

use super::bkpsram;
use super::record::{self, EventEntry};

// Re-use the same KIND_* code space as the blackbox `/events`
// topic. Importing here so a future change to the kind values is a
// single-source-of-truth edit.
use crate::blackbox::topics::events::{
    KIND_ESTIMATOR_DOWN, KIND_ESTIMATOR_UP, KIND_FAILSAFE, KIND_FAILSAFE_CLEAR, KIND_RC_LOSS,
    KIND_RC_RECOVERED,
};
#[cfg(feature = "outer_mpc")]
use crate::blackbox::topics::events::{
    KIND_MISSION_EXECUTING, KIND_MISSION_IDLE, KIND_MISSION_PLANNING,
};

const TICK_PERIOD_MS: u64 = 100;

// `drain_latest` is generic over `Subscriber<...>`, so each call site
// uses the concrete type inferred from the channel — no per-channel
// alias needed here.

/// Steady-state post-mortem task. Spawn once on the thread executor
/// after [`super::bkpsram::enable`] and [`super::recovery::boot_recovery`]
/// have run.
#[embassy_executor::task]
pub async fn postmortem_task() {
    // Take subscriber slots. If any fail to acquire, log and exit —
    // the post-mortem record will only carry events / no live
    // snapshots, which is degraded but still useful.
    let mut att_sub = match VEHICLE_ATTITUDE.subscriber() {
        Ok(s) => Some(s),
        Err(_) => {
            defmt::warn!("postmortem: VEHICLE_ATTITUDE SUBS exhausted; snapshots disabled");
            None
        }
    };
    let mut odom_sub = match VEHICLE_ODOMETRY.subscriber() {
        Ok(s) => Some(s),
        Err(_) => {
            defmt::warn!("postmortem: VEHICLE_ODOMETRY SUBS exhausted; snapshots disabled");
            None
        }
    };
    let mut rc_sub = match RC_INPUT.subscriber() {
        Ok(s) => Some(s),
        Err(_) => {
            defmt::warn!("postmortem: RC_INPUT SUBS exhausted; rc snapshot disabled");
            None
        }
    };

    let mut prev_failsafe = FAILSAFE_ACTIVE.load(Ordering::Acquire);
    let mut prev_est_ready = ESTIMATOR_READY.load(Ordering::Acquire);
    let mut prev_rc_healthy = RC_LINK_HEALTHY.load(Ordering::Acquire);
    #[cfg(feature = "outer_mpc")]
    let mut prev_mission = MISSION_STATE.load(Ordering::Acquire);
    let mut event_seq: u32 = 0;

    let boot_instant = Instant::now();
    let mut ticker = Ticker::every(embassy_time::Duration::from_millis(TICK_PERIOD_MS));

    defmt::info!("postmortem: task running ({}ms tick)", TICK_PERIOD_MS);

    loop {
        ticker.next().await;

        // ── Drain snapshot subscribers. Take the latest only — the
        // ring is for events, not data. `try_next_message` returns
        // `None` once the queue is empty; we keep the `latest` we
        // ever saw and discard the rest.
        let latest_att = drain_latest(att_sub.as_mut());
        let latest_odom = drain_latest(odom_sub.as_mut());
        let latest_rc = drain_latest(rc_sub.as_mut());

        // ── Edge-detect health atomics + append events.
        let mut events_to_push: heapless::Vec<(u8, u32), 8> = heapless::Vec::new();

        let now_failsafe = FAILSAFE_ACTIVE.load(Ordering::Acquire);
        if now_failsafe != prev_failsafe {
            if now_failsafe {
                let reason = FAILSAFE_REASON.load(Ordering::Acquire);
                let _ = events_to_push.push((KIND_FAILSAFE, reason as u32));
            } else {
                let _ = events_to_push.push((KIND_FAILSAFE_CLEAR, 0));
            }
            prev_failsafe = now_failsafe;
        }

        let now_est = ESTIMATOR_READY.load(Ordering::Acquire);
        if now_est != prev_est_ready {
            let kind = if now_est {
                KIND_ESTIMATOR_UP
            } else {
                KIND_ESTIMATOR_DOWN
            };
            let _ = events_to_push.push((kind, 0));
            prev_est_ready = now_est;
        }

        let now_rc = RC_LINK_HEALTHY.load(Ordering::Acquire);
        if now_rc != prev_rc_healthy {
            let kind = if now_rc {
                KIND_RC_RECOVERED
            } else {
                KIND_RC_LOSS
            };
            let _ = events_to_push.push((kind, 0));
            prev_rc_healthy = now_rc;
        }

        #[cfg(feature = "outer_mpc")]
        {
            let now_m = MISSION_STATE.load(Ordering::Acquire);
            if now_m != prev_mission {
                let kind = match crate::control::MissionState::from_u8(now_m) {
                    crate::control::MissionState::Planning => KIND_MISSION_PLANNING,
                    crate::control::MissionState::Executing => KIND_MISSION_EXECUTING,
                    crate::control::MissionState::Idle => KIND_MISSION_IDLE,
                };
                let _ = events_to_push.push((kind, prev_mission as u32));
                prev_mission = now_m;
            }
        }

        // ── Commit the entire batch under one BKPSRAM write window.
        // We hold no embassy lock; `bkpsram::with_record_mut` is a
        // raw-pointer access that can't deadlock.
        if !events_to_push.is_empty()
            || latest_att.is_some()
            || latest_odom.is_some()
            || latest_rc.is_some()
        {
            let timestamp_ms = boot_instant.elapsed().as_millis() as u32;
            bkpsram::with_record_mut(|r| {
                for (kind, data) in events_to_push.iter() {
                    event_seq = event_seq.wrapping_add(1);
                    r.push_event(EventEntry {
                        timestamp_ms,
                        kind: *kind,
                        _pad: [0; 3],
                        data: *data,
                        seq: event_seq,
                    });
                }
                if let Some(att) = latest_att.as_ref() {
                    let q = att.orientation.as_ref();
                    r.snapshots.attitude_quat_wijk = [q.w, q.i, q.j, q.k];
                }
                if let Some(odom) = latest_odom.as_ref() {
                    r.snapshots.position_xyz = [
                        odom.pose.position.x,
                        odom.pose.position.y,
                        odom.pose.position.z,
                    ];
                    r.snapshots.velocity_xyz = [
                        odom.twist.linear.x,
                        odom.twist.linear.y,
                        odom.twist.linear.z,
                    ];
                }
                if let Some(rc) = latest_rc.as_ref() {
                    let n = (rc.channel_count as usize)
                        .min(rc.channels.len())
                        .min(r.snapshots.rc_channels.len());
                    for i in 0..n {
                        r.snapshots.rc_channels[i] = rc.channels[i];
                    }
                    r.snapshots.rc_channel_count = n as u8;
                }
                r.header.uptime_ms = timestamp_ms;
                record::finalize(r);
            });
        }
    }
}

/// Drain a subscriber, keeping only the latest message. `Lagged`
/// counts are discarded — we don't track drops at this layer; the
/// existence of a snapshot at all is the signal.
fn drain_latest<M, const C: usize, const S: usize, const P: usize>(
    sub: Option<&mut Subscriber<'static, CriticalSectionRawMutex, M, C, S, P>>,
) -> Option<M>
where
    M: Clone,
{
    let sub = sub?;
    let mut latest: Option<M> = None;
    while let Some(wr) = sub.try_next_message() {
        if let WaitResult::Message(m) = wr {
            latest = Some(m);
        }
    }
    latest
}
