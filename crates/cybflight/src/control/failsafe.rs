//! Failsafe: monitors RC link health and controller liveness, disarms on failure.
//!
//! Single task monitors two independent conditions:
//!
//! ## RC link failsafe
//! Betaflight-inspired two-stage failsafe:
//! - **Stage 1 (guard period):** 1.5 s after first frame timeout (150 ms).
//!   If RC recovers during the guard period, return to normal immediately.
//! - **Stage 2 (DROP_IT):** Disarm via `ARM_STATE`, block re-arming via
//!   `FAILSAFE_ACTIVE` until 500 ms of continuous valid RC data is received.
//!
//! ## Controller watchdog
//! If the inner loop hasn't published a motor command for `CTRL_TIMEOUT`,
//! disarm immediately. Catches sustained failures: ESKF divergence, NaN
//! in the controller, stale odometry, or stale RC setpoints.

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use embassy_time::{with_timeout, Duration, Instant};

use crate::motors::ARM_STATE;
use crate::sensors::RC_INPUT;
use cybflight_msgs as msgs;

/// Reason code for the most recent `enter_failsafe` call. Encoded
/// as `u8` so it can be carried verbatim in the `data` field of a
/// blackbox `KIND_FAILSAFE` event. Stable on disk — never
/// renumber; assign new values for new conditions.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, defmt::Format)]
pub enum FailsafeReason {
    /// No failsafe in effect (or not yet recorded).
    None = 0,
    /// Inner-loop motor commands silent for `>CTRL_TIMEOUT` (500 ms).
    /// Catches sustained controller failures: ESKF divergence
    /// downstream effects, NaN in the controller, stale odometry
    /// or RC setpoints feeding the controller.
    ControllerTimeout = 1,
    /// RC frames absent through `GUARD_PERIOD` (1500 ms) without
    /// recovery during the guard window.
    RcLoss = 2,
}

// ---------------------------------------------------------------------------
// Timing constants
// ---------------------------------------------------------------------------

/// No valid RC frame for this long triggers Stage 1.
/// BF: `RXLOSS_TRIGGER_INTERVAL` = 150 ms.
const RXLOSS_TRIGGER: Duration = Duration::from_millis(150);

/// Total time from first frame loss before Stage 2 activates.
/// BF: `failsafe_delay` default = 15 (×100 ms) = 1.5 s.
const GUARD_PERIOD: Duration = Duration::from_millis(1500);

/// Continuous valid RC data required to clear failsafe after Stage 2.
/// BF: `failsafe_recovery_delay` default = 5 (×100 ms) = 500 ms.
const RECOVERY_PERIOD: Duration = Duration::from_millis(500);

/// If no motor command published for this long, disarm.
/// Must be longer than any single-frame skip (odom stale = 100 ms, RC stale = 250 ms)
/// but short enough to catch sustained failures before the vehicle falls far.
const CTRL_TIMEOUT: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// `true` while failsafe is active. Checked by the arming state
/// machine in `sensors/rc.rs` to block re-arming (mirrors BF
/// `ARMING_DISABLED_FAILSAFE`).
///
/// **Write order convention:** `enter_failsafe` writes
/// [`FAILSAFE_REASON`] *before* this atomic, both with
/// `Ordering::Release`. Readers using `Acquire` to load
/// `FAILSAFE_ACTIVE` are guaranteed to see the reason that
/// triggered this transition because the Release/Acquire pair on
/// this atomic carries every prior store in the same task.
pub static FAILSAFE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Reason code carried alongside [`FAILSAFE_ACTIVE`]. Set by
/// `enter_failsafe` *before* `FAILSAFE_ACTIVE` flips to true, and
/// reset to `FailsafeReason::None as u8` when failsafe clears.
/// Read with `Acquire` after observing a true→true edge on
/// `FAILSAFE_ACTIVE`. The blackbox recorder mirrors this into the
/// `data` field of a `KIND_FAILSAFE` event.
pub static FAILSAFE_REASON: AtomicU8 = AtomicU8::new(FailsafeReason::None as u8);

/// `true` whenever an RC frame has arrived within `RXLOSS_TRIGGER`
/// (150 ms). Goes false on the first RC timeout in any Phase and
/// true again on the next valid frame. Independent of
/// `FAILSAFE_ACTIVE`: a brief drop that recovers within the guard
/// period flips this atomic twice without ever flipping
/// `FAILSAFE_ACTIVE`. The blackbox uses this to log `KIND_RC_LOSS`
/// / `KIND_RC_RECOVERED` events — useful for diagnosing marginal
/// RC links that don't quite trigger a hard failsafe.
pub static RC_LINK_HEALTHY: AtomicBool = AtomicBool::new(true);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Disarm and enter failsafe state.
///
/// Stores `FAILSAFE_REASON` *before* `FAILSAFE_ACTIVE` so a
/// reader observing the `false→true` edge on `FAILSAFE_ACTIVE`
/// (Acquire) is guaranteed to see the matching reason
/// (carried by the same Release).
fn enter_failsafe(reason: FailsafeReason) {
    defmt::error!("FAILSAFE: {:?} — DISARMING", reason);
    FAILSAFE_REASON.store(reason as u8, Ordering::Release);
    FAILSAFE_ACTIVE.store(true, Ordering::Release);
    ARM_STATE.signal(msgs::ArmDisarm {
        timestamp: Instant::now(),
        armed: false,
    });
    crate::status::STATUS
        .sender()
        .send(crate::status::SystemStatus::Failsafe);
    // Clear any in-flight mission so a subsequent arm does not resume a
    // stale trajectory. Safe no-op when no mission is active.
    #[cfg(feature = "outer_mpc")]
    {
        super::MISSION_ABORT_REQUESTED.store(false, Ordering::Release);
        super::MISSION_TRAJECTORY_SLOT.lock(|s| {
            let _ = s.borrow_mut().take();
            super::MISSION_STATE.store(
                super::MissionState::Idle as u8,
                Ordering::Release,
            );
        });
    }
}

/// Check if the controller heartbeat has timed out.
/// INDI stamps LAST_CONTROLLER_PUBLISH every tick it publishes motor commands.
fn controller_timed_out() -> bool {
    match super::LAST_CONTROLLER_PUBLISH.lock(|c| c.get()) {
        Some(t) => Instant::now().duration_since(t) > CTRL_TIMEOUT,
        None => false, // Controller hasn't started yet — not a failure
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

enum Phase {
    /// Normal operation — RC frames arriving.
    Idle,
    /// Stage 1: RC lost, counting down guard period before disarm.
    GuardPeriod { loss_start: Instant },
    /// Stage 2 complete: disarmed, waiting for RC recovery.
    Landed { recovery_start: Option<Instant> },
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

#[embassy_executor::task]
pub async fn failsafe_task() {
    let mut sub = RC_INPUT
        .subscriber()
        .expect("failsafe: RC_INPUT subscriber");
    let mut phase = Phase::Idle;

    defmt::info!("Failsafe: monitoring RC link + controller heartbeat");

    loop {
        // --- Controller watchdog: check on every loop iteration ---
        if controller_timed_out() {
            enter_failsafe(FailsafeReason::ControllerTimeout);
            super::LAST_CONTROLLER_PUBLISH.lock(|c| c.set(None));
            phase = Phase::Landed {
                recovery_start: None,
            };
            // Fall through to Landed phase which waits for RC recovery.
        }

        match phase {
            // ----------------------------------------------------------
            // IDLE: wait for RC frames, timeout starts guard period
            // ----------------------------------------------------------
            Phase::Idle => {
                match with_timeout(RXLOSS_TRIGGER, sub.next_message_pure()).await {
                    Ok(_rc) => {
                        // Valid frame — stay idle. Idempotent — no-op
                        // if already true; observable transitions only
                        // happen on edges, which is what the recorder
                        // edge-detects.
                        RC_LINK_HEALTHY.store(true, Ordering::Release);
                    }
                    Err(_timeout) => {
                        let now = Instant::now();
                        // Mark the link unhealthy *before* changing
                        // Phase so any reader that observes the new
                        // Phase already sees the down-edge.
                        RC_LINK_HEALTHY.store(false, Ordering::Release);
                        defmt::warn!(
                            "Failsafe: RC frame timeout ({}ms) — guard period started",
                            RXLOSS_TRIGGER.as_millis(),
                        );
                        phase = Phase::GuardPeriod { loss_start: now };
                    }
                }
            }

            // ----------------------------------------------------------
            // GUARD PERIOD: wait for recovery or guard expiry
            // ----------------------------------------------------------
            Phase::GuardPeriod { loss_start } => {
                let elapsed = Instant::now().duration_since(loss_start);
                let remaining = GUARD_PERIOD.checked_sub(elapsed);

                match remaining {
                    None | Some(Duration::MIN) => {
                        enter_failsafe(FailsafeReason::RcLoss);
                        phase = Phase::Landed {
                            recovery_start: None,
                        };
                    }
                    Some(remaining) => {
                        // Still within guard period — try to receive a frame.
                        let wait = remaining.min(RXLOSS_TRIGGER);
                        match with_timeout(wait, sub.next_message_pure()).await {
                            Ok(_rc) => {
                                // RC recovered before commit — flip
                                // RC_LINK_HEALTHY back true so the
                                // recorder logs a `KIND_RC_RECOVERED`
                                // (paired with the prior
                                // `KIND_RC_LOSS`). No `KIND_FAILSAFE`
                                // is logged because failsafe never
                                // committed.
                                RC_LINK_HEALTHY.store(true, Ordering::Release);
                                defmt::info!("Failsafe: RC recovered during guard period");
                                phase = Phase::Idle;
                            }
                            Err(_timeout) => {
                                // Still no data — loop will re-check guard expiry.
                                // RC_LINK_HEALTHY stays false (already false).
                            }
                        }
                    }
                }
            }

            // ----------------------------------------------------------
            // LANDED: disarmed, wait for sustained RC recovery
            // ----------------------------------------------------------
            Phase::Landed { recovery_start } => {
                match with_timeout(RXLOSS_TRIGGER, sub.next_message_pure()).await {
                    Ok(_rc) => {
                        // Frame arrived. RC link is healthy regardless
                        // of whether we've accumulated enough recovery
                        // time to clear failsafe yet.
                        RC_LINK_HEALTHY.store(true, Ordering::Release);
                        let start = recovery_start.unwrap_or(Instant::now());
                        if Instant::now().duration_since(start) >= RECOVERY_PERIOD {
                            // 500 ms of continuous valid RC — clear failsafe.
                            // Reason cleared *after* ACTIVE so a reader
                            // briefly seeing `ACTIVE=false, REASON=<old>`
                            // is preferable to the inverse (where
                            // ACTIVE=true with a stale REASON would be
                            // ambiguous).
                            FAILSAFE_ACTIVE.store(false, Ordering::Release);
                            FAILSAFE_REASON.store(
                                FailsafeReason::None as u8,
                                Ordering::Release,
                            );
                            crate::status::STATUS
                                .sender()
                                .send(crate::status::SystemStatus::Disarmed);
                            defmt::info!(
                                "Failsafe: RC recovered for {}ms — arming re-enabled",
                                RECOVERY_PERIOD.as_millis(),
                            );
                            phase = Phase::Idle;
                        } else {
                            phase = Phase::Landed {
                                recovery_start: Some(start),
                            };
                        }
                    }
                    Err(_timeout) => {
                        // Still no data — reset recovery timer.
                        RC_LINK_HEALTHY.store(false, Ordering::Release);
                        phase = Phase::Landed {
                            recovery_start: None,
                        };
                    }
                }
            }
        }
    }
}
