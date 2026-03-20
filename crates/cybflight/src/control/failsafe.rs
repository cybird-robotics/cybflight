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

use core::sync::atomic::{AtomicBool, Ordering};

use embassy_time::{with_timeout, Duration, Instant};

use crate::motors::ARM_STATE;
use crate::sensors::RC_INPUT;
use cybflight_msgs as msgs;

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
pub static FAILSAFE_ACTIVE: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Disarm and enter failsafe state.
fn enter_failsafe(reason: &str) {
    defmt::error!("FAILSAFE: {} — DISARMING", reason);
    FAILSAFE_ACTIVE.store(true, Ordering::Release);
    ARM_STATE.signal(msgs::ArmDisarm {
        timestamp: Instant::now(),
        armed: false,
    });
    crate::status::STATUS
        .sender()
        .send(crate::status::SystemStatus::Failsafe);
}

/// Check if the controller heartbeat has timed out.
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
            enter_failsafe("controller silent for >500ms");
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
                        // Valid frame — stay idle.
                    }
                    Err(_timeout) => {
                        let now = Instant::now();
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
                        enter_failsafe("RC loss for >1500ms");
                        phase = Phase::Landed {
                            recovery_start: None,
                        };
                    }
                    Some(remaining) => {
                        // Still within guard period — try to receive a frame.
                        let wait = remaining.min(RXLOSS_TRIGGER);
                        match with_timeout(wait, sub.next_message_pure()).await {
                            Ok(_rc) => {
                                defmt::info!("Failsafe: RC recovered during guard period");
                                phase = Phase::Idle;
                            }
                            Err(_timeout) => {
                                // Still no data — loop will re-check guard expiry.
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
                        let start = recovery_start.unwrap_or(Instant::now());
                        if Instant::now().duration_since(start) >= RECOVERY_PERIOD {
                            // 500 ms of continuous valid RC — clear failsafe.
                            FAILSAFE_ACTIVE.store(false, Ordering::Release);
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
                        phase = Phase::Landed {
                            recovery_start: None,
                        };
                    }
                }
            }
        }
    }
}
