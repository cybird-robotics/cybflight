//! Cross-cutting state-estimation failsafe state machine.
//!
//! Owns the parts of post-update bookkeeping that are not specific
//! to the measurement source:
//!
//! - Consecutive-jump cascade detection.
//! - Consecutive-reject cascade detection.
//! - Staleness gate (no accepted measurement for `stale_ms`).
//! - Post-update NaN detection (`!eskf.is_initialized()`).
//! - Convergence detection on the gyro-bias-cov diagonal.
//! - Telemetry counters for pos / att / vel / jump rejections and
//!   inflations.
//!
//! Source-specific guards (`EskfGpsGuard`, future `EskfMocapGuard`)
//! compose `EskfFailsafe` and add their own measurement-update flow
//! and quality bookkeeping (RTK debounce, Vicon "tracking strong"
//! flag, etc.). The split is the same one the audit identified: ~70%
//! of the existing `EskfGpsGuard` was already general-purpose
//! state-estimation failsafe; this lifts it out so mocap (and any
//! future exteroceptive source) can plug in.
//!
//! Behaviour-preservation: every counter, every cascade ordering,
//! and every drop-`converged` decision in this module mirrors what
//! the original `EskfGpsGuard::on_pvt` did. The `cybflight_sim`
//! integration tests and the `gps_guard::tests` unit tests are the
//! bit-stable safety net for the extraction.

use crate::eskf::{Eskf, UpdateOutcome};

/// Which gyro-bias covariance trace to use for the convergence check.
///
/// The choice depends on whether the measurement source observes all
/// three orientation axes:
///
/// * `All` — sum the full diagonal. Use for sources that observe yaw
///   (mocap pose, magnetometer-aided fusion, etc.). Yaw bias actually
///   converges, so the trace is a meaningful "we trust the bias
///   estimate" gate.
/// * `XyOnly` — sum only x and y. Use for sources that don't observe
///   yaw (GPS-only, before a course-of-motion / magnetometer update is
///   wired in). Yaw bias variance never decreases, so a 3-axis trace
///   would never cross any sensible threshold and `converged` would
///   stay false forever.
#[derive(Clone, Copy, Debug)]
pub enum ConvergenceAxes {
    All,
    XyOnly,
}

#[derive(Clone, Copy, Debug)]
pub struct EskfFailsafeConfig {
    pub max_consecutive_jumps: u32,
    pub max_consecutive_rejects: u32,
    pub stale_ms: u64,
    pub gyro_bias_cov_trace_thresh: f32,
    pub convergence_axes: ConvergenceAxes,
}

impl EskfFailsafeConfig {
    pub fn new(
        max_consecutive_jumps: u32,
        max_consecutive_rejects: u32,
        stale_ms: u64,
        gyro_bias_cov_trace_thresh: f32,
        convergence_axes: ConvergenceAxes,
    ) -> Self {
        Self {
            max_consecutive_jumps,
            max_consecutive_rejects,
            stale_ms,
            gyro_bias_cov_trace_thresh,
            convergence_axes,
        }
    }
}

/// What a `record_*_outcome` call decided. The source-specific guard
/// translates this into its own outcome type and (optionally) acts on
/// `JumpCascade` by re-initialising the filter at the latest
/// measurement; `RejectCascade` and `NanAfterUpdate` are
/// caller-fatal: the guard's `is_ready()` flips false until something
/// (re-init or normal convergence) restores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailsafeAction {
    /// No cascade fired. Outcome processed; counters updated.
    None,
    /// A failure mode tripped a gate. `converged` has been dropped.
    Disarmed(DisarmCause),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisarmCause {
    /// `consecutive_jumps` reached `max_consecutive_jumps`. The
    /// caller decides whether to re-init at the latest measurement
    /// (and call `note_reinit`) or ride out on dead-reckoning.
    JumpCascade { consecutive: u32 },
    /// `consecutive_rejects` reached `max_consecutive_rejects`. The
    /// filter is producing systematic rejections; let staleness be
    /// the catch-all.
    RejectCascade { consecutive: u32 },
    /// `eskf.is_initialized()` returned false after the update — the
    /// filter went non-finite. The next measurement should re-init;
    /// the caller's source-specific code does the re-init and calls
    /// `note_reinit`.
    NanAfterUpdate,
}

#[derive(Clone, Copy, Debug)]
pub struct TickOutcome {
    pub is_stale: bool,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct FailsafeSnapshot {
    pub converged: bool,
    pub consecutive_jumps: u32,
    pub consecutive_rejects: u32,
    pub last_accept_ms: u64,
    pub pos_reject_total: u32,
    pub pos_inflated_total: u32,
    pub att_reject_total: u32,
    pub att_inflated_total: u32,
    pub jump_total: u32,
    pub vel_reject_total: u32,
    pub vel_inflated_total: u32,
}

pub struct EskfFailsafe {
    cfg: EskfFailsafeConfig,
    converged: bool,
    consecutive_jumps: u32,
    consecutive_rejects: u32,
    last_accept_ms: u64,
    pos_reject_total: u32,
    pos_inflated_total: u32,
    att_reject_total: u32,
    att_inflated_total: u32,
    jump_total: u32,
    vel_reject_total: u32,
    vel_inflated_total: u32,
}

impl EskfFailsafe {
    /// `anchor_ms` seeds the staleness clock. Pass the timestamp of
    /// the bootstrap measurement (origin-anchor PVT for GPS, first
    /// mocap pose for Vicon).
    pub fn new(cfg: EskfFailsafeConfig, anchor_ms: u64) -> Self {
        Self {
            cfg,
            converged: false,
            consecutive_jumps: 0,
            consecutive_rejects: 0,
            last_accept_ms: anchor_ms,
            pos_reject_total: 0,
            pos_inflated_total: 0,
            att_reject_total: 0,
            att_inflated_total: 0,
            jump_total: 0,
            vel_reject_total: 0,
            vel_inflated_total: 0,
        }
    }

    pub fn config(&self) -> &EskfFailsafeConfig {
        &self.cfg
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn snapshot(&self) -> FailsafeSnapshot {
        FailsafeSnapshot {
            converged: self.converged,
            consecutive_jumps: self.consecutive_jumps,
            consecutive_rejects: self.consecutive_rejects,
            last_accept_ms: self.last_accept_ms,
            pos_reject_total: self.pos_reject_total,
            pos_inflated_total: self.pos_inflated_total,
            att_reject_total: self.att_reject_total,
            att_inflated_total: self.att_inflated_total,
            jump_total: self.jump_total,
            vel_reject_total: self.vel_reject_total,
            vel_inflated_total: self.vel_inflated_total,
        }
    }

    /// Record the outcome of a position-only update (e.g.
    /// `Eskf::update_pos_sparse`). Bumps pos counters, drives the
    /// jump and reject cascades, and runs the post-update NaN check
    /// against `eskf`.
    pub fn record_pos_outcome(
        &mut self,
        eskf: &Eskf,
        outcome: UpdateOutcome,
        now_ms: u64,
    ) -> FailsafeAction {
        self.record_outcome_internal(eskf, outcome, now_ms, /* is_pose: */ false)
    }

    /// Record the outcome of a joint pose update (pos + att applied
    /// in one Kalman step, e.g. mocap `Eskf::update_pose`). Bumps
    /// both pos and att counters; cascade logic is identical to
    /// `record_pos_outcome`.
    pub fn record_pose_outcome(
        &mut self,
        eskf: &Eskf,
        outcome: UpdateOutcome,
        now_ms: u64,
    ) -> FailsafeAction {
        self.record_outcome_internal(eskf, outcome, now_ms, /* is_pose: */ true)
    }

    /// Record a velocity-only update outcome (e.g. GPS
    /// `Eskf::update_vel`). Velocity is supplementary — it bumps
    /// `vel_*_total` for telemetry but does not feed the jump or
    /// reject cascade.
    pub fn record_vel_outcome(&mut self, outcome: UpdateOutcome) {
        match outcome {
            UpdateOutcome::Accepted { inflated: true } => {
                self.vel_inflated_total = self.vel_inflated_total.wrapping_add(1);
            }
            UpdateOutcome::Accepted { inflated: false } => {}
            _ => {
                self.vel_reject_total = self.vel_reject_total.wrapping_add(1);
            }
        }
    }

    /// Per-predict-tick housekeeping: lift `converged` to true if
    /// the gyro-bias-cov trace has dropped below threshold, and drop
    /// it back to false if no accepted measurement has arrived in
    /// `stale_ms`.
    pub fn on_predict_tick(&mut self, eskf: &Eskf, now_ms: u64) -> TickOutcome {
        if !self.converged {
            let trace = match self.cfg.convergence_axes {
                ConvergenceAxes::All => eskf.gyro_bias_cov_trace(),
                ConvergenceAxes::XyOnly => eskf.gyro_bias_cov_trace_xy(),
            };
            if trace < self.cfg.gyro_bias_cov_trace_thresh {
                self.converged = true;
            }
        }
        let age = now_ms.saturating_sub(self.last_accept_ms);
        let is_stale = age > self.cfg.stale_ms;
        if is_stale && self.converged {
            self.converged = false;
        }
        TickOutcome {
            is_stale,
            converged: self.converged,
        }
    }

    /// Caller signal: "I just re-initialised the filter at a fresh
    /// measurement." Resets cascade counters, refreshes the
    /// staleness clock, and explicitly sets `converged=false` (a
    /// freshly re-initialised filter is by definition not converged).
    pub fn note_reinit(&mut self, at_ms: u64) {
        self.converged = false;
        self.consecutive_jumps = 0;
        self.consecutive_rejects = 0;
        self.last_accept_ms = at_ms;
    }

    /// Read-only view of `last_accept_ms`. The source-specific guard
    /// uses this to align its own timestamps (e.g. mocap's
    /// `last_pose_ts`).
    pub fn last_accept_ms(&self) -> u64 {
        self.last_accept_ms
    }

    fn record_outcome_internal(
        &mut self,
        eskf: &Eskf,
        outcome: UpdateOutcome,
        now_ms: u64,
        is_pose: bool,
    ) -> FailsafeAction {
        // Telemetry counters (unconditional). Inflated/Accepted
        // outcomes bump the inflation counter; non-jump rejections
        // bump the rejection counter; jumps bump jump_total only
        // (the jump-cascade branch below counts those separately
        // from generic rejects, matching the original GPS task
        // behaviour).
        match outcome {
            UpdateOutcome::Accepted { inflated: true } => {
                self.pos_inflated_total = self.pos_inflated_total.wrapping_add(1);
                if is_pose {
                    self.att_inflated_total = self.att_inflated_total.wrapping_add(1);
                }
            }
            UpdateOutcome::Accepted { inflated: false } => {}
            UpdateOutcome::JumpRejected => {
                self.jump_total = self.jump_total.wrapping_add(1);
            }
            _ => {
                self.pos_reject_total = self.pos_reject_total.wrapping_add(1);
                if is_pose {
                    self.att_reject_total = self.att_reject_total.wrapping_add(1);
                }
            }
        }

        // Cascade tracking. Jump and reject counters update
        // unconditionally; the *Disarmed action* is first-fire-wins
        // — jump cascade preempts reject cascade on the same frame
        // because `converged` flips to false before reject-cascade's
        // `if self.converged` gate runs. NaN-after-update preempts
        // both at the bottom of the function. This ordering
        // preserves the behaviour the GPS task had pre-extraction.
        let mut action = FailsafeAction::None;

        if outcome.is_jump() {
            self.consecutive_jumps = self.consecutive_jumps.saturating_add(1);
            if self.consecutive_jumps >= self.cfg.max_consecutive_jumps {
                self.converged = false;
                action = FailsafeAction::Disarmed(DisarmCause::JumpCascade {
                    consecutive: self.consecutive_jumps,
                });
            }
        } else {
            self.consecutive_jumps = 0;
        }

        if outcome.is_accepted() {
            self.last_accept_ms = now_ms;
            self.consecutive_rejects = 0;
        } else {
            self.consecutive_rejects = self.consecutive_rejects.saturating_add(1);
            if matches!(action, FailsafeAction::None)
                && self.converged
                && self.consecutive_rejects >= self.cfg.max_consecutive_rejects
            {
                self.converged = false;
                action = FailsafeAction::Disarmed(DisarmCause::RejectCascade {
                    consecutive: self.consecutive_rejects,
                });
            }
        }

        // Post-update NaN guard. If the filter went non-finite on
        // this frame, override the cascade outcome — NaN is more
        // catastrophic, and the caller's wrapper needs to know to
        // re-init on the next measurement.
        if !eskf.is_initialized() {
            self.converged = false;
            self.consecutive_jumps = 0;
            self.consecutive_rejects = 0;
            action = FailsafeAction::Disarmed(DisarmCause::NanAfterUpdate);
        }

        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eskf::EskfConfig;
    use nalgebra::{UnitQuaternion, Vector3};

    fn make_eskf() -> Eskf {
        let mut e = Eskf::new(EskfConfig {
            max_pos_jump_m: 3.0,
            ..EskfConfig::default()
        });
        e.init_with_cov(
            Vector3::zeros(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
            Vector3::new(0.1, 0.1, 10.0),
        );
        e
    }

    fn cfg() -> EskfFailsafeConfig {
        EskfFailsafeConfig::new(
            2,
            5,
            2_000,
            0.002,
            ConvergenceAxes::XyOnly,
        )
    }

    #[test]
    fn jump_cascade_drops_converged_at_limit() {
        let mut f = EskfFailsafe::new(cfg(), 0);
        let e = make_eskf();
        // Force converged=true synthetically — the cascade gate is
        // gated on `converged` being true; there's no other clean
        // way to flip it without driving on_predict_tick through a
        // real predict loop, which a unit test shouldn't do.
        f.converged = true;

        // First jump: bumps to 1, no disarm yet.
        let a = f.record_pos_outcome(&e, UpdateOutcome::JumpRejected, 100);
        assert_eq!(a, FailsafeAction::None);
        assert_eq!(f.snapshot().consecutive_jumps, 1);
        assert!(f.converged());

        // Second jump: hits limit (2), Disarmed{JumpCascade}.
        let a = f.record_pos_outcome(&e, UpdateOutcome::JumpRejected, 200);
        assert_eq!(
            a,
            FailsafeAction::Disarmed(DisarmCause::JumpCascade { consecutive: 2 })
        );
        assert!(!f.converged());
        assert_eq!(f.snapshot().jump_total, 2);
    }

    #[test]
    fn note_reinit_clears_cascade_and_refreshes_clock() {
        let mut f = EskfFailsafe::new(cfg(), 0);
        let e = make_eskf();
        f.converged = true;
        let _ = f.record_pos_outcome(&e, UpdateOutcome::JumpRejected, 100);
        let _ = f.record_pos_outcome(&e, UpdateOutcome::JumpRejected, 200);
        assert!(!f.converged());
        assert_eq!(f.snapshot().consecutive_jumps, 2);

        f.note_reinit(300);
        let s = f.snapshot();
        assert_eq!(s.consecutive_jumps, 0);
        assert_eq!(s.consecutive_rejects, 0);
        assert_eq!(s.last_accept_ms, 300);
        assert!(!f.converged());
    }

    #[test]
    fn accepted_clears_consecutive_jumps_and_rejects() {
        let mut f = EskfFailsafe::new(cfg(), 0);
        let e = make_eskf();
        f.converged = true;
        let _ = f.record_pos_outcome(&e, UpdateOutcome::JumpRejected, 100);
        let _ = f.record_pos_outcome(&e, UpdateOutcome::InverseFailed, 200);
        assert_eq!(f.snapshot().consecutive_jumps, 0); // reset on non-jump
        assert_eq!(f.snapshot().consecutive_rejects, 2); // jump + inverse-fail

        let a = f.record_pos_outcome(&e, UpdateOutcome::Accepted { inflated: false }, 300);
        assert_eq!(a, FailsafeAction::None);
        let s = f.snapshot();
        assert_eq!(s.consecutive_jumps, 0);
        assert_eq!(s.consecutive_rejects, 0);
        assert_eq!(s.last_accept_ms, 300);
    }

    #[test]
    fn pose_outcome_bumps_both_pos_and_att_counters() {
        let mut f = EskfFailsafe::new(cfg(), 0);
        let e = make_eskf();
        let _ = f.record_pose_outcome(&e, UpdateOutcome::Accepted { inflated: true }, 100);
        let s = f.snapshot();
        assert_eq!(s.pos_inflated_total, 1);
        assert_eq!(s.att_inflated_total, 1);

        let _ = f.record_pose_outcome(&e, UpdateOutcome::InverseFailed, 200);
        let s = f.snapshot();
        assert_eq!(s.pos_reject_total, 1);
        assert_eq!(s.att_reject_total, 1);
    }

    #[test]
    fn on_predict_tick_marks_stale_after_window() {
        let mut f = EskfFailsafe::new(cfg(), 0);
        let e = make_eskf();
        f.converged = true;
        let t = f.on_predict_tick(&e, cfg().stale_ms + 10);
        assert!(t.is_stale);
        assert!(!t.converged);
    }
}
