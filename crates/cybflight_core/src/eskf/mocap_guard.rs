//! Mocap (Vicon-style 6-DoF rigid-body) failsafe wrapper around
//! [`EskfFailsafe`].
//!
//! Owns only the mocap-specific bits:
//!
//! - Joint pos+att update via `Eskf::update_pose` with the
//!   configured σ_pos / σ_att.
//! - Re-init at the current pose when the filter has gone non-finite
//!   on entry (NaN guard).
//! - Mocap-specific outcome enum (no `carr_soln` quality fields —
//!   mocap doesn't have an RTK analog; the source is either tracking
//!   strongly or dropping frames entirely).
//!
//! Cross-cutting failsafe behaviour — jump cascade detection, reject
//! cascade, NaN-after-update, staleness, gyro-bias-cov convergence,
//! telemetry counters — lives in [`super::failsafe::EskfFailsafe`]
//! and is shared with `EskfGpsGuard`.
//!
//! # Mocap-specific policy at the jump cascade limit
//!
//! Unlike GPS — which re-anchors the filter at the latest RTK-fixed
//! PVT to break a long-outage drift wedge — mocap **does not**
//! re-init on jump cascade. A Vicon "jump" almost always means a
//! rigid-body re-association (the volume picked up a different body
//! and started reporting its pose); the operator's IMU dead-reckoning
//! is more trustworthy than the misidentified pose. Riding out IMU
//! until the cascade heals (or `mocap_stale_ms` fires as the
//! catch-all) is the safer policy for the typical flight envelope:
//! mocap arrives at 100–360 Hz so the staleness window is
//! O(100 ms), well within an IMU's drift budget.
//!
//! The wrapper makes this decision: it translates the failsafe's
//! `Disarmed(JumpCascade)` action into `MocapGuardOutcome::
//! DisarmedJumpCascade` without calling `note_reinit`.
//!
//! # The escape hatch
//!
//! Riding out IMU is only safe because it is *bounded*. Left alone the
//! cascade never heals: `consecutive_jumps` clears only on a non-jump
//! outcome, and no pose can produce one while it sits outside the jump
//! gate — so a filter whose position has separated from truth by more
//! than `max_pos_jump_m` rejects every pose forever and the vehicle
//! stays unarmable until a power cycle. That is reachable from an
//! entirely routine sequence: uplink drops while disarmed, operator
//! carries the airframe to the pad, uplink returns.
//!
//! `on_pose` therefore takes `armed`. **Disarmed**, once wedged, the
//! guard re-anchors at a pose stream that agrees with itself across
//! `reanchor_frames` within `reanchor_radius_m` — a stationary
//! airframe satisfies this, a re-associated (generally moving) body
//! does not. **Armed**, the policy is unchanged and absolute: no
//! re-anchor, ride out IMU, escalate through staleness and the fault
//! path. This is the mocap analog of `EskfGpsGuard::
//! reinit_min_carr_soln`, using self-consistency where GPS uses the
//! RTK quality flag.

use nalgebra::{UnitQuaternion, Vector3};

use super::failsafe::{
    ConvergenceAxes, DisarmCause, EskfFailsafe, EskfFailsafeConfig, FailsafeAction,
};
use super::gps_guard::{FilterReason, ReinitCause};
use super::{Eskf, UpdateOutcome};

/// One incoming mocap pose. Pos+att applied as a single 6-D Kalman
/// step so the gain for both channels is computed against the same
/// prior P (avoids the position-update collapsing P before the
/// attitude update sees it).
#[derive(Clone, Copy, Debug)]
pub struct MocapPose {
    pub position: Vector3<f32>,
    pub orientation: UnitQuaternion<f32>,
    pub timestamp_ms: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct MocapGuardConfig {
    pub max_consecutive_jumps: u32,
    pub max_consecutive_rejects: u32,
    pub mocap_stale_ms: u64,
    pub gyro_bias_cov_trace_thresh: f32,
    pub mocap_pos_std: f32,
    pub mocap_att_std: f32,
    /// Mutual-agreement radius for the disarmed re-anchor escape hatch
    /// (m). See [`Self::reanchor_frames`] and the module-level policy note.
    pub reanchor_radius_m: f32,
    /// Successive mutually-agreeing poses the re-anchor requires. The
    /// other half of the radius above; see [`DEFAULT_REANCHOR_FRAMES`].
    pub reanchor_frames: u32,
}

/// Successive mutually-agreeing poses required before the disarmed
/// re-anchor fires. At 100–360 Hz this is 14–50 ms of a stationary,
/// self-consistent pose stream — long enough that a re-associated body
/// flickering between two rigid bodies cannot satisfy it, short enough
/// that an operator who has set the airframe down is not left waiting.
pub const DEFAULT_REANCHOR_FRAMES: usize = 5;

impl Default for MocapGuardConfig {
    fn default() -> Self {
        Self {
            // Tighter than GPS (3 vs 2): a Vicon flip/re-association
            // is binary — either the volume has the right body or
            // the wrong body — so 3 frames is enough to be confident
            // it's a fault, not noise.
            max_consecutive_jumps: 3,
            // Looser than GPS (8 vs 5): mocap is much higher-rate
            // (100–360 Hz vs 5 Hz), so 8 frames is still ~25–80 ms.
            max_consecutive_rejects: 8,
            mocap_stale_ms: 100,
            gyro_bias_cov_trace_thresh: 0.003,
            mocap_pos_std: 0.01,
            mocap_att_std: 0.03,
            // 10 cm: an order of magnitude above the 1 cm pose σ (so
            // a genuinely stationary airframe clears it comfortably)
            // and an order of magnitude below the 1 m jump gate (so
            // it cannot be satisfied by the drifting, ambiguous
            // stream a re-association produces).
            reanchor_radius_m: 0.10,
            reanchor_frames: DEFAULT_REANCHOR_FRAMES as u32,
        }
    }
}

impl MocapGuardConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_consecutive_jumps: u32,
        max_consecutive_rejects: u32,
        mocap_stale_ms: u64,
        gyro_bias_cov_trace_thresh: f32,
        mocap_pos_std: f32,
        mocap_att_std: f32,
        reanchor_radius_m: f32,
    ) -> Self {
        Self {
            max_consecutive_jumps,
            max_consecutive_rejects,
            mocap_stale_ms,
            gyro_bias_cov_trace_thresh,
            mocap_pos_std,
            mocap_att_std,
            reanchor_radius_m,
            reanchor_frames: DEFAULT_REANCHOR_FRAMES as u32,
        }
    }
}

/// Result of `EskfMocapGuard::on_pose`. Mirrors the GPS guard's
/// outcome shape minus the GPS-only variants (no usability gate
/// reasons because mocap has none, no per-channel σ feedback because
/// mocap σ is configured statically).
#[derive(Clone, Copy, Debug)]
pub enum MocapGuardOutcome {
    Accepted {
        inflated: bool,
    },
    RejectedJump {
        consecutive: u32,
    },
    RejectedFilter {
        reason: FilterReason,
    },
    NonFinitePose,
    Reinitialised {
        at_pos: Vector3<f32>,
        at_orient: UnitQuaternion<f32>,
        cause: ReinitCause,
    },
    DisarmedJumpCascade {
        consecutive: u32,
    },
    DisarmedRejectCascade {
        consecutive: u32,
    },
    DisarmedNanState,
}

/// Per-tick outcome from `on_predict_tick`. Same shape as
/// `super::gps_guard::TickOutcome` — the wrapper publishes
/// `is_ready` to the arming-gate atomic and uses `is_stale` purely
/// for annunciation.
///
/// `is_stale` must **not** be used to withhold odometry: the
/// module-level policy is to ride the gap out on IMU dead-reckoning,
/// and a caller that stops publishing instead starves the outer
/// loop's own (much tighter) odometry-freshness gate, which
/// cascades all the way to the DShot watchdog cutting thrust. A
/// sustained outage is escalated by the fault path
/// (`eskf_fault_pos_timeout_s`), not by this flag.
#[derive(Clone, Copy, Debug)]
pub struct TickOutcome {
    pub is_ready: bool,
    pub is_stale: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct MocapGuardSnapshot {
    pub converged: bool,
    pub consecutive_jumps: u32,
    pub consecutive_rejects: u32,
    pub pos_reject_total: u32,
    pub pos_inflated_total: u32,
    pub att_reject_total: u32,
    pub att_inflated_total: u32,
    pub jump_total: u32,
    pub last_pose_accept_ms: u64,
}

pub struct EskfMocapGuard {
    cfg: MocapGuardConfig,
    failsafe: EskfFailsafe,
    /// Candidate anchor for the disarmed re-anchor hatch: the position
    /// of the first pose in the current agreeing run, plus how many
    /// poses have agreed with it so far. `None` whenever the guard is
    /// not in the wedged-and-disarmed state.
    reanchor_ref: Option<Vector3<f32>>,
    reanchor_count: u32,
}

impl EskfMocapGuard {
    /// `anchor_ms` is the timestamp of the first mocap pose that the
    /// wrapper has used to seed the ESKF. Seeds the staleness clock.
    pub fn new(cfg: MocapGuardConfig, anchor_ms: u64) -> Self {
        let failsafe_cfg = EskfFailsafeConfig::new(
            cfg.max_consecutive_jumps,
            cfg.max_consecutive_rejects,
            cfg.mocap_stale_ms,
            cfg.gyro_bias_cov_trace_thresh,
            // Mocap pose observes all three orientation axes —
            // unlike GPS-only which can't observe yaw.
            ConvergenceAxes::All,
        );
        Self {
            cfg,
            failsafe: EskfFailsafe::new(failsafe_cfg, anchor_ms),
            reanchor_ref: None,
            reanchor_count: 0,
        }
    }

    pub fn config(&self) -> &MocapGuardConfig {
        &self.cfg
    }

    pub fn is_ready(&self) -> bool {
        self.failsafe.converged()
    }

    pub fn snapshot(&self) -> MocapGuardSnapshot {
        let f = self.failsafe.snapshot();
        MocapGuardSnapshot {
            converged: f.converged,
            consecutive_jumps: f.consecutive_jumps,
            consecutive_rejects: f.consecutive_rejects,
            pos_reject_total: f.pos_reject_total,
            pos_inflated_total: f.pos_inflated_total,
            att_reject_total: f.att_reject_total,
            att_inflated_total: f.att_inflated_total,
            jump_total: f.jump_total,
            last_pose_accept_ms: f.last_accept_ms,
        }
    }

    /// Drive the guard with one incoming mocap pose. Mirrors the
    /// `EskfGpsGuard::on_pvt` shape but for the joint pose-update
    /// flow.
    ///
    /// `armed` gates the re-anchor escape hatch: the ride-out-IMU
    /// policy above is only safe because it is bounded. In flight it is
    /// absolute — a re-associated body must never capture the filter.
    /// On the ground it must not be, or the cascade is a permanent
    /// wedge (see `reanchor_radius_m`).
    pub fn on_pose(
        &mut self,
        eskf: &mut Eskf,
        pose: &MocapPose,
        armed: bool,
    ) -> MocapGuardOutcome {
        // Non-finite-pose gate. Cheap to do here so a corrupted
        // transport frame (ESP bridge / COBS decode) never reaches
        // `update_pose`.
        if !pose.position.iter().all(|v| v.is_finite())
            || !pose.orientation.into_inner().coords.iter().all(|v| v.is_finite())
        {
            return MocapGuardOutcome::NonFinitePose;
        }

        // NaN re-init on entry. The filter's predict step nullified
        // its `initialized` flag (NaN propagated through), so the
        // next mocap pose re-seeds it from this pose. This is the
        // mocap analog of the GPS guard's same-named branch.
        if !eskf.is_initialized() {
            self.reinit_filter(eskf, pose);
            return MocapGuardOutcome::Reinitialised {
                at_pos: pose.position,
                at_orient: pose.orientation,
                cause: ReinitCause::NanState,
            };
        }

        // Joint pos+att update — single Kalman step so the gain for
        // both channels uses the same prior P. The σ values come
        // from cfg (statically tuned, unlike GPS which derives σ
        // from `h_acc_mm` per fix).
        let outcome = eskf.update_pose(
            pose.position,
            pose.orientation,
            self.cfg.mocap_pos_std,
            self.cfg.mocap_att_std,
        );

        // Hand outcome to the cross-cutting failsafe — bumps both
        // pos and att counters (record_pose_outcome), drives the
        // jump+reject cascades, runs the post-update NaN check.
        let action = self.failsafe.record_pose_outcome(eskf, outcome, pose.timestamp_ms);

        // Default outcome from the update itself; the cascade match
        // below may override.
        let mut returned = match outcome {
            UpdateOutcome::Accepted { inflated } => MocapGuardOutcome::Accepted { inflated },
            UpdateOutcome::JumpRejected => MocapGuardOutcome::RejectedJump {
                consecutive: self.failsafe.snapshot().consecutive_jumps,
            },
            other => MocapGuardOutcome::RejectedFilter {
                reason: filter_reason(other),
            },
        };

        // Failsafe-driven overrides. Mocap policy at the jump
        // cascade limit: do NOT re-init. The IMU-integrated estimate
        // is more trustworthy than a rigid-body-re-associated pose
        // stream; ride out IMU until the cascade heals or
        // `mocap_stale_ms` fires.
        match action {
            FailsafeAction::None => {}
            FailsafeAction::Disarmed(DisarmCause::JumpCascade { consecutive }) => {
                returned = MocapGuardOutcome::DisarmedJumpCascade { consecutive };
            }
            FailsafeAction::Disarmed(DisarmCause::RejectCascade { consecutive }) => {
                returned = MocapGuardOutcome::DisarmedRejectCascade { consecutive };
            }
            FailsafeAction::Disarmed(DisarmCause::NanAfterUpdate) => {
                returned = MocapGuardOutcome::DisarmedNanState;
            }
        }

        // --- Bounded re-anchor escape hatch (disarmed only) ---
        //
        // Riding out IMU is the right policy *while the cascade can
        // still heal*. It cannot heal on its own: `consecutive_jumps`
        // only clears on a non-jump outcome, and no pose can produce
        // one while it sits outside the jump gate. So once the filter's
        // position and the true position have separated by more than
        // `max_pos_jump_m`, every subsequent pose is rejected forever
        // and the vehicle is unarmable until a power cycle.
        //
        // The dominant real case is mundane: the uplink drops while
        // disarmed, the operator carries the airframe to the pad, the
        // uplink returns. Requiring the stream to agree with *itself*
        // across `reanchor_frames` before re-seeding distinguishes that
        // (a stationary airframe, poses clustered within centimetres)
        // from the re-association this guard exists to reject (poses
        // tracking a different, generally moving body). Gating on
        // `!armed` keeps the in-flight policy exactly as it was.
        let wedged =
            self.failsafe.snapshot().consecutive_jumps >= self.cfg.max_consecutive_jumps;
        if !armed && wedged {
            let agrees = self
                .reanchor_ref
                .is_some_and(|r| (pose.position - r).norm() <= self.cfg.reanchor_radius_m);
            if agrees {
                self.reanchor_count = self.reanchor_count.saturating_add(1);
            } else {
                self.reanchor_ref = Some(pose.position);
                self.reanchor_count = 1;
            }
            if self.reanchor_count >= self.cfg.reanchor_frames {
                self.reinit_filter(eskf, pose);
                self.clear_reanchor();
                return MocapGuardOutcome::Reinitialised {
                    at_pos: pose.position,
                    at_orient: pose.orientation,
                    cause: ReinitCause::JumpCascade,
                };
            }
        } else {
            self.clear_reanchor();
        }

        returned
    }

    fn clear_reanchor(&mut self) {
        self.reanchor_ref = None;
        self.reanchor_count = 0;
    }

    /// Drive the guard at the IMU/predict cadence. Convergence and
    /// staleness handling lives in the failsafe; this method
    /// re-shapes the failsafe's tick outcome into the wrapper-facing
    /// `TickOutcome`. `is_ready` collapses to `converged` because
    /// mocap has no quality-debounce flag (unlike GPS's RTK gate).
    pub fn on_predict_tick(&mut self, eskf: &Eskf, now_ms: u64) -> TickOutcome {
        let tick = self.failsafe.on_predict_tick(eskf, now_ms);
        TickOutcome {
            is_ready: tick.converged,
            is_stale: tick.is_stale,
        }
    }

    fn reinit_filter(&mut self, eskf: &mut Eskf, pose: &MocapPose) {
        eskf.init(
            pose.position,
            pose.orientation,
            Vector3::zeros(),
            Vector3::zeros(),
        );
        self.failsafe.note_reinit(pose.timestamp_ms);
    }
}

fn filter_reason(outcome: UpdateOutcome) -> FilterReason {
    match outcome {
        UpdateOutcome::InverseFailed => FilterReason::InverseFailed,
        UpdateOutcome::InflationCapExceeded => FilterReason::InflationCapExceeded,
        UpdateOutcome::NaNAfterUpdate => FilterReason::NaNAfterUpdate,
        UpdateOutcome::NotInitialized => FilterReason::NotInitialized,
        _ => FilterReason::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eskf::EskfConfig;

    fn make_eskf() -> Eskf {
        let mut e = Eskf::new(EskfConfig::default());
        e.init(
            Vector3::zeros(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
        );
        e
    }

    fn good_pose(t_ms: u64, pos: Vector3<f32>) -> MocapPose {
        MocapPose {
            position: pos,
            orientation: UnitQuaternion::identity(),
            timestamp_ms: t_ms,
        }
    }

    /// Drive a clean 100 Hz pose stream until the guard reports
    /// converged, returning the timestamp of the last accepted pose.
    /// Convergence is driven by the gyro-bias covariance trace falling
    /// below the threshold, which only re-evaluates on `on_predict_tick`
    /// — so ticks are interleaved at the pose rate.
    fn converge(g: &mut EskfMocapGuard, e: &mut Eskf) -> u64 {
        // Level hover: specific force is +g in body z, no rotation. The
        // gyro-bias covariance only shrinks through predict/update
        // coupling, so the IMU has to actually run.
        let accel = Vector3::new(0.0, 0.0, 9.81);
        let gyro = Vector3::zeros();
        let mut t = 0u64;
        for _ in 0..5_000 {
            for _ in 0..10 {
                e.predict(accel, gyro, 0.001);
            }
            t += 10;
            let _ = g.on_pose(e, &good_pose(t, Vector3::zeros()), false);
            let _ = g.on_predict_tick(e, t);
            if g.is_ready() {
                return t;
            }
        }
        panic!(
            "guard never converged on a clean stream (trace={})",
            e.gyro_bias_cov_trace()
        );
    }

    /// Mocap policy **in flight**: jump cascade at the limit DOES NOT
    /// re-init the filter. The wrapper translates the failsafe's
    /// `Disarmed{JumpCascade}` directly into `DisarmedJumpCascade` and
    /// the IMU-integrated estimate is preserved.
    ///
    /// Note the jump-cascade disarm is *not* gated on `converged` (only
    /// the reject cascade is — `failsafe.rs`), so no warmup is needed
    /// here; the counter alone drives it.
    #[test]
    fn jump_cascade_disarms_without_reinit() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Frames 2 m past the origin — above the 1.0 m jump gate in
        // EskfConfig::default(), so every one is JumpRejected. Armed,
        // so the re-anchor hatch is closed no matter how long the
        // stream agrees with itself.
        for k in 0..8 {
            let t = (k as u64) * 10;
            let _ = g.on_pose(&mut e, &good_pose(t, Vector3::new(2.0, 0.0, 0.0)), true);
        }
        // Filter should NOT have snapped to (2,0,0) — re-init on
        // jump cascade is the GPS-only policy. The mocap path
        // preserves the IMU-integrated estimate (still near origin
        // because the predict was never run).
        let pos = e.position();
        assert!(
            (pos - Vector3::zeros()).norm() < 0.5,
            "mocap guard must NOT re-init on jump cascade (GPS policy); pos={pos:?}"
        );
        assert!(g.snapshot().jump_total >= 3);
    }

    #[test]
    fn nan_state_on_entry_reinits_at_pose() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Force the filter into the non-initialised state by hand —
        // public API doesn't have a mutator for this without driving
        // the predict to NaN, which a unit test shouldn't do.
        // Instead, replace `e` with a fresh non-initialised one.
        let mut e_uninit = Eskf::new(EskfConfig::default());
        assert!(!e_uninit.is_initialized());

        let pose = good_pose(100, Vector3::new(1.0, 2.0, 3.0));
        let outcome = g.on_pose(&mut e_uninit, &pose, false);
        assert!(
            matches!(
                outcome,
                MocapGuardOutcome::Reinitialised {
                    cause: ReinitCause::NanState,
                    ..
                }
            ),
            "expected Reinitialised{{NanState}}, got {outcome:?}"
        );
        assert!(e_uninit.is_initialized());
        // Filter was re-seeded at the pose.
        let pos = e_uninit.position();
        assert!((pos - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-5);
        let _ = g.snapshot(); // sanity: doesn't panic
        // Suppress unused
        let _ = e;
    }

    #[test]
    fn non_finite_pose_is_rejected() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        let pose = MocapPose {
            position: Vector3::new(f32::NAN, 0.0, 0.0),
            orientation: UnitQuaternion::identity(),
            timestamp_ms: 100,
        };
        let outcome = g.on_pose(&mut e, &pose, false);
        assert!(
            matches!(outcome, MocapGuardOutcome::NonFinitePose),
            "expected NonFinitePose, got {outcome:?}"
        );
    }

    #[test]
    fn accepted_pose_bumps_both_pos_and_att_telemetry() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Tiny pos delta — well below the 1.0 m jump gate.
        let _ = g.on_pose(&mut e, &good_pose(10, Vector3::new(0.001, 0.0, 0.0)), false);
        let s = g.snapshot();
        // Inflated counters depend on whether the residual triggered
        // R-inflation — for a 1 mm offset against MOCAP_POS_STD=0.01
        // it shouldn't, so we assert the *non*-inflated path.
        assert_eq!(s.pos_inflated_total, 0);
        assert_eq!(s.att_inflated_total, 0);
        assert_eq!(s.pos_reject_total, 0);
        assert_eq!(s.att_reject_total, 0);
        // last_pose_accept_ms refreshed.
        assert_eq!(s.last_pose_accept_ms, 10);
    }

    /// The escape hatch: disarmed, a wedged filter re-anchors once the
    /// pose stream has agreed with itself for `reanchor_frames`. Without
    /// this the jump cascade is unrecoverable short of a power cycle.
    #[test]
    fn disarmed_jump_cascade_reanchors_on_agreeing_stream() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        let far = Vector3::new(3.0, 0.0, 0.0);

        // Wedge it: 3 jumpy frames trip the cascade (max_consecutive_jumps).
        for k in 0..3 {
            let o = g.on_pose(&mut e, &good_pose((k as u64) * 10, far), false);
            assert!(
                !matches!(o, MocapGuardOutcome::Reinitialised { .. }),
                "re-anchored before the agreement run completed: {o:?}"
            );
        }
        assert!(g.snapshot().consecutive_jumps >= cfg.max_consecutive_jumps);

        // Now the stream agrees with itself. The run needs
        // reanchor_frames poses: the first seeds the reference, so the
        // hatch fires on the reanchor_frames'th agreeing frame.
        let mut reanchored = false;
        for k in 3..(3 + DEFAULT_REANCHOR_FRAMES + 2) {
            let o = g.on_pose(&mut e, &good_pose((k as u64) * 10, far), false);
            if matches!(
                o,
                MocapGuardOutcome::Reinitialised {
                    cause: ReinitCause::JumpCascade,
                    ..
                }
            ) {
                reanchored = true;
                break;
            }
        }
        assert!(reanchored, "disarmed wedge never re-anchored");
        // Filter re-seeded at the pose, cascade counters cleared.
        assert!((e.position() - far).norm() < 1e-5);
        assert_eq!(g.snapshot().consecutive_jumps, 0);
        assert!(!g.is_ready(), "a freshly re-anchored filter is not converged");
    }

    /// The hatch is disarmed-only. Armed, the same agreeing stream must
    /// never capture the filter — that is the re-association case the
    /// ride-out-IMU policy exists to reject.
    #[test]
    fn armed_jump_cascade_never_reanchors() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        let far = Vector3::new(3.0, 0.0, 0.0);
        for k in 0..(DEFAULT_REANCHOR_FRAMES * 4) {
            let o = g.on_pose(&mut e, &good_pose((k as u64) * 10, far), true);
            assert!(
                !matches!(o, MocapGuardOutcome::Reinitialised { .. }),
                "armed re-anchor at frame {k}: {o:?}"
            );
        }
        assert!(
            (e.position() - Vector3::zeros()).norm() < 0.5,
            "armed filter must keep the IMU estimate; pos={:?}",
            e.position()
        );
    }

    /// A stream that does *not* agree with itself — the moving,
    /// misidentified rigid body — never satisfies the hatch even
    /// disarmed.
    #[test]
    fn disarmed_reanchor_requires_self_agreement() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        for k in 0..(DEFAULT_REANCHOR_FRAMES * 4) {
            // Each frame steps well beyond reanchor_radius_m from the
            // last, so the agreement run resets every time.
            let pos = Vector3::new(3.0 + (k as f32) * 0.5, 0.0, 0.0);
            let o = g.on_pose(&mut e, &good_pose((k as u64) * 10, pos), false);
            assert!(
                !matches!(o, MocapGuardOutcome::Reinitialised { .. }),
                "re-anchored on a disagreeing stream at frame {k}: {o:?}"
            );
        }
    }

    /// Mocap twin of `gps_guard::staleness_drops_ready`. This is the
    /// arming-gate half of the staleness contract; the *publishing* half
    /// (odometry keeps flowing while stale) lives in the firmware task.
    #[test]
    fn mocap_staleness_drops_ready() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        converge(&mut g, &mut e);

        let stale_t = g.snapshot().last_pose_accept_ms + cfg.mocap_stale_ms + 10;
        let tick = g.on_predict_tick(&e, stale_t);
        assert!(tick.is_stale);
        assert!(!tick.is_ready);
    }

    /// Mocap twin of `gps_guard::staleness_clears_on_fresh_accept` —
    /// staleness must be recoverable, not a latch.
    #[test]
    fn mocap_staleness_clears_on_fresh_accept() {
        let cfg = MocapGuardConfig::default();
        let mut g = EskfMocapGuard::new(cfg, 0);
        let mut e = make_eskf();
        converge(&mut g, &mut e);

        let stale_t = g.snapshot().last_pose_accept_ms + cfg.mocap_stale_ms + 10;
        assert!(g.on_predict_tick(&e, stale_t).is_stale);

        // A fresh accepted pose re-arms the staleness clock.
        let _ = g.on_pose(&mut e, &good_pose(stale_t, Vector3::zeros()), false);
        assert!(!g.on_predict_tick(&e, stale_t + 1).is_stale);
    }

    // Reject-cascade coverage lives in `failsafe.rs`, which can hand
    // `record_pos_outcome` a synthetic `UpdateOutcome`. Reaching it
    // through this wrapper would need a contrived singular update, and
    // the only mocap-specific part is the outcome mapping to
    // `DisarmedRejectCascade`.
}
