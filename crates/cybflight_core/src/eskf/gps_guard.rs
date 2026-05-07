//! GPS / NAV-PVT failsafe wrapper around [`EskfFailsafe`].
//!
//! Owns only the GPS-specific bits:
//!
//! - PVT usability gate (fix type, SV count, h_acc).
//! - Carrier-solution-scaled σ floor for the position update.
//! - Re-init policy at jump-cascade limit (gated on `carr_soln >=
//!   reinit_min_carr_soln`).
//! - Debounced RTK quality gate (carr_soln ≥ 2 sustained for fix /
//!   carr_soln < 2 sustained for loss).
//! - Source-specific telemetry snapshot fields (`last_carr_soln`,
//!   `last_num_sv`, `last_h_acc_mm`).
//!
//! Cross-cutting failsafe behaviour — jump cascade, reject cascade,
//! NaN-after-update, staleness, gyro-bias-cov convergence, telemetry
//! counters — lives in [`super::failsafe::EskfFailsafe`] and is shared
//! with future sources (mocap, etc.). The guard composes it.
//!
//! The guard does **not** own the `Eskf`, the wall clock, or any logging
//! sink. It takes `&mut Eskf` per call, takes `now_ms: u64` from the
//! caller, and returns structured outcome enums the caller (firmware
//! Embassy task or host-side sim runner) can dispatch on.

use nalgebra::{UnitQuaternion, Vector3};

use super::failsafe::{
    ConvergenceAxes, DisarmCause, EskfFailsafe, EskfFailsafeConfig, FailsafeAction,
};
use super::{Eskf, UpdateOutcome};

/// One incoming PVT translated into the ESKF's ENU frame. The geodetic
/// conversion (LLH → ENU) is the firmware wrapper's responsibility; the
/// sim already lives in ENU and synthesises the RTK quality fields.
#[derive(Clone, Copy, Debug)]
pub struct GpsFix {
    pub enu_pos: Vector3<f32>,
    pub enu_vel: Vector3<f32>,
    pub h_acc_mm: u32,
    pub v_acc_mm: u32,
    pub s_acc_mm_s: u32,
    pub num_sv: u8,
    pub fix_type: u8,
    pub carr_soln: u8,
    pub timestamp_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsabilityReason {
    LowFixType,
    InsufficientSv,
    HorizontalAccuracyTooLoose,
    NonFinitePosition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterReason {
    InverseFailed,
    InflationCapExceeded,
    NaNAfterUpdate,
    NotInitialized,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReinitCause {
    /// `Eskf::is_initialized()` returned false on entry — propagated NaN
    /// from a corrupted predict.
    NanState,
    /// Filter survived predict but `consecutive_jumps >=
    /// max_consecutive_jumps` and the fix is RTK-fixed enough to seed a
    /// re-init from. (Phase 5 — not emitted in Phase 1.)
    JumpCascade,
}

/// Result of `EskfGpsGuard::on_pvt`. The guard collapses the multi-step
/// PVT pipeline (usability → update → cascade → debounce) into a single
/// structured outcome the wrapper can match on for logging / telemetry.
///
/// Disarmed* variants additionally clear `converged` so `is_ready()`
/// reports false; they do not by themselves restart the filter. Callers
/// that need a re-init see `Reinitialised`.
#[derive(Clone, Copy, Debug)]
pub enum GpsGuardOutcome {
    PvtAccepted {
        inflated: bool,
        sigma_pos_m: f32,
    },
    PvtRejectedUsability {
        reason: UsabilityReason,
    },
    PvtRejectedJump {
        consecutive: u32,
    },
    PvtRejectedFilter {
        reason: FilterReason,
    },
    Reinitialised {
        at_enu: Vector3<f32>,
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

/// Result of `EskfGpsGuard::on_predict_tick`. `is_ready` is the
/// `converged && rtk_quality_ok` AND that the wrapper publishes to the
/// arming-gate atomic. `is_stale` tells the wrapper to *skip* odometry
/// publishing this tick (mirrors the firmware's `continue` when GPS is
/// stale).
#[derive(Clone, Copy, Debug)]
pub struct TickOutcome {
    pub is_ready: bool,
    pub is_stale: bool,
}

/// Read-only telemetry view of the guard's internal state. Callers use
/// this to populate logs, the shell `ESTIMATOR_STATUS` enum, or sim
/// `StepRecord`s.
#[derive(Clone, Copy, Debug)]
pub struct GuardSnapshot {
    pub converged: bool,
    pub rtk_quality_ok: bool,
    pub consecutive_jumps: u32,
    pub consecutive_rejects: u32,
    pub last_carr_soln: u8,
    pub last_num_sv: u8,
    pub last_h_acc_mm: u32,
    pub pos_reject_total: u32,
    pub pos_inflated_total: u32,
    pub jump_total: u32,
    /// Cumulative count of velocity updates rejected by the filter
    /// (inverse-failed, inflation-cap exceeded, NaN-after-update).
    /// Counted independently of position rejections — vel and pos
    /// are separate measurements that can fail for different reasons.
    pub vel_reject_total: u32,
    /// Cumulative count of velocity updates that were absorbed with
    /// `R` inflated to admit a large innovation. High values here
    /// suggest the reported `s_acc_mm_s` is over-confident.
    pub vel_inflated_total: u32,
    pub last_gps_accept_ms: u64,
}

/// Guard tunables. `Default` mirrors the constants currently embedded in
/// `crates/cybflight/src/estimation/eskf_imu_gps.rs`.
#[derive(Clone, Copy, Debug)]
pub struct GpsGuardConfig {
    pub max_consecutive_jumps: u32,
    pub max_consecutive_rejects: u32,
    pub gps_stale_ms: u64,
    pub rtk_fix_debounce_ms: u64,
    pub rtk_loss_debounce_ms: u64,
    pub gyro_bias_cov_trace_xy_thresh: f32,
    pub init_yaw_cov: f32,
    pub gps_min_sv: u8,
    pub gps_h_acc_max_mm: u32,
    pub pos_sigma_floor_fix_m: f32,
    pub pos_sigma_floor_float_m: f32,
    pub pos_sigma_floor_none_m: f32,
    pub vel_sigma_floor_m_s: f32,
    /// Minimum `carr_soln` for a jump-cascade-triggered re-init to fire.
    /// Default 2 = RTK-fixed only — re-seeding the ENU state from a
    /// stand-alone fix would inject metre-scale absolute bias relative
    /// to the carr_soln=2 origin anchor. Receivers without RTK should
    /// drop this to 0.
    pub reinit_min_carr_soln: u8,
}

impl Default for GpsGuardConfig {
    fn default() -> Self {
        Self {
            max_consecutive_jumps: 2,
            max_consecutive_rejects: 5,
            gps_stale_ms: 2_000,
            rtk_fix_debounce_ms: 2_000,
            rtk_loss_debounce_ms: 1_000,
            gyro_bias_cov_trace_xy_thresh: 0.002,
            init_yaw_cov: 10.0,
            gps_min_sv: 6,
            gps_h_acc_max_mm: 50_000,
            pos_sigma_floor_fix_m: 0.05,
            pos_sigma_floor_float_m: 0.30,
            pos_sigma_floor_none_m: 2.0,
            vel_sigma_floor_m_s: 0.10,
            reinit_min_carr_soln: 2,
        }
    }
}

impl GpsGuardConfig {
    /// Explicit constructor. The host simulation uses this so future
    /// safety-tuning changes (e.g. tightening
    /// `max_consecutive_jumps`, or relaxing `reinit_min_carr_soln` for
    /// non-RTK builds) don't silently shift the regression snapshot.
    /// Firmware code may keep using `Default::default()` — it *wants*
    /// updated defaults.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_consecutive_jumps: u32,
        max_consecutive_rejects: u32,
        gps_stale_ms: u64,
        rtk_fix_debounce_ms: u64,
        rtk_loss_debounce_ms: u64,
        gyro_bias_cov_trace_xy_thresh: f32,
        init_yaw_cov: f32,
        gps_min_sv: u8,
        gps_h_acc_max_mm: u32,
        pos_sigma_floor_fix_m: f32,
        pos_sigma_floor_float_m: f32,
        pos_sigma_floor_none_m: f32,
        vel_sigma_floor_m_s: f32,
        reinit_min_carr_soln: u8,
    ) -> Self {
        Self {
            max_consecutive_jumps,
            max_consecutive_rejects,
            gps_stale_ms,
            rtk_fix_debounce_ms,
            rtk_loss_debounce_ms,
            gyro_bias_cov_trace_xy_thresh,
            init_yaw_cov,
            gps_min_sv,
            gps_h_acc_max_mm,
            pos_sigma_floor_fix_m,
            pos_sigma_floor_float_m,
            pos_sigma_floor_none_m,
            vel_sigma_floor_m_s,
            reinit_min_carr_soln,
        }
    }
}

pub struct EskfGpsGuard {
    cfg: GpsGuardConfig,
    /// Cross-cutting failsafe — owns convergence, cascade tracking,
    /// staleness, NaN-after-update, and all the telemetry counters
    /// (pos / att / vel / jump). The wrapper here adds GPS-specific
    /// concerns on top.
    failsafe: EskfFailsafe,
    rtk_quality_ok: bool,
    rtk_fix_streak_start_ms: Option<u64>,
    rtk_loss_streak_start_ms: Option<u64>,
    last_carr_soln: u8,
    last_num_sv: u8,
    last_h_acc_mm: u32,
}

impl EskfGpsGuard {
    /// `anchor_ms` is the timestamp of the origin-anchor PVT (which by
    /// construction was carr_soln ≥ 2 — see [`is_pvt_origin_anchor`]).
    /// Seeds the staleness clock and starts the RTK fix streak ticking.
    pub fn new(cfg: GpsGuardConfig, anchor_ms: u64) -> Self {
        let failsafe_cfg = EskfFailsafeConfig::new(
            cfg.max_consecutive_jumps,
            cfg.max_consecutive_rejects,
            cfg.gps_stale_ms,
            cfg.gyro_bias_cov_trace_xy_thresh,
            // GPS-only build can't observe yaw — use 2-axis trace.
            ConvergenceAxes::XyOnly,
        );
        Self {
            cfg,
            failsafe: EskfFailsafe::new(failsafe_cfg, anchor_ms),
            rtk_quality_ok: false,
            rtk_fix_streak_start_ms: Some(anchor_ms),
            rtk_loss_streak_start_ms: None,
            last_carr_soln: 2,
            last_num_sv: 0,
            last_h_acc_mm: 0,
        }
    }

    pub fn config(&self) -> &GpsGuardConfig {
        &self.cfg
    }

    pub fn is_ready(&self) -> bool {
        self.failsafe.converged() && self.rtk_quality_ok
    }

    pub fn snapshot(&self) -> GuardSnapshot {
        let f = self.failsafe.snapshot();
        GuardSnapshot {
            converged: f.converged,
            rtk_quality_ok: self.rtk_quality_ok,
            consecutive_jumps: f.consecutive_jumps,
            consecutive_rejects: f.consecutive_rejects,
            last_carr_soln: self.last_carr_soln,
            last_num_sv: self.last_num_sv,
            last_h_acc_mm: self.last_h_acc_mm,
            pos_reject_total: f.pos_reject_total,
            pos_inflated_total: f.pos_inflated_total,
            jump_total: f.jump_total,
            vel_reject_total: f.vel_reject_total,
            vel_inflated_total: f.vel_inflated_total,
            last_gps_accept_ms: f.last_accept_ms,
        }
    }

    /// Drive the guard with one incoming PVT. Mirrors the GPS-branch
    /// pipeline of the firmware task: usability gate → telemetry
    /// snapshot → RTK streak update → NaN re-init → ESKF update → jump
    /// cascade → reject cascade → post-update NaN check → RTK debounce.
    pub fn on_pvt(&mut self, eskf: &mut Eskf, fix: &GpsFix) -> GpsGuardOutcome {
        if let Some(reason) = usability_reason(fix, self.cfg.gps_min_sv, self.cfg.gps_h_acc_max_mm)
        {
            // A stream of unusable PVTs is itself a signal that RTK
            // quality has degraded — treat as a loss event for the
            // streak so `rtk_quality_ok` actually flips, rather than
            // freezing on the last good value and waiting for
            // `gps_stale_ms` to fire as the catch-all. The receiver-
            // reported `carr_soln` field is meaningless on a fix that
            // failed our own usability gate (low SV count, h_acc
            // explosion, non-finite position), so we override
            // unconditionally rather than reading it.
            self.rtk_fix_streak_start_ms = None;
            if self.rtk_loss_streak_start_ms.is_none() {
                self.rtk_loss_streak_start_ms = Some(fix.timestamp_ms);
            }
            if self.rtk_quality_ok
                && let Some(t) = self.rtk_loss_streak_start_ms
                && fix.timestamp_ms.saturating_sub(t) > self.cfg.rtk_loss_debounce_ms
            {
                self.rtk_quality_ok = false;
            }
            return GpsGuardOutcome::PvtRejectedUsability { reason };
        }

        // Telemetry snapshot — only updated on usable PVTs (matches the
        // firmware task: a stream of unusable frames freezes these
        // fields and lets staleness be the catch-all).
        self.last_carr_soln = fix.carr_soln;
        self.last_num_sv = fix.num_sv;
        self.last_h_acc_mm = fix.h_acc_mm;

        // RTK streak tracking. The debounce eval at the end of this
        // function reads these timestamps.
        if fix.carr_soln >= 2 {
            self.rtk_loss_streak_start_ms = None;
            if self.rtk_fix_streak_start_ms.is_none() {
                self.rtk_fix_streak_start_ms = Some(fix.timestamp_ms);
            }
        } else {
            self.rtk_fix_streak_start_ms = None;
            if self.rtk_loss_streak_start_ms.is_none() {
                self.rtk_loss_streak_start_ms = Some(fix.timestamp_ms);
            }
        }

        // NaN re-init on entry. Mid-flight non-finite state: re-seed
        // at the *current* GPS pos, not the LLH origin (re-anchoring
        // at origin would teleport the estimate back to launch).
        if !eskf.is_initialized() {
            self.reinit_filter(eskf, fix);
            return GpsGuardOutcome::Reinitialised {
                at_enu: fix.enu_pos,
                cause: ReinitCause::NanState,
            };
        }

        // σ derived from u-blox accuracy with a carr_soln-scaled floor.
        let sigma_pos = ((fix.h_acc_mm.max(fix.v_acc_mm) as f32) * 1e-3)
            .max(self.pos_sigma_floor(fix.carr_soln));

        let pos_outcome = eskf.update_pos_sparse(fix.enu_pos, sigma_pos);

        // Velocity update — runs only when the position update was
        // accepted (an RTK ambiguity slip biases pos+vel together;
        // the failsafe story belongs to the pos channel). σ_vel from
        // u-blox `s_acc_mm_s` with the cfg-level floor.
        if pos_outcome.is_accepted() {
            let sigma_vel =
                ((fix.s_acc_mm_s as f32) * 1e-3).max(self.cfg.vel_sigma_floor_m_s);
            let vel_outcome = eskf.update_vel(fix.enu_vel, sigma_vel);
            self.failsafe.record_vel_outcome(vel_outcome);
        }

        // Hand the pos outcome to the cross-cutting failsafe. It bumps
        // counters, drives the cascade gates, and runs the post-update
        // NaN check. Returns either `None` (nothing extra to do) or
        // `Disarmed(cause)` — we translate `cause` into the
        // GPS-specific outcome enum below.
        let action = self.failsafe.record_pos_outcome(eskf, pos_outcome, fix.timestamp_ms);

        // Headline outcome from the pos update — used when the
        // failsafe didn't disarm. `JumpRejected` carries the live
        // cascade counter for the warning log.
        let mut returned = match pos_outcome {
            UpdateOutcome::Accepted { inflated } => GpsGuardOutcome::PvtAccepted {
                inflated,
                sigma_pos_m: sigma_pos,
            },
            UpdateOutcome::JumpRejected => GpsGuardOutcome::PvtRejectedJump {
                consecutive: self.failsafe.snapshot().consecutive_jumps,
            },
            other => GpsGuardOutcome::PvtRejectedFilter {
                reason: filter_reason(other),
            },
        };

        // Failsafe-driven overrides. JumpCascade is the GPS-specific
        // decision point: re-init at the offending PVT if its carrier
        // solution is trustworthy (carr_soln >= reinit_min_carr_soln);
        // otherwise stay disarmed and wait for the cascade to heal
        // (multipath clears / RTK re-locks) or `gps_stale_ms` to fire.
        match action {
            FailsafeAction::None => {}
            FailsafeAction::Disarmed(DisarmCause::JumpCascade { consecutive }) => {
                if fix.carr_soln >= self.cfg.reinit_min_carr_soln {
                    self.reinit_filter(eskf, fix);
                    return GpsGuardOutcome::Reinitialised {
                        at_enu: fix.enu_pos,
                        cause: ReinitCause::JumpCascade,
                    };
                }
                returned = GpsGuardOutcome::DisarmedJumpCascade { consecutive };
            }
            FailsafeAction::Disarmed(DisarmCause::RejectCascade { consecutive }) => {
                returned = GpsGuardOutcome::DisarmedRejectCascade { consecutive };
            }
            FailsafeAction::Disarmed(DisarmCause::NanAfterUpdate) => {
                returned = GpsGuardOutcome::DisarmedNanState;
            }
        }

        // RTK debounce evaluation. Two-direction transition:
        //   ok=false → true on rtk_fix_debounce_ms sustained carr_soln≥2.
        //   ok=true  → false on rtk_loss_debounce_ms sustained carr_soln<2.
        let now = fix.timestamp_ms;
        if self.rtk_quality_ok {
            if let Some(t) = self.rtk_loss_streak_start_ms
                && now.saturating_sub(t) > self.cfg.rtk_loss_debounce_ms
            {
                self.rtk_quality_ok = false;
            }
        } else if let Some(t) = self.rtk_fix_streak_start_ms
            && now.saturating_sub(t) > self.cfg.rtk_fix_debounce_ms
        {
            self.rtk_quality_ok = true;
        }

        returned
    }

    /// Drive the guard at the IMU/predict cadence. Convergence and
    /// staleness handling lives in the failsafe; this method
    /// translates the failsafe's tick outcome into the GPS-guard-
    /// flavoured `TickOutcome` (which adds `is_ready = converged AND
    /// rtk_quality_ok` for the wrapper's `ESTIMATOR_READY` atomic).
    pub fn on_predict_tick(&mut self, eskf: &Eskf, now_ms: u64) -> TickOutcome {
        let tick = self.failsafe.on_predict_tick(eskf, now_ms);
        TickOutcome {
            is_ready: tick.converged && self.rtk_quality_ok,
            is_stale: tick.is_stale,
        }
    }

    fn pos_sigma_floor(&self, carr_soln: u8) -> f32 {
        match carr_soln {
            2 => self.cfg.pos_sigma_floor_fix_m,
            1 => self.cfg.pos_sigma_floor_float_m,
            _ => self.cfg.pos_sigma_floor_none_m,
        }
    }

    fn reinit_filter(&mut self, eskf: &mut Eskf, fix: &GpsFix) {
        eskf.init_with_cov(
            fix.enu_pos,
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
            Vector3::new(0.1, 0.1, self.cfg.init_yaw_cov),
        );
        self.failsafe.note_reinit(fix.timestamp_ms);
    }
}

fn usability_reason(fix: &GpsFix, min_sv: u8, h_acc_max_mm: u32) -> Option<UsabilityReason> {
    if fix.fix_type < 3 {
        Some(UsabilityReason::LowFixType)
    } else if fix.num_sv < min_sv {
        Some(UsabilityReason::InsufficientSv)
    } else if fix.h_acc_mm > h_acc_max_mm {
        Some(UsabilityReason::HorizontalAccuracyTooLoose)
    } else if !fix.enu_pos.iter().all(|v| v.is_finite()) {
        Some(UsabilityReason::NonFinitePosition)
    } else {
        None
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

/// Predicate the caller uses to gate the *origin-anchor* PVT — stricter
/// than per-update usability: requires `carr_soln ≥ 2` so the ENU frame
/// is centimeter-accurate from the start. Float-RTK and stand-alone
/// fixes carry metre-scale absolute bias that would bake a permanent
/// offset into every subsequent setpoint.
pub fn is_pvt_origin_anchor(fix: &GpsFix, min_sv: u8, h_acc_max_mm: u32) -> bool {
    usability_reason(fix, min_sv, h_acc_max_mm).is_none() && fix.carr_soln >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eskf::EskfConfig;

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

    fn good_fix(t_ms: u64, pos: Vector3<f32>) -> GpsFix {
        GpsFix {
            enu_pos: pos,
            enu_vel: Vector3::zeros(),
            h_acc_mm: 50,
            v_acc_mm: 80,
            s_acc_mm_s: 100,
            num_sv: 12,
            fix_type: 3,
            carr_soln: 2,
            timestamp_ms: t_ms,
        }
    }

    /// A clean stream of carr_soln=2 fixes raises rtk_quality_ok after
    /// the debounce window elapses.
    #[test]
    fn rtk_debounce_acquires_after_window() {
        let cfg = GpsGuardConfig::default();
        let mut g = EskfGpsGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Step in 200 ms increments through the 2 s debounce window.
        for k in 0..15 {
            let t = (k as u64) * 200;
            let _ = g.on_pvt(&mut e, &good_fix(t, Vector3::zeros()));
        }
        assert!(g.snapshot().rtk_quality_ok);
    }

    /// At `max_consecutive_jumps`, a carr_soln=2 fix triggers a
    /// re-init at the offending PVT. The filter snaps to the new
    /// position; the cascade counter is reset to 0; subsequent
    /// matching fixes are accepted normally.
    #[test]
    fn jump_cascade_with_rtk_fix_triggers_reinit() {
        let cfg = GpsGuardConfig::default();
        let mut g = EskfGpsGuard::new(cfg, 0);
        let mut e = make_eskf();
        for k in 0..15 {
            let _ = g.on_pvt(&mut e, &good_fix((k as u64) * 200, Vector3::zeros()));
        }
        assert_eq!(g.snapshot().consecutive_jumps, 0);

        // Two consecutive jumpy fixes 5 m east of origin. The first
        // is `JumpRejected` and bumps the counter to 1. The second
        // hits the cascade limit (`MAX_CONSECUTIVE_JUMPS = 2`) and
        // triggers re-init.
        let _ = g.on_pvt(&mut e, &good_fix(3000, Vector3::new(5.0, 0.0, 0.0)));
        let outcome = g.on_pvt(&mut e, &good_fix(3200, Vector3::new(5.0, 0.0, 0.0)));
        assert!(
            matches!(
                outcome,
                GpsGuardOutcome::Reinitialised {
                    cause: ReinitCause::JumpCascade,
                    ..
                }
            ),
            "expected Reinitialised{{JumpCascade}}, got {outcome:?}"
        );
        // Filter should now be at the new GPS position.
        let pos = e.position();
        assert!(
            (pos - Vector3::new(5.0, 0.0, 0.0)).norm() < 0.1,
            "filter not re-anchored: pos={pos:?}"
        );
        // Cascade counter cleared so subsequent jumps don't double-count.
        assert_eq!(g.snapshot().consecutive_jumps, 0);
    }

    /// At cascade limit but with `carr_soln < reinit_min_carr_soln`,
    /// re-init is refused and we fall back to disarming. This
    /// preserves the safety invariant that we never seed from a
    /// known-biased measurement source.
    #[test]
    fn jump_cascade_without_rtk_fix_disarms_only() {
        let cfg = GpsGuardConfig::default();
        let mut g = EskfGpsGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Use carr_soln=0 (stand-alone) fixes throughout — below the
        // default `reinit_min_carr_soln=2` threshold.
        let standalone = |t_ms: u64, pos: Vector3<f32>| GpsFix {
            carr_soln: 0,
            ..good_fix(t_ms, pos)
        };
        // Warm up enough to flip `converged` artificially via a
        // direct mutation isn't easy in this test; instead, drive the
        // cascade with `converged=false`. The test's value is purely
        // to confirm the re-init path is *not* taken on stand-alone.
        let _ = g.on_pvt(&mut e, &standalone(3000, Vector3::new(5.0, 0.0, 0.0)));
        let outcome = g.on_pvt(&mut e, &standalone(3200, Vector3::new(5.0, 0.0, 0.0)));
        // No Reinitialised — the carrier solution didn't permit it.
        assert!(
            !matches!(outcome, GpsGuardOutcome::Reinitialised { .. }),
            "stand-alone fix must NOT trigger re-init: {outcome:?}"
        );
        // Filter is unchanged at origin.
        let pos = e.position();
        assert!(
            pos.norm() < 0.5,
            "filter should NOT have re-anchored at biased PVT: pos={pos:?}"
        );
    }

    /// Staleness flips ready=false in `on_predict_tick`.
    #[test]
    fn staleness_drops_ready() {
        let cfg = GpsGuardConfig::default();
        let mut g = EskfGpsGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Force the converged + rtk paths via a long good stream.
        for k in 0..15 {
            let _ = g.on_pvt(&mut e, &good_fix((k as u64) * 200, Vector3::zeros()));
        }
        // No new PVTs; advance the predict clock past gps_stale_ms.
        let stale_t = g.snapshot().last_gps_accept_ms + cfg.gps_stale_ms + 100;
        let tick = g.on_predict_tick(&e, stale_t);
        assert!(tick.is_stale);
        assert!(!tick.is_ready);
    }

    /// Sustained carr_soln<2 after a period of carr_soln=2 flips
    /// `rtk_quality_ok` from true to false (loss-debounce direction).
    /// Mirror of `rtk_debounce_acquires_after_window`.
    #[test]
    fn rtk_debounce_loses_after_window() {
        let cfg = GpsGuardConfig::default();
        let mut g = EskfGpsGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Acquire RTK quality first.
        for k in 0..15 {
            let _ = g.on_pvt(&mut e, &good_fix((k as u64) * 200, Vector3::zeros()));
        }
        assert!(g.snapshot().rtk_quality_ok, "setup: should have acquired");

        // Now stream carr_soln=1 (float-RTK) fixes for >rtk_loss_debounce_ms.
        // Use small position drift so we don't trip the jump gate.
        let float_fix = |t_ms: u64| GpsFix {
            carr_soln: 1,
            ..good_fix(t_ms, Vector3::zeros())
        };
        let base = 15 * 200u64;
        for k in 0..10 {
            let t = base + (k as u64) * 200; // 2 s window
            let _ = g.on_pvt(&mut e, &float_fix(t));
        }
        assert!(
            !g.snapshot().rtk_quality_ok,
            "rtk_quality_ok should have flipped false after sustained carr_soln<2"
        );
    }

    /// Audit Medium #5 regression: a stream of *unusable* PVTs (e.g.
    /// h_acc explosion) — even ones that report carr_soln=2 — must
    /// flip `rtk_quality_ok` to false via the loss-debounce. Without
    /// this, RTK quality telemetry freezes on the last good value
    /// until `gps_stale_ms` fires.
    #[test]
    fn unusable_pvts_flip_rtk_quality_false() {
        let cfg = GpsGuardConfig::default();
        let mut g = EskfGpsGuard::new(cfg, 0);
        let mut e = make_eskf();
        for k in 0..15 {
            let _ = g.on_pvt(&mut e, &good_fix((k as u64) * 200, Vector3::zeros()));
        }
        assert!(g.snapshot().rtk_quality_ok, "setup: should have acquired");

        // Stream PVTs that pass through the receiver as carr_soln=2
        // but fail our usability gate (h_acc 100 m, way above the
        // 50 m threshold). Each is rejected with PvtRejectedUsability.
        let unusable = |t_ms: u64| GpsFix {
            h_acc_mm: 100_000,
            ..good_fix(t_ms, Vector3::zeros())
        };
        let base = 15 * 200u64;
        for k in 0..10 {
            let t = base + (k as u64) * 200;
            let outcome = g.on_pvt(&mut e, &unusable(t));
            assert!(
                matches!(outcome, GpsGuardOutcome::PvtRejectedUsability { .. }),
                "expected PvtRejectedUsability, got {outcome:?}"
            );
        }
        assert!(
            !g.snapshot().rtk_quality_ok,
            "rtk_quality_ok should have flipped false on a stream of \
             unusable PVTs (audit Medium #5)"
        );
    }

    /// After staleness fires, a fresh accepted PVT updates
    /// `last_gps_accept_ms` so subsequent ticks no longer report
    /// stale. The estimator can then re-converge through the normal
    /// gyro-bias-cov pathway.
    #[test]
    fn staleness_clears_on_fresh_accept() {
        let cfg = GpsGuardConfig::default();
        let mut g = EskfGpsGuard::new(cfg, 0);
        let mut e = make_eskf();
        // Acquire RTK quality.
        for k in 0..15 {
            let _ = g.on_pvt(&mut e, &good_fix((k as u64) * 200, Vector3::zeros()));
        }
        // Drive into the stale state via on_predict_tick.
        let stale_t = g.snapshot().last_gps_accept_ms + cfg.gps_stale_ms + 100;
        let stale_tick = g.on_predict_tick(&e, stale_t);
        assert!(stale_tick.is_stale);

        // Fresh PVT arrives at `stale_t`. Should be accepted and bump
        // last_gps_accept_ms to fresh_t. Tick at fresh_t+1 ms must not
        // report stale.
        let fresh_t = stale_t + 50;
        let outcome = g.on_pvt(&mut e, &good_fix(fresh_t, Vector3::zeros()));
        assert!(
            matches!(outcome, GpsGuardOutcome::PvtAccepted { .. }),
            "fresh PVT should be accepted: {outcome:?}"
        );
        assert_eq!(g.snapshot().last_gps_accept_ms, fresh_t);

        let recover_tick = g.on_predict_tick(&e, fresh_t + 1);
        assert!(
            !recover_tick.is_stale,
            "ticking just after a fresh accept must not report stale"
        );
    }
}
