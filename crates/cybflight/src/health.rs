//! System-health snapshot — single source of truth for "is this airframe
//! ready to fly".
//!
//! All inputs the arming gates (`sensors/rc.rs::ArmStateMachine`) consult
//! are aggregated here so that:
//!
//! 1. The `health` shell command can answer "how f*cked are we?" without
//!    duplicating gate logic, and
//! 2. The latched `LATEST_BLOCK_REASON` from the last arm attempt is
//!    surfaced alongside the live snapshot — useful when the per-attempt
//!    summary line has already scrolled out of the defmt buffer.
//!
//! ## Why fold "GPS health" in here rather than ship a separate `gpshealth`
//!
//! GPS quality matters only insofar as it lets the ESKF anchor its position
//! state. A "happy GPS" with stale fixes or low SV count that nonetheless
//! left the ESKF degraded is *not* healthy from the airframe's perspective;
//! a temporarily noisy GPS that the ESKF is still confidently fusing is.
//! The truthful answer to "is GPS ok?" is encoded in the ESKF status, not
//! in raw `fix_type` / `num_sv`. So this module reports both sides — the
//! ESKF/Mahony status (the airframe's truth) and, on `est_pos_gps` builds,
//! the latest GPS fix as supporting detail. A focused `gpshealth` would be
//! redundant with the `health` view; the existing `gps` shell command stays
//! as the raw-data dump for engineers who want lat/lon/h_acc directly.

use core::fmt;

use embassy_time::Instant;

use crate::sensors::rc::{
    BlockReason, LATEST_BLOCK_REASON, LINK_STATS_MAX_AGE_MS, MAX_ESKF_MAHONY_DISAGREE_DEG,
    MAX_TILT_DEG, MIN_LINK_QUALITY, THROTTLE_MINCHECK,
};

/// Live link snapshot supplied by the caller. We don't read this from a
/// global because the canonical source is the ArmStateMachine running on
/// the RC executor; the shell peeks the `RC_LINK_STATUS` PubSub instead.
pub struct LinkSnapshot {
    pub active: bool,
    pub age_ms: u64,
    pub quality: u8,
}

/// Aggregated, point-in-time view of every input the arming gates check,
/// plus a couple of useful "context" fields (armed state, last block
/// reason). All fields are best-effort — anything that's gated behind a
/// cargo feature degrades to a sensible default when the feature is off.
pub struct SystemHealth {
    pub failsafe_active: bool,
    pub armed: bool,
    pub link: LinkSnapshot,
    pub throttle: Option<u16>,

    pub estimator_ready: bool,
    pub estimator_degraded: bool,
    pub estimator_faults: u32,
    pub estimator_severe: bool,
    pub estimator_phase: EstimatorPhaseSummary,

    pub mahony_ready: bool,
    pub mahony_attitude_deg: Option<(f32, f32, f32)>,

    pub attitude_health_bits: u8,
    pub attitude_health_required: u8,

    /// Wall-clock timestamps of the last accepted measurement update
    /// per channel — read straight from the wrappers' atomics. Used
    /// by the report formatter to distinguish "no fresh pose
    /// arriving" from "pose arriving but rejected" — both manifest
    /// as `POS_STALE` in the fault bitfield, but only one of them
    /// refreshes these clocks.
    pub last_pos_update: Option<Instant>,
    pub last_vel_update: Option<Instant>,
    pub last_att_update: Option<Instant>,

    pub last_block_reason: Option<(BlockReason, Instant)>,
}

#[derive(Clone, Copy)]
pub enum EstimatorPhaseSummary {
    AwaitingExteroceptive,
    Converging {
        roll_deg: f32,
        pitch_deg: f32,
        yaw_deg: f32,
    },
    Running {
        roll_deg: f32,
        pitch_deg: f32,
        yaw_deg: f32,
    },
    /// Reported when the firmware was built without an ESKF feature; no
    /// estimator is running, so position-class checks are vacuously fine.
    NotConfigured,
}

impl SystemHealth {
    /// Sample every gate input. The caller supplies the live link state
    /// and latest throttle PWM (peeked from `RC_LINK_STATUS` / `RC_INPUT`)
    /// because those don't have global mirrors.
    pub fn snapshot(link: LinkSnapshot, throttle: Option<u16>) -> Self {
        use core::sync::atomic::Ordering;

        let failsafe_active =
            crate::control::failsafe::FAILSAFE_ACTIVE.load(Ordering::Acquire);
        let armed = crate::motors::IS_ARMED.load(Ordering::Acquire);

        #[cfg(any(feature = "est_pos_mocap", feature = "est_pos_gps"))]
        let (
            estimator_ready,
            estimator_degraded,
            estimator_faults,
            estimator_severe,
            estimator_phase,
            mahony_ready,
            mahony_attitude_deg,
            attitude_health_bits,
            attitude_health_required,
        ) = {
            use crate::estimation::{att_health, EstimatorPhase};
            let phase = crate::estimation::ESTIMATOR_STATUS.lock(|c| c.get());
            let phase_summary = match phase {
                EstimatorPhase::AwaitingExteroceptive => {
                    EstimatorPhaseSummary::AwaitingExteroceptive
                }
                EstimatorPhase::Converging {
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                    ..
                } => EstimatorPhaseSummary::Converging {
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                },
                EstimatorPhase::Running {
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                    ..
                } => EstimatorPhaseSummary::Running {
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                },
            };
            // Mahony AHRS is not present on this branch — readers see a
            // "ready" sentinel and an empty quaternion so the cross-check
            // gates that depend on it never block.
            let mahony_attitude_deg: Option<(f32, f32, f32)> = None;
            (
                crate::estimation::ESTIMATOR_READY.load(Ordering::Acquire),
                crate::estimation::ESKF_DEGRADED.load(Ordering::Acquire),
                crate::estimation::ESKF_FAULTS.load(Ordering::Acquire),
                crate::estimation::ESKF_SEVERE_FAULT.load(Ordering::Acquire),
                phase_summary,
                true,
                mahony_attitude_deg,
                crate::estimation::ATTITUDE_HEALTH.load(Ordering::Acquire),
                att_health::ALL_OK,
            )
        };

        #[cfg(not(any(feature = "est_pos_mocap", feature = "est_pos_gps")))]
        let (
            estimator_ready,
            estimator_degraded,
            estimator_faults,
            estimator_severe,
            estimator_phase,
            mahony_ready,
            mahony_attitude_deg,
            attitude_health_bits,
            attitude_health_required,
        ) = (
            true,
            false,
            0u32,
            false,
            EstimatorPhaseSummary::NotConfigured,
            true,
            None,
            0u8,
            0u8,
        );

        Self {
            failsafe_active,
            armed,
            link,
            throttle,
            estimator_ready,
            estimator_degraded,
            estimator_faults,
            estimator_severe,
            estimator_phase,
            mahony_ready,
            mahony_attitude_deg,
            attitude_health_bits,
            attitude_health_required,
            #[cfg(any(feature = "est_pos_mocap", feature = "est_pos_gps"))]
            last_pos_update: crate::estimation::ESKF_LAST_POS_UPDATE.lock(|c| c.get()),
            #[cfg(any(feature = "est_pos_mocap", feature = "est_pos_gps"))]
            last_vel_update: crate::estimation::ESKF_LAST_VEL_UPDATE.lock(|c| c.get()),
            #[cfg(any(feature = "est_pos_mocap", feature = "est_pos_gps"))]
            last_att_update: crate::estimation::ESKF_LAST_ATT_UPDATE.lock(|c| c.get()),
            #[cfg(not(any(feature = "est_pos_mocap", feature = "est_pos_gps")))]
            last_pos_update: None,
            #[cfg(not(any(feature = "est_pos_mocap", feature = "est_pos_gps")))]
            last_vel_update: None,
            #[cfg(not(any(feature = "est_pos_mocap", feature = "est_pos_gps")))]
            last_att_update: None,
            last_block_reason: LATEST_BLOCK_REASON.lock(|c| c.get()),
        }
    }

    /// First gate that would currently reject arming, in the same order
    /// `ArmStateMachine` evaluates them. Returns `None` if all gates pass
    /// (the bird is ready to arm — modulo the debounce hold).
    pub fn first_blocker(&self) -> Option<BlockReason> {
        if self.failsafe_active {
            return Some(BlockReason::Failsafe);
        }
        if let Some(thr) = self.throttle {
            if thr > THROTTLE_MINCHECK {
                return Some(BlockReason::ThrottleNotMin {
                    value: thr,
                    max: THROTTLE_MINCHECK,
                });
            }
        }
        if !self.link.active {
            return Some(BlockReason::LinkInactive);
        }
        if self.link.age_ms > LINK_STATS_MAX_AGE_MS {
            return Some(BlockReason::LinkStale {
                age_ms: self.link.age_ms,
            });
        }
        if self.link.quality < MIN_LINK_QUALITY {
            return Some(BlockReason::LinkLowQuality {
                quality: self.link.quality,
                min: MIN_LINK_QUALITY,
            });
        }
        if !self.estimator_ready {
            return Some(BlockReason::EstimatorNotReady);
        }
        let attitude_rp = match self.estimator_phase {
            EstimatorPhaseSummary::Running {
                roll_deg,
                pitch_deg,
                ..
            }
            | EstimatorPhaseSummary::Converging {
                roll_deg,
                pitch_deg,
                ..
            } => Some((roll_deg, pitch_deg)),
            _ => None,
        };
        if let Some((r, p)) = attitude_rp {
            if r.abs() > MAX_TILT_DEG || p.abs() > MAX_TILT_DEG {
                return Some(BlockReason::TiltOutOfEnvelope {
                    roll_deg: r,
                    pitch_deg: p,
                });
            }
        }
        if self.attitude_health_required != 0
            && self.attitude_health_bits & self.attitude_health_required
                != self.attitude_health_required
        {
            return Some(BlockReason::SensorHealthDegraded {
                bits: self.attitude_health_bits,
                required: self.attitude_health_required,
            });
        }
        if self.estimator_degraded {
            return Some(BlockReason::EskfDegraded {
                faults: self.estimator_faults,
            });
        }
        if !self.mahony_ready {
            return Some(BlockReason::MahonyNotReady);
        }
        if let (Some((er, ep)), Some((mr, mp, _))) = (attitude_rp, self.mahony_attitude_deg) {
            if (er - mr).abs() > MAX_ESKF_MAHONY_DISAGREE_DEG
                || (ep - mp).abs() > MAX_ESKF_MAHONY_DISAGREE_DEG
            {
                return Some(BlockReason::EskfMahonyTiltDisagreement {
                    eskf_roll_deg: er,
                    eskf_pitch_deg: ep,
                    mahony_roll_deg: mr,
                    mahony_pitch_deg: mp,
                });
            }
        }
        None
    }

    /// Render a multi-line, human-scannable report.
    pub fn write_report(&self, w: &mut impl fmt::Write) -> fmt::Result {
        let tl_dr = match self.first_blocker() {
            None if self.armed => "ARMED — flying.",
            None => "READY — all gates pass; flip the arm switch.",
            Some(r) => {
                writeln!(w, "BLOCKED — first failing gate: {:?}", DebugReason(r))?;
                ""
            }
        };
        if !tl_dr.is_empty() {
            writeln!(w, "{}", tl_dr)?;
        }
        writeln!(
            w,
            "  failsafe_active : {}",
            yes_no(self.failsafe_active)
        )?;
        writeln!(w, "  armed           : {}", yes_no(self.armed))?;
        writeln!(
            w,
            "  rc link         : active={} age_ms={} lq={}/{}",
            yes_no(self.link.active),
            self.link.age_ms,
            self.link.quality,
            MIN_LINK_QUALITY,
        )?;
        match self.throttle {
            Some(v) => writeln!(
                w,
                "  throttle        : {}us (max for arm: {}us)",
                v, THROTTLE_MINCHECK
            )?,
            None => writeln!(w, "  throttle        : (no recent RC frame)")?,
        }
        match self.estimator_phase {
            EstimatorPhaseSummary::NotConfigured => {
                writeln!(w, "  estimator       : not configured")?;
            }
            EstimatorPhaseSummary::AwaitingExteroceptive => {
                writeln!(w, "  estimator       : AwaitingExteroceptive")?;
            }
            EstimatorPhaseSummary::Converging {
                roll_deg,
                pitch_deg,
                yaw_deg,
            } => {
                writeln!(
                    w,
                    "  estimator       : Converging (rpy = {:.1}/{:.1}/{:.1} deg)",
                    roll_deg, pitch_deg, yaw_deg
                )?;
            }
            EstimatorPhaseSummary::Running {
                roll_deg,
                pitch_deg,
                yaw_deg,
            } => {
                writeln!(
                    w,
                    "  estimator       : Running (rpy = {:.1}/{:.1}/{:.1} deg)",
                    roll_deg, pitch_deg, yaw_deg
                )?;
            }
        }
        writeln!(
            w,
            "  estimator flags : ready={} degraded={} severe={} faults=0x{:04x}",
            yes_no(self.estimator_ready),
            yes_no(self.estimator_degraded),
            yes_no(self.estimator_severe),
            self.estimator_faults,
        )?;
        // Data-freshness line: how long ago each measurement channel
        // was last accepted. Disambiguates "no fresh pose arriving"
        // from "pose arriving but rejected" — both manifest as
        // POS_STALE in the fault bitfield, but only the former
        // freezes these clocks. If you see POS_STALE asserted but
        // pos_age_ms ticking down, fresh poses *are* arriving and
        // recovery is in progress; if it's frozen at >2000ms,
        // upstream (Vicon / GPS receiver) hasn't resumed.
        write!(w, "  data freshness  :")?;
        let now = embassy_time::Instant::now();
        let age_ms = |t: Option<Instant>| -> Option<u64> {
            t.map(|t| now.saturating_duration_since(t).as_millis())
        };
        match age_ms(self.last_pos_update) {
            Some(ms) => write!(w, " pos={}ms", ms)?,
            None => write!(w, " pos=never")?,
        }
        match age_ms(self.last_vel_update) {
            Some(ms) => write!(w, " vel={}ms", ms)?,
            None => write!(w, " vel=never")?,
        }
        match age_ms(self.last_att_update) {
            Some(ms) => write!(w, " att={}ms", ms)?,
            None => write!(w, " att=never")?,
        }
        writeln!(
            w,
            " (stale at >{}ms)",
            (crate::estimation::POS_TIMEOUT_S * 1000.0) as u32,
        )?;
        // Decoded breakdown of the fault bits — only emitted when at
        // least one bit is set, so the healthy report stays clean.
        // Operators reading the live shell don't have the bit layout
        // memorised; the hex value above is the canonical encoding,
        // this line is the reading aid.
        if self.estimator_faults != 0 {
            write!(w, "  fault breakdown :")?;
            let mut first = true;
            let mut emit = |w: &mut dyn fmt::Write, name: &str| -> fmt::Result {
                if first {
                    write!(w, " {}", name)?;
                    first = false;
                } else {
                    write!(w, " | {}", name)?;
                }
                Ok(())
            };
            use crate::estimation::fault;
            if self.estimator_faults & fault::NAN_RESET != 0 {
                emit(w, "NAN_RESET")?;
            }
            if self.estimator_faults & fault::POS_STALE != 0 {
                emit(w, "POS_STALE")?;
            }
            if self.estimator_faults & fault::VEL_STALE != 0 {
                emit(w, "VEL_STALE")?;
            }
            if self.estimator_faults & fault::ATT_STALE != 0 {
                emit(w, "ATT_STALE")?;
            }
            if self.estimator_faults & fault::COV_TRACE_BLOWUP != 0 {
                emit(w, "COV_TRACE_BLOWUP")?;
            }
            if self.estimator_faults & fault::GUARD_JUMP_CASCADE != 0 {
                emit(w, "GUARD_JUMP_CASCADE")?;
            }
            if self.estimator_faults & fault::GUARD_REJECT_CASCADE != 0 {
                emit(w, "GUARD_REJECT_CASCADE")?;
            }
            // Surface unknown high bits so future additions don't
            // silently disappear from the breakdown when the firmware
            // pulls in a newer estimation::fault module.
            const KNOWN: u32 = fault::NAN_RESET
                | fault::POS_STALE
                | fault::VEL_STALE
                | fault::ATT_STALE
                | fault::COV_TRACE_BLOWUP
                | fault::GUARD_JUMP_CASCADE
                | fault::GUARD_REJECT_CASCADE;
            let unknown = self.estimator_faults & !KNOWN;
            if unknown != 0 {
                if first {
                    write!(w, " unknown=0x{:x}", unknown)?;
                } else {
                    write!(w, " | unknown=0x{:x}", unknown)?;
                }
            }
            writeln!(w)?;
        }
        match self.mahony_attitude_deg {
            Some((r, p, y)) => writeln!(
                w,
                "  mahony          : ready={} (rpy = {:.1}/{:.1}/{:.1} deg)",
                yes_no(self.mahony_ready),
                r,
                p,
                y
            )?,
            None => writeln!(
                w,
                "  mahony          : ready={} (no attitude sample)",
                yes_no(self.mahony_ready)
            )?,
        }
        writeln!(
            w,
            "  sensors         : bits=0b{:03b} required=0b{:03b}",
            self.attitude_health_bits, self.attitude_health_required
        )?;
        match self.last_block_reason {
            Some((reason, t)) => writeln!(
                w,
                "  last reject     : {:?} (latched at t={}ms)",
                DebugReason(reason),
                t.as_millis()
            )?,
            None => writeln!(w, "  last reject     : (none since last arm/disarm)")?,
        }
        Ok(())
    }
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// `BlockReason` derives `defmt::Format` but not `Debug`. Wrap it so we can
/// pretty-print to `core::fmt::Write` without adding `Debug` to the enum.
struct DebugReason(BlockReason);

impl fmt::Debug for DebugReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            BlockReason::Failsafe => write!(f, "Failsafe"),
            BlockReason::ThrottleNotMin { value, max } => {
                write!(f, "ThrottleNotMin({}us > {}us)", value, max)
            }
            BlockReason::LinkInactive => write!(f, "LinkInactive"),
            BlockReason::LinkStale { age_ms } => write!(f, "LinkStale({}ms)", age_ms),
            BlockReason::LinkLowQuality { quality, min } => {
                write!(f, "LinkLowQuality({}/{}%)", quality, min)
            }
            BlockReason::EstimatorNotReady => write!(f, "EstimatorNotReady"),
            BlockReason::TiltOutOfEnvelope {
                roll_deg,
                pitch_deg,
            } => write!(
                f,
                "TiltOutOfEnvelope(roll={:.1} pitch={:.1} deg)",
                roll_deg, pitch_deg
            ),
            BlockReason::SensorHealthDegraded { bits, required } => write!(
                f,
                "SensorHealthDegraded(bits=0b{:03b} need=0b{:03b})",
                bits, required
            ),
            BlockReason::EskfDegraded { faults } => {
                write!(f, "EskfDegraded(faults=0x{:04x})", faults)
            }
            BlockReason::MahonyNotReady => write!(f, "MahonyNotReady"),
            BlockReason::EskfMahonyTiltDisagreement {
                eskf_roll_deg,
                eskf_pitch_deg,
                mahony_roll_deg,
                mahony_pitch_deg,
            } => write!(
                f,
                "EskfMahonyTiltDisagreement(ESKF=[{:.1},{:.1}] Mahony=[{:.1},{:.1}] deg)",
                eskf_roll_deg, eskf_pitch_deg, mahony_roll_deg, mahony_pitch_deg
            ),
        }
    }
}
