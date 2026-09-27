//! Offline-prebaked mission trajectories (multi-profile).
//!
//! Each [`MissionProfile`] holds a self-contained schedule that the
//! `mission_planner` task can feed directly into a [`MincoSnap`] solver,
//! bypassing the on-device BFGS optimizer. Profiles are solved offline by
//! a heavier planner; each one lives in its own **`missions/*.yaml`**
//! file (workspace root, planner-native `start`/`waypoints`/`durations`
//! or absolute `timestamps` form) and is baked at build time by
//! `build.rs` into the generated `PROFILES` table `include!`d below.
//! Adding a mission = dropping the planner's output file in `missions/`
//! and rebuilding — the bake performs the conversions that used to be
//! done by hand (durations → timestamps, duplicate-head drop) and
//! validates the data, so a malformed mission fails the build, not the
//! flight.
//!
//! `timestamps[i]` is the absolute time of arrival at `waypoints[i]`
//! measured from the trajectory start. Segment durations are recovered
//! by consecutive differences, with the first segment running from
//! `t=0` to `timestamps[0]`.
//!
//! ## MINCO contract (n = `profile.num_pieces()`)
//!
//! For an `n`-piece trajectory, `MincoSnap` consumes `n−1` intermediate
//! waypoints plus a tail. The schedule provides `n` waypoints —
//! `wp[0..n−1]` are the intermediates and `wp[n−1]` is the tail. None of
//! the waypoint slots holds the head pose; that comes from the live
//! `ACTIVE_POSITION_SETPOINT` (or [`MissionProfile::start_pos`] when
//! `OFFLINE_USE_YAML_START` is set).
//!
//! ## Profile selection
//!
//! Selection is runtime, addressed by `(env, variant, speed)`. The active
//! index lives in [`ACTIVE_PROFILE`] (an `AtomicUsize`); the shell
//! `mission set <env> <variant> <speed>` writes it and the value is
//! mirrored into `VehicleParams.mission_profile` for flash persistence.
//! At boot, [`init_active_from_index`] re-applies the persisted choice.
//!
//! ⚠ The persisted value is an **index into the name-sorted baked
//! table** — adding or removing a mission file reorders it. Re-run
//! `mission set` (and `param save`) after changing the mission set; the
//! boot log and `mission get` always show the active mission *name*.
//!
//! [`MincoSnap`]: cybflight_core::trajectory_planning::minco_snap::MincoSnap

#![cfg(feature = "outer_mpc")]

use core::sync::atomic::{AtomicUsize, Ordering};

/// Which differential-flatness map converts this mission's flat outputs
/// (acceleration/jerk + desired yaw) into the attitude and body-rate
/// references the outer loop tracks. Baked from the mission YAML's
/// `flatness_map` key; `TiltYaw` is the default.
///
/// The two conventions interpret the yaw input ψ differently and trade
/// singularity locations:
///
/// - `TiltYaw`: ψ is the intrinsic tilt-then-yaw angle
///   (`flatness_to_state_tilt_yaw` family; attitude = tilt(z_B) ∘
///   yaw(ψ)). Numerically robust through 90° tilt; singular only at
///   full inversion (`z_B = −ẑ`, handled by a substituted flip). The
///   flown compass heading deviates from ψ as tilt grows.
/// - `TrueYaw`: ψ is the compass heading of the body-x projection
///   (`flatness_to_state_true_yaw`, the Faessler/RPG nominal reference
///   inputs; x_B ∝ y_C × z_B). Heading tracks ψ exactly regardless of
///   tilt, but the construction is singular when the thrust axis goes
///   horizontal along the heading frame's y-axis (90° roll).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlatnessMap {
    TiltYaw,
    TrueYaw,
}

/// One offline trajectory schedule.
#[derive(Clone, Copy)]
pub struct MissionProfile {
    /// Mission name (= `missions/<name>.yaml` file stem), shown by
    /// `mission list` / `mission get`.
    pub name: &'static str,
    /// Selection axis: environment.
    pub env: &'static str,
    /// Selection axis: trajectory family (e.g. `"splits"`, `"drag"`).
    pub variant: &'static str,
    /// Selection axis: speed tier.
    pub speed: &'static str,
    /// Recorded mission start position [m]. Used as the head pose only when
    /// `OFFLINE_USE_YAML_START` is `true` in `mission_planner.rs`; otherwise
    /// the live `ACTIVE_POSITION_SETPOINT` is the head.
    pub start_pos: [f32; 3],
    /// Intermediate + tail waypoints (length == `timestamps.len()`).
    pub waypoints: &'static [[f32; 3]],
    /// Absolute time of arrival at each waypoint [s] (length == `waypoints.len()`).
    pub timestamps: &'static [f32],
    /// Desired yaw at each waypoint [rad] (length == `waypoints.len()`),
    /// or `None` to hold the yaw setpoint latched at mission entry.
    /// Interpreted under this mission's [`FlatnessMap`] convention
    /// (compass heading for `TrueYaw`; intrinsic tilt-then-yaw angle
    /// for `TiltYaw`, which deviates from the compass heading of
    /// body-x as tilt grows).
    pub headings: Option<&'static [f32]>,
    /// Look-forward yaw mode: yaw points at the reference position
    /// `yaw_lookahead_dt_s` ahead. The bake guarantees `headings` is
    /// `None` when set (lookahead wins at load).
    pub lookahead: bool,
    /// Lookahead horizon [s] (bake-validated to
    /// `vehicle_yaml::mission::YAW_LOOKAHEAD_DT_RANGE_S`).
    pub yaw_lookahead_dt_s: f32,
    /// Slew-rate cap [rad/s] on the lookahead yaw reference (bake-
    /// validated finite and > 0; the outer loop further caps it at the
    /// model's yaw-rate bound).
    pub yaw_lookahead_max_rate_rad_s: f32,
    /// Flatness-map convention for this mission's references.
    pub flatness_map: FlatnessMap,
}

impl MissionProfile {
    pub const fn num_pieces(&self) -> usize {
        self.waypoints.len()
    }

    /// Whether this profile's environment matches the firmware build.
    ///
    /// `est_pos_mocap` builds (indoor lab / HIL) only accept `env="indoor"`
    /// profiles; `est_pos_gps` builds (outdoor) only accept `env="outdoor"`.
    /// The `outer_mpc + est_pos_*` mutual-exclusion is enforced at compile
    /// time in `control/mod.rs`.
    pub fn is_compatible_with_build(&self) -> bool {
        self.env.eq_ignore_ascii_case(BUILD_ENV)
    }
}

// ─── Build-time environment ──────────────────────────────────────────
//
// Selected by the position-source feature:
// - `est_pos_mocap` → "indoor" (VICON / lab)
// - `est_pos_gps`   → "outdoor" (u-blox F9 NAV-PVT or UM982 BESTNAV)
//
// The compile_error guards in `control/mod.rs` ensure exactly one is set
// whenever this module is compiled (`outer_mpc` requires it).

/// Indoor build (mocap / VICON position source).
#[cfg(feature = "est_pos_mocap")]
pub const BUILD_ENV: &str = "indoor";

/// Outdoor build (GPS position source).
#[cfg(feature = "est_pos_gps")]
pub const BUILD_ENV: &str = "outdoor";

// ─── Baked mission data ──────────────────────────────────────────────
//
// Generated from `missions/*.yaml` by `build.rs::bake_missions`:
// per-mission statics, `PROFILES` (name-sorted), `OFFLINE_MAX_PIECES`,
// and the per-env `DEFAULT_PROFILE_INDEX_{INDOOR,OUTDOOR}` consts.

include!(concat!(env!("OUT_DIR"), "/missions.rs"));

// Compile-time backstop over the generated data (the bake validates the
// same invariants with better messages; this guards against a future
// hand-edited include). If `OFFLINE_MAX_PIECES` needs to grow past
// `MAX_PIECES` in `cybflight_core/src/trajectory_planning/mod.rs`, grow
// that first — the bake enforces the bound.
const _: () = {
    let mut i = 0;
    while i < PROFILES.len() {
        let p = PROFILES[i];
        assert!(
            p.waypoints.len() == p.timestamps.len(),
            "MissionProfile waypoint/timestamp length mismatch",
        );
        assert!(
            p.waypoints.len() >= 1,
            "MissionProfile must have at least one waypoint",
        );
        assert!(
            p.waypoints.len() <= OFFLINE_MAX_PIECES,
            "MissionProfile exceeds OFFLINE_MAX_PIECES",
        );
        if let Some(h) = p.headings {
            assert!(
                h.len() == p.waypoints.len(),
                "MissionProfile headings/waypoints length mismatch",
            );
        }
        assert!(
            p.yaw_lookahead_dt_s > 0.0,
            "MissionProfile yaw_lookahead_dt_s must be > 0",
        );
        assert!(
            p.yaw_lookahead_max_rate_rad_s > 0.0,
            "MissionProfile yaw_lookahead_max_rate_rad_s must be > 0",
        );
        i += 1;
    }
};

// Default index must point at a real entry in PROFILES.
const _: () = assert!((DEFAULT_PROFILE_INDEX as usize) < PROFILES.len());

/// Per-env fallback default (generated: `<env>_splits_slow` when present,
/// else the first mission of the env).
#[cfg(feature = "est_pos_mocap")]
const ENV_DEFAULT_PROFILE_INDEX: u8 = DEFAULT_PROFILE_INDEX_INDOOR;
#[cfg(feature = "est_pos_gps")]
const ENV_DEFAULT_PROFILE_INDEX: u8 = DEFAULT_PROFILE_INDEX_OUTDOOR;

/// Default profile applied at boot when flash holds no valid setting, or
/// when a persisted index is incompatible with the current build env.
/// Index into [`PROFILES`]: the vehicle YAML's `default_mission` (name,
/// resolved and env-validated at bake) when declared, else the per-env
/// fallback above.
pub const DEFAULT_PROFILE_INDEX: u8 = match crate::vehicle::BAKED_DEFAULT_MISSION_INDEX {
    Some(i) => i,
    None => ENV_DEFAULT_PROFILE_INDEX,
};

/// Active profile index. Read by `mission_planner::plan_offline()`,
/// written by `init_active_from_index()` (boot) and the shell.
static ACTIVE_PROFILE: AtomicUsize = AtomicUsize::new(DEFAULT_PROFILE_INDEX as usize);

/// Returns the currently active profile.
pub fn active() -> &'static MissionProfile {
    let raw = ACTIVE_PROFILE.load(Ordering::Acquire);
    let idx = if raw < PROFILES.len() {
        raw
    } else {
        DEFAULT_PROFILE_INDEX as usize
    };
    PROFILES[idx]
}

/// Returns the active profile's index in [`PROFILES`].
pub fn active_index() -> u8 {
    let raw = ACTIVE_PROFILE.load(Ordering::Acquire);
    if raw < PROFILES.len() {
        raw as u8
    } else {
        DEFAULT_PROFILE_INDEX
    }
}

/// Sets the active profile by index. Returns `false` if `idx` is
/// out of range or the profile's `env` does not match [`BUILD_ENV`].
/// Selection is left unchanged on rejection.
pub fn set_active(idx: u8) -> bool {
    let i = idx as usize;
    if i >= PROFILES.len() {
        return false;
    }
    if !PROFILES[i].is_compatible_with_build() {
        return false;
    }
    ACTIVE_PROFILE.store(i, Ordering::Release);
    true
}

/// Re-applies a persisted profile index at boot. Falls back to
/// [`DEFAULT_PROFILE_INDEX`] (which is itself env-gated) when the
/// persisted index is out of range OR points at a profile whose `env`
/// does not match [`BUILD_ENV`] — e.g., flash from a prior firmware
/// build that flew the same airframe in the other environment.
pub fn init_active_from_index(idx: u8) {
    let i = idx as usize;
    let resolved = if i < PROFILES.len() && PROFILES[i].is_compatible_with_build() {
        i
    } else {
        DEFAULT_PROFILE_INDEX as usize
    };
    ACTIVE_PROFILE.store(resolved, Ordering::Release);
}

/// Resolves `(env, variant, speed)` to a profile index. Case-insensitive.
/// Pure registry lookup — does not enforce build-env compatibility; the
/// shell layer (and `set_active`) gate that separately so the user gets a
/// distinct "incompatible env" error vs. "unknown mission".
pub fn find(env: &str, variant: &str, speed: &str) -> Option<u8> {
    PROFILES.iter().enumerate().find_map(|(i, p)| {
        (p.env.eq_ignore_ascii_case(env)
            && p.variant.eq_ignore_ascii_case(variant)
            && p.speed.eq_ignore_ascii_case(speed))
        .then_some(i as u8)
    })
}

/// Resolves a mission name (= YAML file stem) to a profile index.
/// Case-insensitive. Same gating split as [`find`].
pub fn find_by_name(name: &str) -> Option<u8> {
    PROFILES
        .iter()
        .enumerate()
        .find_map(|(i, p)| p.name.eq_ignore_ascii_case(name).then_some(i as u8))
}
