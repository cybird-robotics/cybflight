//! Baked vehicle configuration.
//!
//! There is no "default drone" anymore: the physical identity (mass,
//! inertia, motor geometry/thrust) comes from `vehicles/<VEHICLE>.yaml`,
//! selected via the `VEHICLE` env var (see `.env` / Justfile) and
//! validated at **build time** by `build.rs` — a missing identity field is
//! a compile error, and a typo'd tuning key is a compile error.
//!
//! `default_params()` reconstructs the baked [`FirmwareConfig`] from the
//! generated full-snapshot table. Runtime `param set` + `param save`
//! overrides (the KV store) layer on top per-key at boot.

use cybflight_core::params::FirmwareConfig;

include!(concat!(env!("OUT_DIR"), "/baked_params.rs"));

// Schema-drift guard: the table is generated against the same
// cybflight-core the firmware links, so the counts must agree.
const _: () = assert!(BAKED_PARAMS.len() == cybflight_core::params::PARAM_COUNT);

/// The baked configuration for this build's vehicle
/// ([`BAKED_VEHICLE`]).
///
/// Serves as both the boot-time base (KV overrides are replayed on top)
/// and the "baked defaults" reference the KV store diffs and prunes
/// against.
pub fn default_params() -> FirmwareConfig {
    let mut cfg = FirmwareConfig::scaffold();
    for (name, value) in BAKED_PARAMS {
        // Hard assert (release too): a baked name the registry doesn't
        // know would silently leave the scaffold's ZEROED airframe in
        // place — a zero mass flowing into 1/mass paths is strictly
        // worse than a loud disarmed boot panic. Unreachable in practice
        // (table and registry come from the same schema), which is
        // exactly why it must not be a debug_assert.
        assert!(cfg.set_named(name, value), "baked param not in registry");
    }
    // The offline-mission default is a firmware/feature decision
    // (BUILD_ENV-dependent), not a vehicle property — it overrides
    // whatever the scaffold/YAML carried. A KV override (from `mission
    // set` + save) still wins at boot.
    #[cfg(feature = "outer_mpc")]
    {
        cfg.trajectory.mission_profile = crate::control::offline_mission::DEFAULT_PROFILE_INDEX;
    }
    cfg
}
