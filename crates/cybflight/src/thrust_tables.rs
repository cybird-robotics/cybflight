//! Baked-at-build-time thrust tables.
//!
//! `build.rs` parses every CSV under `data/thrust_tables/`, validates the
//! contents (size, monotonicity, finite values), and emits one
//! `pub static <NAME>: ThrustTable<TABLE_N>` per file. The MCU never parses
//! CSV at runtime — malformed data fails the build, not the flight.
//!
//! To use one in INDI, declare it in the vehicle YAML:
//! `airframe.thrust_model: { type: table, table: a2rl_0114 }` — the bake
//! resolves the stem to the generated static and emits
//! `crate::vehicle::BAKED_THRUST_MODEL`.

include!(concat!(env!("OUT_DIR"), "/thrust_tables.rs"));
