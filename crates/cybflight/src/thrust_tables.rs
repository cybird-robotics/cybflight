//! Baked-at-build-time thrust tables.
//!
//! `build.rs` parses every CSV under `data/thrust_tables/`, validates the
//! contents (size, monotonicity, finite values), and emits one
//! `pub static <NAME>: ThrustTable<TABLE_N>` per file. The MCU never parses
//! CSV at runtime — malformed data fails the build, not the flight.
//!
//! To use one in INDI, set `THRUST_MODEL` in `control/indi_task.rs` to
//! `ThrustModel::Table(&crate::thrust_tables::A2RL_0114)`.

include!(concat!(env!("OUT_DIR"), "/thrust_tables.rs"));
