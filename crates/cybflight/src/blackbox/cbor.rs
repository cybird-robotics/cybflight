//! CBOR encoder for blackbox payloads — re-export shim.
//!
//! The implementation moved to [`cybflight_core::cbor`] so it (and the
//! compact wire formats in [`cybflight_core::blackbox_wire`]) run
//! their unit tests on the host via `just test`. This firmware crate
//! only compiles for thumbv7em, so a `#[cfg(test)]` module here would
//! never execute — which is how the framer's tests in `mcap.rs` sat
//! dead for months.
//!
//! Kept as a module (rather than fixing 15 import sites) so topic
//! encoders keep writing `use crate::blackbox::cbor::{self, CborWriter}`.

pub use cybflight_core::cbor::{CborWriter, OutOfSpace, Result};
