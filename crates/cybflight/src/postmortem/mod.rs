//! Post-mortem logging subsystem — captures "why did the drone die"
//! data that survives reboot and full power loss.
//!
//! ## Why this exists
//!
//! The SD-card blackbox cannot be trusted with crash forensics: a
//! half-written 512 B sector at the trailing edge can leave the FAT
//! directory entry pointing at zero bytes. The most valuable seconds
//! of any crash — the ones containing the failsafe trip, the panic,
//! or the brownout that *caused* the crash — are precisely the seconds
//! we can't trust SD to preserve.
//!
//! ## What this captures
//!
//! - Reset cause from `RCC.RSR` + `PWR.CSR1` (read in `pre_init`
//!   before flags are cleared).
//! - Last fatal: panic site, HardFault frame + SCB registers, or PVD
//!   brownout marker.
//! - Last-N flight events (same KIND_* code space as the blackbox
//!   `/events` topic — extended with `KIND_PANIC` etc).
//! - Last attitude / setpoint / RC snapshot, refreshed continuously
//!   so the PVD path doesn't have to gather data during the brownout
//!   window.
//!
//! ## Storage
//!
//! Primary: STM32H7's D3-domain backup SRAM (4 KiB at `0x3880_0000`).
//! Byte-atomic, no erase, survives soft-reset and (with VBAT wired) full
//! power loss. See [`bkpsram`].
//!
//! Fallback (Stage A, single-slot, NOT YET IMPLEMENTED — no
//! `flash_mirror` module exists): a flash-resident copy written by the
//! PVD IRQ + on graceful shutdown so the LiPo-yank crash mode (where
//! 3.3 V collapses with VBAT) still has a copy. Placement must avoid
//! both the app image (bank 1 is app flash up to 1792K, see `memory.x`)
//! and the param KV store (bank-2 sectors 6+7).
//!
//! ## Surfacing
//!
//! Three paths, all driven from the same record:
//!
//! 1. Shell: `postmortem show` reads BKPSRAM and prints a summary.
//! 2. defmt: at boot, the recovery code logs a one-line summary.
//! 3. SD blackbox: when the next flight session opens, the recorder
//!    emits `KIND_BOOT_POSTMORTEM` events bracketing the prior boot's
//!    event ring as `/events` records — so post-flight tools see the
//!    fault inline with the new flight's events.
//!
//! ## Hard rules for fault paths
//!
//! `fault.rs` (panic, HardFault, PVD) does **no** allocation, **no**
//! await, and **no** Mutex. Callers from those paths use only
//! [`bkpsram::with_record_mut`] + [`record::finalize`] — pure raw-
//! pointer writes. State this rule at the top of any new file added
//! under this module.

pub mod bkpsram;
pub mod fault;
pub mod record;
pub mod recovery;
// Feature-independent since the reboot-loop debugging of 2026-08-14: the
// capture + `resetcause` shell verb must work on builds WITHOUT
// `postmortem` (the bench mocap build compiles it out), so the module
// lives at the crate root. Re-exported here so `postmortem::reset_cause`
// paths (recovery, shell `postmortem show`, blackbox bracket) still work.
pub use crate::reset_cause;
pub mod task;
