//! Append-only key-value parameter store over two flash sectors.
//!
//! Replaces the monolithic blob (see `params.rs` module docs): flash holds
//! *override records only*, so a schema change no longer invalidates the
//! whole image — unknown keys are skipped, missing keys keep their baked
//! defaults, and one subsystem's evolution can't wipe another's
//! calibration.
//!
//! # Record format
//!
//! Everything is written in whole 32-byte words ([`WORD`]) — the STM32H7
//! flash programming unit — so each record is a single program operation.
//!
//! ```text
//! Sector header (word 0 of a sector):
//!   [0..4]   magic  = "PRMS"
//!   [4..8]   generation: u32 (monotonic across compactions)
//!   [8..12]  crc32 over bytes [0..8]
//!   [12..32] 0xFF
//!
//! Record (words 1..):
//!   [0..4]   magic  = "PRMR"
//!   [4..8]   fnv1a-32 hash of the parameter name
//!   [8..12]  value as f32 LE (ParamValue::as_f32 projection)
//!   [12..16] crc32 over bytes [4..12]
//!   [16..32] 0xFF
//! ```
//!
//! Records are appended; on load, later records win ("last-wins" replay
//! onto the baked defaults). A blank or invalid word ends the log — a torn
//! append therefore costs at most the record being written.
//!
//! # No tombstones, and what that costs
//!
//! A record can only say "override this key to X"; there is no record
//! meaning "this key has no override". So [`save`] cannot remove a key
//! from the store — reverting a param to its baked value appends a
//! record holding that value, which is indistinguishable from having
//! deliberately tuned it there. That is correct until the baked defaults
//! move underneath it: re-flash with an edited vehicle YAML and the
//! stale record shadows the new default, so the YAML edit appears not to
//! take effect. Records also carry no vehicle identity, so flashing a
//! different vehicle onto the same board replays the previous
//! airframe's overrides onto the new one's defaults, matched by name
//! hash alone.
//!
//! [`save_pruned`] is the way out: it rewrites the store as *exactly*
//! the set differing from the baked defaults, so `reset` + prune really
//! does un-override a key. It pays an erase, which is why it is not the
//! default.
//!
//! # Sectors, generations, compaction
//!
//! Two erase units (H7: sectors 6+7, 128 KB each). Exactly one is
//! *active* — the valid header with the highest generation. When an
//! append doesn't fit, [`save`] compacts: the other sector is erased, a
//! new header (generation + 1) plus the *pruned* override set (only
//! params differing from the baked defaults) is written, and the old
//! sector is erased last (and its header is written only after every
//! record, so a half-written compaction target never wins the
//! generation arbitration). Power loss between the header write and the
//! old-sector erase leaves two valid headers; the higher generation
//! wins at the next boot.
//!
//! Appends never erase — `param save` is cheap (one word per changed
//! param). Only compaction (rare), first-time formatting and an
//! explicit [`save_pruned`] pay the 1–2 s blocking erase; the flash
//! implementation is responsible for extending the watchdog there.
//! `save_pruned` reuses this same rewrite path, just entered on demand
//! instead of when the sector fills, so it inherits its crash safety.
//!
//! # Known hazard (unchanged from the blob era)
//!
//! Reads are memory-mapped. On the H7, a word whose programming was
//! interrupted by power loss can carry an uncorrectable-ECC state that
//! faults on read. The blob design had the same exposure across its whole
//! erase+write window; here the window is a single word program. CRCs
//! guard logical corruption only.

use crate::param_registry::{ParamGroup, ParamName};

/// Flash programming word (STM32H7: 256-bit flash word).
pub const WORD: usize = 32;

const HDR_MAGIC: u32 = u32::from_le_bytes(*b"PRMS");
const REC_MAGIC: u32 = u32::from_le_bytes(*b"PRMR");

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoreError {
    /// Flash program/erase operation failed.
    Flash,
    /// The override set doesn't fit even in a freshly compacted sector.
    Full,
    /// Internal registry inconsistency (a param index with no name).
    Corrupt,
    /// The operation was refused because the vehicle is armed.
    Armed,
}

/// Minimal flash abstraction: two erase units addressed as sectors 0 and 1.
///
/// Implementations: the firmware's embassy-flash adapter (sectors 6/7 of
/// bank 2 — the opposite bank from code), and the RAM mock in this
/// module's tests.
pub trait KvFlash {
    /// Bytes per erase unit. Must be a multiple of [`WORD`].
    fn sector_size(&self) -> usize;
    /// Read `buf.len()` bytes from byte `offset` within `sector`.
    fn read(&mut self, sector: u8, offset: usize, buf: &mut [u8]);
    /// Program one erased, word-aligned word.
    fn write_word(&mut self, sector: u8, offset: usize, word: &[u8; WORD])
    -> Result<(), StoreError>;
    /// Erase a whole sector (all bytes → 0xFF). May stall the CPU
    /// on-target; implementations must handle watchdog care.
    fn erase_sector(&mut self, sector: u8) -> Result<(), StoreError>;
}

/// FNV-1a 32-bit over a parameter name. Stable across builds as long as
/// the *name* is stable — which is exactly the schema contract.
pub fn name_hash(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn hash_of_idx<G: ParamGroup>(idx: usize) -> Option<u32> {
    ParamName::of::<G>(idx).map(|n| name_hash(n.as_str()))
}

fn crc(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

fn header_word(generation: u32) -> [u8; WORD] {
    let mut w = [0xFFu8; WORD];
    w[0..4].copy_from_slice(&HDR_MAGIC.to_le_bytes());
    w[4..8].copy_from_slice(&generation.to_le_bytes());
    let c = crc(&w[0..8]);
    w[8..12].copy_from_slice(&c.to_le_bytes());
    w
}

fn record_word(hash: u32, value: f32) -> [u8; WORD] {
    let mut w = [0xFFu8; WORD];
    w[0..4].copy_from_slice(&REC_MAGIC.to_le_bytes());
    w[4..8].copy_from_slice(&hash.to_le_bytes());
    w[8..12].copy_from_slice(&value.to_le_bytes());
    let c = crc(&w[4..12]);
    w[12..16].copy_from_slice(&c.to_le_bytes());
    w
}

fn parse_header(w: &[u8; WORD]) -> Option<u32> {
    let magic = u32::from_le_bytes(w[0..4].try_into().unwrap());
    let generation = u32::from_le_bytes(w[4..8].try_into().unwrap());
    let stored = u32::from_le_bytes(w[8..12].try_into().unwrap());
    (magic == HDR_MAGIC && crc(&w[0..8]) == stored).then_some(generation)
}

fn parse_record(w: &[u8; WORD]) -> Option<(u32, f32)> {
    let magic = u32::from_le_bytes(w[0..4].try_into().unwrap());
    if magic != REC_MAGIC {
        return None;
    }
    let stored = u32::from_le_bytes(w[12..16].try_into().unwrap());
    if crc(&w[4..12]) != stored {
        return None;
    }
    let hash = u32::from_le_bytes(w[4..8].try_into().unwrap());
    let value = f32::from_le_bytes(w[8..12].try_into().unwrap());
    Some((hash, value))
}

/// The active sector, if any: `(sector, generation, append_offset, clean)`.
///
/// `append_offset` is the byte offset of the first word that is not a
/// valid record. `clean` is true when that word is fully erased — i.e.
/// appending there is legal. A torn append (non-blank invalid word) makes
/// the tail dirty; [`save`] then routes through compaction, which rewrites
/// into the freshly erased other sector.
fn active<F: KvFlash>(flash: &mut F) -> Option<(u8, u32, usize, bool)> {
    let mut best: Option<(u8, u32)> = None;
    for sector in 0..2u8 {
        let mut w = [0u8; WORD];
        flash.read(sector, 0, &mut w);
        if let Some(generation) = parse_header(&w)
            && best.map(|(_, g)| generation > g).unwrap_or(true)
        {
            best = Some((sector, generation));
        }
    }
    let (sector, generation) = best?;
    let words = flash.sector_size() / WORD;
    let mut offset = words * WORD; // sector full of valid records
    let mut clean = false;
    for i in 1..words {
        let mut w = [0u8; WORD];
        flash.read(sector, i * WORD, &mut w);
        if parse_record(&w).is_none() {
            offset = i * WORD;
            clean = w.iter().all(|&b| b == 0xFF);
            break;
        }
    }
    Some((sector, generation, offset, clean))
}

/// Upper bound on schema size for the stack-allocated hash lookup
/// table. Far above today's ~170 params; a compile-time-checked cap,
/// not a silent truncation.
pub const MAX_PARAMS: usize = 512;

/// Precomputed `name_hash` for every registry index — turns record
/// lookup from O(records × N × name-walk) into O(records × N) integer
/// compares with the hashes computed exactly once per operation.
struct HashTable {
    hashes: [u32; MAX_PARAMS],
    count: usize,
}

impl HashTable {
    fn build<G: ParamGroup>() -> Self {
        assert!(G::COUNT <= MAX_PARAMS, "param schema exceeds MAX_PARAMS");
        let mut t = HashTable {
            hashes: [0; MAX_PARAMS],
            count: G::COUNT,
        };
        for (idx, slot) in t.hashes[..G::COUNT].iter_mut().enumerate() {
            *slot = hash_of_idx::<G>(idx).unwrap_or(0);
        }
        t
    }

    fn find(&self, hash: u32) -> Option<usize> {
        self.hashes[..self.count].iter().position(|&h| h == hash)
    }
}

/// Replay the active sector's records onto `cfg` (typically the baked
/// defaults). Returns the number of records applied; unknown names are
/// counted in `skipped` (a firmware downgrade/upgrade case, not an
/// error) and records that fail validation — non-finite values or
/// values outside the parameter's `ParamMeta` range — are counted in
/// `rejected`. Flash contents are NEVER trusted: range enforcement at
/// the shell/YAML edges says nothing about records written by other
/// firmware versions, so every record is re-validated here, at the one
/// choke point every persisted value must pass through.
pub fn load<F: KvFlash, G: ParamGroup>(flash: &mut F, cfg: &mut G) -> LoadReport {
    let mut report = LoadReport::default();
    let Some((sector, _, end, _)) = active(flash) else {
        return report;
    };
    let table = HashTable::build::<G>();
    let mut w = [0u8; WORD];
    let mut offset = WORD;
    while offset < end {
        flash.read(sector, offset, &mut w);
        offset += WORD;
        let Some((hash, value)) = parse_record(&w) else {
            break;
        };
        match table.find(hash) {
            Some(idx) => {
                if value.is_finite() && G::param_meta(idx).in_range(value) {
                    cfg.param_set_f32(idx, value);
                    report.applied += 1;
                } else {
                    report.rejected += 1;
                }
            }
            None => report.skipped += 1,
        }
    }
    report
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoadReport {
    /// Records applied to the config.
    pub applied: u32,
    /// Records whose name hash matched no parameter in this build's schema.
    pub skipped: u32,
    /// Records rejected by validation (non-finite or out of the
    /// parameter's `ParamMeta` range).
    pub rejected: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SaveReport {
    /// Records appended (params whose value changed vs the stored state).
    pub appended: u32,
    /// Whether a compaction cycle ran.
    pub compacted: bool,
}

/// Bit-exact f32 comparison (the store's notion of "unchanged"). Avoids
/// both `==` NaN pitfalls and -0.0/0.0 aliasing surprises.
fn f32_eq(a: f32, b: f32) -> bool {
    a.to_le_bytes() == b.to_le_bytes()
}

fn value_of<G: ParamGroup>(cfg: &G, idx: usize) -> f32 {
    cfg.param_get(idx).map_or(0.0, |v| v.as_f32())
}

/// True if any parameter of `current` differs from `baked` — i.e. whether
/// a pruned store would hold any records at all.
fn has_overrides<G: ParamGroup>(current: &G, baked: &G) -> bool {
    (0..G::COUNT).any(|idx| !f32_eq(value_of(current, idx), value_of(baked, idx)))
}

/// Persist `current`: append one record per parameter whose value differs
/// from the *stored effective* state (baked defaults + existing records).
/// Compacts into the other sector when the log is full, pruning records
/// equal to `baked`. Formats sector 0 on first use.
///
/// Cheap by design — no erase, one word per changed param. The cost is
/// that reverting a param appends a record holding the baked value rather
/// than removing anything (there are no tombstones), so the key stays
/// pinned against a later change to the baked defaults. Use
/// [`save_pruned`] to drop those.
///
/// `baked` must be the same compile-time defaults the boot path overlays.
pub fn save<F: KvFlash, G: ParamGroup + Clone>(
    flash: &mut F,
    current: &G,
    baked: &G,
) -> Result<SaveReport, StoreError> {
    save_inner(flash, current, baked, false)
}

/// Persist `current` as the *whole* override set: rewrite the store into
/// the other sector holding exactly the params that differ from `baked`,
/// and nothing else.
///
/// This is the only way to make a key stop being overridden. Because the
/// log has no tombstone record, [`save`] can only ever add "override to
/// X"; a `param reset` followed by a plain save therefore writes
/// `X == baked`, which is correct today and stale the moment the baked
/// defaults change (a re-flashed vehicle YAML). Two cases need that
/// undone rather than restated:
///
/// - A YAML edit that a stale record shadows. `reset` the key, then
///   prune, and the store no longer mentions it.
/// - A vehicle swap. Records carry a name hash and a value, never a
///   vehicle identity, so the previous airframe's saved mass/inertia
///   replay onto the new one's defaults. `reset all` + prune is the
///   clean slate.
///
/// Costs an erase (~1–2 s blocking on the H7, watchdog-extended by the
/// [`KvFlash`] impl) and is refused while armed, which is why it is a
/// separate entry point instead of the default. Crash-safe by the same
/// records-first/header-last ordering as compaction.
///
/// `SaveReport::appended` counts the records that *survived* — the size
/// of the pruned override set, not a delta.
pub fn save_pruned<F: KvFlash, G: ParamGroup + Clone>(
    flash: &mut F,
    current: &G,
    baked: &G,
) -> Result<SaveReport, StoreError> {
    save_inner(flash, current, baked, true)
}

fn save_inner<F: KvFlash, G: ParamGroup + Clone>(
    flash: &mut F,
    current: &G,
    baked: &G,
    prune: bool,
) -> Result<SaveReport, StoreError> {
    let mut report = SaveReport::default();

    // Effective stored state = baked + replay.
    let mut effective = baked.clone();
    load(flash, &mut effective);

    // Diff: indices to append.
    let mut pending = 0usize;
    for idx in 0..G::COUNT {
        if !f32_eq(value_of(current, idx), value_of(&effective, idx)) {
            pending += 1;
        }
    }

    let state = active(flash);

    if prune {
        // A prune is defined by what the store ends up holding, not by a
        // delta, so `pending == 0` is not an exit condition — the whole
        // point is to rewrite a log whose replay already matches
        // `current` but does so via redundant records. The one genuine
        // no-op is an unformatted store with nothing to put in it: don't
        // spend an erase proving it is already empty.
        if state.is_none() && !has_overrides(current, baked) {
            return Ok(report);
        }
    } else if pending == 0 {
        return Ok(report);
    }

    let words = flash.sector_size() / WORD;
    // A prune must rewrite, so it never takes the append path however
    // much room the active sector has left.
    let fits = !prune
        && match state {
            Some((_, _, offset, clean)) => clean && (words * WORD - offset) / WORD >= pending,
            None => false,
        };

    let (sector, mut offset) = if let (Some((sector, _, offset, _)), true) = (state, fits) {
        (sector, offset)
    } else {
        // Format or compact into the other (or first) sector.
        report.compacted = state.is_some();
        let (target, generation) = match state {
            Some((s, g, _, _)) => (1 - s, g + 1),
            None => (0, 1),
        };
        // Pruned override set relative to the baked defaults.
        let overrides = (0..G::COUNT)
            .filter(|&idx| !f32_eq(value_of(current, idx), value_of(baked, idx)))
            .count();
        if overrides + 1 > words {
            return Err(StoreError::Full);
        }
        flash.erase_sector(target)?;
        // Records FIRST, header LAST: until the header lands, the target
        // sector has no valid header and loses the generation
        // arbitration — a power cut mid-compaction leaves the fully
        // intact old sector authoritative instead of a silently
        // truncated new one.
        let mut off = WORD;
        for idx in 0..G::COUNT {
            let v = value_of(current, idx);
            if !f32_eq(v, value_of(baked, idx)) {
                let hash = hash_of_idx::<G>(idx).ok_or(StoreError::Corrupt)?;
                flash.write_word(target, off, &record_word(hash, v))?;
                off += WORD;
                report.appended += 1;
            }
        }
        flash.write_word(target, 0, &header_word(generation))?;
        // New sector complete — retire the old one last (power-loss safe:
        // between the header write and this erase, two valid headers
        // exist and the higher generation wins).
        if let Some((old, _, _, _)) = state {
            flash.erase_sector(old)?;
        }
        return Ok(report);
    };

    // Plain append.
    for idx in 0..G::COUNT {
        let v = value_of(current, idx);
        if !f32_eq(v, value_of(&effective, idx)) {
            let hash = hash_of_idx::<G>(idx).ok_or(StoreError::Corrupt)?;
            flash.write_word(sector, offset, &record_word(hash, v))?;
            offset += WORD;
            report.appended += 1;
        }
    }
    Ok(report)
}

// ---------------------------------------------------------------------------
// Tests (host): RAM mock flash
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{FirmwareConfig, PARAM_COUNT};

    const SECTOR: usize = 4096; // 128 words — small for fast fill tests

    struct MockFlash {
        mem: [Vec<u8>; 2],
        erases: u32,
        writes: u32,
    }

    impl MockFlash {
        fn new() -> Self {
            Self {
                mem: [vec![0xFF; SECTOR], vec![0xFF; SECTOR]],
                erases: 0,
                writes: 0,
            }
        }
    }

    impl KvFlash for MockFlash {
        fn sector_size(&self) -> usize {
            SECTOR
        }
        fn read(&mut self, sector: u8, offset: usize, buf: &mut [u8]) {
            buf.copy_from_slice(&self.mem[sector as usize][offset..offset + buf.len()]);
        }
        fn write_word(
            &mut self,
            sector: u8,
            offset: usize,
            word: &[u8; WORD],
        ) -> Result<(), StoreError> {
            assert_eq!(offset % WORD, 0, "unaligned write");
            let dst = &mut self.mem[sector as usize][offset..offset + WORD];
            // NOR semantics: can only clear bits from erased state.
            assert!(dst.iter().all(|&b| b == 0xFF), "write to non-erased word");
            dst.copy_from_slice(word);
            self.writes += 1;
            Ok(())
        }
        fn erase_sector(&mut self, sector: u8) -> Result<(), StoreError> {
            self.mem[sector as usize].fill(0xFF);
            self.erases += 1;
            Ok(())
        }
    }

    fn baked() -> FirmwareConfig {
        crate::params::tests_support::test_config()
    }

    /// The store keys on fnv1a-32 of the param name — a collision between
    /// two schema names would silently alias them in flash.
    #[test]
    fn no_name_hash_collisions() {
        let mut seen = std::collections::HashMap::new();
        for idx in 0..PARAM_COUNT {
            let name = ParamName::of::<FirmwareConfig>(idx).unwrap();
            let h = name_hash(name.as_str());
            if let Some(prev) = seen.insert(h, name.as_str().to_string()) {
                panic!("hash collision: {} vs {}", prev, name.as_str());
            }
        }
    }

    #[test]
    fn blank_flash_loads_nothing_and_first_save_formats() {
        let mut flash = MockFlash::new();
        let mut cfg = baked();
        assert_eq!(load(&mut flash, &mut cfg), LoadReport::default());

        let mut tuned = baked();
        assert!(tuned.set_named("mpc_w_pos_x", 550.0));
        assert!(tuned.set_named("mass", 0.71));
        let r = save(&mut flash, &tuned, &baked()).unwrap();
        assert_eq!(r.appended, 2);
        assert_eq!(flash.erases, 1); // formatting erase of sector 0

        let mut reloaded = baked();
        let lr = load(&mut flash, &mut reloaded);
        assert_eq!(lr.applied, 2);
        assert_eq!(lr.skipped, 0);
        assert_eq!(reloaded.mpc.pos_weight[0], 550.0);
        assert_eq!(reloaded.airframe.body.mass_kg, 0.71);
    }

    #[test]
    fn unchanged_save_appends_nothing() {
        let mut flash = MockFlash::new();
        let mut tuned = baked();
        tuned.set_named("indi_rate_r", 90.0);
        save(&mut flash, &tuned, &baked()).unwrap();
        let writes_before = flash.writes;
        let r = save(&mut flash, &tuned, &baked()).unwrap();
        assert_eq!(r.appended, 0);
        assert_eq!(flash.writes, writes_before);
    }

    #[test]
    fn last_wins_and_revert_to_default_round_trips() {
        let mut flash = MockFlash::new();
        let base = baked();

        let mut tuned = base.clone();
        tuned.set_named("indi_sync_hz", 20.0);
        save(&mut flash, &tuned, &base).unwrap();

        // Revert to the baked value: appends a default-valued record
        // (pruned later, at compaction).
        let r = save(&mut flash, &base, &base).unwrap();
        assert_eq!(r.appended, 1);

        let mut reloaded = base.clone();
        load(&mut flash, &mut reloaded);
        assert_eq!(
            reloaded.indi.controller.sync_filter_hz,
            base.indi.controller.sync_filter_hz
        );
    }

    #[test]
    fn compaction_prunes_and_preserves() {
        let mut flash = MockFlash::new();
        let base = baked();

        // A persistent override that must survive every compaction.
        let mut tuned = base.clone();
        tuned.set_named("mass", 0.83);
        save(&mut flash, &tuned, &base).unwrap();

        // Churn one param until the 128-word sector must compact.
        for i in 0..200 {
            let mut t = tuned.clone();
            t.set_named("mpc_w_pos_x", 500.0 + i as f32);
            save(&mut flash, &t, &base).unwrap();
            tuned = t;
        }

        let (sector, generation, offset, _) = active(&mut flash).unwrap();
        assert!(generation > 1, "expected at least one compaction");
        // After the last compaction + subsequent appends, the log holds far
        // fewer records than the 200 writes we made.
        assert!(offset / WORD <= 128);

        let mut reloaded = base.clone();
        let lr = load(&mut flash, &mut reloaded);
        assert_eq!(reloaded.airframe.body.mass_kg, 0.83);
        assert_eq!(reloaded.mpc.pos_weight[0], 699.0);
        assert!(lr.applied >= 2);
        // Only one sector is valid after compaction.
        let other = 1 - sector;
        let mut w = [0u8; WORD];
        flash.read(other, 0, &mut w);
        assert!(parse_header(&w).is_none(), "old sector not retired");
    }

    #[test]
    fn higher_generation_wins_after_interrupted_compaction() {
        let mut flash = MockFlash::new();
        // Hand-craft the power-loss-between-erases state: both sectors
        // valid, different generations and values.
        flash.write_word(0, 0, &header_word(3)).unwrap();
        flash
            .write_word(0, WORD, &record_word(name_hash("mass"), 0.91))
            .unwrap();
        flash.write_word(1, 0, &header_word(2)).unwrap();
        flash
            .write_word(1, WORD, &record_word(name_hash("mass"), 0.55))
            .unwrap();

        let mut cfg = baked();
        load(&mut flash, &mut cfg);
        assert_eq!(cfg.airframe.body.mass_kg, 0.91);
    }

    #[test]
    fn torn_record_ends_log_without_losing_prior_records() {
        let mut flash = MockFlash::new();
        let base = baked();
        let mut tuned = base.clone();
        tuned.set_named("mass", 0.66);
        save(&mut flash, &tuned, &base).unwrap();

        // Simulate a torn append: garbage word after the valid record.
        let mut torn = [0u8; WORD];
        torn[0..4].copy_from_slice(&REC_MAGIC.to_le_bytes());
        torn[4..8].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        // (bad CRC)
        flash.write_word(0, 2 * WORD, &torn).unwrap();

        let mut reloaded = base.clone();
        let lr = load(&mut flash, &mut reloaded);
        assert_eq!(lr.applied, 1);
        assert_eq!(reloaded.airframe.body.mass_kg, 0.66);

        // The torn word occupies the append point and is not erased, so a
        // plain append is illegal (the NOR-semantics mock would panic).
        // `save` must detect the dirty tail and route through compaction.
        let mut t2 = base.clone();
        t2.set_named("mass", 0.77);
        let r = save(&mut flash, &t2, &base).unwrap();
        assert!(r.compacted, "dirty tail must force compaction");
        let mut re2 = base.clone();
        load(&mut flash, &mut re2);
        assert_eq!(re2.airframe.body.mass_kg, 0.77);
    }

    /// A compaction target that lost power before its header was written
    /// must NOT win the arbitration — the intact old sector stays
    /// authoritative (records-first/header-last ordering).
    #[test]
    fn headerless_compaction_target_loses_arbitration() {
        let mut flash = MockFlash::new();
        // Old sector: valid, generation 5, one override.
        flash.write_word(0, 0, &header_word(5)).unwrap();
        flash
            .write_word(0, WORD, &record_word(name_hash("mass"), 0.83))
            .unwrap();
        // Target sector: records written, power lost before the header.
        flash
            .write_word(1, WORD, &record_word(name_hash("mass"), 0.11))
            .unwrap();

        let mut cfg = baked();
        load(&mut flash, &mut cfg);
        assert_eq!(cfg.airframe.body.mass_kg, 0.83, "old sector must win");

        // The next save recovers: it must not trust the headerless sector.
        let mut t = baked();
        t.set_named("mass", 0.9);
        save(&mut flash, &t, &baked()).unwrap();
        let mut re = baked();
        load(&mut flash, &mut re);
        assert_eq!(re.airframe.body.mass_kg, 0.9);
    }

    /// Flash records are never trusted: non-finite and out-of-range
    /// values are rejected on replay and counted, leaving the baked
    /// value in place.
    #[test]
    fn replay_rejects_invalid_records() {
        let mut flash = MockFlash::new();
        flash.write_word(0, 0, &header_word(1)).unwrap();
        // mass has ParamMeta range [0.05, 20.0] kg.
        flash
            .write_word(0, WORD, &record_word(name_hash("mass"), 0.001))
            .unwrap();
        flash
            .write_word(0, 2 * WORD, &record_word(name_hash("mass"), f32::NAN))
            .unwrap();
        flash
            .write_word(0, 3 * WORD, &record_word(name_hash("indi_sync_hz"), f32::INFINITY))
            .unwrap();
        flash
            .write_word(0, 4 * WORD, &record_word(name_hash("mass"), 0.71))
            .unwrap();

        let base = baked();
        let mut cfg = base.clone();
        let r = load(&mut flash, &mut cfg);
        assert_eq!(r.rejected, 3);
        assert_eq!(r.applied, 1);
        assert_eq!(cfg.airframe.body.mass_kg, 0.71);
        assert_eq!(
            cfg.indi.controller.sync_filter_hz,
            base.indi.controller.sync_filter_hz
        );
    }

    #[test]
    fn unknown_hash_is_skipped_not_fatal() {
        let mut flash = MockFlash::new();
        flash.write_word(0, 0, &header_word(1)).unwrap();
        flash
            .write_word(0, WORD, &record_word(name_hash("param_from_the_future"), 1.0))
            .unwrap();
        flash
            .write_word(0, 2 * WORD, &record_word(name_hash("mass"), 0.62))
            .unwrap();
        let mut cfg = baked();
        let lr = load(&mut flash, &mut cfg);
        assert_eq!(lr.skipped, 1);
        assert_eq!(lr.applied, 1);
        assert_eq!(cfg.airframe.body.mass_kg, 0.62);
    }
    // ── save_pruned ────────────────────────────────────────────────────

    /// The motivating bug. Reverting a param with a plain `save` records
    /// the baked value rather than dropping the key, so the record then
    /// shadows a later change to the baked defaults — a re-flashed vehicle
    /// YAML whose edit "does not take effect". Same mechanism as a vehicle
    /// swap, where the previous airframe's records replay onto the new
    /// one's defaults. `save_pruned` drops the record instead.
    #[test]
    fn prune_lets_a_changed_baked_default_take_effect() {
        let mut flash = MockFlash::new();
        let base = baked();
        let original = base.mpc.pos_weight[0];
        assert_ne!(original, 700.0, "test would be vacuous");

        // Tune a param, then decide the baked value was right after all.
        let mut tuned = base.clone();
        assert!(tuned.set_named("mpc_w_pos_x", 550.0));
        save(&mut flash, &tuned, &base).unwrap();
        let r = save(&mut flash, &base, &base).unwrap();
        assert_eq!(r.appended, 1, "revert appends a baked-valued record");

        // Re-flash with an edited YAML: the baked default moves.
        let mut edited = base.clone();
        assert!(edited.set_named("mpc_w_pos_x", 700.0));

        // Boot the new firmware — the stale record shadows the edit.
        let mut booted = edited.clone();
        load(&mut flash, &mut booted);
        assert_eq!(
            booted.mpc.pos_weight[0], original,
            "stale record must shadow the new baked default (the bug)",
        );

        // `param reset mpc_w_pos_x` + plain `param save` fixes *this*
        // boot, but only by appending yet another baked-valued record —
        // the can kicked one YAML edit down the road, not removed.
        assert_eq!(save(&mut flash, &edited, &edited).unwrap().appended, 1);
        let mut booted = edited.clone();
        load(&mut flash, &mut booted);
        assert_eq!(booted.mpc.pos_weight[0], 700.0, "correct for now");

        // Prove it: the next YAML edit is shadowed exactly as before.
        let mut edited2 = base.clone();
        assert!(edited2.set_named("mpc_w_pos_x", 900.0));
        let mut booted = edited2.clone();
        load(&mut flash, &mut booted);
        assert_eq!(booted.mpc.pos_weight[0], 700.0, "shadowed again");

        // `param reset` + `param save --prune` drops the key instead.
        let r = save_pruned(&mut flash, &edited2, &edited2).unwrap();
        assert_eq!(r.appended, 0, "nothing differs from baked");
        assert!(r.compacted);
        let mut booted = edited2.clone();
        let lr = load(&mut flash, &mut booted);
        assert_eq!(lr.applied, 0, "store must be empty");
        assert_eq!(booted.mpc.pos_weight[0], 900.0);

        // And it stays fixed across further YAML edits.
        let mut edited3 = base.clone();
        assert!(edited3.set_named("mpc_w_pos_x", 1100.0));
        let mut booted = edited3.clone();
        load(&mut flash, &mut booted);
        assert_eq!(booted.mpc.pos_weight[0], 1100.0);
    }

    #[test]
    fn prune_keeps_live_overrides_and_drops_the_rest() {
        let mut flash = MockFlash::new();
        let base = baked();

        let mut tuned = base.clone();
        assert!(tuned.set_named("mass", 0.83));
        assert!(tuned.set_named("indi_sync_hz", 20.0));
        save(&mut flash, &tuned, &base).unwrap();

        // Revert one of the two; the other is a real tune worth keeping.
        let mut reverted = tuned.clone();
        assert!(reverted.set_named("indi_sync_hz", base.indi.controller.sync_filter_hz));
        let r = save_pruned(&mut flash, &reverted, &base).unwrap();
        assert_eq!(r.appended, 1);
        assert!(r.compacted);

        // Header + exactly one record.
        let (_, _, offset, _) = active(&mut flash).unwrap();
        assert_eq!(offset, 2 * WORD);

        let mut re = base.clone();
        let lr = load(&mut flash, &mut re);
        assert_eq!(lr.applied, 1);
        assert_eq!(re.airframe.body.mass_kg, 0.83);
        assert_eq!(
            re.indi.controller.sync_filter_hz,
            base.indi.controller.sync_filter_hz
        );
    }

    /// A prune is defined by the resulting store, not by a delta, so it
    /// must still rewrite when the replay already matches `current` —
    /// that is precisely the redundant-record case it exists to clean up.
    /// The one true no-op is an unformatted store with nothing to put in
    /// it: don't spend a 1-2 s erase proving it is already empty.
    #[test]
    fn prune_with_nothing_to_do_does_not_erase() {
        let mut flash = MockFlash::new();
        let base = baked();
        assert_eq!(save_pruned(&mut flash, &base, &base).unwrap(), SaveReport::default());
        assert_eq!(flash.erases, 0);
        assert_eq!(flash.writes, 0);
    }

    #[test]
    fn repeated_prune_is_stable() {
        let mut flash = MockFlash::new();
        let base = baked();
        let mut tuned = base.clone();
        assert!(tuned.set_named("mass", 0.83));

        // First prune formats sector 0 (no prior store to retire).
        let r = save_pruned(&mut flash, &tuned, &base).unwrap();
        assert!(!r.compacted);
        let (s1, g1, off1, _) = active(&mut flash).unwrap();

        let r = save_pruned(&mut flash, &tuned, &base).unwrap();
        assert_eq!(r.appended, 1);
        let (s2, g2, off2, _) = active(&mut flash).unwrap();
        assert_eq!(s2, 1 - s1, "a rewrite alternates sectors");
        assert!(g2 > g1);
        assert_eq!(off2, off1, "same content, same length");

        let mut w = [0u8; WORD];
        flash.read(s1, 0, &mut w);
        assert!(parse_header(&w).is_none(), "old sector not retired");

        let mut re = base.clone();
        load(&mut flash, &mut re);
        assert_eq!(re.airframe.body.mass_kg, 0.83);
    }

    /// A torn tail makes a plain append illegal; a prune already routes
    /// through the rewrite path, so it recovers the store as a side effect.
    #[test]
    fn prune_recovers_a_torn_tail() {
        let mut flash = MockFlash::new();
        let base = baked();
        let mut tuned = base.clone();
        assert!(tuned.set_named("mass", 0.66));
        save(&mut flash, &tuned, &base).unwrap();

        let mut torn = [0u8; WORD];
        torn[0..4].copy_from_slice(&REC_MAGIC.to_le_bytes());
        torn[4..8].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        flash.write_word(0, 2 * WORD, &torn).unwrap();

        let r = save_pruned(&mut flash, &tuned, &base).unwrap();
        assert_eq!(r.appended, 1);
        let (_, _, offset, clean) = active(&mut flash).unwrap();
        assert_eq!(offset, 2 * WORD);
        assert!(clean, "tail is appendable again");

        let mut re = base.clone();
        load(&mut flash, &mut re);
        assert_eq!(re.airframe.body.mass_kg, 0.66);
    }
}
