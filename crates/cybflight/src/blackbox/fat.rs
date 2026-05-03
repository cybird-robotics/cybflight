//! FAT-filesystem write path for the blackbox subsystem.
//!
//! Layers (top → bottom):
//! ```text
//!   write_file<P: FileBody>
//!     └─ embedded_fatfs::FileSystem        ── FAT directory + file ops
//!          └─ embedded_partitions::Scheme  ── auto-detect MBR vs superfloppy,
//!                                              slice into FAT partition
//!               └─ block_device_adapters::BufStream  ── byte ↔ 512 B block shim
//!                    └─ embassy_stm32::sdmmc::StorageDevice  ── BlockDevice<512>
//!                         └─ embassy_stm32::sdmmc::Sdmmc
//! ```
//!
//! Why the partition layer is non-optional: SD cards typically ship
//! with a Master Boot Record at LBA 0 and the FAT volume starting at
//! LBA 8192. Handing `BufStream` straight to
//! `embedded_fatfs::FileSystem::new` would make it parse the MBR as a
//! FAT BPB and reject it. `Scheme::open` reads sector 0 and either
//! returns a `StreamSlice` over the FAT partition (MBR case) or the
//! original stream untouched (superfloppy).
//!
//! One [`SdmmcBlockStore::open_session`] holds the `StorageDevice`
//! alive across all FAT operations, so `acquire → mount → create →
//! write → flush → unmount` pays exactly one CMD0/ACMD41 per request.
//!
//! ## Adding a new file body
//!
//! Implement [`FileBody`] for a payload type and call [`write_file`].
//! The trait is the seam by which the blackbox skeleton, future MCAP
//! capture, and the touch-sd canned text all share this single FAT
//! mount/unmount path.

use crate::hal::sdmmc;
use block_device_adapters::{BufStream, BufStreamError};
use embedded_fatfs::{
    DefaultTimeProvider, Error as FatError, FileSystem, FsOptions, LossyOemCpConverter,
};
use embedded_io_async::Write;
use embedded_partitions::mbr::{Error as MbrError, Mbr, Scheme};

use super::sdmmc_block::SdmmcBlockStore;

/// Max filename surfaced to callers in [`EntrySummary`]. Long enough
/// to fit `flight_NNNN.mcap` plus headroom for any other names a
/// user might drop on the card.
pub const MAX_FILENAME: usize = 32;
pub type EntryName = heapless::String<MAX_FILENAME>;

/// One root-directory entry, materialised so callers can drop the
/// underlying `DirEntry` borrow before consuming results. Names are
/// best-effort ASCII (non-ASCII units / OEM bytes become `?`).
#[derive(Clone)]
pub struct EntrySummary {
    pub name: EntryName,
    /// File length in bytes, capped to `u32::MAX` (4 GiB). Files
    /// larger than 4 GiB are unrepresentable on FAT32 anyway.
    pub size: u32,
}

/// Unified per-op error: covers both source-side failures (no
/// pubsub subscriber slot, sample timeout) and sink-side failures
/// (each stage of the FAT pipeline).
///
/// Variants are listed in roughly the order an op encounters them:
/// pre-FAT data acquisition first, then card → partition → FS →
/// file → flush → unmount.
#[derive(Clone, Copy, defmt::Format)]
pub enum OpError {
    // ── Pre-FAT (data source) ─────────────────────────────────────
    /// Pubsub channel ran out of subscriber slots — bump the `SUBS`
    /// const on the channel.
    NoSubscriberSlot,
    /// Subscriber didn't see a fresh message within the op's
    /// timeout — sensor task may be down or the topic is silent.
    NoMessageInTimeout,
    // ── Sink (FAT) ────────────────────────────────────────────────
    /// Couldn't acquire the SD card itself (CMD0/ACMD41).
    CardAcquire,
    /// Sector 0 looks like neither an MBR nor a FAT BPB — card is
    /// blank, GPT-formatted, or its first sector is corrupt.
    NoPartitionTable,
    /// MBR present but no FAT-typed partition entry — card is
    /// formatted as ext4/exFAT/etc.
    NoFatPartition,
    /// MBR / partition I/O failed before mount.
    PartitionIo,
    /// Couldn't parse the FAT superblock — partition exists but is
    /// not FAT32/FAT16, or its boot sector is corrupted.
    Mount,
    /// Couldn't create the directory entry — disk full, name collision, etc.
    Create,
    /// Body's `write` returned an I/O fault, or `flush` failed.
    Write,
    /// Final unmount-time flush failed; data may not have hit the card.
    Unmount,
}

/// Pluggable file payload — what to write into the just-created FAT
/// file. Implementors are responsible for byte counting; the count is
/// returned to the caller for shell-side reporting.
///
/// Stage-1 of the blackbox plan ships two impls: [`Bytes`] (for the
/// existing `touch sd` canned text) and `McapSkeleton` (in
/// `super::skeleton`). Future stages add `McapCapture<...>` with the
/// same shape — generic over `Write`, no FAT details leak in.
#[allow(async_fn_in_trait)]
pub trait FileBody {
    /// Write the body. Return total bytes written so the shell can
    /// report file size without a separate stat.
    async fn write<W: Write>(&mut self, w: &mut W) -> Result<u32, W::Error>;
}

/// Plain byte payload. Used by `touch sd`.
pub struct Bytes<'a>(pub &'a [u8]);

impl FileBody for Bytes<'_> {
    async fn write<W: Write>(&mut self, w: &mut W) -> Result<u32, W::Error> {
        w.write_all(self.0).await?;
        Ok(self.0.len() as u32)
    }
}

/// Open the card, locate the FAT partition (or accept a superfloppy
/// layout), create `path` in the root directory truncated to empty,
/// hand it to `body.write(...)`, then flush + unmount.
///
/// Body is borrowed by `&mut` rather than consumed so the caller can
/// read post-write state from it (e.g. message / drop counters
/// updated during a capture loop). Existing callers pass a freshly-
/// constructed body bound to a local variable; once Rust drops it at
/// statement end, the borrow ends.
///
/// Returns the byte count reported by the body on success.
pub async fn write_file<P: FileBody>(
    store: &mut SdmmcBlockStore,
    path: &str,
    body: &mut P,
) -> Result<u32, OpError> {
    let storage = store.open_session().await.map_err(|_| {
        defmt::warn!("blackbox/fat: open_session failed");
        OpError::CardAcquire
    })?;

    // BlockDevice<512> → byte-level Read/Write/Seek
    let buf = BufStream::<_, 512>::new(storage);

    // **MBR-only.** SD cards from any modern host OS ship with an
    // MBR; the superfloppy layout (FAT BPB at LBA 0) is 1990s legacy.
    // Accepting both doubles every FAT-layer monomorphization
    // (`FileSystem<StreamSlice<BufStream<_>>>` vs
    // `FileSystem<BufStream<_>>`), inflating the binary by tens of
    // KB. The unsupported branches return a clear error so the user
    // knows to reformat.
    match Scheme::open(buf).await.map_err(map_mbr_error)? {
        Scheme::Mbr(mbr) => mount_mbr_partition(mbr, path, body).await,
        Scheme::Superfloppy(_) => {
            defmt::warn!(
                "blackbox/fat: card uses superfloppy layout (FAT BPB at LBA 0); \
                 reformat as MBR + FAT32 (the format any desktop OS uses by default)"
            );
            Err(OpError::NoPartitionTable)
        }
        Scheme::Unknown(_) => {
            defmt::warn!(
                "blackbox/fat: sector 0 has no MBR signature and no FAT BPB \
                 (reformat as MBR + FAT32)"
            );
            Err(OpError::NoPartitionTable)
        }
    }
}

async fn mount_mbr_partition<IO, P>(
    mut mbr: Mbr<IO>,
    path: &str,
    body: &mut P,
) -> Result<u32, OpError>
where
    IO: embedded_io_async::Read + Write + embedded_io_async::Seek,
    P: FileBody,
{
    let idx = mbr
        .iter_used()
        .find(|(_, p)| p.is_fat())
        .map(|(i, _)| i)
        .ok_or_else(|| {
            defmt::warn!("blackbox/fat: no FAT-typed partition in MBR");
            OpError::NoFatPartition
        })?;
    defmt::info!("blackbox/fat: mounting MBR partition #{}", idx);

    let mut slice = mbr.open_partition(idx).await.map_err(|_| {
        // We deliberately omit the inner error via Debug2Format —
        // the Debug impl pulls in formatting code for every error
        // variant of the partition stack and inflates the binary by
        // ~5 KB. The generic `PartitionIo` is enough to point a user
        // at "card or partition table broken; reformat".
        defmt::warn!("blackbox/fat: open_partition failed");
        OpError::PartitionIo
    })?;
    write_into_filesystem(&mut slice, path, body).await
}

/// Mount the filesystem on the given byte stream, write the file via
/// `body`, flush + unmount.
///
/// The flush dance is non-obvious, so document it: `file.flush()`
/// commits the dirty directory entry plus the BufStream block cache.
/// We then explicitly `close()` the file rather than letting it drop
/// — `Drop` only warns on a still-dirty entry, it does NOT flush.
/// `fs.unmount()` writes the FS info sector and clears the FAT
/// dirty flag; without it, on next mount the host OS sees the volume
/// in a "needs check" state and may discard writes from this
/// session.
async fn write_into_filesystem<IO, P>(io: IO, path: &str, body: &mut P) -> Result<u32, OpError>
where
    IO: embedded_io_async::Read + Write + embedded_io_async::Seek,
    P: FileBody,
{
    let fs = FileSystem::new(io, FsOptions::new()).await.map_err(|e| {
        log_fat_error("mount", &e);
        OpError::Mount
    })?;

    let bytes = {
        let root = fs.root_dir();
        let mut file = root.create_file(path).await.map_err(|e| {
            log_fat_error("create_file", &e);
            OpError::Create
        })?;
        file.truncate().await.map_err(|e| {
            log_fat_error("truncate", &e);
            OpError::Write
        })?;
        let n = body.write(&mut file).await.map_err(|e| {
            log_fat_error("body.write", &e);
            OpError::Write
        })?;
        file.flush().await.map_err(|e| {
            log_fat_error("flush", &e);
            OpError::Write
        })?;
        // Explicit close (consumes file) — `Drop` only *warns* on a
        // still-dirty entry, so we pin the close before unmount.
        file.close().await.map_err(|e| {
            log_fat_error("close", &e);
            OpError::Write
        })?;
        n
    };

    fs.unmount().await.map_err(|e| {
        log_fat_error("unmount", &e);
        OpError::Unmount
    })?;
    defmt::info!("blackbox/fat: wrote /{} ({} bytes)", path, bytes);
    Ok(bytes)
}

fn map_mbr_error<E: core::fmt::Debug>(e: MbrError<E>) -> OpError {
    defmt::warn!(
        "blackbox/fat: MBR scheme open: {:?}",
        defmt::Debug2Format(&e)
    );
    OpError::PartitionIo
}

/// Open the card, locate the FAT volume, walk its **root directory**
/// and call `visit` once per regular file. Skips subdirectories and
/// hidden/system entries silently. Mounts read-only-ish (no writes
/// happen) and unmounts cleanly.
///
/// `visit` receives the file name as best-effort ASCII (UCS-2 LFN
/// units outside ASCII become `?`) plus the file size in bytes
/// (saturated at `u32::MAX`). Use it to populate a heapless::Vec, run
/// a max-scan, etc.
pub async fn scan_root_files<F>(store: &mut SdmmcBlockStore, visit: &mut F) -> Result<(), OpError>
where
    F: FnMut(&str, u32),
{
    let storage = store.open_session().await.map_err(|_| {
        defmt::warn!("blackbox/fat: open_session failed");
        OpError::CardAcquire
    })?;
    let buf = BufStream::<_, 512>::new(storage);
    // MBR-only — see `write_file` for rationale.
    match Scheme::open(buf).await.map_err(map_mbr_error)? {
        Scheme::Mbr(mbr) => scan_mbr_partition(mbr, visit).await,
        Scheme::Superfloppy(_) | Scheme::Unknown(_) => {
            defmt::warn!("blackbox/fat: scan: not an MBR-formatted card (reformat as MBR + FAT32)");
            Err(OpError::NoPartitionTable)
        }
    }
}

/// Scan the root for the highest existing `<prefix>NNNN<suffix>` and
/// return that NNNN. Returns 0 when no matching file exists, so
/// callers should add 1 to obtain the next sequence to use.
///
/// Matching is case-insensitive — FAT short-name aliases come back
/// uppercase, so a card formatted on a host that lost its LFN
/// entries still resolves correctly.
pub async fn scan_highest_seq(
    store: &mut SdmmcBlockStore,
    prefix: &str,
    suffix: &str,
) -> Result<u32, OpError> {
    let mut highest: u32 = 0;
    let mut visit = |name: &str, _size: u32| {
        if let Some(n) = parse_seq(name, prefix, suffix) {
            if n > highest {
                highest = n;
            }
        }
    };
    scan_root_files(store, &mut visit).await?;
    Ok(highest)
}

/// Parse `"<prefix>NNNN<suffix>"` (case-insensitive) into NNNN.
/// Accepts any unsigned u32; the typical forms are 4-digit zero-
/// padded, but a card that's been hand-edited might carry shorter
/// or longer numbers and they should still bump the counter.
fn parse_seq(name: &str, prefix: &str, suffix: &str) -> Option<u32> {
    if name.len() < prefix.len() + suffix.len() {
        return None;
    }
    let (head, rest) = name.split_at(prefix.len());
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let (mid, tail) = rest.split_at(rest.len() - suffix.len());
    if !tail.eq_ignore_ascii_case(suffix) {
        return None;
    }
    mid.parse().ok()
}

async fn scan_mbr_partition<IO, F>(mut mbr: Mbr<IO>, visit: &mut F) -> Result<(), OpError>
where
    IO: embedded_io_async::Read + Write + embedded_io_async::Seek,
    F: FnMut(&str, u32),
{
    let idx = mbr
        .iter_used()
        .find(|(_, p)| p.is_fat())
        .map(|(i, _)| i)
        .ok_or_else(|| {
            defmt::warn!("blackbox/fat: no FAT-typed partition in MBR");
            OpError::NoFatPartition
        })?;
    let mut slice = mbr.open_partition(idx).await.map_err(|e| {
        defmt::warn!(
            "blackbox/fat: open_partition: {:?}",
            defmt::Debug2Format(&e)
        );
        OpError::PartitionIo
    })?;
    scan_in_filesystem(&mut slice, visit).await
}

async fn scan_in_filesystem<IO, F>(io: IO, visit: &mut F) -> Result<(), OpError>
where
    IO: embedded_io_async::Read + Write + embedded_io_async::Seek,
    F: FnMut(&str, u32),
{
    let fs = FileSystem::new(io, FsOptions::new()).await.map_err(|e| {
        log_fat_error("mount", &e);
        OpError::Mount
    })?;
    {
        let root = fs.root_dir();
        let mut iter = root.iter();
        let mut name_buf: EntryName = heapless::String::new();
        loop {
            match iter.next().await {
                None => break,
                Some(Ok(entry)) => {
                    if !entry.is_file() {
                        continue;
                    }
                    name_buf.clear();
                    fill_entry_name(&entry, &mut name_buf);
                    let size = entry.len().min(u32::MAX as u64) as u32;
                    visit(name_buf.as_str(), size);
                }
                Some(Err(e)) => {
                    log_fat_error("dir.iter", &e);
                    // Surface as Write-class error so callers see a single
                    // unified OpError; this isn't a write but it's an I/O
                    // fault in the FS layer.
                    return Err(OpError::Write);
                }
            }
        }
    }
    fs.unmount().await.map_err(|e| {
        log_fat_error("unmount", &e);
        OpError::Unmount
    })?;
    Ok(())
}

/// Render a directory entry's name into the supplied heapless string,
/// preferring the LFN if present. ASCII chars copied verbatim;
/// anything else becomes `?`. Caller-owned buffer so we don't pay a
/// stack-spill per entry.
fn fill_entry_name<IO, TP, OCC>(
    entry: &embedded_fatfs::DirEntry<'_, IO, TP, OCC>,
    out: &mut EntryName,
) where
    IO: embedded_fatfs::ReadWriteSeek,
    TP: embedded_fatfs::TimeProvider,
    OCC: embedded_fatfs::OemCpConverter,
{
    if let Some(units) = entry.long_file_name_as_ucs2_units() {
        for &u in units {
            // 0x0000 = NUL (post-name), 0xFFFF = padding past the
            // active LFN length boundary.
            if u == 0 || u == 0xFFFF {
                break;
            }
            let c = if u < 0x80 { u as u8 as char } else { '?' };
            if out.push(c).is_err() {
                break;
            }
        }
        if !out.is_empty() {
            return;
        }
    }
    // Fall back to the 11-byte short name (8 + 3, space-padded).
    let raw = entry.short_file_name_as_bytes();
    let (base, ext) = raw.split_at(raw.len().min(8));
    for &b in base {
        if b == b' ' || b == 0 {
            continue;
        }
        let c = if b < 0x80 { b as char } else { '?' };
        if out.push(c).is_err() {
            return;
        }
    }
    if ext.iter().any(|&b| b != b' ' && b != 0) {
        let _ = out.push('.');
        for &b in ext {
            if b == b' ' || b == 0 {
                continue;
            }
            let c = if b < 0x80 { b as char } else { '?' };
            if out.push(c).is_err() {
                return;
            }
        }
    }
}

// Compile-time assertion that the embedded-fatfs FileSystem type we
// instantiate above has the time provider / OEM converter defaults
// the `DirEntry` helper expects. Keeps the seemingly-unused imports
// honest if the upstream defaults ever change.
const _: () = {
    fn _assert<TP: embedded_fatfs::TimeProvider, OCC: embedded_fatfs::OemCpConverter>() {}
    let _ = _assert::<DefaultTimeProvider, LossyOemCpConverter>;
};

fn log_fat_error<E: core::fmt::Debug>(stage: &'static str, _err: &FatError<E>) {
    // Same trade as `open_partition`: emitting the FatError variant
    // via Debug2Format pulls a multi-KB Debug impl into the binary
    // for every IO/FS error case. The stage tag (mount / create_file
    // / flush / etc) plus the OpError variant returned to the caller
    // is sufficient operationally.
    defmt::warn!("blackbox/fat: {}: error", stage);
}

// Keep these symbols in scope so a future blackbox MCAP path that
// wants to peek at the underlying error variants can reuse them.
const _: () = {
    let _ = core::mem::size_of::<BufStreamError<sdmmc::Error>>();
};
