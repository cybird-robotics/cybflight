//! `BlockStore` adapter for the STM32 SDMMC peripheral.
//!
//! This is the **only** place in the firmware that touches
//! `embassy-stm32::sdmmc`. Generic recorder logic in `touch_sd::handle`
//! sees the SD card exclusively through the `BlockStore` trait.
//!
//! Re-architecting to a different backend (SPI NOR, FRAM, MRAM, ...)
//! means adding a sibling adapter file here — nothing in `touch_sd/mod.rs`
//! changes.

use core::cell::Cell;

use aligned::{Aligned, A4};
use block_device_driver::BlockDevice;
use cybflight_drivers::blackbox_storage::{BlockStore, BlockStoreError, BLOCK_SIZE};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::blocking_mutex::Mutex;
use static_cell::ConstStaticCell;

use crate::hal::sdmmc::sd::{Card, CmdBlock, DataBlock, StorageDevice};
use crate::hal::sdmmc::Sdmmc;
use crate::hal::time::Hertz;

/// SDMMC kernel-clock target. 25 MHz keeps margins on cheap microSDs;
/// matches Betaflight's default.
pub const SDMMC_FREQ: Hertz = Hertz(25_000_000);

/// Adapter that owns the `Sdmmc` peripheral plus a long-lived
/// `CmdBlock` (the DMA scratch buffer the HAL needs for ACMD41).
///
/// Each `BlockStore` call opens **one** `StorageDevice` scope that
/// covers init + the actual op + drop. `embassy-stm32`'s API binds
/// `StorageDevice<'_, '_, Card>` to a `&mut Sdmmc<'_>` borrow, so it
/// can't be parked in a struct field without self-referential
/// gymnastics. Coalescing init and op into one scope avoids two
/// CMD0/ACMD41 cycles per request and keeps the card in a known
/// "just-acquired" state for the data transfer — cheap microSDs are
/// finicky about back-to-back full re-acquisitions.
///
/// `capacity_blocks` reports the most recent successful acquisition's
/// CSD-derived capacity. Returns 0 before the first successful op.
pub struct SdmmcBlockStore {
    sdmmc: Sdmmc<'static>,
    cmd_block: CmdBlock,
    capacity: u32,
    /// CMD25 write-combining batch, taken once from [`BATCH_BUF`].
    batch: &'static mut [Block; BATCH_BLOCKS],
}

/// One 512 B block with the word alignment the SDMMC IDMA requires.
pub type Block = Aligned<A4, [u8; BLOCK_SIZE]>;

/// Write-combining depth: 64 × 512 B = 32 KB — exactly one cluster on
/// the recommended card format (`mkfs.fat -F 32 -s 64`; see
/// docs/blackbox.md §0). `embedded-fatfs` touches the FAT table (a
/// non-sequential write) once per cluster allocation, which ends a
/// batch, so a batch can never usefully exceed one cluster; 64 KB
/// clusters (the only thing a deeper buffer would serve) fail to
/// mount on ≤ 4 GB cards (FAT16 cluster-count trap) and sit past the
/// FAT spec's 32 KB compatibility ceiling anyway.
///
/// **Do not raise this to 128.** ac76d9a did, adding 32 KB of `.bss`,
/// and every 8 kHz build then boot-looped (Booting-LED heartbeat only,
/// no USB) while 1 kHz builds were fine — bisected by flashing to that
/// single commit and confirmed fixed by reverting it. The linker still
/// reported ~145 KB of headroom, so the mechanism is not a plain RAM
/// shortfall; the 64 KB static shifted the whole `.bss` layout at the
/// ODR where the IMU rings (3 × 192 slots) and SPI/SDMMC DMA load are
/// largest. Root cause not yet isolated — treat any further growth of
/// large statics on 8 kHz builds as needing a bench boot test.
pub const BATCH_BLOCKS: usize = 64;

const ZERO_BLOCK: Block = Aligned([0u8; BLOCK_SIZE]);
/// Static home for the batch so it never transits a task stack (the
/// store is moved by value into `blackbox_task`).
static BATCH_BUF: ConstStaticCell<[Block; BATCH_BLOCKS]> =
    ConstStaticCell::new([ZERO_BLOCK; BATCH_BLOCKS]);

/// One acquired-card session: the HAL's `StorageDevice` behind the
/// CMD25 write combiner. Lend it to `BufStream` by `&mut` and call
/// [`WriteCombiner::flush`] once the filesystem is unmounted.
pub type SdSession<'a> = WriteCombiner<'a, StorageDevice<'a, 'static, Card>>;

impl SdmmcBlockStore {
    /// Panics if called twice — one store per boot (there is one
    /// SDMMC peripheral and one batch buffer).
    pub fn new(sdmmc: Sdmmc<'static>) -> Self {
        Self {
            sdmmc,
            cmd_block: CmdBlock::new(),
            capacity: 0,
            batch: BATCH_BUF.take(),
        }
    }

    /// Acquire the SD card and return a live `StorageDevice` for one
    /// session of multi-op work (e.g. mounting a FAT filesystem).
    /// Reuses the held `CmdBlock` so the FAT layer can pay the
    /// CMD0/ACMD41 cost exactly once per session, instead of once per
    /// block as the [`BlockStore`] impl does.
    ///
    /// The returned session borrows `&mut self`; release it (drop)
    /// before reusing the store. Writes are batched into CMD25
    /// multi-block transfers — the caller **must** `flush()` the
    /// session after its last write (i.e. after `fs.unmount()`), or
    /// the tail batch never reaches the card.
    pub async fn open_session(&mut self) -> Result<SdSession<'_>, BlockStoreError> {
        let Self {
            sdmmc,
            cmd_block,
            capacity,
            batch,
        } = self;
        let storage = StorageDevice::new_sd_card(sdmmc, cmd_block, SDMMC_FREQ)
            .await
            .map_err(|_| {
                // Plain message — Debug2Format on the SDMMC `Error`
                // enum pulls in formatting code for every variant
                // and adds non-trivial bytes per call site. Card not
                // present / unseated / locked all surface as
                // `BlockStoreError::Io` upstream.
                defmt::warn!("sdmmc: open_session failed");
                BlockStoreError::Io
            })?;
        let cap = storage.card().csd.block_count();
        *capacity = cap.min(u32::MAX as u64) as u32;
        Ok(WriteCombiner::new(storage, &mut batch[..]))
    }
}

// ─── CMD25 write combiner ───────────────────────────────────────────────

/// Coalesces sequential single-block writes into one multi-block
/// transfer (CMD25) so the card programs many pages per command
/// instead of paying command + busy-wait + program latency per 512 B
/// — the single-block CMD24 path measured ~2.0–2.3 MB/s peak on
/// commodity microSDs, well under what sequential CMD25 achieves.
///
/// Sits *below* `BufStream` (which only ever hands out one block at a
/// time, so the HAL's CMD25 path was unreachable) and *above* the
/// HAL's `StorageDevice` (whose `BlockDevice` impl already routes
/// multi-block slices to `write_blocks`).
///
/// Batch boundaries:
/// - a write that is not the next sequential block flushes the batch
///   first (FAT-table / directory-entry writes do this naturally),
/// - a full buffer flushes,
/// - a read overlapping the pending range flushes first (read-after-
///   write consistency); non-overlapping reads pass straight through
///   so `BufStream`'s cache reloads and FAT-sector reads don't break
///   batching,
/// - [`flush`](Self::flush) drains explicitly — required after
///   unmount because the `BlockDevice` trait has no flush hook.
///
/// Error semantics stay per-block: if a CMD25 batch fails, the same
/// blocks are retried one at a time with CMD24, so a fault surfaces
/// exactly as it did before — just possibly up to one batch (≤32 KB)
/// later than the write that produced the data.
pub struct WriteCombiner<'b, D> {
    dev: D,
    buf: &'b mut [Block],
    start: u32,
    len: usize,
    stats: IoStats,
}

/// What the card was actually asked to do during one session —
/// turns "is CMD25 working?" from inference into measurement. Read
/// via [`LAST_IO_STATS`] after the session's final flush.
#[derive(Clone, Copy, Default, defmt::Format)]
pub struct IoStats {
    /// Write commands issued (CMD25 batches + single CMD24s).
    pub writes: u32,
    /// Blocks written in total.
    pub blocks_written: u32,
    /// Largest batch handed to one write command.
    pub max_batch: u16,
    /// Read commands passed through to the card.
    pub reads: u32,
    /// Blocks read in total.
    pub blocks_read: u32,
    /// CMD25 batches that failed and were retried block-by-block.
    pub fallbacks: u32,
}

/// Stats of the most recent session, published by `fat::finish_session`.
pub static LAST_IO_STATS: Mutex<CriticalSectionRawMutex, Cell<IoStats>> =
    Mutex::new(Cell::new(IoStats {
        writes: 0,
        blocks_written: 0,
        max_batch: 0,
        reads: 0,
        blocks_read: 0,
        fallbacks: 0,
    }));

impl<'b, D: BlockDevice<BLOCK_SIZE, Align = A4>> WriteCombiner<'b, D> {
    fn new(dev: D, buf: &'b mut [Block]) -> Self {
        Self {
            dev,
            buf,
            start: 0,
            len: 0,
            stats: IoStats::default(),
        }
    }

    pub fn stats(&self) -> IoStats {
        self.stats
    }

    #[inline]
    fn overlaps(&self, addr: u32, n: usize) -> bool {
        self.len != 0 && addr < self.start.wrapping_add(self.len as u32) && addr.wrapping_add(n as u32) > self.start
    }

    /// Drain the pending batch to the card. Idempotent.
    pub async fn flush(&mut self) -> Result<(), D::Error> {
        let n = self.len;
        if n == 0 {
            return Ok(());
        }
        // Clear first: whatever happens below, never re-flush the same
        // bytes on a later call.
        self.len = 0;
        let start = self.start;
        self.stats.writes += 1;
        self.stats.blocks_written += n as u32;
        self.stats.max_batch = self.stats.max_batch.max(n as u16);
        match self.dev.write(start, &self.buf[..n]).await {
            Ok(()) => Ok(()),
            Err(_) if n > 1 => {
                self.stats.fallbacks += 1;
                // Degrade to the pre-CMD25 behaviour so partial-write
                // salvage semantics are unchanged: each block gets its
                // own CMD24; the first failing one reports.
                defmt::warn!(
                    "sdmmc: CMD25 batch @{} x{} failed, retrying per block",
                    start,
                    n
                );
                for i in 0..n {
                    self.dev.write(start + i as u32, &self.buf[i..i + 1]).await?;
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl<D: BlockDevice<BLOCK_SIZE, Align = A4>> BlockDevice<BLOCK_SIZE> for WriteCombiner<'_, D> {
    type Error = D::Error;
    type Align = A4;

    async fn read(&mut self, block_address: u32, data: &mut [Block]) -> Result<(), Self::Error> {
        if self.overlaps(block_address, data.len()) {
            self.flush().await?;
        }
        self.stats.reads += 1;
        self.stats.blocks_read += data.len() as u32;
        self.dev.read(block_address, data).await
    }

    async fn write(&mut self, block_address: u32, data: &[Block]) -> Result<(), Self::Error> {
        let mut addr = block_address;
        for block in data {
            let sequential = self.len != 0 && addr == self.start.wrapping_add(self.len as u32);
            if self.len != 0 && (!sequential || self.len == self.buf.len()) {
                self.flush().await?;
            }
            if self.len == 0 {
                self.start = addr;
            }
            self.buf[self.len] = Aligned(**block);
            self.len += 1;
            addr = addr.wrapping_add(1);
        }
        Ok(())
    }

    async fn size(&mut self) -> Result<u64, Self::Error> {
        self.dev.size().await
    }
}

/// Reinterpret a 512 B byte buffer as a word-aligned `DataBlock`. The
/// HAL requires word alignment for DMA; copying through a stack
/// `DataBlock` is the simplest way to obtain it from a caller-owned
/// `&[u8; 512]`.
fn datablock_from_bytes(buf: &[u8; BLOCK_SIZE]) -> DataBlock {
    let mut data = DataBlock::new();
    let dst: &mut [u8; BLOCK_SIZE] =
        unsafe { &mut *(data.0.as_mut_ptr() as *mut [u8; BLOCK_SIZE]) };
    dst.copy_from_slice(buf);
    data
}

fn datablock_to_bytes(data: &DataBlock, buf: &mut [u8; BLOCK_SIZE]) {
    let src: &[u8; BLOCK_SIZE] = unsafe { &*(data.0.as_ptr() as *const [u8; BLOCK_SIZE]) };
    buf.copy_from_slice(src);
}

impl BlockStore for SdmmcBlockStore {
    async fn read_block(
        &mut self,
        index: u32,
        buf: &mut [u8; BLOCK_SIZE],
    ) -> Result<(), BlockStoreError> {
        defmt::debug!("sdmmc: acquire for read sector={}", index);
        let mut storage =
            StorageDevice::new_sd_card(&mut self.sdmmc, &mut self.cmd_block, SDMMC_FREQ)
                .await
                .map_err(|_| {
                    defmt::warn!("sdmmc: acquire failed");
                    BlockStoreError::Io
                })?;
        let cap = storage.card().csd.block_count();
        self.capacity = cap.min(u32::MAX as u64) as u32;
        if index >= self.capacity {
            return Err(BlockStoreError::OutOfRange);
        }
        let mut data = DataBlock::new();
        storage.read_block(index, &mut data).await.map_err(|_| {
            defmt::warn!("sdmmc: read_block({}) failed", index);
            BlockStoreError::Io
        })?;
        datablock_to_bytes(&data, buf);
        Ok(())
    }

    async fn write_block(
        &mut self,
        index: u32,
        buf: &[u8; BLOCK_SIZE],
    ) -> Result<(), BlockStoreError> {
        defmt::debug!("sdmmc: acquire for write sector={}", index);
        let mut storage =
            StorageDevice::new_sd_card(&mut self.sdmmc, &mut self.cmd_block, SDMMC_FREQ)
                .await
                .map_err(|_| {
                    defmt::warn!("sdmmc: acquire failed");
                    BlockStoreError::Io
                })?;
        let cap = storage.card().csd.block_count();
        self.capacity = cap.min(u32::MAX as u64) as u32;
        defmt::info!(
            "sdmmc: card OK ({} sectors, {} MiB)",
            self.capacity,
            (self.capacity as u64 * BLOCK_SIZE as u64) / (1024 * 1024),
        );
        if index >= self.capacity {
            return Err(BlockStoreError::OutOfRange);
        }
        let data = datablock_from_bytes(buf);
        storage.write_block(index, &data).await.map_err(|_| {
            defmt::warn!("sdmmc: write_block({}) failed", index);
            BlockStoreError::Io
        })
    }

    fn capacity_blocks(&self) -> u32 {
        self.capacity
    }
}
