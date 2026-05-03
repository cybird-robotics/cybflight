//! `BlockStore` adapter for the STM32 SDMMC peripheral.
//!
//! This is the **only** place in the firmware that touches
//! `embassy-stm32::sdmmc`. Generic recorder logic in `touch_sd::handle`
//! sees the SD card exclusively through the `BlockStore` trait.
//!
//! Re-architecting to a different backend (SPI NOR, FRAM, MRAM, ...)
//! means adding a sibling adapter file here — nothing in `touch_sd/mod.rs`
//! changes.

use cybflight_drivers::blackbox_storage::{BlockStore, BlockStoreError, BLOCK_SIZE};

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
}

impl SdmmcBlockStore {
    pub fn new(sdmmc: Sdmmc<'static>) -> Self {
        Self {
            sdmmc,
            cmd_block: CmdBlock::new(),
            capacity: 0,
        }
    }

    /// Acquire the SD card and return a live `StorageDevice` for one
    /// session of multi-op work (e.g. mounting a FAT filesystem).
    /// Reuses the held `CmdBlock` so the FAT layer can pay the
    /// CMD0/ACMD41 cost exactly once per session, instead of once per
    /// block as the [`BlockStore`] impl does.
    ///
    /// The returned `StorageDevice` borrows `&mut self`; release it
    /// (drop, or `core::mem::drop`) before reusing the store.
    pub async fn open_session(
        &mut self,
    ) -> Result<StorageDevice<'_, 'static, Card>, BlockStoreError> {
        let storage = StorageDevice::new_sd_card(&mut self.sdmmc, &mut self.cmd_block, SDMMC_FREQ)
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
        self.capacity = cap.min(u32::MAX as u64) as u32;
        Ok(storage)
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
