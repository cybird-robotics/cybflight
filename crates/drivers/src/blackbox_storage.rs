//! Generic block-store abstraction for blackbox / flight-data-recorder
//! storage backends.
//!
//! This trait is the architectural seam between the board-agnostic
//! recorder code (which lives in `cybflight`) and the board-specific
//! storage peripheral (SDMMC today; SPI NOR / FRAM / MRAM tomorrow).
//!
//! Trait lives in `cybflight-drivers` so it stays free of any
//! `embassy-stm32` dependency. The concrete adapter that wraps the
//! STM32 SDMMC peripheral lives in `cybflight` itself, where binding
//! to `embassy-stm32` is allowed.

#[derive(Debug, Clone, Copy, PartialEq, Eq, defmt::Format)]
pub enum BlockStoreError {
    /// Block index is past the device's reported capacity.
    OutOfRange,
    /// Underlying I/O failure (timeout, CRC, card pulled, ...).
    Io,
    /// No media present, or the device has not yet been initialized.
    NotReady,
}

/// Sector size used by every backend. Matches SDHC/SDXC and the
/// dominant erase-block granularity of SPI NOR parts.
pub const BLOCK_SIZE: usize = 512;

/// Async block-aligned read/write store.
///
/// All implementors:
/// - operate on 512-byte sectors;
/// - report capacity in sectors via [`capacity_blocks`];
///   `0` means "no media present / not initialized";
/// - return [`BlockStoreError::OutOfRange`] for indices ≥ capacity.
///
/// `#[allow(async_fn_in_trait)]` matches the project's other driver
/// traits (`ReadImu`, `ReadBaro`, ...). Implementors are monomorphized
/// at every use site so there is no `dyn`-dispatch concern.
#[allow(async_fn_in_trait)]
pub trait BlockStore {
    /// Read one 512 B sector into `buf`.
    async fn read_block(
        &mut self,
        index: u32,
        buf: &mut [u8; BLOCK_SIZE],
    ) -> Result<(), BlockStoreError>;

    /// Write one 512 B sector from `buf`.
    async fn write_block(
        &mut self,
        index: u32,
        buf: &[u8; BLOCK_SIZE],
    ) -> Result<(), BlockStoreError>;

    /// Total addressable sectors. `0` means no media / not initialized.
    fn capacity_blocks(&self) -> u32;
}