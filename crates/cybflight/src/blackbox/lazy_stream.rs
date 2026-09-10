//! `LazyBufStream` — a write-optimised replacement for
//! `block_device_adapters::BufStream`.
//!
//! `BufStream` is a read-modify-write cache: every time the byte
//! cursor enters a new block it **reads that block from the card**
//! before letting you write into it — even when the write goes on to
//! overwrite all 512 bytes, which is what a streaming recorder does
//! for every data block. Measured on a bench log that read costs
//! ~1 ms of CMD17 per 512 B, i.e. most of the recorder's wall clock,
//! and it is exactly the per-block round trip the CMD25 write
//! combiner below us was meant to remove.
//!
//! This adapter keeps `BufStream`'s exact API and semantics (same
//! fast paths, same error type, same seek model) but loads a block
//! **lazily**: entering a block costs nothing; writes that start at
//! the block's first byte (or extend a prefix already written this
//! visit) go straight into the cache, and the card is only read when
//! something genuinely needs the old contents — a read outside the
//! written prefix, a write that leaves a hole, or a flush of a
//! partially written block (the untouched tail must be preserved:
//! directory / FAT sectors are written through this same stream).
//!
//! Net effect for the recorder's sequential-append pattern: zero
//! reads per data block; the only reads left are FAT-table and
//! directory sectors, plus one merge read per partial flush (the 1 s
//! `FLUSH_INTERVAL` commits and session close).

use aligned::Aligned;
use block_device_adapters::BufStreamError;
use block_device_driver::{slice_to_blocks, slice_to_blocks_mut, BlockDevice};
use embedded_io_async::{Read, Seek, SeekFrom, Write};

pub struct LazyBufStream<T: BlockDevice<SIZE>, const SIZE: usize> {
    inner: T,
    buffer: Aligned<T::Align, [u8; SIZE]>,
    current_block: u32,
    current_offset: u64,
    /// Cache differs from the card and must be written back.
    dirty: bool,
    /// Cache holds the card's contents for `current_block` (after a
    /// load or a full-block write). When false, only `[0, filled)`
    /// is valid.
    loaded: bool,
    /// Length of the valid prefix of an unloaded block, built by
    /// sequential writes since the block was entered.
    filled: usize,
}

impl<T: BlockDevice<SIZE>, const SIZE: usize> LazyBufStream<T, SIZE> {
    const ALIGN: usize = core::mem::align_of::<Aligned<T::Align, [u8; SIZE]>>();

    pub fn new(inner: T) -> Self {
        Self {
            inner,
            buffer: Aligned([0; SIZE]),
            current_block: u32::MAX,
            current_offset: 0,
            dirty: false,
            loaded: false,
            filled: 0,
        }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }

    #[inline]
    fn pointer_block_start(&self) -> u32 {
        (self.current_offset / SIZE as u64)
            .try_into()
            .expect("Block larger than 2TB")
    }

    #[inline]
    fn pointer_block_start_addr(&self) -> u64 {
        self.pointer_block_start() as u64 * SIZE as u64
    }

    /// Make the whole cache valid for `current_block`, reading the
    /// card and merging under any prefix written since entry.
    async fn ensure_loaded(&mut self) -> Result<(), T::Error> {
        if self.loaded {
            return Ok(());
        }
        if self.filled == 0 {
            let buf = &mut self.buffer[..];
            self.inner
                .read(self.current_block, slice_to_blocks_mut(buf))
                .await?;
        } else {
            let mut tmp: Aligned<T::Align, [u8; SIZE]> = Aligned([0; SIZE]);
            self.inner
                .read(self.current_block, slice_to_blocks_mut(&mut tmp[..]))
                .await?;
            let f = self.filled;
            self.buffer[f..].copy_from_slice(&tmp[f..]);
        }
        self.loaded = true;
        Ok(())
    }

    /// Write back the cache if dirty. A partially written unloaded
    /// block is merged with the card first so its untouched tail
    /// survives.
    async fn flush_cache(&mut self) -> Result<(), T::Error> {
        if !self.dirty {
            return Ok(());
        }
        if !self.loaded && self.filled < SIZE {
            self.ensure_loaded().await?;
        }
        self.dirty = false;
        self.loaded = true;
        self.inner
            .write(self.current_block, slice_to_blocks(&self.buffer[..]))
            .await
    }

    /// Point the cache at the cursor's block without reading it.
    async fn select_block(&mut self) -> Result<(), T::Error> {
        let block = self.pointer_block_start();
        if block != self.current_block {
            self.flush_cache().await?;
            self.current_block = block;
            self.loaded = false;
            self.dirty = false;
            self.filled = 0;
        }
        Ok(())
    }
}

impl<T: BlockDevice<SIZE>, const SIZE: usize> embedded_io_async::ErrorType
    for LazyBufStream<T, SIZE>
{
    type Error = BufStreamError<T::Error>;
}

impl<T: BlockDevice<SIZE>, const SIZE: usize> Read for LazyBufStream<T, SIZE> {
    async fn read(&mut self, mut buf: &mut [u8]) -> Result<usize, Self::Error> {
        let mut total = 0;
        let target = buf.len();
        loop {
            let bytes_read = if buf.len() % SIZE == 0
                && buf.as_ptr().cast::<u8>() as usize % Self::ALIGN == 0
                && self.current_offset % SIZE as u64 == 0
            {
                // Direct multi-block read (same fast path as BufStream).
                let block = self.pointer_block_start();
                self.inner.read(block, slice_to_blocks_mut(buf)).await?;
                buf.len()
            } else {
                let block_start = self.pointer_block_start_addr();
                self.select_block().await?;
                let buffer_offset = (self.current_offset - block_start) as usize;
                let end = core::cmp::min(buffer_offset + buf.len(), SIZE);
                if !(self.loaded || end <= self.filled) {
                    self.ensure_loaded().await?;
                }
                let n = end - buffer_offset;
                buf[..n].copy_from_slice(&self.buffer[buffer_offset..end]);
                buf = &mut buf[n..];
                n
            };
            self.current_offset += bytes_read as u64;
            total += bytes_read;
            if total == target {
                return Ok(total);
            }
        }
    }
}

impl<T: BlockDevice<SIZE>, const SIZE: usize> Write for LazyBufStream<T, SIZE> {
    async fn write(&mut self, mut buf: &[u8]) -> Result<usize, Self::Error> {
        let mut total = 0;
        let target = buf.len();
        loop {
            let bytes_written = if buf.len() % SIZE == 0
                && buf.as_ptr().cast::<u8>() as usize % Self::ALIGN == 0
                && self.current_offset % SIZE as u64 == 0
            {
                // Direct multi-block write (same fast path as BufStream).
                let block = self.pointer_block_start();
                self.inner.write(block, slice_to_blocks(buf)).await?;
                buf.len()
            } else {
                let block_start = self.pointer_block_start_addr();
                self.select_block().await?;
                let buffer_offset = (self.current_offset - block_start) as usize;
                let end = core::cmp::min(buffer_offset + buf.len(), SIZE);
                // A write touching or inside the valid prefix needs no
                // load — this is the sequential-append case. Anything
                // that would leave a hole needs the card's bytes first.
                if !self.loaded && buffer_offset > self.filled {
                    self.ensure_loaded().await?;
                }
                let n = end - buffer_offset;
                self.buffer[buffer_offset..end].copy_from_slice(&buf[..n]);
                buf = &buf[n..];
                if !self.loaded {
                    self.filled = self.filled.max(end);
                    if self.filled == SIZE {
                        self.loaded = true;
                    }
                }
                self.dirty = true;
                if end == SIZE {
                    self.flush_cache().await?;
                }
                n
            };
            self.current_offset += bytes_written as u64;
            total += bytes_written;
            if total == target {
                return Ok(total);
            }
        }
    }

    async fn flush(&mut self) -> Result<(), Self::Error> {
        self.flush_cache().await?;
        Ok(())
    }
}

impl<T: BlockDevice<SIZE>, const SIZE: usize> Seek for LazyBufStream<T, SIZE> {
    async fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        self.current_offset = match pos {
            SeekFrom::Start(x) => x,
            SeekFrom::End(x) => (self.inner.size().await? as i64 - x) as u64,
            SeekFrom::Current(x) => (self.current_offset as i64 + x) as u64,
        };
        Ok(self.current_offset)
    }
}
