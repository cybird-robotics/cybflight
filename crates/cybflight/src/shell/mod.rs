use core::fmt::Write;

use embassy_usb::{
    class::cdc_acm::CdcAcmClass,
    driver::{Driver, EndpointError},
};

pub mod format;

// ---------------------------------------------------------------------------
// WriteBuf — core::fmt::Write adapter for fixed byte slices
// ---------------------------------------------------------------------------
pub struct WriteBuf<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> WriteBuf<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf[..self.pos]
    }
}

impl Write for WriteBuf<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = &mut self.buf[self.pos..];
        if bytes.len() > remaining.len() {
            return Err(core::fmt::Error);
        }
        remaining[..bytes.len()].copy_from_slice(bytes);
        self.pos += bytes.len();
        Ok(())
    }
}

/// Write `data` to the CDC class in ≤64-byte packets.
///
/// A zero-length packet (ZLP) is appended when `data` is an exact multiple of
/// 64 bytes, signalling end-of-transfer to the USB host.
pub async fn write_all<'d>(
    class: &mut CdcAcmClass<'d, impl Driver<'d>>,
    data: &[u8],
) -> Result<(), EndpointError> {
    for chunk in data.chunks(64) {
        class.write_packet(chunk).await?;
    }
    if !data.is_empty() && data.len() % 64 == 0 {
        class.write_packet(&[]).await?;
    }
    Ok(())
}
