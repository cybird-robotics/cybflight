use core::fmt::Write;

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
