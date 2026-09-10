//! Minimal CBOR encoder for blackbox message payloads.
//!
//! Encodes definite-length maps / arrays / primitives into a
//! caller-provided byte slice. No allocator, no recursion, no
//! intermediate buffers — the writer just appends bytes.
//!
//! Lives in `cybflight-core` (rather than the firmware crate) so the
//! encoder and the wire formats built on it ([`crate::blackbox_wire`])
//! are host-testable: the firmware crate only compiles for thumbv7em,
//! so `#[cfg(test)]` modules there never run.
//!
//! ## Why hand-rolled
//!
//! `serde_cbor` and `ciborium` need an allocator. `minicbor` is no_std
//! but adds a derive crate and trait machinery we don't need yet. Our
//! payloads are flat maps (`{topic_field: scalar, ...}`) or flat
//! positional arrays, so the entire encoder is ~80 lines.
//!
//! ## Wire format crash course
//!
//! Every CBOR item starts with one byte: `(major_type << 5) | info`.
//! `info` 0–23 is inline; 24/25/26/27 mean "1/2/4/8 follow-up bytes".
//! Multi-byte fields are **big-endian** (note: MCAP record framing is
//! little-endian — easy to mix up).
//!
//! Major types we support:
//! - 0: unsigned int
//! - 1: negative int (encoded as `-1 - n`)
//! - 2: byte string
//! - 3: text string (UTF-8)
//! - 4: array (definite length)
//! - 5: map (definite length, key/value alternating)
//! - 7: float / simple — `0xfa` = single, `0xfb` = double
//!
//! For map(n) / array(n), the caller is responsible for then writing
//! exactly `n` items / `2n` items.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfSpace;

pub type Result<T = ()> = core::result::Result<T, OutOfSpace>;

pub struct CborWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> CborWriter<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes appended so far. Pass `&buf[..writer.pos()]` to the MCAP
    /// `write_message` call.
    pub fn pos(&self) -> usize {
        self.pos
    }

    fn put(&mut self, b: u8) -> Result {
        if self.pos >= self.buf.len() {
            return Err(OutOfSpace);
        }
        self.buf[self.pos] = b;
        self.pos += 1;
        Ok(())
    }

    fn put_slice(&mut self, src: &[u8]) -> Result {
        if self.pos + src.len() > self.buf.len() {
            return Err(OutOfSpace);
        }
        self.buf[self.pos..self.pos + src.len()].copy_from_slice(src);
        self.pos += src.len();
        Ok(())
    }

    /// Emit a CBOR head: 1-byte type+info or type byte plus 1/2/4/8
    /// big-endian follow-up bytes. Used by every other method.
    fn put_head(&mut self, major: u8, val: u64) -> Result {
        let mt = major << 5;
        if val < 24 {
            self.put(mt | val as u8)
        } else if val <= u8::MAX as u64 {
            self.put(mt | 24)?;
            self.put(val as u8)
        } else if val <= u16::MAX as u64 {
            self.put(mt | 25)?;
            self.put_slice(&(val as u16).to_be_bytes())
        } else if val <= u32::MAX as u64 {
            self.put(mt | 26)?;
            self.put_slice(&(val as u32).to_be_bytes())
        } else {
            self.put(mt | 27)?;
            self.put_slice(&val.to_be_bytes())
        }
    }

    pub fn u64(&mut self, val: u64) -> Result {
        self.put_head(0, val)
    }

    pub fn i64(&mut self, val: i64) -> Result {
        if val >= 0 {
            self.put_head(0, val as u64)
        } else {
            // CBOR negatives encode as `-1 - n`.
            self.put_head(1, (-(val + 1)) as u64)
        }
    }

    pub fn f32(&mut self, val: f32) -> Result {
        self.put(0xfa)?; // major 7, info 26 = single-precision
        self.put_slice(&val.to_be_bytes())
    }

    pub fn f64(&mut self, val: f64) -> Result {
        self.put(0xfb)?; // major 7, info 27 = double-precision
        self.put_slice(&val.to_be_bytes())
    }

    pub fn str(&mut self, s: &str) -> Result {
        self.put_head(3, s.len() as u64)?;
        self.put_slice(s.as_bytes())
    }

    pub fn bytes(&mut self, b: &[u8]) -> Result {
        self.put_head(2, b.len() as u64)?;
        self.put_slice(b)
    }

    /// Begin a definite-length array. Caller writes exactly `n` items
    /// next.
    pub fn array(&mut self, n: u64) -> Result {
        self.put_head(4, n)
    }

    /// Begin a definite-length map. Caller writes exactly `2n` values
    /// next (alternating key, value).
    pub fn map(&mut self, n: u64) -> Result {
        self.put_head(5, n)
    }

    pub fn bool(&mut self, val: bool) -> Result {
        // major 7, simple values: false=20, true=21
        self.put(if val { 0xf5 } else { 0xf4 })
    }

    /// CBOR `null` (major 7, simple value 22). Used for optional
    /// fields whose absence carries different meaning than a sentinel
    /// (e.g. a missing motor-telemetry frame vs a true zero RPM).
    pub fn null(&mut self) -> Result {
        self.put(0xf6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(f: impl FnOnce(&mut CborWriter) -> Result) -> std::vec::Vec<u8> {
        let mut buf = [0u8; 64];
        let mut w = CborWriter::new(&mut buf);
        f(&mut w).unwrap();
        let n = w.pos();
        buf[..n].to_vec()
    }

    #[test]
    fn head_boundaries() {
        // RFC 8949 boundary values for the head encoding.
        assert_eq!(enc(|w| w.u64(0)), [0x00]);
        assert_eq!(enc(|w| w.u64(23)), [0x17]);
        assert_eq!(enc(|w| w.u64(24)), [0x18, 24]);
        assert_eq!(enc(|w| w.u64(255)), [0x18, 255]);
        assert_eq!(enc(|w| w.u64(256)), [0x19, 0x01, 0x00]);
        assert_eq!(enc(|w| w.u64(65536)), [0x1a, 0, 1, 0, 0]);
        assert_eq!(
            enc(|w| w.u64(u64::MAX)),
            [0x1b, 255, 255, 255, 255, 255, 255, 255, 255]
        );
    }

    #[test]
    fn negatives() {
        // -1 encodes as major 1, value 0; -500 as major 1, value 499.
        assert_eq!(enc(|w| w.i64(-1)), [0x20]);
        assert_eq!(enc(|w| w.i64(-500)), [0x39, 0x01, 0xf3]);
        assert_eq!(enc(|w| w.i64(42)), [0x18, 42]);
    }

    #[test]
    fn floats_and_simples() {
        // 1.5f32 = 0x3FC00000 big-endian after the 0xfa marker.
        assert_eq!(enc(|w| w.f32(1.5)), [0xfa, 0x3f, 0xc0, 0x00, 0x00]);
        assert_eq!(enc(|w| w.bool(true)), [0xf5]);
        assert_eq!(enc(|w| w.bool(false)), [0xf4]);
        assert_eq!(enc(|w| w.null()), [0xf6]);
    }

    #[test]
    fn strings_arrays_maps() {
        assert_eq!(enc(|w| w.str("ab")), [0x62, b'a', b'b']);
        assert_eq!(enc(|w| w.array(3)), [0x83]);
        assert_eq!(enc(|w| w.map(2)), [0xa2]);
    }

    #[test]
    fn out_of_space_is_reported() {
        let mut buf = [0u8; 4];
        let mut w = CborWriter::new(&mut buf);
        // 5 bytes into a 4-byte buffer must fail. The writer is
        // poisoned after an error (the marker byte may have landed);
        // callers discard the whole record on Err.
        assert_eq!(w.f32(1.0), Err(OutOfSpace));
    }
}
