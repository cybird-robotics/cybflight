//! Minimal CBOR encoder for blackbox message payloads.
//!
//! Encodes definite-length maps / arrays / primitives into a
//! caller-provided byte slice. No allocator, no recursion, no
//! intermediate buffers — the writer just appends bytes.
//!
//! ## Why hand-rolled
//!
//! `serde_cbor` and `ciborium` need an allocator. `minicbor` is no_std
//! but adds a derive crate and trait machinery we don't need yet. Our
//! payloads are flat maps (`{topic_field: scalar, ...}`) so the entire
//! encoder is ~80 lines.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, defmt::Format)]
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
}
