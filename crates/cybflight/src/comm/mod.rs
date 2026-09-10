//! ESP bridge communication — COBS framing utilities.

pub mod esp_bridge;
pub mod health_wire;
pub mod time_sync;

use cybflight_msgs::wire;

/// Maximum COBS-encoded frame size including delimiter.
const MAX_RAW_FRAME: usize = wire::FRAME_HEADER_SIZE + wire::MAX_WIRE_PAYLOAD;
const MAX_COBS_BUF: usize = cobs::max_encoding_length(MAX_RAW_FRAME) + 1; // +1 for 0x00 delimiter

/// Accumulates incoming UART bytes and yields COBS-decoded frames on 0x00 delimiter.
pub struct FrameAccumulator {
    buf: [u8; MAX_COBS_BUF],
    pos: usize,
}

impl FrameAccumulator {
    pub fn new() -> Self {
        Self {
            buf: [0; MAX_COBS_BUF],
            pos: 0,
        }
    }

    /// Feed a single byte. Returns `Some(decoded_len)` when a complete frame
    /// is received and successfully COBS-decoded. The decoded data is available
    /// in `self.data()[..decoded_len]` until the next `feed()` call.
    pub fn feed(&mut self, byte: u8) -> Option<usize> {
        if byte == 0x00 {
            if self.pos == 0 {
                return None; // empty frame
            }
            let pos = self.pos;
            self.pos = 0;
            cobs::decode_in_place(&mut self.buf[..pos]).ok()
        } else if self.pos < self.buf.len() {
            self.buf[self.pos] = byte;
            self.pos += 1;
            None
        } else {
            // Buffer overflow — discard frame.
            self.pos = 0;
            None
        }
    }

    /// Access the internal buffer (valid after a successful `feed()` return).
    pub fn data(&self) -> &[u8] {
        &self.buf
    }
}

/// Build a raw frame `[msg_id, seq, payload...]`, COBS-encode it into
/// `dest`, and append the 0x00 delimiter.
///
/// Returns the number of bytes written, or `0` if the frame does not
/// fit. A caller batches many frames into one buffer and cannot know in
/// advance which one will overrun it. Both the COBS encode and the
/// delimiter write used to index `dest` unchecked, so an oversubscribed
/// batch buffer panicked in the middle of a telemetry tick rather than
/// dropping a frame — the one outcome a downlink should never have.
/// Refusing up front lets the caller drop and count it instead.
pub fn encode_frame(msg_id: u8, seq: u8, payload: &[u8], dest: &mut [u8]) -> usize {
    let raw_len = wire::FRAME_HEADER_SIZE + payload.len();
    // The scratch frame is fixed-size, and the destination must hold the
    // encoding plus its delimiter. COBS output length depends on where
    // the zero bytes fall, so this tests the worst case: conservative by
    // a byte or two per 254, which costs nothing here and keeps the
    // check independent of the payload's content.
    if raw_len > MAX_RAW_FRAME || cobs::max_encoding_length(raw_len) + 1 > dest.len() {
        return 0;
    }
    let mut raw = [0u8; MAX_RAW_FRAME];
    raw[0] = msg_id;
    raw[1] = seq;
    raw[wire::FRAME_HEADER_SIZE..raw_len].copy_from_slice(payload);

    let encoded_len = cobs::encode(&raw[..raw_len], &mut dest[..]);
    dest[encoded_len] = 0x00;
    encoded_len + 1
}
