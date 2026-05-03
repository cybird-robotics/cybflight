//! Minimal MCAP record framer over `embedded_io_async::Write`.
//!
//! Implements the wire format from the [MCAP specification][spec].
//! Stage 1 of the blackbox plan only emits **Magic / Header /
//! DataEnd / Footer**, which together form a structurally valid
//! empty file; later stages add `Schema`, `Channel`, and `Message`
//! on top of the same primitives.
//!
//! [spec]: https://mcap.dev/spec
//!
//! ## File layout (any MCAP file)
//!
//! ```text
//!   magic                        ── 8 bytes,  \x89MCAP0\r\n
//!   Header record                ── op 0x01
//!   <data section: 0+ records>   ── Schemas / Channels / Messages / ...
//!   DataEnd record               ── op 0x0F
//!   <summary section: optional>  ── empty in our minimal case
//!   Footer record                ── op 0x02
//!   magic                        ── 8 bytes (same as the header magic)
//! ```
//!
//! Records are framed as `op (1B) | data_len (8B LE u64) | data`.
//!
//! Why hand-rolled: the canonical `mcap` crate is `std`-only. The
//! framing rules are short and stable, so a no_std implementation is
//! ~80 lines and stays under our control.

use embedded_io_async::Write;

/// `\x89MCAP0\r\n` — both leading and trailing.
pub const MAGIC: [u8; 8] = [0x89, b'M', b'C', b'A', b'P', b'0', b'\r', b'\n'];

/// Op-code constants from the MCAP spec. Only the ones we currently
/// emit are listed; add to this list as later stages need them.
pub mod op {
    pub const HEADER: u8 = 0x01;
    pub const FOOTER: u8 = 0x02;
    pub const SCHEMA: u8 = 0x03;
    pub const CHANNEL: u8 = 0x04;
    pub const MESSAGE: u8 = 0x05;
    pub const METADATA: u8 = 0x0C;
    pub const DATA_END: u8 = 0x0F;
}

/// Emit the 8-byte file magic. Used at both the start *and* end of
/// every MCAP file.
pub async fn write_magic<W: Write>(w: &mut W) -> Result<(), W::Error> {
    w.write_all(&MAGIC).await
}

/// Emit a record prelude: 1 byte op + 8 byte little-endian body length.
/// Caller is responsible for then writing exactly `body_len` more bytes.
async fn write_prelude<W: Write>(w: &mut W, op: u8, body_len: u64) -> Result<(), W::Error> {
    let mut head = [0u8; 9];
    head[0] = op;
    head[1..9].copy_from_slice(&body_len.to_le_bytes());
    w.write_all(&head).await
}

/// Emit a u32 length-prefixed UTF-8 string (MCAP "String").
async fn write_string<W: Write>(w: &mut W, s: &str) -> Result<(), W::Error> {
    w.write_all(&(s.len() as u32).to_le_bytes()).await?;
    w.write_all(s.as_bytes()).await
}

/// Emit the Header record (op 0x01).
///
/// `profile` is the MCAP "profile" string — leave empty unless you
/// follow a published profile (e.g. `"ros1"`). `library` identifies
/// the writer; we pass `cybflight v<version>`.
pub async fn write_header<W: Write>(
    w: &mut W,
    profile: &str,
    library: &str,
) -> Result<(), W::Error> {
    let body_len = 4 + profile.len() + 4 + library.len();
    write_prelude(w, op::HEADER, body_len as u64).await?;
    write_string(w, profile).await?;
    write_string(w, library).await
}

/// Emit the DataEnd record (op 0x0F). `data_section_crc` is
/// optional (set to 0 if not computed); we pass 0 in Stage 1.
pub async fn write_data_end<W: Write>(w: &mut W, data_section_crc: u32) -> Result<(), W::Error> {
    write_prelude(w, op::DATA_END, 4).await?;
    w.write_all(&data_section_crc.to_le_bytes()).await
}

/// Emit the Footer record (op 0x02). For files with no summary
/// section all three fields are zero (the convention for "no summary
/// present"). We always pass zeros in Stage 1; later stages with a
/// summary section will fill in `summary_start` etc.
pub async fn write_footer<W: Write>(
    w: &mut W,
    summary_start: u64,
    summary_offset_start: u64,
    summary_crc: u32,
) -> Result<(), W::Error> {
    write_prelude(w, op::FOOTER, 20).await?;
    let mut body = [0u8; 20];
    body[0..8].copy_from_slice(&summary_start.to_le_bytes());
    body[8..16].copy_from_slice(&summary_offset_start.to_le_bytes());
    body[16..20].copy_from_slice(&summary_crc.to_le_bytes());
    w.write_all(&body).await
}

/// Emit a Schema record (op 0x03). `id` must be > 0; `data` is the
/// raw schema document (e.g. JSON Schema bytes for `encoding =
/// "jsonschema"`).
pub async fn write_schema<W: Write>(
    w: &mut W,
    id: u16,
    name: &str,
    encoding: &str,
    data: &[u8],
) -> Result<(), W::Error> {
    let body_len = 2 + 4 + name.len() + 4 + encoding.len() + 4 + data.len();
    write_prelude(w, op::SCHEMA, body_len as u64).await?;
    w.write_all(&id.to_le_bytes()).await?;
    write_string(w, name).await?;
    write_string(w, encoding).await?;
    w.write_all(&(data.len() as u32).to_le_bytes()).await?;
    w.write_all(data).await
}

/// Emit a Channel record (op 0x04) with empty metadata. Set
/// `schema_id = 0` if there is no associated schema (Foxglove won't
/// be able to decode the messages, but the file is still valid).
///
/// `message_encoding` is a free-form identifier interpreted by the
/// reader: `"cbor"`, `"json"`, `"protobuf"`, etc.
pub async fn write_channel<W: Write>(
    w: &mut W,
    id: u16,
    schema_id: u16,
    topic: &str,
    message_encoding: &str,
) -> Result<(), W::Error> {
    // Metadata is `Map<string,string>` encoded as `uint32 length |
    // (string,string)*`. Empty map = `0u32`.
    let body_len = 2 + 2 + 4 + topic.len() + 4 + message_encoding.len() + 4;
    write_prelude(w, op::CHANNEL, body_len as u64).await?;
    w.write_all(&id.to_le_bytes()).await?;
    w.write_all(&schema_id.to_le_bytes()).await?;
    write_string(w, topic).await?;
    write_string(w, message_encoding).await?;
    w.write_all(&0u32.to_le_bytes()).await
}

/// Emit a Message record (op 0x05). `data` is the raw payload bytes
/// in whatever encoding was declared on the channel (CBOR, JSON, ...).
/// Times are nanoseconds (Foxglove will use any monotonic basis; for
/// us, ns since boot).
pub async fn write_message<W: Write>(
    w: &mut W,
    channel_id: u16,
    sequence: u32,
    log_time_ns: u64,
    publish_time_ns: u64,
    data: &[u8],
) -> Result<(), W::Error> {
    let body_len = 2 + 4 + 8 + 8 + data.len();
    write_prelude(w, op::MESSAGE, body_len as u64).await?;
    let mut head = [0u8; 22];
    head[0..2].copy_from_slice(&channel_id.to_le_bytes());
    head[2..6].copy_from_slice(&sequence.to_le_bytes());
    head[6..14].copy_from_slice(&log_time_ns.to_le_bytes());
    head[14..22].copy_from_slice(&publish_time_ns.to_le_bytes());
    w.write_all(&head).await?;
    w.write_all(data).await
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;

    /// Tiny `embedded_io_async::Write` impl for unit testing the
    /// framer on the host. Captures all bytes into a `Vec<u8>`.
    struct VecWriter(Vec<u8>);

    impl embedded_io_async::ErrorType for VecWriter {
        type Error = core::convert::Infallible;
    }

    impl embedded_io_async::Write for VecWriter {
        async fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }
        async fn flush(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn skeleton_byte_layout() {
        let mut w = VecWriter(Vec::new());
        embassy_futures::block_on(async {
            write_magic(&mut w).await.unwrap();
            write_header(&mut w, "", "cybflight").await.unwrap();
            write_data_end(&mut w, 0).await.unwrap();
            write_footer(&mut w, 0, 0, 0).await.unwrap();
            write_magic(&mut w).await.unwrap();
        });
        let buf = w.0;

        // Sanity: 8 magic + (1+8+17 header) + (1+8+4 data_end) + (1+8+20 footer) + 8 magic
        assert_eq!(buf.len(), 8 + 26 + 13 + 29 + 8);
        assert_eq!(&buf[..8], &MAGIC);
        assert_eq!(buf[8], op::HEADER);
        assert_eq!(&buf[buf.len() - 8..], &MAGIC);
    }
}
