//! GCR (Grey Coded Response) telemetry decode from edge timings.
//!
//! Implements `decodeTelemetryPacket` (BF `pwm_output_dshot_shared.c:164-208`).

/// GCR 5-to-4 decode lookup table.
///
/// Invalid entries are 0xFF (BF uses 0, but we use 0xFF for explicit error detection).
const GCR_DECODE: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 9,    10,   11,   0xFF, 13,   14,   15,
    0xFF, 0xFF, 2,    3,    0xFF, 5,    6,    7,
    0xFF, 0,    8,    1,    0xFF, 4,    12,   0xFF,
];

/// Decode a GCR telemetry packet from edge timings.
///
/// `edge_timings` contains timer capture values for each edge transition.
/// `count` is the number of valid entries in `edge_timings`.
/// `ticks_per_bit` is the timer tick count for one GCR bit period (e.g. 16).
///
/// Returns the decoded 12-bit telemetry value, or `None` if decoding fails.
pub fn decode_telemetry_packet(
    edge_timings: &[u32],
    count: usize,
    ticks_per_bit: u32,
) -> Option<u16> {
    if count < 2 {
        return None;
    }

    let half = ticks_per_bit / 2;
    let mut value: u32 = 0;
    let mut bits: u32 = 0;
    let mut old_value = edge_timings[0];

    for i in 1..=count.min(edge_timings.len()) {
        let len = if i < count && i < edge_timings.len() {
            let diff = edge_timings[i].wrapping_sub(old_value);
            if bits >= 21 {
                break;
            }
            (diff + half) / ticks_per_bit
        } else {
            // Pad final gap to reach exactly 21 bits
            21 - bits
        };

        if len == 0 || len > 21 {
            return None;
        }

        value <<= len;
        value |= 1 << (len - 1);
        if i < edge_timings.len() {
            old_value = edge_timings[i];
        }
        bits += len;
    }

    if bits != 21 {
        return None;
    }

    // Decode 4 × 5-bit GCR nibbles from the lower 20 bits
    let n0 = GCR_DECODE[(value & 0x1F) as usize];
    let n1 = GCR_DECODE[((value >> 5) & 0x1F) as usize];
    let n2 = GCR_DECODE[((value >> 10) & 0x1F) as usize];
    let n3 = GCR_DECODE[((value >> 15) & 0x1F) as usize];

    // Check for invalid GCR nibbles
    if n0 == 0xFF || n1 == 0xFF || n2 == 0xFF || n3 == 0xFF {
        return None;
    }

    let decoded: u32 = (n0 as u32)
        | ((n1 as u32) << 4)
        | ((n2 as u32) << 8)
        | ((n3 as u32) << 12);

    // Checksum: XOR of all nibbles must equal 0xF
    let csum = decoded ^ (decoded >> 8);
    let csum = csum ^ (csum >> 4);
    if (csum & 0xF) != 0xF {
        return None;
    }

    // Return upper 12 bits (strip 4-bit checksum)
    Some((decoded >> 4) as u16)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::*;

    /// GCR 4-to-5 encode table (from AP bdshot_encoder.py:58).
    const GCR_ENCODE: [u8; 16] = [
        0x19, 0x1B, 0x12, 0x13, 0x1D, 0x15, 0x16, 0x17,
        0x1A, 0x09, 0x0A, 0x0B, 0x1E, 0x0D, 0x0E, 0x0F,
    ];

    /// Test-only: GCR-encode a 12-bit value into edge timings.
    fn gcr_encode(value12: u16, ticks_per_bit: u32) -> Vec<u32> {
        // Build 16-bit value with checksum (nibble XOR = 0xF)
        let n2 = ((value12 >> 8) & 0xF) as u8;
        let n1 = ((value12 >> 4) & 0xF) as u8;
        let n0 = (value12 & 0xF) as u8;
        let csum = (n2 ^ n1 ^ n0 ^ 0xF) & 0xF;
        let encoded16: u16 = (value12 << 4) | csum as u16;

        // GCR encode 4 nibbles (LSB first) into 20 GCR bits, then prepend start bit
        let gcr_n0 = GCR_ENCODE[(encoded16 & 0xF) as usize];
        let gcr_n1 = GCR_ENCODE[((encoded16 >> 4) & 0xF) as usize];
        let gcr_n2 = GCR_ENCODE[((encoded16 >> 8) & 0xF) as usize];
        let gcr_n3 = GCR_ENCODE[((encoded16 >> 12) & 0xF) as usize];

        let gcr20: u32 = (gcr_n0 as u32)
            | ((gcr_n1 as u32) << 5)
            | ((gcr_n2 as u32) << 10)
            | ((gcr_n3 as u32) << 15);

        // Prepend start bit (bit 20)
        let gcr21 = gcr20 | (1 << 20);

        // Convert 21-bit value to edge timings.
        // Each '1' bit in the value (MSB first) represents an edge.
        let mut edges = Vec::new();
        for msb_pos in 0..21u32 {
            let bit_pos = 20 - msb_pos;
            if (gcr21 >> bit_pos) & 1 == 1 {
                edges.push(msb_pos * ticks_per_bit);
            }
        }

        edges
    }

    #[test]
    fn decode_known_vector_0x000() {
        // V=0x000: Encoded16=0x000F, checksum=0xF
        let edges: &[u32] = &[0, 16, 32, 80, 96, 112, 160, 176, 192, 240, 272, 288, 304, 320];
        let result = decode_telemetry_packet(edges, edges.len(), 16);
        assert_eq!(result, Some(0x000));
    }

    #[test]
    fn decode_known_vector_0xfff() {
        // V=0xFFF: Encoded16=0xFFF0, GCR nibbles=[0x0f,0x0f,0x0f,0x19]
        let edges: &[u32] = &[0, 32, 48, 64, 80, 112, 128, 144, 160, 192, 208, 224, 240, 256, 272, 320];
        let result = decode_telemetry_packet(edges, edges.len(), 16);
        assert_eq!(result, Some(0xFFF));
    }

    #[test]
    fn round_trip_specific_values() {
        let test_values: &[u16] = &[0x000, 0x001, 0x064, 0x0C8, 0x100, 0x7FF, 0xFFF];
        for &v in test_values {
            let edges = gcr_encode(v, 16);
            let decoded = decode_telemetry_packet(&edges, edges.len(), 16);
            assert_eq!(decoded, Some(v), "round-trip failed for 0x{:03X}", v);
        }
    }

    #[test]
    fn round_trip_random_values() {
        // Simple LCG for deterministic "random" values
        let mut rng: u32 = 0xDEAD_BEEF;
        for _ in 0..50 {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = (rng >> 16) as u16 & 0xFFF;
            let edges = gcr_encode(v, 16);
            let decoded = decode_telemetry_packet(&edges, edges.len(), 16);
            assert_eq!(decoded, Some(v), "round-trip failed for 0x{:03X}", v);
        }
    }

    #[test]
    fn too_few_edges_returns_none() {
        let edges: &[u32] = &[0];
        assert_eq!(decode_telemetry_packet(edges, 1, 16), None);
        assert_eq!(decode_telemetry_packet(&[], 0, 16), None);
    }

    #[test]
    fn corrupted_timing_returns_none() {
        // Start with valid edges for 0x000, then corrupt one
        let mut edges: Vec<u32> = std::vec![0, 16, 32, 80, 96, 112, 160, 176, 192, 240, 272, 288, 304, 320];
        edges[3] = 999; // corrupt a timing
        let result = decode_telemetry_packet(&edges, edges.len(), 16);
        assert_eq!(result, None);
    }
}
