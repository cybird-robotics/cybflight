//! DShot frame encoding.
//!
//! Implements `prepareDshotPacket` (BF `dshot.c:115-141`) and DMA buffer fill.

use super::DSHOT_DMA_BUFFER_SIZE;

/// Encode a DShot packet from throttle value + flags.
///
/// Returns the 16-bit frame: `(throttle << 1 | telem) << 4 | checksum`.
/// In bidirectional mode the checksum is inverted.
pub const fn encode_packet(throttle: u16, telemetry_request: bool, bidirectional: bool) -> u16 {
    let packet = (throttle << 1) | (telemetry_request as u16);

    // Checksum: XOR of three nibbles
    let mut csum = packet ^ (packet >> 4) ^ (packet >> 8);
    if bidirectional {
        csum = !csum;
    }
    csum &= 0xF;

    (packet << 4) | csum
}

/// Fill an 18-word DMA buffer with timer compare values for a DShot frame.
///
/// `bit_0` and `bit_1` are the timer CCR values representing a 0-bit and 1-bit
/// respectively (hardware-specific, e.g. 7/14 for DShot600 at 12 MHz timer).
/// The buffer is filled MSB-first, with 2 trailing zero words for the reset period.
pub fn packet_to_dma_buffer(frame: u16, buf: &mut [u32; DSHOT_DMA_BUFFER_SIZE], bit_0: u32, bit_1: u32) {
    for i in 0..16 {
        let bit = (frame >> (15 - i)) & 1;
        buf[i] = if bit != 0 { bit_1 } else { bit_0 };
    }
    buf[16] = 0;
    buf[17] = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_packet_known_vectors() {
        // Hand-verified against BF prepareDshotPacket algorithm
        assert_eq!(encode_packet(0, false, false), 0x0000);
        assert_eq!(encode_packet(0, true, true), 0x001E);
        assert_eq!(encode_packet(48, false, false), 0x0606);
        assert_eq!(encode_packet(48, false, true), 0x0609);
        assert_eq!(encode_packet(48, true, true), 0x0618);
        assert_eq!(encode_packet(2047, false, false), 0xFFEE);
        assert_eq!(encode_packet(2047, true, true), 0xFFF0);
    }

    #[test]
    fn dma_buffer_fill() {
        let frame = 0x0609u16; // throttle=48, no telem, bidirectional
        let mut buf = [0u32; 18];
        packet_to_dma_buffer(frame, &mut buf, 7, 14);

        // 0x0609 = 0000_0110_0000_1001
        let expected: [u32; 18] = [
            7, 7, 7, 7, 7, 14, 14, 7, 7, 7, 7, 7, 14, 7, 7, 14, 0, 0,
        ];
        assert_eq!(buf, expected);
    }

    #[test]
    fn dma_buffer_all_ones() {
        let frame = 0xFFFFu16;
        let mut buf = [0u32; 18];
        packet_to_dma_buffer(frame, &mut buf, 5, 10);

        for i in 0..16 {
            assert_eq!(buf[i], 10, "bit {} should be 1", i);
        }
        assert_eq!(buf[16], 0);
        assert_eq!(buf[17], 0);
    }

    #[test]
    fn dma_buffer_all_zeros() {
        let frame = 0x0000u16;
        let mut buf = [99u32; 18];
        packet_to_dma_buffer(frame, &mut buf, 5, 10);

        for i in 0..16 {
            assert_eq!(buf[i], 5, "bit {} should be 0", i);
        }
        assert_eq!(buf[16], 0);
        assert_eq!(buf[17], 0);
    }
}
