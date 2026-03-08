pub mod crsf;
pub mod ghst;

/// Maximum number of RC channels supported.
pub const MAX_CHANNELS: usize = 16;

/// Decoded RC channel data.
#[derive(Clone, Debug, defmt::Format)]
pub struct RcChannels {
    /// Channel values in PWM microseconds [988..2012].
    pub channels: [u16; MAX_CHANNELS],
    /// Number of valid channels in this update.
    pub channel_count: u8,
}

impl Default for RcChannels {
    fn default() -> Self {
        Self {
            channels: [1500; MAX_CHANNELS],
            channel_count: 0,
        }
    }
}

/// Receiver link quality/statistics.
#[derive(Clone, Debug, Default, defmt::Format)]
pub struct LinkStatistics {
    /// RSSI in dBm (negative value, e.g. -80).
    pub rssi_dbm: i16,
    /// Link quality as percentage [0..100].
    pub link_quality: u8,
    /// Signal-to-noise ratio in dB.
    pub snr: i8,
    /// RF mode index (protocol-specific).
    pub rf_mode: u8,
    /// Uplink TX power in dBm.
    pub tx_power_dbm: i8,
}

/// CRC8-DVB-S2 (polynomial 0xD5), used by both CRSF and GHST.
pub fn crc8_dvb_s2(mut crc: u8, byte: u8) -> u8 {
    crc ^= byte;
    for _ in 0..8 {
        if crc & 0x80 != 0 {
            crc = (crc << 1) ^ 0xD5;
        } else {
            crc <<= 1;
        }
    }
    crc
}

/// CRC8 with polynomial 0xBA, used by CRSF command frames.
pub fn crc8_poly_0xba(mut crc: u8, byte: u8) -> u8 {
    crc ^= byte;
    for _ in 0..8 {
        if crc & 0x80 != 0 {
            crc = (crc << 1) ^ 0xBA;
        } else {
            crc <<= 1;
        }
    }
    crc
}

/// Compute CRC8-DVB-S2 over a byte slice.
pub fn crc8_dvb_s2_buf(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc = crc8_dvb_s2(crc, b);
    }
    crc
}

/// Compute CRC8 poly 0xBA over a byte slice.
pub fn crc8_poly_0xba_buf(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc = crc8_poly_0xba(crc, b);
    }
    crc
}
