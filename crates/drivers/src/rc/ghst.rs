//! GHST (ImmersionRC Ghost) protocol driver.
//!
//! Implements frame parsing, RC channel decoding (10-bit legacy and 12-bit),
//! RSSI/link quality parsing, and telemetry frame construction — all matching
//! the Betaflight reference implementation.

use embedded_io_async::{Read, Write};

use super::{LinkStatistics, RcChannels, MAX_CHANNELS, crc8_dvb_s2_buf};

// ---------------------------------------------------------------------------
// Protocol constants (from ghst_protocol.h)
// ---------------------------------------------------------------------------

pub const GHST_RX_BAUDRATE: u32 = 420000;
pub const GHST_TX_BAUDRATE_FAST: u32 = 400000;

pub const GHST_FRAME_SIZE: usize = 14;
pub const GHST_PAYLOAD_SIZE: usize = 10;

// Addresses
pub const GHST_ADDR_FC: u8 = 0x82;
pub const GHST_ADDR_RX: u8 = 0x89;

// Uplink frame types (RX → FC)
pub const GHST_UL_RC_CHANS_HS4_FIRST: u8 = 0x10;
pub const GHST_UL_RC_CHANS_HS4_5TO8: u8 = 0x10;
pub const GHST_UL_RC_CHANS_HS4_9TO12: u8 = 0x11;
pub const GHST_UL_RC_CHANS_HS4_13TO16: u8 = 0x12;
pub const GHST_UL_RC_CHANS_HS4_RSSI: u8 = 0x13;
pub const GHST_UL_RC_CHANS_HS4_LAST: u8 = 0x1F;

pub const GHST_UL_RC_CHANS_HS4_12_FIRST: u8 = 0x30;
pub const GHST_UL_RC_CHANS_HS4_12_5TO8: u8 = 0x30;
pub const GHST_UL_RC_CHANS_HS4_12_9TO12: u8 = 0x31;
pub const GHST_UL_RC_CHANS_HS4_12_13TO16: u8 = 0x32;
pub const GHST_UL_RC_CHANS_HS4_12_RSSI: u8 = 0x33;
pub const GHST_UL_RC_CHANS_HS4_12_LAST: u8 = 0x3F;

// Downlink frame types (FC → TX)
pub const GHST_DL_PACK_STAT: u8 = 0x23;
pub const GHST_DL_GPS_PRIMARY: u8 = 0x25;
pub const GHST_DL_GPS_SECONDARY: u8 = 0x26;
pub const GHST_DL_MAGBARO: u8 = 0x27;

// Frame structure sizes
pub const GHST_FRAME_LENGTH_CRC: usize = 1;
pub const GHST_FRAME_LENGTH_TYPE: usize = 1;

// Center values
const RC_CTR_VAL_12BIT_PRIMARY: u16 = 2048;
const RC_CTR_VAL_12BIT_AUX: u16 = 128 << 2;

// Misc telemetry flags
pub const GPS_FLAGS_FIX: u8 = 0x01;
pub const GPS_FLAGS_FIX_HOME: u8 = 0x02;
pub const MISC_FLAGS_MAGHEAD: u8 = 0x01;
pub const MISC_FLAGS_BAROALT: u8 = 0x02;
pub const MISC_FLAGS_VARIO: u8 = 0x04;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Events emitted by the GHST driver when parsing incoming frames.
#[derive(Clone, Debug)]
pub enum GhstEvent {
    /// RC channels updated (primary always, aux round-robin).
    RcChannels(RcChannels),
    /// RSSI/link quality frame.
    RssiFrame(LinkStatistics),
    /// Any other frame type.
    Other { frame_type: u8 },
}

/// Driver errors.
#[derive(Debug)]
pub enum Error<E> {
    Io(E),
    BadCrc,
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::Io(e) => defmt::write!(f, "Io({})", e),
            Error::BadCrc => defmt::write!(f, "BadCrc"),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Map `ReadExactError<E>` to `Error<E>`.
fn map_read_err<E>(e: embedded_io_async::ReadExactError<E>) -> Error<E> {
    match e {
        embedded_io_async::ReadExactError::UnexpectedEof => Error::BadCrc,
        embedded_io_async::ReadExactError::Other(e) => Error::Io(e),
    }
}

/// GHST protocol driver, generic over an async read+write transport.
pub struct Ghst<RW> {
    rw: RW,
    /// Internal channel accumulator — primary always updated, aux round-robin.
    channel_data: [u16; MAX_CHANNELS],
    /// Whether we've received at least one RSSI frame (needed to know RF protocol).
    rf_protocol_known: bool,
}

impl<RW> Ghst<RW>
where
    RW: Read + Write,
{
    pub fn new(rw: RW) -> Self {
        let mut channel_data = [RC_CTR_VAL_12BIT_AUX; MAX_CHANNELS];
        // Primary channels default to 12-bit center
        for ch in channel_data.iter_mut().take(4) {
            *ch = RC_CTR_VAL_12BIT_PRIMARY;
        }

        Self {
            rw,
            channel_data,
            rf_protocol_known: false,
        }
    }

    /// Read one complete GHST frame, validate CRC, and parse the payload.
    ///
    /// Blocks until a valid frame addressed to the FC is received.
    pub async fn read_frame(&mut self) -> Result<GhstEvent, Error<RW::Error>> {
        loop {
            // 1. Sync: read bytes until we see GHST_ADDR_FC
            let addr = loop {
                let mut b = [0u8; 1];
                self.rw.read_exact(&mut b).await.map_err(map_read_err)?;
                if b[0] == GHST_ADDR_FC {
                    break b[0];
                }
            };

            // 2. Read remaining 13 bytes (GHST frames are always 14 bytes total)
            let mut frame = [0u8; GHST_FRAME_SIZE];
            frame[0] = addr;
            self.rw
                .read_exact(&mut frame[1..GHST_FRAME_SIZE])
                .await
                .map_err(map_read_err)?;

            // Frame: [addr, len, type, payload..., CRC]
            let len = frame[1] as usize;
            let frame_type = frame[2];

            // Validate frame structure
            let full_frame_len = len + 2; // addr + len_byte + (type + payload + CRC)
            if full_frame_len > GHST_FRAME_SIZE {
                continue;
            }

            // 3. CRC check: CRC covers type + payload (NOT addr, NOT len)
            let crc_idx = full_frame_len - 1;
            let expected_crc = frame[crc_idx];
            let computed_crc = crc8_dvb_s2_buf(&frame[2..crc_idx]);
            if computed_crc != expected_crc {
                continue;
            }

            let payload = &frame[3..crc_idx];

            // 4. Parse by frame type
            let is_legacy =
                frame_type >= GHST_UL_RC_CHANS_HS4_FIRST && frame_type <= GHST_UL_RC_CHANS_HS4_LAST;
            let is_12bit = frame_type >= GHST_UL_RC_CHANS_HS4_12_FIRST
                && frame_type <= GHST_UL_RC_CHANS_HS4_12_LAST;

            if is_legacy || is_12bit {
                if payload.len() < GHST_PAYLOAD_SIZE {
                    continue;
                }

                // Parse RSSI frame first (if applicable)
                match frame_type {
                    GHST_UL_RC_CHANS_HS4_RSSI | GHST_UL_RC_CHANS_HS4_12_RSSI => {
                        self.rf_protocol_known = true;
                        let lq = payload[6];
                        let rssi = payload[7];
                        let rf_protocol = payload[8];
                        let tx_pwr_dbm = payload[9] as i8;

                        let stats = LinkStatistics {
                            rssi_dbm: -(rssi as i16),
                            link_quality: lq,
                            snr: 0,
                            rf_mode: rf_protocol,
                            tx_power_dbm: tx_pwr_dbm,
                        };

                        // Still extract primary channels from RSSI frames
                        self.extract_primary_channels(payload, is_legacy);

                        return Ok(GhstEvent::RssiFrame(stats));
                    }
                    _ => {}
                }

                // Only process channel data after we know the RF protocol
                if !self.rf_protocol_known {
                    continue;
                }

                // Extract primary channels (always present)
                self.extract_primary_channels(payload, is_legacy);

                // Determine aux channel start index
                let start_idx = match frame_type {
                    GHST_UL_RC_CHANS_HS4_5TO8 | GHST_UL_RC_CHANS_HS4_12_5TO8 => Some(4usize),
                    GHST_UL_RC_CHANS_HS4_9TO12 | GHST_UL_RC_CHANS_HS4_12_9TO12 => Some(8),
                    GHST_UL_RC_CHANS_HS4_13TO16 | GHST_UL_RC_CHANS_HS4_12_13TO16 => Some(12),
                    _ => None,
                };

                // Extract aux channels (8-bit, shifted << 2)
                if let Some(si) = start_idx {
                    for i in 0..4 {
                        if si + i < MAX_CHANNELS {
                            self.channel_data[si + i] = (payload[6 + i] as u16) << 2;
                        }
                    }

                    // Apply legacy rescaling to aux channels
                    if is_legacy {
                        for i in 0..4 {
                            if si + i < MAX_CHANNELS {
                                let raw = self.channel_data[si + i];
                                self.channel_data[si + i] =
                                    (5u16.wrapping_mul(raw >> 2)).wrapping_sub(108);
                            }
                        }
                    }
                }

                // Convert internal channel data to PWM and return
                let rc = self.channels_to_pwm();
                return Ok(GhstEvent::RcChannels(rc));
            } else {
                return Ok(GhstEvent::Other { frame_type });
            }
        }
    }

    /// Extract 4 primary 12-bit channels from payload bytes [0..6].
    fn extract_primary_channels(&mut self, payload: &[u8], is_legacy: bool) {
        // 4 × 12-bit channels packed in 6 bytes (48 bits)
        let ch1 = (payload[0] as u16) | ((payload[1] as u16 & 0x0F) << 8);
        let ch2 = ((payload[1] as u16) >> 4) | ((payload[2] as u16) << 4);
        let ch3 = (payload[3] as u16) | ((payload[4] as u16 & 0x0F) << 8);
        let ch4 = ((payload[4] as u16) >> 4) | ((payload[5] as u16) << 4);

        self.channel_data[0] = ch1;
        self.channel_data[1] = ch2;
        self.channel_data[2] = ch3;
        self.channel_data[3] = ch4;

        // Apply legacy rescaling to primary channels: ((5 * raw) >> 2) - 430
        if is_legacy {
            for i in 0..4 {
                let raw = self.channel_data[i];
                self.channel_data[i] = ((5u32 * raw as u32) >> 2) as u16 - 430;
            }
        }
    }

    /// Convert internal channel_data to PWM microseconds.
    ///
    /// Primary (ch0-3): 0.25 * raw + 988
    /// Aux (ch4-15): raw + 988
    fn channels_to_pwm(&self) -> RcChannels {
        let mut channels = [1500u16; MAX_CHANNELS];
        for (i, ch) in channels.iter_mut().enumerate().take(MAX_CHANNELS) {
            let raw = self.channel_data[i];
            if i < 4 {
                *ch = ((raw as f32) * 0.25 + 988.0) as u16;
            } else {
                *ch = raw + 988;
            }
        }
        RcChannels {
            channels,
            channel_count: 16,
        }
    }

    // -----------------------------------------------------------------------
    // Telemetry frame builders (all little-endian, matching BF GHST)
    // -----------------------------------------------------------------------

    /// Write a complete GHST telemetry frame.
    async fn write_frame(
        &mut self,
        frame_type: u8,
        payload: &[u8; GHST_PAYLOAD_SIZE],
    ) -> Result<(), Error<RW::Error>> {
        let mut frame = [0u8; GHST_FRAME_SIZE];
        frame[0] = GHST_ADDR_RX;
        frame[1] = (GHST_PAYLOAD_SIZE + GHST_FRAME_LENGTH_CRC + GHST_FRAME_LENGTH_TYPE) as u8;
        frame[2] = frame_type;
        frame[3..13].copy_from_slice(payload);

        // CRC over type + payload
        let crc = crc8_dvb_s2_buf(&frame[2..13]);
        frame[13] = crc;

        self.rw.write_all(&frame).await.map_err(Error::Io)?;
        Ok(())
    }

    /// Send battery (Pack) telemetry (type 0x23).
    ///
    /// - `voltage_10mv`: battery voltage in units of 10mV
    /// - `current_10ma`: current in units of 10mA
    /// - `mah_10`: capacity drawn in units of 10mAh
    /// - `armed`: true if vehicle is armed
    pub async fn write_pack(
        &mut self,
        voltage_10mv: u16,
        current_10ma: u16,
        mah_10: u16,
        armed: bool,
    ) -> Result<(), Error<RW::Error>> {
        let mut payload = [0u8; GHST_PAYLOAD_SIZE];
        // All little-endian (sbufWriteU16 in BF GHST telemetry)
        payload[0..2].copy_from_slice(&voltage_10mv.to_le_bytes());
        payload[2..4].copy_from_slice(&current_10ma.to_le_bytes());
        payload[4..6].copy_from_slice(&mah_10.to_le_bytes());
        payload[6] = 0; // Rx voltage (not from FC)
        let mut flags = 0u8;
        if !armed {
            flags |= 0x01; // PACK_FLAGS_Disarmed (inverted: set when disarmed)
        }
        payload[7] = flags;
        payload[8] = 0; // tbd2
        payload[9] = 0; // tbd3
        self.write_frame(GHST_DL_PACK_STAT, &payload).await
    }

    /// Send GPS primary telemetry (type 0x25).
    ///
    /// - `lat`, `lon`: raw GPS coordinates (degrees × 10^7)
    /// - `alt_m`: altitude in meters
    pub async fn write_gps_primary(
        &mut self,
        lat: i32,
        lon: i32,
        alt_m: i16,
    ) -> Result<(), Error<RW::Error>> {
        let mut payload = [0u8; GHST_PAYLOAD_SIZE];
        payload[0..4].copy_from_slice(&lat.to_le_bytes());
        payload[4..8].copy_from_slice(&lon.to_le_bytes());
        payload[8..10].copy_from_slice(&alt_m.to_le_bytes());
        self.write_frame(GHST_DL_GPS_PRIMARY, &payload).await
    }

    /// Send GPS secondary telemetry (type 0x26).
    ///
    /// - `speed`: ground speed in 0.1 m/s
    /// - `course`: course in degrees × 10
    /// - `sats`: number of satellites
    /// - `dist_home`: distance to home in units of 10m
    /// - `dir_home`: direction to home in degrees × 10
    /// - `flags`: GPS flags (GPS_FLAGS_FIX, GPS_FLAGS_FIX_HOME)
    pub async fn write_gps_secondary(
        &mut self,
        speed: u16,
        course: u16,
        sats: u8,
        dist_home: u16,
        dir_home: u16,
        flags: u8,
    ) -> Result<(), Error<RW::Error>> {
        let mut payload = [0u8; GHST_PAYLOAD_SIZE];
        payload[0..2].copy_from_slice(&speed.to_le_bytes());
        payload[2..4].copy_from_slice(&course.to_le_bytes());
        payload[4] = sats;
        payload[5..7].copy_from_slice(&dist_home.to_le_bytes());
        payload[7..9].copy_from_slice(&dir_home.to_le_bytes());
        payload[9] = flags;
        self.write_frame(GHST_DL_GPS_SECONDARY, &payload).await
    }

    /// Send magnetometer/barometer telemetry (type 0x27).
    ///
    /// - `yaw_decideg`: magnetic heading in decidegrees
    /// - `alt_m`: barometric altitude in meters
    /// - `vario_cm_s`: vertical speed in cm/s (truncated to i8)
    /// - `flags`: MISC_FLAGS_MAGHEAD | MISC_FLAGS_BAROALT | MISC_FLAGS_VARIO
    pub async fn write_magbaro(
        &mut self,
        yaw_decideg: i16,
        alt_m: i16,
        vario_cm_s: i8,
        flags: u8,
    ) -> Result<(), Error<RW::Error>> {
        let mut payload = [0u8; GHST_PAYLOAD_SIZE];
        payload[0..2].copy_from_slice(&yaw_decideg.to_le_bytes());
        payload[2..4].copy_from_slice(&alt_m.to_le_bytes());
        payload[4] = vario_cm_s as u8;
        // payload[5..9] = 0 (reserved)
        payload[9] = flags;
        self.write_frame(GHST_DL_MAGBARO, &payload).await
    }
}
