//! u-blox GNSS receiver driver — UBX binary protocol, legacy CFG-* commands.
//!
//! Compatible with M8 (incl. SAM-M8Q), M9 / M9P, and ZED-F9P RTK.
//! NOT compatible with M10, which removed legacy CFG-MSG/CFG-RATE/CFG-PRT
//! in favour of the CFG-VALSET key-database. If M10 support is needed
//! later, dispatch on UBX-MON-VER and add a separate init path.
//!
//! Init sends `CFG-MSG NAV-PVT 1` and waits for the matching ACK-ACK,
//! retrying up to `MAX_ATTEMPTS` times. Retries cover the F9P cold-boot
//! window (~1–3 s), during which the receiver silently drops UBX
//! commands; duplicate CFG-MSGs are harmless because the receiver re-ACKs
//! each one. NMEA is left enabled — caller is expected to provide enough
//! UART bandwidth (≥38400 baud) for the receiver's default output.

use embedded_io_async::{Read, Write};

// ---------------------------------------------------------------------------
// UBX protocol constants
// ---------------------------------------------------------------------------

const UBX_SYNC_1: u8 = 0xB5;
const UBX_SYNC_2: u8 = 0x62;

const UBX_CLASS_NAV: u8 = 0x01;
const UBX_CLASS_ACK: u8 = 0x05;
const UBX_CLASS_CFG: u8 = 0x06;

const UBX_NAV_PVT: u8 = 0x07;
const UBX_ACK_ACK: u8 = 0x01;
const UBX_ACK_NAK: u8 = 0x00;

const UBX_CFG_MSG: u8 = 0x01;

const NAV_PVT_LEN: u16 = 92;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Parsed NAV-PVT data.
#[derive(Clone, Debug)]
pub struct NavPvt {
    pub fix_type: u8,
    /// Flags byte bit 0: receiver believes the fix is valid.
    pub gnss_fix_ok: bool,
    /// Flags byte bit 1: differential corrections were applied.
    pub diff_soln: bool,
    /// Flags byte bits 6-7: RTK carrier-phase solution status.
    /// 0 = none, 1 = float, 2 = fixed. Always 0 on M8 (no RTK hardware).
    pub carr_soln: u8,
    pub num_sv: u8,
    pub lon_1e7: i32,
    pub lat_1e7: i32,
    pub alt_msl_mm: i32,
    pub h_acc_mm: u32,
    pub v_acc_mm: u32,
    /// NED velocity components [mm/s].
    pub vel_north_mm_s: i32,
    pub vel_east_mm_s: i32,
    pub vel_down_mm_s: i32,
    pub ground_speed_mm_s: u32,
    pub heading_mot_1e5: i32,
    /// Speed accuracy estimate [mm/s].
    pub s_acc_mm_s: u32,
    pub pdop: u16,
}

/// Driver errors.
#[derive(Debug)]
pub enum Error<E> {
    Io(E),
    BadChecksum,
    Nak,
    Timeout,
}

/// Outcome of a single ACK-scan attempt.
enum AckOutcome {
    Acked,
    Naked,
    Exhausted,
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::Io(e) => defmt::write!(f, "Io({})", e),
            Error::BadChecksum => defmt::write!(f, "BadChecksum"),
            Error::Nak => defmt::write!(f, "Nak"),
            Error::Timeout => defmt::write!(f, "Timeout"),
        }
    }
}

// ---------------------------------------------------------------------------
// Checksum + frame builder
// ---------------------------------------------------------------------------

/// UBX Fletcher-16 checksum over a byte slice.
fn fletcher16(data: &[u8]) -> (u8, u8) {
    let mut ck_a: u8 = 0;
    let mut ck_b: u8 = 0;
    for &b in data {
        ck_a = ck_a.wrapping_add(b);
        ck_b = ck_b.wrapping_add(ck_a);
    }
    (ck_a, ck_b)
}

/// Build a complete UBX frame into `buf`. Returns the number of bytes written.
fn build_frame(buf: &mut [u8], class: u8, id: u8, payload: &[u8]) -> usize {
    let len = payload.len() as u16;
    buf[0] = UBX_SYNC_1;
    buf[1] = UBX_SYNC_2;
    buf[2] = class;
    buf[3] = id;
    buf[4] = len as u8;
    buf[5] = (len >> 8) as u8;
    buf[6..6 + payload.len()].copy_from_slice(payload);
    let (ck_a, ck_b) = fletcher16(&buf[2..6 + payload.len()]);
    buf[6 + payload.len()] = ck_a;
    buf[7 + payload.len()] = ck_b;
    8 + payload.len()
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// u-blox M8/M9/F9P GNSS driver.
pub struct Ublox<RW> {
    rw: RW,
}

/// Map `ReadExactError<E>` to `Error<E>`.
fn map_read_err<E>(e: embedded_io_async::ReadExactError<E>) -> Error<E> {
    match e {
        embedded_io_async::ReadExactError::UnexpectedEof => Error::Timeout,
        embedded_io_async::ReadExactError::Other(e) => Error::Io(e),
    }
}

impl<RW> Ublox<RW>
where
    RW: Read + Write,
{
    /// Initialise the receiver.
    ///
    /// Repeatedly sends `CFG-MSG NAV-PVT 1` and watches for the matching
    /// ACK-ACK; retrying covers the F9P's variable cold-boot window
    /// during which the receiver silently drops UBX commands. Returns
    /// `Error::Nak` if the receiver explicitly rejects the request, or
    /// `Error::Timeout` if no attempt is acknowledged.
    pub async fn new(
        rw: RW,
        delay: &mut impl embedded_hal_async::delay::DelayNs,
    ) -> Result<Self, Error<RW::Error>> {
        // Short prime delay; the bulk of the cold-boot tolerance comes
        // from the retry loop below, not this delay.
        delay.delay_ms(500).await;

        let mut driver = Self { rw };

        let payload = [UBX_CLASS_NAV, UBX_NAV_PVT, 1];
        let mut frame = [0u8; 16];
        let frame_len = build_frame(&mut frame, UBX_CLASS_CFG, UBX_CFG_MSG, &payload);

        // F9P cold-boot can run anywhere from ~1 s to >3 s. Rather than
        // pick a single delay that covers the worst case, send CFG-MSG
        // and watch for ACK over a short window; if absent, resend. The
        // receiver tolerates duplicate CFG-MSG (each one re-ACKs).
        const MAX_ATTEMPTS: u8 = 5;
        const FRAMES_PER_ATTEMPT: u8 = 30;
        for attempt in 0..MAX_ATTEMPTS {
            driver
                .rw
                .write_all(&frame[..frame_len])
                .await
                .map_err(Error::Io)?;
            match driver
                .scan_for_ack(UBX_CLASS_CFG, UBX_CFG_MSG, FRAMES_PER_ATTEMPT)
                .await?
            {
                AckOutcome::Acked => {
                    defmt::info!("UBX: NAV-PVT enabled (attempt {})", attempt);
                    return Ok(driver);
                }
                AckOutcome::Naked => return Err(Error::Nak),
                AckOutcome::Exhausted => {
                    // Receiver wasn't ready; pause and retry.
                    delay.delay_ms(500).await;
                }
            }
        }
        Err(Error::Timeout)
    }

    /// Read up to `budget` UBX frames looking for ACK-ACK / ACK-NAK matching
    /// the given class/id.
    async fn scan_for_ack(
        &mut self,
        expected_class: u8,
        expected_id: u8,
        budget: u8,
    ) -> Result<AckOutcome, Error<RW::Error>> {
        for _ in 0..budget {
            let (class, id, payload_len) = self.read_header().await?;

            if class == UBX_CLASS_ACK && payload_len == 2 {
                let mut ack_payload = [0u8; 2];
                self.read_payload(&mut ack_payload, 2).await?;
                if ack_payload[0] == expected_class && ack_payload[1] == expected_id {
                    if id == UBX_ACK_ACK {
                        return Ok(AckOutcome::Acked);
                    }
                    if id == UBX_ACK_NAK {
                        return Ok(AckOutcome::Naked);
                    }
                }
            } else {
                self.skip_payload(payload_len).await?;
            }
        }
        Ok(AckOutcome::Exhausted)
    }

    /// Read the next NAV-PVT fix. Blocks until a valid NAV-PVT frame arrives.
    pub async fn read_fix(&mut self) -> Result<NavPvt, Error<RW::Error>> {
        loop {
            let (class, id, payload_len) = self.read_header().await?;

            if class == UBX_CLASS_NAV && id == UBX_NAV_PVT && payload_len == NAV_PVT_LEN {
                let mut payload = [0u8; NAV_PVT_LEN as usize];
                self.read_payload(&mut payload, NAV_PVT_LEN).await?;
                return Ok(parse_nav_pvt(&payload));
            }

            self.skip_payload(payload_len).await?;
        }
    }

    /// Read the next driver event. u-blox only produces position fixes.
    pub async fn read_event(&mut self) -> Result<super::GpsEvent, Error<RW::Error>> {
        Ok(super::GpsEvent::Fix(self.read_fix().await?))
    }

    /// Sync on UBX header, return (class, id, payload_length).
    async fn read_header(&mut self) -> Result<(u8, u8, u16), Error<RW::Error>> {
        loop {
            let mut b = [0u8; 1];
            self.rw.read_exact(&mut b).await.map_err(map_read_err)?;
            if b[0] != UBX_SYNC_1 {
                continue;
            }
            self.rw.read_exact(&mut b).await.map_err(map_read_err)?;
            if b[0] != UBX_SYNC_2 {
                continue;
            }
            break;
        }

        let mut hdr = [0u8; 4]; // class, id, len_lo, len_hi
        self.rw.read_exact(&mut hdr).await.map_err(map_read_err)?;
        let payload_len = u16::from_le_bytes([hdr[2], hdr[3]]);
        Ok((hdr[0], hdr[1], payload_len))
    }

    /// Read payload + checksum. Checksum bytes are consumed but not validated;
    /// the GPS module is treated as a trusted source on a wired bus.
    async fn read_payload(&mut self, buf: &mut [u8], len: u16) -> Result<(), Error<RW::Error>> {
        let len = len as usize;
        self.rw
            .read_exact(&mut buf[..len])
            .await
            .map_err(map_read_err)?;

        let mut ck = [0u8; 2];
        self.rw.read_exact(&mut ck).await.map_err(map_read_err)?;
        let _ = ck;
        Ok(())
    }

    /// Skip a message payload + 2-byte checksum.
    async fn skip_payload(&mut self, len: u16) -> Result<(), Error<RW::Error>> {
        let total = len as usize + 2;
        let mut discard = [0u8; 32];
        let mut remaining = total;
        while remaining > 0 {
            let chunk = remaining.min(discard.len());
            self.rw
                .read_exact(&mut discard[..chunk])
                .await
                .map_err(map_read_err)?;
            remaining -= chunk;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NAV-PVT parser
// ---------------------------------------------------------------------------

fn parse_nav_pvt(p: &[u8; 92]) -> NavPvt {
    let flags = p[21];
    NavPvt {
        fix_type: p[20],
        gnss_fix_ok: (flags & 0x01) != 0,
        diff_soln: (flags & 0x02) != 0,
        carr_soln: (flags >> 6) & 0x03,
        num_sv: p[23],
        lon_1e7: i32::from_le_bytes([p[24], p[25], p[26], p[27]]),
        lat_1e7: i32::from_le_bytes([p[28], p[29], p[30], p[31]]),
        alt_msl_mm: i32::from_le_bytes([p[36], p[37], p[38], p[39]]),
        h_acc_mm: u32::from_le_bytes([p[40], p[41], p[42], p[43]]),
        v_acc_mm: u32::from_le_bytes([p[44], p[45], p[46], p[47]]),
        vel_north_mm_s: i32::from_le_bytes([p[48], p[49], p[50], p[51]]),
        vel_east_mm_s: i32::from_le_bytes([p[52], p[53], p[54], p[55]]),
        vel_down_mm_s: i32::from_le_bytes([p[56], p[57], p[58], p[59]]),
        ground_speed_mm_s: i32::from_le_bytes([p[60], p[61], p[62], p[63]]).unsigned_abs(),
        heading_mot_1e5: i32::from_le_bytes([p[64], p[65], p[66], p[67]]),
        s_acc_mm_s: u32::from_le_bytes([p[68], p[69], p[70], p[71]]),
        pdop: u16::from_le_bytes([p[76], p[77]]),
    }
}
