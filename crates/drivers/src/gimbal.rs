//! Z-1Mini gimbal — GCU private-protocol encoder/decoder (`no_std`).
//!
//! Pure byte-level codec for the GCU 私有通信协议 V2.0.6, ported from the
//! `z1mini-gcu` crate (itself a byte-verified port of the vendor `DataConvert.cpp`,
//! cross-checked against the protocol PDF's worked packets). **All I/O lives in the
//! caller** (the gimbal task on the chaser); this module only produces/parses bytes.
//!
//! Transport: UART 8N1, stream control frames at 30–100 Hz. Angles are
//! centidegrees (0.01°). Host→GCU frame:
//! `A8 E5 | len(u16 LE) | ver | main[32] | sub[32] | cmd | params | CRC(u16 BE)`,
//! 72 bytes with no command params.
//!
//! Conventions (verified on hardware / from PDF App.3-4, see project notes): the
//! camera setpoint and the carrier-INS attitude share the standard aerospace
//! convention — **yaw = compass heading (0=N, +CW), pitch = +up, roll =
//! +right-wing-down**. The setpoint yaw field is signed `i16` centideg (±180°);
//! the CarrierIns yaw field is `u16` 0..36000.

const HOST_HEADER: [u8; 2] = [0xA8, 0xE5];
const GCU_HEADER: [u8; 2] = [0x8A, 0x5E];
const VERSION: u8 = 0x02;

/// Length of a control frame with no command parameters (and the GCU reply with
/// no execution-status byte).
pub const FRAME_LEN: usize = 72;

/// Command opcodes (frame byte 69).
pub mod cmd {
    /// Empty / maintain current mode (carries live setpoints).
    pub const MAINTAIN: u8 = 0x00;
    /// Return-to-home (only effective in pointing-follow; no-op in angle/euler/fpv).
    pub const RECENTER: u8 = 0x03;
    /// Angle control: roll/pitch = absolute euler, yaw = relative to body. No INS needed.
    pub const ANGLE: u8 = 0x10;
    /// Euler control: roll/pitch/yaw = absolute euler. Feed CarrierIns for the heading reference.
    pub const EULER: u8 = 0x14;
    /// FPV: roll/pitch/yaw relative to the mount. No INS needed (good for a bench self-test).
    pub const FPV: u8 = 0x1C;
}

/// CRC-16/CCITT (poly `0x1021`, init `0`), nibble-table — the vendor
/// `CalculateCrc16`. Computed over every byte **except** the trailing two CRC
/// bytes; the result is stored **big-endian** (high byte first).
pub fn crc16(data: &[u8]) -> u16 {
    const TAB: [u16; 16] = [
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7, 0x8108, 0x9129, 0xa14a,
        0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    ];
    let mut crc: u16 = 0;
    for &byte in data {
        let da = (crc >> 12) as usize;
        crc <<= 4;
        crc ^= TAB[da ^ (byte >> 4) as usize];
        let da = (crc >> 12) as usize;
        crc <<= 4;
        crc ^= TAB[da ^ (byte & 0x0F) as usize];
    }
    crc
}

/// Camera roll/pitch/yaw control triple, centidegrees. For euler/angle modes the
/// yaw field is a signed angle (±180° → ±18000); the GCU clamps to its mechanical
/// limits and best-efforts beyond them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Setpoint {
    pub roll_cd: i16,
    pub pitch_cd: i16,
    pub yaw_cd: i16,
}

impl Setpoint {
    pub const fn new(roll_cd: i16, pitch_cd: i16, yaw_cd: i16) -> Self {
        Self { roll_cd, pitch_cd, yaw_cd }
    }
}

/// Carrier (aircraft) inertial data carried in the main frame (bytes 12..30).
/// Attitude in centidegrees (yaw `u16` 0..36000 = compass heading), acceleration
/// in 0.01 m/s² and velocity in 0.01 m/s, both in **North/East/Up** order.
/// Supplying it sets the "carrier inertial data valid" status bit so the GCU
/// fuses it (heading reference + maneuver feed-forward).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CarrierIns {
    pub roll_cd: i16,
    pub pitch_cd: i16,
    pub yaw_cd: u16,
    pub acc_n: i16,
    pub acc_e: i16,
    pub acc_u: i16,
    pub vel_n: i16,
    pub vel_e: i16,
    pub vel_u: i16,
}

/// Encode a control frame into `out` (which must be at least [`FRAME_LEN`] bytes).
/// Returns the number of bytes written ([`FRAME_LEN`]).
///
/// `cmd` selects the mode (see [`cmd`]). The setpoint is always marked valid
/// (status bit2). Passing `ins = Some(..)` writes the carrier INS block and sets
/// the "carrier valid" status bit0 (required for [`cmd::EULER`] to hold an
/// absolute world heading); `None` leaves the pod stabilizing on its own IMU
/// (fine for [`cmd::ANGLE`]/[`cmd::FPV`] and the bench self-test).
///
/// Panics if `out.len() < FRAME_LEN`.
pub fn encode(out: &mut [u8], cmd: u8, sp: Setpoint, ins: Option<CarrierIns>) -> usize {
    let b = &mut out[..FRAME_LEN];
    b.fill(0);
    b[0..2].copy_from_slice(&HOST_HEADER);
    b[2..4].copy_from_slice(&(FRAME_LEN as u16).to_le_bytes());
    b[4] = VERSION;
    b[5..7].copy_from_slice(&sp.roll_cd.to_le_bytes());
    b[7..9].copy_from_slice(&sp.pitch_cd.to_le_bytes());
    b[9..11].copy_from_slice(&sp.yaw_cd.to_le_bytes());

    let mut status = 0x04u8; // bit2: control value valid
    if let Some(i) = ins {
        status |= 0x01; // bit0: carrier inertial data valid
        b[12..14].copy_from_slice(&i.roll_cd.to_le_bytes());
        b[14..16].copy_from_slice(&i.pitch_cd.to_le_bytes());
        b[16..18].copy_from_slice(&i.yaw_cd.to_le_bytes());
        b[18..20].copy_from_slice(&i.acc_n.to_le_bytes());
        b[20..22].copy_from_slice(&i.acc_e.to_le_bytes());
        b[22..24].copy_from_slice(&i.acc_u.to_le_bytes());
        b[24..26].copy_from_slice(&i.vel_n.to_le_bytes());
        b[26..28].copy_from_slice(&i.vel_e.to_le_bytes());
        b[28..30].copy_from_slice(&i.vel_u.to_le_bytes());
    }
    b[11] = status;
    // byte 30 (sub-frame request) and 31..69 (reserved / no GNSS sub-frame) stay 0.
    b[69] = cmd;

    let crc = crc16(&b[..FRAME_LEN - 2]);
    b[FRAME_LEN - 2] = (crc >> 8) as u8;
    b[FRAME_LEN - 1] = (crc & 0xFF) as u8;
    FRAME_LEN
}

/// Convenience: an [`cmd::EULER`] frame with carrier INS — the chaser's tracking
/// frame.
pub fn encode_euler(out: &mut [u8], sp: Setpoint, ins: CarrierIns) -> usize {
    encode(out, cmd::EULER, sp, Some(ins))
}

/// A passive "is the pod alive?" probe frame: empty/maintain command, control
/// value **invalid** (so it commands no motion) but with the telemetry sub-frame
/// requested, so the pod answers with a GCU→host frame. Byte-identical to the
/// protocol PDF's "空命令" example. Returns bytes written ([`FRAME_LEN`]).
pub fn encode_heartbeat(out: &mut [u8]) -> usize {
    let b = &mut out[..FRAME_LEN];
    b.fill(0);
    b[0..2].copy_from_slice(&HOST_HEADER);
    b[2..4].copy_from_slice(&(FRAME_LEN as u16).to_le_bytes());
    b[4] = VERSION;
    // status byte 11 = 0 → control invalid (no motion). byte 30 = request sub-frame.
    b[30] = 0x01;
    b[69] = cmd::MAINTAIN;
    let crc = crc16(&b[..FRAME_LEN - 2]);
    b[FRAME_LEN - 2] = (crc >> 8) as u8;
    b[FRAME_LEN - 1] = (crc & 0xFF) as u8;
    FRAME_LEN
}

/// Minimal decoded GCU→host telemetry (for diagnostics / optional readback).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Telemetry {
    /// Pod work mode (byte 5): 0x10 angle, 0x14 euler, 0x16 gaze, 0x1C fpv, …
    pub work_mode: u8,
    /// Camera absolute roll/pitch (centideg) and yaw/heading (0..36000). Valid
    /// only with a valid carrier-INS heading.
    pub cam_roll_cd: i16,
    pub cam_pitch_cd: i16,
    pub cam_yaw_cd: u16,
    /// Command this frame is feedback for (byte 69).
    pub cmd: u8,
    /// Execution status (byte 70): `Some(0)` success, `Some(>0)` failure,
    /// `None` for a 72-byte frame (no status / in-progress).
    pub cmd_status: Option<u8>,
}

/// Parse one complete GCU→host frame. Returns `None` on bad header, length
/// mismatch, or CRC failure. Use a stream framer (resync on `8A 5E`, read the
/// length field) to carve frames out of a UART byte stream first.
pub fn parse(f: &[u8]) -> Option<Telemetry> {
    if f.len() < FRAME_LEN || f[0] != GCU_HEADER[0] || f[1] != GCU_HEADER[1] {
        return None;
    }
    let declared = u16::from_le_bytes([f[2], f[3]]) as usize;
    if declared != f.len() {
        return None;
    }
    let computed = crc16(&f[..declared - 2]);
    let stored = ((f[declared - 2] as u16) << 8) | f[declared - 1] as u16;
    if computed != stored {
        return None;
    }
    Some(Telemetry {
        work_mode: f[5],
        cam_roll_cd: i16::from_le_bytes([f[18], f[19]]),
        cam_pitch_cd: i16::from_le_bytes([f[20], f[21]]),
        cam_yaw_cd: u16::from_le_bytes([f[22], f[23]]),
        cmd: f[69],
        cmd_status: if declared == FRAME_LEN { None } else { Some(f[70]) },
    })
}

/// Streaming framer for GCU→host telemetry over a UART byte stream. Feed bytes
/// with [`push`](Self::push); pull complete CRC-validated frames with
/// [`pull`](Self::pull). Resyncs on the `8A 5E` header and uses the length field.
pub struct FrameScanner {
    buf: [u8; 128],
    len: usize,
}

impl Default for FrameScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameScanner {
    pub const fn new() -> Self {
        Self { buf: [0u8; 128], len: 0 }
    }

    /// Append received bytes. On overflow the oldest half is dropped (the next
    /// header search re-syncs).
    pub fn push(&mut self, data: &[u8]) {
        for &byte in data {
            if self.len == self.buf.len() {
                let half = self.buf.len() / 2;
                self.buf.copy_within(half.., 0);
                self.len -= half;
            }
            self.buf[self.len] = byte;
            self.len += 1;
        }
    }

    /// Pull the next complete, CRC-valid frame, or `None` if more bytes are
    /// needed. Call repeatedly to fully drain. Garbage and CRC failures are
    /// skipped as it re-syncs.
    pub fn pull(&mut self) -> Option<Telemetry> {
        loop {
            // Find the header.
            let mut start = None;
            let mut i = 0;
            while i + 1 < self.len {
                if self.buf[i] == GCU_HEADER[0] && self.buf[i + 1] == GCU_HEADER[1] {
                    start = Some(i);
                    break;
                }
                i += 1;
            }
            let Some(s) = start else {
                // No header yet — keep only a possible trailing header byte.
                if self.len > 1 {
                    self.buf[0] = self.buf[self.len - 1];
                    self.len = 1;
                }
                return None;
            };
            if s > 0 {
                self.buf.copy_within(s..self.len, 0);
                self.len -= s;
            }
            if self.len < 4 {
                return None; // need the length field
            }
            let total = u16::from_le_bytes([self.buf[2], self.buf[3]]) as usize;
            if total < FRAME_LEN || total > self.buf.len() {
                // Implausible length — drop this header byte and re-sync.
                self.buf.copy_within(1..self.len, 0);
                self.len -= 1;
                continue;
            }
            if self.len < total {
                return None; // need the rest of the frame
            }
            let parsed = parse(&self.buf[..total]);
            self.buf.copy_within(total..self.len, 0);
            self.len -= total;
            if parsed.is_some() {
                return parsed;
            }
            // CRC failed — keep scanning.
        }
    }
}

// ───────────────────────────── aiming geometry ─────────────────────────────
//
// Pure (host-testable) functions that turn the chaser's pose + the leader's
// position into the EulerControl setpoint and the CarrierIns attitude, in the
// GCU convention (verified on hardware + PDF App.3/4): yaw = compass heading
// (0=N, +CW), pitch = +up / +nose-up, roll = +right-wing-down. World frame is
// ENU (x=E, y=N, z=U); body frame is FLU (x=fwd, y=left, z=up).

use nalgebra::{UnitQuaternion, Vector3};

/// Radians → centidegrees (0.01°).
const RAD2CD: f32 = 18000.0 / core::f32::consts::PI;

/// EulerControl setpoint to point the camera along `los_enu` — the world-ENU
/// vector from the camera to the target. Roll is held level (0). yaw is the
/// signed compass azimuth (±180°), pitch the elevation (+up).
pub fn aim_setpoint(los_enu: Vector3<f32>) -> Setpoint {
    let az = libm::atan2f(los_enu.x, los_enu.y); // 0=N, +CW (toward E)
    let el = libm::atan2f(los_enu.z, libm::hypotf(los_enu.x, los_enu.y)); // +up
    Setpoint {
        roll_cd: 0,
        pitch_cd: (el * RAD2CD).clamp(-9000.0, 9000.0) as i16,
        yaw_cd: (az * RAD2CD).clamp(-18000.0, 18000.0) as i16,
    }
}

/// The aircraft's absolute attitude — from the body→world ESKF quaternion — as
/// the CarrierIns attitude triple `(roll_cd, pitch_cd, yaw_cd)`: roll =
/// +right-wing-down (±180°), pitch = +nose-up (±90°), yaw = compass heading
/// (`u16`, 0..36000).
pub fn carrier_attitude(q_body_to_world: &UnitQuaternion<f32>) -> (i16, i16, u16) {
    let fwd = q_body_to_world * Vector3::x(); // body +x (forward) in world ENU
    let right = q_body_to_world * (-Vector3::<f32>::y()); // body right = −y (FLU)
    let up = q_body_to_world * Vector3::z(); // body +z (up)

    let heading = libm::atan2f(fwd.x, fwd.y); // 0=N, +CW
    let pitch = libm::atan2f(fwd.z, libm::hypotf(fwd.x, fwd.y)); // +nose up
    let roll = libm::atan2f(-right.z, up.z); // +right-wing-down

    let mut yaw_cd = heading * RAD2CD;
    while yaw_cd < 0.0 {
        yaw_cd += 36000.0;
    }
    while yaw_cd >= 36000.0 {
        yaw_cd -= 36000.0;
    }
    (
        (roll * RAD2CD).clamp(-18000.0, 18000.0) as i16,
        (pitch * RAD2CD).clamp(-9000.0, 9000.0) as i16,
        yaw_cd as u16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // CRC validated against the protocol PDF V2.0.6, Appendix 5 worked packets.
    #[test]
    fn crc_empty_command_vector() {
        // "空命令": header A8E5, len 0x48, ver 02, byte30=0x01 (request sub-frame),
        // all else 0 → CRC 0xFD13.
        let mut p = [0u8; 70];
        p[0] = 0xA8;
        p[1] = 0xE5;
        p[2] = 0x48;
        p[4] = 0x02;
        p[30] = 0x01;
        assert_eq!(crc16(&p), 0xFD13);
    }

    #[test]
    fn crc_pitch_100_vector() {
        // "俯仰控制 控制量 100": pitch=+100 (bytes 7..9), status 0x04, byte30=0x01 → CRC 0xE79F.
        let mut p = [0u8; 70];
        p[0] = 0xA8;
        p[1] = 0xE5;
        p[2] = 0x48;
        p[4] = 0x02;
        p[7] = 0x64; // 100 LE
        p[11] = 0x04;
        p[30] = 0x01;
        assert_eq!(crc16(&p), 0xE79F);
    }

    #[test]
    fn euler_frame_layout() {
        let mut buf = [0u8; FRAME_LEN];
        let ins = CarrierIns { yaw_cd: 9000, vel_n: 50, ..Default::default() };
        let n = encode_euler(&mut buf, Setpoint::new(0, -1500, -9000), ins);
        assert_eq!(n, FRAME_LEN);
        assert_eq!(&buf[0..2], &HOST_HEADER);
        assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), FRAME_LEN as u16);
        assert_eq!(buf[4], VERSION);
        assert_eq!(i16::from_le_bytes([buf[7], buf[8]]), -1500); // setpoint pitch
        assert_eq!(i16::from_le_bytes([buf[9], buf[10]]), -9000); // setpoint yaw (signed)
        assert_eq!(buf[11], 0x05); // bit2 control valid | bit0 carrier valid
        assert_eq!(u16::from_le_bytes([buf[16], buf[17]]), 9000); // carrier yaw
        assert_eq!(i16::from_le_bytes([buf[24], buf[25]]), 50); // carrier vel_n
        assert_eq!(buf[69], cmd::EULER);
        // CRC self-consistent.
        let crc = crc16(&buf[..FRAME_LEN - 2]);
        assert_eq!(((buf[FRAME_LEN - 2] as u16) << 8) | buf[FRAME_LEN - 1] as u16, crc);
    }

    #[test]
    fn encode_without_ins_clears_carrier_bit() {
        let mut buf = [0u8; FRAME_LEN];
        encode(&mut buf, cmd::FPV, Setpoint::new(0, 1000, 0), None);
        assert_eq!(buf[11], 0x04); // control valid, carrier NOT valid
        assert_eq!(buf[69], cmd::FPV);
        // carrier block untouched (zero)
        assert_eq!(u16::from_le_bytes([buf[16], buf[17]]), 0);
    }

    #[test]
    fn parse_roundtrips_and_rejects_bad_crc() {
        let mut f = [0u8; FRAME_LEN];
        f[0] = 0x8A;
        f[1] = 0x5E;
        f[2..4].copy_from_slice(&(FRAME_LEN as u16).to_le_bytes());
        f[4] = VERSION;
        f[5] = cmd::EULER; // work_mode
        f[20..22].copy_from_slice(&(-1500i16).to_le_bytes()); // cam pitch
        f[22..24].copy_from_slice(&14613u16.to_le_bytes()); // cam yaw 146.13°
        f[69] = cmd::EULER;
        let crc = crc16(&f[..FRAME_LEN - 2]);
        f[FRAME_LEN - 2] = (crc >> 8) as u8;
        f[FRAME_LEN - 1] = (crc & 0xFF) as u8;

        let t = parse(&f).expect("valid frame should parse");
        assert_eq!(t.work_mode, cmd::EULER);
        assert_eq!(t.cam_pitch_cd, -1500);
        assert_eq!(t.cam_yaw_cd, 14613);
        assert_eq!(t.cmd, cmd::EULER);
        assert_eq!(t.cmd_status, None);

        let mut bad = f;
        bad[FRAME_LEN - 1] ^= 0xFF;
        assert!(parse(&bad).is_none());

        let mut wrong_header = f;
        wrong_header[0] = 0xA8;
        assert!(parse(&wrong_header).is_none());
    }

    fn close(a: i32, b: i32) -> bool {
        (a - b).abs() <= 2
    }

    #[test]
    fn aim_setpoint_cardinals() {
        // due North, level
        let s = aim_setpoint(Vector3::new(0.0, 10.0, 0.0));
        assert!(close(s.yaw_cd as i32, 0) && close(s.pitch_cd as i32, 0));
        // due East → +90°
        assert!(close(aim_setpoint(Vector3::new(10.0, 0.0, 0.0)).yaw_cd as i32, 9000));
        // due West → −90°
        assert!(close(aim_setpoint(Vector3::new(-10.0, 0.0, 0.0)).yaw_cd as i32, -9000));
        // straight up → pitch +90°
        assert!(close(aim_setpoint(Vector3::new(0.0, 0.0, 10.0)).pitch_cd as i32, 9000));
        // 45° up, to the North
        let s = aim_setpoint(Vector3::new(0.0, 10.0, 10.0));
        assert!(close(s.pitch_cd as i32, 4500) && close(s.yaw_cd as i32, 0));
    }

    #[test]
    fn carrier_attitude_identity_faces_east() {
        // identity: body FLU axes == world ENU axes → forward(x)=East, level.
        let (r, p, y) = carrier_attitude(&UnitQuaternion::identity());
        assert!(close(r as i32, 0) && close(p as i32, 0));
        assert!(close(y as i32, 9000)); // East = compass 90°
    }

    #[test]
    fn carrier_attitude_yaw_to_north() {
        // +90° about world up takes body-forward from East to North.
        let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f32::consts::FRAC_PI_2);
        let (r, p, y) = carrier_attitude(&q);
        assert!(close(r as i32, 0) && close(p as i32, 0));
        assert!(close(y as i32, 0)); // North = compass 0°
    }

    #[test]
    fn carrier_attitude_pitch_nose_up() {
        // rotate body-forward up by 30° (about world −Y) → +30° nose-up.
        let q = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, -1.0, 0.0)),
            core::f32::consts::FRAC_PI_6,
        );
        let (_r, p, _y) = carrier_attitude(&q);
        assert!(close(p as i32, 3000), "pitch_cd={}", p);
    }

    #[test]
    fn heartbeat_is_passive_and_matches_pdf_vector() {
        let mut b = [0u8; FRAME_LEN];
        assert_eq!(encode_heartbeat(&mut b), FRAME_LEN);
        assert_eq!(b[11], 0x00); // control invalid → no motion
        assert_eq!(b[30], 0x01); // requests the telemetry sub-frame
        assert_eq!(b[69], cmd::MAINTAIN);
        // PDF App.5 "空命令" CRC.
        assert_eq!(((b[FRAME_LEN - 2] as u16) << 8) | b[FRAME_LEN - 1] as u16, 0xFD13);
    }

    #[test]
    fn scanner_extracts_valid_frame_from_noise() {
        // Build a valid GCU→host frame.
        let mut f = [0u8; FRAME_LEN];
        f[0] = 0x8A;
        f[1] = 0x5E;
        f[2..4].copy_from_slice(&(FRAME_LEN as u16).to_le_bytes());
        f[4] = VERSION;
        f[5] = cmd::EULER; // work_mode
        let crc = crc16(&f[..FRAME_LEN - 2]);
        f[FRAME_LEN - 2] = (crc >> 8) as u8;
        f[FRAME_LEN - 1] = (crc & 0xFF) as u8;

        let mut s = FrameScanner::new();
        s.push(&[0x11, 0x22, 0x8A]); // leading garbage (incl. a false header byte)
        s.push(&f[..20]); // frame split across pushes
        assert!(s.pull().is_none()); // incomplete so far
        s.push(&f[20..]);
        let t = s.pull().expect("a complete frame");
        assert_eq!(t.work_mode, cmd::EULER);
        assert!(s.pull().is_none());
    }
}
