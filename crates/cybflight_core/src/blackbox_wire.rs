//! Compact wire formats for the blackbox's high-rate topics.
//!
//! The blackbox originally encoded every topic as a CBOR map with
//! full string keys. Self-describing, but on a topic emitted at the
//! IMU rate the keys dominate: `/imu1`'s old map spent 42 of its 89
//! payload bytes re-transmitting `"timestamp_ns"` / `"accel_m_s2"` /
//! `"gyro_rad_s"` / `"temp_c"` up to 8000 times per second — ~330 KB/s
//! of pure key repetition at 8 kHz. The low-rate topics (≤150 Hz)
//! keep their maps; readability there is worth a few KB/s.
//!
//! This module holds the **positional-array** replacements for the
//! topics where the byte rate actually matters (`/imu1`, `/imu1_raw`,
//! `/odometry`, `/health`). `/health` is only 20 Hz, but it is the
//! widest record in the file — 18 string keys made up ~80 % of its
//! 370 B on disk, and at the `Sysid` tier's ~4 KiB/s of headroom that
//! was the difference between affording new fields and not. Element order is fixed and documented both here and
//! in each topic's JSON Schema (`prefixItems`); the schema travels
//! inside every MCAP file, so files stay self-describing.
//!
//! ## Why this lives in `cybflight-core`
//!
//! The firmware crate only compiles for thumbv7em, so its
//! `#[cfg(test)]` modules never run anywhere. These functions are
//! pure `&[f32] → bytes` transforms — putting them here means the
//! golden-byte and worst-case-size tests below actually execute on
//! every `just test`.
//!
//! ## Versioning
//!
//! Changing an element order or adding a field is a wire break. Bump
//! the topic's `SCHEMA_NAME` (e.g. `Imu.v2` → `Imu.v3`) in the same
//! change so a reader can key its decoder on the schema name, and
//! update the golden tests here — they exist precisely to make an
//! accidental layout change fail the build.

use crate::cbor::{CborWriter, Result};

/// Emit-every-Nth rate divider with loss-aware sequence accounting.
///
/// The recorder's `blackbox_rate_div` keeps 1 of every `every`
/// messages on the high-rate topics. Skips must not look like drops in
/// the MCAP sequence numbers, and real drops (`WaitResult::Lagged(n)`,
/// `n` *raw* messages) must widen the sequence hole by the number of
/// **records that would have been emitted**, not by `n` — otherwise a
/// stall that lost 8 raw samples at `every = 4` shows an 8-wide hole
/// for the ~2 records it actually cost, and the divider's phase drifts
/// because the lost messages never ticked it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RateDiv {
    every: u32,
    phase: u32,
}

impl RateDiv {
    /// `every == 0` is treated as 1 (emit everything).
    pub const fn new(every: u32) -> Self {
        Self {
            every: if every == 0 { 1 } else { every },
            phase: 0,
        }
    }

    /// One message received. `true` when it is the one to emit.
    pub fn tick(&mut self) -> bool {
        self.phase += 1;
        if self.phase >= self.every {
            self.phase = 0;
            true
        } else {
            false
        }
    }

    /// `n` raw messages were lost before the next one. Advances the
    /// phase as if they had been received and returns how many records
    /// the loss would have emitted — the width of the sequence hole.
    pub fn lag(&mut self, n: u32) -> u32 {
        let every = self.every as u64;
        let total = self.phase as u64 + n as u64;
        self.phase = (total % every) as u32;
        u32::try_from(total / every).unwrap_or(u32::MAX)
    }
}

/// Encoded size of one `/imu1` (or `/imu1_raw`) record with a
/// worst-case (9-byte) timestamp head: `array(7)` + u64 + 6 × f32.
///
/// f32 always encodes as exactly 5 bytes, so the timestamp head is
/// the only variable-length element and this bound is tight.
pub const IMU_MAX_ENCODED: usize = 1 + 9 + 6 * 5;

/// Encode one IMU sample as a flat positional array:
///
/// `[timestamp_ns, ax, ay, az, gx, gy, gz]`
///
/// accel in m/s², gyro in rad/s — same units and axis order as the
/// old map form. `temp_c` is deliberately absent: it changes at ~1 Hz
/// and cost 12 bytes per sample in the map form (~96 KB/s at 8 kHz);
/// the recorder now reports it in the 20 Hz `/health` record instead.
pub fn encode_imu(scratch: &mut [u8], t_ns: u64, accel: &[f32; 3], gyro: &[f32; 3]) -> Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.array(7)?;
    w.u64(t_ns)?;
    for v in accel {
        w.f32(*v)?;
    }
    for v in gyro {
        w.f32(*v)?;
    }
    Ok(w.pos())
}

/// Encoded size of one `/odometry` record with a worst-case
/// timestamp head: `array(14)` + u64 + 13 × f32.
pub const ODOMETRY_MAX_ENCODED: usize = 1 + 9 + 13 * 5;

/// Encode one odometry sample as a flat positional array:
///
/// `[timestamp_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]`
///
/// position in m, orientation as a `[w, i, j, k]` unit quaternion,
/// linear velocity in m/s, angular velocity in rad/s — same units,
/// frames, and quaternion component order as the old map form.
pub fn encode_odometry(
    scratch: &mut [u8],
    t_ns: u64,
    position: &[f32; 3],
    orientation_wijk: &[f32; 4],
    linear_vel: &[f32; 3],
    angular_vel: &[f32; 3],
) -> Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.array(14)?;
    w.u64(t_ns)?;
    for v in position {
        w.f32(*v)?;
    }
    for v in orientation_wijk {
        w.f32(*v)?;
    }
    for v in linear_vel {
        w.f32(*v)?;
    }
    for v in angular_vel {
        w.f32(*v)?;
    }
    Ok(w.pos())
}

/// Latest RC link statistics as `/health` records them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RcLinkSample {
    /// Link quality, percent of packets received [0..100].
    pub quality: u8,
    /// Uplink RSSI in dBm (negative).
    pub rssi_dbm: i16,
    /// Age of the statistics frame at record time, ms.
    pub age_ms: u32,
}

/// One `/health` sample, as plain values. The firmware gathers these
/// from its atomics; this module only fixes their wire order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HealthRecord {
    pub t_ns: u64,
    pub imu1_temp_c: f32,
    pub failsafe_active: bool,
    pub estimator_ready: bool,
    pub eskf_degraded: bool,
    pub eskf_severe_fault: bool,
    pub eskf_faults: u32,
    pub attitude_health: u8,
    pub nan_resets: u32,
    pub gate_rejects_pos: u32,
    pub gate_rejects_vel: u32,
    pub gate_rejects_att: u32,
    pub last_nis_pos: f32,
    pub last_nis_vel: f32,
    pub last_nis_att: f32,
    /// 0 = never accepted.
    pub last_pos_update_ns: u64,
    pub last_vel_update_ns: u64,
    pub last_att_update_ns: u64,
    /// Longest INDI iteration since the previous record, µs. 0 = no
    /// iteration ran in the window — itself a stall signal.
    pub indi_step_max_us: u32,
    /// Longest gap between INDI iteration starts since the previous
    /// record, µs. Compare against the nominal control period.
    pub indi_period_max_us: u32,
    /// `None` = no link-statistics frame since boot (encoded as three
    /// CBOR `null`s, so a dead link's genuine `quality = 0` stays
    /// distinguishable from "never heard").
    pub rc_link: Option<RcLinkSample>,
}

/// Element count of the `/health` array.
pub const HEALTH_FIELDS: u64 = 23;

/// Encoded size of one `/health` record with every variable-length
/// element at its widest head: `array(23)` + 4 × u64 (9) + 4 × f32 (5)
/// + 4 × bool (1) + 7 × u32 (5: eskf_faults, nan_resets, 3 × gate_rejects,
/// 2 × indi timing) + u8 attitude_health (2) + link: u8 (2) + i16 (3) +
/// u32 (5).
pub const HEALTH_MAX_ENCODED: usize = 1 + 4 * 9 + 4 * 5 + 4 + 7 * 5 + 2 + 2 + 3 + 5;

/// Encode one health sample as a flat positional array:
///
/// `[timestamp_ns, imu1_temp_c, failsafe_active, estimator_ready,
///   eskf_degraded, eskf_severe_fault, eskf_faults, attitude_health,
///   nan_resets, gate_rejects_pos, gate_rejects_vel, gate_rejects_att,
///   last_nis_pos, last_nis_vel, last_nis_att, last_pos_update_ns,
///   last_vel_update_ns, last_att_update_ns, indi_step_max_us,
///   indi_period_max_us, rc_link_quality, rc_rssi_dbm, rc_link_age_ms]`
///
/// `timestamp_ns` stays first: readers (`analysis/read_mcap.py`) take
/// element 0 as the record time when its `prefixItems` title says so.
pub fn encode_health(scratch: &mut [u8], r: &HealthRecord) -> Result<usize> {
    let mut w = CborWriter::new(scratch);
    w.array(HEALTH_FIELDS)?;
    w.u64(r.t_ns)?;
    w.f32(r.imu1_temp_c)?;
    w.bool(r.failsafe_active)?;
    w.bool(r.estimator_ready)?;
    w.bool(r.eskf_degraded)?;
    w.bool(r.eskf_severe_fault)?;
    w.u64(r.eskf_faults as u64)?;
    w.u64(r.attitude_health as u64)?;
    w.u64(r.nan_resets as u64)?;
    w.u64(r.gate_rejects_pos as u64)?;
    w.u64(r.gate_rejects_vel as u64)?;
    w.u64(r.gate_rejects_att as u64)?;
    w.f32(r.last_nis_pos)?;
    w.f32(r.last_nis_vel)?;
    w.f32(r.last_nis_att)?;
    w.u64(r.last_pos_update_ns)?;
    w.u64(r.last_vel_update_ns)?;
    w.u64(r.last_att_update_ns)?;
    w.u64(r.indi_step_max_us as u64)?;
    w.u64(r.indi_period_max_us as u64)?;
    match r.rc_link {
        Some(l) => {
            w.u64(l.quality as u64)?;
            w.i64(l.rssi_dbm as i64)?;
            w.u64(l.age_ms as u64)?;
        }
        None => {
            w.null()?;
            w.null()?;
            w.null()?;
        }
    }
    Ok(w.pos())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plausible mid-flight health sample: 2 min after boot, ESKF
    /// healthy, INDI at 500 Hz, CRSF link at full quality.
    fn steady_health() -> HealthRecord {
        HealthRecord {
            t_ns: 120_000_000_000,
            imu1_temp_c: 35.0,
            failsafe_active: false,
            estimator_ready: true,
            eskf_degraded: false,
            eskf_severe_fault: false,
            eskf_faults: 0,
            attitude_health: 7,
            nan_resets: 0,
            gate_rejects_pos: 0,
            gate_rejects_vel: 0,
            gate_rejects_att: 0,
            last_nis_pos: 0.8,
            last_nis_vel: 1.1,
            last_nis_att: 0.9,
            last_pos_update_ns: 119_990_000_000,
            last_vel_update_ns: 119_990_000_000,
            last_att_update_ns: 119_990_000_000,
            indi_step_max_us: 40,
            indi_period_max_us: 2_100,
            rc_link: Some(RcLinkSample { quality: 100, rssi_dbm: -60, age_ms: 80 }),
        }
    }

    /// The recorder's per-message scratch buffer size. Mirrored here
    /// (rather than imported — the firmware crate can't be a dev-dep
    /// of core) so the worst-case asserts below break if a wire
    /// format outgrows what the recorder can hold.
    const RECORDER_SCRATCH: usize = 512;

    #[test]
    fn rate_div_emits_every_nth() {
        let mut d = RateDiv::new(4);
        let pattern: Vec<bool> = (0..8).map(|_| d.tick()).collect();
        assert_eq!(pattern, [false, false, false, true, false, false, false, true]);
        let mut one = RateDiv::new(1);
        assert!(one.tick() && one.tick());
        let mut zero = RateDiv::new(0);
        assert!(zero.tick());
    }

    #[test]
    fn rate_div_lag_counts_records_not_messages() {
        // At phase 0, losing 8 raw messages at ÷4 costs exactly 2 records
        // and leaves the phase where it was.
        let mut d = RateDiv::new(4);
        assert_eq!(d.lag(8), 2);
        assert_eq!(d, RateDiv::new(4));
        // At phase 3 (next message would emit), losing 8 covers the emits
        // at positions 4 and 8; the following message (position 12) emits.
        let mut d = RateDiv::new(4);
        d.tick();
        d.tick();
        d.tick();
        assert_eq!(d.lag(8), 2);
        assert!(d.tick());
        // Sub-period loss: no record lost, phase advanced.
        let mut d = RateDiv::new(4);
        assert_eq!(d.lag(2), 0);
        assert!(!d.tick());
        assert!(d.tick());
        // ÷1: every lost message is a lost record.
        let mut d = RateDiv::new(1);
        assert_eq!(d.lag(7), 7);
    }

    #[test]
    fn imu_worst_case_size_is_tight_and_fits() {
        let mut buf = [0u8; 64];
        let n = encode_imu(&mut buf, u64::MAX, &[1.0; 3], &[1.0; 3]).unwrap();
        assert_eq!(n, IMU_MAX_ENCODED);
        assert!(IMU_MAX_ENCODED <= RECORDER_SCRATCH);
    }

    #[test]
    fn odometry_worst_case_size_is_tight_and_fits() {
        let mut buf = [0u8; 128];
        let n = encode_odometry(
            &mut buf,
            u64::MAX,
            &[1.0; 3],
            &[1.0; 4],
            &[1.0; 3],
            &[1.0; 3],
        )
        .unwrap();
        assert_eq!(n, ODOMETRY_MAX_ENCODED);
        assert!(ODOMETRY_MAX_ENCODED <= RECORDER_SCRATCH);
    }

    /// Golden bytes: the exact wire image of a known IMU sample.
    /// Guards element order — a swapped accel/gyro or a reordered
    /// quaternion decodes "successfully" into garbage, so only a
    /// byte-exact test catches it.
    #[test]
    fn imu_golden_bytes() {
        let mut buf = [0u8; 64];
        let n = encode_imu(&mut buf, 1000, &[1.5, 0.0, -2.0], &[0.25, -0.5, 1.0]).unwrap();
        #[rustfmt::skip]
        let expect: &[u8] = &[
            0x87,                               // array(7)
            0x19, 0x03, 0xe8,                   // 1000
            0xfa, 0x3f, 0xc0, 0x00, 0x00,       // 1.5
            0xfa, 0x00, 0x00, 0x00, 0x00,       // 0.0
            0xfa, 0xc0, 0x00, 0x00, 0x00,       // -2.0
            0xfa, 0x3e, 0x80, 0x00, 0x00,       // 0.25
            0xfa, 0xbf, 0x00, 0x00, 0x00,       // -0.5
            0xfa, 0x3f, 0x80, 0x00, 0x00,       // 1.0
        ];
        assert_eq!(&buf[..n], expect);
    }

    /// Golden bytes for odometry: element count, order, and the
    /// quaternion's w-first convention.
    #[test]
    fn odometry_golden_bytes() {
        let mut buf = [0u8; 128];
        let n = encode_odometry(
            &mut buf,
            5,
            &[1.0, 2.0, 3.0],
            &[1.0, 0.0, 0.0, 0.0], // identity, w first
            &[0.0; 3],
            &[0.0; 3],
        )
        .unwrap();
        assert_eq!(n, 1 + 1 + 13 * 5); // small timestamp head = 1 byte
        assert_eq!(buf[0], 0x8e); // array(14)
        assert_eq!(buf[1], 0x05); // t = 5
        // position 1.0, 2.0, 3.0
        assert_eq!(&buf[2..7], &[0xfa, 0x3f, 0x80, 0x00, 0x00]);
        assert_eq!(&buf[7..12], &[0xfa, 0x40, 0x00, 0x00, 0x00]);
        assert_eq!(&buf[12..17], &[0xfa, 0x40, 0x40, 0x00, 0x00]);
        // qw = 1.0 comes before qx/qy/qz = 0.0
        assert_eq!(&buf[17..22], &[0xfa, 0x3f, 0x80, 0x00, 0x00]);
        assert_eq!(&buf[22..27], &[0xfa, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn health_worst_case_size_is_tight_and_fits() {
        let mut buf = [0u8; 256];
        let r = HealthRecord {
            t_ns: u64::MAX,
            eskf_faults: u32::MAX,
            attitude_health: u8::MAX,
            nan_resets: u32::MAX,
            gate_rejects_pos: u32::MAX,
            gate_rejects_vel: u32::MAX,
            gate_rejects_att: u32::MAX,
            last_pos_update_ns: u64::MAX,
            last_vel_update_ns: u64::MAX,
            last_att_update_ns: u64::MAX,
            indi_step_max_us: u32::MAX,
            indi_period_max_us: u32::MAX,
            rc_link: Some(RcLinkSample { quality: u8::MAX, rssi_dbm: i16::MIN, age_ms: u32::MAX }),
            ..steady_health()
        };
        let n = encode_health(&mut buf, &r).unwrap();
        assert_eq!(n, HEALTH_MAX_ENCODED);
        assert!(HEALTH_MAX_ENCODED <= RECORDER_SCRATCH);
    }

    /// Golden bytes: element order, the bool/int/float mix, the signed
    /// RSSI, and the tail layout for both link states. A swapped pair
    /// of adjacent u32 counters decodes "successfully" into the wrong
    /// columns, so only a byte-exact test catches it.
    #[test]
    fn health_golden_bytes() {
        let r = HealthRecord {
            t_ns: 5,
            imu1_temp_c: 1.0,
            failsafe_active: true,
            estimator_ready: false,
            eskf_degraded: true,
            eskf_severe_fault: false,
            eskf_faults: 2,
            attitude_health: 7,
            nan_resets: 1,
            gate_rejects_pos: 3,
            gate_rejects_vel: 4,
            gate_rejects_att: 5,
            last_nis_pos: 0.5,
            last_nis_vel: 1.0,
            last_nis_att: 2.0,
            last_pos_update_ns: 0,
            last_vel_update_ns: 6,
            last_att_update_ns: 7,
            indi_step_max_us: 40,
            indi_period_max_us: 300,
            rc_link: Some(RcLinkSample { quality: 99, rssi_dbm: -60, age_ms: 12 }),
        };
        let mut buf = [0u8; 128];
        let n = encode_health(&mut buf, &r).unwrap();
        #[rustfmt::skip]
        let expect: &[u8] = &[
            0x97,                           // array(23)
            0x05,                           // timestamp_ns
            0xfa, 0x3f, 0x80, 0x00, 0x00,   // imu1_temp_c 1.0
            0xf5, 0xf4, 0xf5, 0xf4,         // failsafe, est_ready, degraded, severe
            0x02, 0x07, 0x01,               // eskf_faults, attitude_health, nan_resets
            0x03, 0x04, 0x05,               // gate_rejects pos, vel, att
            0xfa, 0x3f, 0x00, 0x00, 0x00,   // last_nis_pos 0.5
            0xfa, 0x3f, 0x80, 0x00, 0x00,   // last_nis_vel 1.0
            0xfa, 0x40, 0x00, 0x00, 0x00,   // last_nis_att 2.0
            0x00, 0x06, 0x07,               // last_{pos,vel,att}_update_ns
            0x18, 0x28,                     // indi_step_max_us 40
            0x19, 0x01, 0x2c,               // indi_period_max_us 300
            0x18, 0x63,                     // rc_link_quality 99
            0x38, 0x3b,                     // rc_rssi_dbm -60 (= -1 - 59)
            0x0c,                           // rc_link_age_ms 12
        ];
        assert_eq!(&buf[..n], expect);

        // Never-heard link: same element count, three nulls in the tail.
        let n2 = encode_health(&mut buf, &HealthRecord { rc_link: None, ..r }).unwrap();
        assert_eq!(&buf[n2 - 3..n2], &[0xf6, 0xf6, 0xf6]);
        assert_eq!(buf[0], 0x97);
    }

    /// The byte-rate claim in `record_set.rs`'s tier table rests on
    /// these sizes; pin the typical-case (small timestamp is not
    /// typical in flight — timestamps are ns since boot, so the
    /// 9-byte head is the steady state within seconds of boot).
    #[test]
    fn steady_state_sizes() {
        let t = 120_000_000_000u64; // 2 min after boot, ns
        let mut buf = [0u8; 128];
        let n = encode_imu(&mut buf, t, &[0.1; 3], &[0.1; 3]).unwrap();
        assert_eq!(n, 40);
        let n = encode_odometry(&mut buf, t, &[0.1; 3], &[0.1; 4], &[0.1; 3], &[0.1; 3]).unwrap();
        assert_eq!(n, 75);
        // /health: 78 B payload + 22 B MCAP header = the 100 B that
        // `estimated_bytes_per_s` budgets (was 370 B as a keyed map).
        let n = encode_health(&mut buf, &steady_health()).unwrap();
        assert_eq!(n, 78);
    }
}
