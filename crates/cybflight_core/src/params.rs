//! Persistent vehicle parameter container with manual serialization.
//!
//! On-flash layout (little-endian, 160 bytes, aligned to 32-byte flash words):
//!
//! ```text
//! [0x00] magic:   u32 = 0x43594250 ("CYBP")
//! [0x04] version: u32 = 1
//! [0x08] length:  u32 = PAYLOAD_SIZE (136)
//! [0x0C] crc32:   u32 (over payload only)
//! [0x10] payload: 136 bytes
//! [0x98] padding: 8 bytes (zeros)
//! ```

use crate::mixer::{MotorParams, RigidBodyParams, SpinDir};

const MAGIC: u32 = 0x4359_4250; // "CYBP"
const VERSION: u32 = 3;
const HEADER_SIZE: usize = 16; // magic + version + length + crc
/// Body: mass(4) + inertia(9*4=36) = 40 bytes
/// Motor: px(4) + py(4) + spin_dir(4) + max_thrust(4) + torque_coeff(4) = 20 bytes each
/// Control: pos_kp(12) + pos_kd(12) + att_k_rate(12) + rate_roll(12) + rate_pitch(12) + rate_yaw(12) = 72 bytes
/// Total payload: 40 + 4*20 + 72 = 192 bytes
const PAYLOAD_SIZE: usize = 192;
/// Padded to 32-byte flash word boundary: ceil((16+192)/32)*32 = 224
pub const PADDED_SIZE: usize = 224;

/// PID gain triplet.
#[derive(Clone, Copy, Debug)]
pub struct PidGains {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
}

/// Control gains for position, attitude, and rate loops.
///
/// All vector gains are dimension-major: `[roll/x, pitch/y, yaw/z]`.
#[derive(Clone, Debug)]
pub struct ControlGains {
    /// Position proportional gains [x, y, z].
    pub pos_kp: [f32; 3],
    /// Position derivative (velocity) gains [x, y, z].
    pub pos_kd: [f32; 3],
    /// Attitude error to body-rate gains [roll, pitch, yaw].
    pub att_k_rate: [f32; 3],
    /// Rate PID proportional gains [roll, pitch, yaw].
    pub rate_kp: [f32; 3],
    /// Rate PID integral gains [roll, pitch, yaw].
    pub rate_ki: [f32; 3],
    /// Rate PID derivative gains [roll, pitch, yaw].
    pub rate_kd: [f32; 3],
}

impl ControlGains {
    /// Extract per-axis PID gains for the rate controller.
    pub fn rate_pid(&self, axis: usize) -> PidGains {
        PidGains {
            kp: self.rate_kp[axis],
            ki: self.rate_ki[axis],
            kd: self.rate_kd[axis],
        }
    }
}

/// Full vehicle parameter set.
#[derive(Clone, Debug)]
pub struct VehicleParams {
    pub body: RigidBodyParams,
    pub motors: [MotorParams; 4],
    pub control: ControlGains,
}

impl VehicleParams {
    /// Serialize to a flash-ready buffer with header and CRC.
    pub fn to_bytes(&self) -> [u8; PADDED_SIZE] {
        let mut buf = [0u8; PADDED_SIZE];
        // Write payload first (at offset HEADER_SIZE)
        let mut off = HEADER_SIZE;
        off = put_f32(&mut buf, off, self.body.mass_kg);
        for &v in &self.body.inertia_kg_m2 {
            off = put_f32(&mut buf, off, v);
        }
        for m in &self.motors {
            off = put_f32(&mut buf, off, m.position_m[0]);
            off = put_f32(&mut buf, off, m.position_m[1]);
            off = put_u32(
                &mut buf,
                off,
                match m.spin_dir {
                    SpinDir::Cw => 1,
                    SpinDir::Ccw => 0,
                },
            );
            off = put_f32(&mut buf, off, m.max_thrust_n);
            off = put_f32(&mut buf, off, m.torque_coeff_m);
        }
        for &v in &self.control.pos_kp {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.pos_kd {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.att_k_rate {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.rate_kp {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.rate_ki {
            off = put_f32(&mut buf, off, v);
        }
        for &v in &self.control.rate_kd {
            off = put_f32(&mut buf, off, v);
        }
        debug_assert_eq!(off - HEADER_SIZE, PAYLOAD_SIZE);

        // Header
        let payload = &buf[HEADER_SIZE..HEADER_SIZE + PAYLOAD_SIZE];
        let crc = crc32fast::hash(payload);
        put_u32(&mut buf, 0, MAGIC);
        put_u32(&mut buf, 4, VERSION);
        put_u32(&mut buf, 8, PAYLOAD_SIZE as u32);
        put_u32(&mut buf, 12, crc);
        buf
    }

    /// Deserialize from a flash buffer. Returns `None` if magic, version, or CRC mismatch.
    pub fn from_bytes(buf: &[u8; PADDED_SIZE]) -> Option<Self> {
        let magic = get_u32(buf, 0);
        let version = get_u32(buf, 4);
        let length = get_u32(buf, 8);
        let stored_crc = get_u32(buf, 12);

        if magic != MAGIC || version != VERSION || length as usize != PAYLOAD_SIZE {
            return None;
        }

        let payload = &buf[HEADER_SIZE..HEADER_SIZE + PAYLOAD_SIZE];
        if crc32fast::hash(payload) != stored_crc {
            return None;
        }

        let mut off = HEADER_SIZE;
        let mass_kg = get_f32(buf, off);
        off += 4;
        let mut inertia_kg_m2 = [0.0f32; 9];
        for slot in &mut inertia_kg_m2 {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let body = RigidBodyParams {
            mass_kg,
            inertia_kg_m2,
        };

        let mut motors = [MotorParams {
            position_m: [0.0; 2],
            spin_dir: SpinDir::Cw,
            max_thrust_n: 0.0,
            torque_coeff_m: 0.0,
        }; 4];
        for m in &mut motors {
            let px = get_f32(buf, off);
            off += 4;
            let py = get_f32(buf, off);
            off += 4;
            let dir_val = get_u32(buf, off);
            off += 4;
            let max_thrust = get_f32(buf, off);
            off += 4;
            let torque_coeff = get_f32(buf, off);
            off += 4;
            m.position_m = [px, py];
            m.spin_dir = if dir_val == 1 {
                SpinDir::Cw
            } else {
                SpinDir::Ccw
            };
            m.max_thrust_n = max_thrust;
            m.torque_coeff_m = torque_coeff;
        }

        let mut pos_kp = [0.0f32; 3];
        for slot in &mut pos_kp {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut pos_kd = [0.0f32; 3];
        for slot in &mut pos_kd {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut att_k_rate = [0.0f32; 3];
        for slot in &mut att_k_rate {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut rate_kp = [0.0f32; 3];
        for slot in &mut rate_kp {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut rate_ki = [0.0f32; 3];
        for slot in &mut rate_ki {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let mut rate_kd = [0.0f32; 3];
        for slot in &mut rate_kd {
            *slot = get_f32(buf, off);
            off += 4;
        }
        let control = ControlGains {
            pos_kp,
            pos_kd,
            att_k_rate,
            rate_kp,
            rate_ki,
            rate_kd,
        };

        Some(VehicleParams {
            body,
            motors,
            control,
        })
    }

    /// Get a parameter value by key.
    pub fn get(&self, key: ParamKey) -> f32 {
        match key {
            ParamKey::Mass => self.body.mass_kg,
            ParamKey::Ixx => self.body.inertia_kg_m2[0],
            ParamKey::Ixy => self.body.inertia_kg_m2[1],
            ParamKey::Ixz => self.body.inertia_kg_m2[2],
            ParamKey::Iyx => self.body.inertia_kg_m2[3],
            ParamKey::Iyy => self.body.inertia_kg_m2[4],
            ParamKey::Iyz => self.body.inertia_kg_m2[5],
            ParamKey::Izx => self.body.inertia_kg_m2[6],
            ParamKey::Izy => self.body.inertia_kg_m2[7],
            ParamKey::Izz => self.body.inertia_kg_m2[8],
            ParamKey::M0Px => self.motors[0].position_m[0],
            ParamKey::M0Py => self.motors[0].position_m[1],
            ParamKey::M0Spin => self.motors[0].spin_dir as i32 as f32,
            ParamKey::M0Thrust => self.motors[0].max_thrust_n,
            ParamKey::M0Torque => self.motors[0].torque_coeff_m,
            ParamKey::M1Px => self.motors[1].position_m[0],
            ParamKey::M1Py => self.motors[1].position_m[1],
            ParamKey::M1Spin => self.motors[1].spin_dir as i32 as f32,
            ParamKey::M1Thrust => self.motors[1].max_thrust_n,
            ParamKey::M1Torque => self.motors[1].torque_coeff_m,
            ParamKey::M2Px => self.motors[2].position_m[0],
            ParamKey::M2Py => self.motors[2].position_m[1],
            ParamKey::M2Spin => self.motors[2].spin_dir as i32 as f32,
            ParamKey::M2Thrust => self.motors[2].max_thrust_n,
            ParamKey::M2Torque => self.motors[2].torque_coeff_m,
            ParamKey::M3Px => self.motors[3].position_m[0],
            ParamKey::M3Py => self.motors[3].position_m[1],
            ParamKey::M3Spin => self.motors[3].spin_dir as i32 as f32,
            ParamKey::M3Thrust => self.motors[3].max_thrust_n,
            ParamKey::M3Torque => self.motors[3].torque_coeff_m,
            ParamKey::PosKpX => self.control.pos_kp[0],
            ParamKey::PosKpY => self.control.pos_kp[1],
            ParamKey::PosKpZ => self.control.pos_kp[2],
            ParamKey::PosKdX => self.control.pos_kd[0],
            ParamKey::PosKdY => self.control.pos_kd[1],
            ParamKey::PosKdZ => self.control.pos_kd[2],
            ParamKey::AttKrX => self.control.att_k_rate[0],
            ParamKey::AttKrY => self.control.att_k_rate[1],
            ParamKey::AttKrZ => self.control.att_k_rate[2],
            ParamKey::RateKpR => self.control.rate_kp[0],
            ParamKey::RateKpP => self.control.rate_kp[1],
            ParamKey::RateKpY => self.control.rate_kp[2],
            ParamKey::RateKiR => self.control.rate_ki[0],
            ParamKey::RateKiP => self.control.rate_ki[1],
            ParamKey::RateKiY => self.control.rate_ki[2],
            ParamKey::RateKdR => self.control.rate_kd[0],
            ParamKey::RateKdP => self.control.rate_kd[1],
            ParamKey::RateKdY => self.control.rate_kd[2],
        }
    }

    /// Set a parameter value by key.
    pub fn set(&mut self, key: ParamKey, val: f32) {
        match key {
            ParamKey::Mass => self.body.mass_kg = val,
            ParamKey::Ixx => self.body.inertia_kg_m2[0] = val,
            ParamKey::Ixy => self.body.inertia_kg_m2[1] = val,
            ParamKey::Ixz => self.body.inertia_kg_m2[2] = val,
            ParamKey::Iyx => self.body.inertia_kg_m2[3] = val,
            ParamKey::Iyy => self.body.inertia_kg_m2[4] = val,
            ParamKey::Iyz => self.body.inertia_kg_m2[5] = val,
            ParamKey::Izx => self.body.inertia_kg_m2[6] = val,
            ParamKey::Izy => self.body.inertia_kg_m2[7] = val,
            ParamKey::Izz => self.body.inertia_kg_m2[8] = val,
            ParamKey::M0Px => self.motors[0].position_m[0] = val,
            ParamKey::M0Py => self.motors[0].position_m[1] = val,
            ParamKey::M0Spin => {
                self.motors[0].spin_dir = if val > 0.0 {
                    SpinDir::Cw
                } else {
                    SpinDir::Ccw
                }
            }
            ParamKey::M0Thrust => self.motors[0].max_thrust_n = val,
            ParamKey::M0Torque => self.motors[0].torque_coeff_m = val,
            ParamKey::M1Px => self.motors[1].position_m[0] = val,
            ParamKey::M1Py => self.motors[1].position_m[1] = val,
            ParamKey::M1Spin => {
                self.motors[1].spin_dir = if val > 0.0 {
                    SpinDir::Cw
                } else {
                    SpinDir::Ccw
                }
            }
            ParamKey::M1Thrust => self.motors[1].max_thrust_n = val,
            ParamKey::M1Torque => self.motors[1].torque_coeff_m = val,
            ParamKey::M2Px => self.motors[2].position_m[0] = val,
            ParamKey::M2Py => self.motors[2].position_m[1] = val,
            ParamKey::M2Spin => {
                self.motors[2].spin_dir = if val > 0.0 {
                    SpinDir::Cw
                } else {
                    SpinDir::Ccw
                }
            }
            ParamKey::M2Thrust => self.motors[2].max_thrust_n = val,
            ParamKey::M2Torque => self.motors[2].torque_coeff_m = val,
            ParamKey::M3Px => self.motors[3].position_m[0] = val,
            ParamKey::M3Py => self.motors[3].position_m[1] = val,
            ParamKey::M3Spin => {
                self.motors[3].spin_dir = if val > 0.0 {
                    SpinDir::Cw
                } else {
                    SpinDir::Ccw
                }
            }
            ParamKey::M3Thrust => self.motors[3].max_thrust_n = val,
            ParamKey::M3Torque => self.motors[3].torque_coeff_m = val,
            ParamKey::PosKpX => self.control.pos_kp[0] = val,
            ParamKey::PosKpY => self.control.pos_kp[1] = val,
            ParamKey::PosKpZ => self.control.pos_kp[2] = val,
            ParamKey::PosKdX => self.control.pos_kd[0] = val,
            ParamKey::PosKdY => self.control.pos_kd[1] = val,
            ParamKey::PosKdZ => self.control.pos_kd[2] = val,
            ParamKey::AttKrX => self.control.att_k_rate[0] = val,
            ParamKey::AttKrY => self.control.att_k_rate[1] = val,
            ParamKey::AttKrZ => self.control.att_k_rate[2] = val,
            ParamKey::RateKpR => self.control.rate_kp[0] = val,
            ParamKey::RateKpP => self.control.rate_kp[1] = val,
            ParamKey::RateKpY => self.control.rate_kp[2] = val,
            ParamKey::RateKiR => self.control.rate_ki[0] = val,
            ParamKey::RateKiP => self.control.rate_ki[1] = val,
            ParamKey::RateKiY => self.control.rate_ki[2] = val,
            ParamKey::RateKdR => self.control.rate_kd[0] = val,
            ParamKey::RateKdP => self.control.rate_kd[1] = val,
            ParamKey::RateKdY => self.control.rate_kd[2] = val,
        }
    }
}

/// Named parameter keys for shell access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKey {
    Mass,
    Ixx,
    Ixy,
    Ixz,
    Iyx,
    Iyy,
    Iyz,
    Izx,
    Izy,
    Izz,
    M0Px,
    M0Py,
    M0Spin,
    M0Thrust,
    M0Torque,
    M1Px,
    M1Py,
    M1Spin,
    M1Thrust,
    M1Torque,
    M2Px,
    M2Py,
    M2Spin,
    M2Thrust,
    M2Torque,
    M3Px,
    M3Py,
    M3Spin,
    M3Thrust,
    M3Torque,
    // Position control gains
    PosKpX,
    PosKpY,
    PosKpZ,
    PosKdX,
    PosKdY,
    PosKdZ,
    // Attitude control gains
    AttKrX,
    AttKrY,
    AttKrZ,
    // Rate PID gains (dimension-major)
    RateKpR,
    RateKpP,
    RateKpY,
    RateKiR,
    RateKiP,
    RateKiY,
    RateKdR,
    RateKdP,
    RateKdY,
}

/// All parameter keys in order, for iteration.
pub const ALL_KEYS: &[ParamKey] = &[
    ParamKey::Mass,
    ParamKey::Ixx,
    ParamKey::Ixy,
    ParamKey::Ixz,
    ParamKey::Iyx,
    ParamKey::Iyy,
    ParamKey::Iyz,
    ParamKey::Izx,
    ParamKey::Izy,
    ParamKey::Izz,
    ParamKey::M0Px,
    ParamKey::M0Py,
    ParamKey::M0Spin,
    ParamKey::M0Thrust,
    ParamKey::M0Torque,
    ParamKey::M1Px,
    ParamKey::M1Py,
    ParamKey::M1Spin,
    ParamKey::M1Thrust,
    ParamKey::M1Torque,
    ParamKey::M2Px,
    ParamKey::M2Py,
    ParamKey::M2Spin,
    ParamKey::M2Thrust,
    ParamKey::M2Torque,
    ParamKey::M3Px,
    ParamKey::M3Py,
    ParamKey::M3Spin,
    ParamKey::M3Thrust,
    ParamKey::M3Torque,
    ParamKey::PosKpX,
    ParamKey::PosKpY,
    ParamKey::PosKpZ,
    ParamKey::PosKdX,
    ParamKey::PosKdY,
    ParamKey::PosKdZ,
    ParamKey::AttKrX,
    ParamKey::AttKrY,
    ParamKey::AttKrZ,
    ParamKey::RateKpR,
    ParamKey::RateKpP,
    ParamKey::RateKpY,
    ParamKey::RateKiR,
    ParamKey::RateKiP,
    ParamKey::RateKiY,
    ParamKey::RateKdR,
    ParamKey::RateKdP,
    ParamKey::RateKdY,
];

impl ParamKey {
    /// Parse a parameter name string into a key.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "mass" => Some(Self::Mass),
            "ixx" => Some(Self::Ixx),
            "ixy" => Some(Self::Ixy),
            "ixz" => Some(Self::Ixz),
            "iyx" => Some(Self::Iyx),
            "iyy" => Some(Self::Iyy),
            "iyz" => Some(Self::Iyz),
            "izx" => Some(Self::Izx),
            "izy" => Some(Self::Izy),
            "izz" => Some(Self::Izz),
            "m0_px" => Some(Self::M0Px),
            "m0_py" => Some(Self::M0Py),
            "m0_spin" => Some(Self::M0Spin),
            "m0_thrust" => Some(Self::M0Thrust),
            "m0_torque" => Some(Self::M0Torque),
            "m1_px" => Some(Self::M1Px),
            "m1_py" => Some(Self::M1Py),
            "m1_spin" => Some(Self::M1Spin),
            "m1_thrust" => Some(Self::M1Thrust),
            "m1_torque" => Some(Self::M1Torque),
            "m2_px" => Some(Self::M2Px),
            "m2_py" => Some(Self::M2Py),
            "m2_spin" => Some(Self::M2Spin),
            "m2_thrust" => Some(Self::M2Thrust),
            "m2_torque" => Some(Self::M2Torque),
            "m3_px" => Some(Self::M3Px),
            "m3_py" => Some(Self::M3Py),
            "m3_spin" => Some(Self::M3Spin),
            "m3_thrust" => Some(Self::M3Thrust),
            "m3_torque" => Some(Self::M3Torque),
            "pos_kp_x" => Some(Self::PosKpX),
            "pos_kp_y" => Some(Self::PosKpY),
            "pos_kp_z" => Some(Self::PosKpZ),
            "pos_kd_x" => Some(Self::PosKdX),
            "pos_kd_y" => Some(Self::PosKdY),
            "pos_kd_z" => Some(Self::PosKdZ),
            "att_kr_x" => Some(Self::AttKrX),
            "att_kr_y" => Some(Self::AttKrY),
            "att_kr_z" => Some(Self::AttKrZ),
            "rate_kp_r" => Some(Self::RateKpR),
            "rate_kp_p" => Some(Self::RateKpP),
            "rate_kp_y" => Some(Self::RateKpY),
            "rate_ki_r" => Some(Self::RateKiR),
            "rate_ki_p" => Some(Self::RateKiP),
            "rate_ki_y" => Some(Self::RateKiY),
            "rate_kd_r" => Some(Self::RateKdR),
            "rate_kd_p" => Some(Self::RateKdP),
            "rate_kd_y" => Some(Self::RateKdY),
            _ => None,
        }
    }

    /// Return the canonical name for this key.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mass => "mass",
            Self::Ixx => "ixx",
            Self::Ixy => "ixy",
            Self::Ixz => "ixz",
            Self::Iyx => "iyx",
            Self::Iyy => "iyy",
            Self::Iyz => "iyz",
            Self::Izx => "izx",
            Self::Izy => "izy",
            Self::Izz => "izz",
            Self::M0Px => "m0_px",
            Self::M0Py => "m0_py",
            Self::M0Spin => "m0_spin",
            Self::M0Thrust => "m0_thrust",
            Self::M0Torque => "m0_torque",
            Self::M1Px => "m1_px",
            Self::M1Py => "m1_py",
            Self::M1Spin => "m1_spin",
            Self::M1Thrust => "m1_thrust",
            Self::M1Torque => "m1_torque",
            Self::M2Px => "m2_px",
            Self::M2Py => "m2_py",
            Self::M2Spin => "m2_spin",
            Self::M2Thrust => "m2_thrust",
            Self::M2Torque => "m2_torque",
            Self::M3Px => "m3_px",
            Self::M3Py => "m3_py",
            Self::M3Spin => "m3_spin",
            Self::M3Thrust => "m3_thrust",
            Self::M3Torque => "m3_torque",
            Self::PosKpX => "pos_kp_x",
            Self::PosKpY => "pos_kp_y",
            Self::PosKpZ => "pos_kp_z",
            Self::PosKdX => "pos_kd_x",
            Self::PosKdY => "pos_kd_y",
            Self::PosKdZ => "pos_kd_z",
            Self::AttKrX => "att_kr_x",
            Self::AttKrY => "att_kr_y",
            Self::AttKrZ => "att_kr_z",
            Self::RateKpR => "rate_kp_r",
            Self::RateKpP => "rate_kp_p",
            Self::RateKpY => "rate_kp_y",
            Self::RateKiR => "rate_ki_r",
            Self::RateKiP => "rate_ki_p",
            Self::RateKiY => "rate_ki_y",
            Self::RateKdR => "rate_kd_r",
            Self::RateKdP => "rate_kd_p",
            Self::RateKdY => "rate_kd_y",
        }
    }
}

// -- byte helpers --

fn put_f32(buf: &mut [u8], off: usize, val: f32) -> usize {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    off + 4
}

fn put_u32(buf: &mut [u8], off: usize, val: u32) -> usize {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    off + 4
}

fn get_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixer::SpinDir;

    fn test_params() -> VehicleParams {
        VehicleParams {
            body: RigidBodyParams {
                mass_kg: 0.5,
                inertia_kg_m2: [0.0025, 0.0, 0.0, 0.0, 0.0021, 0.0, 0.0, 0.0, 0.0043],
            },
            motors: [
                MotorParams {
                    position_m: [-0.075, -0.1],
                    spin_dir: SpinDir::Cw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [0.075, -0.1],
                    spin_dir: SpinDir::Ccw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [-0.075, 0.1],
                    spin_dir: SpinDir::Ccw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
                MotorParams {
                    position_m: [0.075, 0.1],
                    spin_dir: SpinDir::Cw,
                    max_thrust_n: 8.5,
                    torque_coeff_m: 0.022,
                },
            ],
            control: ControlGains {
                pos_kp: [4.0, 4.0, 5.0],
                pos_kd: [4.0, 4.0, 4.0],
                att_k_rate: [3.0, 3.0, 1.0],
                rate_kp: [0.1, 0.08, 0.05],
                rate_ki: [0.0, 0.0, 0.0],
                rate_kd: [0.0, 0.0, 0.0],
            },
        }
    }

    #[test]
    fn round_trip() {
        let params = test_params();
        let bytes = params.to_bytes();
        let restored = VehicleParams::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.body.mass_kg, params.body.mass_kg);
        assert_eq!(restored.body.inertia_kg_m2, params.body.inertia_kg_m2);
        for (a, b) in restored.motors.iter().zip(params.motors.iter()) {
            assert_eq!(a.position_m, b.position_m);
            assert_eq!(a.spin_dir, b.spin_dir);
            assert_eq!(a.max_thrust_n, b.max_thrust_n);
            assert_eq!(a.torque_coeff_m, b.torque_coeff_m);
        }
    }

    #[test]
    fn bad_magic_returns_none() {
        let params = test_params();
        let mut bytes = params.to_bytes();
        bytes[0] = 0xFF;
        assert!(VehicleParams::from_bytes(&bytes).is_none());
    }

    #[test]
    fn bad_crc_returns_none() {
        let params = test_params();
        let mut bytes = params.to_bytes();
        bytes[HEADER_SIZE] ^= 0xFF; // corrupt payload
        assert!(VehicleParams::from_bytes(&bytes).is_none());
    }

    #[test]
    fn blank_flash_returns_none() {
        let bytes = [0xFF; PADDED_SIZE];
        assert!(VehicleParams::from_bytes(&bytes).is_none());
    }

    #[test]
    fn get_set_round_trip() {
        let mut params = test_params();
        params.set(ParamKey::Mass, 0.6);
        assert_eq!(params.get(ParamKey::Mass), 0.6);
        params.set(ParamKey::M2Thrust, 9.0);
        assert_eq!(params.get(ParamKey::M2Thrust), 9.0);
    }

    #[test]
    fn param_key_from_str() {
        assert_eq!(ParamKey::from_str("mass"), Some(ParamKey::Mass));
        assert_eq!(ParamKey::from_str("m3_torque"), Some(ParamKey::M3Torque));
        assert_eq!(ParamKey::from_str("invalid"), None);
    }
}
