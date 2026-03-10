// RC channel mapping: calibration, expo/rates, and normalization.
//
// Converts raw RC PWM values (typically 988–2012 µs) into normalized
// thrust [0,1] and rate [-1,1] commands.

/// Number of control axes: roll, pitch, throttle, yaw.
pub const NUM_AXES: usize = 4;

/// Output axis order from `RcMapper::map`.
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    Throttle = 0,
    Roll = 1,
    Pitch = 2,
    Yaw = 3,
}

/// Calibration for a single RC channel.
#[derive(Debug, Clone)]
pub struct ChannelCalibration {
    /// Index into the raw RC channel array.
    pub index: usize,
    pub min: i16,
    pub max: i16,
    /// Center value. Set equal to `min` for throttle (maps to [0,1]).
    pub center: i16,
    pub invert: bool,
}

impl ChannelCalibration {
    pub fn centered(index: usize) -> Self {
        Self {
            index,
            min: 988,
            max: 2012,
            center: 1500,
            invert: false,
        }
    }

    pub fn throttle(index: usize) -> Self {
        Self {
            index,
            min: 988,
            max: 2012,
            center: 988,
            invert: false,
        }
    }

    pub fn normalize(&self, value: i16) -> f32 {
        let v = if self.center <= self.min {
            // Throttle-style: [0, 1]
            let travel = (self.max - self.min) as f32;
            if travel <= 0.0 {
                return 0.0;
            }
            (value - self.min) as f32 / travel
        } else {
            // Centered stick: [-1, 1]
            if value >= self.center {
                let half = (self.max - self.center) as f32;
                if half <= 0.0 { 0.0 } else { (value - self.center) as f32 / half }
            } else {
                let half = (self.center - self.min) as f32;
                if half <= 0.0 { 0.0 } else { (value - self.center) as f32 / half }
            }
        };
        let v = if self.invert { -v } else { v };
        v.clamp(-1.0, 1.0)
    }
}

/// Rate/expo/deadband settings for one axis.
#[derive(Debug, Clone)]
pub struct ChannelSetting {
    /// Maximum rate in the output unit (e.g. rad/s for rates, 1.0 for throttle).
    pub rate: f32,
    /// Expo factor [0,1]. 0 = linear, 1 = full cubic.
    pub expo: f32,
    /// Deadband around center [0,1]. Values below this are snapped to 0.
    pub deadband: f32,
}

impl Default for ChannelSetting {
    fn default() -> Self {
        Self {
            rate: 1.0,
            expo: 0.0,
            deadband: 0.0,
        }
    }
}

impl ChannelSetting {
    pub fn new(rate: f32, expo: f32, deadband: f32) -> Self {
        Self { rate, expo, deadband }
    }

    pub fn apply(&self, x: f32) -> f32 {
        let x = self.apply_deadband(x);
        self.apply_expo(x) * self.rate
    }

    pub fn apply_throttle(&self, x: f32) -> f32 {
        (self.apply_expo(x) * self.rate).clamp(0.0, 1.0)
    }

    fn apply_deadband(&self, x: f32) -> f32 {
        if self.deadband > 0.0 && x.abs() < self.deadband {
            0.0
        } else if self.deadband > 0.0 {
            let scale = 1.0 / (1.0 - self.deadband);
            ((x.abs() - self.deadband) * scale).copysign(x)
        } else {
            x
        }
    }

    fn apply_expo(&self, x: f32) -> f32 {
        if self.expo > 0.0 {
            x * x * x * self.expo + x * (1.0 - self.expo)
        } else {
            x
        }
    }
}

/// Thrust + body-rate setpoint output.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThrustRates {
    /// Normalized thrust [0, 1].
    pub thrust: f32,
    /// Roll rate command (output units depend on `ChannelSetting::rate`).
    pub roll_rate: f32,
    /// Pitch rate command.
    pub pitch_rate: f32,
    /// Yaw rate command.
    pub yaw_rate: f32,
}

/// Maps raw RC channel values to thrust + rate setpoints.
#[derive(Debug, Clone)]
pub struct RcMapper {
    pub roll: ChannelCalibration,
    pub pitch: ChannelCalibration,
    pub throttle: ChannelCalibration,
    pub yaw: ChannelCalibration,
    pub settings: RcSettings,
}

/// Rate/expo settings for all four axes.
#[derive(Debug, Clone, Default)]
pub struct RcSettings {
    pub roll: ChannelSetting,
    pub pitch: ChannelSetting,
    pub throttle: ChannelSetting,
    pub yaw: ChannelSetting,
}

impl RcMapper {
    /// Create a mapper with AETR channel order (typical for CRSF/GHST).
    /// Channels: 0=Roll, 1=Pitch, 2=Throttle, 3=Yaw.
    pub fn aetr(settings: RcSettings) -> Self {
        Self {
            roll: ChannelCalibration::centered(0),
            pitch: ChannelCalibration::centered(1),
            throttle: ChannelCalibration::throttle(2),
            yaw: ChannelCalibration::centered(3),
            settings,
        }
    }

    /// Create a mapper with TAER channel order.
    /// Channels: 0=Throttle, 1=Roll, 2=Pitch, 3=Yaw.
    pub fn taer(settings: RcSettings) -> Self {
        Self {
            throttle: ChannelCalibration::throttle(0),
            roll: ChannelCalibration::centered(1),
            pitch: ChannelCalibration::centered(2),
            yaw: ChannelCalibration::centered(3),
            settings,
        }
    }

    /// Map raw RC channels to thrust + rates.
    pub fn map(&self, channels: &[u16; 16]) -> ThrustRates {
        let thr_norm = self.throttle.normalize(channels[self.throttle.index] as i16);
        let roll_norm = self.roll.normalize(channels[self.roll.index] as i16);
        let pitch_norm = self.pitch.normalize(channels[self.pitch.index] as i16);
        let yaw_norm = self.yaw.normalize(channels[self.yaw.index] as i16);

        ThrustRates {
            thrust: self.settings.throttle.apply_throttle(thr_norm),
            roll_rate: self.settings.roll.apply(roll_norm),
            pitch_rate: self.settings.pitch.apply(pitch_norm),
            yaw_rate: self.settings.yaw.apply(yaw_norm),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centered_normalization() {
        let c = ChannelCalibration::centered(0);
        assert!((c.normalize(988) - (-1.0)).abs() < 0.01);
        assert_eq!(c.normalize(1500), 0.0);
        assert!((c.normalize(2012) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_centered_inverted() {
        let mut c = ChannelCalibration::centered(0);
        c.invert = true;
        assert!((c.normalize(988) - 1.0).abs() < 0.01);
        assert_eq!(c.normalize(1500), 0.0);
        assert!((c.normalize(2012) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_throttle_normalization() {
        let c = ChannelCalibration::throttle(0);
        assert!((c.normalize(988) - 0.0).abs() < 0.01);
        assert!((c.normalize(1500) - 0.5).abs() < 0.01);
        assert!((c.normalize(2012) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_deadband() {
        let s = ChannelSetting::new(1.0, 0.0, 0.1);
        assert_eq!(s.apply(0.0), 0.0);
        assert_eq!(s.apply(0.05), 0.0);
        assert!(s.apply(0.5).abs() > 0.0);
    }

    #[test]
    fn test_expo() {
        let s = ChannelSetting::new(1.0, 0.5, 0.0);
        assert_eq!(s.apply(0.0), 0.0);
        assert_eq!(s.apply(1.0), 1.0);
        // Expo should make midpoint smaller than linear
        assert!(s.apply(0.5) < 0.5);
    }

    #[test]
    fn test_mapper_aetr() {
        let mapper = RcMapper::aetr(RcSettings::default());
        let mut channels = [1500u16; 16];
        channels[2] = 988; // throttle min
        let out = mapper.map(&channels);
        assert!(out.thrust.abs() < 0.02);
        assert!(out.roll_rate.abs() < 0.02);
        assert!(out.pitch_rate.abs() < 0.02);
        assert!(out.yaw_rate.abs() < 0.02);

        channels[2] = 2012; // throttle max
        channels[0] = 2012; // roll max
        let out = mapper.map(&channels);
        assert!((out.thrust - 1.0).abs() < 0.02);
        assert!((out.roll_rate - 1.0).abs() < 0.02);
    }
}
