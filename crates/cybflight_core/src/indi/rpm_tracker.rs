// Per-motor RPM validity tracking for INDI.
//
// Tracks DShot telemetry validity per motor and determines:
// - Whether to use G2 for each motor
// - Whether to failsafe (all motors lost)

/// RPM validity status for one motor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RpmStatus {
    /// Valid RPM reading.
    Valid,
    /// Single corrupt frame — using last good value.
    LastGood,
    /// Too many consecutive invalid frames — G2 zeroed for this motor.
    Invalid,
}

/// Per-motor RPM validity tracker.
pub struct RpmTracker<const N: usize> {
    last_good_omega: [f32; N],
    invalid_count: [u16; N],
    /// Consecutive valid frames since last invalid (for recovery hysteresis).
    valid_count: [u16; N],
    /// True if this motor was previously in Invalid state (G2 zeroed).
    was_invalid: [bool; N],
    all_invalid_count: u16,
}

/// Result of one RPM update cycle.
pub struct RpmUpdate<const N: usize> {
    /// Motor speed to use (rad/s). Either fresh, last-good, or zero.
    pub omega: [f32; N],
    /// Whether G2 should be active for each motor.
    pub g2_valid: [bool; N],
    /// Per-motor status.
    pub status: [RpmStatus; N],
    /// True if all motors have been invalid for too long → failsafe.
    pub failsafe: bool,
}

/// Input telemetry value for one motor.
#[derive(Clone, Copy, Debug)]
pub enum RpmInput {
    /// Valid eRPM reading.
    Erpm(u32),
    /// Motor stopped (valid, RPM=0).
    Stopped,
    /// Invalid reading (noise, dropout, EDT frame, etc.)
    Invalid,
}

impl<const N: usize> RpmTracker<N> {
    pub fn new() -> Self {
        Self {
            last_good_omega: [0.0; N],
            invalid_count: [0; N],
            valid_count: [0; N],
            was_invalid: [false; N],
            all_invalid_count: 0,
        }
    }

    /// Update with new telemetry readings.
    ///
    /// `erpm_to_rads`: conversion factor from eRPM to rad/s.
    /// `invalid_limit`: consecutive invalid frames before zeroing G2 for one motor.
    /// `all_invalid_limit`: consecutive frames with ALL motors invalid before failsafe.
    /// `recovery_count`: consecutive valid frames required to re-enable G2 after it was zeroed.
    pub fn update(
        &mut self,
        inputs: &[RpmInput; N],
        erpm_to_rads: f32,
        invalid_limit: u16,
        all_invalid_limit: u16,
        recovery_count: u16,
    ) -> RpmUpdate<N> {
        let mut omega = [0.0f32; N];
        let mut g2_valid = [false; N];
        let mut status = [RpmStatus::Invalid; N];
        let mut any_valid = false;

        for i in 0..N {
            match inputs[i] {
                RpmInput::Erpm(erpm) => {
                    let w = erpm as f32 * erpm_to_rads;
                    omega[i] = w;
                    self.last_good_omega[i] = w;
                    self.invalid_count[i] = 0;
                    self.valid_count[i] = self.valid_count[i].saturating_add(1);
                    status[i] = RpmStatus::Valid;
                    any_valid = true;
                    // Recovery hysteresis: require N consecutive valid frames
                    // before re-enabling G2 after it was zeroed.
                    if self.was_invalid[i] {
                        g2_valid[i] = self.valid_count[i] >= recovery_count;
                        if g2_valid[i] {
                            self.was_invalid[i] = false;
                        }
                    } else {
                        g2_valid[i] = true;
                    }
                }
                RpmInput::Stopped => {
                    omega[i] = 0.0;
                    self.last_good_omega[i] = 0.0;
                    self.invalid_count[i] = 0;
                    self.valid_count[i] = self.valid_count[i].saturating_add(1);
                    status[i] = RpmStatus::Valid;
                    any_valid = true;
                    if self.was_invalid[i] {
                        g2_valid[i] = self.valid_count[i] >= recovery_count;
                        if g2_valid[i] {
                            self.was_invalid[i] = false;
                        }
                    } else {
                        g2_valid[i] = true;
                    }
                }
                RpmInput::Invalid => {
                    self.invalid_count[i] = self.invalid_count[i].saturating_add(1);
                    self.valid_count[i] = 0;
                    if self.invalid_count[i] < invalid_limit {
                        omega[i] = self.last_good_omega[i];
                        g2_valid[i] = !self.was_invalid[i]; // still invalid if recovering
                        status[i] = RpmStatus::LastGood;
                        any_valid = true;
                    } else {
                        omega[i] = 0.0;
                        g2_valid[i] = false;
                        self.was_invalid[i] = true;
                        status[i] = RpmStatus::Invalid;
                    }
                }
            }
        }

        if any_valid {
            self.all_invalid_count = 0;
        } else {
            self.all_invalid_count = self.all_invalid_count.saturating_add(1);
        }

        RpmUpdate {
            omega,
            g2_valid,
            status,
            failsafe: self.all_invalid_count >= all_invalid_limit,
        }
    }
}

/// Takeoff detection: determines whether the vehicle is on the ground.
///
/// When touching ground, INDI runs in non-incremental (NDI) mode.
/// When airborne, it runs in incremental mode.
pub fn is_touching_ground(
    gyro_magnitude_sq: f32,
    accel_magnitude_sq: f32,
    thrust_command_low: bool,
) -> bool {
    // Thresholds matching indiflight's isTouchingGround()
    let gyro_thresh_rad_s = 100.0_f32 * core::f32::consts::PI / 180.0;
    let gyro_low = gyro_magnitude_sq < gyro_thresh_rad_s * gyro_thresh_rad_s;
    let accel_high = accel_magnitude_sq > (0.8 * 9.81) * (0.8 * 9.81);
    gyro_low && accel_high && thrust_command_low
}

#[cfg(test)]
mod tests {
    use super::*;

    const ERPM_TO_RADS: f32 = 100.0 / 7.0 / 60.0 * core::f32::consts::TAU; // 14-pole motor

    // -----------------------------------------------------------------------
    // RpmTracker tests
    // -----------------------------------------------------------------------

    #[test]
    fn valid_rpm_returns_correct_omega() {
        let mut tracker = RpmTracker::<4>::new();
        let inputs = [RpmInput::Erpm(10000), RpmInput::Erpm(10000),
                      RpmInput::Erpm(10000), RpmInput::Erpm(10000)];
        let result = tracker.update(&inputs, ERPM_TO_RADS, 50, 50, 5);

        assert!(result.g2_valid.iter().all(|&v| v));
        assert!(result.status.iter().all(|&s| s == RpmStatus::Valid));
        assert!(!result.failsafe);
        assert!(result.omega[0] > 0.0);
    }

    #[test]
    fn stopped_motor_is_valid() {
        let mut tracker = RpmTracker::<4>::new();
        let inputs = [RpmInput::Stopped, RpmInput::Erpm(10000),
                      RpmInput::Erpm(10000), RpmInput::Erpm(10000)];
        let result = tracker.update(&inputs, ERPM_TO_RADS, 50, 50, 5);

        assert!(result.g2_valid[0]);
        assert_eq!(result.status[0], RpmStatus::Valid);
        assert_eq!(result.omega[0], 0.0);
    }

    #[test]
    fn single_corrupt_frame_uses_last_good() {
        let mut tracker = RpmTracker::<4>::new();

        // First: valid reading
        let valid = [RpmInput::Erpm(20000); 4];
        let r1 = tracker.update(&valid, ERPM_TO_RADS, 50, 50, 5);
        let saved_omega = r1.omega[0];

        // Second: one motor invalid
        let corrupt = [RpmInput::Invalid, RpmInput::Erpm(20000),
                       RpmInput::Erpm(20000), RpmInput::Erpm(20000)];
        let r2 = tracker.update(&corrupt, ERPM_TO_RADS, 50, 50, 5);

        assert!(r2.g2_valid[0], "single corrupt should keep G2 active");
        assert_eq!(r2.status[0], RpmStatus::LastGood);
        assert_eq!(r2.omega[0], saved_omega, "should use last good omega");
        assert!(!r2.failsafe);
    }

    #[test]
    fn n_consecutive_invalid_zeros_g2() {
        let mut tracker = RpmTracker::<4>::new();
        let invalid_limit = 5u16;

        // First: valid
        tracker.update(&[RpmInput::Erpm(20000); 4], ERPM_TO_RADS, invalid_limit, 50, 5);

        // Then: motor 0 goes invalid for N frames
        for _ in 0..invalid_limit {
            let inputs = [RpmInput::Invalid, RpmInput::Erpm(20000),
                          RpmInput::Erpm(20000), RpmInput::Erpm(20000)];
            let r = tracker.update(&inputs, ERPM_TO_RADS, invalid_limit, 50, 5);

            if r.status[0] == RpmStatus::Invalid {
                assert!(!r.g2_valid[0], "G2 should be zeroed after N invalid");
                assert_eq!(r.omega[0], 0.0);
                // Other motors still valid
                assert!(r.g2_valid[1]);
                assert!(!r.failsafe, "single motor loss should not failsafe");
                return;
            }
        }
        panic!("should have reached Invalid status within limit");
    }

    #[test]
    fn all_motors_invalid_triggers_failsafe() {
        let mut tracker = RpmTracker::<4>::new();
        let all_invalid_limit = 10u16;

        // Valid first
        tracker.update(&[RpmInput::Erpm(20000); 4], ERPM_TO_RADS, 5, all_invalid_limit, 5);

        // All motors invalid for enough frames
        for frame in 0..(5 + all_invalid_limit) {
            let r = tracker.update(
                &[RpmInput::Invalid; 4], ERPM_TO_RADS, 5, all_invalid_limit, 5,
            );
            if r.failsafe {
                assert!(frame >= 5, "failsafe should not trigger before invalid_limit");
                return;
            }
        }
        panic!("failsafe should have triggered");
    }

    #[test]
    fn recovery_from_invalid_resets_counter() {
        let mut tracker = RpmTracker::<4>::new();

        // Valid
        tracker.update(&[RpmInput::Erpm(20000); 4], ERPM_TO_RADS, 50, 50, 5);

        // 3 invalid frames
        for _ in 0..3 {
            tracker.update(&[RpmInput::Invalid; 4], ERPM_TO_RADS, 50, 50, 5);
        }

        // Valid again — should reset
        let r = tracker.update(&[RpmInput::Erpm(20000); 4], ERPM_TO_RADS, 50, 50, 5);
        assert!(r.g2_valid.iter().all(|&v| v));
        assert!(r.status.iter().all(|&s| s == RpmStatus::Valid));
        assert!(!r.failsafe);
    }

    // -----------------------------------------------------------------------
    // Takeoff detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn on_ground_all_conditions_met() {
        // Low gyro, high accel (~1g), low thrust
        let gyro_sq = 0.1_f32 * 0.1; // ~5.7 deg/s
        let accel_sq = 9.81_f32 * 9.81; // 1g
        assert!(is_touching_ground(gyro_sq, accel_sq, true));
    }

    #[test]
    fn airborne_high_gyro() {
        // High gyro > 100 deg/s → not touching ground
        let gyro_rad = 120.0_f32 * core::f32::consts::PI / 180.0;
        let gyro_sq = gyro_rad * gyro_rad;
        let accel_sq = 9.81_f32 * 9.81;
        assert!(!is_touching_ground(gyro_sq, accel_sq, true));
    }

    #[test]
    fn airborne_low_accel() {
        // Low accel < 0.8g → freefall / maneuvering
        let accel = 0.5 * 9.81;
        let accel_sq = accel * accel;
        assert!(!is_touching_ground(0.0, accel_sq, true));
    }

    #[test]
    fn airborne_high_thrust() {
        // High thrust command → definitely flying
        let accel_sq = 9.81_f32 * 9.81;
        assert!(!is_touching_ground(0.0, accel_sq, false));
    }

    #[test]
    fn boundary_gyro_threshold() {
        // Exactly at 100 deg/s boundary
        let thresh = 100.0_f32 * core::f32::consts::PI / 180.0;
        let just_below = thresh - 0.01;
        let just_above = thresh + 0.01;
        let accel_sq = 9.81_f32 * 9.81;

        assert!(is_touching_ground(just_below * just_below, accel_sq, true));
        assert!(!is_touching_ground(just_above * just_above, accel_sq, true));
    }

    #[test]
    fn boundary_accel_threshold() {
        // Exactly at 0.8g boundary
        let thresh = 0.8 * 9.81;
        let just_below = thresh - 0.01;
        let just_above = thresh + 0.01;

        assert!(!is_touching_ground(0.0, just_below * just_below, true));
        assert!(is_touching_ground(0.0, just_above * just_above, true));
    }
}
