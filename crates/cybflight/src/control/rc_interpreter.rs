//! RC interpreter: subscribes to raw RC channels and publishes
//! `ManualControlSetpoint` (thrust + body rates) via `RcMapper`.

use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3};

use cybflight_core::rc::rc_mapping::{ChannelSetting, RcMapper, RcSettings};
use cybflight_msgs as msgs;

use crate::estimate::VEHICLE_ODOMETRY;
use crate::sensors::{MANUAL_CONTROL, RC_INPUT, STABILIZED_CONTROL};
use super::flight_mode::{FlightMode, HYSTERESIS, MODE_ANGLE_MAX, MODE_AUTO_MAX, MODE_CHANNEL};

/// Number of consecutive frames a new mode must be seen before committing.
const MODE_DEBOUNCE_COUNT: u8 = 5;

// ---------------------------------------------------------------------------
// Mode debouncer
// ---------------------------------------------------------------------------

struct ModeDebouncer {
    current: FlightMode,
    pending: FlightMode,
    count: u8,
}

impl ModeDebouncer {
    fn new() -> Self {
        Self {
            current: FlightMode::Acro,
            pending: FlightMode::Acro,
            count: 0,
        }
    }

    fn update(&mut self, raw_mode: FlightMode) -> FlightMode {
        if raw_mode == self.pending {
            self.count = self.count.saturating_add(1);
        } else {
            self.pending = raw_mode;
            self.count = 1;
        }
        if self.count >= MODE_DEBOUNCE_COUNT {
            self.current = self.pending;
        }
        self.current
    }
}

// ---------------------------------------------------------------------------
// RC settings
// ---------------------------------------------------------------------------

/// Create default rate-mode settings.
///
/// Roll/pitch: 800 deg/s max rate, moderate expo.
/// Yaw: 400 deg/s max rate.
/// Throttle: linear, full range.
fn default_acro_settings() -> RcSettings {
    RcSettings {
        roll: ChannelSetting::new(270.0_f32.to_radians(), 0.0, 0.02),
        pitch: ChannelSetting::new(270.0_f32.to_radians(), 0.0, 0.02),
        yaw: ChannelSetting::new(90.0_f32.to_radians(), 0.0, 0.02),
        throttle: ChannelSetting::default(),
    }
}

fn default_angle_settings() -> RcSettings {
    RcSettings {
        roll: ChannelSetting::new(35.0_f32.to_radians(), 0.0, 0.02),
        pitch: ChannelSetting::new(35.0_f32.to_radians(), 0.0, 0.02),
        yaw: ChannelSetting::new(180.0_f32.to_radians(), 0.0, 0.02),
        throttle: ChannelSetting::default(),
    }
}

// ---------------------------------------------------------------------------
// Flight mode decoding with hysteresis
// ---------------------------------------------------------------------------

/// Decode AUX3 three-position switch into FlightMode with hysteresis.
///
/// Thresholds shift depending on the current mode so that the switch value must
/// cross threshold ± HYSTERESIS before a transition is recognised, preventing
/// oscillation when the PWM value hovers near a boundary.
fn decode_flight_mode(channels: &[u16; 16], channel_count: u8, current: FlightMode) -> FlightMode {
    if (channel_count as usize) <= MODE_CHANNEL {
        return FlightMode::Acro;
    }

    let aux3 = channels[MODE_CHANNEL];
    match current {
        FlightMode::Auto => {
            if aux3 > MODE_AUTO_MAX + HYSTERESIS {
                FlightMode::Angle
            } else {
                FlightMode::Auto
            }
        }
        FlightMode::Angle => {
            if aux3 < MODE_AUTO_MAX - HYSTERESIS {
                FlightMode::Auto
            } else if aux3 > MODE_ANGLE_MAX + HYSTERESIS {
                FlightMode::Acro
            } else {
                FlightMode::Angle
            }
        }
        FlightMode::Acro => {
            if aux3 < MODE_ANGLE_MAX - HYSTERESIS {
                FlightMode::Angle
            } else {
                FlightMode::Acro
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract yaw angle from a unit quaternion (ZYX Euler convention).
fn extract_yaw(q: &UnitQuaternion<f32>) -> f32 {
    let (_roll, _pitch, yaw) = q.euler_angles();
    yaw
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    let mut sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");
    let rate_pub = MANUAL_CONTROL.immediate_publisher();
    let angle_pub = STABILIZED_CONTROL.immediate_publisher();
    let mapper = RcMapper::aetr(default_acro_settings(), default_angle_settings());

    let mut odom_sub = VEHICLE_ODOMETRY
        .subscriber()
        .expect("rc_interpreter: VEHICLE_ODOMETRY subscriber");

    let mut debouncer = ModeDebouncer::new();
    let mut prev_mode = FlightMode::Acro;

    loop {
        let rc = sub.next_message_pure().await;

        // Decode flight mode from AUX3 with hysteresis, then debounce.
        let raw_mode = decode_flight_mode(&rc.channels, rc.channel_count, debouncer.current);
        let mode = debouncer.update(raw_mode);

        match mode {
            FlightMode::Acro => {
                let tr = mapper.map_rates(&rc.channels);
                rate_pub.publish_immediate(msgs::ManualControlSetpoint {
                    timestamp: Instant::now(),
                    thrust: tr.thrust,
                    roll_rate: tr.roll_rate,
                    pitch_rate: tr.pitch_rate,
                    yaw_rate: tr.yaw_rate,
                });
                defmt::debug!(
                    "RC acro: thrust={} roll_rate={} pitch_rate={} yaw_rate={}",
                    tr.thrust, tr.roll_rate, tr.pitch_rate, tr.yaw_rate
                );
            }
            FlightMode::Angle => {
                let ta = mapper.map_angles(&rc.channels);
                angle_pub.publish_immediate(msgs::StablizedControlSetpoint {
                    timestamp: Instant::now(),
                    thrust: ta.thrust,
                    roll: ta.roll,
                    pitch: ta.pitch,
                    yaw: ta.yaw,
                });
                defmt::debug!(
                    "RC angle: thrust={} roll={} pitch={} yaw={}",
                    ta.thrust, ta.roll, ta.pitch, ta.yaw
                );
            }
            FlightMode::Auto => {
                // Auto mode: autonomy provides its own setpoints.
                defmt::debug!("RC auto: no manual setpoint published");
            }
        }

        // On transition *into* Auto, snapshot current odometry as initial setpoint
        // with level attitude (zero roll/pitch) and zero velocity.
        if mode != prev_mode {
            defmt::info!("flight mode: {} → {}", prev_mode, mode);

            if mode == FlightMode::Auto {
                if let Some(odom) = odom_sub.try_next_message_pure() {
                    let yaw = extract_yaw(&odom.pose.orientation);
                    let level_orientation = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);
                    super::AUTO_SETPOINT.signal(msgs::VehicleOdometry {
                        timestamp: Instant::now(),
                        pose: msgs::Pose {
                            position: odom.pose.position,
                            orientation: level_orientation,
                        },
                        twist: msgs::Twist {
                            linear: Vector3::zeros(),
                            angular: Vector3::zeros(),
                        },
                    });
                    defmt::info!("auto setpoint: pos={} yaw={}", odom.pose.position, yaw);
                } else {
                    defmt::warn!("auto mode entered but no odometry available for initial setpoint");
                }
            }

            prev_mode = mode;
        }
        super::FLIGHT_MODE.signal(mode);
    }
}
