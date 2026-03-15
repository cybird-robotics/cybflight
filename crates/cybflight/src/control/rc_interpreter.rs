//! RC interpreter: subscribes to raw RC channels and publishes
//! `ManualControlSetpoint` (thrust + body rates) via `RcMapper`.

use embassy_time::Instant;

use cybflight_core::rc::rc_mapping::{RcMapper, RcSettings, ChannelSetting};
use cybflight_msgs as msgs;

use crate::sensors::{MANUAL_CONTROL, RC_INPUT};

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

#[embassy_executor::task]
pub async fn rc_interpreter_task() {
    let mut sub = RC_INPUT
        .subscriber()
        .expect("rc_interpreter: RC_INPUT subscriber");
    let pub_ = MANUAL_CONTROL.immediate_publisher();
    let mapper = RcMapper::aetr(default_acro_settings());

    loop {
        let rc = sub.next_message_pure().await;
        let tr = mapper.map(&rc.channels);
        pub_.publish_immediate(msgs::ManualControlSetpoint {
            timestamp: Instant::now(),
            thrust: tr.thrust,
            roll_rate: tr.roll_rate,
            pitch_rate: tr.pitch_rate,
            yaw_rate: tr.yaw_rate,
        });
        defmt::debug!(
            "RC: ch[0..4]={} {} {} {} {}, setpoint: thrust={} roll_rate={} pitch_rate={} yaw_rate={}",
            rc.channels[0],
            rc.channels[1],
            rc.channels[2],
            rc.channels[3],
            rc.channels[4],
            tr.thrust,
            tr.roll_rate,
            tr.pitch_rate,
            tr.yaw_rate
        );
    }
}
