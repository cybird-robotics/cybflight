pub mod imu;
use crate::msgs;

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, channel::Channel};

pub static FUSED_IMU: Channel<CriticalSectionRawMutex, msgs::Imu, 4> = Channel::new();
