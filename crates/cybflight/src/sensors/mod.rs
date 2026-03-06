pub mod imu;

use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, channel::Channel};

use crate::ImuSample;

pub static FUSED_IMU: Channel<CriticalSectionRawMutex, ImuSample, 4> = Channel::new();
