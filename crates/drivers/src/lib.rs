#![cfg_attr(not(test), no_std)]

pub mod baro;
pub mod beeper;
pub mod blackbox_storage;
pub mod dshot;
pub mod gimbal;
pub mod gps;
pub mod imu;
pub mod led;
pub mod mag;
pub mod rc;

/// Yield to the executor once, allowing other tasks to run.
/// Essential for cooperative scheduling when polling in a loop on a shared bus.
pub async fn yield_now() {
    let mut yielded = false;
    core::future::poll_fn(|cx| {
        if yielded {
            core::task::Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            core::task::Poll::Pending
        }
    })
    .await;
}
