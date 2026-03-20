use cybflight_drivers::led::Led;
use embassy_futures::select::{Either, select};
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, watch::Watch};
use embassy_time::Timer;
use embedded_hal::digital::{OutputPin, StatefulOutputPin};

use crate::hal;

#[derive(Clone, Copy, PartialEq, Eq, defmt::Format)]
pub enum SystemStatus {
    /// Boot / pre-calibration: slow heartbeat.
    Booting,
    /// Disarmed, ready to arm: slow blink (1 Hz).
    Disarmed,
    /// Armed: solid on.
    Armed,
    /// Failsafe active: rapid blink.
    Failsafe,
}

pub static STATUS: Watch<CriticalSectionRawMutex, SystemStatus, 2> = Watch::new();

/// Returns the blink pattern for a given status as `(on_ms, off_ms)` phases.
fn pattern(status: SystemStatus) -> &'static [(u64, u64)] {
    match status {
        SystemStatus::Booting => &[(200, 800)],
        SystemStatus::Disarmed => &[(500, 500)],
        SystemStatus::Armed => &[(1000, 0)],
        SystemStatus::Failsafe => &[(50, 50)],
    }
}

type StatusLed = Led<hal::gpio::Output<'static>>;

#[embassy_executor::task]
pub async fn task(led: StatusLed) {
    run(led).await
}

pub async fn run<P: OutputPin + StatefulOutputPin>(mut led: Led<P>) {
    let mut receiver = STATUS.receiver().unwrap();
    let mut status = receiver.changed().await;

    'outer: loop {
        let phases = pattern(status);
        for &(on_ms, off_ms) in phases {
            led.on();
            if on_ms > 0 {
                match select(Timer::after_millis(on_ms), receiver.changed()).await {
                    Either::Second(new) => {
                        status = new;
                        continue 'outer;
                    }
                    _ => {}
                }
            }
            led.off();
            if off_ms > 0 {
                match select(Timer::after_millis(off_ms), receiver.changed()).await {
                    Either::Second(new) => {
                        status = new;
                        continue 'outer;
                    }
                    _ => {}
                }
            }
        }
    }
}
