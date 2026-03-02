use cybflight_drivers::led::Led;
use embassy_futures::select::{select, Either};
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, watch::Watch};
use embassy_time::Timer;
use embedded_hal::digital::{OutputPin, StatefulOutputPin};

#[derive(Clone, Copy, PartialEq, Eq, defmt::Format)]
pub enum SystemStatus {
    Alive,
}

pub static STATUS: Watch<CriticalSectionRawMutex, SystemStatus, 2> = Watch::new();

/// Returns the blink pattern for a given status as `(on_ms, off_ms)` phases.
fn pattern(status: SystemStatus) -> &'static [(u64, u64)] {
    match status {
        SystemStatus::Alive => &[(500, 500)],
    }
}

pub async fn run<P: OutputPin + StatefulOutputPin>(
    mut led0: Led<P>,
    mut led1: Led<P>,
    mut led2: Led<P>,
) {
    led1.off();
    led2.off();

    let mut receiver = STATUS.receiver().unwrap();
    let mut status = receiver.changed().await;

    'outer: loop {
        let phases = pattern(status);
        for &(on_ms, off_ms) in phases {
            led0.on();
            if on_ms > 0 {
                match select(Timer::after_millis(on_ms), receiver.changed()).await {
                    Either::Second(new) => {
                        status = new;
                        continue 'outer;
                    }
                    _ => {}
                }
            }
            led0.off();
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
