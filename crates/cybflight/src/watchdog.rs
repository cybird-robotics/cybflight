use crate::hal::pac;

/// Initialize and start the Independent Watchdog (IWDG1).
///
/// Once started, IWDG cannot be stopped — it must be fed periodically via
/// [`feed()`] or the MCU will reset. This is intentional: if the firmware
/// hangs completely, the watchdog resets the system and ESCs disarm on their
/// own when DShot frames stop arriving.
///
/// Configuration: LSI ≈ 32 kHz, prescaler /32 → 1 kHz tick, reload = 500
/// → **~500 ms timeout**.
pub fn init() {
    let iwdg = pac::IWDG1;

    // Unlock PR/RLR registers
    iwdg.kr().write(|w| w.set_key(pac::iwdg::vals::Key::ENABLE));

    // Prescaler /32  → 32 kHz / 32 = 1 kHz
    iwdg.pr().write(|w| w.set_pr(pac::iwdg::vals::Pr::DIVIDE_BY32));

    // Reload 500 → 500 ms timeout
    iwdg.rlr().write(|w| w.set_rl(500));

    // Start the watchdog — enables LSI and begins countdown (irreversible).
    // Initially counts with default values (prescaler /4, RL=0xFFF ≈ 512 ms).
    iwdg.kr().write(|w| w.set_key(pac::iwdg::vals::Key::START));

    // Wait for shadow register update (requires LSI running, which START enabled)
    while iwdg.sr().read().pvu() || iwdg.sr().read().rvu() {}

    // Feed to load the new reload value into the down-counter
    feed();

    defmt::info!("IWDG started (~500 ms timeout)");
}

/// Feed (reload) the watchdog counter. Must be called before the timeout
/// expires or the MCU will reset.
#[inline]
pub fn feed() {
    pac::IWDG1
        .kr()
        .write(|w| w.set_key(pac::iwdg::vals::Key::RESET));
}

/// Watchdog feeder task — runs at lowest priority on the thread executor.
///
/// Feeds the IWDG every 200 ms. If the thread executor stalls for >500 ms
/// (the IWDG timeout), the MCU resets.
#[embassy_executor::task]
pub async fn iwdg_feed_task() {
    loop {
        feed();
        embassy_time::Timer::after_millis(200).await;
    }
}
