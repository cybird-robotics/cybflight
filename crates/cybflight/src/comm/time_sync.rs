//! UTC clock synchronization via ESP32 PTP follow-up.
//!
//! The ESP32 runs SNTP against the GS PC (which acts as NTP server) and
//! periodically sends `WireTimeSync` frames over UART.  Each frame carries:
//!   - `esp_send_ntp_us`:  NTP time captured before UART write (primary)
//!   - `prev_tx_complete_ntp_us`: NTP time captured after the *previous*
//!     write completed (refined — eliminates UART delay estimation)
//!
//! Internal timestamps remain monotonic `Instant`.  UTC conversion happens
//! only at the telemetry serialization boundary via [`to_utc_us`].
//!
//! When time sync is not established (no GS / debug mode), [`to_utc_us`]
//! falls back to boot-relative microseconds so telemetry still works.

use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, Ordering};
use critical_section::Mutex;
use embassy_time::Instant;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Time sync state protected by critical section.
static SYNC_STATE: Mutex<RefCell<Option<SyncState>>> = Mutex::new(RefCell::new(None));

struct SyncState {
    smoothed_offset_us: i64,
    prev_rx_instant_us: u64,
    consecutive_rejects: u8,
    last_ping_rtt_us: u64,
    last_ping_clock_err_us: i64,
}

/// Whether at least one valid time sync has been received.
static SYNCED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Estimated UART frame delay for an 18-byte WireTimeSync frame.
///
/// 18 raw bytes → ~20 COBS-encoded bytes.
/// At 921 600 baud, 10 bits/byte: 20 × 10 / 921 600 ≈ 217 µs.
const UART_FRAME_DELAY_US: i64 = 217;

/// EMA smoothing: α ≈ 26/256 ≈ 0.10.
const EMA_ALPHA_NUM: i64 = 26;
const EMA_ALPHA_DEN: i64 = 256;

/// Reject offset jumps larger than this (µs).
const MAX_OFFSET_JUMP_US: i64 = 10_000; // 10 ms

/// After this many consecutive rejections, accept the new offset as a step
/// change (e.g. NTP server corrected a large drift).
const MAX_CONSECUTIVE_REJECTS: u8 = 5;

// ---------------------------------------------------------------------------
// Diagnostics snapshot (for shell / telemetry)
// ---------------------------------------------------------------------------

/// Time sync diagnostics for display and telemetry.
#[derive(Clone, Copy, defmt::Format)]
pub struct TimeSyncStatus {
    /// Whether time sync is established.
    pub synced: bool,
    /// Current smoothed UTC offset (monotonic + offset = UTC). 0 if unsynced.
    pub offset_us: i64,
    /// Last ping round-trip time in µs. 0 if no ping received.
    pub ping_rtt_us: u64,
    /// Last ping clock error in µs (gs_send - our_estimate). 0 if no ping.
    pub ping_clock_err_us: i64,
}

/// Get a snapshot of current time sync diagnostics.
pub fn status() -> TimeSyncStatus {
    critical_section::with(|cs| match &*SYNC_STATE.borrow_ref(cs) {
        Some(state) => TimeSyncStatus {
            synced: true,
            offset_us: state.smoothed_offset_us,
            ping_rtt_us: state.last_ping_rtt_us,
            ping_clock_err_us: state.last_ping_clock_err_us,
        },
        None => TimeSyncStatus {
            synced: false,
            offset_us: 0,
            ping_rtt_us: 0,
            ping_clock_err_us: 0,
        },
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert a monotonic `Instant` to UTC microseconds since UNIX epoch.
///
/// Falls back to boot-relative microseconds when time sync is not
/// established, so telemetry still works during debug without a GS.
pub fn to_utc_us(instant: Instant) -> i64 {
    critical_section::with(|cs| {
        let monotonic = instant.as_micros() as i64;
        match &*SYNC_STATE.borrow_ref(cs) {
            Some(state) => monotonic + state.smoothed_offset_us,
            None => monotonic,
        }
    })
}

/// Returns `true` once at least one valid sync has been received.
pub fn is_synced() -> bool {
    SYNCED.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Processing incoming messages
// ---------------------------------------------------------------------------

/// Process a `WireTimeSync` frame received from the ESP32.
///
/// Uses the refined PTP follow-up offset when available (previous frame's
/// TX-complete ↔ previous frame's RX instant), falling back to the primary
/// estimate (send-time − RX + UART delay) for the very first frame.
pub fn process_time_sync(
    rx_instant: Instant,
    esp_send_ntp_us: i64,
    prev_tx_complete_ntp_us: i64,
) {
    let rx_us = rx_instant.as_micros() as i64;

    critical_section::with(|cs| {
        let mut state_ref = SYNC_STATE.borrow_ref_mut(cs);

        // Compute raw offset.
        let raw_offset = match &*state_ref {
            Some(state) if prev_tx_complete_ntp_us != 0 && state.prev_rx_instant_us != 0 => {
                // Refined PTP follow-up: no UART delay estimation needed.
                // prev_tx_complete_ntp_us ≈ prev_rx_instant (in shared timebase)
                prev_tx_complete_ntp_us - state.prev_rx_instant_us as i64
            }
            _ => {
                // Primary fallback for first sync frame.
                esp_send_ntp_us - rx_us + UART_FRAME_DELAY_US
            }
        };

        match &mut *state_ref {
            None => {
                // First sync — accept directly.
                *state_ref = Some(SyncState {
                    smoothed_offset_us: raw_offset,
                    prev_rx_instant_us: rx_us as u64,
                    consecutive_rejects: 0,
                    last_ping_rtt_us: 0,
                    last_ping_clock_err_us: 0,
                });
                SYNCED.store(true, Ordering::Relaxed);
                defmt::info!("Time sync: first offset = {} us", raw_offset);
            }
            Some(state) => {
                // Outlier rejection with step-change recovery.
                let delta = raw_offset - state.smoothed_offset_us;
                if delta.unsigned_abs() > MAX_OFFSET_JUMP_US as u64 {
                    state.consecutive_rejects += 1;
                    if state.consecutive_rejects >= MAX_CONSECUTIVE_REJECTS {
                        // Sustained offset change — accept as new baseline.
                        defmt::warn!(
                            "Time sync: {} consecutive rejects, accepting step change {} us",
                            state.consecutive_rejects,
                            delta
                        );
                        state.smoothed_offset_us = raw_offset;
                        state.consecutive_rejects = 0;
                    } else {
                        defmt::warn!("Time sync: offset jump {} us, rejecting ({}/{})",
                            delta, state.consecutive_rejects, MAX_CONSECUTIVE_REJECTS);
                    }
                    state.prev_rx_instant_us = rx_us as u64;
                    return;
                }

                state.consecutive_rejects = 0;

                // EMA smoothing.
                state.smoothed_offset_us = (EMA_ALPHA_NUM * raw_offset
                    + (EMA_ALPHA_DEN - EMA_ALPHA_NUM) * state.smoothed_offset_us)
                    / EMA_ALPHA_DEN;
                state.prev_rx_instant_us = rx_us as u64;
            }
        }
    });
}

/// Process a `WirePingResp` for RTT measurement and clock-sync validation.
pub fn process_ping_resp(
    rx_instant: Instant,
    stm32_send_us: u64,
    _gs_recv_time_us: i64,
    gs_send_time_us: i64,
) {
    let rtt_us = rx_instant.as_micros().saturating_sub(stm32_send_us);
    let one_way_us = rtt_us / 2;

    // Cross-validate: our UTC estimate at RX minus one-way should ≈ gs_send_time.
    let our_estimate = to_utc_us(rx_instant) - one_way_us as i64;
    let clock_err = gs_send_time_us - our_estimate;

    // Store for diagnostics.
    critical_section::with(|cs| {
        if let Some(state) = &mut *SYNC_STATE.borrow_ref_mut(cs) {
            state.last_ping_rtt_us = rtt_us;
            state.last_ping_clock_err_us = clock_err;
        }
    });

    defmt::info!(
        "Ping: RTT={} us, one_way={} us, clock_err={} us",
        rtt_us,
        one_way_us,
        clock_err
    );
}
