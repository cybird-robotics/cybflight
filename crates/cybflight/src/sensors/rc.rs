//! RC receiver sensor tasks for CRSF and GHST protocols.
//!
//! Each task reads frames from the UART, publishes RC channel data and link
//! statistics to the pub/sub channels, and sends telemetry in the inter-frame gaps.

use core::cell::{Cell, RefCell};
use core::sync::atomic::Ordering;

use embassy_sync::blocking_mutex::{raw::CriticalSectionRawMutex, Mutex as BlockingMutex};
use embassy_time::Instant;

use crate::hal;
use crate::motors::ARM_STATE;
use crate::status;
use cybflight_msgs as msgs;

/// Latest gate-rejection reason from the most recent arm attempt, plus the
/// instant it was latched. Cleared on disarm and on successful arm. Read by
/// the `health` shell command so users can ask "why didn't it arm?" after
/// the fact, even after the per-attempt summary line has scrolled away.
pub static LATEST_BLOCK_REASON:
    BlockingMutex<CriticalSectionRawMutex, Cell<Option<(BlockReason, Instant)>>> =
    BlockingMutex::new(Cell::new(None));

/// Most recent RC channel frame, mirrored into a shared cell as it is
/// published to `RC_INPUT`. The `health` shell command reads this synchronously
/// — a fresh `RC_INPUT.subscriber()` cannot recover history, so peeking the
/// PubSub directly returns no data even when frames are arriving normally.
pub static LATEST_RC_INPUT:
    BlockingMutex<CriticalSectionRawMutex, RefCell<Option<msgs::RcInput>>> =
    BlockingMutex::new(RefCell::new(None));

/// Most recent CRSF/GHST link statistics, same caching rationale as
/// `LATEST_RC_INPUT`.
pub static LATEST_RC_LINK_STATUS:
    BlockingMutex<CriticalSectionRawMutex, RefCell<Option<msgs::RcLinkStatus>>> =
    BlockingMutex::new(RefCell::new(None));

pub type RcUart = hal::usart::BufferedUart<'static>;

// ---------------------------------------------------------------------------
// Arming parameters
// ---------------------------------------------------------------------------

/// RC arm channel index (0-based). Channel 6 on the transmitter (AUX2).
pub(crate) const ARM_CHANNEL: usize = 5;
/// PWM threshold: armed when channel value exceeds this (µs).
pub(crate) const ARM_THRESHOLD: u16 = 1500;
/// Throttle channel index (AETR order: index 2 = throttle).
pub(crate) const THROTTLE_CHANNEL: usize = 2;
/// Throttle must be below this to arm (µs). Matches BF `rxConfig.mincheck` default.
pub(crate) const THROTTLE_MINCHECK: u16 = 1050;
/// Arm switch must be held for this long before arming (ms).
/// BF does not debounce the arm switch; this is a cybflight safety addition.
const ARM_SWITCH_HOLD_MS: u64 = 100;

/// Hold time after the arm switch is flipped before we print the rejection
/// summary if gates have not passed. Long enough that a normal-debounce arm
/// completes silently before the deadline; short enough that a stuck attempt
/// gets one immediate "ARM rejected" line.
const ARM_REJECT_SUMMARY_MS: u64 = 200;
/// Minimum link quality to allow arming [0..100].
pub(crate) const MIN_LINK_QUALITY: u8 = 50;
/// Link stats older than this block arming (ms). Mirrors BF `ARMING_DISABLED_RX_FAILSAFE`.
pub(crate) const LINK_STATS_MAX_AGE_MS: u64 = 500;
/// Tilt envelope: roll/pitch must be within this magnitude (degrees).
pub(crate) const MAX_TILT_DEG: f32 = 30.0;
/// ESKF/Mahony cross-check tolerance for roll & pitch (degrees).
pub(crate) const MAX_ESKF_MAHONY_DISAGREE_DEG: f32 = 10.0;

// ---------------------------------------------------------------------------
// Arming state machine
// ---------------------------------------------------------------------------

/// Arming state machine with debounce, throttle gate, link quality gate,
/// and estimator readiness gate.
///
/// Enforces four pre-conditions before allowing arm:
/// 1. Throttle at minimum (BF: `ARMING_DISABLED_THROTTLE`, threshold = `mincheck`)
/// 2. RC link active with acceptable quality (BF: `ARMING_DISABLED_RX_FAILSAFE`)
/// 3. ESKF estimator initialised and running (`est_eskf` only)
/// 4. Arm switch held for debounce duration (cybflight safety addition)
///
/// Disarming via switch is always immediate (no debounce) for safety.
///
/// On every arm attempt (switch flipped to arm position), the state machine
/// latches the latest gate-rejection reason and prints a single summary
/// after `ARM_REJECT_SUMMARY_MS`, so the operator can confirm the FCU is
/// alive and see *why* arming was refused.
struct ArmStateMachine {
    armed: bool,
    /// Timestamp when arm switch first entered "arm" position (for hold debounce).
    switch_arm_start: Option<Instant>,
    /// Sticky timestamp of the current arm attempt. Set on the rising edge of
    /// the arm switch, cleared on the falling edge. Distinct from
    /// `switch_arm_start` (which resets every time a gate fails).
    attempt_start: Option<Instant>,
    /// Whether we have already printed the rejection summary for the
    /// current attempt. Reset on the falling edge.
    summary_printed: bool,
    /// Latest gate-rejection reason for the current attempt.
    last_block_reason: Option<BlockReason>,
    /// Last known link quality [0..100].
    link_quality: u8,
    /// Whether any link stats frame has been received.
    link_active: bool,
    /// Timestamp of the most recent link stats frame.
    link_stats_time: Instant,
}

#[derive(Clone, Copy, defmt::Format)]
pub enum BlockReason {
    Failsafe,
    ThrottleNotMin { value: u16, max: u16 },
    LinkInactive,
    LinkStale { age_ms: u64 },
    LinkLowQuality { quality: u8, min: u8 },
    EstimatorNotReady,
    TiltOutOfEnvelope { roll_deg: f32, pitch_deg: f32 },
    SensorHealthDegraded { bits: u8, required: u8 },
    EskfDegraded { faults: u32 },
    MahonyNotReady,
    EskfMahonyTiltDisagreement {
        eskf_roll_deg: f32,
        eskf_pitch_deg: f32,
        mahony_roll_deg: f32,
        mahony_pitch_deg: f32,
    },
}

impl ArmStateMachine {
    fn new() -> Self {
        Self {
            armed: false,
            switch_arm_start: None,
            attempt_start: None,
            summary_printed: false,
            last_block_reason: None,
            link_quality: 0,
            link_active: false,
            link_stats_time: Instant::now(),
        }
    }

    /// Record a gate rejection. Latches the reason; if the attempt has
    /// been ongoing for `ARM_REJECT_SUMMARY_MS` and we have not yet
    /// printed a summary, emit one now.
    fn note_block(&mut self, now: Instant, reason: BlockReason) {
        self.switch_arm_start = None;
        self.last_block_reason = Some(reason);
        LATEST_BLOCK_REASON.lock(|c| c.set(Some((reason, now))));
        if self.summary_printed {
            return;
        }
        if let Some(start) = self.attempt_start {
            if now.duration_since(start).as_millis() as u64 >= ARM_REJECT_SUMMARY_MS {
                defmt::info!("ARM rejected: {}", reason);
                self.summary_printed = true;
            }
        }
    }

    /// Feed latest link statistics. Call on every link-stats frame.
    fn update_link(&mut self, quality: u8) {
        self.link_quality = quality;
        self.link_active = true;
        self.link_stats_time = Instant::now();
    }

    /// Evaluate arm/disarm on each RC channel frame.
    fn update_channels(&mut self, channels: &[u16; 16], channel_count: u8) {
        if (channel_count as usize) <= ARM_CHANNEL {
            return;
        }

        let now = Instant::now();
        let switch_armed = channels[ARM_CHANNEL] > ARM_THRESHOLD;

        // --- Disarm: always immediate, no gates ---
        if !switch_armed {
            self.switch_arm_start = None;
            self.attempt_start = None;
            self.summary_printed = false;
            self.last_block_reason = None;
            LATEST_BLOCK_REASON.lock(|c| c.set(None));
            if self.armed {
                self.armed = false;
                defmt::info!(
                    "DISARMED (ch{}={})",
                    ARM_CHANNEL + 1,
                    channels[ARM_CHANNEL]
                );
                ARM_STATE.signal(msgs::ArmDisarm {
                    timestamp: now,
                    armed: false,
                });
                status::STATUS
                    .sender()
                    .send(status::SystemStatus::Disarmed);
            }
            return;
        }

        // --- Switch is in arm position; check gates ---
        if self.armed {
            return; // already armed
        }

        // Rising edge: open a new attempt window so the rejection summary
        // is bounded to one print regardless of how long the user holds.
        self.attempt_start.get_or_insert(now);

        // Gate 1: failsafe not active (BF: ARMING_DISABLED_FAILSAFE)
        if crate::control::failsafe::FAILSAFE_ACTIVE.load(Ordering::Acquire) {
            self.note_block(now, BlockReason::Failsafe);
            return;
        }

        // Gate 2: throttle at minimum (BF: ARMING_DISABLED_THROTTLE)
        if channels[THROTTLE_CHANNEL] > THROTTLE_MINCHECK {
            self.note_block(
                now,
                BlockReason::ThrottleNotMin {
                    value: channels[THROTTLE_CHANNEL],
                    max: THROTTLE_MINCHECK,
                },
            );
            return;
        }

        // Gate 3: link active & quality (BF: ARMING_DISABLED_RX_FAILSAFE)
        if !self.link_active {
            self.note_block(now, BlockReason::LinkInactive);
            return;
        }
        let link_age_ms = now.duration_since(self.link_stats_time).as_millis() as u64;
        if link_age_ms > LINK_STATS_MAX_AGE_MS {
            self.note_block(now, BlockReason::LinkStale { age_ms: link_age_ms });
            return;
        }
        if self.link_quality < MIN_LINK_QUALITY {
            self.note_block(
                now,
                BlockReason::LinkLowQuality {
                    quality: self.link_quality,
                    min: MIN_LINK_QUALITY,
                },
            );
            return;
        }

        // Gate 4: ESKF estimator ready (Mahony has no convergence phase)
        #[cfg(feature = "est_eskf")]
        if !crate::estimation::ESTIMATOR_READY.load(Ordering::Acquire) {
            self.note_block(now, BlockReason::EstimatorNotReady);
            return;
        }

        // Gate 5: switch hold duration (debounce)
        let start = *self.switch_arm_start.get_or_insert(now);
        if now.duration_since(start).as_millis() < ARM_SWITCH_HOLD_MS {
            return;
        }

        // All gates passed — arm
        self.armed = true;
        self.attempt_start = None;
        self.summary_printed = false;
        self.last_block_reason = None;
        LATEST_BLOCK_REASON.lock(|c| c.set(None));
        defmt::info!(
            "ARMED (ch{}={}, throttle={}, lq={}%)",
            ARM_CHANNEL + 1,
            channels[ARM_CHANNEL],
            channels[THROTTLE_CHANNEL],
            self.link_quality,
        );
        ARM_STATE.signal(msgs::ArmDisarm {
            timestamp: now,
            armed: true,
        });
        status::STATUS
            .sender()
            .send(status::SystemStatus::Armed);
    }
}

// ---------------------------------------------------------------------------
// CRSF
// ---------------------------------------------------------------------------

#[cfg(feature = "rx_crsf")]
pub mod crsf_runner {
    use super::*;
    use cybflight_drivers::rc::crsf::{Crsf, CrsfEvent};
    use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::Subscriber};

    type AttitudeSub =
        Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleAttitude, 4, 6, 1>;

    /// CRSF runner: reads frames, publishes channels/link stats, sends telemetry.
    pub struct CrsfRunner {
        crsf: Crsf<RcUart>,
        telemetry_idx: u8,
        attitude_sub: Option<AttitudeSub>,
    }

    impl CrsfRunner {
        pub fn new(uart: RcUart) -> Self {
            let attitude_sub = super::super::VEHICLE_ATTITUDE.subscriber().ok();
            Self {
                crsf: Crsf::new(uart),
                telemetry_idx: 0,
                attitude_sub,
            }
        }

        pub async fn run(&mut self) -> ! {
            let rc_pub = super::super::RC_INPUT.immediate_publisher();
            let link_pub = super::super::RC_LINK_STATUS.immediate_publisher();
            let mut arm = ArmStateMachine::new();

            loop {
                match self.crsf.read_frame().await {
                    Ok(event) => {
                        let now = Instant::now();
                        match event {
                            CrsfEvent::RcChannelsPacked(rc)
                            | CrsfEvent::SubsetRcChannels(rc) => {
                                let frame = msgs::RcInput {
                                    timestamp: now,
                                    channels: rc.channels,
                                    channel_count: rc.channel_count,
                                };
                                rc_pub.publish_immediate(frame.clone());
                                LATEST_RC_INPUT.lock(|c| c.replace(Some(frame)));
                                arm.update_channels(&rc.channels, rc.channel_count);

                                // Send one telemetry frame per RC frame received
                                self.send_telemetry().await;
                            }
                            CrsfEvent::LinkStatistics(stats)
                            | CrsfEvent::LinkStatisticsTx(stats) => {
                                arm.update_link(stats.link_quality);
                                let link = msgs::RcLinkStatus {
                                    timestamp: now,
                                    rssi_dbm: stats.rssi_dbm,
                                    link_quality: stats.link_quality,
                                    snr: stats.snr,
                                    rf_mode: stats.rf_mode,
                                };
                                link_pub.publish_immediate(link.clone());
                                LATEST_RC_LINK_STATUS.lock(|c| c.replace(Some(link)));
                            }
                            CrsfEvent::SpeedProposal { port_id, baud } => {
                                defmt::info!(
                                    "CRSF V3 speed proposal: port={} baud={}",
                                    port_id,
                                    baud
                                );
                                // Accept the speed proposal
                                if let Err(e) =
                                    self.crsf.accept_speed_proposal(port_id, baud).await
                                {
                                    defmt::warn!("Failed to send speed response: {}", e);
                                }
                                // Note: actual UART baudrate change would need to be
                                // handled by the board_init layer via take_baudrate_hint()
                            }
                            CrsfEvent::Other { .. } => {}
                        }
                    }
                    Err(e) => {
                        defmt::warn!("CRSF read error: {}", e);
                    }
                }
            }
        }

        /// Send the next scheduled telemetry frame.
        /// Round-robin: heartbeat → attitude → flight mode
        ///
        // TODO: expand telemetry schedule once more sensor channels exist:
        //   - battery: subscribe to BATTERY channel, send write_battery()
        //   - GPS: subscribe to GPS channel, send write_gps()
        //   - baro: subscribe to BARO channel, send write_baro_altitude() / write_vario()
        //   - flight mode: read actual mode from control logic instead of hardcoded "ACRO"
        async fn send_telemetry(&mut self) {
            match self.telemetry_idx % 3 {
                0 => {
                    if let Err(e) = self.crsf.write_heartbeat().await {
                        defmt::warn!("CRSF telemetry heartbeat error: {}", e);
                    }
                }
                1 => {
                    // Send attitude if available
                    if let Some(ref mut sub) = self.attitude_sub {
                        if let Some(att) = sub.try_next_message_pure() {
                            let q = att.orientation;
                            let (roll, pitch, yaw) = q.euler_angles();
                            if let Err(e) =
                                self.crsf.write_attitude(pitch, roll, yaw).await
                            {
                                defmt::warn!("CRSF telemetry attitude error: {}", e);
                            }
                        }
                    }
                }
                _ => {
                    if let Err(e) = self.crsf.write_flight_mode(b"ACRO").await {
                        defmt::warn!("CRSF telemetry flight mode error: {}", e);
                    }
                }
            }
            self.telemetry_idx = self.telemetry_idx.wrapping_add(1);
        }
    }

    #[embassy_executor::task]
    pub async fn crsf_task(uart: RcUart) {
        let mut runner = CrsfRunner::new(uart);
        runner.run().await;
    }
}

// ---------------------------------------------------------------------------
// GHST
// ---------------------------------------------------------------------------

#[cfg(feature = "rx_ghst")]
pub mod ghst_runner {
    use super::*;
    use cybflight_drivers::rc::ghst::{Ghst, GhstEvent};
    use embassy_time::{with_timeout, Duration, Timer};

    /// GHST runner: reads frames, publishes channels/link stats, sends telemetry.
    pub struct GhstRunner {
        ghst: Ghst<RcUart>,
        telemetry_idx: u8,
    }

    /// Time to wait for the first GHST frame before warning about pin assignment.
    const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(3);
    /// Time to wait for subsequent frames before warning about signal loss.
    const FRAME_TIMEOUT: Duration = Duration::from_secs(1);

    impl GhstRunner {
        pub fn new(uart: RcUart) -> Self {
            Self {
                ghst: Ghst::new(uart),
                telemetry_idx: 0,
            }
        }

        pub async fn run(&mut self) -> ! {
            let rc_pub = super::super::RC_INPUT.immediate_publisher();
            let link_pub = super::super::RC_LINK_STATUS.immediate_publisher();
            let mut arm = ArmStateMachine::new();

            let mut frame_count: u32 = 0;
            let mut err_count: u32 = 0;
            let mut timeout_count: u32 = 0;
            let mut got_first_frame = false;

            defmt::info!("GHST: waiting for first frame (timeout {}s)...",
                FIRST_FRAME_TIMEOUT.as_secs());

            loop {
                let timeout = if got_first_frame {
                    FRAME_TIMEOUT
                } else {
                    FIRST_FRAME_TIMEOUT
                };

                match with_timeout(timeout, self.ghst.read_frame()).await {
                    Err(_elapsed) => {
                        timeout_count = timeout_count.wrapping_add(1);
                        if !got_first_frame {
                            defmt::error!(
                                "GHST: NO DATA after {}s — check wiring! \
                                 Verify RC receiver is connected to the correct UART TX pin \
                                 (half-duplex). (timeouts={})",
                                FIRST_FRAME_TIMEOUT.as_secs(),
                                timeout_count,
                            );
                        } else {
                            defmt::warn!(
                                "GHST: frame timeout ({}s no data, timeouts={}, frames={}, errs={})",
                                FRAME_TIMEOUT.as_secs(),
                                timeout_count,
                                frame_count,
                                err_count,
                            );
                        }
                        continue;
                    }
                    Ok(Err(e)) => {
                        err_count = err_count.wrapping_add(1);
                        defmt::warn!("GHST read error: {} (total errs={})", e, err_count);
                        continue;
                    }
                    Ok(Ok(event)) => {
                        if !got_first_frame {
                            got_first_frame = true;
                            defmt::info!("GHST: first frame received OK");
                        }
                        frame_count = frame_count.wrapping_add(1);
                        let now = Instant::now();
                        match event {
                            GhstEvent::RcChannels(rc) => {
                                if frame_count % 500 == 1 {
                                    defmt::info!(
                                        "GHST ch[0..4]=[{},{},{},{}] frames={} errs={} timeouts={}",
                                        rc.channels[0],
                                        rc.channels[1],
                                        rc.channels[2],
                                        rc.channels[3],
                                        frame_count,
                                        err_count,
                                        timeout_count,
                                    );
                                }

                                let frame = msgs::RcInput {
                                    timestamp: now,
                                    channels: rc.channels,
                                    channel_count: rc.channel_count,
                                };
                                rc_pub.publish_immediate(frame.clone());
                                LATEST_RC_INPUT.lock(|c| c.replace(Some(frame)));
                                arm.update_channels(&rc.channels, rc.channel_count);

                                // Guard delay: GHST requires 1ms minimum gap after
                                // the last received byte before transmitting telemetry
                                // on the half-duplex bus (BF: GHST_RX_TO_TELEMETRY_MIN_US).
                                Timer::after_micros(1000).await;
                                self.send_telemetry().await;
                            }
                            GhstEvent::RssiFrame(stats) => {
                                arm.update_link(stats.link_quality);
                                defmt::debug!(
                                    "GHST link: rssi={}dBm lq={}% rf_mode={}",
                                    stats.rssi_dbm,
                                    stats.link_quality,
                                    stats.rf_mode,
                                );
                                let link = msgs::RcLinkStatus {
                                    timestamp: now,
                                    rssi_dbm: stats.rssi_dbm,
                                    link_quality: stats.link_quality,
                                    snr: stats.snr,
                                    rf_mode: stats.rf_mode,
                                };
                                link_pub.publish_immediate(link.clone());
                                LATEST_RC_LINK_STATUS.lock(|c| c.replace(Some(link)));
                            }
                            GhstEvent::Other { frame_type } => {
                                defmt::debug!("GHST unknown frame type={:#x}", frame_type);
                            }
                        }
                    }
                }
            }
        }

        /// Send the next scheduled telemetry frame.
        /// Round-robin: pack → magbaro
        ///
        // TODO: expand telemetry schedule once more sensor channels exist:
        //   - battery: subscribe to BATTERY channel, send real values in write_pack()
        //   - GPS: subscribe to GPS channel, send write_gps_primary() / write_gps_secondary()
        //   - baro/mag: subscribe to BARO + VEHICLE_ATTITUDE channels, send real
        //     yaw/alt/vario in write_magbaro()
        async fn send_telemetry(&mut self) {
            match self.telemetry_idx % 2 {
                0 => {
                    // Battery pack status — placeholder values until battery system exists
                    if let Err(e) = self.ghst.write_pack(0, 0, 0, false).await {
                        defmt::warn!("GHST telemetry pack error: {}", e);
                    }
                }
                _ => {
                    // Magbaro — placeholder values
                    if let Err(e) = self.ghst.write_magbaro(0, 0, 0, 0).await {
                        defmt::warn!("GHST telemetry magbaro error: {}", e);
                    }
                }
            }
            self.telemetry_idx = self.telemetry_idx.wrapping_add(1);
        }
    }

    #[embassy_executor::task]
    pub async fn ghst_task(uart: RcUart) {
        let mut runner = GhstRunner::new(uart);
        runner.run().await;
    }
}
