//! RC receiver sensor tasks for CRSF and GHST protocols.
//!
//! Each task reads frames from the UART, publishes RC channel data and link
//! statistics to the pub/sub channels, and sends telemetry in the inter-frame gaps.

use embassy_time::{Instant, Timer};

use crate::hal;
use crate::motors::ARM_STATE;
use cybflight_msgs as msgs;

pub type RcUart = hal::usart::BufferedUart<'static>;

/// RC arm channel index (0-based). Channel 6 on the transmitter.
const ARM_CHANNEL: usize = 5;
/// PWM threshold: armed when channel value exceeds this.
const ARM_THRESHOLD: u16 = 1500;

fn publish_arm_state(channels: &[u16; 16], channel_count: u8, was_armed: &mut bool) {
    if (channel_count as usize) <= ARM_CHANNEL {
        return;
    }
    let armed = channels[ARM_CHANNEL] > ARM_THRESHOLD;
    if armed != *was_armed {
        *was_armed = armed;
        if armed {
            defmt::info!("RC: ARMED (ch6={})", channels[ARM_CHANNEL]);
        } else {
            defmt::info!("RC: DISARMED (ch6={})", channels[ARM_CHANNEL]);
        }
        ARM_STATE.signal(msgs::ArmDisarm {
            timestamp: Instant::now(),
            armed,
        });
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
        Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleAttitude, 4, 4, 1>;

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
            let mut was_armed = false;

            loop {
                match self.crsf.read_frame().await {
                    Ok(event) => {
                        let now = Instant::now();
                        match event {
                            CrsfEvent::RcChannelsPacked(rc)
                            | CrsfEvent::SubsetRcChannels(rc) => {
                                rc_pub.publish_immediate(msgs::RcInput {
                                    timestamp: now,
                                    channels: rc.channels,
                                    channel_count: rc.channel_count,
                                });
                                publish_arm_state(&rc.channels, rc.channel_count, &mut was_armed);

                                // Send one telemetry frame per RC frame received
                                self.send_telemetry().await;
                            }
                            CrsfEvent::LinkStatistics(stats)
                            | CrsfEvent::LinkStatisticsTx(stats) => {
                                link_pub.publish_immediate(msgs::RcLinkStatus {
                                    timestamp: now,
                                    rssi_dbm: stats.rssi_dbm,
                                    link_quality: stats.link_quality,
                                    snr: stats.snr,
                                    rf_mode: stats.rf_mode,
                                });
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
    use embassy_time::{with_timeout, Duration};

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
            let mut was_armed = false;

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

                                rc_pub.publish_immediate(msgs::RcInput {
                                    timestamp: now,
                                    channels: rc.channels,
                                    channel_count: rc.channel_count,
                                });
                                publish_arm_state(&rc.channels, rc.channel_count, &mut was_armed);

                                // Guard delay: GHST requires 1ms minimum gap after
                                // the last received byte before transmitting telemetry
                                // on the half-duplex bus (BF: GHST_RX_TO_TELEMETRY_MIN_US).
                                Timer::after_micros(1000).await;
                                self.send_telemetry().await;
                            }
                            GhstEvent::RssiFrame(stats) => {
                                defmt::debug!(
                                    "GHST link: rssi={}dBm lq={}% rf_mode={}",
                                    stats.rssi_dbm,
                                    stats.link_quality,
                                    stats.rf_mode,
                                );
                                link_pub.publish_immediate(msgs::RcLinkStatus {
                                    timestamp: now,
                                    rssi_dbm: stats.rssi_dbm,
                                    link_quality: stats.link_quality,
                                    snr: stats.snr,
                                    rf_mode: stats.rf_mode,
                                });
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
