//! RC receiver sensor tasks for CRSF and GHST protocols.
//!
//! Each task reads frames from the UART, publishes RC channel data and link
//! statistics to the pub/sub channels, and sends telemetry in the inter-frame gaps.

use embassy_time::{Instant, Timer};

use crate::hal;
use cybflight_msgs as msgs;

pub type RcUart = hal::usart::BufferedUart<'static>;

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

    /// GHST runner: reads frames, publishes channels/link stats, sends telemetry.
    pub struct GhstRunner {
        ghst: Ghst<RcUart>,
        telemetry_idx: u8,
    }

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

            loop {
                match self.ghst.read_frame().await {
                    Ok(event) => {
                        let now = Instant::now();
                        match event {
                            GhstEvent::RcChannels(rc) => {
                                rc_pub.publish_immediate(msgs::RcInput {
                                    timestamp: now,
                                    channels: rc.channels,
                                    channel_count: rc.channel_count,
                                });

                                // Guard delay: GHST requires 1ms minimum gap after
                                // the last received byte before transmitting telemetry
                                // on the half-duplex bus (BF: GHST_RX_TO_TELEMETRY_MIN_US).
                                Timer::after_micros(1000).await;
                                self.send_telemetry().await;
                            }
                            GhstEvent::RssiFrame(stats) => {
                                link_pub.publish_immediate(msgs::RcLinkStatus {
                                    timestamp: now,
                                    rssi_dbm: stats.rssi_dbm,
                                    link_quality: stats.link_quality,
                                    snr: stats.snr,
                                    rf_mode: stats.rf_mode,
                                });
                            }
                            GhstEvent::Other { .. } => {}
                        }
                    }
                    Err(e) => {
                        defmt::warn!("GHST read error: {}", e);
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
