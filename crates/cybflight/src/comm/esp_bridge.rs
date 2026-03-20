//! ESP bridge — RX (uplink) + TX (downlink) embassy tasks.
//!
//! Uses DMA-backed UART for minimal interrupt overhead:
//! - TX: batch-encodes all frames into one buffer, single DMA transfer per tick
//! - RX: `read_until_idle()` — DMA fills buffer, wakes on idle line detection

use embassy_time::Timer;

use crate::hal::usart::{UartRx, UartTx};
use crate::{control, sensors};
use cybflight_msgs::wire::{
    self, WireArmDisarm, WireAttitudeControlSetpoint, WireBaroSample, WireDshotTelemetry,
    WireGpsFix, WireImu, WireMagSample, WireMessage,
    WireOcpSolverOutput, WirePing, WirePingResp, WirePositionControlSetpoint, WirePose,
    WireRcInput, WireRcLinkStatus, WireTimeSync, WireTimeSyncStatus, WireVehicleAttitude,
    WireVehicleOdometry,
};

use super::{encode_frame, FrameAccumulator};

// ---------------------------------------------------------------------------
// RX task — receive uplink messages from ESP32 via DMA + idle line detection
// ---------------------------------------------------------------------------

#[embassy_executor::task]
pub async fn esp_bridge_rx_task(mut rx: UartRx<'static, crate::hal::mode::Async>) {
    let pose_pub = sensors::VICON_POSE.immediate_publisher();
    let mut acc = FrameAccumulator::new();
    let mut read_buf = [0u8; 128];

    defmt::info!("ESP bridge RX task started (DMA)");

    loop {
        match rx.read_until_idle(&mut read_buf).await {
            Ok(n) => {
                for &b in &read_buf[..n] {
                    if let Some(len) = acc.feed(b) {
                        let frame = &acc.data()[..len];
                        if len < wire::FRAME_HEADER_SIZE {
                            continue;
                        }
                        let msg_id = frame[0];
                        // frame[1] is seq — ignored for now
                        let payload = &frame[wire::FRAME_HEADER_SIZE..len];

                        match msg_id {
                            wire::msg_id::POSE => {
                                if let Some(wp) = WirePose::from_bytes(payload) {
                                    let rx_time = embassy_time::Instant::now();
                                    pose_pub.publish_immediate(wp.to_vicon_pose(rx_time));
                                }
                            }
                            wire::msg_id::TIME_SYNC => {
                                if let Some(ts) = WireTimeSync::from_bytes(payload) {
                                    let rx_time = embassy_time::Instant::now();
                                    super::time_sync::process_time_sync(
                                        rx_time,
                                        ts.esp_send_ntp_us,
                                        ts.prev_tx_complete_ntp_us,
                                    );
                                }
                            }
                            wire::msg_id::PING_RESP => {
                                if let Some(pr) = WirePingResp::from_bytes(payload) {
                                    let rx_time = embassy_time::Instant::now();
                                    super::time_sync::process_ping_resp(
                                        rx_time,
                                        pr.stm32_send_us,
                                        pr.gs_recv_time_us,
                                        pr.gs_send_time_us,
                                    );
                                }
                            }
                            _ => {
                                defmt::warn!("ESP bridge: unknown msg_id={}", msg_id);
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // UART error — wait briefly before retrying.
                Timer::after_millis(10).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TX task — batch-encode and DMA-send downlink messages at 100 Hz
// ---------------------------------------------------------------------------

/// Timer period for TX polling (100 Hz).
const TX_PERIOD_MS: u64 = 10;

/// Ping interval in TX ticks (500 × 10ms = 5s).
const PING_INTERVAL_TICKS: u32 = 500;

/// Convert a monotonic Instant to UTC microseconds for telemetry.
fn utc_ts(instant: embassy_time::Instant) -> u64 {
    super::time_sync::to_utc_us(instant) as u64
}

#[embassy_executor::task]
pub async fn esp_bridge_tx_task(mut tx: UartTx<'static, crate::hal::mode::Async>) {
    let mut imu1_sub = sensors::IMU_1.subscriber().unwrap();
    let mut imu2_sub = sensors::IMU_2.subscriber().unwrap();
    let mut att_sub = sensors::VEHICLE_ATTITUDE.subscriber().unwrap();
    let mut rc_sub = sensors::RC_INPUT.subscriber().unwrap();
    let mut rc_link_sub = sensors::RC_LINK_STATUS.subscriber().unwrap();
    let mut dshot_sub = sensors::DSHOT_TELEMETRY.subscriber().unwrap();
    let mut ocp_sub = control::OCP_SOLVER_OUTPUT.subscriber().unwrap();
    let mut gps_sub = sensors::GPS_FIX.subscriber().unwrap();
    let mut mag_ext_sub = sensors::MAG_EXT.subscriber().unwrap();
    let mut mag_int_sub = sensors::MAG_INT.subscriber().unwrap();
    let mut baro1_sub = sensors::BARO_1.subscriber().unwrap();
    let mut baro2_sub = sensors::BARO_2.subscriber().unwrap();
    let mut att_ctrl_sub = control::ATTITUDE_CONTROL_SETPOINT.subscriber().unwrap();
    let mut pos_ctrl_sub = control::POSITION_CONTROL_SETPOINT.subscriber().unwrap();
    let mut arm_sub = crate::ARM_DISARM.subscriber().unwrap();
    let mut odom_sub = sensors::VEHICLE_ODOMETRY.subscriber().unwrap();

    let mut seq: u8 = 0;
    // Batch buffer: holds all COBS-encoded frames for one tick.
    // Worst case ~16 frames × ~85 bytes each ≈ 1360 bytes; 1536 gives headroom.
    let mut batch = [0u8; 1536];

    // Ping state.
    let mut ping_counter: u32 = 0;
    let mut ping_id: u32 = 0;

    defmt::info!("ESP bridge TX task started (DMA)");

    loop {
        Timer::after_millis(TX_PERIOD_MS).await;

        // Drain each subscriber — only send the latest message per tick.
        // try_next_message_pure() returns None when no unread messages,
        // so low-rate channels send at their actual rate (no duplicates)
        // and high-rate channels are capped at 100 Hz.
        //
        // All frames are batch-encoded into one contiguous buffer,
        // then sent as a single DMA transfer (one interrupt total).
        //
        // All telemetry timestamps are converted to UTC via time_sync.

        let mut pos = 0;

        if let Some(m) = drain_latest(&mut imu1_sub) {
            let mut w = WireImu::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut imu2_sub) {
            let mut w = WireImu::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_with_id(&w, wire::msg_id::IMU_2, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut att_sub) {
            let mut w = WireVehicleAttitude::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut rc_sub) {
            let mut w = WireRcInput::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut rc_link_sub) {
            let mut w = WireRcLinkStatus::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut dshot_sub) {
            let mut w = WireDshotTelemetry::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut ocp_sub) {
            let mut w = WireOcpSolverOutput::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut gps_sub) {
            let mut w = WireGpsFix::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut mag_ext_sub) {
            let mut w = WireMagSample::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut mag_int_sub) {
            let mut w = WireMagSample::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_with_id(&w, wire::msg_id::MAG_INT, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut baro1_sub) {
            let mut w = WireBaroSample::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut baro2_sub) {
            let mut w = WireBaroSample::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_with_id(&w, wire::msg_id::BARO_2, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut att_ctrl_sub) {
            let mut w = WireAttitudeControlSetpoint::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut pos_ctrl_sub) {
            let mut w = WirePositionControlSetpoint::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut arm_sub) {
            let mut w = WireArmDisarm::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut odom_sub) {
            let mut w = WireVehicleOdometry::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }

        // Periodic ping + time sync status telemetry (every 5s).
        ping_counter += 1;
        if ping_counter >= PING_INTERVAL_TICKS {
            ping_counter = 0;
            let ping = WirePing {
                ping_id,
                stm32_send_us: embassy_time::Instant::now().as_micros(),
            };
            ping_id += 1;
            pos += encode_and_advance(&ping, &mut seq, &mut batch[pos..]);

            // Send time sync diagnostics to ground.
            let s = super::time_sync::status();
            let status = WireTimeSyncStatus {
                synced: s.synced as u8,
                offset_us: s.offset_us,
                ping_rtt_us: s.ping_rtt_us,
                ping_clock_err_us: s.ping_clock_err_us,
            };
            pos += encode_and_advance(&status, &mut seq, &mut batch[pos..]);
        }

        if pos > 0 {
            let _ = tx.write(&batch[..pos]).await;
        }
    }
}

/// COBS-encode a wire message into the batch buffer. Returns bytes written.
fn encode_and_advance<W: WireMessage>(msg: &W, seq: &mut u8, dest: &mut [u8]) -> usize {
    encode_with_id(msg, W::MSG_ID, seq, dest)
}

/// COBS-encode a wire message with an explicit msg_id (for secondary instances
/// like IMU_2, BARO_2, MAG_INT that share a wire layout with their primary).
fn encode_with_id<W: WireMessage>(msg: &W, msg_id: u8, seq: &mut u8, dest: &mut [u8]) -> usize {
    let len = encode_frame(msg_id, *seq, msg.as_bytes(), dest);
    *seq = seq.wrapping_add(1);
    len
}

/// Drain subscriber to get only the most recent message since last poll.
/// Returns `None` if no new messages (avoids duplicate sends).
fn drain_latest<PSB, T>(sub: &mut embassy_sync::pubsub::subscriber::Sub<'_, PSB, T>) -> Option<T>
where
    PSB: embassy_sync::pubsub::PubSubBehavior<T> + ?Sized,
    T: Clone,
{
    let mut last = None;
    while let Some(m) = sub.try_next_message_pure() {
        last = Some(m);
    }
    last
}
