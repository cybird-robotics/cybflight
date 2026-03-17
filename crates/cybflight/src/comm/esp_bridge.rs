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
    WireGpsFix, WireImu, WireMagSample, WireManualControlSetpoint, WireMessage,
    WireOcpSolverOutput, WirePose, WireRcInput, WireRcLinkStatus, WireVehicleAttitude,
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
    let mut manual_sub = sensors::MANUAL_CONTROL.subscriber().unwrap();
    let mut arm_sub = crate::ARM_DISARM.subscriber().unwrap();

    let mut seq: u8 = 0;
    // Batch buffer: holds all COBS-encoded frames for one tick.
    // Worst case ~14 frames × ~60 bytes each = ~840 bytes; 1024 gives headroom.
    let mut batch = [0u8; 1024];

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

        let mut pos = 0;

        if let Some(m) = drain_latest(&mut imu1_sub) {
            pos += encode_and_advance(&WireImu::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut imu2_sub) {
            pos += encode_with_id(&WireImu::from_msg(&m), wire::msg_id::IMU_2, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut att_sub) {
            pos += encode_and_advance(&WireVehicleAttitude::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut rc_sub) {
            pos += encode_and_advance(&WireRcInput::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut rc_link_sub) {
            pos += encode_and_advance(&WireRcLinkStatus::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut dshot_sub) {
            pos += encode_and_advance(&WireDshotTelemetry::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut ocp_sub) {
            pos += encode_and_advance(&WireOcpSolverOutput::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut gps_sub) {
            pos += encode_and_advance(&WireGpsFix::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut mag_ext_sub) {
            pos += encode_and_advance(&WireMagSample::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut mag_int_sub) {
            pos += encode_with_id(&WireMagSample::from_msg(&m), wire::msg_id::MAG_INT, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut baro1_sub) {
            pos += encode_and_advance(&WireBaroSample::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut baro2_sub) {
            pos += encode_with_id(&WireBaroSample::from_msg(&m), wire::msg_id::BARO_2, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut att_ctrl_sub) {
            pos += encode_and_advance(&WireAttitudeControlSetpoint::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut manual_sub) {
            pos += encode_and_advance(&WireManualControlSetpoint::from_msg(&m), &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut arm_sub) {
            pos += encode_and_advance(&WireArmDisarm::from_msg(&m), &mut seq, &mut batch[pos..]);
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
