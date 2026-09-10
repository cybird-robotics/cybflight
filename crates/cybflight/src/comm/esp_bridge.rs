//! ESP bridge — RX (uplink) + TX (downlink) embassy tasks.
//!
//! Uses DMA-backed UART for minimal interrupt overhead:
//! - TX: batch-encodes all frames into one buffer, single DMA transfer per tick
//! - RX: `read_until_idle()` — DMA fills buffer, wakes on idle line detection

use embassy_time::Timer;

use crate::hal::usart::{UartRx, UartTx};
use crate::{control, sensors};
use cybflight_msgs::NeighborStates;
#[cfg(feature = "outer_mpc")]
use cybflight_msgs::wire::WireMissionStatus;
#[cfg(feature = "est_eskf")]
use cybflight_msgs::wire::WirePositionControlSetpoint;
// NOTE: the health records are built by `health_wire::snapshot_{system,gps}`
// and forwarded by type inference, so `WireSystemHealth` / `WireGpsHealth`
// need no import here — the downlink itself is live (see DECIM_HEALTH).
use cybflight_msgs::wire::{
    self, WireArmDisarm, WireAttitudeControlSetpoint, WireDshotTelemetry, WireFleetState,
    WireGpsFix, WireMessage, WireOcpSolverOutput, WirePing, WirePingResp, WirePose, WirePowerStatus,
    WireRcInput, WireRcLinkStatus, WireTimeSync, WireTimeSyncStatus, WireVehicleAttitude,
    WireVehicleIdentity, WireVehicleOdometry,
};
#[cfg(feature = "dev_telem")]
use cybflight_msgs::wire::WireMotorStateTelemetry;
#[cfg(feature = "dev_telem")]
use cybflight_msgs::wire::{WireBaroSample, WireImu, WireMagSample};

use super::{FrameAccumulator, encode_frame};

// ---------------------------------------------------------------------------
// RX task — receive uplink messages from ESP32 via DMA + idle line detection
// ---------------------------------------------------------------------------

#[embassy_executor::task]
pub async fn esp_bridge_rx_task(mut rx: UartRx<'static, crate::hal::mode::Async>) {
    let pose_pub = sensors::VICON_POSE.immediate_publisher();
    let neighbor_tx = sensors::NEIGHBOR_STATES.sender();
    // Chaser only: republish the leader's pose from the direct PEER_POSE link.
    #[cfg(feature = "role_chaser")]
    let leader_tx = sensors::LEADER_POSE.sender();
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
                            wire::msg_id::FLEET_STATE => {
                                if let Some(fs) = WireFleetState::from_bytes(payload) {
                                    let rx_time = embassy_time::Instant::now();
                                    let states = NeighborStates::from_wire(&fs, rx_time);
                                    neighbor_tx.send(states);
                                }
                            }
                            #[cfg(feature = "role_chaser")]
                            wire::msg_id::PEER_POSE => {
                                if let Some(wp) = wire::WirePeerPose::from_bytes(payload) {
                                    let rx_time = embassy_time::Instant::now();
                                    let pp = cybflight_msgs::PeerPose::from_wire(&wp, rx_time);
                                    leader_tx.send(pp);
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

// ── Per-topic downlink decimation ────────────────────────────────────
//
// The TX task ticks at 100 Hz and `drain_latest` already caps each
// topic's downlink rate at 100 Hz. To reduce the WiFi/Rerun load we
// further decimate by gating each subscriber drain on `(tick %
// DECIM_<topic>) == 0`. Numbers chosen for "rerun-grade visualization
// is fine, no perceptible loss":
//   100 Hz / 1  = 100 Hz   (unchanged)
//   100 Hz / 2  =  50 Hz   (visualization-grade)
//   100 Hz / 4  =  25 Hz   (tuning-grade)
//   100 Hz / 100 = 1 Hz    (battery / status)
// `drain_latest` discards backlog so downsampling is lossy by design —
// each emission is the *latest* sample, never an averaged window.
const DECIM_VEHICLE_ODOMETRY: u32 = 4; //  25 Hz — rerun trajectory
const DECIM_VEHICLE_ATTITUDE: u32 = 4; //  25 Hz — rerun pose
const DECIM_ATTITUDE_CONTROL_SETPOINT: u32 = 4; //  25 Hz — outer_loop ref
// OCP solver output and POSITION_CONTROL_SETPOINT producers run at 50 Hz;
// decimate the downlink to 25 Hz to cut WiFi packet rate.
const DECIM_OCP_SOLVER: u32 = 4; //  25 Hz — OCP solver output
const DECIM_POSITION_CONTROL_SETPOINT: u32 = 4; //  25 Hz — outer_loop pos ref
const DECIM_RC_INPUT: u32 = 4; //  25 Hz — stick viz
const DECIM_DSHOT_TELEM: u32 = 4; //  25 Hz — eRPM tuning
#[cfg(feature = "dev_telem")]
const DECIM_MOTOR_STATE_TELEM: u32 = 4; //  25 Hz — KF state tuning
const DECIM_ACTUATOR_MOTORS: u32 = 4; //  25 Hz — motor cmd
const DECIM_POWER_STATUS: u32 = 100; //   1 Hz — battery
const DECIM_HEALTH: u32 = 100; //   1 Hz — system / GPS health snapshot
/// 0.2 Hz — firmware identity. Every field is fixed at build time, so
/// this is a pure self-announcement; the only requirement is that the GCS
/// learns it promptly enough after a drone appears (or reboots) to check
/// the roster before anyone arms. 5 s is well inside pre-flight.
const DECIM_IDENTITY: u32 = 500;
#[cfg(feature = "dev_telem")]
const DECIM_IMU: u32 = 4; //  25 Hz — raw IMU (debug)
#[cfg(feature = "dev_telem")]
const DECIM_MAG: u32 = 4; //  25 Hz — raw mag (debug)
#[cfg(feature = "dev_telem")]
const DECIM_BARO: u32 = 4; //  25 Hz — raw baro (debug)

/// Convert a monotonic Instant to UTC microseconds for telemetry.
fn utc_ts(instant: embassy_time::Instant) -> u64 {
    super::time_sync::to_utc_us(instant) as u64
}

#[embassy_executor::task]
pub async fn esp_bridge_tx_task(mut tx: UartTx<'static, crate::hal::mode::Async>) {
    // ── Subscribers, partitioned by purpose ──
    //
    // Always-on (visualization / control monitoring / health):
    let mut att_sub = crate::subscribe_or_park!(sensors::VEHICLE_ATTITUDE, "VEHICLE_ATTITUDE");
    let mut rc_sub = crate::subscribe_or_park!(sensors::RC_INPUT, "RC_INPUT");
    let mut rc_link_sub = crate::subscribe_or_park!(sensors::RC_LINK_STATUS, "RC_LINK_STATUS");
    let mut dshot_sub =
        crate::subscribe_or_park!(control::PROCESSED_DSHOT_TELEM, "PROCESSED_DSHOT_TELEM");
    // PROCESSED_MOTOR_STATE — tuning-grade only; gated behind `dev_telem`
    // because the GCS dashboard uses /health and /gpshealth instead. The
    // wire type stays public in cybflight-msgs for future-compat consumers.
    #[cfg(feature = "dev_telem")]
    let mut motor_state_sub =
        crate::subscribe_or_park!(control::PROCESSED_MOTOR_STATE, "PROCESSED_MOTOR_STATE");
    let mut ocp_sub = crate::subscribe_or_park!(control::OCP_SOLVER_OUTPUT, "OCP_SOLVER_OUTPUT");
    let mut gps_sub = crate::subscribe_or_park!(sensors::GPS_FIX, "GPS_FIX");
    // POWER_STATUS is NOT tuning-grade — pack voltage is the one flight
    // quantity with a hard deadline attached, and the operator can't read the
    // FC's buzzer from the ground station. 22 B/s at 1 Hz, so it rides along
    // with the health snapshots rather than behind `dev_telem`.
    let mut power_sub = crate::subscribe_or_park!(sensors::POWER_STATUS, "POWER_STATUS");
    let mut att_ctrl_sub = crate::subscribe_or_park!(
        control::ATTITUDE_CONTROL_SETPOINT,
        "ATTITUDE_CONTROL_SETPOINT"
    );
    #[cfg(feature = "est_eskf")]
    let mut pos_ctrl_sub = crate::subscribe_or_park!(
        control::POSITION_CONTROL_SETPOINT,
        "POSITION_CONTROL_SETPOINT"
    );
    let mut arm_sub = crate::subscribe_or_park!(crate::ARM_DISARM, "ARM_DISARM");
    let mut odom_sub = crate::subscribe_or_park!(sensors::VEHICLE_ODOMETRY, "VEHICLE_ODOMETRY");
    let mut motor_sub =
        crate::subscribe_or_park!(control::ACTUATOR_MOTORS_TELEM, "ACTUATOR_MOTORS_TELEM");
    #[cfg(feature = "outer_mpc")]
    let mut mission_status_sub =
        crate::subscribe_or_park!(control::MISSION_STATUS, "MISSION_STATUS");

    // Dev-only raw sensor channels — gated behind `dev_telem`. The ESKF
    // consumes these internally; the rerun viewport doesn't visualize
    // raw sensor signals during normal flight, so they're dropped on the
    // floor by default to free WiFi/Rerun bandwidth.
    #[cfg(feature = "dev_telem")]
    let mut imu1_sub = crate::subscribe_or_park!(sensors::IMU_1, "IMU_1");
    #[cfg(feature = "dev_telem")]
    let mut imu2_sub = crate::subscribe_or_park!(sensors::IMU_2, "IMU_2");
    #[cfg(feature = "dev_telem")]
    let mut mag_ext_sub = crate::subscribe_or_park!(sensors::MAG_EXT, "MAG_EXT");
    #[cfg(feature = "dev_telem")]
    let mut mag_int_sub = crate::subscribe_or_park!(sensors::MAG_INT, "MAG_INT");
    #[cfg(feature = "dev_telem")]
    let mut baro1_sub = crate::subscribe_or_park!(sensors::BARO_1, "BARO_1");
    #[cfg(feature = "dev_telem")]
    let mut baro2_sub = crate::subscribe_or_park!(sensors::BARO_2, "BARO_2");

    let mut seq: u8 = 0;
    // Batch buffer: holds all COBS-encoded frames for one tick.
    // Worst case ~16 frames × ~85 bytes each ≈ 1360 bytes; 1536 gives headroom.
    let mut batch = [0u8; 1536];

    // Ping state.
    let mut ping_counter: u32 = 0;
    let mut ping_id: u32 = 0;
    // Free-running tick counter for per-topic decimation (mod by DECIM_*).
    let mut tick: u32 = 0;

    // Leader peer-pose state: ENU origin (held once the GPS ESKF anchors) and
    // the live 3D-fix flag, for the direct WirePeerPose link to the chaser.
    // Gated at runtime by the `peer_pose_en` param (live — re-read on
    // PARAM_VERSION change) so the heavy 100 Hz downlink is off unless the
    // leader's vehicle explicitly enables it. The role itself stays a
    // compile-time feature (hardware topology).
    #[cfg(feature = "role_leader")]
    let mut peer_origin: Option<cybflight_core::geodetic::LlhOrigin> = None;
    #[cfg(feature = "role_leader")]
    let mut peer_fix3d = false;
    #[cfg(feature = "role_leader")]
    let mut peer_pose_enabled = crate::params::get().system.peer_pose_enable;
    #[cfg(feature = "role_leader")]
    let mut peer_pose_param_ver =
        crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);

    defmt::info!("ESP bridge TX task started (DMA)");

    loop {
        Timer::after_millis(TX_PERIOD_MS).await;
        tick = tick.wrapping_add(1);

        // Live param: pick up `param set peer_pose_en` without a reboot.
        // Cheap (one atomic load per 10 ms tick; the clone only happens on
        // an actual version change).
        #[cfg(feature = "role_leader")]
        {
            let v = crate::params::PARAM_VERSION.load(core::sync::atomic::Ordering::Acquire);
            if v != peer_pose_param_ver {
                peer_pose_param_ver = v;
                peer_pose_enabled = crate::params::get().system.peer_pose_enable;
            }
        }

        // Drain each subscriber — only send the latest message per
        // *eligible* tick. Per-topic `DECIM_*` further throttles the
        // 100 Hz tick to the rate appropriate for that topic's
        // visualization / tuning need; topics not in their slot this
        // tick are simply not drained, so backlog accumulates and the
        // next eligible tick still sees the latest sample.
        //
        // Edge-driven topics (ARM_DISARM) and natively-low-rate topics
        // (GPS, RC_LINK_STATUS, MISSION_STATUS) are NOT decimated here —
        // their producers already publish at the right rate.
        //
        // All frames are batch-encoded into one contiguous buffer, then
        // sent as a single DMA transfer (one interrupt total).
        // All telemetry timestamps are converted to UTC via time_sync.

        let mut pos = 0;

        // ── Visualization-grade @ 50 Hz ──
        if tick % DECIM_VEHICLE_ATTITUDE == 0 {
            if let Some(m) = drain_latest(&mut att_sub) {
                let mut w = WireVehicleAttitude::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }
        // Drain odometry once per tick: the GCS odometry downlink is decimated
        // to 25 Hz, while the leader's direct peer-pose link to the chaser (when
        // the `peer_pose_en` param is set) reuses the same fresh sample at the
        // full 100 Hz tick.
        let odom = drain_latest(&mut odom_sub);
        if let Some(ref m) = odom {
            if tick % DECIM_VEHICLE_ODOMETRY == 0 {
                let mut w = WireVehicleOdometry::from_msg(m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
            // Leader → chaser peer pose: re-express our fused ESKF position as
            // absolute LLH so the chaser can reproject it into its own ENU.
            // Disabled by default; enable with `param set peer_pose_en 1`.
            #[cfg(feature = "role_leader")]
            if peer_pose_enabled {
                if peer_origin.is_none() {
                    peer_origin = sensors::GNSS_ORIGIN.try_take();
                }
                if let Some(o) = peer_origin {
                    let (lat_rad, lon_rad, alt_m) = o.enu_to_llh(m.pose.position);
                    let converged = crate::estimation::ESTIMATOR_READY
                        .load(core::sync::atomic::Ordering::Acquire);
                    let w = wire::WirePeerPose::new(
                        utc_ts(m.timestamp),
                        lat_rad,
                        lon_rad,
                        alt_m,
                        converged,
                        peer_fix3d,
                    );
                    pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
                }
            }
        }
        if tick % DECIM_ATTITUDE_CONTROL_SETPOINT == 0 {
            if let Some(m) = drain_latest(&mut att_ctrl_sub) {
                let mut w = WireAttitudeControlSetpoint::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }
        // POSITION_CONTROL_SETPOINT producer is 50 Hz — decimate to 25 Hz.
        #[cfg(feature = "est_eskf")]
        if tick % DECIM_POSITION_CONTROL_SETPOINT == 0 {
            if let Some(m) = drain_latest(&mut pos_ctrl_sub) {
                let mut w = WirePositionControlSetpoint::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }

        // ── Tuning-grade @ 25 Hz ──
        if tick % DECIM_RC_INPUT == 0 {
            if let Some(m) = drain_latest(&mut rc_sub) {
                let mut w = WireRcInput::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }
        if tick % DECIM_DSHOT_TELEM == 0 {
            if let Some(m) = drain_latest(&mut dshot_sub) {
                let mut w = WireDshotTelemetry::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }
        #[cfg(feature = "dev_telem")]
        if tick % DECIM_MOTOR_STATE_TELEM == 0 {
            if let Some(m) = drain_latest(&mut motor_state_sub) {
                let mut w = WireMotorStateTelemetry::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }
        if tick % DECIM_ACTUATOR_MOTORS == 0 {
            if let Some(m) = drain_latest(&mut motor_sub) {
                let mut w = wire::WireActuatorMotors::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }

        // ── Solver / mission status ──
        // OCP producer is 50 Hz — decimate the downlink to 25 Hz.
        // MISSION_STATUS at 10 Hz (already decimated by outer_loop), GPS at
        // 5–10 Hz, RC_LINK_STATUS at ~10 Hz, and ARM_DISARM is event-driven;
        // all forwarded producer-paced.
        if tick % DECIM_OCP_SOLVER == 0 {
            if let Some(m) = drain_latest(&mut ocp_sub) {
                let mut w = WireOcpSolverOutput::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }
        if let Some(m) = drain_latest(&mut gps_sub) {
            let mut w = WireGpsFix::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            // Track 3D-fix liveness for the leader's peer-pose status flag.
            // Unconditional under role_leader (cheap; keeps the flag warm
            // for the moment peer_pose_en flips on).
            #[cfg(feature = "role_leader")]
            {
                peer_fix3d = m.fix_type >= 3;
            }
        }
        if let Some(m) = drain_latest(&mut rc_link_sub) {
            let mut w = WireRcLinkStatus::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        if let Some(m) = drain_latest(&mut arm_sub) {
            let mut w = WireArmDisarm::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }
        #[cfg(feature = "outer_mpc")]
        if let Some(m) = drain_latest(&mut mission_status_sub) {
            let mut w = WireMissionStatus::from_msg(&m);
            w.timestamp_us = utc_ts(m.timestamp);
            pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
        }

        // ── Battery @ 1 Hz ──
        if tick % DECIM_POWER_STATUS == 0 {
            if let Some(m) = drain_latest(&mut power_sub) {
                let mut w = WirePowerStatus::from_msg(&m);
                w.timestamp_us = utc_ts(m.timestamp);
                pos += encode_and_advance(&w, &mut seq, &mut batch[pos..]);
            }
        }

        // ── Health snapshots @ 1 Hz ──
        // Mirror what the live `health` / `gpshealth` shell commands
        // print. Source-of-truth atomics are read by
        // `comm::health_wire::snapshot_*`. Same wall-clock instant for
        // both records so the GCS dashboard renders coherent state.
        if tick % DECIM_HEALTH == 0 {
            let now = embassy_time::Instant::now();
            let mut sys = super::health_wire::snapshot_system(now);
            sys.timestamp_us = utc_ts(now);
            pos += encode_and_advance(&sys, &mut seq, &mut batch[pos..]);
            let mut gps = super::health_wire::snapshot_gps(now);
            gps.timestamp_us = utc_ts(now);
            pos += encode_and_advance(&gps, &mut seq, &mut batch[pos..]);
        }

        // ── Firmware identity @ 0.2 Hz ──
        // Lets the ground station verify a binding it otherwise assumes:
        // it routes poses to this drone by IP, with nothing confirming
        // that the board at that IP is running THIS airframe's firmware.
        // A mis-flash silently pairs one machine's mass/inertia/gains
        // with another machine's pose stream — which is precisely the
        // rigid-body re-association the mocap guard exists to reject,
        // except arriving pre-authenticated through the correct channel.
        if tick % DECIM_IDENTITY == 0 {
            let now = embassy_time::Instant::now();
            let ident = WireVehicleIdentity {
                timestamp_us: utc_ts(now),
                airframe_name: match crate::vehicle::BAKED_AIRFRAME_NAME {
                    Some(n) => WireVehicleIdentity::pack_ident(n),
                    None => [0u8; wire::IDENT_LEN],
                },
                vehicle: WireVehicleIdentity::pack_ident(crate::vehicle::BAKED_VEHICLE),
                pos_source: if cfg!(feature = "est_pos_mocap") {
                    wire::pos_source::MOCAP
                } else if cfg!(feature = "est_pos_gps") {
                    wire::pos_source::GPS
                } else {
                    wire::pos_source::UNKNOWN
                },
                _pad: [0u8; 7],
            };
            pos += encode_and_advance(&ident, &mut seq, &mut batch[pos..]);
        }

        // ── Dev-only raw channels (gated by `dev_telem`) @ 25 Hz ──
        #[cfg(feature = "dev_telem")]
        if tick % DECIM_IMU == 0 {
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
        }
        #[cfg(feature = "dev_telem")]
        if tick % DECIM_MAG == 0 {
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
        }
        #[cfg(feature = "dev_telem")]
        if tick % DECIM_BARO == 0 {
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

        // Surface batch-buffer saturation on the ping cadence. Silence
        // here would mean a topic quietly stopped reaching the ground
        // station whenever several decimation counters coincided.
        if tick % PING_INTERVAL_TICKS == 0 {
            let dropped = BATCH_DROPPED.swap(0, core::sync::atomic::Ordering::Relaxed);
            if dropped > 0 {
                defmt::warn!(
                    "esp_bridge: {} downlink frames dropped — batch buffer ({} B) full",
                    dropped,
                    batch.len(),
                );
            }
        }
    }
}

/// Frames dropped because the per-tick batch buffer was full.
///
/// Every downlink topic writes into one fixed `batch` buffer sized
/// against a budget in the comments above. Adding a topic, or a burst
/// where several decimation counters line up on the same tick, eats that
/// headroom silently — and the encode used to panic when it ran out.
/// Dropping and counting is the right failure for telemetry.
static BATCH_DROPPED: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// COBS-encode a wire message into the batch buffer. Returns bytes written,
/// or 0 if it did not fit.
fn encode_and_advance<W: WireMessage>(msg: &W, seq: &mut u8, dest: &mut [u8]) -> usize {
    encode_with_id(msg, W::MSG_ID, seq, dest)
}

/// COBS-encode a wire message with an explicit msg_id (for secondary instances
/// like IMU_2, BARO_2, MAG_INT that share a wire layout with their primary).
fn encode_with_id<W: WireMessage>(msg: &W, msg_id: u8, seq: &mut u8, dest: &mut [u8]) -> usize {
    let len = encode_frame(msg_id, *seq, msg.as_bytes(), dest);
    // The sequence number advances either way. It is a single counter
    // across every topic, so leaving it unadvanced on a drop would hand
    // the ground station a contiguous sequence over a real gap and hide
    // the loss it exists to expose.
    *seq = seq.wrapping_add(1);
    if len == 0 {
        BATCH_DROPPED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
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
