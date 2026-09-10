//! Chaser gimbal task — aims a Z-1Mini at the leader over USART2 (chaser only).
//!
//! Reads the chaser's own fused pose (`VEHICLE_ODOMETRY`) and the leader's pose
//! (`LEADER_POSE`, from the direct ESP link), reprojects the leader's absolute
//! LLH into the chaser's local ENU (`GNSS_ORIGIN`), and streams EulerControl
//! setpoints + CarrierIns to the pod at 100 Hz.
//!
//! Mode: EulerControl (0x14) + full CarrierIns — the pod fuses the carrier INS
//! (true heading from the UM982) to hold the camera on an absolute world
//! orientation. Gating: track only when our ESKF is converged, we have a 3D fix,
//! and the leader pose is converged + 3D-fixed + fresh; otherwise hold the last
//! setpoint (never recenter). The pod clamps/best-efforts at its mechanical
//! limits — no firmware envelope guard.

use core::sync::atomic::Ordering;

use cybflight_drivers::gimbal as proto;
use embassy_time::{with_timeout, Duration, Instant, Timer};
use nalgebra::{UnitQuaternion, Vector3};

use crate::hal::usart::{BufferedUartRx, BufferedUartTx};
use crate::sensors;

/// Control stream rate (Hz) and period.
const STREAM_PERIOD_MS: u64 = 10; // 100 Hz
/// Drop to "hold" if the leader pose is older than this.
const LEADER_MAX_AGE: Duration = Duration::from_millis(300);

fn clamp_i16(v: f32) -> i16 {
    v.clamp(-32767.0, 32767.0) as i16
}

#[embassy_executor::task]
pub async fn gimbal_task(mut tx: BufferedUartTx<'static>, mut rx: BufferedUartRx<'static>) {
    defmt::info!("gimbal: task started (USART2)");

    // 1) Wait for the pod to actually be listening: stream passive heartbeats and
    //    listen for a valid GCU reply. Per-read timeout (never blocks), retried
    //    indefinitely — so whether the pod boots slower than the FC or is powered
    //    on afterwards, it's still picked up. Only this task waits; the rest of
    //    the board boots normally.
    wait_for_pod(&mut tx, &mut rx).await;

    // 2) Boot self-test: a small INS-free AngleControl sweep so the operator sees
    //    the gimbal move — now guaranteed to land because the pod is listening.
    self_test(&mut tx).await;

    // 3) Wait for the GPS ESKF to anchor the local ENU origin (held for flight).
    let origin = sensors::GNSS_ORIGIN.wait().await;
    defmt::info!("gimbal: ENU origin acquired, tracking enabled once converged");

    let mut odom_sub = crate::subscribe_or_park!(sensors::VEHICLE_ODOMETRY, "VEHICLE_ODOMETRY");
    let mut gps_sub = crate::subscribe_or_park!(sensors::GPS_FIX, "GPS_FIX");
    let mut leader_rx = match sensors::LEADER_POSE.receiver() {
        Some(r) => r,
        None => {
            defmt::error!("gimbal: LEADER_POSE receiver exhausted — task parked");
            loop {
                Timer::after_secs(60).await;
            }
        }
    };

    let mut buf = [0u8; proto::FRAME_LEN];
    let mut last_sp = proto::Setpoint::default();
    let mut own_pos = Vector3::zeros();
    let mut own_q = UnitQuaternion::identity();
    let mut own_vel: Vector3<f32> = Vector3::zeros();
    let mut prev_vel: Vector3<f32> = Vector3::zeros();
    let mut accel: Vector3<f32> = Vector3::zeros(); // ENU kinematic accel (d vel/dt, low-passed)
    let mut fix3d = false;
    let mut rxbuf = [0u8; 96];
    let mut rx_scan = proto::FrameScanner::new();

    loop {
        Timer::after_millis(STREAM_PERIOD_MS).await;

        // Best-effort drain of the pod's telemetry so its bytes can't back up the
        // RX buffer (discarded for now — wire it to logging later if useful).
        while let Ok(Ok(k)) = with_timeout(
            Duration::from_micros(300),
            embedded_io_async::Read::read(&mut rx, &mut rxbuf),
        )
        .await
        {
            if k == 0 {
                break;
            }
            rx_scan.push(&rxbuf[..k]);
            while rx_scan.pull().is_some() {}
        }

        // Drain to the freshest own pose; derive world accel from Δvelocity.
        let mut got = false;
        while let Some(o) = odom_sub.try_next_message_pure() {
            own_pos = o.pose.position;
            own_q = o.pose.orientation;
            own_vel = o.twist.linear;
            got = true;
        }
        if got {
            let raw = (own_vel - prev_vel) * (1000.0 / STREAM_PERIOD_MS as f32);
            accel = accel * 0.8 + raw * 0.2; // light LP on the differentiated accel
            prev_vel = own_vel;
        }
        while let Some(g) = gps_sub.try_next_message_pure() {
            fix3d = g.fix_type >= 3;
        }

        let now = Instant::now();
        let own_ready = crate::estimation::ESTIMATOR_READY.load(Ordering::Acquire);
        let leader = leader_rx.try_get();
        let leader_ok = leader.is_some_and(|l| l.usable(now, LEADER_MAX_AGE));
        let track = own_ready && fix3d && leader_ok;

        // Carrier INS: our aircraft attitude (heading from UM982-fused yaw) +
        // ENU velocity/accel reordered to North/East/Up, in 0.01 units.
        let (roll_cd, pitch_cd, yaw_cd) = proto::carrier_attitude(&own_q);
        let ins = proto::CarrierIns {
            roll_cd,
            pitch_cd,
            yaw_cd,
            acc_n: clamp_i16(accel.y * 100.0),
            acc_e: clamp_i16(accel.x * 100.0),
            acc_u: clamp_i16(accel.z * 100.0),
            vel_n: clamp_i16(own_vel.y * 100.0),
            vel_e: clamp_i16(own_vel.x * 100.0),
            vel_u: clamp_i16(own_vel.z * 100.0),
        };

        if track {
            // SAFETY: leader_ok implies leader.is_some().
            let l = leader.unwrap();
            let leader_enu = origin.llh_to_enu(l.lat_rad, l.lon_rad, l.alt_m);
            let los = leader_enu - own_pos; // camera 5 cm lever-arm dropped (≤1°)
            last_sp = proto::aim_setpoint(los);
        }
        // Not tracking → hold the last setpoint (pod keeps the last world
        // orientation); never recenter.
        let n = proto::encode_euler(&mut buf, last_sp, ins);
        let _ = embedded_io_async::Write::write_all(&mut tx, &buf[..n]).await;
    }
}

/// Probe the pod until it answers. Streams passive heartbeats (no motion) and
/// listens for a valid GCU→host frame, with a per-read timeout so it never
/// blocks. Retries indefinitely — each read is bounded — so a pod that is still
/// booting when the FC comes up, or powered on later, is still detected. Only
/// this task waits here; the board's boot sequence is unaffected.
async fn wait_for_pod(tx: &mut BufferedUartTx<'static>, rx: &mut BufferedUartRx<'static>) {
    let mut hb = [0u8; proto::FRAME_LEN];
    let n = proto::encode_heartbeat(&mut hb);
    let mut rxbuf = [0u8; 96];
    let mut scan = proto::FrameScanner::new();
    let mut announced = false;
    loop {
        let _ = embedded_io_async::Write::write_all(tx, &hb[..n]).await;
        match with_timeout(
            Duration::from_millis(120),
            embedded_io_async::Read::read(rx, &mut rxbuf),
        )
        .await
        {
            Ok(Ok(k)) if k > 0 => {
                scan.push(&rxbuf[..k]);
                if scan.pull().is_some() {
                    defmt::info!("gimbal: pod responding on USART2");
                    return;
                }
            }
            _ => {
                if !announced {
                    defmt::info!("gimbal: waiting for pod to respond on USART2…");
                    announced = true;
                }
            }
        }
    }
}

/// Boot self-test: an aggressive full-range-of-motion sweep on all three axes so
/// the operator can confirm the link and see the pod exercise its full travel.
///
/// Uses FPV mode (0x1C): all axes relative to the mount, INS-free, and — unlike
/// angle/euler — **not** subject to the >45° auto-recenter protection, so the pod
/// drives to its mechanical stops. We command ±180° on each axis; the pod clamps
/// to its actual limits (pitch −105…+145°, yaw ±160°, roll to its stop). Motion
/// is ramped between waypoints so it's a visible sweep, not a jump.
async fn self_test(tx: &mut BufferedUartTx<'static>) {
    let mut buf = [0u8; proto::FRAME_LEN];
    const FULL: i16 = 18_000; // ±180° commanded (pod clamps to its mechanical ROM)
    // (roll, pitch, yaw) centidegrees — each axis swept to both stops, then center.
    const WPTS: [(i16, i16, i16); 7] = [
        (0, FULL, 0),    // pitch up to stop
        (0, -FULL, 0),   // pitch down to stop
        (0, 0, FULL),    // yaw right to stop
        (0, 0, -FULL),   // yaw left to stop
        (FULL, 0, 0),    // roll right to stop
        (-FULL, 0, 0),   // roll left to stop
        (0, 0, 0),       // re-center
    ];
    let mut prev = (0i16, 0i16, 0i16);
    for &wp in WPTS.iter() {
        const STEPS: i32 = 30; // ~0.45 s per segment at 15 ms/step
        for s in 1..=STEPS {
            let lerp = |a: i16, b: i16| (a as i32 + (b as i32 - a as i32) * s / STEPS) as i16;
            let sp = proto::Setpoint::new(lerp(prev.0, wp.0), lerp(prev.1, wp.1), lerp(prev.2, wp.2));
            let n = proto::encode(&mut buf, proto::cmd::FPV, sp, None);
            let _ = embedded_io_async::Write::write_all(tx, &buf[..n]).await;
            Timer::after_millis(15).await;
        }
        prev = wp;
    }
    defmt::info!("gimbal: full-ROM self-test sweep complete");
}
