//! Always-on Mahony complementary attitude filter.
//!
//! Guarantees an attitude estimate exists in **every** build —
//! including `outer_rate` vehicles with no mocap and no GPS, where no
//! ESKF task is spawned and `/odometry` never publishes. Subscribes
//! to the ~1 kHz [`sensors::IMU_1_DECIM`] mirror (not the full-rate
//! `IMU_1`, so this thread-executor task is not woken 8000×/s to
//! discard 7 of 8 samples), runs the IMU-only (no magnetometer)
//! Mahony filter at that rate, and publishes
//! [`sensors::MAHONY_ATTITUDE`] at ~100 Hz — the source of the
//! blackbox `/attitude` topic.
//!
//! Deliberately **outside the control loop**: nothing consumes this
//! estimate for flight control. On ESKF builds it runs alongside the
//! ESKF as an independent cross-check (~200–300 FLOPs per 1 kHz step,
//! well under 0.1 % of the M7) — logging both lets a postmortem
//! distinguish "mocap/GPS went bad" from "IMU went bad". On no-ESKF
//! builds it additionally feeds `VEHICLE_ATTITUDE`, so the CRSF /
//! ESP-bridge / shell-stream / postmortem consumers see attitude too
//! (the channel's PUBS=1 contract holds: in those builds no ESKF
//! publisher exists).
//!
//! IMU-only Mahony observes tilt (gravity direction) but not yaw:
//! expect yaw drift, same as the GPS build's ESKF without a heading
//! source. Fine for logging; not a heading reference.
//!
//! Revived from the `robust_ahrs` branch (`c69b55a`) minus the AHRS
//! supervisor / fallback-control machinery — this task is
//! logging/telemetry only. The `health.rs` ESKF↔Mahony arming
//! cross-check stays on its "ready" sentinel; wiring it up is a
//! separate, deliberate change.

use core::time::Duration;

use embassy_sync::pubsub::WaitResult;
use embassy_time::Instant;
use nalgebra::{UnitQuaternion, Vector3};

use cybflight_core::mahony::{Mahony, MahonyError};

use crate::msgs;
use crate::sensors;

/// Update rate: the `IMU_1_DECIM` mirror's rate (IMU ODR /
/// `rates::IMU_DECIM_DIV` ≈ 1 kHz). Mahony gains are rate-robust here
/// because dt is measured from sample timestamps, not assumed.
const UPDATE_RATE_HZ: u32 = (crate::rates::IMU_ODR_HZ as u32) / crate::rates::IMU_DECIM_DIV;

/// Publish decimation relative to the ~1 kHz update rate → ~100 Hz.
/// Sets the `/attitude` blackbox rate (~5.5 KB/s) and, on no-ESKF
/// builds, the `VEHICLE_ATTITUDE` telemetry rate.
const PUBLISH_DECIMATION: u32 = 10;

/// Drop update steps with `dt` outside this range — first sample
/// after subscribe, or a long executor stall.
const MAX_DT_S: f32 = 0.05;

/// Updates before the estimate is considered converged (~1 s at
/// 1 kHz). Purely informational — the estimate is published from the
/// first valid update so the log shows convergence too.
const WARMUP_UPDATES: u32 = 1000;

fn orientation_is_finite(q: &UnitQuaternion<f32>) -> bool {
    let v = q.as_vector();
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite() && v.w.is_finite()
}

#[embassy_executor::task]
pub async fn mahony_task() {
    let mut imu_sub = sensors::IMU_1_DECIM
        .subscriber()
        .expect("mahony: IMU_1_DECIM subscriber");
    let att_pub = sensors::MAHONY_ATTITUDE.immediate_publisher();
    // Sole VEHICLE_ATTITUDE publisher when no ESKF task exists.
    #[cfg(not(any(feature = "est_pos_mocap", feature = "est_pos_gps")))]
    let vehicle_att_pub = sensors::VEHICLE_ATTITUDE.immediate_publisher();

    // Gains and the two sensor-norm floors come from the `mahony` group
    // (reboot-flagged, so one snapshot covers the task's lifetime). They
    // were previously `Mahony::new()` literals, which left the attitude
    // reference the arming gate cross-checks against untunable.
    let mut filter: Mahony<f32> = {
        let p = crate::params::get();
        let m = &p.mahony;
        let mut f = Mahony::with_gains(
            nalgebra::Vector3::from_element(m.kp),
            nalgebra::Vector3::from_element(m.ki),
        );
        // Both setters reject a non-positive threshold. Range validation
        // already keeps these above zero at every write, so a rejection
        // here means the value never reached the filter — keep the
        // constructor default rather than fail the task, per the
        // "degrade, never panic" rule.
        if f.set_min_accel_norm(m.min_accel_g * p.site.gravity_m_s2)
            .is_err()
        {
            defmt::warn!("Mahony: mahony_min_accel_g rejected, keeping default");
        }
        if f.set_min_mag_norm(m.min_mag_ut).is_err() {
            defmt::warn!("Mahony: mahony_min_mag_ut rejected, keeping default");
        }
        f
    };
    let mut last_t: Option<Instant> = None;
    let mut pub_decim: u32 = 0;
    let mut warmup: u32 = 0;

    defmt::info!(
        "Mahony: attitude filter started (update ~{} Hz, publish ~{} Hz)",
        UPDATE_RATE_HZ,
        UPDATE_RATE_HZ / PUBLISH_DECIMATION,
    );

    loop {
        let imu = match imu_sub.next_message().await {
            WaitResult::Message(m) => m,
            WaitResult::Lagged(_) => continue,
        };

        // `saturating_duration_since`, NOT `duration_since`: an IMU
        // sample can legitimately be OLDER than `last_t`. This task
        // runs on the thread executor and can fall a full CAP behind
        // the publisher (32 samples = 32 ms on the mirror); a `Lagged`
        // continue then a ring wrap can hand us a stale sample.
        // `duration_since` panics on that underflow — which took the
        // board into a panic → sys_reset → IWDG boot loop on every
        // 8 kHz power-up, with only the `Booting` heartbeat alive.
        // (Same bug class the ESKF hit on `imu_1khz`: see
        // eskf_imu_mocap.rs.) Saturated, a stale sample yields dt = 0
        // and the guard below skips it; `last_t` still advances.
        let dt_s = match last_t {
            Some(prev) => {
                imu.timestamp.saturating_duration_since(prev).as_micros() as f32 * 1e-6
            }
            None => 0.0,
        };
        last_t = Some(imu.timestamp);
        if dt_s <= 0.0 || dt_s > MAX_DT_S {
            continue;
        }
        let dt = Duration::from_micros((dt_s * 1e6) as u64);

        match filter.update(imu.gyro_rad_s, imu.accel_m_s2, None, dt) {
            Ok(_) => {}
            // Free-fall / launch transients: gyro-only propagation
            // inside `update` still ran; skip nothing.
            Err(MahonyError::ZeroAcceleration) => {}
            Err(_) => continue,
        }

        // Pathological gyro spikes can drive the orientation into NaN
        // even with a sane accel. Reset to identity and re-warm.
        if !orientation_is_finite(&filter.orientation()) {
            filter.set_orientation(UnitQuaternion::identity());
            filter.set_gyro_bias(Vector3::zeros());
            warmup = 0;
            defmt::warn!("Mahony: non-finite orientation, reset to identity");
            continue;
        }

        warmup = warmup.saturating_add(1);
        if warmup == WARMUP_UPDATES {
            defmt::info!("Mahony: warm-up complete");
        }

        pub_decim += 1;
        if pub_decim < PUBLISH_DECIMATION {
            continue;
        }
        pub_decim = 0;

        let m = msgs::VehicleAttitude {
            timestamp: imu.timestamp,
            orientation: filter.orientation(),
        };
        att_pub.publish_immediate(m.clone());
        #[cfg(not(any(feature = "est_pos_mocap", feature = "est_pos_gps")))]
        vehicle_att_pub.publish_immediate(m);
        #[cfg(any(feature = "est_pos_mocap", feature = "est_pos_gps"))]
        let _ = m;
    }
}
