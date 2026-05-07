//! ESKF fault-detection integration tests. These drive `cybflight_core::eskf`
//! directly through pathological measurement sequences and assert that the
//! `EskfHealth` counters move the way the fortification plan calls for —
//! gate rejections accumulate without state corruption, NaN inputs trip the
//! reset path exactly once, and saturating-noise IMU input doesn't blow up
//! the covariance through the safeguards (`clamp_covariance_diagonal`,
//! `state_is_finite`).
//!
//! The firmware-side `evaluate_faults` helper is covered by unit tests on
//! the firmware crate; what these integration tests prove is that the
//! filter's *internal* health surface is wired up correctly and that the
//! existing numeric guards still hold.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test autotest_faults -- --nocapture

use cybflight_core::eskf::{Eskf, EskfConfig};
use nalgebra::{UnitQuaternion, Vector3};

fn fresh_eskf() -> Eskf {
    let mut e = Eskf::new(EskfConfig::default());
    e.init(
        Vector3::zeros(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    e
}

#[test]
fn outlier_position_rain_increments_counter_without_corrupting_state() {
    // Simulate a flapping mocap source: every other frame is a 100 m
    // outlier, interleaved with reasonable updates. After many cycles the
    // gate counter should be high but the filter should still be tracking
    // truth (since the gate rejects the bad ones).
    let mut e = fresh_eskf();
    let truth = Vector3::new(1.0, 2.0, 3.0);
    for _ in 0..200 {
        e.update_pos(truth, 0.05);
        e.update_pos(Vector3::new(100.0, 0.0, 0.0), 0.05);
    }
    let h = e.health();
    assert!(
        h.gate_rejects_pos > 100,
        "expected ≥100 rejections, got {}",
        h.gate_rejects_pos
    );
    let pos = e.position();
    assert!(
        (pos - truth).norm() < 0.5,
        "filter drifted to {pos:?}, expected near {truth:?}"
    );
    assert_eq!(h.nan_resets, 0);
    println!(
        "outlier rain: rejects={} final_pos=[{:.3},{:.3},{:.3}]",
        h.gate_rejects_pos, pos.x, pos.y, pos.z
    );
}

#[test]
fn nan_input_trips_reset_exactly_once() {
    // A single NaN-laden predict should bump nan_resets and clear
    // `is_initialized`. Subsequent NaN predicts (filter already
    // un-initialised) must NOT double-count.
    let mut e = fresh_eskf();
    let bad_accel = Vector3::new(f32::NAN, 0.0, 0.0);
    e.predict(bad_accel, Vector3::zeros(), 0.001);
    assert_eq!(e.health().nan_resets, 1);
    assert!(!e.is_initialized());
    // Filter is now uninitialised — predict should be a no-op.
    e.predict(bad_accel, Vector3::zeros(), 0.001);
    assert_eq!(
        e.health().nan_resets,
        1,
        "nan_resets must not increment while uninitialised"
    );
}

#[test]
fn saturating_noise_imu_does_not_explode_covariance() {
    // Drive the filter at 1 kHz with comically large IMU values that
    // would produce numerically unstable covariance in a naive
    // implementation. The clamp + Joseph form should keep things sane —
    // no NaN reset, finite covariance trace.
    let mut e = fresh_eskf();
    let mut rng_state: u32 = 0xC0FFEE;
    let mut next_pseudo = || {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 8) as f32 / (1u32 << 23) as f32) - 1.0
    };
    let dt = 1e-3;
    for _ in 0..1000 {
        let accel = Vector3::new(
            20.0 * next_pseudo(),
            20.0 * next_pseudo(),
            -9.81 + 20.0 * next_pseudo(),
        );
        let gyro = Vector3::new(next_pseudo(), next_pseudo(), next_pseudo()) * 5.0;
        e.predict(accel, gyro, dt);
    }
    let h = e.health();
    assert_eq!(h.nan_resets, 0, "saturating IMU produced NaN reset");
    assert!(e.is_initialized());
    let pos_trace = e.pos_cov_trace();
    let vel_trace = e.vel_cov_trace();
    assert!(pos_trace.is_finite() && pos_trace > 0.0);
    assert!(vel_trace.is_finite() && vel_trace > 0.0);
    println!(
        "saturating-noise IMU: pos_cov_trace={pos_trace:.2} vel_cov_trace={vel_trace:.2}"
    );
}

#[test]
fn velocity_outliers_count_separately_from_position() {
    // Two channels — make sure pos and vel counters don't share state.
    let mut e = fresh_eskf();
    e.update_vel(Vector3::new(500.0, 0.0, 0.0), 0.1);
    e.update_pos(Vector3::new(0.01, 0.0, 0.0), 0.5); // inlier — no count.
    let h = e.health();
    assert_eq!(h.gate_rejects_vel, 1);
    assert_eq!(h.gate_rejects_pos, 0);
}

#[test]
fn baro_outliers_dont_affect_other_counters() {
    let mut e = fresh_eskf();
    e.update_altitude(10_000.0);
    let h = e.health();
    assert_eq!(h.gate_rejects_baro, 1);
    assert_eq!(h.gate_rejects_pos, 0);
    assert_eq!(h.gate_rejects_vel, 0);
    assert_eq!(h.gate_rejects_att, 0);
    assert_eq!(h.gate_rejects_mag, 0);
}
