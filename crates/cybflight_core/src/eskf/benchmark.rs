//! Runtime benchmark: dense vs. sparse ESKF predict/update_pose.
//!
//! Wall-clock timing on the host x86_64 is not a perfect proxy for
//! Cortex-M7 cycles (different vector widths, different FMA scheduling,
//! different cache hierarchy), but the *ratio* between the dense and
//! sparse implementations should track closely on the H743: both paths
//! call the same nalgebra primitives, and the sparse version simply
//! avoids structurally-zero work in the same SIMD-able loops.
//!
//! Run: `cargo test -p cybflight-core --target x86_64-unknown-linux-gnu \
//!       --features std --release --lib eskf::benchmark -- --nocapture`
//!
//! `--release` is essential — a debug build hides the optimizer wins
//! the sparse path is meant to expose.

use super::eskf::{Eskf, EskfConfig};
use nalgebra::{UnitQuaternion, Vector3};
use std::time::Instant;

/// Number of operations per timed run. Larger gives more stable timings
/// at the cost of test runtime; ~10k iterations × 5 inner repeats
/// gives statistically stable mean wall-clock at <1 s test runtime.
const N_PREDICT_ITERS: usize = 100_000;
const N_UPDATE_ITERS: usize = 30_000;
const N_REPEATS: usize = 5;

fn fresh_filter() -> Eskf {
    let mut e = Eskf::new(EskfConfig::default());
    e.init(
        Vector3::zeros(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    e
}

/// Time a closure that runs `n` iterations, returning wall-clock seconds.
fn time_iters<F: FnMut(usize)>(n: usize, mut body: F) -> f64 {
    // Warm up: caches, branch predictor.
    for i in 0..1000 {
        body(i);
    }
    let t0 = Instant::now();
    for i in 0..n {
        body(i);
    }
    let dt = t0.elapsed();
    dt.as_secs_f64()
}

/// Best-of-N timing: reduces variance from OS scheduling, frequency
/// scaling, etc. The minimum over repeats is closer to the steady-state
/// per-call cost than the mean.
fn best_of<F: FnMut() -> f64>(repeats: usize, mut f: F) -> f64 {
    (0..repeats)
        .map(|_| f())
        .fold(f64::INFINITY, |acc, x| acc.min(x))
}

/// Geometrically-meaningful synthetic IMU: small accel + gyro
/// perturbations on top of nominal hover. Exercises the F-matrix's
/// non-trivial blocks (R⁻, M_a, M_b) at flight-like magnitudes.
fn imu_at(i: usize) -> (Vector3<f32>, Vector3<f32>) {
    let g = Vector3::new(0.0, 0.0, -9.81);
    let phase = i as f32 * 1e-3;
    let accel = -g
        + Vector3::new(
            0.5 * phase.sin(),
            0.4 * (phase * 0.3).cos(),
            0.2 * (phase * 0.7).sin(),
        );
    let gyro = Vector3::new(
        0.5 * phase.sin(),
        0.3 * (phase * 0.5).cos(),
        0.8 * (phase * 0.2).sin(),
    );
    (accel, gyro)
}

fn pose_at(i: usize) -> (Vector3<f32>, UnitQuaternion<f32>) {
    let phase = i as f32 * 1e-3;
    // Slow walk so successive poses pass the gate without inflation.
    let pos = Vector3::new(0.001 * phase, 0.0, 0.0);
    let q = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.001 * phase));
    (pos, q)
}

#[test]
fn benchmark_predict_dense_vs_sparse() {
    let dt = 1e-3_f32;

    // Dense path
    let dense_secs = best_of(N_REPEATS, || {
        let mut e = fresh_filter();
        time_iters(N_PREDICT_ITERS, |i| {
            let (a, g) = imu_at(i);
            e.predict(a, g, dt);
        })
    });

    // Sparse path
    let sparse_secs = best_of(N_REPEATS, || {
        let mut e = fresh_filter();
        time_iters(N_PREDICT_ITERS, |i| {
            let (a, g) = imu_at(i);
            e.predict_sparse(a, g, dt);
        })
    });

    let dense_ns = dense_secs * 1e9 / N_PREDICT_ITERS as f64;
    let sparse_ns = sparse_secs * 1e9 / N_PREDICT_ITERS as f64;
    let speedup = dense_ns / sparse_ns;

    eprintln!("=== predict() ===");
    eprintln!(
        "  dense:  {dense_ns:>7.1} ns/call  ({:.3} s for {N_PREDICT_ITERS} iters)",
        dense_secs
    );
    eprintln!(
        "  sparse: {sparse_ns:>7.1} ns/call  ({:.3} s for {N_PREDICT_ITERS} iters)",
        sparse_secs
    );
    eprintln!("  speedup: {speedup:.2}×");

    // Lower bound: sparse should not be slower than dense. Loose enough
    // (0.95×) to absorb timing jitter; if sparse is consistently slower
    // than dense even with --release, that's a real regression.
    assert!(
        speedup > 0.95,
        "sparse predict slower than dense: {speedup:.2}×"
    );
}

#[test]
fn benchmark_update_pose_dense_vs_sparse() {
    // Update tests need the filter in a "running" regime — covariance
    // must be representative of a settled flight state, not initial
    // 1.0-on-the-diagonal P. Pre-roll a few hundred predict steps.
    let dt = 1e-3_f32;
    let preroll = |e: &mut Eskf| {
        for i in 0..1000 {
            let (a, g) = imu_at(i);
            e.predict(a, g, dt);
        }
    };

    let dense_secs = best_of(N_REPEATS, || {
        let mut e = fresh_filter();
        preroll(&mut e);
        time_iters(N_UPDATE_ITERS, |i| {
            let (p, q) = pose_at(i);
            let _ = e.update_pose(p, q, 0.05, 0.05);
        })
    });

    let sparse_secs = best_of(N_REPEATS, || {
        let mut e = fresh_filter();
        preroll(&mut e);
        time_iters(N_UPDATE_ITERS, |i| {
            let (p, q) = pose_at(i);
            let _ = e.update_pose_sparse(p, q, 0.05, 0.05);
        })
    });

    let dense_ns = dense_secs * 1e9 / N_UPDATE_ITERS as f64;
    let sparse_ns = sparse_secs * 1e9 / N_UPDATE_ITERS as f64;
    let speedup = dense_ns / sparse_ns;

    eprintln!("=== update_pose() ===");
    eprintln!(
        "  dense:  {dense_ns:>7.1} ns/call  ({:.3} s for {N_UPDATE_ITERS} iters)",
        dense_secs
    );
    eprintln!(
        "  sparse: {sparse_ns:>7.1} ns/call  ({:.3} s for {N_UPDATE_ITERS} iters)",
        sparse_secs
    );
    eprintln!("  speedup: {speedup:.2}×");

    assert!(
        speedup > 0.95,
        "sparse update_pose slower than dense: {speedup:.2}×"
    );
}

/// Combined benchmark mirroring the firmware's actual workload:
/// 1 kHz predict + 100 Hz pose updates = ~10 predicts per update.
/// This is what shows up in the H743's CPU budget.
#[test]
fn benchmark_combined_workload() {
    let dt = 1e-3_f32;
    let n_seconds = 1; // simulated wall-clock seconds
    let n_predicts = n_seconds * 1000;
    let n_updates_per_predict = 1; // 1 update per 10 predicts → 100 Hz
    let inner_loop = move |e: &mut Eskf| {
        for i in 0..n_predicts {
            let (a, g) = imu_at(i);
            // Inline the dispatch through predict() so we hit the same
            // call-site shape as the firmware.
            // (Replaced by predict_sparse below in the sparse arm.)
            e.predict(a, g, dt);
            if i % 10 == 9 {
                for k in 0..n_updates_per_predict {
                    let (p, q) = pose_at(i + k);
                    let _ = e.update_pose(p, q, 0.05, 0.05);
                }
            }
        }
    };
    let inner_loop_sparse = move |e: &mut Eskf| {
        for i in 0..n_predicts {
            let (a, g) = imu_at(i);
            e.predict_sparse(a, g, dt);
            if i % 10 == 9 {
                for k in 0..n_updates_per_predict {
                    let (p, q) = pose_at(i + k);
                    let _ = e.update_pose_sparse(p, q, 0.05, 0.05);
                }
            }
        }
    };

    let dense_secs = best_of(N_REPEATS, || {
        let mut e = fresh_filter();
        let t0 = Instant::now();
        // Run multiple seconds-worth of work back-to-back to amortize
        // measurement overhead.
        for _ in 0..50 {
            inner_loop(&mut e);
        }
        t0.elapsed().as_secs_f64()
    });
    let sparse_secs = best_of(N_REPEATS, || {
        let mut e = fresh_filter();
        let t0 = Instant::now();
        for _ in 0..50 {
            inner_loop_sparse(&mut e);
        }
        t0.elapsed().as_secs_f64()
    });

    let total_calls = (n_predicts + n_updates_per_predict * (n_predicts / 10)) * 50;
    let dense_us_per_sec = dense_secs * 1e6 / 50.0;
    let sparse_us_per_sec = sparse_secs * 1e6 / 50.0;
    let speedup = dense_secs / sparse_secs;

    eprintln!("=== combined 1 s of firmware workload ===");
    eprintln!(
        "  dense:  {:.0} µs of host CPU per simulated second  (avg {:.1} ns/call over {} calls)",
        dense_us_per_sec,
        dense_secs * 1e9 / total_calls as f64,
        total_calls
    );
    eprintln!(
        "  sparse: {:.0} µs of host CPU per simulated second  (avg {:.1} ns/call over {} calls)",
        sparse_us_per_sec,
        sparse_secs * 1e9 / total_calls as f64,
        total_calls
    );
    eprintln!("  speedup: {speedup:.2}×");

    assert!(
        speedup > 0.95,
        "sparse workload slower than dense: {speedup:.2}×"
    );
}
