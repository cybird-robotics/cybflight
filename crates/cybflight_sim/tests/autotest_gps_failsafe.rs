//! Failsafe regression for the GPS-driven ESKF guard.
//!
//! Models the audit-flagged "permanent jump-cascade wedge" failure
//! mode (`docs/architecture.md` and the eskf_imu_gps audit): a
//! sustained GPS shift — what RTK ambiguity loss / multipath
//! wraparound looks like in practice — pushes every fresh PVT 5 m
//! away from the IMU-integrated position. The 3 m jump gate fires
//! every frame, the consecutive-jump counter saturates, and
//! `MAX_CONSECUTIVE_JUMPS` triggers.
//!
//! With Phase-1 guard semantics (the docstring promises a re-init at
//! the cascade limit but the code only drops `converged`) the filter
//! never re-anchors and `jump_total` keeps growing for the rest of
//! the run. With the Phase-5 fix (re-init at the offending PVT when
//! `carr_soln >= reinit_min_carr_soln`) the first cascade is the
//! last cascade — `jump_total` is bounded.
//!
//! Two assertions:
//!   1. The cascade gate fired during the fault window
//!      (`max consecutive_jumps >= 2`).
//!   2. After a recovery deadline well past `RTK_FIX_DEBOUNCE`, no
//!      new jumps have been recorded for at least 3 s (`jump_total`
//!      is stable). Phase-1 code FAILS this — that's the bug
//!      demonstration. Phase-5 PASSES.
//!
//! Run:
//!   cargo test -p cybflight-sim --target x86_64-unknown-linux-gnu \
//!       --profile release-host --test autotest_gps_failsafe -- --nocapture

use cybflight_sim::{
    controller::MpcIndiController,
    plant::QuadPlant,
    runner::{MissionRunner, StepRecord},
    scenario::Scenario,
    sensors::{FaultedGps, NoisyGps, NoisyImu, OutageGps},
};
use nalgebra::Vector3;

const FAULT_START_S: f32 = 8.0;
/// Sustained fault — extends past the simulation horizon. Models a
/// real-world RTK ambiguity loss / multipath wrap that doesn't heal.
const FAULT_END_S: f32 = 1.0e9;
const FAULT_BIAS_M: Vector3<f32> = Vector3::new(5.0, 0.0, 0.0);

/// Pick the latest history sample at or before `t`. `None` if no
/// sample is recorded that early.
fn sample_at_or_before(history: &[StepRecord], t_target_s: f32) -> Option<&StepRecord> {
    history
        .iter()
        .rev()
        .find(|r| r.t <= t_target_s)
}

#[test]
fn jump_cascade_recovers_after_sustained_gps_excursion() {
    let inner_gps = NoisyGps::isotropic(0xDEADBEEF, 5.0, 0.5, 0.2);
    let faulted = FaultedGps::new(inner_gps, FAULT_START_S, FAULT_END_S, FAULT_BIAS_M);

    let mut scenario = Scenario::mission(
        "mission_square_gps_failsafe",
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
    )
    .with_gps(Box::new(faulted));
    // Plenty of headroom past the cascade trigger so we can observe
    // jump-counter stability (or lack thereof) for several seconds.
    scenario.terminal_hold_s = 12.0;
    // The mission is expected to NOT track well — the failsafe is
    // the focus. Loose pass-criteria stop the runner from bailing on
    // a tracking-error early-exit before we get our assertion data.
    scenario.pass_criteria.terminal_pos_err_m = 50.0;
    scenario.pass_criteria.rms_pos_err_m = 50.0;
    scenario.pass_criteria.peak_tilt_rad = 89.0_f32.to_radians();
    scenario.pass_criteria.geofence_min = Vector3::new(-50.0, -50.0, -2.0);
    scenario.pass_criteria.geofence_max = Vector3::new(50.0, 50.0, 50.0);

    let mut controller = MpcIndiController::from_params(&scenario.vehicle_params);
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    // ── Assertion 1: cascade fired ────────────────────────────────────
    // Use the cumulative `jump_total` counter rather than
    // `consecutive_jumps`. After the Phase-5 fix, `consecutive_jumps`
    // is reset to 0 *inside* the same `on_pvt` call that triggers
    // the re-init, so the history-recorded peak may never reach the
    // cascade limit (the `=2 → 0` transition happens between
    // history samples).
    let max_jump_total = out
        .history
        .iter()
        .filter_map(|r| r.estimator.as_ref())
        .map(|s| s.jump_total)
        .max()
        .unwrap_or(0);
    assert!(
        max_jump_total >= 2,
        "expected the jump gate to fire at least twice during the fault \
         window — got jump_total={max_jump_total}. Either the bias is \
         too small to trip the 3 m gate, or the FaultedGps decorator \
         is not wired correctly."
    );

    // ── Assertion 2: jump_total is stable after recovery deadline ────
    // Pick two samples post-recovery; if `jump_total` still grew
    // between them, the guard is wedged (Phase 1 bug). After the
    // Phase-5 re-init the cascade fires once, the filter re-anchors
    // at the biased GPS, subsequent fixes match the new anchor, and
    // `jump_total` plateaus.
    let last_sample = out
        .history
        .last()
        .expect("scenario must produce at least one history sample");
    let observation_end_t = last_sample.t;
    let observation_start_t = observation_end_t - 3.0;

    let early = sample_at_or_before(&out.history, observation_start_t).expect(
        "scenario should produce samples ≥3 s before end — increase terminal_hold_s",
    );
    let late = last_sample;

    let early_est = early
        .estimator
        .as_ref()
        .expect("GPS scenario must populate estimator snapshots");
    let late_est = late
        .estimator
        .as_ref()
        .expect("GPS scenario must populate estimator snapshots");

    let new_jumps = late_est.jump_total.saturating_sub(early_est.jump_total);

    println!(
        "autotest_gps_failsafe: max_jump_total={max_jump_total} \
         (cascade gate at MAX_CONSECUTIVE_JUMPS=2 fired)",
    );
    println!(
        "autotest_gps_failsafe: at t={:.2}s — converged={} rtk_ok={} jump_total={} \
         consecutive_jumps={} est=({:.2},{:.2},{:.2}) truth=({:.2},{:.2},{:.2})",
        late.t,
        late_est.converged,
        late_est.rtk_quality_ok,
        late_est.jump_total,
        late_est.consecutive_jumps,
        late.estimator_position.unwrap_or_default().x,
        late.estimator_position.unwrap_or_default().y,
        late.estimator_position.unwrap_or_default().z,
        late.position.x,
        late.position.y,
        late.position.z,
    );
    println!(
        "autotest_gps_failsafe: window [{:.2}s, {:.2}s] new jumps = {} \
         (Phase 1 expected to grow; Phase 5 expected = 0)",
        early.t, late.t, new_jumps,
    );

    assert_eq!(
        new_jumps, 0,
        "guard is wedged: {} new GPS jump rejections recorded between t={:.2}s \
         and t={:.2}s — the cascade did not heal. \
         This is the Phase-1 bug: `MAX_CONSECUTIVE_JUMPS` drops `converged` \
         but does not re-anchor the filter, so every subsequent biased PVT \
         keeps tripping the jump gate. The Phase-5 fix calls \
         `init_with_cov` at the offending PVT (gated on \
         `carr_soln >= reinit_min_carr_soln`), absorbing the shift in one step.",
        new_jumps, early.t, late.t,
    );
}

/// Critical #2 regression: a transient GPS *outage* followed by
/// recovery into a position discrepancy is exactly the "long-outage
/// cascade wedge" the audit flagged. Real-world variants:
///   - IMU drifts during outage; fresh post-outage GPS doesn't match.
///   - Multipath / RTK ambiguity slip happens to coincide with the
///     outage end, so the first usable frame is shifted.
///
/// Either way the first post-outage frame trips the 3 m jump gate,
/// the cascade fires, and the Phase-5 re-init absorbs the shift.
/// This test composes [`OutageGps`] (window with no usable PVT)
/// with [`FaultedGps`] (post-outage position bias) — the IMU side
/// is left clean because the in-loop ESKF estimates and cancels
/// constant accel/gyro biases during pre-outage operation, which
/// would silently negate any IMU-side bias added here.
#[test]
fn gps_outage_recovers_via_jump_cascade_reinit() {
    const OUTAGE_START_S: f32 = 6.0;
    const OUTAGE_END_S: f32 = 10.0;
    const POST_OUTAGE_BIAS_M: Vector3<f32> = Vector3::new(5.0, 0.0, 0.0);

    let inner_gps = NoisyGps::isotropic(0xCAFEBABE, 5.0, 0.5, 0.2);
    let outage = OutageGps::new(inner_gps, OUTAGE_START_S, OUTAGE_END_S);
    // After the outage clears, the receiver returns with a 5 m
    // east shift. From the failsafe's point of view this is
    // indistinguishable from "IMU drifted during outage" — both
    // produce a sustained pos discrepancy that trips the jump gate.
    let post_outage_shifted = FaultedGps::new(
        outage,
        OUTAGE_END_S,
        f32::INFINITY,
        POST_OUTAGE_BIAS_M,
    );

    // Light IMU noise without bias — bias would be silently
    // cancelled by the ESKF's pre-outage bias estimation, so it's
    // not a useful knob here. The GPS-side shift drives the test.
    let imu = NoisyImu::isotropic(0xCAFEBABE, 0.001, 0.05);

    let mut scenario = Scenario::mission(
        "mission_square_gps_outage",
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
    )
    .with_imu(Box::new(imu))
    .with_gps(Box::new(post_outage_shifted));
    // Plenty of headroom past the outage so we can observe both
    // the staleness gate firing and the post-outage recovery.
    scenario.terminal_hold_s = 14.0;
    // Loose pass-criteria — we're testing failsafe recovery, not
    // tracking quality. The biased IMU + outage will produce
    // significant tracking error; what matters is that the
    // estimator state machine recovers to ready=true.
    scenario.pass_criteria.terminal_pos_err_m = 50.0;
    scenario.pass_criteria.rms_pos_err_m = 50.0;
    scenario.pass_criteria.peak_tilt_rad = 89.0_f32.to_radians();
    scenario.pass_criteria.geofence_min = Vector3::new(-50.0, -50.0, -2.0);
    scenario.pass_criteria.geofence_max = Vector3::new(50.0, 50.0, 50.0);

    let mut controller = MpcIndiController::from_params(&scenario.vehicle_params);
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    // ── Assertion 1: cascade fired post-outage ────────────────────────
    // At least one jump must be recorded after GPS resumes —
    // confirming the IMU bias actually drove the estimate far
    // enough off truth to trip the gate.
    let post_outage_jumps = out
        .history
        .iter()
        .filter(|r| r.t >= OUTAGE_END_S)
        .filter_map(|r| r.estimator.as_ref())
        .map(|s| s.jump_total)
        .max()
        .unwrap_or(0)
        .saturating_sub(
            out.history
                .iter()
                .filter(|r| r.t < OUTAGE_END_S)
                .filter_map(|r| r.estimator.as_ref())
                .map(|s| s.jump_total)
                .max()
                .unwrap_or(0),
        );
    assert!(
        post_outage_jumps >= 1,
        "expected at least one GPS jump after outage end (IMU should have \
         drifted past the 3 m gate during the outage) — got {post_outage_jumps} \
         new jumps. Either accel bias is too small or outage too short."
    );

    // ── Assertion 2: ready re-acquires by end of run ─────────────────
    // After the cascade-driven re-init, RTK debounce takes 2 s to
    // re-acquire. Sample the last record; expect ready = true.
    let last = out
        .history
        .last()
        .expect("scenario must produce history samples");
    let last_est = last
        .estimator
        .as_ref()
        .expect("GPS scenario must populate estimator snapshots");
    let ready = last_est.converged && last_est.rtk_quality_ok;

    println!(
        "gps_outage_recovers: at t={:.2}s — converged={} rtk_ok={} \
         jump_total={} consecutive_jumps={} post_outage_jumps={}",
        last.t,
        last_est.converged,
        last_est.rtk_quality_ok,
        last_est.jump_total,
        last_est.consecutive_jumps,
        post_outage_jumps,
    );

    assert!(
        ready,
        "ESTIMATOR_READY did not recover by end of run (t={:.2}s). \
         Final snapshot: converged={} rtk_quality_ok={} jump_total={}. \
         The Phase-5 re-init should have absorbed the post-outage shift \
         and convergence + RTK debounce should have flipped ready back true.",
        last.t, last_est.converged, last_est.rtk_quality_ok, last_est.jump_total,
    );
}

/// "Clean recovery" companion to [`gps_outage_recovers_via_jump_cascade_reinit`]:
/// a transient outage with NO post-outage shift. With `PerfectImu`,
/// the IMU integrates correctly during the outage so when GPS
/// resumes the fresh fix matches the estimator. Recovery should NOT
/// require the cascade re-init path — the staleness gate fires
/// during the outage, then heals naturally on the next accepted
/// PVT.
///
/// This locks in the basic staleness-recovery happy path so a
/// future change to the guard that breaks it (e.g. failing to
/// refresh `last_gps_accept_ms` on accept) is caught here, not
/// only in the harder cascade scenario.
#[test]
fn gps_outage_clean_recovery_no_cascade_needed() {
    const OUTAGE_START_S: f32 = 6.0;
    const OUTAGE_END_S: f32 = 10.0;

    let inner_gps = NoisyGps::isotropic(0xFEEDFACE, 5.0, 0.5, 0.2);
    let outage = OutageGps::new(inner_gps, OUTAGE_START_S, OUTAGE_END_S);

    let mut scenario = Scenario::mission(
        "mission_square_gps_clean_outage",
        Vector3::new(0.0, 0.0, 1.0),
        &[
            Vector3::new(3.0, 0.0, 1.0),
            Vector3::new(3.0, 3.0, 1.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
    )
    .with_gps(Box::new(outage));
    scenario.terminal_hold_s = 14.0;
    // Tracking-quality criteria are loose because the controller will
    // briefly fly without GPS during the outage. Failsafe behaviour
    // is the focus.
    scenario.pass_criteria.terminal_pos_err_m = 1.0;
    scenario.pass_criteria.rms_pos_err_m = 1.0;
    scenario.pass_criteria.peak_tilt_rad = 80.0_f32.to_radians();

    let mut controller = MpcIndiController::from_params(&scenario.vehicle_params);
    let mut plant = QuadPlant::new(scenario.vehicle_params.clone(), &scenario.sim_params, 1.0 / 8000.0);
    let runner = MissionRunner::new(Default::default());
    let out = runner.run(&mut scenario, &mut plant, &mut controller);

    // ── Assertion 1: staleness fired during the outage ────────────────
    // last_gps_accept_ms should freeze at the last pre-outage accept.
    // The runner ticks `on_predict_tick` at the controller rate, so by
    // OUTAGE_END (4 s of no usable PVTs, vs 2 s gps_stale_ms threshold)
    // we expect at least one history sample with `is_stale`-equivalent
    // state — i.e. converged=false despite previously being true.
    let pre_outage_converged = out
        .history
        .iter()
        .filter(|r| r.t < OUTAGE_START_S)
        .filter_map(|r| r.estimator.as_ref())
        .any(|s| s.converged);
    let stale_period_unconverged = out
        .history
        .iter()
        .filter(|r| {
            r.t >= OUTAGE_START_S + 2.5 // past gps_stale_ms after outage start
                && r.t < OUTAGE_END_S
        })
        .filter_map(|r| r.estimator.as_ref())
        .all(|s| !s.converged);
    assert!(
        pre_outage_converged,
        "expected pre-outage convergence, but no history sample reported \
         converged=true before t={OUTAGE_START_S:.2}s. The 2 s warm-up may \
         not be enough — extend or check the gyro_bias_cov_trace_xy threshold."
    );
    assert!(
        stale_period_unconverged,
        "expected staleness to drop converged=false during the outage \
         (t≥{:.2}s, t<{:.2}s) — never observed.",
        OUTAGE_START_S + 2.5,
        OUTAGE_END_S,
    );

    // ── Assertion 2: NO cascade-driven re-init occurred ──────────────
    // With PerfectImu, the IMU position matches truth during the
    // outage. Fresh post-outage GPS shouldn't trip the 3 m jump gate.
    // jump_total should stay at 0 throughout the run (or at most
    // pre-existing jumps from start-up transients, which we cap at 1
    // for slack).
    let max_jump_total = out
        .history
        .iter()
        .filter_map(|r| r.estimator.as_ref())
        .map(|s| s.jump_total)
        .max()
        .unwrap_or(0);
    assert!(
        max_jump_total <= 1,
        "clean outage recovery should not need the cascade path — \
         expected jump_total ≤ 1, got {max_jump_total}. If this fails, \
         the IMU has drifted unexpectedly during the outage and the \
         scenario is no longer a 'clean' test."
    );

    // ── Assertion 3: ready re-acquires after the outage ──────────────
    let last = out
        .history
        .last()
        .expect("scenario must produce history samples");
    let last_est = last
        .estimator
        .as_ref()
        .expect("GPS scenario must populate estimator snapshots");
    let ready = last_est.converged && last_est.rtk_quality_ok;

    println!(
        "gps_outage_clean: at t={:.2}s — converged={} rtk_ok={} \
         jump_total={} consecutive_jumps={}",
        last.t,
        last_est.converged,
        last_est.rtk_quality_ok,
        last_est.jump_total,
        last_est.consecutive_jumps,
    );

    assert!(
        ready,
        "ESTIMATOR_READY did not recover by end of run after a clean \
         outage. Final snapshot: converged={} rtk_quality_ok={} \
         jump_total={}. The staleness gate should clear on the first \
         post-outage accept and convergence should re-acquire from \
         normal gyro-bias estimation.",
        last_est.converged, last_est.rtk_quality_ok, last_est.jump_total,
    );
}
