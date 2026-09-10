//! Effective IMU / inner-loop rate for this build.
//!
//! This is the single `cfg!` site for the `imu_1khz` build knob (vehicle
//! YAML `build: imu_rate`); control logic consumes the consts below and
//! never touches the feature directly. `bsp::PRIMARY_GYRO_ODR_HZ` remains
//! the board's *native* full rate — this module decides what the firmware
//! actually programs and runs at.

use cybflight_drivers::imu::icm426xx::OutputDataRate;

/// Effective primary-IMU ODR = INDI inner-loop rate for this build.
///
/// The IMU driver's `sample_rate_hz()` must agree with this — enforced by a
/// `defmt::assert!` at each ICM init site in `board_init`.
pub const IMU_ODR_HZ: f32 = if cfg!(feature = "imu_1khz") {
    1000.0
} else {
    crate::bsp::PRIMARY_GYRO_ODR_HZ
};

/// Driver-facing ODR selection for ICM426xx boards (single source with
/// [`IMU_ODR_HZ`]).
pub const ICM_ODR: OutputDataRate = if cfg!(feature = "imu_1khz") {
    OutputDataRate::Odr1kHzLowNoise
} else {
    OutputDataRate::Odr8kHz
};

/// Group-delay budget for the INDI rate-derivative estimator's SG stage.
///
/// The `rate_dot` path is the one INDI signal whose delay is NOT set by
/// `indi_sync_hz`: the SG filter adds `(window − 1) / 2` samples on top of
/// the shared post-biquad, so it is pure delay mismatch against `spf_fs`,
/// `u_state_fs` and `omega_fs`. 2 ms is ~11 % of the 18.8 ms group delay of
/// a 12 Hz sync filter — small enough not to matter, and the value the
/// hand-tuned 1 kHz window already implied.
const RATE_DOT_SG_MAX_GROUP_DELAY_S: f32 = 2.0e-3;

/// Upper bound on the SG window regardless of rate. Past ~13 taps the fit
/// stops buying noise rejection on a signal the post-biquad has to smooth
/// anyway, and `air-filters` caps the window at 19.
const RATE_DOT_SG_WINDOW_MAX: i32 = 13;

/// Largest odd SG window whose group delay fits
/// [`RATE_DOT_SG_MAX_GROUP_DELAY_S`] at `odr_hz`, capped both ways.
///
/// `pub` and callable at runtime: INDI calls it with its *control* rate
/// (`IMU_ODR_HZ / indi_ctrl_div`, a vehicle param), not the IMU ODR —
/// 2 kHz → 9 taps (2 ms), which is what the 8 kHz / 4 vehicles run.
///
/// Derived rather than hand-picked per build so that *every* IMU rate gets
/// a chosen delay instead of an inherited one. The two rates that were
/// previously hardcoded come out unchanged, and the third board no longer
/// falls through to the 8 kHz value by accident:
///
/// | ODR | half-window | window | SG group delay |
/// |---|---|---|---|
/// | 8 kHz (ICM default) | 16 → capped 6 | 13 | 0.75 ms |
/// | 3.2 kHz (BMI270)    | 6            | 13 | 1.88 ms |
/// | 1 kHz (`imu_1khz`)  | 2            |  5 | 2.00 ms |
pub const fn rate_dot_sg_window(odr_hz: f32) -> i32 {
    let half = (odr_hz * RATE_DOT_SG_MAX_GROUP_DELAY_S) as i32;
    let window = 2 * half + 1;
    if window < 3 {
        3
    } else if window > RATE_DOT_SG_WINDOW_MAX {
        RATE_DOT_SG_WINDOW_MAX
    } else {
        window
    }
}

/// Savitzky–Golay window for the INDI rate-derivative estimator, sized for
/// this build's [`IMU_ODR_HZ`]. See [`rate_dot_sg_window`] for the table.
///
/// The SG filter runs at the loop rate (there is no decimation stage), so
/// the window is the only thing setting its span; less SG smoothing at
/// 1 kHz is acceptable because the chip's 227 Hz UI filter already
/// band-limits the gyro in that mode.
pub const RATE_DOT_SG_WINDOW: i32 = rate_dot_sg_window(IMU_ODR_HZ);

/// Wall-clock producer backlog the IMU-rate PubSub channels absorb before
/// a consumer starts seeing `Lagged`.
///
/// Sized in *time*, not samples, because the thing it has to survive is a
/// consumer stall, and stalls are wall-clock events: an SD-card
/// garbage-collection pause during a blackbox session is 50–250 ms on
/// consumer cards, an ESP-bridge or USB backpressure hiccup is a few ms.
/// A fixed sample count silently means eight times less tolerance at
/// 8 kHz than at 1 kHz, which is exactly backwards — the faster build is
/// the one under more pressure.
///
/// 8 ms does not cover a full card GC pause; nothing bounded does, short
/// of a megabyte of queue. It covers the ordinary case (block-write
/// latency, a preempted drain, a burst of higher-priority work) so that
/// drops become a genuine-overload signal instead of routine noise.
/// 24 ms: bench logs at the sysid tier show the recorder's CMD25 flush
/// stalls at 10–18 ms (a ~16 KB cluster batch + FAT-table write on a
/// commodity card). At the previous 8 ms every stall lost
/// `stall − 8 ms` of IMU samples (~7–9 % of `/imu1`); 24 ms covers the
/// observed distribution with margin. Cost at 8 kHz: 192 slots × 3
/// channels ≈ 27 KiB `.bss`.
const IMU_PUBSUB_BUFFER_S: f32 = 24.0e-3;

/// Slots in the IMU-rate PubSub channels, from
/// [`IMU_PUBSUB_BUFFER_S`] at `odr_hz`.
///
/// | ODR | slots | buffered |
/// |---|---|---|
/// | 8 kHz (ICM default) | 192 | 24 ms |
/// | 3.2 kHz (BMI270)    | 76 | 23.8 ms |
/// | 1 kHz (`imu_1khz`)  | 24 | 24 ms |
const fn imu_pubsub_cap(odr_hz: f32) -> usize {
    let n = (odr_hz * IMU_PUBSUB_BUFFER_S) as usize;
    // Floor at the historical value so no build ends up shallower than
    // what already flew.
    if n < 4 { 4 } else { n }
}

/// CAP for `IMU_1` / `IMU_1_RAW` / `IMU_2`, sized for this build's
/// [`IMU_ODR_HZ`]. See [`imu_pubsub_cap`] for the table.
///
/// Costs `3 × CAP × (size_of::<msgs::Imu>() + 8)` bytes of `.bss` — about
/// 9 KiB at 8 kHz, so the 8 kHz builds pay for the depth they need and
/// the 1 kHz builds don't. `just size` reports the current headroom.
///
/// **This sizes jitter tolerance, not throughput.** If a consumer is
/// structurally slower than the producer — the blackbox recorder's
/// `Large` tier at 8 kHz is the case to watch, where total SD write
/// bandwidth is the binding constraint — a deeper queue only delays the
/// drops and makes them burstier. Fix throughput there, not CAP.
pub const IMU_PUBSUB_CAP: usize = imu_pubsub_cap(IMU_ODR_HZ);

/// ESKF odometry publish rate. Both ESKF tasks decimate their predict to
/// ~1 kHz (`PREDICT_DECIMATION`) and publish on every predict
/// (`ODOM_DECIMATION` = 1), so `sensors::VEHICLE_ODOMETRY` runs at 1 kHz
/// in **every** build. Unlike the IMU channels it does not scale with
/// [`IMU_ODR_HZ`], which is why the CAP below is a plain constant.
pub const ODOM_PUBLISH_HZ: f32 = 1000.0;

/// Blackbox odometry decimation: `sensors::BLACKBOX_ODOMETRY` carries
/// one ESKF sample in N, so the recorder's channel buffers **records**.
///
/// 8 → 125 Hz nominal. The requirement is "≥ 100 Hz at the reader", and
/// the recorder still drops some of what it is offered, so the source
/// rate is set above the target rather than at it: 125 Hz offered leaves
/// ≥ 100 Hz on the card with room for the residual loss.
///
/// Why a separate channel rather than a divider on `VEHICLE_ODOMETRY`:
/// dividing behind a PubSub costs exactly what it saves — the discarded
/// sample has already taken a channel slot and a drain-budget iteration
/// — and `VEHICLE_ODOMETRY` itself cannot be thinned at the publisher
/// because the inner loop, the ESKF consumers and the outer loop read
/// it. Mirroring is the same pattern [`IMU_DECIM_DIV`] uses for
/// `sensors::IMU_1_DECIM`. Measured 2026-09-09: at `rate_div 2` the
/// post-buffer divider halved `/imu1_raw`'s per-pass ceiling (24 → 12
/// records) and its logged rate fell 489 → 268 Hz.
pub const BLACKBOX_ODOM_DECIM: u32 = 8;

/// Rate on `sensors::BLACKBOX_ODOMETRY` = [`ODOM_PUBLISH_HZ`] /
/// [`BLACKBOX_ODOM_DECIM`].
pub const BLACKBOX_ODOM_HZ: f32 = ODOM_PUBLISH_HZ / BLACKBOX_ODOM_DECIM as f32;

/// Consumer-stall tolerance for `sensors::BLACKBOX_ODOMETRY`.
///
/// Sized against the *recorder's* absence, not an SD stall: sharing the
/// thread executor with the MPC, its p99 gap between drain passes was
/// 77–212 ms across the 2026-09-09 flights. 200 ms covers that; at
/// 125 Hz that is 25 slots, and because these are records the whole
/// buffer survives to the file instead of a fraction of it.
const BLACKBOX_ODOM_BUFFER_S: f32 = 200.0e-3;

/// CAP for `sensors::BLACKBOX_ODOMETRY`, from
/// [`BLACKBOX_ODOM_BUFFER_S`] at [`BLACKBOX_ODOM_HZ`]. 25 slots =
/// 200 ms. Costs `25 × (size_of::<msgs::VehicleOdometry>() + 8)` of
/// `.bss`; `just size` reports the headroom.
pub const BLACKBOX_ODOM_PUBSUB_CAP: usize = {
    let n = (BLACKBOX_ODOM_HZ * BLACKBOX_ODOM_BUFFER_S) as usize;
    if n < 8 { 8 } else { n }
};

/// Blackbox power-telemetry decimation: `sensors::POWER_TELEM` carries
/// one `power_task` tick in N.
///
/// 10 → 10 Hz from the task's 100 Hz ADC tick. Thinned at the publisher
/// for the same reason as `IMU_1_RAW`, and safe there for the same
/// reason: the recorder is `POWER_TELEM`'s only subscriber (every other
/// consumer — INDI thrust table, shell, GCS — reads `POWER_STATUS`,
/// which is untouched and stays at 100 Hz).
///
/// **What 10 Hz keeps and loses.** `batt_lpf_hz` defaults to 2 Hz
/// (τ ≈ 80 ms), so the *filtered* pack voltage is still ~5× oversampled:
/// sag trends and the thrust table's operating voltage are intact. The
/// *raw* voltage under a throttle punch has content well above 5 Hz and
/// is aliased at this rate — measuring how far the filter lags a
/// transient needs the full 100 Hz, i.e. `blackbox set large` or a
/// bespoke session.
pub const BLACKBOX_POWER_DECIM: u32 = 10;

/// Rate on `sensors::POWER_TELEM` = 100 Hz `power_task` tick /
/// [`BLACKBOX_POWER_DECIM`]. At the channel's CAP of 8 that is 800 ms of
/// buffer, far beyond the recorder's worst measured absence.
pub const BLACKBOX_POWER_HZ: f32 = 100.0 / BLACKBOX_POWER_DECIM as f32;

/// Rate of the INDI telemetry mirrors (`ACTUATOR_MOTORS_TELEM`,
/// `PROCESSED_MOTOR_STATE`, `TRACKING_ERROR`) at the sysid tier, where
/// `RecordSet::fast_indi_telem` raises them from 100 Hz.
pub const INDI_TELEM_SYSID_HZ: f32 = 500.0;

/// Consumer-stall tolerance for the INDI telemetry mirrors.
///
/// Sized against the recorder's absence between drain passes, which is
/// set by how long the outer loop holds the shared thread executor. At
/// `mpc_rate_hz` 100 a solve blocks 7.27 ms of every 10 ms, so the
/// recorder runs in ~2.7 ms slices and its service interval measures
/// p99 48.5 ms, max 191.9 ms (flight_0010, 2026-09-09). The previous
/// 32.3 ms (CAP 16 at 500 Hz) sat below that p99 and lost 33 % of both
/// topics; 97 ms clears it with ~2x margin.
///
/// Not sized for the max: covering 191.9 ms would need CAP 96, and the
/// tier already offers ~136 KiB/s against ~140 sustainable — past this
/// point a deeper queue stops eliminating drops and starts moving them
/// onto `/imu1_raw`, which drains last by design. See
/// `blackbox::record_set`'s drop-priority section.
const INDI_TELEM_BUFFER_S: f32 = 97.0e-3;

/// CAP for the INDI telemetry mirrors, from [`INDI_TELEM_BUFFER_S`] at
/// [`INDI_TELEM_SYSID_HZ`]. 48 slots = 97 ms.
///
/// `MotorStateTelemetry` is the expensive one at ~80 B/slot
/// (`Option<f32>` has no niche), so the pair costs ~3.6 KiB of `.bss`
/// over the old 16. `just size` reports the headroom; boot-test on an
/// 8 kHz vehicle before flying it.
pub const INDI_TELEM_PUBSUB_CAP: usize = {
    let n = (INDI_TELEM_SYSID_HZ * INDI_TELEM_BUFFER_S) as usize;
    if n < 16 { 16 } else { n }
};

/// Decimation from the IMU ODR to the ~1 kHz `sensors::IMU_1_DECIM`
/// mirror (8 at 8 kHz, 3 at 3.2 kHz, 1 at 1 kHz). Consumers that only
/// need ~1 kHz (the Mahony filter) subscribe there instead of `IMU_1`,
/// so the full-rate publish does not wake them 8000×/s to discard 7 of
/// every 8 samples on the thread executor.
pub const IMU_DECIM_DIV: u32 = {
    let d = (IMU_ODR_HZ / 1000.0) as u32;
    if d == 0 { 1 } else { d }
};

#[cfg(all(feature = "imu_1khz", feature = "board_micoair743v2"))]
compile_error!(
    "imu_1khz applies to ICM426xx boards; micoair743v2 (BMI270) runs at its native 3.2 kHz"
);

// This crate only ever builds for thumbv7em, so a `#[cfg(test)]` module here
// would never run. These are const assertions instead: they are checked on
// every firmware build of every board, which is the stronger guarantee for a
// compile-time constant anyway.
const _: () = {
    // The three real IMU rates. The first two must reproduce the values that
    // were hardcoded before this became a derivation, so no flying vehicle
    // changed behaviour; the third is the board that used to inherit the
    // 8 kHz window by accident.
    assert!(rate_dot_sg_window(8000.0) == 13, "8 kHz ICM window changed");
    assert!(rate_dot_sg_window(1000.0) == 5, "imu_1khz window changed");
    assert!(rate_dot_sg_window(3200.0) == 13, "3.2 kHz BMI270 window changed");
    assert!(rate_dot_sg_window(2000.0) == 9, "2 kHz (8 kHz / indi_ctrl_div 4) window changed");
    // Structural invariants of whatever this build resolved to: the SG
    // filter requires an odd window in [3, 19].
    assert!(RATE_DOT_SG_WINDOW % 2 == 1, "SG window must be odd");
    assert!(RATE_DOT_SG_WINDOW >= 3, "SG window below the filter minimum");
    assert!(
        RATE_DOT_SG_WINDOW <= RATE_DOT_SG_WINDOW_MAX,
        "SG window above the cap",
    );

    // Same three rates for the PubSub depth. These pin the float
    // arithmetic in `imu_pubsub_cap` — the derivation rounds, so the
    // table in its doc comment is only trustworthy if it is asserted.
    assert!(imu_pubsub_cap(8000.0) == 192, "8 kHz IMU pubsub CAP changed");
    assert!(imu_pubsub_cap(1000.0) == 24, "imu_1khz pubsub CAP changed");
    assert!(imu_pubsub_cap(3200.0) == 76, "3.2 kHz IMU pubsub CAP changed");
    // Never shallower than the pre-derivation value that flew.
    assert!(IMU_PUBSUB_CAP >= 4, "IMU pubsub CAP below the historical floor");

    // Odometry is 1 kHz in every build, so unlike the IMU CAP this is a
    // single value rather than a table — pin it, and keep it at or above
    // the CAP=8 that flew before 2026-09-09.
    assert!(BLACKBOX_ODOM_HZ >= 100.0, "blackbox odometry below the 100 Hz sysid floor");
    assert!(BLACKBOX_ODOM_PUBSUB_CAP == 25, "blackbox odometry pubsub CAP changed");
    assert!(BLACKBOX_POWER_HZ >= 10.0, "blackbox power below the 10 Hz floor");
    assert!(INDI_TELEM_PUBSUB_CAP == 48, "INDI telemetry pubsub CAP changed");
    assert!(INDI_TELEM_PUBSUB_CAP >= 16, "INDI telemetry CAP below the historical floor");
};
