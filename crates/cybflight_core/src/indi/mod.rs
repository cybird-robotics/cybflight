/// Fallback low-pass cutoff when a configured one is structurally unusable
/// (non-finite or ≤ 0). Matches the `indi_sync_hz` schema default, so the
/// degraded controller behaves like an untuned but flyable one rather than
/// like a broken filter.
const FALLBACK_CUTOFF_HZ: f32 = 12.0;

/// Largest fraction of the sample rate we will place a low-pass cutoff at.
/// `air-filters` rejects `fc >= fs/2` outright; 0.4 keeps the response
/// meaningful rather than merely legal.
const MAX_CUTOFF_FRACTION: f32 = 0.4;

/// Clamp a low-pass cutoff to something the biquad builder will accept at
/// `sample_hz`, so no parameter value can panic the inner loop.
///
/// This exists because the INDI filter cutoffs are runtime parameters while
/// the sample rate is a compile-time build knob, and the two are validated
/// independently: `indi_sync_hz` is schema-bounded to 500 Hz, which is
/// legal at the 8 kHz default and is *exactly Nyquist* on an `imu_1khz`
/// build — where `BiquadFilter::new(...).expect(...)` would panic
/// `IndiController::new` into a flash-persistent boot loop. Per
/// docs/safety_protocol.md rule 2, a config problem degrades; it never
/// kills the task. Callers that care can compare the result against what
/// they asked for and log the difference.
pub fn clamp_cutoff_hz(cutoff_hz: f32, sample_hz: f32) -> f32 {
    let requested = if cutoff_hz.is_finite() && cutoff_hz > 0.0 {
        cutoff_hz
    } else {
        FALLBACK_CUTOFF_HZ
    };
    let max = MAX_CUTOFF_FRACTION * sample_hz;
    if requested < max { requested } else { max }
}

pub mod controller;
pub mod effectiveness;
pub mod linearization;
pub mod rate_dot_estimator;
pub mod rpm_notch;
pub mod rpm_tracker;
pub mod thrust_table;
