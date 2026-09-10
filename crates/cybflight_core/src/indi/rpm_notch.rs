// RPM-tracking notch filter bank for gyro and accel signals.
//
// Each motor's known rotational frequency drives a cascade of biquad notch
// filters that suppress the narrow-band vibration the motor injects into the
// IMU. Without this stage, the only way to keep INDI's gyro→motor→vibration
// loop from oscillating at hover is to set the post-derivative `sync_filter`
// cutoff so low (~10 Hz) that the rate loop can't track aggressive
// trajectories. With it, the cutoff can be raised into the 60–80 Hz region
// without the loop closing on the airframe's structural mode.
//
// Ported from Betaflight `flight/rpm_filter.c` and `sensors/acceleration.c`.
// See `tmp/indi_c/rpm_filter.c` for the reference implementation.
//
// Key design decisions:
//
//   * **DirectForm1 biquads.** Coefficients change between samples (round-
//     robin batched updates). DF1 stores past *signal* values in its delay
//     line, so coefficient changes don't cause transients. DF2 stores
//     intermediate state derived from the coefficients and would glitch.
//
//   * **Round-robin batched coefficient refresh.** Recomputing all
//     `NU·NH·N_AXES` notches every sample is wasted work — motor frequency
//     moves slowly. Instead, refresh `notches_per_call` cursor positions per
//     call so the full table cycles within ~1 ms.
//
//   * **Per-axis biquads, shared frequency.** Coefficients are identical
//     across axes (the disturbance frequency is axis-invariant), but each
//     axis needs its own delay line. The reference C code computes
//     coefficients on ROLL and copies the five floats to PITCH/YAW; the
//     air-filters Rust API doesn't expose coefficient cloning, so we
//     instead call `set_cutoff_frequency_hz` on all three axes' biquads at
//     each refreshed cursor position — three trig recomputes instead of
//     one trig + 5-coefficient copy. With round-robin batching the cost
//     is trivial (see CPU budget in the implementation plan).
//
//   * **External weight crossfade.** When a motor approaches `min_hz`, the
//     notch fades to passthrough via `out = w·biquad(v) + (1−w)·v`. The
//     biquad still runs at w=0 so its delay line stays warm — avoids
//     transients when the notch re-engages on motor spin-up.
//
//     The fade is symmetric: it also runs *down* as a harmonic approaches
//     `max_hz` (0.48·sample rate), reaching passthrough at and above it.
//     Fading only at the bottom edge and clamping at the top left a
//     full-authority notch parked on `max_hz` whenever a harmonic ran off
//     the end of the band — attenuating a frequency where no motor tone
//     exists. At 8 kHz `max_hz` is 3840 Hz and no harmonic ever reaches it,
//     so this is invisible there; at 1 kHz (`imu_1khz`) it is 480 Hz, which
//     a hovering quad's 2nd harmonic already exceeds.
//
//   * **Safety**: non-finite motor frequency is treated as "motor at idle"
//     (clamped to `min_hz`, weight=0 → passthrough). The bank never
//     introduces NaN into the signal and never panics.

use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm1,
};
use air_filters::{CommonConfigurableFilter, Filter};
use nalgebra::Vector3;

const N_AXES: usize = 3;
/// Target time window over which the round-robin cursor visits every notch.
/// At 8 kHz with 12 gyro notches this works out to 2 refreshed per call;
/// at 4 notches (accel, 1 harmonic) it works out to 1.
const REFRESH_WINDOW_S: f32 = 0.001;

/// Cascade of RPM-tracking notch filters across `N_AXES` axes, `NU` motors,
/// and `NH` harmonics per motor. Coefficients are recomputed in a round-
/// robin so the entire bank refreshes within ~1 ms regardless of how many
/// notches it owns.
///
/// Generic over motor count and harmonic count so a single implementation
/// serves both the gyro path (typically 3 harmonics) and the accel path
/// (typically 1 harmonic).
pub struct RpmNotchBank<const NU: usize, const NH: usize> {
    /// `[axis][motor][harmonic]`. Each axis has its own delay line; the
    /// coefficients across axes for the same `(motor, harmonic)` are kept in
    /// sync by `update`.
    notches: [[[BiquadFilter<f32, DirectForm1<f32>>; NH]; NU]; N_AXES],
    /// `[motor][harmonic]`. Crossfade weight in `[0, 1]`. 0 = passthrough,
    /// 1 = full notch. Same value applies on every axis.
    weights: [[f32; NH]; NU],
    min_hz: f32,
    max_hz: f32,
    fade_range_hz: f32,
    /// Round-robin cursor.
    motor_idx: usize,
    harm_idx: usize,
    /// Notches refreshed per call to `update`. Sized so the cursor visits
    /// every `(motor, harmonic)` within `REFRESH_WINDOW_S`.
    notches_per_call: usize,
}

impl<const NU: usize, const NH: usize> RpmNotchBank<NU, NH> {
    /// Construct a bank with all notches initialised at `min_hz` (so they
    /// start as passthrough until the first `update`). `q` controls notch
    /// width: higher = narrower = less off-band phase loss but worse
    /// rejection if motor-frequency tracking is off; 5.0 matches Betaflight.
    /// `fade_range_hz` is the window above `min_hz` over which the notch
    /// fades in.
    pub fn new(sample_hz: f32, q: f32, min_hz: f32, fade_range_hz: f32) -> Self {
        // 0.48·Nyquist matches the C reference; staying back from the actual
        // Nyquist limit avoids biquad config errors at very high motor RPM.
        let max_hz = 0.48 * sample_hz;
        let make = || {
            let cfg = BiquadFilterConfigBuilder::direct_form_1()
                .sample_frequency_hz(sample_hz)
                .filter_type(BiquadFilterType::Notch)
                .cutoff_frequency_hz(min_hz)
                .q(q)
                .build()
                .expect("rpm_notch: biquad config invalid");
            BiquadFilter::new(cfg)
        };
        let total = NU * NH;
        // ceil(total / loops_per_window). Saturate at >=1 so a slow loop
        // (sample_hz·REFRESH_WINDOW_S < 1) still makes progress.
        let loops_per_window = (sample_hz * REFRESH_WINDOW_S).max(1.0);
        let notches_per_call = (libm::ceilf(total as f32 / loops_per_window) as usize).max(1);
        Self {
            notches: core::array::from_fn(|_| {
                core::array::from_fn(|_| core::array::from_fn(|_| make()))
            }),
            weights: [[0.0; NH]; NU],
            min_hz,
            max_hz,
            fade_range_hz,
            motor_idx: 0,
            harm_idx: 0,
            notches_per_call,
        }
    }

    /// Refresh the next batch of notches with current per-motor frequencies
    /// (Hz). Call once per loop iteration. Non-finite or out-of-range motor
    /// frequencies fade the corresponding notch to passthrough.
    pub fn update(&mut self, motor_freq_hz: &[f32; NU]) {
        for _ in 0..self.notches_per_call {
            let f_motor = motor_freq_hz[self.motor_idx];
            let f_h = ((self.harm_idx + 1) as f32) * f_motor;
            // Non-finite freq → fall back to min_hz (weight=0 → passthrough).
            // Hardens against bad dshot decode propagating into filter coeffs.
            let f_clamped = if f_h.is_finite() {
                f_h.clamp(self.min_hz, self.max_hz)
            } else {
                self.min_hz
            };
            // Symmetric crossfade. The lower edge fades in over
            // `[min_hz, min_hz + fade]`; the upper edge fades back out over
            // `[max_hz - fade, max_hz]` and is fully passthrough at or above
            // `max_hz`. The upper term reads the UNCLAMPED harmonic, so a
            // tone that has run past the band gets weight 0 instead of a
            // full-authority notch pinned to `max_hz`. `min` of the two
            // keeps the weight sane even if the fade bands overlap on a
            // narrow band (low sample rate, wide `fade_range_hz`).
            let weight = if f_h.is_finite() {
                let up = (f_h - self.min_hz) / self.fade_range_hz;
                let down = (self.max_hz - f_h) / self.fade_range_hz;
                up.min(down).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.weights[self.motor_idx][self.harm_idx] = weight;
            for axis in 0..N_AXES {
                // set_cutoff_frequency_hz can only fail on Nyquist violation;
                // we already clamped to 0.48·sample, so this is infallible
                // in practice. Discard the Result rather than unwrap to
                // guarantee no panic path if a future config change widens
                // the clamp.
                let _ = self.notches[axis][self.motor_idx][self.harm_idx]
                    .set_cutoff_frequency_hz(f_clamped);
            }
            // advance cursor (harmonic-major within motor)
            self.harm_idx += 1;
            if self.harm_idx >= NH {
                self.harm_idx = 0;
                self.motor_idx = (self.motor_idx + 1) % NU;
            }
        }
    }

    /// Apply all `NU·NH` notches on `axis` in cascade with crossfade weight.
    /// Cascade order is irrelevant (LTI). The biquad always runs even at
    /// w=0 so its delay line stays warm and re-engages without transients.
    #[inline]
    pub fn apply(&mut self, axis: usize, value: f32) -> f32 {
        debug_assert!(axis < N_AXES);
        let mut v = value;
        for motor in 0..NU {
            for harm in 0..NH {
                let raw = self.notches[axis][motor][harm].apply(v);
                let w = self.weights[motor][harm];
                v = w * raw + (1.0 - w) * v;
            }
        }
        v
    }

    /// Apply the bank to all three axes of a `Vector3`.
    #[inline]
    pub fn apply_xyz(&mut self, v: Vector3<f32>) -> Vector3<f32> {
        Vector3::new(self.apply(0, v.x), self.apply(1, v.y), self.apply(2, v.z))
    }

    /// Reset every biquad's delay line to zero and zero the weights and
    /// round-robin cursor. Call on disarm to avoid stale state on re-arm.
    /// Per the safety protocol: this never disarms or skips publishing on
    /// its own — the caller decides the policy.
    pub fn reset(&mut self) {
        for axis in 0..N_AXES {
            for motor in 0..NU {
                for harm in 0..NH {
                    let _ = self.notches[axis][motor][harm].reset(0.0);
                }
            }
        }
        self.weights = [[0.0; NH]; NU];
        self.motor_idx = 0;
        self.harm_idx = 0;
    }

    #[cfg(test)]
    pub(crate) fn weight(&self, motor: usize, harm: usize) -> f32 {
        self.weights[motor][harm]
    }

    #[cfg(test)]
    pub(crate) fn notches_per_call(&self) -> usize {
        self.notches_per_call
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_HZ: f32 = 8000.0;
    const Q: f32 = 5.0;
    const MIN_HZ: f32 = 100.0;
    const FADE_HZ: f32 = 50.0;

    /// Drive a sinusoid at `freq_hz` through the bank, return the RMS of the
    /// last `n_tail` samples to ignore filter startup transient.
    fn rms_response(
        bank: &mut RpmNotchBank<4, 3>,
        sine_hz: f32,
        motor_freqs: &[f32; 4],
        n_total: usize,
        n_tail: usize,
    ) -> f32 {
        // Settle motor frequencies through the full round-robin cycle first
        // (the cursor needs ~1 ms worth of calls to reach every notch).
        for _ in 0..(NU_TIMES_NH * 4) {
            bank.update(motor_freqs);
        }
        let mut sum_sq = 0.0;
        let mut count = 0;
        let dt = 1.0 / SAMPLE_HZ;
        for n in 0..n_total {
            let t = n as f32 * dt;
            let x = libm::sinf(2.0 * core::f32::consts::PI * sine_hz * t);
            bank.update(motor_freqs);
            let y = bank.apply(0, x);
            if n >= n_total - n_tail {
                sum_sq += y * y;
                count += 1;
            }
        }
        libm::sqrtf(sum_sq / count as f32)
    }

    const NU_TIMES_NH: usize = 4 * 3;

    #[test]
    fn passthrough_when_motors_idle() {
        // All motors so slow that even the top harmonic stays below min_hz
        // → every weight stays 0 → bank is a passthrough.
        // With NH=3 and min_hz=100, the 3rd harmonic of 30 Hz is 90 Hz <
        // min_hz. Pick motor freq < min_hz / NH to be safe.
        let mut bank = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        let idle = [30.0; 4];
        // Pump enough updates to refresh the entire bank.
        for _ in 0..(NU_TIMES_NH * 4) {
            bank.update(&idle);
        }
        for m in 0..4 {
            for h in 0..3 {
                assert_eq!(
                    bank.weight(m, h),
                    0.0,
                    "weight should be 0 at idle (motor {m} harm {h})"
                );
            }
        }
        // Output should equal input within float precision.
        let mut sum_diff_sq = 0.0;
        for n in 0..2000 {
            let x = libm::sinf(2.0 * core::f32::consts::PI * 200.0 * n as f32 / SAMPLE_HZ);
            bank.update(&idle);
            let y = bank.apply(1, x);
            sum_diff_sq += (x - y) * (x - y);
        }
        let rms_diff = libm::sqrtf(sum_diff_sq / 2000.0);
        assert!(rms_diff < 1e-6, "expected near-perfect passthrough, got rms_diff={rms_diff}");
    }

    #[test]
    fn full_notch_at_motor_frequency() {
        // Drive a 200 Hz sinusoid; set motor freq = 200 Hz so the
        // fundamental (h=1) lands on it. With Q=5 and full weight, the
        // attenuation should be substantial.
        let mut bank = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        let on_target = [200.0; 4];
        let on_response = rms_response(&mut bank, 200.0, &on_target, 4000, 2000);
        // Reset and re-measure with motors far away so no notch lands on the sine.
        let mut bank2 = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        let off_target = [800.0; 4];
        let off_response = rms_response(&mut bank2, 200.0, &off_target, 4000, 2000);
        assert!(
            on_response < 0.3 * off_response,
            "expected ≥3× attenuation on-target; got on={on_response} off={off_response}"
        );
    }

    /// Upper edge of the crossfade, at the 1 kHz (`imu_1khz`) sample rate
    /// where it actually binds: `max_hz` = 0.48·1000 = 480 Hz. A harmonic
    /// that runs past the band must fade to passthrough, not sit at full
    /// weight on a notch pinned to 480 Hz.
    #[test]
    fn weights_fade_to_zero_above_max_hz() {
        const FS: f32 = 1000.0;
        const MAX_HZ: f32 = 0.48 * FS; // 480
        let cases: &[(f32, f32)] = &[
            (300.0, 1.0),             // mid-band, full notch
            (MAX_HZ - FADE_HZ, 1.0),  // 430: bottom of the upper fade band
            (MAX_HZ - FADE_HZ / 2.0, 0.5), // 455: halfway back down
            (MAX_HZ, 0.0),            // 480: passthrough
            (666.0, 0.0),             // 40k-RPM fundamental — off the end
            (5000.0, 0.0),            // absurd, still passthrough
        ];
        for (f, expected_weight) in cases {
            let mut bank = RpmNotchBank::<4, 1>::new(FS, Q, MIN_HZ, FADE_HZ);
            let freqs = [*f; 4];
            for _ in 0..(4 * 4) {
                bank.update(&freqs);
            }
            for motor in 0..4 {
                let w = bank.weight(motor, 0);
                assert!(
                    libm::fabsf(w - expected_weight) < 1e-4,
                    "freq={f} motor={motor}: expected weight {expected_weight}, got {w}"
                );
            }
        }
    }

    #[test]
    fn weights_fade_smoothly_through_min_hz() {
        // As motor frequency rises through min_hz..(min_hz+fade_range), the
        // weight on the fundamental harmonic should ramp linearly from 0 to 1.
        let cases: &[(f32, f32)] = &[
            (90.0, 0.0),                       // below min_hz
            (100.0, 0.0),                      // at min_hz
            (125.0, 0.5),                      // halfway up fade band
            (150.0, 1.0),                      // top of fade band
            (300.0, 1.0),                      // well above
            (f32::NAN, 0.0),                   // non-finite → passthrough
            (f32::INFINITY, 0.0),              // non-finite → passthrough
            (-50.0, 0.0),                      // negative → clamped to min_hz
        ];
        for (f, expected_weight) in cases {
            // Reset the cursor by constructing a fresh bank — we only need
            // to verify the weight after one full round-robin.
            let mut bank_local = RpmNotchBank::<4, 1>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
            let freqs = [*f; 4];
            // Pump through enough updates to refresh every notch at least once.
            for _ in 0..(4 * 4) {
                bank_local.update(&freqs);
            }
            for motor in 0..4 {
                let w = bank_local.weight(motor, 0);
                assert!(
                    libm::fabsf(w - expected_weight) < 1e-4,
                    "freq={f} motor={motor}: expected weight {expected_weight}, got {w}"
                );
            }
        }
    }

    #[test]
    fn coefficient_sweep_no_transient() {
        // DF1 invariant: changing biquad coefficients between samples must not
        // produce output spikes. Feed a constant input through the bank while
        // sweeping motor frequency from min_hz upward — the output should
        // remain bounded and non-explosive even as coefficients change every
        // call.
        let mut bank = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        let constant_input = 1.0;
        let mut max_abs = 0.0f32;
        // Sweep 100..2000 Hz over 8000 samples (= 1 s of audio).
        for n in 0..8000 {
            let f = 100.0 + (1900.0 * n as f32 / 8000.0);
            let freqs = [f; 4];
            bank.update(&freqs);
            // Skip the first ~50 samples to ignore the natural step-response
            // transient (DC step entering a notch with non-trivial passband).
            let y = bank.apply(0, constant_input);
            if n > 50 {
                max_abs = max_abs.max(libm::fabsf(y));
            }
        }
        // The notch is unity-gain at DC, so the steady-state response to a
        // constant input is the constant. Allow some headroom for the
        // transient the sweep introduces, but anything blowing up would
        // exceed even 10× input — flag that as instability.
        assert!(
            max_abs < 5.0,
            "coefficient sweep produced runaway output: max_abs={max_abs}"
        );
        // And specifically: no NaN/Inf.
        assert!(max_abs.is_finite(), "non-finite output during sweep");
    }

    #[test]
    fn nan_motor_freq_does_not_propagate_to_signal() {
        // Safety contract per docs/safety_protocol.md: bad input must not
        // poison the output. A NaN motor frequency must NOT produce a NaN
        // gyro/accel sample — the bank treats it as "motor idle" and falls
        // back to passthrough.
        let mut bank = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        let bad = [f32::NAN, 200.0, 300.0, f32::INFINITY];
        for n in 0..1000 {
            let x = libm::sinf(2.0 * core::f32::consts::PI * 200.0 * n as f32 / SAMPLE_HZ);
            bank.update(&bad);
            for axis in 0..3 {
                let y = bank.apply(axis, x);
                assert!(y.is_finite(), "non-finite output at sample {n} axis {axis}: {y}");
            }
        }
    }

    #[test]
    fn reset_clears_state_and_weights() {
        let mut bank = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        let active = [400.0; 4];
        // Run for a while to build up delay-line state and non-zero weights.
        for n in 0..4000 {
            let x = libm::sinf(2.0 * core::f32::consts::PI * 400.0 * n as f32 / SAMPLE_HZ);
            bank.update(&active);
            let _ = bank.apply(0, x);
        }
        // Sanity check: weights actually built up (not a no-op test).
        let any_active = (0..4).any(|m| (0..3).any(|h| bank.weight(m, h) > 0.5));
        assert!(any_active, "weights should be non-zero before reset");
        bank.reset();
        for m in 0..4 {
            for h in 0..3 {
                assert_eq!(bank.weight(m, h), 0.0, "weight should be 0 after reset");
            }
        }
        // After reset, feeding zeros should yield zeros (delay lines cleared).
        for _ in 0..100 {
            for axis in 0..3 {
                assert_eq!(bank.apply(axis, 0.0), 0.0);
            }
        }
    }

    #[test]
    fn round_robin_visits_every_notch_within_window() {
        // The cursor must visit every (motor, harmonic) within the refresh
        // window. At 8 kHz with NU=4, NH=3 → ceil(12 / 8) = 2 per call,
        // so the bank should fully refresh in ceil(12/2) = 6 calls.
        let bank = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        assert_eq!(bank.notches_per_call(), 2);

        // For NH=1 (accel): ceil(4 / 8) = 1 per call, 4 calls to refresh.
        let bank_accel = RpmNotchBank::<4, 1>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        assert_eq!(bank_accel.notches_per_call(), 1);
    }

    #[test]
    fn batched_refresh_reaches_all_motors() {
        // After running update() enough times for one full window, every
        // motor's weight should reflect the latest motor frequency.
        let mut bank = RpmNotchBank::<4, 3>::new(SAMPLE_HZ, Q, MIN_HZ, FADE_HZ);
        let freqs = [500.0, 600.0, 700.0, 800.0];
        // Conservative upper bound: ceil(NU*NH / notches_per_call) = 6 calls.
        for _ in 0..16 {
            bank.update(&freqs);
        }
        for m in 0..4 {
            for h in 0..3 {
                assert_eq!(
                    bank.weight(m, h),
                    1.0,
                    "motor {m} harm {h} weight should be 1.0 (well above min_hz)"
                );
            }
        }
    }
}
