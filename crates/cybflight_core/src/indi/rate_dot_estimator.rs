//! Body angular-acceleration estimator: decimated Savitzky–Golay first
//! derivative followed by a single biquad low-pass.
//!
//! The INDI controller historically computed `rate_dot` by backward-differencing
//! the gyro at loop rate and smoothing the result with a biquad. A backward
//! difference is the noisiest possible derivative and puts all the burden on
//! the post-biquad. A Savitzky–Golay derivative is a least-squares
//! polynomial-fit derivative over a small window, which rejects sample noise
//! at the source.
//!
//! Both stages run at the full loop rate. An earlier version decimated the
//! gyro stream to a fixed SG rate first, on the theory that an SG window is
//! designed for a specific modest sample rate; in practice every firmware
//! and sim configuration set the target equal to the loop rate, so the
//! decimation was inert everywhere while still costing a held-value
//! (zero-order-hold) code path that would have injected staircase delay
//! into the `rate_dot` signal had anyone ever enabled it. The window is
//! instead sized per build from a group-delay budget — see
//! `cybflight::rates::RATE_DOT_SG_WINDOW` — which is what actually keeps
//! the SG span sensible across IMU rates.

use air_filters::fir::savitzky_golay::{SavitzkyGolayFilter, SgConfigBuilder};
use air_filters::iir::biquad::{
    BiquadFilter, BiquadFilterConfigBuilder, BiquadFilterType, DirectForm2,
};
use air_filters::Filter;
use nalgebra::Vector3;

type Biquad = BiquadFilter<f32, DirectForm2<f32>>;
type Sg = SavitzkyGolayFilter<f32>;

/// Configuration for [`RateDotEstimator`].
#[derive(Clone, Copy)]
pub struct RateDotEstimatorConfig {
    /// SG window size (odd, ≥ 3, ≤ 19).
    pub sg_window_size: i32,
    /// SG polynomial order (1 ≤ order ≤ 3, ≤ window_size - 1).
    pub sg_order: i32,
    /// Post-biquad low-pass cutoff (Hz). Clamped against the loop rate by
    /// [`super::clamp_cutoff_hz`] so a runtime parameter can never panic
    /// the filter builder on a low-rate build.
    pub post_cutoff_hz: f32,
}

/// Streaming body-acceleration estimator = SG first-derivative → single
/// biquad low-pass, both at the loop rate.
pub struct RateDotEstimator {
    sg: [Sg; 3],
    post: [Biquad; 3],
    raw: Vector3<f32>,
    filtered: Vector3<f32>,
}

impl RateDotEstimator {
    pub fn new(loop_rate_hz: f32, cfg: &RateDotEstimatorConfig) -> Self {
        let sg_rate_hz = loop_rate_hz;
        let post_cutoff_hz = super::clamp_cutoff_hz(cfg.post_cutoff_hz, sg_rate_hz);

        let make_sg = || {
            SavitzkyGolayFilter::new(
                SgConfigBuilder::new()
                    .window_size(cfg.sg_window_size)
                    .order(cfg.sg_order)
                    .deriv_order(1)
                    .sample_frequency_hz(sg_rate_hz)
                    .build()
                    .expect("indi: SG filter config invalid"),
            )
        };
        let make_post = || {
            BiquadFilter::new(
                BiquadFilterConfigBuilder::direct_form_2()
                    .sample_frequency_hz(sg_rate_hz)
                    .filter_type(BiquadFilterType::LowPass)
                    .cutoff_frequency_hz(post_cutoff_hz)
                    .build()
                    .expect("indi: rate_dot post biquad config invalid"),
            )
        };

        Self {
            sg: [make_sg(), make_sg(), make_sg()],
            post: [make_post(), make_post(), make_post()],
            raw: Vector3::zeros(),
            filtered: Vector3::zeros(),
        }
    }

    /// Feed one gyro sample at loop rate. Advances the SG + post-biquad.
    pub fn update(&mut self, gyro_rad_s: &Vector3<f32>) {
        let sg0 = self.sg[0].apply(gyro_rad_s[0]);
        let sg1 = self.sg[1].apply(gyro_rad_s[1]);
        let sg2 = self.sg[2].apply(gyro_rad_s[2]);
        self.raw = Vector3::new(sg0, sg1, sg2);

        let f0 = self.post[0].apply(sg0);
        let f1 = self.post[1].apply(sg1);
        let f2 = self.post[2].apply(sg2);
        self.filtered = Vector3::new(f0, f1, f2);
    }

    /// SG first-derivative output *before* the post-biquad. Exposed as the
    /// "raw" angular-acceleration signal for external observers.
    pub fn raw(&self) -> Vector3<f32> {
        self.raw
    }

    /// Final post-biquad output — the `rate_dot_fs` signal consumed by the
    /// INDI control law.
    pub fn filtered(&self) -> Vector3<f32> {
        self.filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(window: i32) -> RateDotEstimatorConfig {
        RateDotEstimatorConfig {
            sg_window_size: window,
            sg_order: 2,
            post_cutoff_hz: 15.0,
        }
    }

    /// A constant-slope gyro ramp must produce a correct derivative estimate
    /// at every IMU rate the firmware builds for, with that rate's
    /// delay-budgeted window.
    #[test]
    fn ramp_derivative_tracks_slope_at_every_imu_rate() {
        for (rate_hz, window) in [(1000.0f32, 5), (3200.0, 13), (8000.0, 13)] {
            let mut est = RateDotEstimator::new(rate_hz, &cfg(window));
            let slope_rad_s2 = 10.0;
            let dt = 1.0 / rate_hz;
            for i in 0..(rate_hz as usize / 5) {
                let w = slope_rad_s2 * dt * i as f32;
                est.update(&Vector3::new(w, 0.0, 0.0));
            }
            let raw = est.raw();
            assert!(
                (raw[0] - slope_rad_s2).abs() < 0.5,
                "{rate_hz} Hz / window {window}: SG derivative {} should track slope {slope_rad_s2}",
                raw[0],
            );
        }
    }

    /// A post cutoff at or past Nyquist must clamp, not panic the builder.
    /// `indi_sync_hz`'s schema maximum is 500 Hz, which is exactly Nyquist
    /// on an `imu_1khz` build.
    #[test]
    fn post_cutoff_at_nyquist_does_not_panic() {
        let mut est = RateDotEstimator::new(
            1000.0,
            &RateDotEstimatorConfig {
                post_cutoff_hz: 500.0,
                ..cfg(5)
            },
        );
        for _ in 0..50 {
            est.update(&Vector3::new(0.1, -0.2, 0.3));
        }
        assert!(est.filtered().iter().all(|v| v.is_finite()));
    }
}
