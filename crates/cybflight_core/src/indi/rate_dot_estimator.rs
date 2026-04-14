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
//! An SG derivative filter is typically designed for a specific (modest)
//! sample rate. Feeding it 8 kHz gyro samples directly would make the
//! SG window span too little time for the polynomial fit to be meaningful.
//! This estimator therefore decimates the input to a target SG rate
//! (~1 kHz by default) by taking every `decim`-th sample, where
//! `decim = round(loop_rate_hz / target_rate_hz)`. The SG filter and the
//! post-biquad both run at the decimated rate; on intermediate loop-rate
//! calls the estimator returns the last held value. The post-biquad's low
//! cutoff strongly attenuates any aliased content from the naive
//! stride-decimation.

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
    /// Target SG sample rate. The actual decimated rate is
    /// `loop_rate_hz / round(loop_rate_hz / target_rate_hz)` (≥ 1).
    pub sg_target_rate_hz: f32,
    /// Post-biquad low-pass cutoff (Hz), evaluated at the decimated rate.
    pub post_cutoff_hz: f32,
}

/// Streaming body-acceleration estimator = SG first-derivative (decimated)
/// → single biquad low-pass.
pub struct RateDotEstimator {
    sg: [Sg; 3],
    post: [Biquad; 3],
    decim: u32,
    decim_count: u32,
    raw: Vector3<f32>,
    filtered: Vector3<f32>,
}

impl RateDotEstimator {
    pub fn new(loop_rate_hz: f32, cfg: &RateDotEstimatorConfig) -> Self {
        let decim = (libm::roundf(loop_rate_hz / cfg.sg_target_rate_hz) as u32).max(1);
        let sg_rate_hz = loop_rate_hz / decim as f32;

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
                    .cutoff_frequency_hz(cfg.post_cutoff_hz)
                    .build()
                    .expect("indi: rate_dot post biquad config invalid"),
            )
        };

        Self {
            sg: [make_sg(), make_sg(), make_sg()],
            post: [make_post(), make_post(), make_post()],
            decim,
            // Start at `decim` so the very first `update` call immediately
            // advances the SG/biquad instead of returning zero for the first
            // `decim - 1` calls.
            decim_count: decim,
            raw: Vector3::zeros(),
            filtered: Vector3::zeros(),
        }
    }

    /// Feed one gyro sample at loop rate. Advances the SG + post-biquad only
    /// on decimation boundaries; between boundaries the held outputs are
    /// reused.
    pub fn update(&mut self, gyro_rad_s: &Vector3<f32>) {
        self.decim_count += 1;
        if self.decim_count < self.decim {
            return;
        }
        self.decim_count = 0;

        let sg0 = self.sg[0].apply(gyro_rad_s[0]);
        let sg1 = self.sg[1].apply(gyro_rad_s[1]);
        let sg2 = self.sg[2].apply(gyro_rad_s[2]);
        self.raw = Vector3::new(sg0, sg1, sg2);

        let f0 = self.post[0].apply(sg0);
        let f1 = self.post[1].apply(sg1);
        let f2 = self.post[2].apply(sg2);
        self.filtered = Vector3::new(f0, f1, f2);
    }

    /// SG first-derivative output *before* the post-biquad (held between
    /// decimation boundaries). Used as the "raw" angular-acceleration signal
    /// consumed by the online learner, which applies its own matched filter.
    pub fn raw(&self) -> Vector3<f32> {
        self.raw
    }

    /// Final post-biquad output (held between decimation boundaries) — the
    /// `rate_dot_fs` signal consumed by the INDI control law.
    pub fn filtered(&self) -> Vector3<f32> {
        self.filtered
    }

    /// Effective decimation factor actually applied (for diagnostics/tests).
    pub fn decim(&self) -> u32 {
        self.decim
    }
}
