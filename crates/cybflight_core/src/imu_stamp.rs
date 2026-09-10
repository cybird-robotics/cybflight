//! Sample-time reconstruction for a fixed-ODR IMU read by a task that can
//! be late.
//!
//! The chip samples on its own crystal at exactly `T = 1/ODR` (±1 %); the
//! reader task observes each sample some variable time later (wake
//! latency + SPI burst, plus whatever else was running on its executor).
//! Stamping `now()` at observation therefore puts that jitter into every
//! timestamp: 125 / 225 / 25 / 125 µs spacing in a log whose true spacing
//! is 125 / 125 / 125 / 125. Latched DRDY makes this visible (a late read
//! is *kept* instead of dropped), so the stamps have to be reconstructed.
//!
//! Model: `stamp_n = stamp_{n-1} + T`, softly pulled toward the observed
//! wake time so the two clocks (IMU crystal vs MCU timebase) cannot
//! drift apart unboundedly, and re-anchored — with the gap counted as
//! **lost samples** — when the observation falls a sample or more behind
//! for two consecutive reads. Two, because a single late read (up to a
//! full period, e.g. the reader waiting out an INDI step) is
//! indistinguishable from one lost sample by time alone; what separates
//! them is the *next* read: after lateness it lands back on the grid,
//! after a loss it is still a period behind.
//!
//! All arithmetic is in plain integers of whatever unit the caller
//! feeds in, so this is testable on the host and independent of the
//! firmware's `Instant` type. The tests use microseconds
//! ([`SampleStamper::new`]); the firmware uses Q16 fixed-point timer
//! ticks ([`SampleStamper::from_period`]) so the per-sample path on the
//! control executor is shifts/adds only — no µs↔tick conversions and no
//! 64-bit software divides (Cortex-M7 has none in hardware). The only
//! division left is on the rare lost-sample path.

/// Reconstructs on-grid sample timestamps and counts lost samples.
#[derive(Clone, Debug)]
pub struct SampleStamper {
    period: i64,
    /// `period / 2`, precomputed: the late-read threshold.
    half_period: i64,
    /// `period / PLL_MAX_DIV`, precomputed: the per-sample PLL cap.
    max_adj: i64,
    prev: Option<i64>,
    behind_streak: u8,
}

/// One stamped sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamped {
    /// Reconstructed sample time, in the caller's units. Never later than
    /// the observation time it was derived from.
    pub stamp: i64,
    /// Samples judged lost immediately before this one (0 normally).
    pub lost: u32,
}

impl SampleStamper {
    /// PLL gain as a shift: the stamp moves 1/16 of the residual toward the
    /// observed time each sample. Filters ±100 µs lateness jitter to ~6 µs
    /// while tracking a 1 % clock-rate difference (1.25 µs/sample at 8 kHz)
    /// with a steady-state residual of ~32 µs — a quarter period, half the
    /// loss threshold. (1/32 settles at ~64 µs and false-trips it.)
    const PLL_SHIFT: u32 = 4;
    /// Cap on the per-sample PLL correction (`T / 8`), so a single wild
    /// observation cannot bend the grid.
    const PLL_MAX_DIV: i64 = 8;

    /// Microsecond units: `period = 1e6 / odr_hz`.
    pub fn new(odr_hz: f32) -> Self {
        let period_us = if odr_hz.is_finite() && odr_hz > 0.0 {
            (1.0e6 / odr_hz) as i64
        } else {
            1
        };
        Self::from_period(period_us)
    }

    /// Arbitrary units: the nominal sample period in whatever unit
    /// `stamp` will be fed. Use a fixed-point unit fine enough that the
    /// period's rounding error is well under the 1 % the PLL tracks.
    pub fn from_period(period: i64) -> Self {
        let period = period.max(1);
        Self {
            period,
            half_period: period / 2,
            max_adj: period / Self::PLL_MAX_DIV,
            prev: None,
            behind_streak: 0,
        }
    }

    /// Forget the grid; the next observation re-anchors it (use after an
    /// IMU recovery).
    pub fn reset(&mut self) {
        self.prev = None;
        self.behind_streak = 0;
    }

    /// Feed the observation time of one sample (taken as early as the
    /// reader can — right after data-ready, before or after the burst,
    /// but consistently).
    pub fn stamp(&mut self, observed: i64) -> Stamped {
        let t = self.period;
        let Some(prev) = self.prev else {
            self.prev = Some(observed);
            return Stamped { stamp: observed, lost: 0 };
        };
        let recon = prev + t;
        // Positive: the observation is later than the grid predicts.
        let err = observed - recon;

        if err > self.half_period {
            self.behind_streak = self.behind_streak.saturating_add(1);
        } else {
            self.behind_streak = 0;
        }

        let stamped = if self.behind_streak >= 2 {
            // Still ≥ half a period behind on the second consecutive read:
            // real samples were lost, not just a late wake. Re-anchor on
            // the observation and count the missing grid points.
            let lost = ((err + self.half_period) / t).max(0) as u32;
            self.behind_streak = 0;
            Stamped { stamp: observed, lost }
        } else {
            // Arithmetic shift: floor toward -inf, fine for a gain.
            let adj = (err >> Self::PLL_SHIFT).clamp(-self.max_adj, self.max_adj);
            // A sample cannot have been taken after it was observed.
            let stamp = (recon + adj).min(observed);
            Stamped { stamp, lost: 0 }
        };
        self.prev = Some(stamped.stamp);
        stamped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: i64 = 125;

    fn run(seq: &[i64]) -> Vec<Stamped> {
        let mut s = SampleStamper::new(8000.0);
        seq.iter().map(|&o| s.stamp(o)).collect()
    }

    #[test]
    fn on_time_reads_land_on_the_grid() {
        let out = run(&[1000, 1125, 1250, 1375, 1500]);
        let d: Vec<i64> = out.windows(2).map(|w| w[1].stamp - w[0].stamp).collect();
        assert_eq!(d, vec![T; 4]);
        assert!(out.iter().all(|s| s.lost == 0));
    }

    #[test]
    fn a_single_late_read_keeps_grid_spacing_and_is_not_a_loss() {
        // Third sample observed 100 µs late (reader waited out an INDI
        // step), fourth back on time.
        let out = run(&[1000, 1125, 1350, 1375, 1500]);
        let d: Vec<i64> = out.windows(2).map(|w| w[1].stamp - w[0].stamp).collect();
        for x in &d {
            assert!((x - T).abs() <= T / 8, "spacing {x} strayed from the grid");
        }
        assert!(out.iter().all(|s| s.lost == 0));
    }

    #[test]
    fn one_lost_sample_is_counted_and_reanchors() {
        // Sample at 1250 never observed; the reader sees 1375 and 1500.
        let out = run(&[1000, 1125, 1375, 1500, 1625]);
        let lost: u32 = out.iter().map(|s| s.lost).sum();
        assert_eq!(lost, 1);
        // After re-anchoring the grid is back to true spacing.
        let tail = &out[3..];
        assert_eq!(tail[1].stamp - tail[0].stamp, T);
    }

    #[test]
    fn a_long_gap_counts_all_missing_samples() {
        // 10 samples missing (a ~1.4 ms stall).
        let obs: Vec<i64> = [1000, 1125].into_iter().chain((0..3).map(|i| 1125 + 11 * T + i * T)).collect();
        let out = run(&obs);
        let lost: u32 = out.iter().map(|s| s.lost).sum();
        assert_eq!(lost, 10);
    }

    #[test]
    fn clock_rate_mismatch_is_tracked_without_false_losses() {
        // IMU crystal 1 % fast: samples every 123.75 µs; observe with a
        // constant 20 µs read offset. Over 2000 samples the stamp must
        // stay within half a period of the observation and never report
        // a loss.
        let mut s = SampleStamper::new(8000.0);
        let mut lost = 0;
        for n in 0..2000i64 {
            let true_t = (n as f64 * 123.75) as i64;
            let st = s.stamp(true_t + 20);
            lost += st.lost;
            let resid = (true_t + 20) - st.stamp;
            assert!(resid.abs() < T / 2, "n={n} resid={resid}");
        }
        assert_eq!(lost, 0);
        // And 1 % slow.
        let mut s = SampleStamper::new(8000.0);
        for n in 0..2000i64 {
            let true_t = (n as f64 * 126.25) as i64;
            let st = s.stamp(true_t + 20);
            lost += st.lost;
            let resid = (true_t + 20) - st.stamp;
            assert!(resid.abs() < T / 2, "slow n={n} resid={resid}");
        }
        assert_eq!(lost, 0);
    }

    #[test]
    fn stamp_never_exceeds_observation() {
        let mut s = SampleStamper::new(8000.0);
        // Observations arriving early relative to the grid (fast crystal)
        // must clamp, not run ahead.
        for n in 0..200i64 {
            let o = n * 120;
            let st = s.stamp(o);
            assert!(st.stamp <= o);
        }
    }

    #[test]
    fn q16_tick_units_match_microsecond_units() {
        // Firmware feeds Q16 timer ticks (32 768 Hz). The same physical
        // observation sequence must yield the same lost count and grid
        // spacing (within one tick) as the µs run.
        const TICK_HZ: f64 = 32_768.0;
        let to_q16 = |us: i64| ((us as f64 * TICK_HZ / 1e6) * 65536.0) as i64;
        let period_q16 = ((TICK_HZ * 65536.0) / 8000.0) as i64;
        let mut s = SampleStamper::from_period(period_q16);
        let obs = [1000, 1125, 1375, 1500, 1625, 1750];
        let out: Vec<Stamped> = obs.iter().map(|&o| s.stamp(to_q16(o))).collect();
        assert_eq!(out.iter().map(|s| s.lost).sum::<u32>(), 1);
        let d = out[5].stamp - out[4].stamp;
        assert!((d - period_q16).abs() <= 65536, "spacing {d} vs {period_q16}");
        assert!(out.iter().zip(obs).all(|(s, o)| s.stamp <= to_q16(o)));
    }

    #[test]
    fn reset_reanchors_silently() {
        let mut s = SampleStamper::new(1000.0);
        s.stamp(0);
        s.stamp(1000);
        s.reset();
        let st = s.stamp(50_000);
        assert_eq!(st, Stamped { stamp: 50_000, lost: 0 });
    }
}
