use core::clone::Clone;
use num_traits::float::Float;
use num_traits::FloatConst;

#[derive(Debug)]
pub enum ButterworthError {
    InvalidSamplingFrequency,
    InvalidCutoffFrequency,
    NyquistCriterionViolation,
    SizeMismatch,
}

/// Butterworth filter
pub struct ButterworthFilter<T: Float, const DIM: usize> {
    den: [T; 2],
    num: [T; 2],
    input: [[T; DIM]; 2],
    output: [[T; DIM]; 2],
    cutoff_frequency: T,
    sampling_frequency: T,
    curr_idx: usize,
}

impl<T: Float + FloatConst, const DIM: usize> ButterworthFilter<T, DIM> {
    /// Initialize the Butterworth filter context.
    ///
    /// This function initializes the Butterworth filter context with the specified
    /// cutoff frequency and sampling frequency.
    ///
    /// # Parameters
    /// - `cutoff_frequency` Cutoff frequency in Hz.
    /// - `sampling_frequency` Sampling frequency in Hz.
    ///
    /// # Returns
    /// - `Ok(ButterworthFilter)` if the filter is initialized successfully.
    /// - `Err(ButterworthError)` if the cutoff frequency is non-positive, non-finite, or if the
    /// sampling frequency is non-positive, non-finite, or not more than twice the cutoff
    /// frequency.
    pub fn new(
        cutoff_frequency: T,
        sampling_frequency: T,
        initial_input: Option<[T; DIM]>,
        initial_output: Option<[T; DIM]>,
    ) -> Self {
        debug_assert!(
            cutoff_frequency.is_finite() && cutoff_frequency > T::zero(),
            "cutoff frequency must be positive and finite"
        );
        debug_assert!(
            sampling_frequency.is_finite() && sampling_frequency > T::zero(),
            "sampling frequency must be positive and finite"
        );
        debug_assert!(
             sampling_frequency > T::from(2.0).unwrap() * cutoff_frequency,
            "sampling frequency must be more than twice the cutoff frequency to satisfy Nyquist criterion"
        );

        let mut filter = ButterworthFilter {
            den: [T::zero(); 2],
            num: [T::zero(); 2],
            input: [initial_input.unwrap_or_else(|| [T::zero(); DIM]); 2],
            output: [initial_output.unwrap_or_else(|| [T::zero(); DIM]); 2],
            cutoff_frequency,
            sampling_frequency,
            curr_idx: 0,
        };

        filter.reset_numden();
        filter
    }

    /// Set the cutoff frequency of the Butterworth filter.
    ///
    /// This function sets the cutoff frequency for the Butterworth filter and
    /// recalculates the filter coefficients.
    ///
    /// # Parameters
    /// - `cutoff_frequency` New cutoff frequency in Hz.
    ///
    /// # Returns
    /// - `Err(ButterworthError)` if the cutoff frequency is non-positive, non-finite, or if the
    /// sampling frequency is not more than twice the new cutoff frequency.
    pub fn set_cutoff_frequency(&mut self, cutoff_frequency: T) -> Result<(), ButterworthError> {
        if cutoff_frequency <= T::zero() || !cutoff_frequency.is_finite() {
            return Err(ButterworthError::InvalidCutoffFrequency);
        }

        if self.sampling_frequency <= T::from(2.0).unwrap() * cutoff_frequency {
            return Err(ButterworthError::NyquistCriterionViolation);
        }

        self.cutoff_frequency = cutoff_frequency;
        self.reset_numden();
        Ok(())
    }

    /// Set the sampling frequency of the Butterworth filter.
    ///
    /// This function sets the sampling frequency for the Butterworth filter and
    /// recalculates the filter coefficients.
    ///
    /// # Parameters
    /// - `sampling_frequency` New sampling frequency in Hz.
    ///
    /// # Returns
    /// - `Err(ButterworthError)` if the sampling frequency is non-positive, non-finite, or not
    /// more than twice the cutoff frequency.
    pub fn set_sampling_frequency(
        &mut self,
        sampling_frequency: T,
    ) -> Result<(), ButterworthError> {
        if sampling_frequency <= T::zero() || !sampling_frequency.is_finite() {
            return Err(ButterworthError::InvalidSamplingFrequency);
        }

        if sampling_frequency <= T::from(2.0).unwrap() * self.cutoff_frequency {
            return Err(ButterworthError::NyquistCriterionViolation);
        }

        self.sampling_frequency = sampling_frequency;
        self.reset_numden();
        Ok(())
    }

    /// compute the butterworth filter output for a given sample.
    ///
    /// this function computes the output of the butterworth filter for a given
    /// input sample and updates the internal state of the filter.
    ///
    /// # Parameters
    /// - `sample` input sample array
    ///
    /// # Returns
    /// - `[T; DIM]` containing the filtered output if the computation is successful.
    pub fn compute(&mut self, sample: &[T; DIM]) -> [T; DIM] {
        let y_nm2 = self.prev_output(1).clone();
        let y_nm1 = self.output[self.curr_idx];

        let x_nm2 = self.prev_input(1).clone();
        self.curr_idx = (self.curr_idx + 1) % 2;

        let x_nm1 = self.prev_input(1);

        let mut result = [T::zero(); DIM];
        for i in 0..DIM {
            result[i] = sample[i] * self.num[0] + x_nm1[i] * self.num[1] + x_nm2[i] * self.num[0]
                - y_nm1[i] * self.den[0]
                - y_nm2[i] * self.den[1];
        }
        self.input[self.curr_idx] = (*sample).into();
        self.output[self.curr_idx] = result;
        result
    }

    /// Reset the input and output buffers of the Butterworth filter.
    ///
    /// This function resets the input and output buffers of the Butterworth filter to the
    /// specified initial values or to zero if no initial values are provided.
    ///
    /// # Parameters
    /// - `initial_input` Optional initial input array to reset the input buffer. If `None`, the
    /// input buffer will be reset to zero.
    /// - `initial_output` Optional initial output array to reset the output buffer. If `None`, the
    /// output buffer will be reset to zero.
    pub fn reset_input_output(
        &mut self,
        initial_input: Option<[T; DIM]>,
        initial_output: Option<[T; DIM]>,
    ) {
        self.curr_idx = 0;
        for inp in &mut self.input {
            *inp = initial_input.unwrap_or_else(|| [T::zero(); DIM]);
        }
        for outp in &mut self.output {
            *outp = initial_output.unwrap_or_else(|| [T::zero(); DIM]);
        }
        self.curr_idx = 0;
    }

    fn reset_numden(&mut self) {
        let k = (T::PI() * self.cutoff_frequency / self.sampling_frequency).tan();
        let k2 = k * k;
        let poly = k2 + T::SQRT_2() * k + T::one();

        let b0 = k2 / poly;
        let a1 = T::from(2.0).unwrap() * (k2 - T::one()) / poly;
        let a2 = (k2 - T::SQRT_2() * k + T::one()) / poly;

        self.num[0] = b0;
        self.num[1] = T::from(2.0).unwrap() * b0;
        self.den[0] = a1;
        self.den[1] = a2;
    }

    fn prev_input(&self, idx: usize) -> &[T; DIM] {
        &self.input[(self.curr_idx + 2 - idx) % 2]
    }

    fn prev_output(&self, idx: usize) -> &[T; DIM] {
        &self.output[(self.curr_idx + 2 - idx) % 2]
    }
}

#[cfg(test)]
mod tests {

    use core::f64::consts::PI;
    use float_cmp::assert_approx_eq;

    use crate::butterworth::{ButterworthError, ButterworthFilter};

    #[test]
    fn butterworth_impulse_response() {
        let mut bw = ButterworthFilter::new(40.0, 100.0, None, None);

        const EXPECTED: [f32; 10] = [
            0.6389455251590224,
            0.5475887728761645,
            -0.2506954995302367,
            0.06049454749474992,
            0.03434341454513083,
            -0.06422609909766833,
            0.05923216261452053,
            -0.041188570644668146,
            0.02262660178837562,
            -0.008859056897431019,
        ];

        const INPUT: [f32; 10] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for (x, expected) in INPUT.into_iter().zip(EXPECTED) {
            let result = bw.compute(&[x])[0];
            assert_approx_eq!(f32, result, expected);
        }
    }

    #[test]
    fn butterworth_dc_gain() {
        // A Butterworth LPF must pass DC with unity gain
        let mut bw = ButterworthFilter::<f64, 1>::new(40.0, 1000.0, None, None);
        let mut output = 0.0_f64;
        for _ in 0..2000 {
            output = bw.compute(&[1.0])[0];
        }
        assert!(
            (output - 1.0).abs() < 1e-9,
            "DC gain should be 1.0, got {output}"
        );
    }

    #[test]
    fn butterworth_minus3db_at_cutoff() {
        // By definition of the cutoff frequency, the gain must be exactly 1/sqrt(2)
        let fc = 40.0_f64;
        let fs = 1000.0_f64;
        let mut bw = ButterworthFilter::<f64, 1>::new(fc, fs, None, None);

        let mut x_rms = 0.0_f64;
        let mut y_rms = 0.0_f64;
        for i in 0..5000_usize {
            let x = (2.0 * PI * fc / fs * i as f64).sin();
            let y = bw.compute(&[x])[0];
            if i >= 1000 {
                x_rms += x * x;
                y_rms += y * y;
            }
        }

        let gain = (y_rms / x_rms).sqrt();
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!(
            (gain - expected).abs() < 0.001,
            "gain at cutoff should be {expected:.4}, got {gain:.4}"
        );
    }

    #[test]
    fn butterworth_passband_gain() {
        // At fc/10, the passband should be flat to within 1%
        let fc = 100.0_f64;
        let fs = 2000.0_f64;
        let f_test = fc / 10.0;
        let mut bw = ButterworthFilter::<f64, 1>::new(fc, fs, None, None);

        let mut x_rms = 0.0_f64;
        let mut y_rms = 0.0_f64;
        for i in 0..5000_usize {
            let x = (2.0 * PI * f_test / fs * i as f64).sin();
            let y = bw.compute(&[x])[0];
            if i >= 1000 {
                x_rms += x * x;
                y_rms += y * y;
            }
        }

        let gain = (y_rms / x_rms).sqrt();
        assert!(gain > 0.99, "passband gain should be > 0.99, got {gain:.4}");
    }

    #[test]
    fn butterworth_stopband_attenuation() {
        // At 10*fc, 2nd-order 40dB/decade rolloff gives gain ~ 1/sqrt(1 + 10^4) ≈ 0.01
        let fc = 40.0_f64;
        let fs = 1000.0_f64;
        let f_test = 400.0_f64;
        let mut bw = ButterworthFilter::<f64, 1>::new(fc, fs, None, None);

        let mut x_rms = 0.0_f64;
        let mut y_rms = 0.0_f64;
        for i in 0..5000_usize {
            let x = (2.0 * PI * f_test / fs * i as f64).sin();
            let y = bw.compute(&[x])[0];
            if i >= 1000 {
                x_rms += x * x;
                y_rms += y * y;
            }
        }

        let gain = (y_rms / x_rms).sqrt();
        assert!(gain < 0.01, "stopband gain should be < 0.01, got {gain:.6}");
    }

    #[test]
    fn butterworth_multidim_independence() {
        // Each dimension of a multi-dim filter must behave as an independent 1D filter
        let fc = 40.0_f32;
        let fs = 100.0_f32;
        let mut bw3 = ButterworthFilter::<f32, 3>::new(fc, fs, None, None);
        let mut bw1 = [
            ButterworthFilter::<f32, 1>::new(fc, fs, None, None),
            ButterworthFilter::<f32, 1>::new(fc, fs, None, None),
            ButterworthFilter::<f32, 1>::new(fc, fs, None, None),
        ];

        // Stagger impulses across dimensions so each channel sees a distinct signal
        let signals: [[f32; 10]; 3] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        for i in 0..10 {
            let sample = [signals[0][i], signals[1][i], signals[2][i]];
            let out3 = bw3.compute(&sample);
            for (dim, bw) in bw1.iter_mut().enumerate() {
                let out1 = bw.compute(&[signals[dim][i]])[0];
                assert_approx_eq!(f32, out3[dim], out1);
            }
        }
    }

    #[test]
    fn butterworth_reset_input_output() {
        let mut bw = ButterworthFilter::<f32, 1>::new(40.0, 100.0, None, None);

        for _ in 0..10 {
            bw.compute(&[1.0_f32]);
        }
        bw.reset_input_output(None, None);

        let mut bw_fresh = ButterworthFilter::<f32, 1>::new(40.0, 100.0, None, None);
        for i in 0..10 {
            let x = if i == 0 { 1.0_f32 } else { 0.0 };
            let r_reset = bw.compute(&[x])[0];
            let r_fresh = bw_fresh.compute(&[x])[0];
            assert_approx_eq!(f32, r_reset, r_fresh);
        }
    }

    #[test]
    fn butterworth_set_cutoff_frequency() {
        let mut bw = ButterworthFilter::<f64, 1>::new(40.0, 1000.0, None, None);

        assert!(bw.set_cutoff_frequency(80.0).is_ok());
        assert!(matches!(
            bw.set_cutoff_frequency(0.0),
            Err(ButterworthError::InvalidCutoffFrequency)
        ));
        assert!(matches!(
            bw.set_cutoff_frequency(-1.0),
            Err(ButterworthError::InvalidCutoffFrequency)
        ));
        // 2 * 600 > 1000
        assert!(matches!(
            bw.set_cutoff_frequency(600.0),
            Err(ButterworthError::NyquistCriterionViolation)
        ));
        // fs == 2 * fc is also rejected
        assert!(matches!(
            bw.set_cutoff_frequency(500.0),
            Err(ButterworthError::NyquistCriterionViolation)
        ));
    }

    #[test]
    fn butterworth_set_sampling_frequency() {
        let mut bw = ButterworthFilter::<f64, 1>::new(40.0, 1000.0, None, None);

        assert!(bw.set_sampling_frequency(2000.0).is_ok());
        assert!(matches!(
            bw.set_sampling_frequency(0.0),
            Err(ButterworthError::InvalidSamplingFrequency)
        ));
        assert!(matches!(
            bw.set_sampling_frequency(-1.0),
            Err(ButterworthError::InvalidSamplingFrequency)
        ));
        // 50 <= 2 * 40
        assert!(matches!(
            bw.set_sampling_frequency(50.0),
            Err(ButterworthError::NyquistCriterionViolation)
        ));
        // 80 == 2 * 40 is also rejected
        assert!(matches!(
            bw.set_sampling_frequency(80.0),
            Err(ButterworthError::NyquistCriterionViolation)
        ));
    }
}
