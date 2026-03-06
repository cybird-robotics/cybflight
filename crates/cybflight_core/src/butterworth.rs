use core::clone::Clone;
use core::convert::TryInto;
use core::f64::consts::PI;
use num_traits::float::Float;

#[derive(Debug)]
pub enum ButterworthError {
    InvalidSamplingFrequency,
    InvalidCutoffFrequency,
    SamplingTheoremViolation,
    SizeMismatch,
}

/// Butterworth filter context structure.
///
/// This structure holds the state of the Butterworth filter, including
/// coefficients, input and output buffers, and the current index for
/// processing.
///
pub struct ButterworthFilter<T: Float, const DIM: usize> {
    den: [T; 2],
    num: [T; 2],
    input: [[T; DIM]; 2],  // Assuming Dim = 3 for a 3D vector
    output: [[T; DIM]; 2], // Assuming Dim = 3 for a 3D vector
    cutoff_frequency: T,
    sampling_frequency: T,
    curr_idx: usize,
}

impl<T: Float, const DIM: usize> ButterworthFilter<T, DIM> {
    /// Initialize the Butterworth filter context.
    ///
    /// This function initializes the Butterworth filter context with the specified
    /// cutoff frequency and sampling frequency.
    ///
    /// @param cutoff_frequency Cutoff frequency in Hz.
    /// @param sampling_frequency Sampling frequency in Hz.
    /// @return true if initialization is successful
    /// @return false if either cutoff or sampling frequency is non-positive or
    /// non-finite, or if the sampling frequency is less than twice the cutoff
    /// frequency.
    ///
    pub fn new(cutoff_frequency: T, sampling_frequency: T) -> Result<Self, ButterworthError> {
        if cutoff_frequency <= T::zero() {
            return Err(ButterworthError::InvalidCutoffFrequency);
        }

        if sampling_frequency <= T::zero() {
            return Err(ButterworthError::InvalidSamplingFrequency);
        }

        if sampling_frequency < T::from(2.0).unwrap() * cutoff_frequency {
            return Err(ButterworthError::SamplingTheoremViolation);
        }

        let mut filter = ButterworthFilter {
            den: [T::zero(); 2],
            num: [T::zero(); 2],
            input: [[T::zero(); DIM]; 2],
            output: [[T::zero(); DIM]; 2],
            cutoff_frequency,
            sampling_frequency,
            curr_idx: 0,
        };

        filter.reset_numden();
        Ok(filter)
    }

    /// @brief Set the cutoff frequency of the Butterworth filter.
    ///
    /// This function sets the cutoff frequency for the Butterworth filter and
    /// recalculates the filter coefficients.
    ///
    /// @param cutoff_frequency New cutoff frequency in Hz.
    /// @return true if the cutoff frequency is set successfully
    /// @return false if the cutoff frequency is non-positive, non-finite, or twice
    /// the cutoff frequency is greater than the sampling frequency.
    pub fn set_cutoff_frequency(&mut self, cutoff_frequency: T) -> bool {
        if cutoff_frequency <= T::zero()
            || self.sampling_frequency < T::from(2.0).unwrap() * cutoff_frequency
        {
            return false;
        }

        self.cutoff_frequency = cutoff_frequency;
        self.reset_numden();
        true
    }

    /// @brief Set the sampling frequency of the Butterworth filter.
    ///
    /// This function sets the sampling frequency for the Butterworth filter and
    /// recalculates the filter coefficients.
    ///
    /// @param sampling_frequency New sampling frequency in Hz.
    /// @return true if the sampling frequency is set successfully
    /// @return false if the sampling frequency is non-positive, non-finite, or
    /// less than twice the cutoff frequency.
    pub fn set_sampling_frequency(&mut self, sampling_frequency: T) -> bool {
        if sampling_frequency <= T::zero()
            || sampling_frequency < T::from(2.0).unwrap() * self.cutoff_frequency
        {
            return false;
        }

        self.sampling_frequency = sampling_frequency;
        self.reset_numden();
        true
    }

    /// @brief compute the butterworth filter output for a given sample.
    ///
    /// this function computes the output of the butterworth filter for a given
    /// input sample and updates the internal state of the filter.
    ///
    /// @param sample pointer to the input sample array of size dim.
    /// @param output pointer to the output sample array of size dim.
    /// @return true if the computation is successful
    /// @return false if the context is not initialized or if the input sample is
    /// not valid.
    pub fn compute(&mut self, sample: &[T]) -> Result<[T; DIM], ButterworthError> {
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
        self.input[self.curr_idx] = sample
            .try_into()
            .map_err(|_| ButterworthError::SizeMismatch)?;
        self.output[self.curr_idx] = result;
        Ok(result)
    }

    pub fn reset(&mut self) {
        self.curr_idx = 0;
        for inp in &mut self.input {
            *inp = [T::zero(); DIM];
        }
        for outp in &mut self.output {
            *outp = [T::zero(); DIM];
        }
    }

    fn reset_numden(&mut self) {
        let k = (T::from(PI).unwrap() * self.cutoff_frequency / self.sampling_frequency).tan();
        let k2 = k * k;
        let poly = k2 + T::from(2.0).unwrap().sqrt() * k + T::one();

        let b0 = k2 / poly;
        let a1 = T::from(2.0).unwrap() * (k2 - T::one()) / poly;
        let a2 = (k2 - T::from(2.0).unwrap().sqrt() * k + T::one()) / poly;

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

    use float_cmp::{ApproxEq, F32Margin};

    use crate::butterworth::ButterworthFilter;

    #[test]
    fn butterworth_test() {
        let bw = ButterworthFilter::<f32, 1>::new(40.0, 100.0);
        assert!(bw.is_some());
        let mut bw = bw.unwrap();

        const EXPECTED_IMPULSE_RESPONSE: [f32; 10] = [
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

        const STEP: [f32; 10] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for i in 0..10 {
            let result = bw.compute(&[STEP[i]]).unwrap()[0];
            assert!(result.approx_eq(EXPECTED_IMPULSE_RESPONSE[i], F32Margin::default()));
        }
    }
}
