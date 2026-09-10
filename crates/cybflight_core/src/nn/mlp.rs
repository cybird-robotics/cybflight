//! Fixed-topology multi-layer perceptron inference, `no_std` and
//! allocation-free.
//!
//! Layout matches what PyTorch exports: for each layer, the weight matrix
//! `W` is `[out_features, in_features]` in **row-major** order, immediately
//! followed by the `[out_features]` bias vector. Layers are concatenated in
//! forward order into one flat `&[f32]`.
//!
//! That layout is not a choice — it is `nn.Linear.weight.numpy().tobytes()`.
//! Keeping it verbatim means the exporter is a memcpy and there is no
//! transpose anywhere that could silently disagree with the trainer.
//!
//! Every hidden layer applies [`Activation`]; the last layer is linear —
//! the standard stable-baselines3 actor stack (`mlp_extractor.policy_net`
//! followed by `action_net`). A network whose trained head has its own
//! output nonlinearity folds that into the caller, which is invariably
//! reading the head through a scaling map anyway.

/// Default widest layer supported. Two scratch buffers of `W` floats live
/// on the stack during [`Mlp::forward`] (1 KiB total at 128); a wider
/// network sets `W` explicitly rather than making every user pay for it.
pub const MAX_WIDTH: usize = 128;

/// Hidden-layer nonlinearity. A property of the trained checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    /// Exact GELU, `0.5·x·(1 + erf(x/√2))` — PyTorch's `nn.GELU()` default
    /// (`approximate='none'`). The tanh approximation is *not* used: its
    /// ~1e-3 error compounds across a 512-wide tower into far more than
    /// the tolerance a port-fidelity test can carry.
    Gelu,
    /// `tanh` — Stable-Baselines3's `MlpPolicy` default hidden activation.
    Tanh,
}

impl Activation {
    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Relu => x.max(0.0),
            Activation::Gelu => {
                0.5 * x * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2))
            }
            Activation::Tanh => libm::tanhf(x),
        }
    }
}

/// One `nn.Linear` layer's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerShape {
    pub inputs: u16,
    pub outputs: u16,
}

impl LayerShape {
    pub const fn new(inputs: u16, outputs: u16) -> Self {
        Self { inputs, outputs }
    }

    /// Number of `f32` this layer consumes from the flat weight slice.
    pub const fn weight_count(&self) -> usize {
        (self.inputs as usize) * (self.outputs as usize) + self.outputs as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpError {
    /// The flat weight slice is not exactly the length the shapes imply.
    /// Length is checked eagerly at construction so `forward` cannot panic.
    WeightLen { expected: usize, actual: usize },
    /// A layer is wider than [`MAX_WIDTH`].
    TooWide { width: usize },
    /// No layers.
    Empty,
    /// Caller's input or output slice does not match the network's dims.
    BadIo,
    /// Consecutive layers disagree on dimension.
    Discontinuous { layer: usize },
}

/// A borrowed network. Holds no weights of its own, so the same type serves
/// a `&'static` flash blob on the target and a heap buffer in the host sim.
///
/// `W` bounds the layer width and hence the two stack scratch buffers
/// [`Mlp::forward`] uses; it is a compile-time property of the topology,
/// so a 64-wide policy never pays for a 512-wide one.
#[derive(Clone, Copy, Debug)]
pub struct Mlp<'a, const W: usize = MAX_WIDTH> {
    weights: &'a [f32],
    shapes: &'a [LayerShape],
    activation: Activation,
}

impl<'a, const W: usize> Mlp<'a, W> {
    /// Validate shapes against the weight slice. After this returns `Ok`,
    /// [`Mlp::forward`] is total: every index it computes is in bounds.
    pub fn new(
        weights: &'a [f32],
        shapes: &'a [LayerShape],
        activation: Activation,
    ) -> Result<Self, MlpError> {
        if shapes.is_empty() {
            return Err(MlpError::Empty);
        }
        let mut expected = 0usize;
        for (i, s) in shapes.iter().enumerate() {
            let w = (s.outputs as usize).max(s.inputs as usize);
            if w > W {
                return Err(MlpError::TooWide { width: w });
            }
            if i > 0 && shapes[i - 1].outputs != s.inputs {
                return Err(MlpError::Discontinuous { layer: i });
            }
            expected += s.weight_count();
        }
        if weights.len() != expected {
            return Err(MlpError::WeightLen { expected, actual: weights.len() });
        }
        Ok(Self { weights, shapes, activation })
    }

    pub fn input_dim(&self) -> usize {
        self.shapes[0].inputs as usize
    }

    pub fn output_dim(&self) -> usize {
        self.shapes[self.shapes.len() - 1].outputs as usize
    }

    /// Run the network. `input.len()` must equal [`Mlp::input_dim`] and
    /// `output.len()` [`Mlp::output_dim`].
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<(), MlpError> {
        if input.len() != self.input_dim() || output.len() != self.output_dim() {
            return Err(MlpError::BadIo);
        }
        let mut src = [0.0f32; W];
        let mut dst = [0.0f32; W];
        src[..input.len()].copy_from_slice(input);

        let last = self.shapes.len() - 1;
        let mut off = 0usize;
        for (li, s) in self.shapes.iter().enumerate() {
            let (nin, nout) = (s.inputs as usize, s.outputs as usize);
            let w = &self.weights[off..off + nin * nout];
            off += nin * nout;
            let b = &self.weights[off..off + nout];
            off += nout;

            for o in 0..nout {
                let row = &w[o * nin..(o + 1) * nin];
                let mut acc = b[o];
                for i in 0..nin {
                    acc += row[i] * src[i];
                }
                dst[o] = if li < last { self.activation.apply(acc) } else { acc };
            }
            core::mem::swap(&mut src, &mut dst);
        }
        output.copy_from_slice(&src[..self.output_dim()]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_weight_length() {
        let shapes = [LayerShape::new(2, 3), LayerShape::new(3, 1)];
        // correct is 2*3+3 + 3*1+1 = 13
        let w = [0.0f32; 12];
        assert_eq!(
            Mlp::<MAX_WIDTH>::new(&w, &shapes, Activation::Relu).err(),
            Some(MlpError::WeightLen { expected: 13, actual: 12 })
        );
    }

    #[test]
    fn rejects_discontinuous_layers() {
        let shapes = [LayerShape::new(2, 3), LayerShape::new(4, 1)];
        let w = [0.0f32; 2 * 3 + 3 + 4 * 1 + 1];
        assert_eq!(
            Mlp::<MAX_WIDTH>::new(&w, &shapes, Activation::Relu).err(),
            Some(MlpError::Discontinuous { layer: 1 })
        );
    }

    /// `W` bounds the layer width, and a network that exceeds it is
    /// rejected at construction rather than overflowing the scratch buffer.
    #[test]
    fn rejects_layers_wider_than_the_scratch_buffer() {
        let shapes = [LayerShape::new(2, 4)];
        let w = [0.0f32; 2 * 4 + 4];
        assert_eq!(
            Mlp::<3>::new(&w, &shapes, Activation::Relu).err(),
            Some(MlpError::TooWide { width: 4 })
        );
    }

    /// GELU must be the exact erf form, not the tanh approximation: the
    /// two differ by ~1e-3 near |x| = 2, which is the whole reason the
    /// port carries an erf.
    #[test]
    fn gelu_matches_the_exact_erf_definition() {
        for &x in &[-3.0f32, -0.5, 0.0, 0.7, 2.0] {
            let want = 0.5 * x * (1.0 + libm::erff(x / core::f32::consts::SQRT_2));
            assert!((Activation::Gelu.apply(x) - want).abs() < 1e-7, "x={x}");
        }
        assert!(Activation::Gelu.apply(-10.0).abs() < 1e-6);
    }

    #[test]
    fn two_layer_forward_matches_hand_computation() {
        // layer0: W=[[1,2],[0,-1],[3,1]] b=[0,1,-2]  (3x2)
        // layer1: W=[[1,1,1]]           b=[0.5]      (1x3)
        let shapes = [LayerShape::new(2, 3), LayerShape::new(3, 1)];
        let w = [
            1.0, 2.0, 0.0, -1.0, 3.0, 1.0, // W0 row-major [out,in]
            0.0, 1.0, -2.0, // b0
            1.0, 1.0, 1.0, // W1
            0.5, // b1
        ];
        let mlp = Mlp::<MAX_WIDTH>::new(&w, &shapes, Activation::Relu).unwrap();
        let mut out = [0.0f32; 1];
        mlp.forward(&[1.0, 1.0], &mut out).unwrap();
        // h = relu([1*1+2*1+0, 0*1-1*1+1, 3*1+1*1-2]) = relu([3,0,2]) = [3,0,2]
        // y = 3+0+2+0.5 = 5.5
        assert!((out[0] - 5.5).abs() < 1e-6, "got {}", out[0]);
    }

    #[test]
    fn activation_is_not_applied_to_the_output_layer() {
        let shapes = [LayerShape::new(1, 1), LayerShape::new(1, 1)];
        // identity then negate: output must stay negative
        let w = [1.0, 0.0, -1.0, 0.0];
        let mlp = Mlp::<MAX_WIDTH>::new(&w, &shapes, Activation::Relu).unwrap();
        let mut out = [0.0f32; 1];
        mlp.forward(&[2.0], &mut out).unwrap();
        assert!((out[0] + 2.0).abs() < 1e-6, "got {}", out[0]);
    }
}
