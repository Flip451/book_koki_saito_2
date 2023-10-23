use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::num_traits::Zero;

use super::layer::Layer;

struct Repeat {
    n: Option<usize>,
}

struct InputOfRepeatLayer {
    input: Array1<f64>,
    n: usize,
}

struct DInputOfRepeatLayer {
    dinput: Array1<f64>,
}

struct OutputOfRepeatLayer {
    out: Array2<f64>,
}

impl Layer for Repeat
{
    type Input = InputOfRepeatLayer;
    type Output = OutputOfRepeatLayer;
    type DInput = DInputOfRepeatLayer;

    fn new() -> Self {
        Self {
            n: None,
        }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output
    {
        let Self::Input { input, n } = input;
        self.n = Some(n);
        Self::Output {
            out: input.broadcast((n, input.len())).unwrap().to_owned(),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.n.is_some());
        Self::DInput {
            dinput: dout.out.sum_axis(Axis(0)).to_owned(),
        }
    }
}
