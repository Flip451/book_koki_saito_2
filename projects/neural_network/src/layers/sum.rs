use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::num_traits::Zero;

use super::layer::Layer;

struct Sum {
    n: Option<usize>,
}

struct InputOfSumLayer {
    input: Array2<f64>,
}

struct DInputOfSumLayer {
    dinput: Array2<f64>,
}

struct OutputOfSumLayer {
    out: Array1<f64>,
}

impl Layer for Sum {
    type Input = InputOfSumLayer;
    type Output = OutputOfSumLayer;
    type DInput = DInputOfSumLayer;

    fn new() -> Self {
        Self { n: None }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input } = input;
        self.n = Some(input.shape()[0]);
        Self::Output {
            out: input.sum_axis(Axis(0)).to_owned(),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.n.is_some());
        Self::DInput {
            dinput: dout
                .out
                .broadcast((self.n.unwrap(), dout.out.len()))
                .unwrap()
                .to_owned(),
        }
    }
}
