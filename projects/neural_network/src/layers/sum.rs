use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::num_traits::Zero;

use super::layer::Layer;

struct Sum {
    n: Option<usize>,
}

struct InputOfSumLayer<T> {
    input: Array2<T>,
}

struct DInputOfSumLayer<T> {
    dinput: Array2<T>,
}

struct OutputOfSumLayer<T> {
    out: Array1<T>,
}

impl<T> Layer<T> for Sum
where
    T: std::ops::Add<Output = T> + Clone + Zero,
{
    type Input<U> = InputOfSumLayer<U>;
    type Output<U> = OutputOfSumLayer<U>;
    type DInput<U> = DInputOfSumLayer<U>;

    fn new() -> Self {
        Self { n: None }
    }

    fn forward(&mut self, input: Self::Input<T>) -> Self::Output<T>
    where
        T: std::ops::Add<Output = T>,
    {
        let Self::Input { input } = input;
        self.n = Some(input.shape()[0]);
        Self::Output {
            out: input.sum_axis(Axis(0)).to_owned(),
        }
    }

    fn backward(&self, dout: Self::Output<T>) -> Self::DInput<T> {
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
