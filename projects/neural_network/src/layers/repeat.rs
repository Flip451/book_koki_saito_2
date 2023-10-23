use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::num_traits::Zero;

use super::layer::Layer;

struct Repeat {
    n: Option<usize>,
}

struct InputOfRepeatLayer<T> {
    input: Array1<T>,
    n: usize,
}

struct DInputOfRepeatLayer<T> {
    dinput: Array1<T>,
}

struct OutputOfRepeatLayer<T> {
    out: Array2<T>,
}

impl<T> Layer<T> for Repeat
where
    T: std::ops::Add<Output = T> + Clone + Zero,
{
    type Input<U> = InputOfRepeatLayer<U>;
    type Output<U> = OutputOfRepeatLayer<U>;
    type DInput<U> = DInputOfRepeatLayer<U>;

    fn new() -> Self {
        Self {
            n: None,
        }
    }

    fn forward(&mut self, input: Self::Input<T>) -> Self::Output<T>
    where
        T: std::ops::Add<Output = T>,
    {
        let Self::Input { input, n } = input;
        self.n = Some(n);
        Self::Output {
            out: input.broadcast((n, input.len())).unwrap().to_owned(),
        }
    }

    fn backward(&self, dout: Self::Output<T>) -> Self::DInput<T> {
        Self::DInput {
            dinput: dout.out.sum_axis(Axis(0)).to_owned(),
        }
    }
}
