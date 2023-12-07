use std::marker::PhantomData;

use ndarray::{Array1, Array2, Axis};

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

struct Repeat<M2, M1> {
    n: Option<usize>,
    ph2: PhantomData<M2>,
    ph1: PhantomData<M1>,
}

struct InputOfRepeatLayer<M1> {
    input: M1,
    n: usize,
}

struct DInputOfRepeatLayer<M1> {
    dinput: M1,
}

struct OutputOfRepeatLayer<M2> {
    out: M2,
}

impl<M2, M1> Layer<M2, M1> for Repeat<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfRepeatLayer<M1>;
    type Output = OutputOfRepeatLayer<M2>;
    type DInput = DInputOfRepeatLayer<M1>;

    fn new() -> Self {
        Self {
            n: None,
            ph2: PhantomData,
            ph1: PhantomData,
        }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input, n } = input;
        self.n = Some(n);
        let len = input.len();
        Self::Output {
            out: M2::broadcast_1d_array(input, (n, len)),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.n.is_some());
        Self::DInput {
            dinput: dout.out.sum_axis_zero().to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_repeat_layer() {
        // test forward
        let mut repeat = Repeat::new();
        let input = InputOfRepeatLayer {
            input: array![1., 2., 3., 4., 5., 6.],
            n: 3,
        };
        let output = repeat.forward(input);

        assert_eq!(
            output.out,
            array![
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.]
            ]
        );

        // test backward
        let dout = OutputOfRepeatLayer {
            out: array![
                [7., 8., 9., 10., 11., 12.],
                [13., 14., 15., 16., 17., 18.],
                [19., 20., 21., 22., 23., 24.],
            ],
        };
        let dinput = repeat.backward(dout);
        assert_eq!(dinput.dinput, array![39., 42., 45., 48., 51., 54.]);
    }
}
