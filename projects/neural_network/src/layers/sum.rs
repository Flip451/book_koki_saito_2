use std::marker::PhantomData;

use ndarray::{Array1, Array2, Axis};

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

struct Sum<M2, M1> {
    n: Option<usize>,
    ph2: PhantomData<M2>,
    ph1: PhantomData<M1>,
}

struct InputOfSumLayer<M2> {
    input: M2,
}

struct DInputOfSumLayer<M2> {
    dinput: M2,
}

struct OutputOfSumLayer<M1> {
    out: M1,
}

impl<M2, M1> Layer<M2, M1> for Sum<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfSumLayer<M2>;
    type Output = OutputOfSumLayer<M1>;
    type DInput = DInputOfSumLayer<M2>;

    fn new() -> Self {
        Self {
            n: None,
            ph2: PhantomData,
            ph1: PhantomData,
        }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input } = input;
        self.n = Some(input.dim().0);
        Self::Output {
            out: input.sum_axis_zero().to_owned(),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.n.is_some());
        let len = dout.out.len();
        Self::DInput {
            dinput: M2::broadcast_1d_array(dout.out, (self.n.unwrap(), len)),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_sum_layer() {
        // test forward
        let mut sum = Sum::new();
        let input = InputOfSumLayer {
            input: array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        };
        let output = sum.forward(input);
        assert_eq!(output.out, array![12., 15., 18.]);

        // test backward
        let dout = OutputOfSumLayer {
            out: array![10., 11., 12.],
        };
        let dinput = sum.backward(dout);
        assert_eq!(
            dinput.dinput,
            array![[10., 11., 12.], [10., 11., 12.], [10., 11., 12.]]
        );
    }
}
