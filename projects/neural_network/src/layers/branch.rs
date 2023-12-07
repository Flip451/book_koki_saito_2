use std::marker::PhantomData;

use ndarray::Array2;

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

struct Branch<M2, M1>(PhantomData<M2>, PhantomData<M1>);

struct InputOfBranchLayer<M2> {
    input: M2,
}

struct DInputOfBranchLayer<M2> {
    dinput: M2,
}

struct OutputOfBranchLayer<M2> {
    a: M2,
    b: M2,
}

impl<M2, M1> Layer<M2, M1> for Branch<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfBranchLayer<M2>;
    type Output = OutputOfBranchLayer<M2>;
    type DInput = DInputOfBranchLayer<M2>;

    fn new() -> Self {
        Self(PhantomData, PhantomData)
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        Self::Output {
            a: input.input.clone(),
            b: input.input,
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        Self::DInput {
            dinput: dout.a + dout.b,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_branch_layer() {
        // test forward
        let mut branch = Branch::new();
        let input = InputOfBranchLayer {
            input: array![[1., 2., 3.], [4., 5., 6.]],
        };
        let output = branch.forward(input);

        let expected = array![[1., 2., 3.], [4., 5., 6.]];
        assert_eq!(output.a, expected);
        assert_eq!(output.b, expected);

        // test backward
        let dout = OutputOfBranchLayer {
            a: array![[1., 2., 3.], [4., 5., 6.]],
            b: array![[7., 8., 9.], [10., 11., 12.]],
        };

        let dinput = branch.backward(dout);
        let expected = array![[8., 10., 12.], [14., 16., 18.]];
        assert_eq!(dinput.dinput, expected);
    }
}
