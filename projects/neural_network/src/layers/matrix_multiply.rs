/*
    Y = X.dot(A)
    ∂L/∂X = ∂L/∂Y.dot(A^T)
    ∂L/∂A = X^T.dot(∂L/∂Y)
*/

use std::marker::PhantomData;

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

struct MatMul<M2, M1> {
    x: Option<M2>,
    a: Option<M2>,
    ph: PhantomData<M1>,
}

struct InputOfMatMulLayer<M2> {
    x: M2,
    a: M2,
}

struct DInputOfMatMulLayer<M2> {
    dx: M2,
    da: M2,
}

struct OutputOfMatMulLayer<M2> {
    out: M2,
}

impl<M2, M1> Layer<M2, M1> for MatMul<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfMatMulLayer<M2>;
    type Output = OutputOfMatMulLayer<M2>;
    type DInput = DInputOfMatMulLayer<M2>;

    fn new() -> Self {
        Self {
            a: None,
            x: None,
            ph: PhantomData,
        }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { x, a } = input;
        self.x = Some(x.clone());
        self.a = Some(a.clone());
        Self::Output {
            out: x.dot(self.a.as_ref().unwrap()),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.x.is_some());
        assert!(self.a.is_some());
        let x = self.x.as_ref().unwrap();
        let a = self.a.as_ref().unwrap();
        let Self::Output { out: dout } = dout;
        Self::DInput {
            dx: dout.dot(&a.t()).to_owned(),
            da: x.t().dot(&dout).to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_matrix_multiply_layer() {
        // test forward
        let mut matmul = MatMul::new();
        let input = InputOfMatMulLayer {
            x: array![[1., 2., 3.], [4., 5., 6.]],
            a: array![[1., 2.], [3., 4.], [5., 6.]],
        };
        let output = matmul.forward(input);
        assert_eq!(output.out, array![[22., 28.], [49., 64.]]);

        // test backward
        let dout = OutputOfMatMulLayer {
            out: array![[7., 8.], [9., 10.]],
        };
        let dinput = matmul.backward(dout);

        assert_eq!(dinput.dx, array![[23., 53., 83.], [29., 67., 105.]]);
        assert_eq!(dinput.da, array![[43., 48.], [59., 66.], [75., 84.]])
    }
}
