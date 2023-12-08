/*
    Y = X.dot(A) + B
    ∂L/∂X = ∂L/∂Y.dot(A^T)
    ∂L/∂A = X^T.dot(∂L/∂Y)
    ∂L/∂B = N ∂L/∂Y where N = first dimension of X = X.shape[0]
*/

use std::marker::PhantomData;

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

pub(crate) struct Affine<M2, M1> {
    x: Option<M2>,
    a: Option<M2>,
    ph: PhantomData<M1>,
}

pub(crate) struct InputOfAffineLayer<M2, M1> {
    pub(crate) x: M2,
    pub(crate) a: M2,
    pub(crate) b: M1,
}

pub(crate) struct DInputOfAffineLayer<M2, M1> {
    pub(crate) dx: M2,
    pub(crate) da: M2,
    pub(crate) db: M1,
}

pub(crate) struct OutputOfAffineLayer<M2> {
    out: M2,
}

impl<M2> OutputOfAffineLayer<M2> {
    pub fn into_value(self) -> M2 {
        self.out
    }
}

impl<M2> From<M2> for OutputOfAffineLayer<M2> {
    fn from(value: M2) -> Self {
        Self { out: value }
    }
}

impl<M2, M1> Layer<M2, M1> for Affine<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfAffineLayer<M2, M1>;
    type Output = OutputOfAffineLayer<M2>;
    type DInput = DInputOfAffineLayer<M2, M1>;

    fn new() -> Self {
        Self {
            a: None,
            x: None,
            ph: PhantomData,
        }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { x, a, b } = input;
        self.x = Some(x.clone());
        self.a = Some(a.clone());
        Self::Output {
            out: x.dot(self.a.as_ref().unwrap()) + b,
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
            db: dout.sum_axis_zero().to_owned(),
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
        let mut affine = Affine::new();
        let input = InputOfAffineLayer {
            x: array![[1., 2., 3.], [4., 5., 6.]],
            a: array![[1., 2.], [3., 4.], [5., 6.]],
            b: array![7., 8.],
        };
        let output = affine.forward(input);
        assert_eq!(output.out, array![[29., 36.], [56., 72.]]);

        // test backward
        let dout = OutputOfAffineLayer {
            out: array![[7., 8.], [9., 10.]],
        };
        let dinput = affine.backward(dout);

        assert_eq!(dinput.dx, array![[23., 53., 83.], [29., 67., 105.]]);
        assert_eq!(dinput.da, array![[43., 48.], [59., 66.], [75., 84.]]);
        assert_eq!(dinput.db, array![16., 18.]);
    }
}
