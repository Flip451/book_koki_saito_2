/*
    Y = X.dot(A) + B
    ∂L/∂X = ∂L/∂Y.dot(A^T)
    ∂L/∂A = X^T.dot(∂L/∂Y)
    ∂L/∂B = N ∂L/∂Y where N = first dimension of X = X.shape[0]
*/

use ndarray::{Array1, Array2, Axis};

use super::layer::Layer;

pub(crate) struct Affine {
    x: Option<Array2<f64>>,
    a: Option<Array2<f64>>,
}

pub(crate) struct InputOfAffineLayer {
    pub(crate) x: Array2<f64>,
    pub(crate) a: Array2<f64>,
    pub(crate) b: Array1<f64>,
}

pub(crate) struct DInputOfAffineLayer {
    pub(crate) dx: Array2<f64>,
    pub(crate) da: Array2<f64>,
    pub(crate) db: Array1<f64>,
}

pub(crate) struct OutputOfAffineLayer {
    out: Array2<f64>,
}

impl Layer for Affine {
    type Input = InputOfAffineLayer;
    type Output = OutputOfAffineLayer;
    type DInput = DInputOfAffineLayer;

    fn new() -> Self {
        Self { a: None, x: None }
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
            db: dout.sum_axis(Axis(0)).to_owned(),
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
