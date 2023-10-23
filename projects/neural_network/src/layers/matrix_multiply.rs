/*
    Y = X.dot(A)
    ∂L/∂X = ∂L/∂Y.dot(A^T)
    ∂L/∂A = X^T.dot(∂L/∂Y)
*/

use ndarray::Array2;

use super::layer::Layer;

struct MatMul {
    x: Option<Array2<f64>>,
    a: Option<Array2<f64>>,
}

struct InputOfMatMulLayer {
    x: Array2<f64>,
    a: Array2<f64>,
}

struct DInputOfMatMulLayer {
    dx: Array2<f64>,
    da: Array2<f64>,
}

struct OutputOfMatMulLayer {
    out: Array2<f64>,
}

impl<_T> Layer<_T> for MatMul {
    type Input<_U> = InputOfMatMulLayer;
    type Output<_V> = OutputOfMatMulLayer;
    type DInput<_W> = DInputOfMatMulLayer;

    fn new() -> Self {
        Self {
            a: None,
            x: None,
        }
    }

    fn forward(&mut self, input: Self::Input<_T>) -> Self::Output<_T> {
        let Self::Input::<_T> { x, a } = input;
        self.x = Some(x.clone());
        self.x = Some(a.clone());
        Self::Output::<_T> {
            out: x.dot(self.a.as_ref().unwrap()),
        }
    }

    fn backward(&self, dout: Self::Output<_T>) -> Self::DInput<_T> {
        assert!(self.x.is_some());
        assert!(self.a.is_some());
        let x = self.a.as_ref().unwrap();
        let a = self.a.as_ref().unwrap();
        let Self::Output::<_T> { out: dout } = dout;
        Self::DInput::<_T> {
            dx: dout.dot(&a.t()).to_owned(),
            da: x.t().dot(&dout).to_owned(),
        }
    }
}
