/*
    y = 1/(1 + exp(-x))
    ∂L/∂x = y(1-y)∂L/∂y
*/

use ndarray::Array2;

use super::layer::Layer;

struct Sigmoid {
    out: Option<Array2<f64>>,
}

struct InputOfSigmoidLayer {
    input: Array2<f64>,
}

struct DInputOfSigmoidLayer {
    dinput: Array2<f64>,
}

struct OutputOfSigmoidLayer {
    out: Array2<f64>,
}

impl<_T> Layer<_T> for Sigmoid {
    type Input<_U> = InputOfSigmoidLayer;
    type Output<_V> = OutputOfSigmoidLayer;
    type DInput<_W> = DInputOfSigmoidLayer;
    fn new() -> Self {
        Self { out: None }
    }

    fn forward(&mut self, input: Self::Input<_T>) -> Self::Output<_T> {
        let Self::Input::<_T> { input } = input;
        let out = input.mapv_into(|x| 1. / (1. + (-x).exp()));
        self.out = Some(out.clone());
        Self::Output::<_T> { out }
    }

    fn backward(&self, dout: Self::Output<_T>) -> Self::DInput<_T> {
        assert!(self.out.is_some());
        let Self::Output::<_T> { out } = dout;

        let dinput = out.clone() * (1.0 - out) * self.out.as_ref().unwrap();
        Self::DInput::<_T> { dinput }
    }
}
