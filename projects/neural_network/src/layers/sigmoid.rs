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

impl Layer for Sigmoid {
    type Input = InputOfSigmoidLayer;
    type Output = OutputOfSigmoidLayer;
    type DInput = DInputOfSigmoidLayer;
    fn new() -> Self {
        Self { out: None }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input } = input;
        let out = input.mapv_into(|x| 1. / (1. + (-x).exp()));
        self.out = Some(out.clone());
        Self::Output { out }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.out.is_some());
        let Self::Output { out } = dout;

        let dinput = out.clone() * (1.0 - out) * self.out.as_ref().unwrap();
        Self::DInput { dinput }
    }
}
