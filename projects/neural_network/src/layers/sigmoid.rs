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
        let out = self.out.as_ref().unwrap();
        let Self::Output { out: dout } = dout;

        let dinput = out * (Array2::ones(out.dim()) - out) * dout;
        Self::DInput { dinput }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_sigmoid() {
        // test forward
        let mut sigmoid = Sigmoid::new();
        let input = InputOfSigmoidLayer {
            input: array![[1., 2., 3.], [4., 5., 6.]],
        };
        let output = sigmoid.forward(input);

        let expected = array![
            [
                (1.) / (1. + (-1_f64).exp()),
                (1.) / (1. + (-2_f64).exp()),
                (1.) / (1. + (-3_f64).exp())
            ],
            [
                (1.) / (1. + (-4_f64).exp()),
                (1.) / (1. + (-5_f64).exp()),
                (1.) / (1. + (-6_f64).exp())
            ]
        ];
        assert_eq!(output.out, expected);

        // test backward
        let dout = OutputOfSigmoidLayer {
            out: array![[7., 8., 9.], [10., 11., 12.]],
        };
        let dinput = sigmoid.backward(dout);

        // numerical gradient
        // dy/dx = 1/[4cosh^2(x/2)]
        let expected = array![
            [
                (7.) / 4. / 0.5_f64.cosh().powf(2.),
                (8.) / 4. / 1_f64.cosh().powf(2.),
                (9.) / 4. / 1.5_f64.cosh().powf(2.),
            ],
            [
                (10.) / 4. / 2_f64.cosh().powf(2.),
                (11.) / 4. / 2.5_f64.cosh().powf(2.),
                (12.) / 4. / 3_f64.cosh().powf(2.),
            ]
        ];
        dinput
            .dinput
            .into_iter()
            .zip(expected)
            .for_each(|(dinput, expected)| {
                assert_abs_diff_eq!(dinput, expected, epsilon = 1e-14);
            });
    }
}
