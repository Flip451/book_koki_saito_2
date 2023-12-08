/*
    y = 1/(1 + exp(-x))
    ∂L/∂x = y(1-y)∂L/∂y
*/

use std::marker::PhantomData;

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

pub(crate) struct Sigmoid<M2, M1> {
    out: Option<M2>,
    ph: PhantomData<M1>,
}

pub(crate) struct InputOfSigmoidLayer<M2> {
    input: M2,
}

impl<M2> From<M2> for InputOfSigmoidLayer<M2> {
    fn from(value: M2) -> Self {
        Self { input: value }
    }
}

pub(crate) struct DInputOfSigmoidLayer<M2> {
    dinput: M2,
}

impl<M2> DInputOfSigmoidLayer<M2> {
    pub fn into_value(self) -> M2 {
        self.dinput
    }
}

impl<M2> From<M2> for DInputOfSigmoidLayer<M2> {
    fn from(value: M2) -> Self {
        Self { dinput: value }
    }
}

pub(crate) struct OutputOfSigmoidLayer<M2> {
    out: M2,
}

impl<M2> OutputOfSigmoidLayer<M2> {
    pub fn into_value(self) -> M2 {
        self.out
    }
}

impl<M2> From<M2> for OutputOfSigmoidLayer<M2> {
    fn from(value: M2) -> Self {
        Self { out: value }
    }
}

impl<M2, M1> Layer<M2, M1> for Sigmoid<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfSigmoidLayer<M2>;
    type Output = OutputOfSigmoidLayer<M2>;
    type DInput = DInputOfSigmoidLayer<M2>;
    fn new() -> Self {
        Self {
            out: None,
            ph: PhantomData,
        }
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

        let dinput = out.clone() * (M2::ones_like(&out) - out.clone()) * dout;
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
                (1.) / (1. + (-1_f32).exp()),
                (1.) / (1. + (-2_f32).exp()),
                (1.) / (1. + (-3_f32).exp())
            ],
            [
                (1.) / (1. + (-4_f32).exp()),
                (1.) / (1. + (-5_f32).exp()),
                (1.) / (1. + (-6_f32).exp())
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
                (7.) / 4. / 0.5_f32.cosh().powf(2.),
                (8.) / 4. / 1_f32.cosh().powf(2.),
                (9.) / 4. / 1.5_f32.cosh().powf(2.),
            ],
            [
                (10.) / 4. / 2_f32.cosh().powf(2.),
                (11.) / 4. / 2.5_f32.cosh().powf(2.),
                (12.) / 4. / 3_f32.cosh().powf(2.),
            ]
        ];
        dinput
            .dinput
            .into_iter()
            .zip(expected)
            .for_each(|(dinput, expected)| {
                assert_abs_diff_eq!(dinput, expected, epsilon = 1e-6);
            });
    }
}
