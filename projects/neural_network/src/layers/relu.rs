/*
    y = ReLU(x)
      = { x  if x > 0
        { 0  if x <= 0

    ∂L/∂x = { ∂L/∂y  if x > 0
            { 0      if x <= 0
*/

use ndarray::{Array2, Array1};

use super::layer::Layer;

pub(crate) struct ReLU {
    filter: Option<Array2<bool>>,
}

pub(crate) struct InputOfReLULayer {
    input: Array2<f32>,
}

impl From<Array2<f32>> for InputOfReLULayer {
    fn from(value: Array2<f32>) -> Self {
        Self { input: value }
    }
}

pub(crate) struct DInputOfReLULayer {
    dinput: Array2<f32>,
}

impl Into<Array2<f32>> for DInputOfReLULayer {
    fn into(self) -> Array2<f32> {
        self.dinput
    }
}

pub(crate) struct OutputOfReLULayer {
    out: Array2<f32>,
}

impl From<Array2<f32>> for OutputOfReLULayer {
    fn from(value: Array2<f32>) -> Self {
        Self { out: value }
    }
}

impl Into<Array2<f32>> for OutputOfReLULayer {
    fn into(self) -> Array2<f32> {
        self.out
    }
}

impl Layer for ReLU {
    type Input = InputOfReLULayer;
    type Output = OutputOfReLULayer;
    type DInput = DInputOfReLULayer;
    fn new() -> Self {
        Self { filter: None }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input } = input;
        let filter = input.clone().mapv_into_any(|x| if x > 0. { true } else { false });
        self.filter = Some(filter);
        let out = input.mapv_into(|x| if x > 0. { x } else { 0. });
        Self::Output { out }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.filter.is_some());
        let filter = self.filter.as_ref().unwrap();
        let Self::Output { out: dout } = dout;

        let dinput_1d: Array1<f32> = dout.into_iter().zip(filter).map(|(dout, filter)| {
            if *filter {
                dout
            } else {
                0.
            }
        }).collect();
        let dinput = dinput_1d.into_shape((filter.shape()[0], filter.shape()[1])).unwrap();
        Self::DInput { dinput }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_relu() {
        // test forward
        let mut relu = ReLU::new();
        let input = InputOfReLULayer {
            input: array![[1., -2., 18.], [-3., 3., 5.]],
        };
        let output = relu.forward(input);

        let expected = array![[1., 0., 18.,], [0., 3., 5.]];
        assert_eq!(output.out, expected);

        // test backward
        let dout = OutputOfReLULayer {
            out: array![[7., 8., 9.], [10., 11., 12.]],
        };
        let dinput = relu.backward(dout);

        // numerical gradient
        let expected = array![[7., 0., 9.], [0., 11., 12.]];
        dinput
            .dinput
            .into_iter()
            .zip(expected)
            .for_each(|(dinput, expected)| {
                assert_eq!(dinput, expected);
            });
    }
}
