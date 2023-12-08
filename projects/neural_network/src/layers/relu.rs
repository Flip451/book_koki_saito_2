/*
    y = ReLU(x)
      = { x  if x > 0
        { 0  if x <= 0

    ∂L/∂x = { ∂L/∂y  if x > 0
            { 0      if x <= 0
*/

use std::marker::PhantomData;

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

pub(crate) struct ReLU<M2, M1> {
    filter: Option<M2>,
    ph: PhantomData<M1>,
}

pub(crate) struct InputOfReLULayer<M2> {
    input: M2,
}

impl<M2> From<M2> for InputOfReLULayer<M2> {
    fn from(value: M2) -> Self {
        Self { input: value }
    }
}

pub(crate) struct DInputOfReLULayer<M2> {
    dinput: M2,
}

impl<M2> DInputOfReLULayer<M2> {
    pub fn into_value(self) -> M2 {
        self.dinput
    }
}

impl<M2> From<M2> for DInputOfReLULayer<M2> {
    fn from(value: M2) -> Self {
        Self { dinput: value }
    }
}

pub(crate) struct OutputOfReLULayer<M2> {
    out: M2,
}

impl<M2> OutputOfReLULayer<M2> {
    pub fn into_value(self) -> M2 {
        self.out
    }
}

impl<M2> From<M2> for OutputOfReLULayer<M2> {
    fn from(value: M2) -> Self {
        Self { out: value }
    }
}

impl<M2, M1> Layer<M2, M1> for ReLU<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfReLULayer<M2>;
    type Output = OutputOfReLULayer<M2>;
    type DInput = DInputOfReLULayer<M2>;
    fn new() -> Self {
        Self { filter: None, ph: PhantomData }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input } = input;
        let filter = input
            .clone()
            .mapv_into(|x| if x > 0. { 1. } else { 0. });
        self.filter = Some(filter);
        let out = input.mapv_into(|x| if x > 0. { x } else { 0. });
        Self::Output { out }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.filter.is_some());
        let filter = self.filter.as_ref().unwrap();
        let Self::Output { out: dout } = dout;
        let dinput = dout * filter.clone();
        Self::DInput { dinput }
    }
}

#[cfg(test)]
mod tests {
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
