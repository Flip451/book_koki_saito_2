use std::ops::{Mul, Sub};

use ndarray::Array2;

use crate::layers::{
    layer::Layer,
    relu::{InputOfReLULayer, OutputOfReLULayer, ReLU},
};

use super::layer::{LayerBase, TransformLayer};

pub(crate) struct ReLULayer {
    relu: ReLU,
    params: ParamsOfReLULayer,
    grads: ParamsOfReLULayer,
}

pub(crate) struct ParamsOfReLULayer();

impl Sub for ParamsOfReLULayer {
    type Output = ParamsOfReLULayer;

    fn sub(self, _: Self) -> Self::Output {
        ParamsOfReLULayer()
    }
}

impl Mul<f64> for ParamsOfReLULayer {
    type Output = Self;

    fn mul(self, _: f64) -> Self::Output {
        ParamsOfReLULayer()
    }
}

impl LayerBase for ReLULayer {
    type Params = ParamsOfReLULayer;

    fn new(params: Self::Params) -> Self {
        let relu = ReLU::new();
        let grads = ParamsOfReLULayer();
        Self {
            relu,
            params,
            grads,
        }
    }

    fn params_and_grads(&mut self) -> (&mut Self::Params, &Self::Params) {
        (&mut self.params, &self.grads)
    }
}

impl TransformLayer for ReLULayer {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.relu
            .forward(InputOfReLULayer::from(input))
            .into()
    }

    fn backward(&mut self, dout: Array2<f64>) -> Array2<f64> {
        self.relu
            .backward(OutputOfReLULayer::from(dout))
            .into()
    }
}
