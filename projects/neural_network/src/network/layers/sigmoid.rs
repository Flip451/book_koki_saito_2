use std::ops::{Mul, Sub};

use ndarray::Array2;

use crate::layers::{
    layer::Layer,
    sigmoid::{InputOfSigmoidLayer, OutputOfSigmoidLayer, Sigmoid},
};

use super::layer::{LayerBase, TransformLayer};

pub(crate) struct SigmoidLayer {
    sigmoid: Sigmoid,
    params: ParamsOfSigmoidLayer,
    grads: ParamsOfSigmoidLayer,
}

pub(crate) struct ParamsOfSigmoidLayer();

impl Sub for ParamsOfSigmoidLayer {
    type Output = ParamsOfSigmoidLayer;

    fn sub(self, _: Self) -> Self::Output {
        ParamsOfSigmoidLayer()
    }
}

impl Mul<f64> for ParamsOfSigmoidLayer {
    type Output = Self;

    fn mul(self, _: f64) -> Self::Output {
        ParamsOfSigmoidLayer()
    }
}

impl LayerBase for SigmoidLayer {
    type Params = ParamsOfSigmoidLayer;

    fn new(params: Self::Params) -> Self {
        let sigmoid = Sigmoid::new();
        let grads = ParamsOfSigmoidLayer();
        Self {
            sigmoid,
            params,
            grads,
        }
    }

    fn params_and_grads(&mut self) -> (&mut Self::Params, &Self::Params) {
        (&mut self.params, &self.grads)
    }
}

impl TransformLayer for SigmoidLayer {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.sigmoid
            .forward(InputOfSigmoidLayer::from(input))
            .into()
    }

    fn backward(&mut self, dout: Array2<f64>) -> Array2<f64> {
        self.sigmoid
            .backward(OutputOfSigmoidLayer::from(dout))
            .into()
    }
}
