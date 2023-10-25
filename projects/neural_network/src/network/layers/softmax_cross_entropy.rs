use std::ops::{Mul, Sub};

use ndarray::Array2;

use crate::layers::{
    layer::Layer,
    softmax_cross_entropy::{
        InputOfSoftmaxCrossEntropyLayer, OutputOfSoftmaxCrossEntropyLayer, SoftmaxCrossEntropy,
    },
};

use super::layer::{LayerBase, LossLayer};

pub(crate) struct SoftmaxCrossEntropyLayer {
    softmax_cross_entropy: SoftmaxCrossEntropy,
    params: ParamsOfSoftmaxCrossEntropyLayer,
    grads: ParamsOfSoftmaxCrossEntropyLayer,
}

pub(crate) struct ParamsOfSoftmaxCrossEntropyLayer();

impl Sub for ParamsOfSoftmaxCrossEntropyLayer {
    type Output = ParamsOfSoftmaxCrossEntropyLayer;

    fn sub(self, _: Self) -> Self::Output {
        ParamsOfSoftmaxCrossEntropyLayer()
    }
}

impl Mul<f64> for ParamsOfSoftmaxCrossEntropyLayer {
    type Output = Self;

    fn mul(self, _: f64) -> Self::Output {
        ParamsOfSoftmaxCrossEntropyLayer()
    }
}

impl LayerBase for SoftmaxCrossEntropyLayer {
    type Params = ParamsOfSoftmaxCrossEntropyLayer;

    fn new(params: Self::Params) -> Self {
        let softmax_cross_entropy = SoftmaxCrossEntropy::new();
        Self {
            softmax_cross_entropy,
            params,
            grads: ParamsOfSoftmaxCrossEntropyLayer(),
        }
    }

    fn params_and_grads(&mut self) -> (&mut Self::Params, &Self::Params) {
        (&mut self.params, &self.grads)
    }
}

impl LossLayer for SoftmaxCrossEntropyLayer {
    fn forward(&mut self, input: Array2<f64>, one_hot_labels: Array2<f64>) -> f64 {
        self.softmax_cross_entropy
            .forward(InputOfSoftmaxCrossEntropyLayer::from(input, one_hot_labels))
            .into()
    }

    fn backward(&mut self, dout: f64) -> Array2<f64> {
        self.softmax_cross_entropy
            .backward(OutputOfSoftmaxCrossEntropyLayer::from(dout))
            .into()
    }
}
