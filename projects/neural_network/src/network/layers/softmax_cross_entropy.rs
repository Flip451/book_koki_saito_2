use crate::{
    layers::{
        layer::Layer,
        softmax_cross_entropy::{
            InputOfSoftmaxCrossEntropyLayer, OutputOfSoftmaxCrossEntropyLayer, SoftmaxCrossEntropy,
        },
    },
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use super::layer::{LayerBase, LossLayer};

pub(crate) struct SoftmaxCrossEntropyLayer<M2, M1> {
    softmax_cross_entropy: SoftmaxCrossEntropy<M2, M1>,
    params: ParamsOfSoftmaxCrossEntropyLayer,
    grads: ParamsOfSoftmaxCrossEntropyLayer,
}

pub(crate) struct ParamsOfSoftmaxCrossEntropyLayer();

impl<M2, M1> LayerBase for SoftmaxCrossEntropyLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
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

impl<M2, M1> LossLayer<M2, M1> for SoftmaxCrossEntropyLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn forward(&mut self, input: M2, one_hot_labels: M2) -> f32 {
        self.softmax_cross_entropy
            .forward(InputOfSoftmaxCrossEntropyLayer::from(input, one_hot_labels))
            .into_value()
    }

    fn backward(&mut self, dout: f32) -> M2 {
        self.softmax_cross_entropy
            .backward(OutputOfSoftmaxCrossEntropyLayer::from(dout))
            .into_value()
    }
}
