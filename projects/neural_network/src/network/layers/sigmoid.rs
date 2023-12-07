use ndarray::Array2;

use crate::{
    layers::{
        layer::Layer,
        sigmoid::{InputOfSigmoidLayer, OutputOfSigmoidLayer, Sigmoid},
    },
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use super::layer::{IntermediateLayer, LayerBase};

pub(crate) struct SigmoidLayer<M2, M1> {
    sigmoid: Sigmoid<M2, M1>,
    params: ParamsOfSigmoidLayer,
    grads: ParamsOfSigmoidLayer,
}

pub(crate) struct ParamsOfSigmoidLayer();

impl<M2, M1> LayerBase for SigmoidLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
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

impl<M2, M1> IntermediateLayer<M2, M1> for SigmoidLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn forward(&mut self, input: M2) -> M2 {
        self.sigmoid
            .forward(InputOfSigmoidLayer::from(input))
            .into_value()
    }

    fn backward(&mut self, dout: M2) -> M2 {
        self.sigmoid
            .backward(OutputOfSigmoidLayer::from(dout))
            .into_value()
    }
}
