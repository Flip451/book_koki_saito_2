use ndarray::Array2;

use crate::{
    layers::{
        layer::Layer,
        relu::{InputOfReLULayer, OutputOfReLULayer, ReLU},
    },
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use super::layer::{IntermediateLayer, LayerBase};

pub(crate) struct ReLULayer<M2, M1> {
    relu: ReLU<M2, M1>,
    params: ParamsOfReLULayer,
    grads: ParamsOfReLULayer,
}

pub(crate) struct ParamsOfReLULayer();

impl<M2, M1> LayerBase for ReLULayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
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

impl<M2, M1> IntermediateLayer<M2, M1> for ReLULayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn forward(&mut self, input: M2) -> M2 {
        self.relu.forward(InputOfReLULayer::from(input)).into_value()
    }

    fn backward(&mut self, dout: M2) -> M2 {
        self.relu.backward(OutputOfReLULayer::from(dout)).into_value()
    }
}
