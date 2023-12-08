use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

pub(crate) trait LayerBase {
    type Params;
    fn new(params: Self::Params) -> Self;
    fn params_and_grads(&mut self) -> (&mut Self::Params, &Self::Params);
}

pub(crate) trait IntermediateLayer<M2, M1>: LayerBase
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn forward(&mut self, input: M2) -> M2;
    fn backward(&mut self, dout: M2) -> M2;
}

pub(crate) trait LossLayer<M2, M1>: LayerBase
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn forward(&mut self, input: M2, one_hot_labels: M2) -> f32;
    fn backward(&mut self, dout: f32) -> M2;
}
