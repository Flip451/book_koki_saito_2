use ndarray::Array2;

pub(crate) trait LayerBase {
    type Params;
    fn new(params: Self::Params) -> Self;
    fn params_and_grads(&mut self) -> (&mut Self::Params, &Self::Params);
}

pub(crate) trait TransformLayer: LayerBase {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, dout: Array2<f32>) -> Array2<f32>;
}

pub(crate) trait LossLayer: LayerBase {
    fn forward(&mut self, input: Array2<f32>, one_hot_labels: Array2<f32>) -> f32;
    fn backward(&mut self, dout: f32) -> Array2<f32>;
}
