use std::ops::{Sub, Mul};

use ndarray::Array2;

pub(crate) trait LayerBase {
    type Params: Sub + Mul<f64, Output = Self::Params>;
    fn new(params: Self::Params) -> Self;
    fn params_and_grads(&mut self) -> (&mut Self::Params, &Self::Params);
}

pub(crate) trait TransformLayer: LayerBase {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, dout: Array2<f64>) -> Array2<f64>;
}

pub(crate) trait LossLayer: LayerBase {
    fn forward(&mut self, input: Array2<f64>, one_hot_labels: Array2<f64>) -> f64;
    fn backward(&mut self, dout: f64) -> Array2<f64>;
}
