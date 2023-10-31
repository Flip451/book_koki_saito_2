use ndarray::Array2;

use crate::optimizer::optimizer::Optimizer;

pub trait Network {
    fn predict(&mut self, input: Array2<f32>) -> Array2<f32>;
    fn forward(&mut self, input: Array2<f32>, one_hot_labels: Array2<f32>) -> f32;
    fn backward(&mut self, dout: f32) -> Array2<f32>;
    fn update<T: Optimizer>(&mut self, optimizer: &T);
}
