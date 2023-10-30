use ndarray::Array2;

use crate::optimizer::optimizer::Optimizer;

pub trait Network {
    fn predict(&mut self, input: Array2<f64>) -> Array2<f64>;
    fn forward(&mut self, input: Array2<f64>, one_hot_labels: Array2<f64>) -> f64;
    fn backward(&mut self, dout: f64) -> Array2<f64>;
    fn update<T: Optimizer>(&mut self, optimizer: &T);
}
