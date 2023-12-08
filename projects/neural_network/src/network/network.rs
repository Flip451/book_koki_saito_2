use crate::optimizer::optimizer::Optimizer;

pub trait Network<M2, M1> {
    fn predict(&mut self, input: M2) -> M2;
    fn forward(&mut self, input: M2, one_hot_labels: M2) -> f32;
    fn backward(&mut self, dout: f32) -> M2;
    fn update<T: Optimizer>(&mut self, optimizer: &T);
}
