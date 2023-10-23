use ndarray::{ArrayView, ArrayViewMut, Dimension};

pub struct LearningRate(f64);

impl LearningRate {
    pub fn new(lr: f64) -> Self {
        Self(lr)
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

pub(super) trait Optimizer {
    fn new(lr: LearningRate) -> Self;
    fn update<D>(self, params: Vec<ArrayViewMut<f64, D>>, grads: Vec<ArrayView<f64, D>>)
    where
        D: Dimension;
}
