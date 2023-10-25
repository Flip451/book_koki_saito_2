use std::ops::{Mul, SubAssign};

pub struct LearningRate(f64);

impl LearningRate {
    pub fn new(lr: f64) -> Self {
        Self(lr)
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

pub trait Optimizer {
    fn new(lr: LearningRate) -> Self;
    fn update<P, G>(&self, params: Vec<&mut P>, grads: Vec<&G>)
    where
        P: SubAssign<G>,
        G: Mul<f64, Output = G> + Clone;
}
