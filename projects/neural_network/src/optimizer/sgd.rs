use std::ops::{Add, Mul};

use super::optimizer::Optimizer;
use super::learning_rate::LearningRate;

pub struct SGD {
    lr: LearningRate,
}

impl SGD {
    pub fn new(lr: LearningRate) -> Self {
        Self { lr }
    }
}

impl<P, G> Optimizer<P, G> for SGD
where
    P: Add<G, Output = P> + Clone,
    G: Mul<f64, Output = G> + Clone,
{
    fn update(&self, params: &mut P, grads: &G) {
        *params = params.clone() + grads.clone() * (-self.lr.value());
    }
}
