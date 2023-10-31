use std::ops::{Add, Mul};

use self::learning_rate::LearningRate;

use super::super::optimizer::Optimizer;

pub mod learning_rate;

pub struct SGD {
    lr: LearningRate,
}

impl Optimizer for SGD {
    type Params = LearningRate;

    fn new(params: Self::Params) -> Self {
        Self { lr: params }
    }

    fn update<P, G>(&self, params: &mut P, grads: &G)
    where
        P: Add<G, Output = P> + Clone,
        G: Mul<f32, Output = G> + Clone,
    {
        *params = params.clone() + grads.clone() * (-self.lr.value());
    }
}
