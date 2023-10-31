use std::ops::{Add, Mul};

pub trait Optimizer {
    type Params;

    fn new(params: Self::Params) -> Self;
    fn update<P, G>(&self, params: &mut P, grads: &G)
    where
        P: Add<G, Output = P> + Clone,
        G: Mul<f32, Output = G> + Clone;
}
