use crate::network::{layers::affine::ParamsOfAffineLayer, simple_network::Optimizer};

use super::optimizer::LearningRate;

pub struct SGD {
    lr: LearningRate,
}

// impl Optimizer for SGD {
//     fn new(lr: LearningRate) -> Self {
//         Self { lr }
//     }

//     fn update<P, G>(&self, params: Vec<&mut P>, grads: Vec<&G>)
//     where
//         P: SubAssign<G>,
//         G: Mul<f64, Output = G> + Clone,
//     {
//         params.into_iter().zip(grads).for_each(|(param, grad)| {
//             *param -= grad.clone() * self.lr.value();
//         })
//     }
// }

impl SGD {
    pub fn new(lr: LearningRate) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {

    fn update(&self, params: &mut ParamsOfAffineLayer, grads: &ParamsOfAffineLayer) {
        *params -= grads.clone() * self.lr.value();
    }
}
