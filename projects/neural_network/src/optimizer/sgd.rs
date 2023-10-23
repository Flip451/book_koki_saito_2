use ndarray::{ArrayView, ArrayViewMut, Dimension};

use super::optimizer::{LearningRate, Optimizer};

struct SGD {
    lr: LearningRate,
}

impl Optimizer for SGD {
    fn new(lr: LearningRate) -> Self {
        Self { lr }
    }

    fn update<D>(self, params: Vec<ArrayViewMut<f64, D>>, grads: Vec<ArrayView<f64, D>>)
    where
        D: Dimension,
    {
        params
            .into_iter()
            .zip(grads)
            .for_each(|(mut param_mat, grad_mat)| {
                let new = param_mat.to_owned() - grad_mat.to_owned() * self.lr.value();
                param_mat.assign(&new);
            })
    }
}
