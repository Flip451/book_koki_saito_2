use std::ops::{Add, Mul};

use ndarray::{Array1, Array2};

use crate::layers::{
    affine::{Affine, DInputOfAffineLayer, InputOfAffineLayer, OutputOfAffineLayer},
    layer::Layer,
};

use super::layer::{LayerBase, TransformLayer};

pub(crate) struct AffineLayer {
    affine: Affine,
    params: ParamsOfAffineLayer,
    grads: ParamsOfAffineLayer,
}

#[derive(Clone)]
pub struct ParamsOfAffineLayer {
    pub(crate) w: Array2<f32>,
    pub(crate) b: Array1<f32>,
}

impl Add for ParamsOfAffineLayer {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ParamsOfAffineLayer {
            w: self.w + rhs.w,
            b: self.b + rhs.b,
        }
    }
}

impl Mul<f32> for ParamsOfAffineLayer {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        ParamsOfAffineLayer {
            w: self.w * rhs,
            b: self.b * rhs,
        }
    }
}

impl LayerBase for AffineLayer {
    type Params = ParamsOfAffineLayer;

    fn new(params: Self::Params) -> Self {
        let affine = Affine::new();
        let grads = ParamsOfAffineLayer {
            w: Array2::zeros(params.w.dim()),
            b: Array1::zeros(params.b.dim()),
        };
        Self {
            affine,
            params,
            grads,
        }
    }

    fn params_and_grads(&mut self) -> (&mut Self::Params, &Self::Params) {
        (&mut self.params, &self.grads)
    }
}

impl TransformLayer for AffineLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        self.affine
            .forward(InputOfAffineLayer {
                x: input,
                a: self.params.w.clone(),
                b: self.params.b.clone(),
            })
            .into()
    }

    fn backward(&mut self, dout: Array2<f32>) -> Array2<f32> {
        let DInputOfAffineLayer { dx, da, db } =
            self.affine.backward(OutputOfAffineLayer::from(dout));
        self.grads.w = da;
        self.grads.b = db;
        dx
    }
}
