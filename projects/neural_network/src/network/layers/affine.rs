use std::ops::{Sub, Mul, SubAssign};

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
    pub(crate) w: Array2<f64>,
    pub(crate) b: Array1<f64>,
}

impl Sub for ParamsOfAffineLayer {
    type Output = ParamsOfAffineLayer;

    fn sub(self, rhs: Self) -> Self::Output {
        ParamsOfAffineLayer {
            w: self.w - rhs.w,
            b: self.b - rhs.b,
        }
    }
}

impl SubAssign for ParamsOfAffineLayer {
    fn sub_assign(&mut self, rhs: Self) {
        self.w = self.w.clone() - rhs.w;
        self.b = self.b.clone() - rhs.b;
    }
}

impl Mul<f64> for ParamsOfAffineLayer {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
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
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.affine
            .forward(InputOfAffineLayer {
                x: input,
                a: self.params.w.clone(),
                b: self.params.b.clone(),
            })
            .into()
    }

    fn backward(&mut self, dout: Array2<f64>) -> Array2<f64> {
        let DInputOfAffineLayer { dx, da, db } =
            self.affine.backward(OutputOfAffineLayer::from(dout));
        self.grads.w = da;
        self.grads.b = db;
        dx
    }
}
