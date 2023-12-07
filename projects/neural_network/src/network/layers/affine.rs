use std::ops::{Add, Mul};

use ndarray::{Array1, Array2};

use crate::{
    layers::{
        affine::{Affine, DInputOfAffineLayer, InputOfAffineLayer, OutputOfAffineLayer},
        layer::Layer,
    },
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use super::layer::{IntermediateLayer, LayerBase};

pub(crate) struct AffineLayer<M2, M1> {
    affine: Affine<M2, M1>,
    params: ParamsOfAffineLayer<M2, M1>,
    grads: ParamsOfAffineLayer<M2, M1>,
}

#[derive(Clone)]
pub struct ParamsOfAffineLayer<M2, M1> {
    pub(crate) w: M2,
    pub(crate) b: M1,
}

impl<M2, M1> Add for ParamsOfAffineLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ParamsOfAffineLayer {
            w: self.w + rhs.w,
            b: self.b + rhs.b,
        }
    }
}

impl<M2, M1> Mul<f32> for ParamsOfAffineLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        ParamsOfAffineLayer {
            w: self.w * rhs,
            b: self.b * rhs,
        }
    }
}

impl<M2, M1> LayerBase for AffineLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Params = ParamsOfAffineLayer<M2, M1>;

    fn new(params: Self::Params) -> Self {
        let affine = Affine::new();
        let grads = ParamsOfAffineLayer {
            w: M2::zeros_like(&params.w),
            b: M1::zeros(params.b.len()),
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

impl<M2, M1> IntermediateLayer<M2, M1> for AffineLayer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn forward(&mut self, input: M2) -> M2 {
        self.affine
            .forward(InputOfAffineLayer {
                x: input,
                a: self.params.w.clone(),
                b: self.params.b.clone(),
            })
            .into_value()
    }

    fn backward(&mut self, dout: M2) -> M2 {
        let DInputOfAffineLayer { dx, da, db } =
            self.affine.backward(OutputOfAffineLayer::from(dout));
        self.grads.w = da;
        self.grads.b = db;
        dx
    }
}
