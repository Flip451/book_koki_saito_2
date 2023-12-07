mod one_hot_label;
use std::marker::PhantomData;

use ndarray::{array, Array1, Array2};

use crate::{
    dataset::dataset::MiniBatch,
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use self::one_hot_label::OneHotLabel;

pub(super) struct PointWithClass<M1> {
    x: f32,
    y: f32,
    class: OneHotLabel<M1>,
}

impl<M1> PointWithClass<M1>
where
    M1: MatrixOneDim,
{
    pub(super) fn new(x: f32, y: f32, class: usize, class_number: usize) -> Self {
        Self {
            x,
            y,
            class: OneHotLabel::new(class, class_number),
        }
    }

    pub(super) fn get_class(&self) -> usize {
        self.class.get_class()
    }

    pub(super) fn get_xy(&self) -> (f32, f32) {
        (self.x, self.y)
    }
}

impl<M2, M1> MiniBatch<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    pub(super) fn from_points(points: &[PointWithClass<M1>]) -> Self {
        let bundled_points: Vec<M1> = points
            .iter()
            .map(|point| M1::from(vec![point.x, point.y]))
            .collect();
        let bundled_inputs = M2::from_1d_arrays(bundled_points);

        let bundled_one_hot_labels: Vec<M1> =
            points.iter().map(|point| point.class.get_array()).collect();
        let bundled_one_hot_labels = M2::from_1d_arrays(bundled_one_hot_labels);

        Self {
            bundled_inputs,
            bundled_one_hot_labels,
            ph: PhantomData,
        }
    }
}
