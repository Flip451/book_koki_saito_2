use std::marker::PhantomData;

use ndarray::Array2;

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

pub trait Dataset<M2, M1>: ExactSizeIterator<Item = MiniBatch<M2, M1>>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn shuffle_and_reset_cursor(&mut self);
    fn test_data(&self) -> MiniBatch<M2, M1>;
}

pub struct MiniBatch<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    pub bundled_inputs: M2,
    pub bundled_one_hot_labels: M2,
    pub ph: PhantomData<M1>,
}
