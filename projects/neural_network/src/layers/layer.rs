use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

pub(crate) trait Layer<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input;
    type Output;
    type DInput;

    fn new() -> Self;
    fn forward(&mut self, input: Self::Input) -> Self::Output;
    fn backward(&self, dout: Self::Output) -> Self::DInput;
}
