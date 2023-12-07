use std::ops::{Add, Div, Mul, Sub};

use ndarray::{Array1, Array2, ArrayView1, Axis, ShapeBuilder, Zip};
use ndarray_rand::{rand_distr::Normal, RandomExt};

use super::matrix_one_dim::MatrixOneDim;

pub trait MatrixTwoDim<M1>:
    Add<Output = Self>
    + Add<M1, Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + Clone
where
    M1: MatrixOneDim,
{
    fn dot(&self, rhs: &Self) -> Self;
    fn t(&mut self) -> Self;
    fn sum_axis_zero(&self) -> M1;
    fn sum(&self) -> f32 {
        self.sum_axis_zero().sum()
    }
    fn mapv_into<F>(self, f: F) -> Self
    where
        F: FnMut(f32) -> f32;
    fn ones_like(&self) -> Self;
    fn zeros_like(&self) -> Self;
    fn dim(&self) -> (usize, usize);
    fn mapv_into_for_each_rows<F>(self, f: F) -> Self
    where
        F: FnMut(M1) -> M1;
    fn from_1d_arrays(arrays: Vec<M1>) -> Self;
    fn broadcast_1d_array(array: M1, shape: (usize, usize)) -> Self;
    fn zip_with<F>(&self, rhs: &Self, f: F) -> Self
    where
        F: FnMut(&f32, &f32) -> f32;
    fn random_normal(dim: (usize, usize), mean: f32, std_dev: f32) -> Self;
}

impl MatrixTwoDim<Array1<f32>> for Array2<f32> {
    fn dot(&self, rhs: &Self) -> Self {
        self.dot(rhs)
    }

    fn t(&mut self) -> Self {
        self.t()
    }

    fn sum_axis_zero(&self) -> Array1<f32> {
        self.sum_axis(Axis(0))
    }

    fn mapv_into<F>(self, f: F) -> Self
    where
        F: FnMut(f32) -> f32,
    {
        self.mapv_into(f)
    }

    fn ones_like(&self) -> Self {
        Array2::ones(self.dim())
    }

    fn zeros_like(&self) -> Self {
        Array2::zeros(self.dim())
    }

    fn dim(&self) -> (usize, usize) {
        self.dim()
    }

    fn mapv_into_for_each_rows<F>(mut self, mut f: F) -> Self
    where
        F: FnMut(Array1<f32>) -> Array1<f32>,
    {
        let (height, width) = self.dim();
        let flattened: Array1<f32> = self
            .rows()
            .into_iter()
            .flat_map(|row| (f)(row.to_owned()))
            .collect();
        flattened.into_shape((height, width)).unwrap()
    }

    fn from_1d_arrays(arrays: Vec<Array1<f32>>) -> Self {
        // ソースに１つ以上の１次元配列が含まれることを要請
        let width = arrays.len();
        assert!(width > 0);

        // すべての1次元配列の長さが等しいことを要請
        let height = arrays[0].len();
        assert!(arrays.iter().all(|array| { array.len() == height }));

        // １次元配列をすべて連結
        let flattened: Array1<f32> = arrays.into_iter().flat_map(|row| row.to_vec()).collect();

        // 連結した１次元配列を２次元配列に変換
        let output = flattened.into_shape((width, height)).unwrap();
        output
    }

    fn broadcast_1d_array(array: Array1<f32>, shape: (usize, usize)) -> Self {
        array.broadcast(shape).unwrap().to_owned()
    }

    fn zip_with<F>(&self, rhs: &Self, mut f: F) -> Self
    where
        F: FnMut(&f32, &f32) -> f32,
    {
        let flattened: Array1<f32> = self.into_iter().zip(rhs).map(|(s, r)| (f)(s, r)).collect();
        flattened.into_shape(self.dim()).unwrap()
    }

    fn random_normal(dim: (usize, usize), mean: f32, std_dev: f32) -> Self {
        Array2::random(dim, Normal::new(mean, std_dev).unwrap())
    }
}
