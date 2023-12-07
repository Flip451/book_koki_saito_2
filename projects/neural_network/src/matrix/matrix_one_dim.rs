use std::ops::{Add, Div, Mul, Sub};

use ndarray::Array1;

pub trait MatrixOneDim:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + Clone
    + From<Vec<f32>>
{
    fn max_value(&self) -> f32;
    fn mapv_into<F>(self, f: F) -> Self
    where
        F: FnMut(f32) -> f32;
    fn sum(&self) -> f32;
    fn len(&self) -> usize;
    fn zeros(len: usize) -> Self;
    fn find_index<F>(&self, f: F) -> Option<usize>
    where
        F: FnMut(f32) -> bool;
    fn into_one_hot(self) -> Self;
}

impl MatrixOneDim for Array1<f32> {
    fn max_value(&self) -> f32 {
        *self.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap()
    }

    fn mapv_into<F>(self, f: F) -> Self
    where
        F: FnMut(f32) -> f32,
    {
        self.mapv_into(f)
    }

    fn sum(&self) -> f32 {
        self.sum()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn zeros(len: usize) -> Self {
        Array1::zeros(len)
    }

    fn find_index<F>(&self, mut f: F) -> Option<usize>
    where
        F: FnMut(f32) -> bool,
    {
        self.iter()
            .enumerate()
            .find(|&(_index, &v)| (f)(v))
            .map(|(index, _)| index)
    }

    fn into_one_hot(self) -> Self {
        let max_value = self.max_value();
        let mut one_hot = vec![0.; self.len()];
        let index = self
            .iter()
            .enumerate()
            .find(|&(_index, &v)| v == max_value)
            .map(|(index, _)| index)
            .unwrap();
        one_hot[index] = 1.;
        Self::from(one_hot)
    }
}
