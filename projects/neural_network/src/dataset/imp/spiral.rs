mod point_with_class;

use std::{collections::HashMap, f32::consts::PI, marker::PhantomData};

use rand::seq::SliceRandom;

use crate::{
    dataset::dataset::{Dataset, MiniBatch},
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use self::point_with_class::PointWithClass;

pub struct SpiralDataset<M2, M1> {
    points: Vec<PointWithClass<M1>>,
    cursor: usize,
    batch_size: usize,
    phantom: PhantomData<(M2)>,
}

pub struct InitParamsOfSpiralDataset {
    pub batch_size: usize,
    pub number_of_class: usize,
    pub point_per_class: usize,
    pub max_angle: f32,
}

impl<M2, M1> SpiralDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    pub fn new(params: InitParamsOfSpiralDataset) -> Self {
        let InitParamsOfSpiralDataset {
            batch_size,
            number_of_class,
            point_per_class,
            max_angle,
        } = params;

        let mut points = Vec::with_capacity(number_of_class * point_per_class);

        for class in 0..number_of_class {
            for i in 0..point_per_class {
                let radius = (i as f32) / (point_per_class as f32);
                let angle = (i as f32) / (point_per_class as f32) * max_angle
                    + (class as f32 / number_of_class as f32) * 2.0 * PI
                    + rand::random::<f32>() * 1.;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let point_with_class = PointWithClass::new(x, y, class, number_of_class);
                points.push(point_with_class);
            }
        }

        Self {
            points,
            cursor: 0,
            batch_size,
            phantom: PhantomData,
        }
    }

    pub fn get_points(&self) -> HashMap<usize, Vec<(f32, f32)>> {
        let mut map = HashMap::new();
        for point in &self.points {
            let class = point.get_class();
            map.entry(class).or_insert(Vec::new()).push(point.get_xy());
        }
        map
    }
}

impl<M2, M1> Dataset<M2, M1> for SpiralDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn shuffle_and_reset_cursor(&mut self) {
        self.points.shuffle(&mut rand::thread_rng());
        self.cursor = 0;
    }

    fn test_data(&self) -> MiniBatch<M2, M1> {
        MiniBatch::from_points(&self.points)
    }
}

impl<M2, M1> Iterator for SpiralDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Item = MiniBatch<M2, M1>;

    fn next(&mut self) -> Option<Self::Item> {
        let number_of_points = self.points.len();
        let rest = number_of_points - self.cursor;
        if rest < self.batch_size {
            None
        } else {
            let mini_batch =
                MiniBatch::from_points(&self.points[self.cursor..(self.cursor + self.batch_size)]);
            self.cursor += self.batch_size;
            Some(mini_batch)
        }
    }
}

impl<M2, M1> ExactSizeIterator for SpiralDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn len(&self) -> usize {
        self.points.len() / self.batch_size
    }
}
