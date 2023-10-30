mod point_with_class;

use std::{collections::HashMap, f64::consts::PI};

use rand::seq::SliceRandom;

use crate::dataset::dataset::{Dataset, MiniBatch};

use self::point_with_class::PointWithClass;

pub struct SpiralDataset {
    points: Vec<PointWithClass>,
    cursor: usize,
    batch_size: usize,
}

pub struct InitParamsOfSpiralDataset {
    pub batch_size: usize,
    pub number_of_class: usize,
    pub point_per_class: usize,
    pub max_angle: f64,
}

impl SpiralDataset {
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
                let radius = (i as f64) / (point_per_class as f64);
                let angle = (i as f64) / (point_per_class as f64) * max_angle
                    + (class as f64 / number_of_class as f64) * 2.0 * PI
                    + rand::random::<f64>() * 1.;
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
        }
    }

    pub fn get_points(&self) -> HashMap<usize, Vec<(f64, f64)>> {
        let mut map = HashMap::new();
        for point in &self.points {
            let class = point.get_class();
            map.entry(class).or_insert(Vec::new()).push(point.get_xy());
        }
        map
    }
}

impl Dataset for SpiralDataset {
    fn shuffle_and_reset_cursor(&mut self) {
        self.points.shuffle(&mut rand::thread_rng());
        self.cursor = 0;
    }

    fn test_data(&self) -> MiniBatch {
        MiniBatch::from_points(&self.points)
    }
}

impl Iterator for SpiralDataset {
    type Item = MiniBatch;

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
