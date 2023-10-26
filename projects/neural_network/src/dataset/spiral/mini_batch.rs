use ndarray::{Array1, Array2};

use super::point_with_class::{PointWithClass, SeriesOfPointWithClass};

pub struct MiniBatch {
    pub bundled_one_hot_labels: Array2<f64>,
    pub bundled_points: Array2<f64>,
}

impl MiniBatch {
    fn from_points(points: &[PointWithClass]) -> Self {
        let bundled_points: Vec<Array1<f64>> =
            points.iter().map(|point| point.get_xy_array()).collect();
        let bundled_points = bundle_1d_arrays_into_2d_array(bundled_points);

        let bundled_one_hot_labels: Vec<Array1<f64>> = points
            .iter()
            .map(|point| point.get_one_hot_label_array())
            .collect();
        let bundled_one_hot_labels = bundle_1d_arrays_into_2d_array(bundled_one_hot_labels);

        Self {
            bundled_one_hot_labels,
            bundled_points,
        }
    }
}

pub struct MiniBatchGetter {
    points: SeriesOfPointWithClass,
    cursor: usize,
    batch_size: usize,
}

impl MiniBatchGetter {
    pub fn new(points: SeriesOfPointWithClass, batch_size: usize) -> Self {
        Self {
            points,
            cursor: 0,
            batch_size,
        }
    }

    pub fn shuffle_and_reset_cursor(&mut self) {
        self.points.shuffle();
        self.cursor = 0;
    }

    pub fn whole_data(&self) -> MiniBatch {
        MiniBatch::from_points(self.points.slice(0, self.points.total_len()))
    }
}

impl Iterator for MiniBatchGetter {
    type Item = MiniBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let number_of_points = self.points.total_len();
        let rest = number_of_points - self.cursor;
        if rest < self.batch_size {
            None
        } else {
            let points = self.points.slice(self.cursor, self.batch_size);
            self.cursor += self.batch_size;
            Some(MiniBatch::from_points(points))
        }
    }
}

fn bundle_1d_arrays_into_2d_array<T>(arrays: Vec<Array1<T>>) -> Array2<T>
where
    T: Clone,
{
    // ソースに１つ以上の１次元配列が含まれることを要請
    let width = arrays.len();
    assert!(width > 0);

    // すべての1次元配列の長さが等しいことを要請
    let height = arrays[0].len();
    assert!(arrays.iter().all(|array| { array.len() == height }));

    // １次元配列をすべて連結
    let flattened: Array1<T> = arrays.into_iter().flat_map(|row| row.to_vec()).collect();

    // 連結した１次元配列を２次元配列に変換
    let output = flattened.into_shape((width, height)).unwrap();
    output
}
