mod one_hot_label;
use ndarray::{array, Array1, Array2};

use crate::dataset::dataset::MiniBatch;

use self::one_hot_label::OneHotLabel;

pub(super) struct PointWithClass {
    x: f64,
    y: f64,
    class: OneHotLabel,
}

impl PointWithClass {
    pub(super) fn new(x: f64, y: f64, class: usize, class_number: usize) -> Self {
        Self {
            x,
            y,
            class: OneHotLabel::new(class, class_number),
        }
    }

    pub(super) fn get_class(&self) -> usize {
        self.class.get_class()
    }

    pub(super) fn get_xy(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

impl MiniBatch {
    pub(super) fn from_points(points: &[PointWithClass]) -> Self {
        let bundled_points: Vec<Array1<f64>> = points
            .iter()
            .map(|point| array![point.x, point.y])
            .collect();
        let bundled_inputs = bundle_1d_arrays_into_2d_array(bundled_points);

        let bundled_one_hot_labels: Vec<Array1<f64>> =
            points.iter().map(|point| point.class.get_array()).collect();
        let bundled_one_hot_labels = bundle_1d_arrays_into_2d_array(bundled_one_hot_labels);

        Self {
            bundled_inputs,
            bundled_one_hot_labels,
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
