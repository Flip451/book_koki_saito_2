use ndarray::{array, Array1};
use rand::prelude::SliceRandom;
use std::f64::consts::PI;

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

struct Class {
    class: usize,
    class_number: usize,
}

impl Class {
    fn new(class: usize, class_number: usize) -> Self {
        assert!(class < class_number);
        Self {
            class,
            class_number,
        }
    }
}

struct OneHotLabel(Vec<f64>);

impl OneHotLabel {
    fn new(class: Class) -> Self {
        let mut label = vec![0.; class.class_number];
        label[class.class] = 1.;
        Self(label)
    }
}

pub(super) struct PointWithClass {
    point: Point,
    one_hot_label: OneHotLabel,
}

impl PointWithClass {
    fn new(x: f64, y: f64, class: usize, class_number: usize) -> Self {
        Self {
            point: Point::new(x, y),
            one_hot_label: OneHotLabel::new(Class::new(class, class_number)),
        }
    }

    pub(super) fn get_xy_array(&self) -> Array1<f64> {
        array![self.point.x, self.point.y]
    }

    pub(super) fn get_one_hot_label_array(&self) -> Array1<f64> {
        Array1::from(self.one_hot_label.0.clone())
    }
}

pub struct SeriesOfPointWithClass {
    number_of_class: usize,
    points: Vec<PointWithClass>,
}

pub struct ParamsForNewSeriesOfPointWithClass {
    pub point_per_class: usize,
    pub number_of_class: usize,
    pub max_angle: f64,
}

impl SeriesOfPointWithClass {
    pub fn new(params: ParamsForNewSeriesOfPointWithClass) -> Self {
        let ParamsForNewSeriesOfPointWithClass {
            point_per_class,
            number_of_class,
            max_angle,
        } = params;
        let mut points = Vec::new();
        for class in 0..params.number_of_class {
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
            number_of_class: params.number_of_class,
            points,
        }
    }

    pub fn get_points(&self) -> Vec<Vec<(f64, f64)>> {
        let mut points = Vec::new();
        for class in 0..self.number_of_class {
            let points_of_class = self
                .points
                .iter()
                .filter(|p| p.one_hot_label.0[class] == 1.)
                .map(|p| (p.point.x, p.point.y))
                .collect();
            points.push(points_of_class);
        }
        points
    }

    pub(super) fn shuffle(&mut self) {
        self.points.shuffle(&mut rand::thread_rng());
    }

    pub(super) fn total_len(&self) -> usize {
        self.points.len()
    }

    pub(super) fn slice(&self, start: usize, len: usize) -> &[PointWithClass] {
        let start = start;
        let end = start + len;
        &self.points[start..end]
    }
}
