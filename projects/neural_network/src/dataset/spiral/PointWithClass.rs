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

struct Class(usize);

impl Class {
    fn new(class: usize) -> Self {
        Self(class)
    }
}

struct PointWithClass {
    point: Point,
    class: Class,
}

impl PointWithClass {
    fn new(x: f64, y: f64, class: usize) -> Self {
        Self {
            point: Point::new(x, y),
            class: Class::new(class),
        }
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
                    + rand::random::<f64>();
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let point_with_class = PointWithClass::new(x, y, class);
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
                .filter(|p| p.class.0 == class)
                .map(|p| (p.point.x, p.point.y))
                .collect();
            points.push(points_of_class);
        }
        points
    }
}
