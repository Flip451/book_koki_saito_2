use ndarray::ArrayView2;

pub mod co_matrix;

pub trait WordMatrix {
    fn view(&self) -> ArrayView2<f32>;
}
