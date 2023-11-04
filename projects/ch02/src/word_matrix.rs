use ndarray::ArrayView2;

pub(crate) mod co_matrix;

pub trait WordMatrix {
    fn view(&self) -> ArrayView2<f32>;
}
