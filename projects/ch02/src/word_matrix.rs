use ndarray::ArrayView2;

pub(crate) mod co_matrix;

pub trait WordMatrix {
    fn array2(&self) -> ArrayView2<f32>;
}
