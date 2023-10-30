use ndarray::Array2;

pub trait Dataset: Iterator<Item = MiniBatch> {
    fn shuffle_and_reset_cursor(&mut self);
    fn test_data(&self) -> MiniBatch;
}

pub struct MiniBatch {
    pub bundled_inputs: Array2<f64>,
    pub bundled_one_hot_labels: Array2<f64>,
}