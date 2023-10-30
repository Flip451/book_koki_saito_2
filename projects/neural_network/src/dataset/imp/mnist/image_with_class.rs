use ndarray::Array1;

mod one_hot_label;
use crate::dataset::dataset::MiniBatch;

use self::one_hot_label::OneHotLabel;

pub(super) struct ImageWithClass {
    image: Array1<f64>,
    class: OneHotLabel,
}

impl ImageWithClass {
    pub(super) fn load_from_files(image_file_path: &'static str, label_file_path: &'static str) -> Vec<Self> {
        todo!()
    }
}

impl MiniBatch {
    pub(super) fn from_images(images: &[ImageWithClass]) -> Self {
        todo!()
    }
}