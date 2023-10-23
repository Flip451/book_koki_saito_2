use ndarray::Array1;

pub(super) struct MnistImage {
    pub(super) one_hot_label: OneHotLabel,
    pub(super) image: Image,
}

impl MnistImage {
    pub(super) fn new(label: u8, pixels: &[u8]) -> Self {
        let mut one_hot_label = Array1::zeros(10);
        one_hot_label[label as usize] = 1.0;
        let image = Array1::from(pixels.to_vec()).mapv(|x| x as f64);
        Self {
            one_hot_label,
            image,
        }
    }
}

pub(super) type Pixel = f64;
pub(super) type Image = Array1<Pixel>;
pub(super) type OneHotLabel = Array1<f64>;
