use ndarray::Array1;
use ndarray_rand::rand_distr::num_traits::One;

pub(super) struct OneHotLabel(Array1<f64>);

impl OneHotLabel {
    pub(super) fn new(class: usize, number_of_class: usize) -> Self {
        assert!(class < number_of_class);
        let mut label = vec![0.; number_of_class];
        label[class] = 1.;
        Self(Array1::from(label))
    }

    pub(super) fn get_array(&self) -> Array1<f64> {
        self.0.clone()
    }

    pub(super) fn get_class(&self) -> usize {
        self.0
            .iter()
            .enumerate()
            .find(|(_index, class)| class.is_one())
            .map(|(index, _)| index)
            .unwrap()
    }
}
