use ndarray::Array1;

pub(super) struct OneHotLabel(Array1<f32>);

impl OneHotLabel {
    pub(super) fn new(class: usize, number_of_class: usize) -> Self {
        assert!(class < number_of_class);
        let mut label = vec![0.; number_of_class];
        label[class] = 1.;
        Self(Array1::from(label))
    }

    pub(super) fn get_array(&self) -> Array1<f32> {
        self.0.clone()
    }
}