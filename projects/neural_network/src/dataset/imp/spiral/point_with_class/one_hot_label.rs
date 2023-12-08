use crate::matrix::matrix_one_dim::MatrixOneDim;

pub(super) struct OneHotLabel<M1>(M1);

impl<M1> OneHotLabel<M1>
where
    M1: MatrixOneDim,
{
    pub(super) fn new(class: usize, number_of_class: usize) -> Self {
        assert!(class < number_of_class);
        let mut label = vec![0.; number_of_class];
        label[class] = 1.;
        Self(M1::from(label))
    }

    pub(super) fn get_array(&self) -> M1 {
        self.0.clone()
    }

    pub(super) fn get_class(&self) -> usize {
        self.0.find_index(|element| element == 1.).unwrap()
    }
}
