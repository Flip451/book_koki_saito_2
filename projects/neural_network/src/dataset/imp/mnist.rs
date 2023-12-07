mod image_with_class;
use std::marker::PhantomData;

use rand::{seq::SliceRandom, thread_rng};

use crate::{
    dataset::dataset::{Dataset, MiniBatch},
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use self::image_with_class::ImageWithClass;

pub struct MnistDataset<M2, M1> {
    train_images: Vec<ImageWithClass<M2, M1>>,
    test_images: Vec<ImageWithClass<M2, M1>>,
    cursor: usize,
    batch_size: usize,
    phantom: PhantomData<M2>,
}

pub struct InitParamsOfMnistDataset {
    pub batch_size: usize,
    pub train_image_file_path: &'static str,
    pub train_label_file_path: &'static str,
    pub test_image_file_path: &'static str,
    pub test_label_file_path: &'static str,
}

impl<M2, M1> MnistDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    pub fn new(params: InitParamsOfMnistDataset) -> Self {
        let InitParamsOfMnistDataset {
            batch_size,
            train_image_file_path,
            train_label_file_path,
            test_image_file_path,
            test_label_file_path,
        } = params;

        let train_images =
            ImageWithClass::load_from_files(train_image_file_path, train_label_file_path);
        let test_images =
            ImageWithClass::load_from_files(test_image_file_path, test_label_file_path);

        Self {
            train_images,
            test_images,
            cursor: 0,
            batch_size,
            phantom: PhantomData,
        }
    }
}

impl<M2, M1> Dataset<M2, M1> for MnistDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn shuffle_and_reset_cursor(&mut self) {
        self.train_images.shuffle(&mut thread_rng());
        self.cursor = 0;
    }

    fn test_data(&self) -> MiniBatch<M2, M1> {
        MiniBatch::from_images(&self.test_images)
    }
}

impl<M2, M1> Iterator for MnistDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Item = MiniBatch<M2, M1>;

    fn next(&mut self) -> Option<Self::Item> {
        let number_of_points = self.train_images.len();
        let rest = number_of_points - self.cursor;
        if rest < self.batch_size {
            None
        } else {
            let mini_batch = MiniBatch::from_images(
                &self.train_images[self.cursor..(self.cursor + self.batch_size)],
            );
            self.cursor += self.batch_size;
            Some(mini_batch)
        }
    }
}

impl<M2, M1> ExactSizeIterator for MnistDataset<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn len(&self) -> usize {
        self.train_images.len() / self.batch_size
    }
}
