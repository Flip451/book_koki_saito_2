mod image_with_class;
use rand::{thread_rng, seq::SliceRandom};

use crate::dataset::dataset::{Dataset, MiniBatch};

use self::image_with_class::ImageWithClass;

pub struct MnistDataset {
    train_images: Vec<ImageWithClass>,
    test_images: Vec<ImageWithClass>,
    cursor: usize,
    batch_size: usize,
}

pub struct InitParamsOfMnistDataset {
    pub batch_size: usize,
    pub train_image_file_path: &'static str,
    pub train_label_file_path: &'static str,
    pub test_image_file_path: &'static str,
    pub test_label_file_path: &'static str,
}

impl MnistDataset {
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
        }
    }
}

impl Dataset for MnistDataset {
    fn shuffle_and_reset_cursor(&mut self) {
        self.train_images.shuffle(&mut thread_rng());
        self.cursor = 0;
    }

    fn test_data(&self) -> MiniBatch {
        MiniBatch::from_images(&self.test_images)
    }
}

impl Iterator for MnistDataset {
    type Item = MiniBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let number_of_points = self.train_images.len();
        let rest = number_of_points - self.cursor;
        if rest < self.batch_size {
            None
        } else {
            let mini_batch =
                MiniBatch::from_images(&self.train_images[self.cursor..(self.cursor + self.batch_size)]);
            self.cursor += self.batch_size;
            Some(mini_batch)
        }
    }
}

impl ExactSizeIterator for MnistDataset {
    fn len(&self) -> usize {
        self.train_images.len() / self.batch_size
    }
}