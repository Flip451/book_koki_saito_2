use std::{fs::File, io::Read, marker::PhantomData};

mod one_hot_label;
use crate::{
    dataset::dataset::MiniBatch,
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
};

use self::one_hot_label::OneHotLabel;

pub(super) struct ImageWithClass<M2, M1> {
    image: M1,
    class: OneHotLabel<M1>,
    ph: PhantomData<M2>,
}

impl<M2, M1> ImageWithClass<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn new(image_pixels: &[u8], label: u8) -> Self {
        let image_pixels = image_pixels.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        let class = OneHotLabel::new(label as usize, 10);
        let image = M1::from(image_pixels).mapv_into(|x| x as f32);
        Self { image, class, ph: PhantomData }
    }

    pub(super) fn load_from_files(
        image_file_path: &'static str,
        label_file_path: &'static str,
    ) -> Vec<Self> {
        let mut f_label = File::open(label_file_path).unwrap();
        let mut f_images = File::open(image_file_path).unwrap();

        let mut buf: [u8; 4] = [0; 4];

        // ラベルデータのマジックナンバーの取得
        f_label
            .read(&mut buf)
            .expect("Error: cannot read magic number of label data.");
        let magic_number = u32::from_be_bytes(buf);
        print!("Magic Number: {}, ", magic_number);
        assert_eq!(magic_number, 2049);

        // ラベルデータから要素数の取得
        f_label
            .read(&mut buf)
            .expect("Error: cannot read number of items in label data.");
        let number_of_items = u32::from_be_bytes(buf) as usize;
        println!("Number of Items: {}, ", number_of_items);

        // マジックナンバーの取得
        f_images
            .read(&mut buf)
            .expect("Error: cannot read magic number of image data.");
        let magic_number = u32::from_be_bytes(buf);
        print!("Magic Number: {}, ", magic_number);
        assert_eq!(magic_number, 2051);

        // 画像数の取得
        f_images
            .read(&mut buf)
            .expect("Error: cannot read number of image from image data.");
        let number_of_images = u32::from_be_bytes(buf) as usize;
        print!("Number of Images: {}, ", number_of_images);
        assert_eq!(number_of_items, number_of_images);

        // 画像の幅（px）の取得
        f_images
            .read(&mut buf)
            .expect("Error: cannot read width of images.");
        let number_of_rows = u32::from_be_bytes(buf) as usize;
        print!("Image Width: {}, ", number_of_rows);

        // 画像の高さ（px）の取得
        f_images
            .read(&mut buf)
            .expect("Error: cannot read height of images.");
        let number_of_columns = u32::from_be_bytes(buf) as usize;
        println!("Image Height: {}", number_of_columns);

        let mut images = Vec::<ImageWithClass<M2, M1>>::with_capacity(number_of_images as usize);

        // ラベルの読み出し
        let mut labels: Vec<u8> = Vec::with_capacity(number_of_images);
        let n = f_label
            .read_to_end(&mut labels)
            .expect("Error: cannot read labels");
        assert_eq!(n, number_of_images);

        // pixel の読み出し
        let mut pixels: Vec<u8> =
            Vec::with_capacity(number_of_rows * number_of_columns * number_of_images);
        let n = f_images
            .read_to_end(&mut pixels)
            .expect("Error: cannot read pixels");
        assert_eq!(n, number_of_rows * number_of_columns * number_of_images);

        for i in 0..number_of_images {
            let offset = i * number_of_rows * number_of_columns;

            images.push(ImageWithClass::new(
                &pixels[offset..(offset + number_of_rows * number_of_columns)],
                labels[i],
            ));
        }

        // 読み損ねたデータがないかをチェック
        let n = f_label.read(&mut buf).unwrap();
        assert_eq!(n, 0);
        let n = f_images.read(&mut buf).unwrap();
        assert_eq!(n, 0);

        images
    }
}

impl<M2, M1> MiniBatch<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    pub(super) fn from_images(images: &[ImageWithClass<M2, M1>]) -> Self {
        let bundled_points: Vec<M1> = images
            .iter()
            .map(|image_with_class| image_with_class.image.clone())
            .collect();
        let bundled_inputs = M2::from_1d_arrays(bundled_points);

        let bundled_one_hot_labels: Vec<M1> = images
            .iter()
            .map(|image_with_class| image_with_class.class.get_array())
            .collect();
        let bundled_one_hot_labels = M2::from_1d_arrays(bundled_one_hot_labels);

        Self {
            bundled_inputs,
            bundled_one_hot_labels,
            ph: PhantomData,
        }
    }
}
