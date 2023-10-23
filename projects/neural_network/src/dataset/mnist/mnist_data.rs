use std::{fs::File, io::Read};

use crate::dataset::mnist::mnist_image::MnistImage;

pub struct MnistData {
    pub image_number: usize,
    pub image_height: usize,
    pub image_width: usize,
    pub(super) images: Vec<MnistImage>,
}

impl MnistData {
    pub fn load(label_file: &str, image_file: &str) -> Self {
        let mut f_label = File::open(label_file).unwrap();
        let mut f_images = File::open(image_file).unwrap();

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

        let mut images = Vec::<MnistImage>::with_capacity(number_of_images as usize);

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

            images.push(MnistImage::new(
                labels[i],
                &pixels[offset..(offset + number_of_rows * number_of_columns)],
            ));
        }

        // 読み損ねたデータがないかをチェック
        let n = f_label.read(&mut buf).unwrap();
        assert_eq!(n, 0);
        let n = f_images.read(&mut buf).unwrap();
        assert_eq!(n, 0);

        Self {
            image_number: number_of_images,
            image_width: number_of_rows,
            image_height: number_of_columns,
            images,
        }
    }
}
