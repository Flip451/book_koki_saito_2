use ndarray::{Array2, Array1};
use rand::thread_rng;

use super::{mnist_image::{Image, OneHotLabel}, mnist_data::MnistData};

pub struct MiniBatch {
    pub bundled_one_hot_labels: Array2<f64>,
    pub bundled_images: Array2<f64>,
}

impl MiniBatch {
    // ランダムに batch_size 個の画像を選択
    pub fn random_choice(mnist_data: &MnistData, batch_size: usize) -> Self {
        // ランダムに画像のインデックスを選択
        // <https://stackoverflow.com/questions/69609250/how-to-choose-several-random-numbers-from-an-interval> を参考に実装
        let mut rng = thread_rng();
        let batch_ids = rand::seq::index::sample(&mut rng, mnist_data.image_number, batch_size);

        let (images, one_hots): (Vec<&Image>, Vec<&OneHotLabel>) = batch_ids
            .iter()
            .map(|batch_id| {
                (
                    &mnist_data.images[batch_id].image,
                    &mnist_data.images[batch_id].one_hot_label,
                )
            })
            .unzip();

        // 各画像（Array1）を束ねて、Array2 を作成
        let bundled_one_hot_labels = bundle_1d_arrays_into_2d_array(one_hots);
        let bundled_images = bundle_1d_arrays_into_2d_array(images);

        Self {
            bundled_one_hot_labels,
            bundled_images,
        }
    }
}


fn bundle_1d_arrays_into_2d_array<T>(arrays: Vec<&Array1<T>>) -> Array2<T>
where
    T: Clone,
{
    // ソースに１つ以上の１次元配列が含まれることを要請
    let width = arrays.len();
    assert!(width > 0);

    // すべての1次元配列の長さが等しいことを要請
    let height = arrays[0].len();
    assert!(arrays.iter().all(|array| { array.len() == height }));

    // １次元配列をすべて連結
    let flattened: Array1<T> = arrays.into_iter().flat_map(|row| row.to_vec()).collect();

    // 連結した１次元配列を２次元配列に変換
    let output = flattened.into_shape((width, height)).unwrap();
    output
}
