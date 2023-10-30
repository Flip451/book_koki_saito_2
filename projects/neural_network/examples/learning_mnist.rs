extern crate neural_network;

use neural_network::{
    dataset::mnist::{mini_batch::MiniBatch, mnist_data::MnistData},
    network::simple_network::SimpleNetwork,
    optimizer::{learning_rate::LearningRate, sgd::SGD},
};

const BATCH_SIZE: usize = 100;
const MAX_EPOCH: usize = 17;
const HIDDNE_SIZES: [usize; 1] = [50];
const LEARNING_RATE: f64 = 0.001;

fn main() {
    // 学習用データの読み込み
    let training_data = MnistData::load(
        "data/train-labels-idx1-ubyte",
        "data/train-images-idx3-ubyte",
    );
    // テスト用データの読み込み
    let test_data = MnistData::load("data/t10k-labels-idx1-ubyte", "data/t10k-images-idx3-ubyte");

    // 画像サイズの取得
    let (image_hight, image_width) = (training_data.image_height, training_data.image_width);
    // 入力サイズの取得
    let input_size = image_hight * image_width;
    // 分類数（＝出力数）の取得
    let output_size = training_data.class_number;

    // ニューラルネットワークの初期化
    let mut network = SimpleNetwork::new(
        input_size,
        HIDDNE_SIZES.to_vec(),
        output_size,
    );

    // 学習率の設定
    let lr = LearningRate::new(LEARNING_RATE);
    // 最適化手法の設定（確率的勾配降下法）
    let optimizer = SGD::new(lr);

    let epoch_size = training_data.image_number / BATCH_SIZE;
    for i in 0..MAX_EPOCH {
        for _ in 0..epoch_size {
            let MiniBatch {
                bundled_images,
                bundled_one_hot_labels,
            } = MiniBatch::random_choice(&training_data, BATCH_SIZE);

            network.forward(bundled_images, bundled_one_hot_labels);

            network.backward(1.);
            network.update(&optimizer);
        }
        // 全データでの評価
        let MiniBatch {
            bundled_images,
            bundled_one_hot_labels,
        } = MiniBatch::random_choice(&training_data, training_data.image_number);
        let loss = network.forward(bundled_images, bundled_one_hot_labels);

        // テストデータでの評価
        let MiniBatch {
            bundled_images,
            bundled_one_hot_labels,
        } = MiniBatch::random_choice(&test_data, test_data.image_number);
        let predict = network.predict(bundled_images);
        let correct_number = predict
            .outer_iter()
            .zip(bundled_one_hot_labels.outer_iter())
            .filter(|(predict, one_hot_label)| {
                let predict = predict
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                let one_hot_label = one_hot_label
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                predict == one_hot_label
            })
            .count();
        let accuracy_rate = correct_number as f64 / test_data.image_number as f64;

        println!("epoch: {}, acc_test: {}, loss_test: {}", i, accuracy_rate, loss);
    }
}
