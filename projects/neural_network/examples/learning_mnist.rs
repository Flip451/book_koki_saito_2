extern crate neural_network;

use neural_network::{
    dataset::{
        dataset::{Dataset, MiniBatch},
        imp::mnist::{InitParamsOfMnistDataset, MnistDataset},
    },
    network::{
        network::Network,
        simple_network::{Activation, SimpleNetwork},
    },
    optimizer::{
        imp::sgd::{learning_rate::LearningRate, SGD},
        optimizer::Optimizer,
    },
};

const BATCH_SIZE: usize = 100;
const MAX_EPOCH: usize = 17;
const HIDDNE_SIZES: [usize; 1] = [50];
const LEARNING_RATE: f64 = 0.001;

fn main() {
    // 学習用データの読み込み
    let params = InitParamsOfMnistDataset {
        batch_size: BATCH_SIZE,
        train_image_file_path: "data/train-images-idx3-ubyte",
        train_label_file_path: "data/train-labels-idx1-ubyte",
        test_image_file_path: "data/t10k-images-idx3-ubyte",
        test_label_file_path: "data/t10k-labels-idx1-ubyte",
    };
    let mut mnist_dataset = MnistDataset::new(params);

    // 入力サイズの取得
    let input_size = 28 * 28;
    // 分類数（＝出力数）の取得
    let output_size = 10;

    // ニューラルネットワークの初期化
    let mut network = SimpleNetwork::new(
        input_size,
        HIDDNE_SIZES.to_vec(),
        output_size,
        Activation::ReLU,
    );

    // 最適化手法の設定（確率的勾配降下法）
    let lr = LearningRate::new(LEARNING_RATE);
    let optimizer = SGD::new(lr);

    for i in 0..MAX_EPOCH {
        mnist_dataset.shuffle_and_reset_cursor();

        for MiniBatch {
            bundled_inputs,
            bundled_one_hot_labels,
        } in &mut mnist_dataset
        {
            network.forward(bundled_inputs, bundled_one_hot_labels);
            network.backward(1.);
            network.update(&optimizer);
        }

        // 全データでの評価
        let MiniBatch {
            bundled_one_hot_labels,
            bundled_inputs,
        } = mnist_dataset.test_data();

        // テストデータのデータ数の取得
        let n = bundled_inputs.dim().0;

        // テストデータでの評価
        let predict = network.predict(bundled_inputs);

        // 正解数の計算
        let correct_number = predict
            .outer_iter()
            .zip(bundled_one_hot_labels.outer_iter())
            .filter(|(predict, one_hot_label)| {
                let predict = predict
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap();
                one_hot_label[predict] == 1.
            })
            .count();
        let accuracy_rate = correct_number as f64 / n as f64;

        println!("epoch: {}, acc: {}", i, accuracy_rate);
    }
}
