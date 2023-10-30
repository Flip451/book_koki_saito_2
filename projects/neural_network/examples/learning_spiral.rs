extern crate neural_network;

use neural_network::{
    dataset::{
        dataset::{Dataset, MiniBatch},
        imp::spiral::{InitParamsOfSpiralDataset, SpiralDataset},
    },
    network::{network::Network, simple_network::SimpleNetwork},
    optimizer::{
        imp::sgd::{learning_rate::LearningRate, SGD},
        optimizer::Optimizer,
    },
};

const BATCH_SIZE: usize = 30;
const MAX_EPOCH: usize = 300;
const HIDDNE_SIZES: [usize; 1] = [10];
const LEARNING_RATE: f64 = 1.;

fn main() {
    // 学習用データの作成
    const NUMBER_OF_CLASS: usize = 3;
    const POINT_PER_CLASS: usize = 100;
    let config = InitParamsOfSpiralDataset {
        point_per_class: POINT_PER_CLASS,
        number_of_class: NUMBER_OF_CLASS,
        max_angle: 1.5 * std::f64::consts::PI,
        batch_size: BATCH_SIZE,
    };
    let mut spiral_dataset = SpiralDataset::new(config);

    // 入力サイズの取得
    let input_size = 2;

    // 分類数（＝出力数）の取得
    let output_size = 3;

    // ニューラルネットワークの初期化
    let mut network = SimpleNetwork::new(input_size, HIDDNE_SIZES.to_vec(), output_size);

    // 最適化手法の設定（確率的勾配降下法）
    let lr = LearningRate::new(LEARNING_RATE);
    let optimizer = SGD::new(lr);

    for i in 0..MAX_EPOCH {
        spiral_dataset.shuffle_and_reset_cursor();

        // 学習の実行
        for MiniBatch {
            bundled_inputs,
            bundled_one_hot_labels,
        } in &mut spiral_dataset
        {
            network.forward(bundled_inputs, bundled_one_hot_labels);
            network.backward(1.);
            network.update(&optimizer);
        }

        // 全データでの評価
        let MiniBatch {
            bundled_one_hot_labels,
            bundled_inputs,
        } = spiral_dataset.test_data();

        // テストデータのデータ数の取得
        let n = bundled_inputs.dim().0;

        // 予測の実行
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

        // 正解率の計算
        let accuracy_rate = correct_number as f64 / n as f64;

        println!("epoch: {}, acc: {}", i, accuracy_rate);
    }
}
