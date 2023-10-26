extern crate neural_network;

use neural_network::{
    dataset::spiral::{
        mini_batch::{MiniBatch, MiniBatchGetter},
        point_with_class::{ParamsForNewSeriesOfPointWithClass, SeriesOfPointWithClass},
    },
    network::simple_network::SimpleNetwork,
    optimizer::{optimizer::LearningRate, sgd::SGD},
};

const BATCH_SIZE: usize = 30;
const MAX_EPOCH: usize = 300;
const HIDDNE_SIZES: [usize; 1] = [10];
const LEARNING_RATE: f64 = 1.;

fn main() {
    // 学習用データの作成
    let config = ParamsForNewSeriesOfPointWithClass {
        point_per_class: 100,
        number_of_class: 3,
        max_angle: 1.5 * std::f64::consts::PI,
    };
    let training_data = SeriesOfPointWithClass::new(config);
    let mut minibatch_getter = MiniBatchGetter::new(training_data, BATCH_SIZE);

    // 入力サイズの取得
    let input_size = 2;
    // 分類数（＝出力数）の取得
    let output_size = 3;

    // ニューラルネットワークの初期化
    let mut network = SimpleNetwork::new(input_size, HIDDNE_SIZES.to_vec(), output_size);

    // 学習率の設定
    let lr = LearningRate::new(LEARNING_RATE);
    // 最適化手法の設定（確率的勾配降下法）
    let optimizer = SGD::new(lr);

    for i in 0..MAX_EPOCH {
        minibatch_getter.shuffle_and_reset_cursor();

        while let Some(MiniBatch {
            bundled_points,
            bundled_one_hot_labels,
        }) = minibatch_getter.next()
        {
            network.forward(bundled_points, bundled_one_hot_labels);

            network.backward(1.);
            network.update(&optimizer);
        }
        // 全データでの評価
        let MiniBatch {
            bundled_one_hot_labels,
            bundled_points,
        } = minibatch_getter.whole_data();

        // 予測の実行
        let predict = network.predict(bundled_points);

        // 正解数の計算
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
                    .find(|&(_, &x)| 1. == x)
                    .unwrap()
                    .0;
                predict == one_hot_label
            })
            .count();

        // 正解率の計算
        let accuracy_rate = correct_number as f64 / (300) as f64;

        println!("epoch: {}, acc: {}", i, accuracy_rate);
    }
}
