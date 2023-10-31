use std::time::Instant;

use crate::{
    dataset::dataset::{Dataset, MiniBatch},
    network::network::Network,
    optimizer::optimizer::Optimizer,
};

pub struct Trainer<Net, Opt>
where
    Net: Network,
    Opt: Optimizer,
{
    network: Net,
    optimizer: Opt,
    loss_list: Vec<f64>,
    acc_list: Vec<f64>,
    eval_interval: Option<usize>,
}

impl<Net, Opt> Trainer<Net, Opt>
where
    Net: Network,
    Opt: Optimizer,
{
    pub fn new(network: Net, optimizer: Opt) -> Self {
        Self {
            network,
            optimizer,
            loss_list: Vec::new(),
            acc_list: Vec::new(),
            eval_interval: None,
        }
    }

    pub fn fit<D: Dataset>(
        &mut self,
        dataset: &mut D,
        max_epoch: usize,
        // max_grad: Option<f64>,
        eval_interval: usize,
    ) {
        self.eval_interval = Some(eval_interval);

        // 損失の合計値の初期化
        let mut total_loss = 0.;
        let mut loss_count = 0;

        // １エポック当たりのイテレーション数
        let max_iter = dataset.len();

        // 実行時間の計測開始
        let start_time = Instant::now();

        // 学習の実行
        for epoch in 0..max_epoch {
            // データのシャッフル
            dataset.shuffle_and_reset_cursor();

            // ミニバッチを取得して学習を実行
            for (
                iters,
                MiniBatch {
                    bundled_inputs,
                    bundled_one_hot_labels,
                },
            ) in dataset.enumerate()
            {
                let loss = self.network.forward(bundled_inputs, bundled_one_hot_labels);
                self.network.backward(1.);
                self.network.update(&self.optimizer);

                // 損失の合計値の更新
                total_loss += loss;
                loss_count += 1;

                // 評価
                if loss_count == eval_interval {
                    // 損失の平均値の計算
                    let loss_avg = total_loss / loss_count as f64;
                    // 実行時間の取得
                    let elapsed_time = start_time.elapsed().as_secs_f64();

                    // 時間、イテレーション、損失の表示
                    println!(
                        "| epoch {:5} | iter {:5} / {:5} | time {:.5} [s] | loss {:.5}",
                        epoch, iters, max_iter, elapsed_time, loss_avg
                    );

                    // 損失の履歴の更新
                    self.loss_list.push(loss_avg);

                    // 損失の合計値の初期化
                    total_loss = 0.;
                    loss_count = 0;
                }
            }

            // テストデータでの評価
            let MiniBatch {
                bundled_one_hot_labels,
                bundled_inputs,
            } = dataset.test_data();

            // テストデータのデータ数の取得
            let n = bundled_inputs.dim().0;

            // 予測の実行
            let predict = self.network.predict(bundled_inputs);

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
            self.acc_list.push(accuracy_rate);
            // 正解率の表示
            println!("| epoch {:5} | acc {:5.5}", epoch, accuracy_rate);
        }
    }

    // fn plot(self);
}
