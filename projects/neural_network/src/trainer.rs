use anyhow::Result;
use plotters::prelude::*;
use std::{iter::zip, time::Instant};

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

        // 学習開始直前の正解率の計算
        let accuracy_rate = self.accuracy(dataset);
        self.acc_list.push(accuracy_rate);

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
                        epoch + 1, iters, max_iter, elapsed_time, loss_avg
                    );

                    // 損失の履歴の更新
                    self.loss_list.push(loss_avg);

                    // 損失の合計値の初期化
                    total_loss = 0.;
                    loss_count = 0;
                }
            }

            // 正解率の計算
            let accuracy_rate = self.accuracy(dataset);
            self.acc_list.push(accuracy_rate);

            // 正解率の表示
            println!("| epoch {:5} | acc {:5.5}", epoch + 1, accuracy_rate);
        }
    }

    fn accuracy<D: Dataset>(&mut self, dataset: &D) -> f64 {
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

        accuracy_rate
    }

    pub fn plot_accuracy(&self, out_path: &'static str) -> Result<()> {
        let max_x: f64 = self.acc_list.len() as f64;
        let max_y: f64 = 1.;
        let x = (0..max_x as usize).map(|x| x as f64);

        // 背景の作成
        let root = BitMapBackend::new(out_path, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        // グラフの描画範囲の設定
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(70)
            .y_label_area_size(80)
            .margin(20)
            .caption("Accuracy", ("Arial", 40.0).into_font())
            .build_cartesian_2d(0_f64..max_x, 0_f64..max_y)?;

        // グラフのx軸、y軸の設定
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("epoch")
            .y_desc("accuracy")
            .x_label_formatter(&|x| format!("{:2.0}", x))
            .y_label_formatter(&|x| format!("{:2.1}", x))
            .axis_desc_style(FontDesc::new(FontFamily::SansSerif, 32., FontStyle::Normal))
            .label_style(FontDesc::new(FontFamily::SansSerif, 24., FontStyle::Normal))
            .draw()?;

        // データの描画設定
        chart.draw_series(LineSeries::new(
            zip(x, self.acc_list.iter()).map(|(x, y)| (x, *y)),
            &BLUE,
        ))?;

        // グラフの描画
        root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
        println!("Result has been saved to {}", out_path);
        Ok(())
    }

    pub fn plot_loss(&self, out_path: &'static str) -> Result<()> {
        let max_x: f64 = self.loss_list.len() as f64;
        let max_y: f64 = *self.loss_list.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap();
        let x = (0..max_x as usize).map(|x| x as f64);

        // 背景の作成
        let root = BitMapBackend::new(out_path, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        // グラフの描画範囲の設定
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(70)
            .y_label_area_size(80)
            .margin(20)
            .caption("Accuracy", ("Arial", 40.0).into_font())
            .build_cartesian_2d(0_f64..max_x, 0_f64..max_y)?;

        // グラフのx軸、y軸の設定
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("iteration x eval_interval")
            .y_desc("accuracy")
            .x_label_formatter(&|x| format!("{:2.0}", x))
            .y_label_formatter(&|x| format!("{:2.1}", x))
            .axis_desc_style(FontDesc::new(FontFamily::SansSerif, 32., FontStyle::Normal))
            .label_style(FontDesc::new(FontFamily::SansSerif, 24., FontStyle::Normal))
            .draw()?;

        // データの描画設定
        chart.draw_series(LineSeries::new(
            zip(x, self.loss_list.iter()).map(|(x, y)| (x, *y)),
            &BLUE,
        ))?;

        // グラフの描画
        root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
        println!("Result has been saved to {}", out_path);
        Ok(())
    }
}
