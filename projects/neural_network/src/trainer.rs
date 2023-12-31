use anyhow::Result;
use plotters::prelude::*;
use std::{iter::zip, time::Instant, marker::PhantomData};

use crate::{
    dataset::dataset::{Dataset, MiniBatch},
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
    network::network::Network,
    optimizer::optimizer::Optimizer,
};

pub struct Trainer<Net, Opt, M2, M1>
where
    Net: Network<M2, M1>,
    Opt: Optimizer,
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    network: Net,
    optimizer: Opt,
    loss_list: Vec<f32>,
    acc_list: Vec<f32>,
    eval_interval: Option<usize>,
    phantom: PhantomData<(M2, M1)>,
}

impl<Net, Opt, M2, M1> Trainer<Net, Opt, M2, M1>
where
    Net: Network<M2, M1>,
    Opt: Optimizer,
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    pub fn new(network: Net, optimizer: Opt) -> Self {
        Self {
            network,
            optimizer,
            loss_list: Vec::new(),
            acc_list: Vec::new(),
            eval_interval: None,
            phantom: PhantomData,
        }
    }

    pub fn fit<D: Dataset<M2, M1>>(
        &mut self,
        dataset: &mut D,
        max_epoch: usize,
        // max_grad: Option<f32>,
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
                    ph: _,
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
                    let loss_avg = total_loss / loss_count as f32;
                    // 実行時間の取得
                    let elapsed_time = start_time.elapsed().as_secs_f32();

                    // 時間、イテレーション、損失の表示
                    println!(
                        "| epoch {:5} | iter {:5} / {:5} | time {:.5} [s] | loss {:.5}",
                        epoch + 1,
                        iters,
                        max_iter,
                        elapsed_time,
                        loss_avg
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

    fn accuracy<D: Dataset<M2, M1>>(&mut self, dataset: &D) -> f32 {
        // テストデータでの評価
        let MiniBatch {
            bundled_one_hot_labels,
            bundled_inputs,
            ph: _,
        } = dataset.test_data();

        // テストデータのデータ数の取得
        let n = bundled_inputs.dim().0;

        // 予測の実行
        let predict = self.network.predict(bundled_inputs);

        // 正解数の計算
        let predict = predict.mapv_into_for_each_rows(|predict| {
            predict.into_one_hot()
        });
        let correct_number = (predict * bundled_one_hot_labels).sum();

        // 正解率の計算
        let accuracy_rate = correct_number as f32 / n as f32;

        accuracy_rate
    }

    pub fn plot_accuracy(&self, out_path: &'static str) -> Result<()> {
        Self::plot(&self.acc_list, out_path, "epoch", "accuracy", "Accuracy")
    }

    pub fn plot_loss(&self, out_path: &'static str) -> Result<()> {
        Self::plot(
            &self.loss_list,
            out_path,
            &format!("iteration (x{:?})", self.eval_interval.unwrap()),
            "loss",
            "Loss",
        )
    }

    fn plot(
        list: &Vec<f32>,
        out_path: &str,
        x_label: &str,
        y_label: &str,
        caption: &str,
    ) -> Result<()> {
        let max_x: f32 = list.len() as f32;
        let max_y: f32 = *list.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap();
        let x = (0..max_x as usize).map(|x| x as f32);

        // 背景の作成
        let root = BitMapBackend::new(out_path, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        // グラフの描画範囲の設定
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(70)
            .y_label_area_size(80)
            .margin(20)
            .caption(caption, ("Arial", 40.0).into_font())
            .build_cartesian_2d(0_f32..max_x, 0_f32..max_y)?;

        // グラフのx軸、y軸の設定
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .x_label_formatter(&|x| format!("{:2.0}", x))
            .y_label_formatter(&|x| format!("{:2.1}", x))
            .axis_desc_style(FontDesc::new(FontFamily::SansSerif, 32., FontStyle::Normal))
            .label_style(FontDesc::new(FontFamily::SansSerif, 24., FontStyle::Normal))
            .draw()?;

        // データの描画設定
        chart.draw_series(LineSeries::new(
            zip(x, list.iter()).map(|(x, y)| (x, *y)),
            &BLUE,
        ))?;

        // グラフの描画
        root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
        println!("Result has been saved to {}", out_path);
        Ok(())
    }
}
