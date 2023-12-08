/*
    softmax
    X = [x_1, x_2, ..., x_n]
    Y = [y_1, y_2, ..., y_n]
    y_i = exp(x_i) / (Σ_{k} exp(x_i))

    交差エントロピー誤差
    t(j) = [0, ..., 0, 1, 0, ..., 0] (データ j の正解)
    t(j) = [t_1, t_2, ..., t_n]
    y(j) = [y_1, y_2, ..., y_n] (データ j に対する予測)
    L = (1 / |対象データ数|) Σ_{j: 対象データ} Σ_{k} (- t_k * ln(y_k))

    ∂L/∂(x_i) = y_i - t_i
*/

use std::marker::PhantomData;

use crate::matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim};

use super::layer::Layer;

pub(crate) struct SoftmaxCrossEntropy<M2, M1> {
    y: Option<M2>,
    t: Option<M2>,
    ph: PhantomData<M1>,
}

pub(crate) struct InputOfSoftmaxCrossEntropyLayer<M2> {
    input: M2,
    t: M2,
}

impl<M2> InputOfSoftmaxCrossEntropyLayer<M2> {
    pub(crate) fn from(input: M2, one_hot_labels: M2) -> Self {
        Self {
            input,
            t: one_hot_labels,
        }
    }
}

pub(crate) struct DInputOfSoftmaxCrossEntropyLayer<M2> {
    dinput: M2,
}

impl<M2> DInputOfSoftmaxCrossEntropyLayer<M2> {
    pub fn into_value(self) -> M2 {
        self.dinput
    }
}

impl<M2> From<M2> for DInputOfSoftmaxCrossEntropyLayer<M2> {
    fn from(value: M2) -> Self {
        Self { dinput: value }
    }
}

pub(crate) struct OutputOfSoftmaxCrossEntropyLayer {
    out: f32,
}

impl OutputOfSoftmaxCrossEntropyLayer {
    pub fn into_value(self) -> f32 {
        self.out
    }
}

impl From<f32> for OutputOfSoftmaxCrossEntropyLayer {
    fn from(out: f32) -> Self {
        Self { out }
    }
}

const TINY_DELTA: f32 = 0.00_000_000_01;

impl<M2, M1> SoftmaxCrossEntropy<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn softmax_1d(input: M1) -> M1 {
        let max = input.max_value();
        let exp = input.mapv_into(move |x| (x - max).exp());
        let sum = exp.sum();
        exp / sum
    }

    fn softmax(input: M2) -> M2 {
        input.mapv_into_for_each_rows(Self::softmax_1d)
    }

    fn cross_entropy(input: M2, t: M2) -> f32 {
        assert_eq!(input.dim(), t.dim());
        let batch_size = input.dim().0;
        input
            .zip_with(&t, |&input, &t| {
                -t * (if input == 0_f32 { TINY_DELTA } else { input }).ln()
            })
            .sum()
            / batch_size as f32
    }
}

impl<M2, M1> Layer<M2, M1> for SoftmaxCrossEntropy<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfSoftmaxCrossEntropyLayer<M2>;
    type Output = OutputOfSoftmaxCrossEntropyLayer;
    type DInput = DInputOfSoftmaxCrossEntropyLayer<M2>;

    fn new() -> Self {
        Self {
            y: None,
            t: None,
            ph: PhantomData,
        }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input, t } = input;
        let y = Self::softmax(input);
        self.y = Some(y.clone());
        self.t = Some(t.clone());
        Self::Output {
            out: Self::cross_entropy(y, t),
        }
    }

    fn backward(&self, _: Self::Output) -> Self::DInput {
        assert!(self.y.is_some());
        assert!(self.t.is_some());

        let y = self.y.as_ref().unwrap().clone();
        let t = self.t.as_ref().unwrap().clone();
        assert_eq!(y.dim(), t.dim());

        let batch_size = y.dim().0 as f32;
        Self::DInput {
            dinput: (y - t) / batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Zip};
    use ndarray_rand::{rand_distr::Normal, RandomExt};

    use super::*;

    #[test]
    fn test_softmax_cross_entropy_layer_foward() {
        let mut softmax_cross_entropy = SoftmaxCrossEntropy::new();
        let input = InputOfSoftmaxCrossEntropyLayer {
            input: array![[1., 2., 18.], [-3., 3., 5.]],
            t: array![[0., 0., 1.], [0., 1., 0.]],
        };
        let output = softmax_cross_entropy.forward(input);

        // input, onehots の第１行目の loss の計算
        let sum1 = (-17_f32).exp() + (-16_f32).exp() + 1.;
        let loss1 = -(1. / sum1).ln();

        // input, onehots の第２行目の loss の計算
        let sum2 = (-8_f32).exp() + (-2_f32).exp() + 1.;
        let loss2 = -((-2_f32).exp() / sum2).ln();

        let expected = (loss1 + loss2) / 2.;
        assert_eq!(output.out, expected);
    }

    #[test]
    fn test_softmax() {
        let input = array![[1., 2., 3.], [4., 5., 6.]];
        let sum1 = (1_f32).exp() + (2_f32).exp() + (3_f32).exp();
        let sum2 = (4_f32).exp() + (5_f32).exp() + (6_f32).exp();
        let expected = array![
            [
                (1_f32).exp() / sum1,
                (2_f32).exp() / sum1,
                (3_f32).exp() / sum1,
            ],
            [
                (4_f32).exp() / sum2,
                (5_f32).exp() / sum2,
                (6_f32).exp() / sum2,
            ]
        ];
        let result = SoftmaxCrossEntropy::softmax(input);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(actual, expected)| {
                assert_abs_diff_eq!(actual, expected);
            });
    }

    #[test]
    fn test_softmax_cross_entropy_layer_backward() {
        let mut softmax_cross_entropy = SoftmaxCrossEntropy::new();

        // 微分を数値計算するための微小量
        const DELTA: f32 = 0.00_000_01;

        // 入力をランダムに生成
        let input = Array::random((13, 7), Normal::new(0., 1.).unwrap());

        // 教師ラベルを適当に生成
        let mut one_hots = Array::zeros((13, 7));
        one_hots
            .rows_mut()
            .into_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let len = row.len();
                row[i % len] = 1.;
            });

        // 逆誤差伝播法を利用した微分の計算
        let _ = softmax_cross_entropy.forward(InputOfSoftmaxCrossEntropyLayer {
            input: input.clone(),
            t: one_hots.clone(),
        });
        let result = softmax_cross_entropy.backward(OutputOfSoftmaxCrossEntropyLayer { out: 1. });
        let expected = (SoftmaxCrossEntropy::softmax(input.clone()) - one_hots) / 13.;

        // input の [i, j] 成分に関する微分を数値計算
        input.indexed_iter().for_each(|((i, j), _input_ij)| {
            assert_abs_diff_eq!(expected[[i, j]], result.dinput[[i, j]]);
        });
    }
}
