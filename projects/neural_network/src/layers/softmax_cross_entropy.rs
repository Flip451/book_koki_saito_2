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

use ndarray::{Array1, Array2, ArrayView1};

use super::layer::Layer;

pub(crate) struct SoftmaxCrossEntropy {
    y: Option<Array2<f64>>,
    t: Option<Array2<f64>>,
}

pub(crate) struct InputOfSoftmaxCrossEntropyLayer {
    input: Array2<f64>,
    t: Array2<f64>,
}

struct DInputOfSoftmaxCrossEntropyLayer {
pub(crate) struct DInputOfSoftmaxCrossEntropyLayer {
    dinput: Array2<f64>,
}

struct OutputOfSoftmaxCrossEntropyLayer {
pub(crate) struct OutputOfSoftmaxCrossEntropyLayer {
    out: f64,
}

const TINY_DELTA: f64 = 0.00_000_000_01;

impl SoftmaxCrossEntropy {
    fn softmax_1d(input: ArrayView1<f64>) -> Array1<f64> {
        let max = input.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap();
        let exp = input.mapv(|x| (x - max).exp());
        let sum = exp.sum();
        exp / sum
    }

    fn softmax(input: Array2<f64>) -> Array2<f64> {
        let (height, width) = input.dim();
        let flattened: Array1<f64> = input
            .rows()
            .into_iter()
            .flat_map(|row| Self::softmax_1d(row))
            .collect();
        flattened.into_shape((height, width)).unwrap()
    }

    fn cross_entropy(input: Array2<f64>, t: Array2<f64>) -> f64 {
        assert_eq!(input.dim(), t.dim());
        let batch_size = input.shape()[0];
        input
            .into_iter()
            .zip(t)
            .map(|(input, t)| -t * (if input == 0. { TINY_DELTA } else { input }).ln())
            .sum::<f64>()
            / batch_size as f64
    }
}

impl Layer for SoftmaxCrossEntropy {
    type Input = InputOfSoftmaxCrossEntropyLayer;
    type Output = OutputOfSoftmaxCrossEntropyLayer;
    type DInput = DInputOfSoftmaxCrossEntropyLayer;

    fn new() -> Self {
        Self { y: None, t: None }
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

        let y = self.y.as_ref().unwrap();
        let t = self.t.as_ref().unwrap();
        assert_eq!(y.dim(), t.dim());

        let batch_size = y.shape()[0] as f64;
        Self::DInput {
            dinput: (y - t) / batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array};
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
        let sum1 = (-17_f64).exp() + (-16_f64).exp() + 1.;
        let loss1 = -(1. / sum1).ln();

        // input, onehots の第２行目の loss の計算
        let sum2 = (-8_f64).exp() + (-2_f64).exp() + 1.;
        let loss2 = -((-2_f64).exp() / sum2).ln();

        let expected = (loss1 + loss2) / 2.;
        assert_eq!(output.out, expected);
    }

    #[test]
    fn test_softmax_cross_entropy_layer_backward() {
        let mut softmax_cross_entropy = SoftmaxCrossEntropy::new();

        // 微分を数値計算するための微小量
        const DELTA: f64 = 0.00_000_01;

        // 入力をランダムに生成
        let input = Array::random((13, 7), Normal::new(0., 1.).unwrap());
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

        // input の [i, j] 成分に関する微分を数値計算
        input.indexed_iter().for_each(|((i, j), input_ij)| {
            let mut input_left = input.clone();
            input_left[[i, j]] = input_ij - DELTA;

            let mut layer_left = SoftmaxCrossEntropy::new();
            let result_left = layer_left.forward(InputOfSoftmaxCrossEntropyLayer {
                input: input_left,
                t: one_hots.clone(),
            });

            let mut input_right = input.clone();
            input_right[[i, j]] = input_ij + DELTA;

            let mut layer_right = SoftmaxCrossEntropy::new();
            let result_right = layer_right.forward(InputOfSoftmaxCrossEntropyLayer {
                input: input_right,
                t: one_hots.clone(),
            });

            let expected = (result_right.out - result_left.out) / (2. * DELTA);
            assert_abs_diff_eq!(expected, result.dinput[[i, j]], epsilon = 1e-8);
        });
    }
}
