use ndarray::{Array, Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};

use crate::{
    matrix::{matrix_one_dim::MatrixOneDim, matrix_two_dim::MatrixTwoDim},
    optimizer::optimizer::Optimizer,
};

use super::{
    layers::{
        affine::{AffineLayer, ParamsOfAffineLayer},
        layer::{IntermediateLayer, LayerBase, LossLayer},
        relu::{ParamsOfReLULayer, ReLULayer},
        sigmoid::{ParamsOfSigmoidLayer, SigmoidLayer},
        softmax_cross_entropy::{ParamsOfSoftmaxCrossEntropyLayer, SoftmaxCrossEntropyLayer},
    },
    network::Network,
};

// ハイパーパラメータ
const MEAN_DISTR: f32 = 0.;
const STD_DEV_DISTR: f32 = 0.01;

enum HiddenLayer<M2, M1> {
    Affine(AffineLayer<M2, M1>),
    Sigmoid(SigmoidLayer<M2, M1>),
    ReLU(ReLULayer<M2, M1>),
}

pub struct SimpleNetwork<M2, M1> {
    layers: Vec<HiddenLayer<M2, M1>>,
    loss_layer: SoftmaxCrossEntropyLayer<M2, M1>,
}

pub enum Activation {
    Sigmoid,
    ReLU,
}

impl<M2, M1> SimpleNetwork<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        activation: Activation,
    ) -> Self {
        let mut layers = vec![];

        let last_layer_size =
            hidden_sizes
                .into_iter()
                .fold(input_size, |last_layer_size, current_layer_size| {
                    // TODO: 重みとバイアスの初期化を M2, M1 に要請する
                    let weight = M2::random_normal(
                        (last_layer_size, current_layer_size),
                        MEAN_DISTR,
                        STD_DEV_DISTR,
                    );
                    let bias = M1::zeros(current_layer_size);

                    layers.push(HiddenLayer::Affine(AffineLayer::new(ParamsOfAffineLayer {
                        w: weight,
                        b: bias,
                    })));

                    match activation {
                        Activation::Sigmoid => {
                            layers.push(HiddenLayer::Sigmoid(SigmoidLayer::new(
                                ParamsOfSigmoidLayer(),
                            )));
                        }
                        Activation::ReLU => {
                            layers.push(HiddenLayer::ReLU(ReLULayer::new(ParamsOfReLULayer())));
                        }
                    }

                    current_layer_size
                });

        let weight = M2::random_normal((last_layer_size, output_size), MEAN_DISTR, STD_DEV_DISTR);
        let bias = M1::zeros(output_size);

        layers.push(HiddenLayer::Affine(AffineLayer::new(ParamsOfAffineLayer {
            w: weight,
            b: bias,
        })));

        Self {
            layers,
            loss_layer: SoftmaxCrossEntropyLayer::new(ParamsOfSoftmaxCrossEntropyLayer()),
        }
    }

    fn params_and_grads(
        &mut self,
    ) -> Vec<(
        &mut ParamsOfAffineLayer<M2, M1>,
        &ParamsOfAffineLayer<M2, M1>,
    )> {
        let mut params_and_grads = vec![];
        for layer in &mut self.layers {
            match layer {
                HiddenLayer::Affine(affine_layer) => {
                    params_and_grads.push(affine_layer.params_and_grads());
                }
                HiddenLayer::Sigmoid(_) => {}
                HiddenLayer::ReLU(_) => {}
            }
        }
        params_and_grads
    }
}

impl<M2, M1> Network<M2, M1> for SimpleNetwork<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    fn predict(&mut self, input: M2) -> M2 {
        let mut input = input;
        for layer in &mut self.layers {
            match layer {
                HiddenLayer::Affine(affine_layer) => {
                    input = affine_layer.forward(input);
                }
                HiddenLayer::Sigmoid(sigmoid_layer) => {
                    input = sigmoid_layer.forward(input);
                }
                HiddenLayer::ReLU(relu_layer) => {
                    input = relu_layer.forward(input);
                }
            }
        }
        input
    }

    fn forward(&mut self, input: M2, one_hot_labels: M2) -> f32 {
        let score = self.predict(input);
        let loss = self.loss_layer.forward(score, one_hot_labels);
        loss
    }

    fn backward(&mut self, dout: f32) -> M2 {
        let mut dout = self.loss_layer.backward(dout);
        for layer in self.layers.iter_mut().rev() {
            match layer {
                HiddenLayer::Affine(affine_layer) => {
                    dout = affine_layer.backward(dout);
                }
                HiddenLayer::Sigmoid(sigmoid_layer) => {
                    dout = sigmoid_layer.backward(dout);
                }
                HiddenLayer::ReLU(relu_layer) => {
                    dout = relu_layer.backward(dout);
                }
            }
        }
        dout
    }

    fn update<T: Optimizer>(&mut self, optimizer: &T) {
        let params_and_grads = self.params_and_grads();
        for (params, grads) in params_and_grads {
            optimizer.update(params, grads);
        }
    }
}
