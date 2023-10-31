use ndarray::{Array, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};

use crate::optimizer::optimizer::Optimizer;

use super::{
    layers::{
        affine::{AffineLayer, ParamsOfAffineLayer},
        layer::{LayerBase, LossLayer, TransformLayer},
        relu::{ParamsOfReLULayer, ReLULayer},
        sigmoid::{ParamsOfSigmoidLayer, SigmoidLayer},
        softmax_cross_entropy::{ParamsOfSoftmaxCrossEntropyLayer, SoftmaxCrossEntropyLayer},
    },
    network::Network,
};

// ハイパーパラメータ
const MEAN_DISTR: f64 = 0.;
const STD_DEV_DISTR: f64 = 0.01;

enum HiddenLayer {
    Affine(AffineLayer),
    Sigmoid(SigmoidLayer),
    ReLU(ReLULayer),
}

pub struct SimpleNetwork {
    layers: Vec<HiddenLayer>,
    loss_layer: SoftmaxCrossEntropyLayer,
}

pub enum Activation {
    Sigmoid,
    ReLU,
}

impl SimpleNetwork {
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        activation: Activation,
    ) -> Self {
        let distribution = Normal::new(MEAN_DISTR, STD_DEV_DISTR).unwrap();

        let mut layers = vec![];

        let last_layer_size =
            hidden_sizes
                .into_iter()
                .fold(input_size, |last_layer_size, current_layer_size| {
                    let weight = Array::random((last_layer_size, current_layer_size), distribution);
                    let bias = Array::zeros(current_layer_size);

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

        let weight = Array::random((last_layer_size, output_size), distribution);
        let bias = Array::zeros(output_size);

        layers.push(HiddenLayer::Affine(AffineLayer::new(ParamsOfAffineLayer {
            w: weight,
            b: bias,
        })));

        Self {
            layers,
            loss_layer: SoftmaxCrossEntropyLayer::new(ParamsOfSoftmaxCrossEntropyLayer()),
        }
    }

    fn params_and_grads(&mut self) -> Vec<(&mut ParamsOfAffineLayer, &ParamsOfAffineLayer)> {
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

impl Network for SimpleNetwork {
    fn predict(&mut self, input: Array2<f64>) -> Array2<f64> {
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

    fn forward(&mut self, input: Array2<f64>, one_hot_labels: Array2<f64>) -> f64 {
        let score = self.predict(input);
        let loss = self.loss_layer.forward(score, one_hot_labels);
        loss
    }

    fn backward(&mut self, dout: f64) -> Array2<f64> {
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
