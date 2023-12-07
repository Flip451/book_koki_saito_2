use ndarray::{Array2, Array1};
use neural_network::{
    dataset::imp::spiral::{InitParamsOfSpiralDataset, SpiralDataset},
    network::{self, simple_network::Activation},
    optimizer::{
        imp::sgd::{learning_rate::LearningRate, SGD},
        optimizer::Optimizer,
    },
    trainer::Trainer,
};

const BATCH_SIZE: usize = 30;
const MAX_EPOCH: usize = 300;
const HIDDNE_SIZES: [usize; 1] = [10];
const LEARNING_RATE: f32 = 1.;

fn main() {
    let params = InitParamsOfSpiralDataset {
        batch_size: BATCH_SIZE,
        number_of_class: 3,
        point_per_class: 100,
        max_angle: 1. * std::f32::consts::PI,
    };
    let mut dataset: SpiralDataset<Array2<f32>, Array1<f32>> = SpiralDataset::new(params);

    let network = network::simple_network::SimpleNetwork::new(
        2,
        HIDDNE_SIZES.to_vec(),
        3,
        Activation::Sigmoid,
    );
    let optimizer = SGD::new(LearningRate::new(LEARNING_RATE));

    let mut trainer = Trainer::new(network, optimizer);

    trainer.fit(&mut dataset, MAX_EPOCH, 10);
    trainer.plot_accuracy("test_acc.png").unwrap();
    trainer.plot_loss("test_loss.png").unwrap();
}
