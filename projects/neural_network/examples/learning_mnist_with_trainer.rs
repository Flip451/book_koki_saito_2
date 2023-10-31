use neural_network::{
    dataset::imp::mnist::{InitParamsOfMnistDataset, MnistDataset},
    network::simple_network::SimpleNetwork,
    optimizer::{
        imp::sgd::{learning_rate::LearningRate, SGD},
        optimizer::Optimizer,
    },
    trainer::Trainer,
};

const BATCH_SIZE: usize = 100;
const MAX_EPOCH: usize = 17;
const HIDDNE_SIZES: [usize; 1] = [50];
const LEARNING_RATE: f64 = 0.001;

fn main() {
    let params = InitParamsOfMnistDataset {
        batch_size: BATCH_SIZE,
        train_image_file_path: "data/train-images-idx3-ubyte",
        train_label_file_path: "data/train-labels-idx1-ubyte",
        test_image_file_path: "data/t10k-images-idx3-ubyte",
        test_label_file_path: "data/t10k-labels-idx1-ubyte",
    };
    let mut dataset = MnistDataset::new(params);

    let network = SimpleNetwork::new(28 * 28, HIDDNE_SIZES.to_vec(), 10);
    let optimizer = SGD::new(LearningRate::new(LEARNING_RATE));

    let mut trainer = Trainer::new(network, optimizer);

    trainer.fit(&mut dataset, MAX_EPOCH, 10);
}
