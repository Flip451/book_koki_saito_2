use ndarray::{Array1, Array2, Axis};

use super::layer::Layer;

struct Sum {
    n: Option<usize>,
}

struct InputOfSumLayer {
    input: Array2<f64>,
}

struct DInputOfSumLayer {
    dinput: Array2<f64>,
}

struct OutputOfSumLayer {
    out: Array1<f64>,
}

impl Layer for Sum {
    type Input = InputOfSumLayer;
    type Output = OutputOfSumLayer;
    type DInput = DInputOfSumLayer;

    fn new() -> Self {
        Self { n: None }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input } = input;
        self.n = Some(input.shape()[0]);
        Self::Output {
            out: input.sum_axis(Axis(0)).to_owned(),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.n.is_some());
        Self::DInput {
            dinput: dout
                .out
                .broadcast((self.n.unwrap(), dout.out.len()))
                .unwrap()
                .to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_sum_layer() {
        // test forward
        let mut sum = Sum::new();
        let input = InputOfSumLayer {
            input: array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        };
        let output = sum.forward(input);
        assert_eq!(output.out, array![12., 15., 18.]);

        // test backward
        let dout = OutputOfSumLayer {
            out: array![10., 11., 12.],
        };
        let dinput = sum.backward(dout);
        assert_eq!(
            dinput.dinput,
            array![
                [10., 11., 12.],
                [10., 11., 12.],
                [10., 11., 12.]
            ]
        );
    }
}