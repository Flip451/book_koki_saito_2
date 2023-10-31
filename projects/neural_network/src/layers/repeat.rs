use ndarray::{Array1, Array2, Axis};

use super::layer::Layer;

struct Repeat {
    n: Option<usize>,
}

struct InputOfRepeatLayer {
    input: Array1<f32>,
    n: usize,
}

struct DInputOfRepeatLayer {
    dinput: Array1<f32>,
}

struct OutputOfRepeatLayer {
    out: Array2<f32>,
}

impl Layer for Repeat {
    type Input = InputOfRepeatLayer;
    type Output = OutputOfRepeatLayer;
    type DInput = DInputOfRepeatLayer;

    fn new() -> Self {
        Self { n: None }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { input, n } = input;
        self.n = Some(n);
        Self::Output {
            out: input.broadcast((n, input.len())).unwrap().to_owned(),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.n.is_some());
        Self::DInput {
            dinput: dout.out.sum_axis(Axis(0)).to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_repeat_layer() {
        // test forward
        let mut repeat = Repeat::new();
        let input = InputOfRepeatLayer {
            input: array![1., 2., 3., 4., 5., 6.],
            n: 3,
        };
        let output = repeat.forward(input);

        assert_eq!(
            output.out,
            array![
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.],
                [1., 2., 3., 4., 5., 6.]
            ]
        );

        // test backward
        let dout = OutputOfRepeatLayer {
            out: array![
                [7., 8., 9., 10., 11., 12.],
                [13., 14., 15., 16., 17., 18.],
                [19., 20., 21., 22., 23., 24.],
            ],
        };
        let dinput = repeat.backward(dout);
        assert_eq!(dinput.dinput, array![39., 42., 45., 48., 51., 54.]);
    }
}
