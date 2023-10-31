use ndarray::Array2;

use super::layer::Layer;

struct Branch {}

struct InputOfBranchLayer {
    input: Array2<f32>,
}

struct DInputOfBranchLayer {
    dinput: Array2<f32>
}

struct OutputOfBranchLayer {
    a: Array2<f32>,
    b: Array2<f32>,
}

impl Layer for Branch
{
    type Input = InputOfBranchLayer;
    type Output = OutputOfBranchLayer;
    type DInput = DInputOfBranchLayer;

    fn new() -> Self {
        Self {}
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output
    {
        Self::Output {
            a: input.input.clone(),
            b: input.input,
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        Self::DInput {
            dinput: dout.a + dout.b,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_branch_layer() {
        // test forward
        let mut branch = Branch::new();
        let input = InputOfBranchLayer {
            input: array![[1., 2., 3.], [4., 5., 6.]],
        };
        let output = branch.forward(input);

        let expected = array![[1., 2., 3.], [4., 5., 6.]];
        assert_eq!(output.a, expected);
        assert_eq!(output.b, expected);

        // test backward
        let dout = OutputOfBranchLayer {
            a: array![[1., 2., 3.], [4., 5., 6.]],
            b: array![[7., 8., 9.], [10., 11., 12.]],
        };

        let dinput = branch.backward(dout);
        let expected = array![[8., 10., 12.], [14., 16., 18.]];
        assert_eq!(dinput.dinput, expected);
    }
}