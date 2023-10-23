use ndarray::Array2;

use super::layer::Layer;

struct Branch {}

struct InputOfBranchLayer {
    input: Array2<f64>,
}

struct DInputOfBranchLayer {
    dinput: Array2<f64>
}

struct OutputOfBranchLayer {
    a: Array2<f64>,
    b: Array2<f64>,
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
