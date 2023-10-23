use super::layer::Layer;

struct Branch {}

struct InputOfBranchLayer<T> {
    input: T
}

struct DInputOfBranchLayer<T> {
    dinput: T
}

struct OutputOfBranchLayer<T> {
    a: T,
    b: T,
}

impl<T> Layer<T> for Branch
where
    T: std::ops::Add<Output = T> + Clone,
{
    type Input<U> = InputOfBranchLayer<U>;
    type Output<U> = OutputOfBranchLayer<U>;
    type DInput<U> = DInputOfBranchLayer<U>;

    fn new() -> Self {
        Self {}
    }

    fn forward(&mut self, input: Self::Input<T>) -> Self::Output<T>
    where
        T: std::ops::Add<Output = T>,
    {
        Self::Output {
            a: input.input.clone(),
            b: input.input,
        }
    }

    fn backward(&self, dout: Self::Output<T>) -> Self::DInput<T> {
        Self::DInput {
            dinput: dout.a + dout.b,
        }
    }
}
