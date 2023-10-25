pub(crate)trait Layer {
    type Input;
    type Output;
    type DInput;

    fn new() -> Self;
    fn forward(&mut self, input: Self::Input) -> Self::Output;
    fn backward(&self, dout: Self::Output) -> Self::DInput;
}