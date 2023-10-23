pub(super)trait Layer<T> {
    type Input<U>;
    type Output<V>;
    type DInput<W>;

    fn new() -> Self;
    fn forward(&mut self, input: Self::Input<T>) -> Self::Output<T>;
    fn backward(&self, dout: Self::Output<T>) -> Self::DInput<T>;
}