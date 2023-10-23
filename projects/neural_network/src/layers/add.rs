use super::layer::Layer;

struct Add {}

struct InputOfAddLayer<T> {
    a: T,
    b: T,
}

struct OutputOfAddLayer<T> {
    out: T,
}

struct DInputOfAddLayer<T> {
    da: T,
    db: T,
}

impl<T> Layer<T> for Add
where
    T: std::ops::Add<Output = T> + Clone,
{
    type Input<U> = InputOfAddLayer<U>;
    type Output<V> = OutputOfAddLayer<V>;
    type DInput<V> = DInputOfAddLayer<V>;

    fn new() -> Self {
        Self {}
    }

    fn forward(&mut self, input: Self::Input<T>) -> Self::Output<T>
    where
        T: std::ops::Add<Output = T>,
    {
        Self::Output {
            out: input.a + input.b,
        }
    }

    fn backward(&self, dout: Self::Output<T>) -> Self::DInput<T> {
        Self::DInput {
            da: dout.out.clone(),
            db: dout.out,
        }
    }
}
