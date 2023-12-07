use std::marker::PhantomData;

use ndarray::Array2;

use crate::matrix::{matrix_two_dim::MatrixTwoDim, matrix_one_dim::MatrixOneDim};

use super::layer::Layer;

struct Add<M2, M1>(PhantomData<M2>, PhantomData<M1>);

struct InputOfAddLayer<M2> {
    a: M2,
    b: M2,
}

struct OutputOfAddLayer<M2> {
    out: M2,
}

struct DInputOfAddLayer<M2> {
    da: M2,
    db: M2,
}

impl<M2, M1> Layer<M2, M1> for Add<M2, M1>
where
    M2: MatrixTwoDim<M1>,
    M1: MatrixOneDim,
{
    type Input = InputOfAddLayer<M2>;
    type Output = OutputOfAddLayer<M2>;
    type DInput = DInputOfAddLayer<M2>;

    fn new() -> Self {
        Self(PhantomData, PhantomData)
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        Self::Output {
            out: input.a + input.b,
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        Self::DInput {
            da: dout.out.clone(),
            db: dout.out,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array1};

    use super::*;

    #[test]
    fn test_add_layer() {
        // z = x + y となっていることををチェック
        let mut add = Add::<Array2<f32>, Array1<f32>>::new();
        let a = array![[1., 2., 3.], [4., 5., 6.]];
        let b = array![[7., 8., 9.], [10., 11., 12.]];
        let input = InputOfAddLayer { a, b };
        let output = add.forward(input);

        let expected = array![[8., 10., 12.], [14., 16., 18.]];
        assert_eq!(output.out, expected);

        // dL/dx = dL/dz, dL/dy = dL/dx となっていることをチェック
        let dout = array![[1., 2., 3.], [4., 5., 6.]];
        let dinput = add.backward(OutputOfAddLayer { out: dout.clone() });
        assert_eq!(dinput.da, dout.clone());
        assert_eq!(dinput.db, dout);
    }
}
