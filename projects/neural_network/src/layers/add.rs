use ndarray::Array2;

use super::layer::Layer;

struct Add {}

struct InputOfAddLayer {
    a: Array2<f64>,
    b: Array2<f64>,
}

struct OutputOfAddLayer {
    out: Array2<f64>,
}

struct DInputOfAddLayer {
    da: Array2<f64>,
    db: Array2<f64>,
}

impl Layer for Add {
    type Input = InputOfAddLayer;
    type Output = OutputOfAddLayer;
    type DInput = DInputOfAddLayer;

    fn new() -> Self {
        Self {}
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
    use ndarray::array;

    use super::*;

    #[test]
    fn test_add_layer() {
        // z = x + y となっていることををチェック
        let mut add = Add::new();
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
