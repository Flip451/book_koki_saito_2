/*
    Y = X.dot(A)
    ∂L/∂X = ∂L/∂Y.dot(A^T)
    ∂L/∂A = X^T.dot(∂L/∂Y)
*/

use ndarray::Array2;

use super::layer::Layer;

struct MatMul {
    x: Option<Array2<f32>>,
    a: Option<Array2<f32>>,
}

struct InputOfMatMulLayer {
    x: Array2<f32>,
    a: Array2<f32>,
}

struct DInputOfMatMulLayer {
    dx: Array2<f32>,
    da: Array2<f32>,
}

struct OutputOfMatMulLayer {
    out: Array2<f32>,
}

impl Layer for MatMul {
    type Input = InputOfMatMulLayer;
    type Output = OutputOfMatMulLayer;
    type DInput = DInputOfMatMulLayer;

    fn new() -> Self {
        Self { a: None, x: None }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        let Self::Input { x, a } = input;
        self.x = Some(x.clone());
        self.a = Some(a.clone());
        Self::Output {
            out: x.dot(self.a.as_ref().unwrap()),
        }
    }

    fn backward(&self, dout: Self::Output) -> Self::DInput {
        assert!(self.x.is_some());
        assert!(self.a.is_some());
        let x = self.x.as_ref().unwrap();
        let a = self.a.as_ref().unwrap();
        let Self::Output { out: dout } = dout;
        Self::DInput {
            dx: dout.dot(&a.t()).to_owned(),
            da: x.t().dot(&dout).to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_matrix_multiply_layer() {
        // test forward
        let mut matmul = MatMul::new();
        let input = InputOfMatMulLayer {
            x: array![[1., 2., 3.], [4., 5., 6.]],
            a: array![[1., 2.], [3., 4.], [5., 6.]],
        };
        let output = matmul.forward(input);
        assert_eq!(
            output.out,
            array![[22., 28.], [49., 64.]]
        );

        // test backward
        let dout = OutputOfMatMulLayer {
            out: array![[7., 8.], [9., 10.]],
        };
        let dinput = matmul.backward(dout);

        assert_eq!(
            dinput.dx,
            array![[23., 53., 83.], [29., 67., 105.]]
        );
        assert_eq!(
            dinput.da,
            array![[43., 48.], [59., 66.], [75., 84.]]
        )
    }
}
