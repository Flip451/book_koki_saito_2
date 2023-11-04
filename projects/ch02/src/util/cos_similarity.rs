use ndarray::ArrayView1;
use ndarray_linalg::Norm;

const EPSILON: f32 = 1e-8;

pub(crate) fn cos_similarity(x: ArrayView1<f32>, y: ArrayView1<f32>, epsilon: Option<f32>) -> f32 {
    let epsilon = epsilon.unwrap_or(EPSILON);
    let nx = x.to_owned() / (x.norm_l2() + epsilon);
    let ny = y.to_owned() / (y.norm_l2() + epsilon);
    nx.dot(&ny)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_cos_similarity() {
        // 適当な値
        let x = array![1., 2., 3., 4., 5.];
        let y = array![8., 7., 6., 5., 4.];
        let x_len = ((1. + 4. + 9. + 16. + 25.) as f32).sqrt();
        let y_len = ((64. + 49. + 36. + 25. + 16.) as f32).sqrt();
        let x_dot_y = 1. * 8. + 2. * 7. + 3. * 6. + 4. * 5. + 5. * 4.;
        assert_abs_diff_eq!(
            cos_similarity(x.view(), y.view(), None),
            x_dot_y / (x_len * y_len)
        );

        // 直交する場合
        let x = array![1., 0., 0., 1.];
        let y = array![0., 7., 6., 0.];
        assert_abs_diff_eq!(cos_similarity(x.view(), y.view(), None), 0.);

        // 逆向きに平行な場合
        let x = array![1., 3., -2., 1.];
        let y = array![-3., -9., 6., -3.];
        assert_abs_diff_eq!(cos_similarity(x.view(), y.view(), None), -1.);
    }
}
