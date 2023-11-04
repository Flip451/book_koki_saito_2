use ndarray::{Array2, ArrayView2};

use crate::corpus::Corpus;

use super::WordMatrix;

pub struct CoMatrix(Array2<f32>);

impl CoMatrix {
    pub fn new(corpus: &Corpus, window_size: usize) -> Self {
        let vocab_size = corpus.id_to_word.len();
        let mut matrix = Array2::<f32>::zeros((vocab_size, vocab_size));

        for (idx, word_id) in corpus.text.iter().enumerate() {
            for i in 1..=window_size {
                let left_idx = idx.checked_sub(i);
                let right_idx = if idx + i < corpus.text.len() {
                    Some(idx + i)
                } else {
                    None
                };

                if let Some(left_idx) = left_idx {
                    let left_word_id = corpus.text[left_idx];
                    matrix[[*word_id, left_word_id]] += 1.;
                }

                if let Some(right_idx) = right_idx {
                    let right_word_id = corpus.text[right_idx];
                    matrix[[*word_id, right_word_id]] += 1.;
                }
            }
        }

        Self(matrix)
    }
}

impl WordMatrix for CoMatrix {
    fn view(&self) -> ArrayView2<f32> {
        self.0.view()
    }
}

impl CoMatrix {
    // PPMI(x, y) = max(0, log2(P(x, y) / P(x)P(y)))
    // P(x, y) / P(x)P(y) = C(x, y) N / C(x)C(y)
    // N: コーパスの単語数（語彙数ではないことに注意）≒ C(*, *)の和
    // C(x, y): 単語xと単語yの共起回数
    // C(x): 単語xの出現回数 ≒ C(x, *)の和
    pub fn ppmi(&self, verbose: bool, eps: Option<f32>) -> Array2<f32> {
        let eps = eps.unwrap_or(1e-8);
        let mut cnt = 0;
        let total = self.0.len();
        let mut ppmi = Array2::<f32>::zeros(self.0.dim());

        let sum = self.0.sum();
        let row_sum = self.0.sum_axis(ndarray::Axis(1));
        let col_sum = self.0.sum_axis(ndarray::Axis(0));

        for ((i, j), &value) in self.0.indexed_iter() {
            let pmi = (value * sum) / (row_sum[i] * col_sum[j]);
            ppmi[[i, j]] = if pmi > 0. { (pmi + eps).log2() } else { 0. };

            if verbose {
                cnt += 1;
                if cnt % 100_000 == 0 {
                    println!("PPMI: {}/{}", cnt, total);
                }
            }
        }

        ppmi
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_co_matrix() {
        let text = "You say goodbye and I say hello.";
        let corpus = Corpus::new(text);
        let co_matrix = CoMatrix::new(&corpus, 1);

        assert_eq!(
            co_matrix.0,
            array![
                [0., 1., 0., 0., 0., 0., 0.], // you
                [1., 0., 1., 0., 1., 1., 0.], // say
                [0., 1., 0., 1., 0., 0., 0.], // goodbye
                [0., 0., 1., 0., 1., 0., 0.], // and
                [0., 1., 0., 1., 0., 0., 0.], // i
                [0., 1., 0., 0., 0., 0., 1.], // hello
                [0., 0., 0., 0., 0., 1., 0.], // .
            ]
        );
    }

    #[test]
    fn test_ppmi() {
        let text = "You say goodbye and I say hello.";
        let corpus = Corpus::new(text);
        let co_matrix = CoMatrix::new(&corpus, 1);
        let ppmi = co_matrix.ppmi(false, None);

        assert_eq!(
            ppmi,
            array![
                [0., 1.8073549, 0., 0., 0., 0., 0.],
                [1.8073549, 0., 0.8073549, 0., 0.8073549, 0.8073549, 0.],
                [0., 0.8073549, 0., 1.8073549, 0., 0., 0.],
                [0., 0., 1.8073549, 0., 1.8073549, 0., 0.],
                [0., 0.8073549, 0., 1.8073549, 0., 0., 0.],
                [0., 0.8073549, 0., 0., 0., 0., 2.807355],
                [0., 0., 0., 0., 0., 2.807355, 0.],
            ]
        );
    }
}
