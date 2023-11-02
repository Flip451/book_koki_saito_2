use ndarray::Array2;

use crate::corpus::Corpus;

struct CoMatrix(Array2<u32>);

impl CoMatrix {
    fn new(corpus: &Corpus, window_size: usize) -> Self {
        let vocab_size = corpus.id_to_word.len();
        let mut matrix = Array2::<u32>::zeros((vocab_size, vocab_size));

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
                    matrix[[*word_id, left_word_id]] += 1;
                }

                if let Some(right_idx) = right_idx {
                    let right_word_id = corpus.text[right_idx];
                    matrix[[*word_id, right_word_id]] += 1;
                }
            }
        }

        Self(matrix)
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
                [0, 1, 0, 0, 0, 0, 0], // you
                [1, 0, 1, 0, 1, 1, 0], // say
                [0, 1, 0, 1, 0, 0, 0], // goodbye
                [0, 0, 1, 0, 1, 0, 0], // and
                [0, 1, 0, 1, 0, 0, 0], // i
                [0, 1, 0, 0, 0, 0, 1], // hello
                [0, 0, 0, 0, 0, 1, 0], // .
            ]
        );
    }
}
