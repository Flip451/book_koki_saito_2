use std::io::{self, Write};

use crate::{
    corpus::{Corpus, Word},
    word_matrix::WordMatrix,
};

use super::cos_similarity::cos_similarity;

pub(crate) fn print_most_similar<T: WordMatrix>(
    query: Word,
    corpus: &Corpus,
    word_matrix: T,
    top: usize,
) {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    most_similar(&mut stdout, query, corpus, word_matrix, top).unwrap();
}

pub(crate) fn most_similar<W: Write, T: WordMatrix>(
    w: &mut W,
    query: Word,
    corpus: &Corpus,
    word_matrix: T,
    top: usize,
) -> io::Result<()> {
    if let None = corpus.word_to_id.get(&query) {
        writeln!(w, "{} is not found", query)?;
        return Ok(());
    }

    writeln!(w, "\n[query] {}", query)?;

    let query_id = corpus.word_to_id[&query];
    let matrix = word_matrix.view();
    let query_vec = matrix.row(query_id);
    let id_to_word = &corpus.id_to_word;

    let mut cos_similarities = word_matrix
        .view()
        .rows()
        .into_iter()
        .enumerate()
        .filter(|&(word_id, _)| word_id != query_id)
        .map(|(word_id, word_vec)| {
            (
                id_to_word.get(&word_id).unwrap(),
                cos_similarity(query_vec, word_vec, None),
            )
        })
        .collect::<Vec<_>>();
    cos_similarities.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    for (word, cos_similarity) in cos_similarities.into_iter().take(top) {
        writeln!(w, "{}: {}", word, cos_similarity)?;
    }

    Ok(())
}

// <https://teratail.com/questions/126598> を参考に実装
#[cfg(test)]
mod tests {
    use crate::word_matrix::co_matrix::CoMatrix;

    use super::*;

    #[test]
    fn test_most_similar() {
        let text = "You say goodbye and I say hello.";
        let corpus = Corpus::new(text);
        let co_matrix = CoMatrix::new(&corpus, 1);

        let mut w = Vec::<u8>::new();
        most_similar(&mut w, "you".to_string(), &corpus, co_matrix, 5).unwrap();

        let expected = "\n[query] you\ngoodbye: 0.70710677\ni: 0.70710677\nhello: 0.70710677\nsay: 0\nand: 0\n";
        assert_eq!(String::from_utf8(w).unwrap(), expected);
    }
}
