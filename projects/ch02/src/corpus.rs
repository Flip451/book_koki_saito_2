use std::collections::HashMap;

type WordId = u32;
type Word = String;

struct Corpus {
    text: Vec<WordId>,
    word_to_id: HashMap<Word, WordId>,
    id_to_word: HashMap<WordId, Word>,
}

impl Corpus {
    fn new(text: &str) -> Self {
        let text = text.to_lowercase();
        let text = text.replace(".", " .");
        let words = text.split_whitespace();

        let mut id_to_word = HashMap::<WordId, Word>::new();
        let mut word_to_id = HashMap::<Word, WordId>::new();
        let mut new_id = 0;

        let text = words
            .map(|word| {
                let word_id = match word_to_id.get(word) {
                    Some(id) => *id,
                    None => {
                        let id = new_id;
                        new_id += 1;
                        word_to_id.insert(word.to_string(), id);
                        id_to_word.insert(id, word.to_string());
                        id
                    }
                };
                word_id
            })
            .collect::<Vec<WordId>>();

        Self {
            text,
            word_to_id,
            id_to_word,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus() {
        let text = "You say goodbye and I say hello.";
        let corpus = Corpus::new(text);

        assert_eq!(corpus.text, vec![0, 1, 2, 3, 4, 1, 5, 6]);

        let mut expected_word_to_id = HashMap::new();
        expected_word_to_id.insert("you".to_string(), 0);
        expected_word_to_id.insert("say".to_string(), 1);
        expected_word_to_id.insert("goodbye".to_string(), 2);
        expected_word_to_id.insert("and".to_string(), 3);
        expected_word_to_id.insert("i".to_string(), 4);
        expected_word_to_id.insert("hello".to_string(), 5);
        expected_word_to_id.insert(".".to_string(), 6);

        assert_eq!(corpus.word_to_id, expected_word_to_id);

        let mut expected_id_to_word = HashMap::new();
        expected_id_to_word.insert(0, "you".to_string());
        expected_id_to_word.insert(1, "say".to_string());
        expected_id_to_word.insert(2, "goodbye".to_string());
        expected_id_to_word.insert(3, "and".to_string());
        expected_id_to_word.insert(4, "i".to_string());
        expected_id_to_word.insert(5, "hello".to_string());
        expected_id_to_word.insert(6, ".".to_string());

        assert_eq!(corpus.id_to_word, expected_id_to_word);
    }
}
