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

        let text = words
            .enumerate()
            .map(|(id, word)| {
                word_to_id.insert(word.to_string(), id as WordId);
                id_to_word.insert(id as WordId, word.to_string());
                *word_to_id.get(word).unwrap()
            })
            .collect::<Vec<WordId>>();

        Self {
            text,
            word_to_id,
            id_to_word,
        }
    }
}