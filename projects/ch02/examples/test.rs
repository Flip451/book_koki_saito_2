use std::fs;

use ch02::{corpus::Corpus, word_matrix::co_matrix::CoMatrix, util::most_similar::print_most_similar};

// <https://github.com/ekg/rsvd/tree/main> を利用するが,
// このリポジトリの Cargo.toml 内で
// ndarray-linalg の features に "openblas-static" が指定されてしまっている.
// これは不要なので、このリポジトリをローカルにクローンして、該当箇所を削除して利用.
// この指定が不要な理由については、<https://github.com/rust-ndarray/ndarray-linalg#for-library-developer> に記述がある（以下、引用）：
/*
    ### For library developer

    If you creating a library depending on this crate, we encourage you not to link any backend:

    ```toml
    [dependencies]
    ndarray = "0.13"
    ndarray-linalg = "0.12"
    ```

    The cargo's feature is additive. If your library (saying `lib1`) set a feature `openblas-static`,
    the application using `lib1` builds ndarray_linalg with `openblas-static` feature though they want to use `intel-mkl-static` backend.

    See [the cargo reference](https://doc.rust-lang.org/cargo/reference/features.html) for detail
*/
use rsvd::rsvd;

const FILE_PATH: &str = "examples/ptb.train.txt";

fn main() {
    let text = fs::read_to_string(FILE_PATH).unwrap();
    let corpus = Corpus::new(&text);
    let co_matrix = CoMatrix::new(&corpus, 1);
    let ppmi = co_matrix.ppmi(false, None);
    let ppmi = ppmi.map(|x| *x as f64);
    let (u, s, vt) = rsvd(&ppmi, 5, 0, None);
    // let (u, s, vt) = ppmi.svd(true, false).unwrap();
    println!("U: {:e}", u);
    println!("S: {:e}", s);

    let queries = ["you", "year", "car", "toyota", "hard", "mix", "left"];
    for query in queries {
        print_most_similar(query.to_string(), &corpus, &co_matrix, 4);
    }
}
