use std::collections::HashSet;

use tokenizers::{
    models::{
        bpe::{Merges, Vocab, BPE},
        wordlevel::WordLevel,
        ModelWrapper,
    },
    Model,
};

use super::{super::bytes_char, MergeTable, SHORT_PIECE_SYMBOLS};

/// The byte-level alphabet as the first 256 entries of a vocabulary.
fn alphabet_vocab() -> (Vocab, [char; 256]) {
    let table = bytes_char();
    let mut vocab = Vocab::default();
    let mut chars: Vec<char> = table.to_vec();
    chars.sort_unstable();
    for ch in chars {
        let id = vocab.len() as u32;
        vocab.insert(ch.to_string(), id);
    }
    (vocab, table)
}

/// A piece's bytes as the alphabet string the model tokenizes.
fn mapped(table: &[char; 256], piece: &[u8]) -> String {
    piece.iter().map(|&b| table[usize::from(b)]).collect()
}

/// A model over the alphabet plus `merges` in that order (each a pair of
/// alphabet strings; their concatenation joins the vocabulary when new).
fn model_with(merges: &[(&str, &str)], ignore_merges: bool) -> BPE {
    let (mut vocab, _) = alphabet_vocab();
    let mut list = Merges::new();
    for (a, b) in merges {
        let merged = format!("{a}{b}");
        if !vocab.contains_key(&merged) {
            let id = vocab.len() as u32;
            vocab.insert(merged, id);
        }
        list.push(((*a).to_owned(), (*b).to_owned()));
    }
    BPE::builder()
        .vocab_and_merges(vocab, list)
        .ignore_merges(ignore_merges)
        .build()
        .expect("a byte-level BPE")
}

/// The ids the model's own merge gives a piece.
fn model_ids(bpe: &BPE, table: &[char; 256], piece: &[u8]) -> Vec<u32> {
    bpe.tokenize(&mapped(table, piece))
        .expect("the model tokenizes every alphabet string")
        .into_iter()
        .map(|token| token.id)
        .collect()
}

fn table_of(bpe: &BPE) -> MergeTable {
    MergeTable::from_model(&ModelWrapper::BPE(bpe.clone())).expect("the table loads")
}

fn table_ids(table: &MergeTable, piece: &[u8]) -> Vec<u32> {
    let mut ids = Vec::new();
    table.merge(piece, &mut ids);
    ids
}

/// Merges that chain (`t` `h` `e` `the` `Ġthe`), repeat (`a` `a` `aa`
/// `aaaa`), reach into non-ASCII (`Ã©` is `é`, `ä¸Ń` is `中`), and a pair
/// whose rank order is not the order its tokens were created in.
fn merges() -> Vec<(&'static str, &'static str)> {
    vec![
        ("t", "h"),
        ("th", "e"),
        ("Ġ", "the"),
        ("a", "a"),
        ("aa", "aa"),
        ("Ã", "©"),
        ("ä", "¸"),
        ("ä¸", "Ń"),
        ("i", "n"),
        ("in", "g"),
        ("Ġ", "t"),
        ("Ġt", "h"),
        ("Ġth", "e"),
        ("e", "r"),
        ("er", "s"),
        ("1", "2"),
        ("12", "3"),
        ("ð", "Ł"),
        ("ðŁ", "ĺ"),
        ("ðŁĺ", "Ģ"),
        ("Ċ", "Ċ"),
        ("ĊĊ", "ĊĊ"),
        ("Ġ", "Ġ"),
        ("ĠĠ", "ĠĠ"),
        ("ĠĠĠĠ", "ĠĠĠĠ"),
        ("r", "e"),
        ("th", "er"),
    ]
}

/// Pieces that exercise the loops: runs of one byte around the stack bound,
/// repeats of one pair, the longest chains, mixed scripts and emoji, pieces
/// with no merge at all, the empty piece's neighbours.
fn pieces() -> Vec<Vec<u8>> {
    let mut pieces: Vec<Vec<u8>> = [
        "a",
        "aa",
        "aaa",
        "aaaa",
        "aaaaa",
        "aaaaaaa",
        "the",
        " the",
        " there",
        "thethe",
        "theother",
        "ing",
        "ringing",
        "xyz",
        "ers",
        "123",
        "1234",
        "é",
        "éé",
        "中",
        "中文",
        "😀",
        "😀😀",
        "\n\n\n\n\n",
        "    ",
        "         ",
        " the the",
        "re",
        "other",
        "thereafter",
    ]
    .iter()
    .map(|s| s.as_bytes().to_vec())
    .collect();
    for len in [
        SHORT_PIECE_SYMBOLS - 1,
        SHORT_PIECE_SYMBOLS,
        SHORT_PIECE_SYMBOLS + 1,
        40,
        257,
    ] {
        pieces.push(vec![b'a'; len]);
        pieces.push("the".repeat(len).into_bytes());
        pieces.push(" ".repeat(len).into_bytes());
        pieces.push("é".repeat(len).into_bytes());
        pieces.push("😀".repeat(len).into_bytes());
        pieces.push(b"xyz".repeat(len));
        pieces.push(b"a the1 in".repeat(len));
    }
    pieces
}

#[test]
fn every_piece_merges_to_the_models_ids() {
    let (_, alphabet) = alphabet_vocab();
    for ignore_merges in [false, true] {
        let bpe = model_with(&merges(), ignore_merges);
        let table = table_of(&bpe);
        assert_eq!(table.ignore_merges, ignore_merges);
        assert_eq!(table.merges(), merges().len());
        for piece in pieces() {
            let expected = if ignore_merges {
                // The whole-piece lookup is the caller's; the table's merge
                // is the model's merge past it.
                let whole = mapped(&alphabet, &piece);
                match bpe.token_to_id(&whole) {
                    Some(_) => continue,
                    None => model_ids(&bpe, &alphabet, &piece),
                }
            } else {
                model_ids(&bpe, &alphabet, &piece)
            };
            assert_eq!(
                table_ids(&table, &piece),
                expected,
                "piece {:?} (ignore_merges {ignore_merges})",
                String::from_utf8_lossy(&piece)
            );
        }
    }
}

#[test]
fn the_leftmost_of_a_repeated_pair_merges_first() {
    // (a, a) is the only merge: "aaa" is [aa, a], never [a, aa].
    let bpe = model_with(&[("a", "a")], false);
    let table = table_of(&bpe);
    let aa = bpe.token_to_id("aa").expect("aa");
    let a = bpe.token_to_id("a").expect("a");
    assert_eq!(table_ids(&table, b"aaa"), vec![aa, a]);
    assert_eq!(table_ids(&table, b"aaaaa"), vec![aa, aa, a]);
    let long = vec![b'a'; SHORT_PIECE_SYMBOLS * 2 + 1];
    let mut expected = vec![aa; SHORT_PIECE_SYMBOLS];
    expected.push(a);
    assert_eq!(table_ids(&table, &long), expected);
}

#[test]
fn rank_order_is_the_merge_lists_order_not_the_vocabularys() {
    // "abc" with (b, c) ranked before (a, b): [a, bc].
    let bpe = model_with(&[("b", "c"), ("a", "b"), ("ab", "c")], false);
    let table = table_of(&bpe);
    let (_, alphabet) = alphabet_vocab();
    assert_eq!(
        table_ids(&table, b"abc"),
        vec![
            bpe.token_to_id("a").expect("a"),
            bpe.token_to_id("bc").expect("bc")
        ]
    );
    assert_eq!(
        table_ids(&table, b"abc"),
        model_ids(&bpe, &alphabet, b"abc")
    );
}

#[test]
fn random_pieces_merge_to_the_models_ids_on_a_random_vocabulary() {
    // A vocabulary of random pairs over a small byte pool, so pieces merge
    // deeply and the same pair recurs at many positions.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut vocab, alphabet) = alphabet_vocab();
    // Letters, a newline, a digit, a space and the two bytes of a space's
    // alphabet form, so pieces hold split multi-byte sequences too.
    let pool: &[u8] = b"abcde\n1 \xc4\xa0";
    let corpus: Vec<u8> = (0..6000)
        .map(|_| pool[next() as usize % pool.len()])
        .collect();
    let mut list = Merges::new();
    let mut seen = HashSet::new();
    let mapped_corpus = mapped(&alphabet, &corpus);
    let chars: Vec<char> = mapped_corpus.chars().collect();
    // Pairs of the corpus, then pairs of the merged tokens, up to 400 merges.
    let mut tokens: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    while list.len() < 400 {
        let i = next() as usize % (tokens.len() - 1);
        let (a, b) = (tokens[i].clone(), tokens[i + 1].clone());
        if !seen.insert((a.clone(), b.clone())) {
            continue;
        }
        let merged = format!("{a}{b}");
        if vocab.contains_key(&merged) {
            continue;
        }
        let id = vocab.len() as u32;
        vocab.insert(merged.clone(), id);
        list.push((a, b));
        tokens[i] = merged;
        tokens.remove(i + 1);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, list)
        .build()
        .expect("a random BPE");
    let table = table_of(&bpe);
    for round in 0..3000 {
        let len = 1 + round % 48;
        let start = next() as usize % (corpus.len() - len);
        let piece = &corpus[start..start + len];
        assert_eq!(
            table_ids(&table, piece),
            model_ids(&bpe, &alphabet, piece),
            "piece {:?}",
            String::from_utf8_lossy(piece)
        );
    }
}

#[test]
fn a_token_produced_by_two_pairs_merges_as_the_model_does() {
    // Every split of "abc" is a merge (the shape of a vocabulary converted
    // from ranks, where a token has one merge per split): the model's
    // expired-entry test reads the produced token, and the rule here reads
    // the pair; the ids agree on every piece around such a token.
    let (_, alphabet) = alphabet_vocab();
    for ignore_merges in [false, true] {
        let bpe = model_with(
            &[
                ("a", "b"),
                ("b", "c"),
                ("ab", "c"),
                ("a", "bc"),
                ("c", "a"),
                ("abc", "a"),
                ("ab", "ca"),
                ("b", "ca"),
                ("a", "bca"),
            ],
            ignore_merges,
        );
        let table = table_of(&bpe);
        for piece in [
            "abc",
            "abca",
            "abcabc",
            "xabcx",
            "ababc",
            "abcbc",
            "aabcca",
            "cabcab",
            "bcabca",
            "abcabcabcabcabcabcabcabcabcabc",
            "aaabbbcccabcabc",
        ] {
            let piece = piece.as_bytes();
            if ignore_merges && bpe.token_to_id(&mapped(&alphabet, piece)).is_some() {
                continue;
            }
            assert_eq!(
                table_ids(&table, piece),
                model_ids(&bpe, &alphabet, piece),
                "piece {:?} (ignore_merges {ignore_merges})",
                String::from_utf8_lossy(piece)
            );
        }
    }
}

#[test]
fn a_vocabulary_without_merges_leaves_the_bytes() {
    let bpe = model_with(&[], false);
    let table = table_of(&bpe);
    let (_, alphabet) = alphabet_vocab();
    for piece in [&b"abc"[..], "é中😀".as_bytes(), &[b'x'; 300]] {
        assert_eq!(table_ids(&table, piece), model_ids(&bpe, &alphabet, piece));
        assert_eq!(table_ids(&table, piece).len(), piece.len());
    }
}

#[test]
fn models_the_table_cannot_stand_in_for_decline() {
    let declined = |bpe: BPE| {
        MergeTable::from_model(&ModelWrapper::BPE(bpe))
            .err()
            .expect("declined")
            .to_string()
    };
    let mut dropout = model_with(&merges(), false);
    dropout.dropout = Some(0.1);
    assert_eq!(declined(dropout), "the model merges with dropout");
    let mut zero_dropout = model_with(&merges(), false);
    zero_dropout.dropout = Some(0.0);
    assert_eq!(table_of(&zero_dropout).merges(), merges().len());
    let mut prefix = model_with(&merges(), false);
    prefix.continuing_subword_prefix = Some("##".to_owned());
    assert_eq!(
        declined(prefix),
        "the model marks subwords with a prefix or a suffix"
    );
    let mut suffix = model_with(&merges(), false);
    suffix.end_of_word_suffix = Some("</w>".to_owned());
    assert_eq!(
        declined(suffix),
        "the model marks subwords with a prefix or a suffix"
    );
    // An empty prefix and suffix, as some checkpoints spell "none": the
    // symbols stay the byte tokens, the table loads and merges as the model.
    let mut empty_marks = model_with(&merges(), false);
    empty_marks.continuing_subword_prefix = Some(String::new());
    empty_marks.end_of_word_suffix = Some(String::new());
    let table = table_of(&empty_marks);
    let (_, alphabet) = alphabet_vocab();
    for piece in pieces() {
        assert_eq!(
            table_ids(&table, &piece),
            model_ids(&empty_marks, &alphabet, &piece),
            "piece {:?}",
            String::from_utf8_lossy(&piece)
        );
    }
    // A byte without a token.
    let table = bytes_char();
    let mut vocab = Vocab::default();
    for ch in table.iter().filter(|&&ch| ch != table[0x7F]) {
        let id = vocab.len() as u32;
        vocab.insert(ch.to_string(), id);
    }
    let missing = BPE::builder()
        .vocab_and_merges(vocab, Merges::new())
        .build()
        .expect("bpe");
    assert_eq!(declined(missing), "byte 0x7f has no token");
    // Not BPE at all.
    let word_level = ModelWrapper::WordLevel(WordLevel::default());
    assert_eq!(
        MergeTable::from_model(&word_level)
            .err()
            .expect("declined")
            .to_string(),
        "the model is not BPE"
    );
}

#[test]
fn the_merges_read_are_the_models_merges_with_a_gap_in_the_ids_too() {
    // The merges the reader takes are the field the full serialization
    // writes, on a vocabulary with a gap in its ids (256..300 unassigned),
    // and the table built from them merges as the model does. That the
    // reader never serializes the vocabulary is `serialized::tests`'.
    let (mut vocab, alphabet) = alphabet_vocab();
    let mut list = Merges::new();
    let mut id = 300;
    for (a, b) in merges() {
        let merged = format!("{a}{b}");
        if !vocab.contains_key(&merged) {
            vocab.insert(merged, id);
            id += 1;
        }
        list.push((a.to_owned(), b.to_owned()));
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, list)
        .build()
        .expect("a vocabulary with a gap builds");
    let read: serde_json::Value =
        serde_json::from_str(&super::serialized::merges_json(&bpe).expect("the merges read"))
            .expect("the merges are json");
    let full = serde_json::to_value(&bpe).expect("the model serializes");
    assert_eq!(read, full["merges"]);
    assert_eq!(read.as_array().map(Vec::len), Some(merges().len()));
    let table = table_of(&bpe);
    for piece in pieces() {
        assert_eq!(
            table_ids(&table, &piece),
            model_ids(&bpe, &alphabet, &piece),
            "piece {:?}",
            String::from_utf8_lossy(&piece)
        );
    }
}

#[test]
fn the_tables_bytes_are_the_grid_the_slots_and_the_ids() {
    let bpe = model_with(&merges(), false);
    let table = table_of(&bpe);
    let grid = 512 * 512 * size_of::<u32>();
    assert!(table.bytes() > grid, "{}", table.bytes());
    assert!(table.bytes() < grid + 64 * 1024, "{}", table.bytes());
}
