use std::collections::{hash_map::Entry, HashSet};

use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;
use tokenizers::{
    models::{
        bpe::{Merges, Vocab, BPE},
        wordlevel::WordLevel,
        ModelWrapper,
    },
    Model,
};

use super::{
    super::bytes_char, ByteTokens, Merge, MergeTable, BUCKET, HOT_SIDE, PAIR_FILTER_BITS_PER_MERGE,
    SHORT_PIECE_SMALL, SHORT_PIECE_SYMBOLS, SHORT_PIECE_TINY,
};

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

/// [`model_with`] whose vocabulary also holds `tokens` that no merge
/// produces (an entry no merge sequence reaches, as some vocabularies have).
fn model_with_tokens(merges: &[(&str, &str)], tokens: &[&str]) -> BPE {
    let (mut vocab, _) = alphabet_vocab();
    for token in tokens {
        if !vocab.contains_key(*token) {
            let id = vocab.len() as u32;
            vocab.insert((*token).to_owned(), id);
        }
    }
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
    MergeTable::from_model(&ModelWrapper::BPE(bpe.clone()), ByteTokens::Alphabet, None)
        .expect("the table loads")
}

fn table_ids(table: &MergeTable, piece: &[u8]) -> Vec<u32> {
    let mut ids = Vec::new();
    table.merge(piece, &mut ids);
    ids
}

/// The ids the table gives a piece the way the encoder asks for them: a
/// piece with a non-ASCII byte from its pre-formed characters first
/// (`MergeTable::merge_preformed`), from its bytes when that declines; and
/// whether the pre-formed path answered.
fn table_ids_preformed(table: &MergeTable, piece: &[u8]) -> (Vec<u32>, bool) {
    let mut ids = Vec::new();
    let answered = !piece.is_ascii() && table.merge_preformed(piece, &mut ids);
    if !answered {
        table.merge(piece, &mut ids);
    }
    (ids, answered)
}

/// The raw bytes an alphabet string stands for.
fn unmapped(table: &[char; 256], entry: &str) -> Vec<u8> {
    entry
        .chars()
        .map(|c| {
            table
                .iter()
                .position(|&a| a == c)
                .expect("an alphabet character") as u8
        })
        .collect()
}

/// A tiktoken vocabulary: the 256 bytes, then `merges` in order, each
/// concatenation a new token at the next rank (one that exists already
/// keeps its rank). With `bytes_last` the byte tokens take the ranks after
/// the merges instead, as a file whose byte tokens are not its first lines.
fn ranks_of(merges: &[(&str, &str)], bytes_last: bool) -> FxHashMap<Vec<u8>, u32> {
    let (_, alphabet) = alphabet_vocab();
    let mut ranks = FxHashMap::default();
    let mut next = if bytes_last { 0 } else { 256 };
    for (a, b) in merges {
        let token = unmapped(&alphabet, &format!("{a}{b}"));
        if let Entry::Vacant(slot) = ranks.entry(token) {
            slot.insert(next);
            next += 1;
        }
    }
    for byte in 0..=255u8 {
        ranks.insert(
            vec![byte],
            if bytes_last {
                next + u32::from(byte)
            } else {
                u32::from(byte)
            },
        );
    }
    ranks
}

/// The file's own encoder over `ranks`, every piece one match.
fn own_encoder(ranks: &FxHashMap<Vec<u8>, u32>) -> CoreBPE {
    CoreBPE::new(ranks.clone(), FxHashMap::default(), r"[\s\S]+").expect("the encoder loads")
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
        SHORT_PIECE_TINY - 1,
        SHORT_PIECE_TINY,
        SHORT_PIECE_TINY + 1,
        SHORT_PIECE_SMALL - 1,
        SHORT_PIECE_SMALL,
        SHORT_PIECE_SMALL + 1,
        SHORT_PIECE_SYMBOLS - 1,
        SHORT_PIECE_SYMBOLS,
        SHORT_PIECE_SYMBOLS + 1,
        40,
        100,
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

/// The pairs a tiktoken vocabulary's table holds: every cut of every token
/// of two bytes or more whose two halves are tokens.
fn split_pairs(ranks: &FxHashMap<Vec<u8>, u32>) -> usize {
    ranks
        .keys()
        .filter(|token| token.len() >= 2)
        .map(|token| {
            (1..token.len())
                .filter(|&cut| {
                    ranks.contains_key(&token[..cut]) && ranks.contains_key(&token[cut..])
                })
                .count()
        })
        .sum()
}

/// A tiktoken vocabulary's table merges every piece as the file's own
/// encoder does: the merges of `merges()` as ranks (bytes first and bytes
/// last), the pieces of `pieces()`, and pieces that are tokens themselves
/// (the encoder looks those up whole; the table merges them, and a complete
/// table reaches them).
#[test]
fn a_tiktoken_vocabulary_merges_as_its_own_encoder_does() {
    for bytes_last in [false, true] {
        let ranks = ranks_of(&merges(), bytes_last);
        let table = MergeTable::from_ranks(&ranks).expect("a complete table");
        assert!(!table.ignore_merges);
        // One entry per split of a token into two tokens.
        assert_eq!(table.merges(), split_pairs(&ranks));
        let own = own_encoder(&ranks);
        let mut tokens: Vec<Vec<u8>> = ranks.keys().cloned().collect();
        tokens.sort();
        for piece in pieces().into_iter().chain(tokens) {
            let expected = match std::str::from_utf8(&piece) {
                Ok(text) => own.encode_ordinary(text),
                // A token that is not text on its own (a byte or a prefix
                // of a multi-byte character): the encoder's whole lookup.
                Err(_) => vec![ranks[&piece]],
            };
            assert_eq!(
                table_ids(&table, &piece),
                expected,
                "piece {:?} (bytes last {bytes_last})",
                String::from_utf8_lossy(&piece)
            );
        }
    }
}

/// On a vocabulary whose merges are a derivation (every token made by the
/// merge the byte-pair rule itself would make), the table read from the
/// ranks, the table read from the equivalent `tokenizers` model and the
/// model agree on every piece: the pairs implied by the ranks are the
/// model's merges, at the same priorities.
#[test]
fn the_table_from_ranks_equals_the_table_from_the_equivalent_model() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (_, alphabet) = alphabet_vocab();
    // A corpus over a small pool, merged by the byte-pair rule itself: the
    // most frequent adjacent pair becomes the next token, 300 times.
    let pool: &[u8] = b"abcdef \n1\xc3\xa9";
    let corpus: Vec<u8> = (0..4000)
        .map(|_| pool[next() as usize % pool.len()])
        .collect();
    let mut symbols: Vec<Vec<u8>> = corpus.iter().map(|&b| vec![b]).collect();
    let mut ranks: FxHashMap<Vec<u8>, u32> = (0..=255u8).map(|b| (vec![b], u32::from(b))).collect();
    let mut list: Vec<(String, String)> = Vec::new();
    // A pair whose concatenation is a token already is passed over (its
    // token has its one derivation), so that the model's merge list is
    // exactly the derivation of every token.
    let mut passed_over: HashSet<(Vec<u8>, Vec<u8>)> = HashSet::new();
    while list.len() < 300 {
        let mut counts: FxHashMap<(Vec<u8>, Vec<u8>), usize> = FxHashMap::default();
        for pair in symbols.windows(2) {
            let pair = (pair[0].clone(), pair[1].clone());
            if pair.0 != b" " && pair.1 != b" " && !passed_over.contains(&pair) {
                *counts.entry(pair).or_default() += 1;
            }
        }
        let Some(((a, b), _)) = counts
            .into_iter()
            .max_by_key(|((a, b), n)| (*n, std::cmp::Reverse((a.clone(), b.clone()))))
        else {
            break;
        };
        let token = [a.as_slice(), b.as_slice()].concat();
        if ranks.contains_key(&token) {
            passed_over.insert((a, b));
            continue;
        }
        ranks.insert(token.clone(), ranks.len() as u32);
        list.push((mapped(&alphabet, &a), mapped(&alphabet, &b)));
        let mut merged = Vec::with_capacity(symbols.len());
        let mut i = 0;
        while i < symbols.len() {
            if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
                merged.push(token.clone());
                i += 2;
            } else {
                merged.push(symbols[i].clone());
                i += 1;
            }
        }
        symbols = merged;
    }
    let from_ranks = MergeTable::from_ranks(&ranks).expect("a complete table");
    let (mut vocab, _) = alphabet_vocab();
    let mut merges = Merges::new();
    for (a, b) in &list {
        let merged = format!("{a}{b}");
        let id = vocab.len() as u32;
        vocab.entry(merged).or_insert(id);
        merges.push((a.clone(), b.clone()));
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("a byte-level BPE");
    let from_model = table_of(&model);
    let own = own_encoder(&ranks);
    let token_of_rank: FxHashMap<u32, &Vec<u8>> =
        ranks.iter().map(|(token, &rank)| (rank, token)).collect();
    for round in 0..3000 {
        let len = 1 + round % 40;
        let piece: Vec<u8> = (0..len)
            .map(|_| pool[next() as usize % pool.len()])
            .collect();
        let text = String::from_utf8_lossy(&piece).into_owned();
        let piece = text.as_bytes();
        let expected: Vec<u32> = own.encode_ordinary(&text);
        assert_eq!(table_ids(&from_ranks, piece), expected, "ranks: {text:?}");
        // The model's ids name the same tokens by their alphabet spelling.
        let via_model: Vec<Vec<u8>> = table_ids(&from_model, piece)
            .iter()
            .map(|&id| unmapped(&alphabet, &model.id_to_token(id).expect("a token")))
            .collect();
        let via_ranks: Vec<Vec<u8>> = expected
            .iter()
            .map(|id| token_of_rank[id].to_vec())
            .collect();
        assert_eq!(via_model, via_ranks, "model: {text:?}");
    }
}

#[test]
fn a_rank_set_the_table_cannot_stand_in_for_declines() {
    let bytes = |ranks: &mut FxHashMap<Vec<u8>, u32>| {
        for byte in 0..=255u8 {
            ranks.insert(vec![byte], u32::from(byte));
        }
    };
    // A byte without a token.
    let mut ranks = FxHashMap::default();
    bytes(&mut ranks);
    ranks.remove([0x7Fu8].as_slice());
    assert_eq!(
        MergeTable::from_ranks(&ranks)
            .err()
            .expect("declined")
            .to_string(),
        "byte 0x7f has no token"
    );
    // A token that is no two tokens' concatenation.
    let mut ranks = FxHashMap::default();
    bytes(&mut ranks);
    ranks.insert(b"ab".to_vec(), 256);
    ranks.insert(b"abcd".to_vec(), 257);
    assert_eq!(
        MergeTable::from_ranks(&ranks)
            .err()
            .expect("declined")
            .to_string(),
        "token 257 is no two tokens' concatenation"
    );
    // A token its own bytes do not merge back into: `ab` merges before `bc`
    // can, and nothing merges `ab` with `c` or `cd`; the file's encoder
    // would still answer the whole piece `abcd` with its token.
    let mut ranks = FxHashMap::default();
    bytes(&mut ranks);
    ranks.insert(b"ab".to_vec(), 256);
    ranks.insert(b"bcd".to_vec(), 257);
    ranks.insert(b"bc".to_vec(), 258);
    ranks.insert(b"abcd".to_vec(), 259);
    assert_eq!(
        MergeTable::from_ranks(&ranks)
            .err()
            .expect("declined")
            .to_string(),
        "token 259 is not the merge of its bytes"
    );
    assert_eq!(own_encoder(&ranks).encode_ordinary("abcd"), vec![259]);
    // The same set without that token is complete.
    ranks.remove(b"abcd".as_slice());
    assert!(MergeTable::from_ranks(&ranks).is_ok());
    // A rank far past the token count: the table of rank ids is dense and
    // bounded, as the decode table of such a file is.
    ranks.insert(b"ab".to_vec(), 1 << 30);
    assert_eq!(
        MergeTable::from_ranks(&ranks)
            .err()
            .expect("declined")
            .to_string(),
        "ranks up to 1073741825 on a table bounded at 4194304"
    );
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
    // On the three sizes of the stack arrays and on the heap.
    for len in [
        SHORT_PIECE_TINY,
        SHORT_PIECE_TINY + 1,
        SHORT_PIECE_SMALL,
        SHORT_PIECE_SMALL + 1,
        SHORT_PIECE_SYMBOLS,
        SHORT_PIECE_SYMBOLS + 1,
        SHORT_PIECE_SYMBOLS * 2 + 1,
    ] {
        let mut expected = vec![aa; len / 2];
        if len % 2 == 1 {
            expected.push(a);
        }
        assert_eq!(table_ids(&table, &vec![b'a'; len]), expected, "{len} a's");
    }
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
    // Lengths across both sizes of the stack arrays and onto the heap.
    for round in 0..3000 {
        let len = 1 + round % 140;
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
        MergeTable::from_model(&ModelWrapper::BPE(bpe), ByteTokens::Alphabet, None)
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
    // A byte without a token: the table stands, and names the pieces that
    // hold the byte as the model's.
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
    let without = MergeTable::from_model(&ModelWrapper::BPE(missing), ByteTokens::Alphabet, None)
        .expect("a table without the byte's token");
    assert!(without.lacks_a_token_for(&[0x7F]));
    assert!(without.lacks_a_token_for(b"a\x7fb"));
    assert!(!without.lacks_a_token_for(b"ab"));
    // Not BPE at all.
    let word_level = ModelWrapper::WordLevel(WordLevel::default());
    assert_eq!(
        MergeTable::from_model(&word_level, ByteTokens::Alphabet, None)
            .err()
            .expect("declined")
            .to_string(),
        "the model is not BPE"
    );
}

/// A vocabulary whose byte tokens are sentencepiece's `<0xHH>`: the table
/// loads with that spelling, and merges a piece's symbols of any kind, a
/// character's token or a byte's, as the model merges that word.
#[test]
fn a_byte_fallback_vocabulary_merges_its_symbols_as_the_model_does() {
    let mut vocab = Vocab::default();
    let add = |vocab: &mut Vocab, token: String| {
        if !vocab.contains_key(&token) {
            let id = vocab.len() as u32;
            vocab.insert(token, id);
        }
    };
    add(&mut vocab, "<unk>".to_owned());
    for byte in 0..=255u8 {
        add(&mut vocab, format!("<0x{byte:02X}>"));
    }
    for c in "▁thequickbrownfoxé中".chars() {
        add(&mut vocab, c.to_string());
    }
    let merges = [
        ("t", "h"),
        ("th", "e"),
        ("▁", "the"),
        ("q", "u"),
        ("qu", "i"),
        ("c", "k"),
        ("qui", "ck"),
        ("▁", "qu"),
        ("é", "é"),
        ("中", "中"),
        ("▁", "é"),
    ];
    let mut list = Merges::new();
    for (a, b) in merges {
        add(&mut vocab, format!("{a}{b}"));
        list.push((a.to_owned(), b.to_owned()));
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, list)
        .byte_fallback(true)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("a byte-fallback BPE");
    let table = MergeTable::from_model(&ModelWrapper::BPE(bpe.clone()), ByteTokens::Hex, None)
        .expect("the table loads with hex byte tokens");
    assert_eq!(table.merges(), merges.len());
    // No merge of such a table starts from bytes: the hot grid is not built
    // (2 MiB per loaded vocabulary otherwise).
    assert!(
        table.hot.is_empty(),
        "a table built for character symbols carries no hot grid"
    );
    assert!(table.bytes() < (HOT_SIDE * HOT_SIDE) as usize * size_of::<u64>());
    // The symbols of a word as `merge_word` makes them: a character's token,
    // else its bytes' tokens.
    let symbols = |word: &str| -> Vec<u32> {
        let mut symbols = Vec::new();
        for c in word.chars() {
            match bpe.token_to_id(&c.to_string()) {
                Some(id) => symbols.push(id),
                None => {
                    let mut utf8 = [0u8; 4];
                    for byte in c.encode_utf8(&mut utf8).bytes() {
                        symbols.push(
                            bpe.token_to_id(&format!("<0x{byte:02X}>"))
                                .expect("a byte token"),
                        );
                    }
                }
            }
        }
        symbols
    };
    for word in [
        "▁the",
        "▁thequick",
        "▁quick▁the",
        "éé",
        "中中中",
        "▁éé",
        "thethe",
        "▁thé",
        "x",
        "😀",
        "▁😀the",
        "▁the▁quick▁brown▁fox▁the▁quick▁brown▁fox▁the▁quick▁brown▁fox",
    ] {
        let mut ids = Vec::new();
        table.merge_symbols(&symbols(word), &mut ids);
        let expected: Vec<u32> = bpe
            .tokenize(word)
            .expect("the model tokenizes the word")
            .into_iter()
            .map(|token| token.id)
            .collect();
        assert_eq!(ids, expected, "word {word:?}");
    }
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

/// The hot grid, indexed by id, answers as the model does on a vocabulary
/// that gives two byte characters one id: the model merges the pair by id
/// for either byte, and so does the table's first round from the grid. (The
/// merge's product is in the vocabulary under both spellings, on one id too:
/// the model names a merge by either byte when it serializes its merges.)
#[test]
fn two_byte_characters_on_one_id_merge_as_the_model_does() {
    let (mut vocab, table) = alphabet_vocab();
    let a = vocab["a"];
    vocab.insert("b".to_owned(), a);
    let ac = vocab.len() as u32;
    vocab.insert("ac".to_owned(), ac);
    vocab.insert("bc".to_owned(), ac);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![("a".to_owned(), "c".to_owned())])
        .build()
        .expect("a byte-level BPE");
    let merges = table_of(&bpe);
    for piece in [&b"ac"[..], b"bc", b"abc", b"cb"] {
        assert_eq!(
            table_ids(&merges, piece),
            model_ids(&bpe, &table, piece),
            "{piece:?}"
        );
    }
    assert_eq!(table_ids(&merges, b"bc"), [ac]);
}

/// The table is the hot grid (`HOT_SIDE` x `HOT_SIDE` words) and its
/// entries: a power of two of buckets, one cache line of four sixteen-byte
/// entries each, at a load of one half or less, the pair filter, and the
/// pre-formed character analysis counted apart.
#[test]
fn the_tables_bytes_are_the_grid_the_filter_and_its_buckets_at_half_load_or_less() {
    let bpe = model_with(&merges(), false);
    let table = table_of(&bpe);
    let merges = merges().len();
    let grid = (HOT_SIDE * HOT_SIDE) as usize * size_of::<u64>();
    let filter = table.pair_filter_bytes();
    assert!(
        filter * 8 >= PAIR_FILTER_BITS_PER_MERGE * merges && filter.is_power_of_two(),
        "{filter} filter bytes for {merges} merges"
    );
    let analysis = table.preformed.bytes();
    let entries = (table.bytes() - size_of::<MergeTable>() - grid - filter - analysis) / 16;
    assert!(entries.is_power_of_two(), "{entries} entries");
    assert!(
        entries >= 2 * merges,
        "{entries} entries for {merges} merges"
    );
    assert!(
        entries < 4 * merges + 8,
        "{entries} entries for {merges} merges"
    );
}

/// Every merge of the model is answered: a pair inside the hot grid's window
/// by the grid, one outside it by the pair filter (a false negative would
/// lose a merge) and its bucket; and pairs that are no merge mostly fail the
/// filter (a filter that admitted them all would save no bucket line).
#[test]
fn every_merge_passes_the_pair_filter_and_a_pair_that_is_none_mostly_does_not() {
    // Filler tokens push the merge products past the hot grid's window, so
    // that the merges among products reach the filter and its buckets.
    let fillers: Vec<String> = (0..HOT_SIDE).map(|i| format!("filler{i}")).collect();
    let fillers: Vec<&str> = fillers.iter().map(String::as_str).collect();
    let bpe = model_with_tokens(&merges(), &fillers);
    let table = table_of(&bpe);
    let id = |token: &str| bpe.token_to_id(token).expect("a token of the model");
    assert!(
        merges()
            .iter()
            .any(|&(a, b)| table.hot_index(id(a), id(b)).is_none()),
        "some merge must fall outside the hot window, or the filter goes untested"
    );
    for (a, b) in merges() {
        assert!(
            table.merge_of(id(a), id(b)) != Merge::NONE,
            "the merge ({a}, {b}) is not answered"
        );
        assert!(
            table.hot_index(id(a), id(b)).is_some() || table.pair_filter_admits(id(a), id(b)),
            "the merge ({a}, {b}) fails the filter"
        );
    }
    let ids = bpe.get_vocab_size() as u64;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut none, mut admitted) = (0, 0);
    while none < 20_000 {
        let (a, b) = ((next() % ids) as u32, (next() % ids) as u32);
        if table.hot_index(a, b).is_some() || table.merge_of(a, b) != Merge::NONE {
            continue;
        }
        none += 1;
        admitted += usize::from(table.pair_filter_admits(a, b));
    }
    assert!(
        admitted * 20 < none,
        "{admitted} of {none} pairs that are no merge pass the filter"
    );
}

/// A lookup walks a full bucket into the next one and stops at the first
/// empty entry it meets: pairs that hash to one bucket, one more than its
/// entries, are all found, and a pair that hashes there too and passes the
/// filter but was never inserted is none.
#[test]
fn a_lookup_walks_a_full_bucket_into_the_next_and_stops_at_the_first_empty_entry() {
    let bpe = model_with(&merges(), false);
    let mut table = table_of(&bpe);
    // Pairs of ids no token of the model has, grouped by their first bucket.
    let mut by_bucket: FxHashMap<usize, Vec<(u32, u32)>> = FxHashMap::default();
    for a in 1_000_000..1_100_000u32 {
        let key = MergeTable::pair_key(a, a + 1);
        by_bucket
            .entry(table.first_bucket(key))
            .or_default()
            .push((a, a + 1));
    }
    let pairs = by_bucket
        .values()
        .max_by_key(|pairs| pairs.len())
        .expect("a bucket with candidates");
    let (inserted, rest) = pairs.split_at(BUCKET + 1);
    for (rank, &(a, b)) in inserted.iter().enumerate() {
        table.insert(a, b, Merge::new(rank as u32, a));
    }
    for (rank, &(a, b)) in inserted.iter().enumerate() {
        assert_eq!(
            table.merge_of(a, b),
            Merge::new(rank as u32, a),
            "pair {rank}"
        );
    }
    // An absent pair that hashes to the full bucket and passes the filter
    // (its bits set as a merge's would be) is none: the walk crosses the
    // full bucket and ends at the next bucket's first empty entry.
    let &(a, b) = rest.first().expect("a pair left over");
    let (word, bits) = table.pair_filter_slot(MergeTable::pair_key(a, b));
    table.pair_filter[word] |= bits;
    assert!(table.pair_filter_admits(a, b));
    assert_eq!(table.merge_of(a, b), Merge::NONE);
}

/// A byte-fallback vocabulary may number its tokens anywhere in the id
/// space, and its table has no hot grid: no pair is answered from the grid
/// whatever the ids, so operands at 2^31 load and merge through the
/// buckets (a window base chosen as a sentinel would have indexed the
/// absent grid at load).
#[test]
fn a_table_without_the_hot_grid_answers_no_pair_from_it_whatever_the_ids() {
    let mut vocab = Vocab::default();
    vocab.insert("<unk>".to_owned(), 0);
    for byte in 0..=255u8 {
        vocab.insert(format!("<0x{byte:02X}>"), 1 + u32::from(byte));
    }
    let (a, b, ab) = (1 << 31, (1 << 31) + 1, (1 << 31) + 2);
    vocab.insert("a".to_owned(), a);
    vocab.insert("b".to_owned(), b);
    vocab.insert("ab".to_owned(), ab);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![("a".to_owned(), "b".to_owned())])
        .byte_fallback(true)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("a byte-fallback BPE");
    let table = MergeTable::from_model(&ModelWrapper::BPE(bpe), ByteTokens::Hex, None)
        .expect("the table loads with ids at 2^31");
    assert!(table.hot.is_empty());
    assert_eq!(table.hot_index(a, b), None);
    assert_eq!(table.merge_of(a, b), Merge::new(0, ab));
    assert_eq!(table.merge_of(a, a), Merge::NONE);
}

/// At a vocabulary's size and the tightest rounding of the filter (131,072
/// merges: exactly eight bits per merge), every inserted pair passes and
/// about one pair in twenty that is no merge does: a filter whose density
/// or hash degraded (bits per merge down, bit positions collapsed) would
/// admit far more.
#[test]
fn the_pair_filter_admits_about_one_absent_pair_in_twenty_at_a_vocabularys_size() {
    const MERGES: usize = 131_072;
    const IDS: u64 = 160_000;
    let mut state = 0x6A09_E667_F3BC_C909u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut table = MergeTable::with_capacity(MERGES, false, false);
    let mut pairs = HashSet::new();
    while pairs.len() < MERGES {
        let (a, b) = ((next() % IDS) as u32, (next() % IDS) as u32);
        if pairs.insert((a, b)) {
            let rank = pairs.len() as u32 - 1;
            table.insert(a, b, Merge::new(rank, IDS as u32 + rank));
        }
    }
    assert!(
        table.pair_filter_bytes() * 8 == PAIR_FILTER_BITS_PER_MERGE * MERGES,
        "{} filter bytes for {MERGES} merges",
        table.pair_filter_bytes()
    );
    for &(a, b) in &pairs {
        assert!(
            table.pair_filter_admits(a, b),
            "the merge ({a}, {b}) fails the filter"
        );
    }
    let (mut none, mut admitted) = (0usize, 0usize);
    while none < 200_000 {
        let (a, b) = ((next() % IDS) as u32, (next() % IDS) as u32);
        if pairs.contains(&(a, b)) {
            continue;
        }
        none += 1;
        admitted += usize::from(table.pair_filter_admits(a, b));
    }
    let rate = admitted as f64 / none as f64;
    assert!(
        (0.03..0.07).contains(&rate),
        "{admitted} of {none} absent pairs pass the filter ({:.2} %)",
        100.0 * rate
    );
}
/// A vocabulary of random merges over a corpus of whole characters of
/// several multi-byte scripts and ASCII, so that character tokens, tokens of
/// several characters and fragment tokens (a token ending inside a character,
/// one starting with continuation bytes) all arise; `seed` varies the draw.
fn random_character_model(seed: u64, merges: usize) -> (BPE, [char; 256], Vec<u8>) {
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut vocab, alphabet) = alphabet_vocab();
    // Ideographs, kana, Hangul, Cyrillic, Greek, a two-byte Latin letter,
    // an emoji, ASCII letters and a space: whole characters, so that the
    // vocabulary grows tokens of them and of their parts.
    let pool: &[&str] = &[
        "中", "文", "日", "本", "語", "の", "한", "국", "п", "р", "и", "в", "ε", "é", "😀", "a",
        "b", " ",
    ];
    let corpus: Vec<u8> = (0..4000)
        .flat_map(|_| pool[next() as usize % pool.len()].as_bytes().to_vec())
        .collect();
    let mut list = Merges::new();
    let mut seen = HashSet::new();
    let mapped_corpus = mapped(&alphabet, &corpus);
    let mut tokens: Vec<String> = mapped_corpus.chars().map(|c| c.to_string()).collect();
    // About three of four multi-byte characters of the pool are whole tokens
    // before the random merges (their byte merges rank first), the rest stay
    // loose bytes: the random merges then join anything adjacent, so whole
    // characters pre-form, multi-character tokens arise, and fragment
    // tokens arise around the loose bytes and decline their characters.
    let mut whole: Vec<String> = Vec::new();
    for entry in pool {
        let m = mapped(&alphabet, entry.as_bytes());
        if m.chars().count() > 1 && next() % 4 != 0 {
            whole.push(m);
        }
    }
    for m in &whole {
        let parts: Vec<String> = m.chars().map(|c| c.to_string()).collect();
        let mut acc = parts[0].clone();
        for part in &parts[1..] {
            let merged = format!("{acc}{part}");
            if seen.insert((acc.clone(), part.clone())) {
                if !vocab.contains_key(&merged) {
                    let id = vocab.len() as u32;
                    vocab.insert(merged.clone(), id);
                }
                list.push((acc.clone(), part.clone()));
            }
            acc = merged;
        }
    }
    let mut rebuilt: Vec<String> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let run = whole.iter().find(|m| {
            let n = m.chars().count();
            i + n <= tokens.len() && tokens[i..i + n].concat() == **m
        });
        match run {
            Some(m) => {
                i += m.chars().count();
                rebuilt.push(m.clone());
            }
            None => {
                rebuilt.push(tokens[i].clone());
                i += 1;
            }
        }
    }
    tokens = rebuilt;
    while list.len() < merges {
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
    (bpe, alphabet, corpus)
}

#[test]
fn pieces_of_multi_byte_scripts_merge_to_the_models_ids_from_pre_formed_characters() {
    // Several draws: in each, some characters pre-form and some are declined
    // (a merge taking the character ranks early, or a fragment token covers
    // part of it next to the bytes the corpus puts beside it), and pieces cut
    // at any byte, inside characters too, must merge to the model's ids.
    let mut pre_formed_somewhere = 0;
    let mut answered_somewhere = 0;
    for seed in [
        0x2545_F491_4F6C_DD1Du64,
        0x9E37_79B9_7F4A_7C15,
        0xD6E8_FEB8_6659_FD93,
    ] {
        let (bpe, alphabet, corpus) = random_character_model(seed, 900);
        let table = table_of(&bpe);
        pre_formed_somewhere += table.preformed_characters();
        let mut state = seed ^ 0xA5A5;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..3000 {
            let len = 1 + round % 300;
            let start = next() as usize % (corpus.len() - len);
            let piece = &corpus[start..start + len];
            let (ids, answered) = table_ids_preformed(&table, piece);
            assert_eq!(
                ids,
                model_ids(&bpe, &alphabet, piece),
                "seed {seed:#x} piece {:?}",
                String::from_utf8_lossy(piece)
            );
            answered_somewhere += usize::from(answered);
        }
        assert!(
            answered_somewhere > 0,
            "seed {seed:#x}: the pre-formed path answered no piece"
        );
        answered_somewhere = 0;
    }
    assert!(pre_formed_somewhere > 0, "no draw pre-formed a character");
}

#[test]
fn a_character_pre_forms_only_under_the_three_conditions() {
    // "中" is E4 B8 AD; its alphabet form is "ä¸Ń". (E4, B8) then (E4B8, AD)
    // form it: a pre-formable character when nothing else touches its bytes.
    let (_, alphabet) = alphabet_vocab();
    let zhong = mapped(&alphabet, "中".as_bytes());
    let (e4, b8, ad) = (
        zhong.chars().next().expect("E4").to_string(),
        zhong.chars().nth(1).expect("B8").to_string(),
        zhong.chars().nth(2).expect("AD").to_string(),
    );
    let e4b8 = format!("{e4}{b8}");
    // (1) and (2) hold, no fragment: pre-formed.
    let bpe = model_with(
        &[(&e4, &b8), (&e4b8, &ad), (&zhong, &zhong)].map(|(a, b)| (a.as_str(), b.as_str())),
        false,
    );
    let table = table_of(&bpe);
    assert_eq!(table.preformed_characters(), 1);
    let (ids, answered) = table_ids_preformed(&table, "中中".as_bytes());
    assert!(
        answered,
        "the pre-formed path answers a piece of pre-formed characters"
    );
    assert_eq!(ids, model_ids(&bpe, &alphabet, "中中".as_bytes()));
    // (2) fails: a merge taking the character ranks before the merge that
    // completes it ((中, 中) at rank 1, (E4B8, AD) at rank 2).
    let bpe = model_with(
        &[(&e4, &b8), (&zhong, &zhong), (&e4b8, &ad)].map(|(a, b)| (a.as_str(), b.as_str())),
        false,
    );
    // A merge that takes the character as an operand ranks before the merge
    // that completes it: the character is not pre-formed (the bytes merge
    // as the model's do) and the ids agree.
    let table = table_of(&bpe);
    assert_eq!(table.preformed_characters(), 0);
    assert!(!table_ids_preformed(&table, "中中".as_bytes()).1);
    assert_eq!(
        table_ids_preformed(&table, "中中".as_bytes()).0,
        model_ids(&bpe, &alphabet, "中中".as_bytes())
    );
    // (1) fails: the token exists but the bytes alone do not reach it (no
    // (E4B8, AD) merge).
    let bpe = model_with_tokens(
        &[(&e4, &b8), (&zhong, &zhong)].map(|(a, b)| (a.as_str(), b.as_str())),
        &[zhong.as_str()],
    );
    assert_eq!(table_of(&bpe).preformed_characters(), 0);
    // (3): a fragment token "a" + E4 B8 (a token ending inside the character)
    // declines the occurrence after an "a" but not elsewhere; the ids agree
    // either way.
    let bpe = model_with(
        &[
            (e4.as_str(), b8.as_str()),
            ("a", e4b8.as_str()),
            (e4b8.as_str(), ad.as_str()),
            (zhong.as_str(), zhong.as_str()),
        ],
        false,
    );
    let table = table_of(&bpe);
    assert_eq!(table.preformed_characters(), 1);
    // The one character after an "a" is declined and the piece merges from
    // its bytes; a run that starts with the character is answered.
    assert!(!table_ids_preformed(&table, "a中".as_bytes()).1);
    assert!(table_ids_preformed(&table, "中中a".as_bytes()).1);
    for text in ["a中中", "b中中", "中中a", "a中", "中a中"] {
        assert_eq!(
            table_ids_preformed(&table, text.as_bytes()).0,
            model_ids(&bpe, &alphabet, text.as_bytes()),
            "{text}"
        );
    }
}

/// A rank vocabulary pre-forms a character under the same three conditions
/// as a model's, to the file's own encoder's ids; the model table of the
/// same vocabulary pre-forms the same character and answers the same
/// pieces, token for token.
#[test]
fn a_rank_vocabulary_pre_forms_its_characters() {
    let (_, alphabet) = alphabet_vocab();
    let zhong = mapped(&alphabet, "中".as_bytes());
    let [e4, b8, ad] = [0, 1, 2].map(|i| zhong.chars().nth(i).expect("a byte").to_string());
    let e4b8 = format!("{e4}{b8}");
    // (1) and (2) hold, no fragment: pre-formed, and the pre-formed path
    // answers a run of the character to the encoder's ids.
    let merges =
        [(&e4, &b8), (&e4b8, &ad), (&zhong, &zhong)].map(|(a, b)| (a.as_str(), b.as_str()));
    let ranks = ranks_of(&merges, false);
    let table = MergeTable::from_ranks(&ranks).expect("a complete table");
    assert_eq!(table.preformed_characters(), 1);
    let model = model_with(&merges, false);
    let from_model = table_of(&model);
    assert_eq!(from_model.preformed_characters(), 1);
    let own = own_encoder(&ranks);
    let token_of_rank: FxHashMap<u32, &Vec<u8>> =
        ranks.iter().map(|(token, &rank)| (rank, token)).collect();
    for text in ["中", "中中", "中中中", "a中", "中a中"] {
        let (ids, answered) = table_ids_preformed(&table, text.as_bytes());
        assert!(answered, "{text}");
        assert_eq!(ids, own.encode_ordinary(text), "{text}");
        // The model table answers the same pieces to the same tokens (its
        // ids name them by their alphabet spelling).
        let (model_ids, model_answered) = table_ids_preformed(&from_model, text.as_bytes());
        assert!(model_answered, "model: {text}");
        let via_model: Vec<Vec<u8>> = model_ids
            .iter()
            .map(|&id| unmapped(&alphabet, &model.id_to_token(id).expect("a token")))
            .collect();
        let via_ranks: Vec<Vec<u8>> = ids.iter().map(|id| token_of_rank[id].to_vec()).collect();
        assert_eq!(via_model, via_ranks, "model: {text}");
    }
    // (2) fails: (中, 中) ranks before (E4B8, AD): the character is not
    // pre-formed, the bytes merge as the encoder's do.
    let ranks = ranks_of(
        &[(&e4, &b8), (&zhong, &zhong), (&e4b8, &ad)].map(|(a, b)| (a.as_str(), b.as_str())),
        false,
    );
    let table = MergeTable::from_ranks(&ranks).expect("a complete table");
    assert_eq!(table.preformed_characters(), 0);
    let own = own_encoder(&ranks);
    for text in ["中中", "a中"] {
        let (ids, answered) = table_ids_preformed(&table, text.as_bytes());
        assert!(!answered, "{text}");
        assert_eq!(ids, own.encode_ordinary(text), "{text}");
    }
}

#[test]
fn a_run_of_one_character_overflows_the_stack_arrays_and_a_pair_no_token_holds_is_cut() {
    // 中 pre-forms and (中, 中) merges: a run of n 中 is one segment of n
    // symbols, merged on the stack arrays up to SHORT_PIECE_SYMBOLS and from
    // its bytes beyond, to the model's ids either way.
    let (_, alphabet) = alphabet_vocab();
    let zhong = mapped(&alphabet, "中".as_bytes());
    let wen = mapped(&alphabet, "文".as_bytes());
    let [e4, b8, ad] = [0, 1, 2].map(|i| zhong.chars().nth(i).expect("a byte").to_string());
    let [e6, x96, x87] = [0, 1, 2].map(|i| wen.chars().nth(i).expect("a byte").to_string());
    let e4b8 = format!("{e4}{b8}");
    let e696 = format!("{e6}{x96}");
    let bpe = model_with(
        &[(&e4, &b8), (&e4b8, &ad), (&zhong, &zhong)].map(|(a, b)| (a.as_str(), b.as_str())),
        false,
    );
    let table = table_of(&bpe);
    assert_eq!(table.preformed_characters(), 1);
    for n in [
        1,
        2,
        3,
        SHORT_PIECE_SYMBOLS - 1,
        SHORT_PIECE_SYMBOLS,
        SHORT_PIECE_SYMBOLS + 1,
        200,
        300,
    ] {
        let piece = "中".repeat(n);
        let (ids, answered) = table_ids_preformed(&table, piece.as_bytes());
        assert!(answered, "{n} characters");
        assert_eq!(
            ids,
            model_ids(&bpe, &alphabet, piece.as_bytes()),
            "{n} characters"
        );
    }
    // 中 and 文 pre-form and no token holds them adjacent: the walk cuts
    // between them and each segment merges alone, to the model's ids.
    let bpe = model_with(
        &[(&e4, &b8), (&e4b8, &ad), (&e6, &x96), (&e696, &x87)]
            .map(|(a, b)| (a.as_str(), b.as_str())),
        false,
    );
    let table = table_of(&bpe);
    assert_eq!(table.preformed_characters(), 2);
    for text in ["中文", "中文中文", "文中", "a中文b"] {
        let (ids, answered) = table_ids_preformed(&table, text.as_bytes());
        assert!(answered, "{text}");
        assert_eq!(ids, model_ids(&bpe, &alphabet, text.as_bytes()), "{text}");
    }
}

/// The model as its section reads into the loading model (the one-pass
/// read the loader's tables take their merges and largest id from).
fn read_of(bpe: &BPE) -> crate::word_cache::LateBoundModel {
    let json =
        serde_json::to_string(&ModelWrapper::BPE(bpe.clone())).expect("the model serializes");
    serde_json::from_str(&json).expect("the loading model reads its section")
}

/// The model's section as JSON text, read into the loading model without
/// the built model's serializer, which walks every id up to the largest
/// (a vocabulary numbered at 2^31 would stall there).
fn section_of(
    vocab: &Vocab,
    merges: &[(&str, &str)],
    byte_fallback: bool,
    unk_token: Option<&str>,
) -> crate::word_cache::LateBoundModel {
    let vocab: std::collections::BTreeMap<&str, u32> = vocab
        .iter()
        .map(|(token, &id)| (token.as_str(), id))
        .collect();
    let section = serde_json::json!({
        "type": "BPE",
        "unk_token": unk_token,
        "byte_fallback": byte_fallback,
        "vocab": vocab,
        "merges": merges,
    });
    serde_json::from_value(section).expect("the loading model reads its section")
}

/// The one-pass read of a model section holds the merges as the ids the
/// built model resolves them to and the vocabulary's largest id, and the
/// table built from it is the table built from the model; a merge whose
/// product the vocabulary spells otherwise (a subword prefix) leaves the
/// pairs unread, and the table declines from the model as without a read.
#[test]
fn the_read_of_a_section_builds_the_table_the_model_builds() {
    use crate::word_cache::ModelView;
    let bpe = model_with(&[("a", "b"), ("ab", "c"), ("d", "e")], false);
    let late = read_of(&bpe);
    let read = late.read().expect("a BPE section leaves a read");
    let merges = MergeTable::merges_of(late.wrapper(), &bpe).expect("the merges resolve");
    assert_eq!(read.pairs.as_deref(), Some(merges.as_slice()));
    assert_eq!(read.max_id, bpe.get_vocab().values().copied().max());
    let with = MergeTable::from_model(late.wrapper(), ByteTokens::Alphabet, Some(read))
        .expect("the table loads from the read");
    let without = MergeTable::from_model(late.wrapper(), ByteTokens::Alphabet, None)
        .expect("the table loads from the model");
    assert_eq!(
        (with.merges(), with.bytes(), with.preformed_characters()),
        (
            without.merges(),
            without.bytes(),
            without.preformed_characters()
        )
    );
    for piece in [&b"abc"[..], b"ab", b"cab", b"de", b"xyz", b"abcde"] {
        assert_eq!(
            table_ids(&with, piece),
            table_ids(&without, piece),
            "{piece:?}"
        );
    }

    let mut vocab = Vocab::default();
    for (id, token) in ["a", "_b", "ab"].into_iter().enumerate() {
        vocab.insert(token.to_owned(), id as u32);
    }
    let prefixed = BPE::builder()
        .vocab_and_merges(vocab, vec![("a".to_owned(), "_b".to_owned())])
        .continuing_subword_prefix("_".to_owned())
        .build()
        .expect("a BPE with a subword prefix");
    let late = read_of(&prefixed);
    let read = late.read().expect("a BPE section leaves a read");
    assert_eq!(
        read.pairs, None,
        "a merge the vocabulary spells otherwise is left to the model"
    );
    assert_eq!(read.max_id, Some(2));
    assert_eq!(
        MergeTable::from_model(late.wrapper(), ByteTokens::Alphabet, Some(read)).err(),
        MergeTable::from_model(late.wrapper(), ByteTokens::Alphabet, None).err()
    );
}

/// The entries read by id are the vocabulary's entries when the ids are the
/// range `0..n`; a vocabulary numbered elsewhere (a byte-fallback one at
/// 2^31) or two tokens on one id take the vocabulary walk, as a model
/// without a read.
#[test]
fn the_entries_by_id_are_the_vocabulary_walk_or_the_walk_itself() {
    use crate::word_cache::{entries_of, ModelView};
    fn sorted(mut entries: Vec<(u32, String)>) -> Vec<(u32, String)> {
        entries.sort();
        entries
    }
    let walk = |bpe: &BPE| {
        sorted(
            bpe.get_vocab()
                .into_iter()
                .map(|(token, id)| (id, token))
                .collect(),
        )
    };

    let dense = model_with(&[("a", "b")], false);
    let late = read_of(&dense);
    assert!(late.read().is_some_and(|read| read.dense));
    assert!(
        late.read().is_some_and(|read| read.tokens.is_some()),
        "a dense vocabulary is listed"
    );
    assert_eq!(
        sorted(
            entries_of(&dense, late.read())
                .map(|(id, token)| (id, token.into_owned()))
                .collect()
        ),
        walk(&dense)
    );
    assert_eq!(
        sorted(
            entries_of(&dense, None)
                .map(|(id, token)| (id, token.into_owned()))
                .collect()
        ),
        walk(&dense)
    );

    let mut vocab = Vocab::default();
    vocab.insert("<unk>".to_owned(), 0);
    for byte in 0..=255u8 {
        vocab.insert(format!("<0x{byte:02X}>"), 1 + u32::from(byte));
    }
    vocab.insert("a".to_owned(), 1 << 31);
    vocab.insert("b".to_owned(), (1 << 31) + 1);
    vocab.insert("ab".to_owned(), (1 << 31) + 2);
    let late = section_of(&vocab, &[("a", "b")], true, Some("<unk>"));
    let sparse = BPE::builder()
        .vocab_and_merges(vocab, vec![("a".to_owned(), "b".to_owned())])
        .byte_fallback(true)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("a byte-fallback BPE numbered at 2^31");
    assert!(late.read().is_some_and(|read| !read.dense));
    assert_eq!(
        late.read().and_then(|read| read.max_id),
        Some((1 << 31) + 2)
    );
    assert_eq!(
        sorted(
            entries_of(&sparse, late.read())
                .map(|(id, token)| (id, token.into_owned()))
                .collect()
        ),
        walk(&sparse)
    );

    let mut vocab = Vocab::default();
    for (token, id) in [("a", 0), ("b", 0), ("c", 2)] {
        vocab.insert(token.to_owned(), id);
    }
    // From the section's text: the built model's serializer writes one token
    // per id, so a read through it would see a gap at 1 rather than the
    // shared id 0, and the gap check, not the duplicate check, would clear
    // `dense`.
    let late = section_of(&vocab, &[], false, None);
    let doubled = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .expect("a BPE with two tokens on one id");
    assert_eq!(late.read().and_then(|read| read.max_id), Some(2));
    assert!(late.read().is_some_and(|read| !read.dense));
    let entries = sorted(
        entries_of(&doubled, late.read())
            .map(|(id, token)| (id, token.into_owned()))
            .collect(),
    );
    assert_eq!(entries.len(), 3, "both tokens of the shared id");
    assert_eq!(entries, walk(&doubled));
    assert!(
        late.read().is_some_and(|read| read.tokens.is_none()),
        "two tokens on one id are walked, not listed"
    );
}

/// The listing holds every token of a dense vocabulary under its id, in
/// one buffer, and nothing past the end.
#[test]
fn the_listing_holds_every_token_under_its_id() {
    use crate::word_cache::Listing;
    let mut vocab = Vocab::default();
    for (id, token) in ["a", "b", "ab", "\u{e9}", "a\u{e9}"]
        .into_iter()
        .enumerate()
    {
        vocab.insert(token.to_owned(), id as u32);
    }
    let listing = Listing::of(&vocab).expect("a dense vocabulary lists");
    assert_eq!(listing.len(), 5);
    for (token, &id) in &vocab {
        assert_eq!(listing.get(id), Some(token.as_str()));
    }
    assert_eq!(listing.get(5), None, "past the end");
    let mut sparse = Vocab::default();
    sparse.insert("a".to_owned(), 0);
    sparse.insert("b".to_owned(), 2);
    assert!(Listing::of(&sparse).is_none(), "a gap: no listing");
    let mut twice = Vocab::default();
    twice.insert("a".to_owned(), 0);
    twice.insert("b".to_owned(), 0);
    assert!(
        Listing::of(&twice).is_none(),
        "two tokens on one id: no listing"
    );
}
