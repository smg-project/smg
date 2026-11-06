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
    super::bytes_char, ByteTokens, MergeTable, SHORT_PIECE_SMALL, SHORT_PIECE_SYMBOLS,
    SHORT_PIECE_TINY,
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

/// The ids the model's own merge gives a piece.
fn model_ids(bpe: &BPE, table: &[char; 256], piece: &[u8]) -> Vec<u32> {
    bpe.tokenize(&mapped(table, piece))
        .expect("the model tokenizes every alphabet string")
        .into_iter()
        .map(|token| token.id)
        .collect()
}

fn table_of(bpe: &BPE) -> MergeTable {
    MergeTable::from_model(&ModelWrapper::BPE(bpe.clone()), ByteTokens::Alphabet)
        .expect("the table loads")
}

fn table_ids(table: &MergeTable, piece: &[u8]) -> Vec<u32> {
    let mut ids = Vec::new();
    table.merge(piece, &mut ids);
    ids
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
        MergeTable::from_model(&ModelWrapper::BPE(bpe), ByteTokens::Alphabet)
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
        MergeTable::from_model(&word_level, ByteTokens::Alphabet)
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
    let table = MergeTable::from_model(&ModelWrapper::BPE(bpe.clone()), ByteTokens::Hex)
        .expect("the table loads with hex byte tokens");
    assert_eq!(table.merges(), merges.len());
    // No merge of such a table starts from bytes: the byte-pair grid is not
    // built (512 KiB per loaded vocabulary otherwise).
    assert!(
        table.byte_grid.is_empty(),
        "a table built for character symbols carries no byte grid"
    );
    assert!(table.bytes() < 256 * 256 * size_of::<u64>());
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

/// The byte-pair grid answers as the bucket table does on a vocabulary
/// that gives two byte characters one id: the model merges the pair by id
/// for either byte, and so does the table's first round. (The merge's
/// product is in the vocabulary under both spellings, on one id too: the
/// model names a merge by either byte when it serializes its merges.)
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

/// The table is the byte-pair grid (256 x 256 words) and its entries: a
/// power of two of buckets, one cache line of four sixteen-byte entries
/// each, at a load of one half or less.
#[test]
fn the_tables_bytes_are_the_grid_and_its_buckets_at_half_load_or_less() {
    let bpe = model_with(&merges(), false);
    let table = table_of(&bpe);
    let merges = merges().len();
    let grid = 256 * 256 * size_of::<u64>();
    let entries = (table.bytes() - size_of::<MergeTable>() - grid) / 16;
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
