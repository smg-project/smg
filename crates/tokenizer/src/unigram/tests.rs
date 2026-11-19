//! The direct Unigram encoder against `tokenizers` itself, on a small
//! tokenizer built in the test: the best path of a word and the paths of
//! equal score, characters no piece covers, the word marker, a character map
//! with an entry of every kind over every short text of the characters it
//! touches, the added tokens, the frame, and the configurations the encoder
//! declines.

use std::cell::Cell;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};
use tokenizers::{models::ModelWrapper, tokenizer::Model, Tokenizer};

use super::UnigramEncoder;
use crate::{
    encode_path::EncodePath,
    huggingface::HuggingFaceTokenizer,
    traits::{Encoder, Encoding},
};

/// The word marker.
const MARKER: &str = "\u{2581}";

/// The added tokens, at the ids 0 to 2. The last is the unknown piece.
const SPECIAL: &[&str] = &["<pad>", "</s>", "<unk>"];

/// The pieces behind the added tokens, each with its score. A letter costs
/// 2, so `ab`, `bc` and `abc` score as their letters do, and the marker
/// with `a` as the two apart: which of two paths of one score a word takes
/// is the library's rule. `ca` and the marker with `ab` score higher than
/// their parts, `aa` lower.
const PIECES: &[(&str, f64)] = &[
    ("\u{2581}", -2.0),
    ("a", -2.0),
    ("b", -2.0),
    ("c", -2.0),
    ("ab", -4.0),
    ("bc", -4.0),
    ("abc", -6.0),
    ("ca", -3.0),
    ("aa", -5.0),
    ("\u{2581}a", -4.0),
    ("\u{2581}ab", -5.0),
    // What the character map of `MAP` makes of its entries.
    ("x", -2.0),
    ("y", -2.0),
    ("z", -2.0),
    ("A", -2.0),
    ("\u{4e2d}", -3.0),
];

/// The entries of the character map, one of each kind: a control character
/// that becomes a space, a carriage return (one cluster with a line feed
/// after it), a character that takes the next one into its cluster, a
/// letter and its mark as one entry, a compatibility form, and a character
/// that goes.
const MAP: &[(&str, &str)] = &[
    ("\t", " "),
    ("\r", "x"),
    ("\u{600}", "y"),
    ("a\u{301}", "z"),
    ("\u{ff21}", "A"),
    ("\u{ad}", ""),
];

/// Texts that reach every step: nothing, whitespace of every kind and in
/// runs, control characters, marks and clusters, characters no piece
/// covers, the marker and the added tokens typed in the text.
const TEXTS: &[&str] = &[
    "",
    " ",
    "abc",
    "a b  c\tab\nbc\r\ncab",
    "  abc  ",
    "\u{a0}a\u{3000}b\u{2028}c",
    "a\u{1}b\u{7f}c",
    "a\u{301}b a\u{301}\u{301} \u{301}a",
    "\u{600}a \u{600}\u{600}b",
    "\u{ff21}\u{ff21}b \u{ad}a\u{ad}",
    "\u{4e2d}\u{6587} a\u{4e2d}b\u{6587}\u{6587}c",
    "\u{1f600} a\u{1f600}b \u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}",
    "\u{2581}a a\u{2581}b \u{2581}\u{2581} c\u{2581}",
    "a</s>b <pad> </s></s> <unk>c",
    "<pad",
    "x\u{e0000}y\u{10ffff}z",
];

/// A precompiled character map of `entries` (a text and what it becomes) as
/// a tokenizer file carries it: the size of the trie in bytes, the trie,
/// then the replacements, each ended by a zero byte. The trie is a double
/// array laid out the plain way: the node of every prefix of an entry has a
/// block of 256 units of its own, one for each byte that may follow it, and
/// the first unit of the block holds the place of the replacement when the
/// prefix is a whole entry.
fn character_map(entries: &[(&str, &str)]) -> String {
    const HAS_VALUE: u32 = 1 << 8;
    const VALUE: u32 = 1 << 31;
    let unit_of = |value: usize| u32::try_from(value).expect("a small character map");
    // The prefixes, the empty one first: each is a node, and node `n` has
    // block `n + 1`.
    let mut prefixes: Vec<&[u8]> = vec![&[]];
    for (from, _) in entries {
        for end in 1..=from.len() {
            let prefix = &from.as_bytes()[..end];
            if !prefixes.contains(&prefix) {
                prefixes.push(prefix);
            }
        }
    }
    let block_of = |prefix: &[u8]| {
        let node = prefixes
            .iter()
            .position(|known| *known == prefix)
            .expect("a prefix of an entry");
        256 * (node + 1)
    };
    let mut units = vec![0u32; 256 * (prefixes.len() + 1)];
    let mut replacements = Vec::new();
    for prefix in &prefixes {
        // The unit of a node lives in the block of its parent, at its last
        // byte, and the root at 0. It names that byte and, as an offset
        // from its own place, its block.
        let (at, byte) = match prefix.split_last() {
            None => (0, 0),
            Some((last, parent)) => (block_of(parent) + usize::from(*last), u32::from(*last)),
        };
        let block = block_of(prefix);
        units[at] = unit_of(at ^ block) << 10 | byte;
        if let Some((_, to)) = entries.iter().find(|(from, _)| from.as_bytes() == *prefix) {
            units[at] |= HAS_VALUE;
            units[block] = VALUE | unit_of(replacements.len());
            replacements.extend_from_slice(to.as_bytes());
            replacements.push(0);
        }
    }
    let mut bytes = unit_of(units.len() * 4).to_le_bytes().to_vec();
    bytes.extend(units.iter().flat_map(|unit| unit.to_le_bytes()));
    bytes.extend(replacements);
    STANDARD.encode(bytes)
}

/// A tokenizer file of the shape the encoder takes.
fn definition() -> Value {
    let added: Vec<Value> = SPECIAL
        .iter()
        .enumerate()
        .map(|(id, content)| {
            json!({
                "id": id,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let vocabulary: Vec<Value> = SPECIAL
        .iter()
        .map(|piece| json!([piece, 0.0]))
        .chain(PIECES.iter().map(|(piece, score)| json!([piece, score])))
        .collect();
    let metaspace = json!({
        "type": "Metaspace",
        "replacement": MARKER,
        "prepend_scheme": "always",
        "split": true,
    });
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Precompiled", "precompiled_charsmap": character_map(MAP)},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [{"type": "WhitespaceSplit"}, metaspace.clone()],
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "special_tokens": {
                "</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]},
            },
        },
        "decoder": metaspace,
        "model": {
            "type": "Unigram",
            "unk_id": 2,
            "vocab": vocabulary,
            "byte_fallback": false,
        },
    })
}

/// `definition()` with `value` at `place`, a JSON pointer.
fn changed(place: &str, value: Value) -> Value {
    let mut definition = definition();
    *definition
        .pointer_mut(place)
        .expect("a place of the definition") = value;
    definition
}

/// The library's tokenizer of `definition`.
fn library(definition: &Value) -> Tokenizer {
    definition
        .to_string()
        .parse()
        .expect("the definition is a tokenizer")
}

/// The library's tokenizer of `definition` and the crate's over the same.
fn pair(definition: &Value) -> (Tokenizer, HuggingFaceTokenizer) {
    let library = library(definition);
    let ours = HuggingFaceTokenizer::from_tokenizer(library.clone());
    (library, ours)
}

fn library_ids(library: &Tokenizer, text: &str, special: bool) -> Vec<u32> {
    library
        .encode(text, special)
        .expect("the library encodes")
        .get_ids()
        .to_vec()
}

/// The crate's ids of `text`, which must come from the direct encoder.
fn direct_ids(ours: &HuggingFaceTokenizer, text: &str, special: bool) -> Vec<u32> {
    match ours.encode(text, special).expect("the crate encodes") {
        Encoding::Plain(ids) => ids,
        other => panic!("{text:?} left the direct encoder: {other:?}"),
    }
}

/// The reason the encoder declines `definition`, if it does.
fn declined(definition: &Value) -> Option<String> {
    UnigramEncoder::from_tokenizer(&*library(definition))
        .err()
        .map(|declined| declined.to_string())
}

/// Every text of up to `longest` of the `letters`, the empty one first.
fn texts_of(letters: &[&str], longest: usize) -> Vec<String> {
    let mut texts = vec![String::new()];
    let mut last = texts.clone();
    for _ in 0..longest {
        last = last
            .iter()
            .flat_map(|text| letters.iter().map(move |letter| format!("{text}{letter}")))
            .collect();
        texts.extend(last.iter().cloned());
    }
    texts
}

#[test]
fn a_word_takes_the_path_of_the_highest_score() {
    let (library, ours) = pair(&definition());
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    assert_eq!(ours.encode_path(), EncodePath::Native);
    // `bc` is the longest piece at `b`, and `b` with `ca` scores higher.
    assert_eq!(
        direct_ids(&ours, "bca", false),
        [id(MARKER), id("b"), id("ca")]
    );
    // The marker with `ab` scores higher than the marker and `abc`.
    assert_eq!(direct_ids(&ours, "abc", false), [id("\u{2581}ab"), id("c")]);
    // Of two paths of one score, the one whose last piece starts first.
    assert_eq!(direct_ids(&ours, "a", false), [id("\u{2581}a")]);
    assert_eq!(direct_ids(&ours, "bc", false), [id(MARKER), id("bc")]);
}

#[test]
fn every_short_word_encodes_as_the_library_does() {
    let (library, ours) = pair(&definition());
    for text in texts_of(&["a", "b", "c"], 7) {
        assert_eq!(
            direct_ids(&ours, &text, false),
            library_ids(&library, &text, false),
            "{text:?}"
        );
    }
}

#[test]
fn characters_no_piece_covers_are_one_unknown_piece() {
    let (library, ours) = pair(&definition());
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    assert_eq!(
        direct_ids(&ours, "a\u{6587}b", false),
        [id("\u{2581}a"), id("<unk>"), id("b")]
    );
    // Two in a row are one piece.
    assert_eq!(
        direct_ids(&ours, "a\u{6587}\u{1f600}b", false),
        [id("\u{2581}a"), id("<unk>"), id("b")]
    );
    // A word of them alone keeps its marker.
    assert_eq!(
        direct_ids(&ours, "\u{6587} \u{6587}", false),
        [id(MARKER), id("<unk>"), id(MARKER), id("<unk>")]
    );
}

#[test]
fn the_marker_starts_every_word_and_every_piece_after_a_marker() {
    let (library, ours) = pair(&definition());
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    // A run of whitespace of any kind ends a word once.
    assert_eq!(
        direct_ids(&ours, " a \u{3000}\n b ", false),
        [id("\u{2581}a"), id(MARKER), id("b")]
    );
    // A marker typed in the text starts a piece as the one before a word
    // does, and a word that starts with one gets no second.
    assert_eq!(
        direct_ids(&ours, "a\u{2581}b", false),
        [id("\u{2581}a"), id(MARKER), id("b")]
    );
    assert_eq!(direct_ids(&ours, "\u{2581}a", false), [id("\u{2581}a")]);
    assert_eq!(
        direct_ids(&ours, "\u{2581}\u{2581}", false),
        [id(MARKER), id(MARKER)]
    );
}

#[test]
fn the_character_map_changes_a_text_as_the_library_does() {
    let (library, ours) = pair(&definition());
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    // A carriage return and the line feed after it are one cluster, and
    // the entry of the first replaces both: the word goes on.
    assert_eq!(
        direct_ids(&ours, "a\r\nb", false),
        [id("\u{2581}a"), id("x"), id("b")]
    );
    assert_eq!(
        direct_ids(&ours, "a\rb\nb", false),
        [id("\u{2581}a"), id("x"), id("b"), id(MARKER), id("b")]
    );
    // The letter after this character is in its cluster and goes with it.
    assert_eq!(
        direct_ids(&ours, "\u{600}ab", false),
        [id(MARKER), id("y"), id("b")]
    );
    // A letter the map keeps alone is half of an entry with its mark.
    assert_eq!(
        direct_ids(&ours, "a\u{301}b", false),
        [id(MARKER), id("z"), id("b")]
    );
    assert_eq!(
        direct_ids(&ours, "\u{ff21}b a\tb", false),
        [
            id(MARKER),
            id("A"),
            id("b"),
            id("\u{2581}a"),
            id(MARKER),
            id("b")
        ]
    );
    // A character that goes leaves its neighbours one word.
    assert_eq!(direct_ids(&ours, "a\u{ad}b", false), [id("\u{2581}ab")]);
    let letters = [
        "a", "b", " ", "\t", "\r", "\n", "\u{301}", "\u{600}", "\u{ff21}", "\u{ad}",
    ];
    for text in texts_of(&letters, 4) {
        assert_eq!(
            direct_ids(&ours, &text, false),
            library_ids(&library, &text, false),
            "{text:?}"
        );
    }
}

#[test]
fn an_added_token_is_cut_from_the_raw_text_as_it_stands() {
    let (library, ours) = pair(&definition());
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    // The text after the token starts a word of its own.
    assert_eq!(
        direct_ids(&ours, "a</s>b", false),
        [id("\u{2581}a"), id("</s>"), id(MARKER), id("b")]
    );
    for text in [
        "</s>",
        "<pad></s>",
        "a <unk> b",
        "</s",
        "a\t</s>\u{301}",
        "\u{600}</s>a",
    ] {
        assert_eq!(
            direct_ids(&ours, text, true),
            library_ids(&library, text, true),
            "{text:?}"
        );
    }
}

#[test]
fn the_encoder_cuts_added_tokens_with_its_own_matcher() {
    let library = library(&definition());
    let Ok(encoder) = UnigramEncoder::from_tokenizer(&*library) else {
        panic!("the encoder takes the definition");
    };
    for text in ["a</s>b", "<pad></s>", "a <unk> b"] {
        assert_eq!(
            encoder.encode(&*library, text, true),
            Some(library_ids(&library, text, true)),
            "{text:?}"
        );
    }
}

#[test]
fn the_frame_is_what_the_post_processor_adds() {
    let template = definition();
    let none = changed("/post_processor", Value::Null);
    for (name, definition) in [("a template", template), ("no post-processor", none)] {
        let (library, ours) = pair(&definition);
        for special in [false, true] {
            for text in TEXTS {
                assert_eq!(
                    direct_ids(&ours, text, special),
                    library_ids(&library, text, special),
                    "{name}, special tokens {special}: {text:?}"
                );
            }
        }
    }
}

#[test]
fn a_word_longer_than_any_piece_encodes_as_the_library_does() {
    let (library, ours) = pair(&definition());
    let text = format!("{}\u{6587}{}", "abc".repeat(2000), "cab".repeat(2000));
    assert_eq!(
        direct_ids(&ours, &text, true),
        library_ids(&library, &text, true)
    );
}

#[test]
fn a_long_piece_is_asked_only_where_its_first_byte_stands() {
    // Pieces of 64, 65 and 300 bytes, all of `b`: the paths over them, and
    // of two paths of one score the one kept, are the library's.
    let mut definition = definition();
    let vocabulary = definition["model"]["vocab"]
        .as_array_mut()
        .expect("the vocabulary of the definition");
    for (len, score) in [(64, -7.0), (65, -8.0), (300, -9.0)] {
        vocabulary.push(json!(["b".repeat(len), score]));
    }
    let (library, ours) = pair(&definition);
    for len in [63, 64, 65, 66, 129, 299, 300, 301, 365, 664] {
        for text in ["b".repeat(len), format!("a{}c", "b".repeat(len))] {
            assert_eq!(
                direct_ids(&ours, &text, true),
                library_ids(&library, &text, true),
                "{len} of `b`"
            );
        }
    }
    // A word of `a` asks the model for the lengths that pieces with its
    // first bytes have (three with the marker's, three with `a`), not for
    // every length up to the longest piece.
    let Ok(encoder) = UnigramEncoder::from_tokenizer(&*library) else {
        panic!("the encoder takes the definition");
    };
    let ModelWrapper::Unigram(model) = library.get_model() else {
        panic!("the model of the definition is Unigram");
    };
    let word = format!("{MARKER}{}", "a".repeat(2000));
    let asked = Cell::new(0_usize);
    let (mut best, mut ids) = (Vec::new(), Vec::new());
    encoder.best_path(
        |text| {
            asked.set(asked.get() + 1);
            model.token_to_id(text)
        },
        &word,
        &mut best,
        &mut ids,
    );
    assert_eq!(ids, library_ids(&library, &word[MARKER.len()..], false));
    assert!(asked.get() <= 3 * 2001, "{} texts asked", asked.get());
}

#[test]
fn a_batch_encodes_as_each_text_alone() {
    let (library, ours) = pair(&definition());
    let batch = ours
        .encode_batch(TEXTS, true)
        .expect("the crate encodes a batch");
    assert_eq!(batch.len(), TEXTS.len());
    for (text, encoding) in TEXTS.iter().zip(&batch) {
        assert!(matches!(encoding, Encoding::Plain(_)), "{text:?}");
        assert_eq!(
            encoding.token_ids(),
            library_ids(&library, text, true),
            "{text:?}"
        );
    }
}

#[test]
fn a_configuration_the_encoder_does_not_reproduce_is_declined_with_its_reason() {
    assert_eq!(declined(&definition()), None);
    let changes = [
        (
            "/truncation",
            json!({"direction": "Right", "max_length": 8, "strategy": "LongestFirst", "stride": 0}),
            "truncation",
        ),
        (
            "/padding",
            json!({
                "strategy": "BatchLongest",
                "direction": "Right",
                "pad_to_multiple_of": null,
                "pad_id": 0,
                "pad_type_id": 0,
                "pad_token": "<pad>",
            }),
            "padding",
        ),
        ("/normalizer", json!({"type": "NFKC"}), "normalizer"),
        ("/normalizer", Value::Null, "normalizer"),
        (
            "/pre_tokenizer",
            json!({"type": "WhitespaceSplit"}),
            "pre-tokenizer",
        ),
        (
            "/pre_tokenizer/pretokenizers/1/prepend_scheme",
            json!("first"),
            "pre-tokenizer",
        ),
        (
            "/pre_tokenizer/pretokenizers/1/split",
            json!(false),
            "pre-tokenizer",
        ),
        (
            "/post_processor",
            json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}),
            "post-processor",
        ),
        ("/model/byte_fallback", json!(true), "bytes"),
        ("/model/unk_id", Value::Null, "unknown piece"),
        (
            "/model",
            json!({
                "type": "WordPiece",
                "unk_token": "<unk>",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"<pad>": 0, "</s>": 1, "<unk>": 2, "a": 3},
            }),
            "not Unigram",
        ),
        // A token the library matches after normalization is not cut from
        // the raw text as it stands.
        ("/added_tokens/0/normalized", json!(true), "added token"),
    ];
    for (place, value, reason) in changes {
        let declined = declined(&changed(place, value));
        assert!(
            declined
                .as_deref()
                .is_some_and(|text| text.contains(reason)),
            "{place}: {declined:?}"
        );
    }
}

#[test]
fn a_declined_tokenizer_encodes_in_the_library_as_before() {
    let definition = changed(
        "/pre_tokenizer/pretokenizers/1/prepend_scheme",
        json!("first"),
    );
    let (library, ours) = pair(&definition);
    assert_eq!(ours.encode_path(), EncodePath::Tokenizers);
    let encoding = ours.encode("abc bca", true).expect("the crate encodes");
    assert!(matches!(encoding, Encoding::Hf(_)));
    assert_eq!(encoding.token_ids(), library_ids(&library, "abc bca", true));
}
