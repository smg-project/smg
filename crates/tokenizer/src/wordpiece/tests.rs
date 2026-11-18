//! The direct WordPiece encoder against `tokenizers` itself, on a small
//! tokenizer built in the test: every flag set of the normalizer over texts
//! that reach each of its rules, every code point, the frame, the added
//! tokens, the bound on a word, and the configurations the encoder declines.

use serde_json::{json, Map, Value};
use tokenizers::{
    normalizers::bert::BertNormalizer, pre_tokenizers::bert::BertPreTokenizer, NormalizedString,
    Normalizer, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
};

use super::{WordPieceEncoder, MAX_ADDED};
use crate::{
    encode_path::EncodePath,
    huggingface::HuggingFaceTokenizer,
    traits::{Encoder, Encoding},
};

/// The flags of the normalizer.
#[derive(Debug, Clone, Copy)]
struct Flags {
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: Option<bool>,
    lowercase: bool,
}

/// The flags of an uncased tokenizer: marks go with the lowercase.
const UNCASED: Flags = Flags {
    clean_text: true,
    handle_chinese_chars: true,
    strip_accents: None,
    lowercase: true,
};

/// The flags of a cased tokenizer.
const CASED: Flags = Flags {
    clean_text: true,
    handle_chinese_chars: true,
    strip_accents: Some(false),
    lowercase: false,
};

/// The added tokens, at the ids 0 to 4.
const SPECIAL: &[&str] = &["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

/// Pieces beyond the single letters: whole words, continuations,
/// punctuation, and what the rules leave of the characters in `TEXTS`.
const WORDS: &[&str] = &[
    "hello",
    "world",
    "un",
    "##aff",
    "##able",
    "##s",
    "##llo",
    ",",
    ".",
    "!",
    "?",
    "[",
    "]",
    "#",
    "'",
    "-",
    "$",
    "\u{bf}",
    "\u{2014}",
    "\u{4e2d}",
    "\u{6587}",
    "\u{e9}",
    "##\u{e9}",
    "\u{df}",
    "##\u{df}",
    "##\u{301}",
    "##\u{307}",
    "##\u{327}",
    "\u{1f600}",
    "\u{1100}",
    "##\u{1161}",
];

/// Each of these is a piece both as the start of a word and inside one.
const LETTERS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// Texts that reach every rule of the normalizer and of the cut.
const TEXTS: &[&str] = &[
    "",
    " ",
    "   ",
    "hello world",
    "Hello, World!",
    "HELLO",
    "unaffable unaffables",
    "helloworld",
    "a.b,c!?",
    "it's",
    "[CLS] hello [SEP]",
    "hello[MASK]world",
    "[mask] [UNK] [PAD][PAD]",
    "[ CLS ]",
    "#hello ##llo ###",
    "a\tb\nc\r\nd",
    // Spaces beyond ASCII.
    "a\u{a0}b\u{2003}c\u{3000}d",
    // Control characters, two of them whitespace to other readers.
    "a\u{85}b\u{1c}c\u{7f}d\u{1}e",
    "\0a\0",
    "a\u{fffd}b",
    // Format characters.
    "a\u{200b}b\u{200d}c\u{ad}d\u{feff}e",
    // A letter with its mark in one character, then in two.
    "h\u{e9}llo he\u{301}llo",
    // Two marks in both orders.
    "e\u{301}\u{327} e\u{327}\u{301}",
    // A removed character between two marks, then a kept one.
    "e\u{301}\0\u{327} e\u{301}a\u{327}",
    // A mark with no letter.
    "\u{301}",
    // A capital that lowercases to two characters; a titlecase digraph.
    "\u{130}stanbul \u{1c5}",
    "stra\u{df}e STRASSE",
    // A sign that decomposes to a letter and a mark.
    "\u{212b}",
    // Ideographs, alone and inside words, in three of their blocks.
    "\u{4e2d}\u{6587}\u{5b57}",
    "hello\u{4e2d}world\u{6587}",
    "\u{3400}\u{20000}\u{f900}",
    // A syllable that decomposes to two pieces of the vocabulary.
    "\u{ac00}",
    // Spacing and nonspacing marks of one script.
    "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947}",
    "\u{bf}qu\u{e9}?",
    "a\u{2014}b\u{2026}",
    // Fullwidth letters and punctuation.
    "\u{ff48}\u{ff45}\u{ff4c}\u{ff4c}\u{ff4f}\u{ff01}",
    "$5 ^ ` ~ + = < > |",
    "\u{1f600} a\u{1f600}b",
    "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}",
    // Unassigned, private use, the last code point.
    "x\u{e0000}y\u{f0000}z\u{10ffff}",
];

/// The vocabulary: each piece once, its id the count of those before it.
fn vocabulary() -> Map<String, Value> {
    let mut vocabulary = Map::new();
    let pieces = SPECIAL
        .iter()
        .chain(WORDS)
        .map(|piece| (*piece).to_owned())
        .chain(
            LETTERS
                .chars()
                .flat_map(|c| [c.to_string(), format!("##{c}")]),
        );
    for piece in pieces {
        if !vocabulary.contains_key(&piece) {
            let id = vocabulary.len();
            vocabulary.insert(piece, json!(id));
        }
    }
    vocabulary
}

/// A tokenizer file of the shape the encoder takes, under `flags`.
fn definition(flags: Flags) -> Value {
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
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": flags.clean_text,
            "handle_chinese_chars": flags.handle_chinese_chars,
            "strip_accents": flags.strip_accents,
            "lowercase": flags.lowercase,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
            ],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]},
            },
        },
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocabulary(),
        },
    })
}

/// `definition(flags)` with more added tokens behind the special ones, at
/// the ids that follow the vocabulary.
fn with_added(flags: Flags, contents: &[&str]) -> Value {
    let mut definition = definition(flags);
    let first = vocabulary().len();
    let added = definition["added_tokens"]
        .as_array_mut()
        .expect("the added tokens are a list");
    for (at, content) in contents.iter().enumerate() {
        added.push(json!({
            "id": first + at,
            "content": content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true,
        }));
    }
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
    WordPieceEncoder::from_tokenizer(&*library(definition))
        .err()
        .map(|declined| declined.to_string())
}

#[test]
fn every_flag_set_encodes_as_the_library_does() {
    // A word at the model's bound, one past it, and one past it in
    // characters that take two bytes each.
    let long = [
        "x".repeat(100),
        "x".repeat(101),
        format!("{} hello", "\u{e9}".repeat(101)),
    ];
    for clean_text in [false, true] {
        for handle_chinese_chars in [false, true] {
            for strip_accents in [None, Some(false), Some(true)] {
                for lowercase in [false, true] {
                    let flags = Flags {
                        clean_text,
                        handle_chinese_chars,
                        strip_accents,
                        lowercase,
                    };
                    let (library, ours) = pair(&definition(flags));
                    assert_eq!(ours.encode_path(), EncodePath::Native, "{flags:?}");
                    let texts = TEXTS.iter().copied().chain(long.iter().map(String::as_str));
                    for text in texts {
                        for special in [false, true] {
                            assert_eq!(
                                direct_ids(&ours, text, special),
                                library_ids(&library, text, special),
                                "{text:?} under {flags:?}, special tokens {special}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Every code point between two letters: the three class tables, the
/// whitespace and ideograph tests, the decomposition and the lowercase of
/// each character. A table that moved away from the library's fails here
/// with the code point.
#[test]
fn every_code_point_encodes_as_the_library_does() {
    const CHUNK: u32 = 256;
    for flags in [UNCASED, CASED] {
        let (library, ours) = pair(&definition(flags));
        for first in (0..=0x10_FFFF).step_by(CHUNK as usize) {
            let chunk: Vec<char> = (first..first + CHUNK).filter_map(char::from_u32).collect();
            let text: String = chunk.iter().flat_map(|&c| ['x', c, 'y', ' ']).collect();
            if direct_ids(&ours, &text, false) == library_ids(&library, &text, false) {
                continue;
            }
            for &c in &chunk {
                let one = format!("x{c}y");
                assert_eq!(
                    direct_ids(&ours, &one, false),
                    library_ids(&library, &one, false),
                    "U+{:04X} under {flags:?}",
                    u32::from(c)
                );
            }
            panic!("the code points from U+{first:04X} differ only together under {flags:?}");
        }
    }
}

#[test]
fn a_word_is_cut_longest_piece_first_and_framed_on_request() {
    let (library, ours) = pair(&definition(UNCASED));
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    assert_eq!(
        direct_ids(&ours, "unaffable", false),
        [id("un"), id("##aff"), id("##able")]
    );
    assert_eq!(
        direct_ids(&ours, "Hello, worlds!", true),
        [
            id("[CLS]"),
            id("hello"),
            id(","),
            id("world"),
            id("##s"),
            id("!"),
            id("[SEP]")
        ]
    );
    // No piece continues the word with this letter: the word is unknown
    // whole, the piece that started it included.
    assert_eq!(
        direct_ids(&ours, "hello\u{3b1} hello", false),
        [id("[UNK]"), id("hello")]
    );
}

#[test]
fn a_word_at_the_bound_is_cut_and_one_past_it_is_unknown() {
    let (library, ours) = pair(&definition(UNCASED));
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    let mut at_the_bound = vec![id("x")];
    at_the_bound.resize(100, id("##x"));
    assert_eq!(direct_ids(&ours, &"x".repeat(100), false), at_the_bound);
    assert_eq!(direct_ids(&ours, &"x".repeat(101), false), [id("[UNK]")]);
}

#[test]
fn an_added_token_is_cut_from_the_raw_text_as_it_stands() {
    let (library, ours) = pair(&definition(UNCASED));
    let id = |piece: &str| {
        library
            .token_to_id(piece)
            .expect("a piece of the vocabulary")
    };
    assert_eq!(
        direct_ids(&ours, "hello[MASK]world", false),
        [id("hello"), id("[MASK]"), id("world")]
    );
    // The token is matched before the text is lowercased, so its lowercase
    // spelling is three words.
    assert_eq!(
        direct_ids(&ours, "[mask]", false),
        [id("["), id("m"), id("##a"), id("##s"), id("##k"), id("]")]
    );
    // A token at each end of the text, two in a row, a start that no token
    // finishes, and a token behind a run beyond ASCII, between two marks
    // and between two removed characters.
    let texts = [
        "[CLS]",
        "[CLS][SEP]",
        "[PAD]hello[PAD]",
        "[",
        "[[CLS]",
        "[CLS",
        "[CLS][",
        "x[SE[SEP]y",
        "[SEP ]",
        "a[b]c",
        "h\u{e9}[SEP]\u{e9}",
        "e\u{301}[MASK]\u{327}",
        "\u{4e2d}[UNK]\u{6587}",
        "a\u{1}[PAD]\u{1}b",
    ];
    for flags in [UNCASED, CASED] {
        let (library, ours) = pair(&definition(flags));
        for text in texts {
            for special in [false, true] {
                assert_eq!(
                    direct_ids(&ours, text, special),
                    library_ids(&library, text, special),
                    "{text:?} under {flags:?}, special tokens {special}"
                );
            }
        }
    }
}

#[test]
fn added_tokens_are_cut_leftmost_first_and_longest_first() {
    // Two tokens that start alike, one of them the start of the other; one
    // that starts inside them; one that starts with a letter and is a piece
    // too; one that starts with a space.
    let (library, ours) = pair(&with_added(UNCASED, &["<a", "<ab", "b>", "hello", " ."]));
    assert_eq!(ours.encode_path(), EncodePath::Native);
    let id = |token: &str| {
        library
            .token_to_id(token)
            .expect("a token of the tokenizer")
    };
    // The longest of the tokens that stand at one place.
    assert_eq!(direct_ids(&ours, "<abc", false), [id("<ab"), id("c")]);
    // The leftmost token: the one that starts inside it is text.
    assert_eq!(direct_ids(&ours, "<ab>", false), [id("<ab"), id("[UNK]")]);
    // A start no token finishes is a character of its class, and the next
    // byte is asked in its turn.
    assert_eq!(direct_ids(&ours, "<b>", false), [id("[UNK]"), id("b>")]);
    // A token is cut inside a word.
    assert_eq!(direct_ids(&ours, "ahello", false), [id("a"), id("hello")]);
    assert_eq!(
        direct_ids(&ours, "hellohello", false),
        [id("hello"), id("hello")]
    );
    let texts = [
        "<a",
        "<ab",
        "<abc",
        "<ab>",
        "<abb>",
        "<b>",
        "<a<ab<abb>",
        "a<ab>b",
        "<",
        "b",
        "b>b>",
        "hello",
        "Hello",
        "hell",
        "hellos",
        "ahello",
        "hellohello",
        "x .",
        " . .",
        ". .",
        "a .b>",
        "h\u{e9}<ab\u{e9}",
    ];
    for text in texts {
        for special in [false, true] {
            assert_eq!(
                direct_ids(&ours, text, special),
                library_ids(&library, text, special),
                "{text:?}, special tokens {special}"
            );
        }
    }
}

#[test]
fn a_set_of_added_tokens_the_pass_does_not_cut_is_declined() {
    // A token that starts beyond ASCII, one that starts with a byte the
    // normalizer removes, and one token more than the encoder tries.
    let many: Vec<String> = (SPECIAL.len()..=MAX_ADDED)
        .map(|n| format!("<{n}>"))
        .collect();
    let many: Vec<&str> = many.iter().map(String::as_str).collect();
    for (contents, reason) in [
        (&["\u{e9}t\u{e9}"][..], "does not start"),
        (&["\u{1}x"][..], "does not start"),
        (&many[..], "more than"),
    ] {
        let declined = declined(&with_added(UNCASED, contents));
        assert!(
            declined
                .as_deref()
                .is_some_and(|text| text.contains(reason)),
            "the added tokens {contents:?}: {declined:?}"
        );
    }
    // As many tokens as the encoder tries are taken.
    let (library, ours) = pair(&with_added(UNCASED, &many[1..]));
    assert_eq!(ours.encode_path(), EncodePath::Native);
    for text in ["<5><32>x<6", "<31><32", "a<10>b<6>"] {
        assert_eq!(
            direct_ids(&ours, text, true),
            library_ids(&library, text, true),
            "{text:?}"
        );
    }
    // Where the normalizer keeps the control characters, a token that
    // starts with one is taken.
    let kept = Flags {
        clean_text: false,
        ..UNCASED
    };
    let (library, ours) = pair(&with_added(kept, &["\u{1}x"]));
    assert_eq!(ours.encode_path(), EncodePath::Native);
    for text in ["a\u{1}xb", "\u{1}", "\u{1}y \u{1}x"] {
        assert_eq!(
            direct_ids(&ours, text, true),
            library_ids(&library, text, true),
            "{text:?}"
        );
    }
}

#[test]
fn the_frame_is_what_the_post_processor_adds() {
    let template = definition(UNCASED);
    let mut pair_processor = definition(UNCASED);
    pair_processor["post_processor"] =
        json!({"type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2]});
    let mut none = definition(UNCASED);
    none["post_processor"] = Value::Null;
    for (name, definition) in [
        ("a template", template),
        ("the pair processor", pair_processor),
        ("no post-processor", none),
    ] {
        let (library, ours) = pair(&definition);
        for special in [false, true] {
            assert_eq!(
                direct_ids(&ours, "hello world", special),
                library_ids(&library, "hello world", special),
                "{name}, special tokens {special}"
            );
        }
    }
}

#[test]
fn a_batch_encodes_as_each_text_alone() {
    let (library, ours) = pair(&definition(UNCASED));
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
    let blocks = [
        (
            "truncation",
            json!({"direction": "Right", "max_length": 8, "strategy": "LongestFirst", "stride": 0}),
            "truncation",
        ),
        (
            "padding",
            json!({
                "strategy": "BatchLongest",
                "direction": "Right",
                "pad_to_multiple_of": null,
                "pad_id": 0,
                "pad_type_id": 0,
                "pad_token": "[PAD]",
            }),
            "padding",
        ),
        ("normalizer", json!({"type": "NFC"}), "normalizer"),
        ("normalizer", Value::Null, "normalizer"),
        (
            "pre_tokenizer",
            json!({"type": "Whitespace"}),
            "pre-tokenizer",
        ),
        (
            "post_processor",
            json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}),
            "post-processor",
        ),
    ];
    for (block, value, reason) in blocks {
        let mut definition = definition(UNCASED);
        definition[block] = value;
        let declined = declined(&definition);
        assert!(
            declined
                .as_deref()
                .is_some_and(|text| text.contains(reason)),
            "a {block} block: {declined:?}"
        );
    }
    // A token the library matches after normalization is not cut from the
    // raw text as it stands.
    let mut definition = definition(UNCASED);
    definition["added_tokens"][4]["normalized"] = json!(true);
    let declined = declined(&definition);
    assert!(
        declined
            .as_deref()
            .is_some_and(|text| text.contains("added token")),
        "a normalized added token: {declined:?}"
    );
}

#[test]
fn a_declined_tokenizer_encodes_in_the_library_as_before() {
    let mut definition = definition(UNCASED);
    definition["pre_tokenizer"] = json!({"type": "Whitespace"});
    let (library, ours) = pair(&definition);
    assert_eq!(ours.encode_path(), EncodePath::Tokenizers);
    let encoding = ours
        .encode("Hello, worlds!", true)
        .expect("the crate encodes");
    assert!(matches!(encoding, Encoding::Hf(_)));
    assert_eq!(
        encoding.token_ids(),
        library_ids(&library, "Hello, worlds!", true)
    );
}

/// The code points `accepts` holds, as inclusive ranges.
fn ranges_where(mut accepts: impl FnMut(char) -> bool) -> Vec<(u32, u32)> {
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for c in (0..=0x10_FFFF).filter_map(char::from_u32) {
        if !accepts(c) {
            continue;
        }
        let code = u32::from(c);
        match ranges.last_mut() {
            Some((_, last)) if *last + 1 == code => *last = code,
            _ => ranges.push((code, code)),
        }
    }
    ranges
}

/// What the library's normalizer leaves of `text`.
fn normalized(normalizer: BertNormalizer, text: &str) -> String {
    let mut normalized = NormalizedString::from(text);
    normalizer
        .normalize(&mut normalized)
        .expect("the normalizer runs");
    normalized.get().to_owned()
}

/// Prints the tables of `classes.rs`, each class read off the library
/// itself one character at a time: what its normalizer removes under
/// `clean_text`, what it drops under `strip_accents`, and what its
/// pre-tokenizer sets apart.
///
/// Run: `cargo test -p llm-tokenizer --lib wordpiece::tests::print_the_class_tables -- --ignored --nocapture`
/// and replace the tables of the file with the output between the markers.
#[test]
#[ignore = "prints the tables of wordpiece/classes.rs"]
#[expect(clippy::print_stdout)]
fn print_the_class_tables() {
    let clean = BertNormalizer::new(true, false, Some(false), false);
    let strip = BertNormalizer::new(false, false, Some(true), false);
    let removed = ranges_where(|c| normalized(clean, &format!("a{c}b")) == "ab");
    let marks = ranges_where(|c| normalized(strip, c.encode_utf8(&mut [0; 4])).is_empty());
    let punctuation = ranges_where(|c| {
        let mut text = PreTokenizedString::from(format!("a{c}b"));
        BertPreTokenizer
            .pre_tokenize(&mut text)
            .expect("the pre-tokenizer runs");
        text.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .len()
            == 3
    });
    let tables = [
        (
            "REMOVED",
            "What `clean_text` removes: NUL, the replacement character and the\n/// library's control characters.",
            removed,
        ),
        (
            "NONSPACING_MARK",
            "The marks `strip_accents` drops after the decomposition.",
            marks,
        ),
        (
            "PUNCTUATION",
            "What the pre-tokenizer sets apart as a word of its own.",
            punctuation,
        ),
    ];
    println!("\n// BEGIN");
    for (name, doc, ranges) in tables {
        println!("/// {doc}");
        println!("pub(super) const {name}: &[(u32, u32)] = &[");
        for (first, last) in ranges {
            println!("    (0x{first:04X}, 0x{last:04X}),");
        }
        println!("];");
        println!();
    }
    println!("// END");
}
