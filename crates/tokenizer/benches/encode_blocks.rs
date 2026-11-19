//! One row that pins the block pass of the native encoder's pre-tokenizer
//! (`src/native/block.rs`): a 10 KB text of long words through the public
//! encode path. The pass marks the piece starts of 64 bytes at a time from
//! flag bytes that a plain loop writes, and the compiler emits that loop as
//! vector compares; nothing in the source asks for vectors, so a toolchain
//! or a change that leaves the loop scalar is silent everywhere else. Built
//! with the loop and SLP vectorizers off, this row reads 2.5 times slower,
//! against a gate of 10 % (`.github/bench-gates/tokenizer-blocks.toml`).
//! The words are long so that the pass, and not the vocabulary's table, is
//! most of the row, and the text ends in a character beyond ASCII because an
//! all-ASCII unit takes the scanner's lean scan and never reaches the pass.
//! A vocabulary built in code, no model download.
#![allow(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use llm_tokenizer::{EncodePath, Encoder, HuggingFaceTokenizer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

/// The GPT-4 family's pre-tokenizer template, the shape the block pass reads.
const TEMPLATE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const WORDS: [&str; 16] = [
    "internationalization",
    "responsibilities",
    "characteristically",
    "misunderstandings",
    "telecommunications",
    "interdisciplinary",
    "counterproductive",
    "disproportionately",
    "institutionalized",
    "unconstitutional",
    "environmentalists",
    "hypersensitivity",
    "compartmentalized",
    "electromagnetically",
    "oversimplification",
    "transcontinental",
];

/// 520 of the words with a space between two, and an ellipsis at the end as
/// the one character beyond ASCII: 10 KB.
fn long_words() -> String {
    let words: Vec<&str> = WORDS.iter().copied().cycle().take(520).collect();
    format!("The {}\u{2026}\n", words.join(" "))
}

/// A byte-level BPE tokenizer with the template as its split and every word
/// of the text whole in its vocabulary (`Ġ` is the byte-level space), so a
/// word is one piece and one token.
fn tokenizer() -> HuggingFaceTokenizer {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocab: serde_json::Map<String, serde_json::Value> = alphabet
        .into_iter()
        .map(String::from)
        .chain(std::iter::once("The".to_owned()))
        .chain(WORDS.iter().map(|word| format!("Ġ{word}")))
        .enumerate()
        .map(|(id, token)| (token, serde_json::json!(id)))
        .collect();
    let definition = serde_json::json!({
        "version": "1.0",
        "added_tokens": [],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": TEMPLATE}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]},
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "ignore_merges": true}
    });
    let pipeline =
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&definition).expect("json"))
            .expect("tokenizer");
    HuggingFaceTokenizer::from_tokenizer(pipeline)
}

fn bench_long_words(c: &mut Criterion) {
    let tokenizer = tokenizer();
    let text = long_words();
    // A pipeline the native encoder declines, or a vocabulary that splits
    // the words, would time something else under this row's name.
    assert!(
        tokenizer.encode_path() == EncodePath::Native,
        "the row times the native encoder"
    );
    let encoding = tokenizer.encode(&text, false).expect("encode");
    assert_eq!(
        encoding.token_ids().len(),
        525,
        "a token a word, and the ellipsis a token a byte"
    );
    let mut group = c.benchmark_group("encode_blocks");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("long_words", |b| {
        b.iter(|| black_box(tokenizer.encode(black_box(&text), false).expect("encode")));
    });
    group.finish();
}

criterion_group!(benches, bench_long_words);
criterion_main!(benches);
