//! What the hottest paths allocate, pinned at today's count, so the next
//! change that adds a copy fails a test rather than moving a bench gate
//! inside its noise. An exact-cache hit hands out one copy of the prompt's
//! ids; a decode step on a byte-level vocabulary allocates nothing; a decode
//! step on a byte-fallback vocabulary pays two strings on its fast path, and a
//! third for a piece longer than four bytes. Each
//! pin states the count it reads today and what moves it, so the change that
//! removes a copy flips the pin in the same commit. A separate executable, so
//! it can declare the counting allocator the pins need.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

mod common;

use std::{collections::BTreeSet, sync::Arc};

use common::counting_alloc::{calls, Counting};
use llm_tokenizer::{
    traits::{Encoder, Tokenizer},
    CacheConfig, CachedTokenizer, HuggingFaceTokenizer, Sequence,
};
use tokenizers::{
    decoders::{
        byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence,
        strip::Strip,
    },
    models::bpe::{Merges, Vocab, BPE},
    normalizers::Replace,
    AddedToken, DecoderWrapper, Tokenizer as HfTokenizer,
};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = calls();
    let out = work();
    (out, calls() - before)
}

/// The byte-level vocabulary of the common shape, wrapped as the gateway
/// serves it.
fn byte_level() -> Arc<dyn Tokenizer> {
    Arc::new(HuggingFaceTokenizer::from_tokenizer(
        common::byte_level_bpe(&common::FEW_MERGES, &[]),
    ))
}

/// The word pieces of the byte-fallback vocabulary that render as four bytes
/// or fewer (ids 3 to 6), and the ones that render longer (ids 7 to 11).
const SHORT_PIECES: std::ops::Range<u32> = 3..7;
const LONG_PIECES: std::ops::Range<u32> = 7..12;

/// A small sentencepiece-style vocabulary with the Llama-class decoder chain
/// (`Replace(U+2581 -> " ")`, `ByteFallback`, `Fuse`, `Strip(" ", 1, 0)`):
/// word pieces behind the word marker, short ones first, the byte tokens of
/// one two-byte character, and two special tokens.
fn byte_fallback() -> Arc<dyn Tokenizer> {
    let pieces = [
        "<unk>",
        "<s>",
        "</s>",
        "\u{2581}the",
        "\u{2581}fox",
        "\u{2581}dog",
        ".",
        "\u{2581}quick",
        "\u{2581}brown",
        "\u{2581}jumps",
        "\u{2581}over",
        "\u{2581}lazy",
        "<0xC3>",
        "<0xA9>",
    ];
    let mut vocab = Vocab::default();
    for (id, piece) in pieces.iter().enumerate() {
        vocab.insert((*piece).to_string(), id as u32);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Merges::new())
        .byte_fallback(true)
        .unk_token("<unk>".into())
        .build()
        .expect("a byte-fallback BPE");
    let mut tokenizer = HfTokenizer::new(model);
    let chain: Vec<DecoderWrapper> = vec![
        Replace::new("\u{2581}", " ")
            .expect("the marker replace")
            .into(),
        ByteFallback::new().into(),
        Fuse::new().into(),
        Strip::new(' ', 1, 0).into(),
    ];
    tokenizer.with_decoder(Some(DecoderSequence::new(chain)));
    tokenizer
        .add_special_tokens([
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ])
        .expect("the special tokens");
    Arc::new(HuggingFaceTokenizer::from_tokenizer(tokenizer))
}

/// Repetitions of a three-word line that encode to twelve ids each on the
/// byte-level vocabulary: a prompt of about 8,200 ids.
const LONG_PROMPT_LINES: usize = 683;

#[test]
fn an_exact_cache_hit_of_an_8k_id_prompt_copies_the_ids_once() {
    let cached = CachedTokenizer::new(
        byte_level(),
        CacheConfig {
            enable_l0: true,
            enable_l1: false,
            ..CacheConfig::default()
        },
    );
    let prompt = "hello the world ".repeat(LONG_PROMPT_LINES);
    let first = cached.encode(&prompt, false).expect("the miss encodes");
    let ids = first.token_ids().len();
    assert!(
        (8_100..8_300).contains(&ids),
        "the prompt is meant to be about 8,200 ids, not {ids}"
    );
    // The first hit after the fill, then the hit the pin reads: nothing
    // lazy is left on the path by then.
    let (warm, _) = allocations_during(|| cached.encode(&prompt, false).expect("a hit"));
    assert_eq!(
        warm.token_ids(),
        first.token_ids(),
        "the hit is the miss's ids"
    );
    let (again, hit) = allocations_during(|| cached.encode(&prompt, false).expect("a hit"));
    assert_eq!(
        again.token_ids(),
        first.token_ids(),
        "the hit is the miss's ids"
    );
    assert_eq!(
        hit,
        1,
        "an exact-cache hit of {ids} ids allocated {hit} times: today the hit copies the cached \
         ids once, into the encoding it hands out ({} bytes), and nothing else; handing out the \
         shared encoding instead takes this to 0, and a second copy takes it to 2",
        ids * 4
    );
}

/// `ids` through one stream as the gateway reads it: each token's text
/// appended into the stream's buffer, cleared per token and keeping its
/// capacity. The allocation calls of every step, in order, and the text.
fn allocations_per_step(tokenizer: Arc<dyn Tokenizer>, ids: &[u32]) -> (Vec<usize>, String) {
    let mut sequence = Sequence::new(tokenizer);
    let mut out = String::with_capacity(64);
    let mut text = String::new();
    let mut steps = Vec::with_capacity(ids.len());
    for &id in ids {
        out.clear();
        let (appended, calls) =
            allocations_during(|| sequence.append_token_into(id, &mut out).expect("a step"));
        assert!(appended, "every token of the stream has text of its own");
        steps.push(calls);
        text.push_str(&out);
    }
    (steps, text)
}

/// The distinct counts among `steps`, for a failure message.
fn distinct(steps: &[usize]) -> BTreeSet<usize> {
    steps.iter().copied().collect()
}

#[test]
fn a_decode_step_on_a_byte_level_vocabulary_allocates_nothing() {
    let tokenizer = byte_level();
    let prompt = "hello the world ".repeat(400);
    let encoding = tokenizer.encode(&prompt, false).expect("an encode");
    let ids = &encoding.token_ids()[..1_000];
    let (steps, text) = allocations_per_step(Arc::clone(&tokenizer), ids);
    assert_eq!(
        text,
        tokenizer.decode(ids, false).expect("a decode"),
        "the stream is the decode"
    );
    assert!(
        steps.iter().all(|&calls| calls == 0),
        "a 1,000-token stream on a byte-level vocabulary allocated {:?} times per token ({} on \
         the first): every step borrows the whole token's text from the table and appends it \
         into the stream's buffer",
        distinct(&steps),
        steps[0]
    );
}

#[test]
fn a_decode_step_on_a_byte_fallback_vocabulary_costs_two_strings_today() {
    let tokenizer = byte_fallback();
    // Word pieces only: the fast path of the step, a plain piece after a
    // settled window, which is what a stream of prose is almost everywhere.
    // Two streams, by the length of the piece's text: up to four bytes, and
    // longer.
    for (pieces, name, expected, growth) in [
        (SHORT_PIECES, "up to four bytes", 2, "fits"),
        (LONG_PIECES, "longer than four bytes", 3, "grows once"),
    ] {
        let count = pieces.len() as u32;
        let ids: Vec<u32> = (0..1_000).map(|k| pieces.start + k % count).collect();
        let (steps, text) = allocations_per_step(Arc::clone(&tokenizer), &ids);
        assert_eq!(
            text,
            tokenizer.decode(&ids, false).expect("a decode"),
            "the stream is the decode"
        );
        let (first, steady) = steps.split_first().expect("a thousand steps");
        assert!(
            steady.iter().all(|&calls| calls == expected),
            "a 1,000-token stream of pieces {name} on a byte-fallback vocabulary allocated {:?} \
             times per steady step (and {first} on the first): today each step copies the \
             piece's text out of the table and renders the one-id window into a new prefix \
             string sized at four bytes per id, which {growth} for such a piece: {expected} \
             allocations per token where the byte-level decoder makes none; a step that borrows \
             the piece and keeps its window takes this to 0, and a step that goes back to the \
             two window decodes per token takes it to 4 or more",
            distinct(steady)
        );
    }
}
