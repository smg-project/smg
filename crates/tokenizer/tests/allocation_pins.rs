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
    chat_template::ChatTemplateParams,
    traits::{Encoder, PromptEncoding, Tokenizer},
    CacheConfig, CachedTokenizer, HuggingFaceTokenizer, Sequence, TiktokenTokenizer,
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

/// A first sight of a piece of up to 128 bytes merges on the stack. What
/// the encode allocates is its list of units, its ids (the list, grown once
/// past its guess) and the thread's cache entry for a piece longer than the
/// packed table keys (the piece and its ids, two boxes): five calls. What
/// moves it: a merge that returns to the heap (the linked list and the
/// candidate heap of the long merge were two more calls), a cache tier that
/// copies differently. The opt-in v1 pipeline, when its feature is compiled
/// in, takes a BPE tokenizer in front of the direct path, so this pin of the
/// direct path's merge is built without that feature.
#[cfg(not(feature = "tokenizers-v1"))]
#[test]
fn a_first_sight_of_a_forty_byte_piece_merges_without_the_heap() {
    let tokenizer = byte_level();
    // The thread's cache and its map tier exist and have room: the pieces
    // below settle both before the measured piece.
    for letter in ["a", "b", "c", "d", "e", "f"] {
        let settled = letter.repeat(40);
        tokenizer.encode(&settled, false).expect("an encode");
    }
    // One piece of forty bytes under the byte-level split, with merges in it.
    let piece = "hello".repeat(8);
    let (encoding, calls) = allocations_during(|| tokenizer.encode(&piece, false));
    let encoding = encoding.expect("an encode");
    assert_eq!(encoding.token_ids().len(), 24, "h e llo, eight times");
    assert_eq!(
        calls, 5,
        "a first sight of a 40-byte piece allocated {calls} times: the units, the ids and the \
         cache entry are five, the merge itself none"
    );
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

/// A byte-level tiktoken vocabulary (one token per byte) with the four control
/// tokens the segmented renderer writes, under a configuration that selects
/// that renderer and the checkpoint's own piece cuts, as the served
/// checkpoints do; its ids are the prompt's bytes, so the count below reads
/// the encode's own work, not a merge table's.
fn segmented_tiktoken_dir() -> tempfile::TempDir {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let dir = tempfile::tempdir().expect("a temporary directory");
    let model: String = (0u32..256)
        .map(|b| format!("{} {b}\n", STANDARD.encode([b as u8])))
        .collect();
    std::fs::write(dir.path().join("tiktoken.model"), model).expect("the model");
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"added_tokens_decoder": {
            "300": {"content": "<|open|>", "special": false},
            "301": {"content": "<|close|>", "special": false},
            "302": {"content": "<|sep|>", "special": false},
            "303": {"content": "<|end_of_msg|>", "special": true}
        }}"#,
    )
    .expect("the tokenizer config");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["KimiK3ForConditionalGeneration"], "model_type": "kimi_k3"}"#,
    )
    .expect("the model config");
    dir
}

/// Forty turns of a conversation, each a sentence or two.
fn forty_messages() -> Vec<serde_json::Value> {
    (0..40)
        .map(|k| {
            let (role, text) = if k % 2 == 0 {
                (
                    "user",
                    "Please summarize the paragraph about the history of computing.",
                )
            } else {
                (
                    "assistant",
                    "Computing began with mechanical calculators and grew into networks.",
                )
            };
            serde_json::json!({"role": role, "content": format!("{text} ({k})")})
        })
        .collect()
}

/// The allocation calls of a forty-turn segmented encode today: the direct
/// encoder's encodes make almost all of it (a unit list and an ids list per
/// piece it is handed, where the file's own encoder made a parts list and a
/// rank list per word it merged); the ids buffer, sized up front at one id per
/// three bytes, makes one and its regrowths on this one-id-per-byte vocabulary
/// (about twice the reservation) make the rest of its share; nothing is made
/// per segment by the segmented encode itself.
const PIN_SEGMENTED_ENCODE_CALLS: usize = 1_572;

#[test]
fn a_segmented_render_of_forty_turns_encodes_its_ids_into_one_buffer() {
    let dir = segmented_tiktoken_dir();
    let tokenizer = TiktokenTokenizer::from_dir(dir.path()).expect("the tokenizer loads");
    let messages = forty_messages();
    let params = || ChatTemplateParams {
        add_generation_prompt: true,
        ..Default::default()
    };
    let render = || {
        tokenizer
            .apply_chat_template_with_encoding(&messages, params(), None)
            .expect("the render")
    };
    // One warm render: whatever is lazy on the path is built here.
    let warm = render();
    let PromptEncoding::Deferred(job) = warm.encoding else {
        panic!("the segmented renderer defers its encode");
    };
    let expected_ids = job.run().expect("the encode").token_ids().to_vec();
    let rendered = render();
    let PromptEncoding::Deferred(job) = rendered.encoding else {
        panic!("the segmented renderer defers its encode");
    };
    let (ids, calls) = allocations_during(|| job.run().expect("the encode"));
    assert_eq!(ids.token_ids(), expected_ids, "the ids are the render's");
    assert_eq!(
        calls,
        PIN_SEGMENTED_ENCODE_CALLS,
        "encoding the segments of a forty-turn render ({} ids; about twenty segments per turn, \
         every piece of a tag's attributes one of them) allocated {calls} times: the direct \
         encoder's encodes make almost all of it (a unit list and an ids list per piece), the \
         ids buffer sized up front at one id per three bytes makes one and its regrowths on \
         this one-id-per-byte vocabulary the rest of its share, and nothing is made per \
         segment by the segmented encode itself; a list per segment (the checkpoint's cut of \
         a segment shorter than its bounds, or a vector around the text) moves this up by the \
         segment count (about 760), a tighter or looser sizing rule by the regrowths it saves \
         or adds, and a change of the encoder by what its encodes make",
        ids.token_ids().len()
    );
}

/// The special token the prefix cache splits at, in the added vocabulary of
/// the byte-level BPE below.
const TURN_END: &str = "<|im_end|>";

/// The `k`th one-turn text. The texts differ in one number of four digits,
/// each digit one id on the byte-level vocabulary, so every prefix has the
/// same bytes and ids and every entry charges the budget the same.
fn turn(k: usize) -> String {
    format!("turn {k:04} the hello{TURN_END}the end")
}

#[test]
fn a_prefix_cache_entry_costs_its_shared_ids_and_an_eviction_round_nothing() {
    let inner: Arc<dyn Tokenizer> = Arc::new(HuggingFaceTokenizer::from_tokenizer(
        common::byte_level_bpe(&common::FEW_MERGES, &[TURN_END]),
    ));
    let prefix_cached = |budget: usize| {
        CachedTokenizer::new(
            Arc::clone(&inner),
            CacheConfig {
                enable_l0: false,
                enable_l1: true,
                l1_max_memory: budget,
                ..CacheConfig::default()
            },
        )
    };
    let texts: Vec<String> = (0..4).map(turn).collect();
    // A cache with room for everything: what one entry charges the budget,
    // and every piece of the texts through the library's own word cache, so
    // the encodes below read nothing lazy.
    let roomy = prefix_cached(1 << 20);
    for text in &texts {
        roomy.encode(text, false).expect("the miss encodes");
    }
    let stats = roomy.l1_cache_stats().expect("the prefix cache is on");
    assert_eq!(stats.entries, 4, "a one-turn text seeds one prefix entry");
    let size = stats.memory_bytes / 4;
    assert_eq!(
        size * 4,
        stats.memory_bytes,
        "the four entries are of one size"
    );
    // A cache with room for less than one entry: the miss path without its
    // insert, the second text on it as on the cache below.
    let roomless = prefix_cached(size - 1);
    roomless.encode(&texts[0], false).expect("a miss");
    let (_, no_insert) = allocations_during(|| roomless.encode(&texts[1], false).expect("a miss"));
    assert_eq!(roomless.l1_cache_stats().map(|s| s.entries), Some(0));
    // Room for three entries, after one pass over the texts and a clear: the
    // map slot an entry's key hashes to gets its table at its first entry
    // and the key index its room at its first key, and a clear keeps both,
    // so the pass below reads the inserts alone. The second insert is the
    // pinned one, the fourth has to make room.
    let cached = prefix_cached(3 * size);
    for text in &texts {
        cached.encode(text, false).expect("a miss");
    }
    cached.clear_cache();
    cached.encode(&texts[0], false).expect("a miss");
    let (_, steady) = allocations_during(|| cached.encode(&texts[1], false).expect("a miss"));
    cached.encode(&texts[2], false).expect("a miss");
    assert_eq!(cached.l1_cache_stats().map(|s| s.entries), Some(3));
    let (_, evicting) = allocations_during(|| cached.encode(&texts[3], false).expect("a miss"));
    assert_eq!(
        cached.l1_cache_stats().map(|s| (s.entries, s.memory_bytes)),
        Some((3, 3 * size)),
        "the fourth entry took the room of one"
    );
    assert!(
        steady >= no_insert,
        "the miss that inserted allocated {steady} times, the one that could not {no_insert}"
    );
    assert_eq!(
        steady - no_insert,
        1,
        "inserting a prefix entry allocated {} times beyond the miss path that leads to it \
         ({no_insert}): today the entry's shared ids, and nothing else; a box around the entry's \
         timestamp moves this to 2, a copy of the ids to 2 as well (the miss that had to make room \
         allocated {evicting} times)",
        steady - no_insert
    );
    assert_eq!(
        evicting, steady,
        "the miss that had to make room allocated {evicting} times where one with room allocated \
         {steady}: an eviction round draws its samples from the key index, reads them and removes \
         one without allocating; a sample list per round moves this up by one, an iterator over \
         a map per sample by one for every shard of that map it crosses"
    );
}
