//! What the hottest paths allocate, pinned at today's count, so the next
//! change that adds a copy fails a test rather than moving a bench gate
//! inside its noise. An exact-cache hit hands out one copy of the prompt's
//! ids; a decode step on a byte-level vocabulary allocates nothing; a decode
//! step on a byte-fallback vocabulary allocates nothing either on its plain-piece
//! path, the piece going from the table into the stream's buffer. Each
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
    CacheConfig, CachedTokenizer, HuggingFaceTokenizer, Sequence, StopRules, StopSequenceConfig,
    StopSequenceDecoder, TiktokenTokenizer,
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
/// the encode allocates is its ids (the list, grown once past its guess) and
/// the thread's cache entry for a piece longer than the packed table keys
/// (the piece and its ids, two boxes): four calls; its list of units lives
/// inline (it was a fifth). What moves it: a list of units on the heap again,
/// a merge that returns to the heap (the linked list and the
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
        calls, 4,
        "a first sight of a 40-byte piece allocated {calls} times: the ids and the cache entry \
         are four (the units live inline), the merge itself none"
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
fn a_decode_step_on_a_byte_fallback_vocabulary_allocates_nothing_on_a_plain_piece() {
    let tokenizer = byte_fallback();
    // Word pieces only: the fast path of the step, a plain piece after a
    // settled window, which is what a stream of prose is almost everywhere.
    // Two streams, by the length of the piece's text: up to four bytes, and
    // longer, since a prefix rendered into a new string would grow for the
    // longer one.
    for (pieces, name) in [
        (SHORT_PIECES, "up to four bytes"),
        (LONG_PIECES, "longer than four bytes"),
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
            steady.iter().all(|&calls| calls == 0),
            "a 1,000-token stream of pieces {name} on a byte-fallback vocabulary allocated {:?} \
             times per steady step (and {first} on the first): today a plain piece after a \
             settled window goes from the table into the stream's buffer and becomes the window \
             on its own without a render, as the byte-level decoder streams a whole token; a \
             step that copies the piece out of the table or renders the one-id window into a \
             new prefix string takes this to 2 (and to 3 for a piece longer than four bytes), \
             and a step that goes back to the two window decodes per token to 4 or more",
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
/// encoder's encodes make almost all of it (an ids list per piece it is
/// handed; its list of units lives inline, and was a second list per piece
/// before, where the file's own encoder made a parts list and a rank list per
/// word it merged); the ids buffer, sized up front at one id per three bytes,
/// makes one and its regrowths on this one-id-per-byte vocabulary (about twice
/// the reservation) make the rest of its share; nothing is made per segment by
/// the segmented encode itself.
const PIN_SEGMENTED_ENCODE_CALLS: usize = 829;

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
         encoder's encodes make almost all of it (an ids list per piece; the units live inline), the \
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

/// A request's stop rules: four stop strings and two stop token ids, as an
/// agent's request carries them.
fn stop_config() -> StopSequenceConfig {
    StopSequenceConfig::default()
        .with_stop_sequence("\n\nHuman:")
        .with_stop_sequence("</s>")
        .with_stop_sequence("###")
        .with_stop_sequence("Observation:")
        .with_stop_token(2)
        .with_stop_token(50_256)
}

/// A request's stop rules are built once and copied into each of its choices'
/// decoders: the copy allocates nothing (the patterns and the automaton share
/// their tables behind reference counts, the ids live inline or behind one
/// more), so a decoder over the copy allocates its own decode state alone,
/// where one built from the request's parameters pays the automaton's build
/// and the ids' sort again; and a stream's first choice, whose decoder is made
/// over the rules built for the request, pays the build and its state, as a
/// decoder from the parameters did.
#[test]
fn a_choices_stop_decoder_copies_the_requests_rules_instead_of_building_them() {
    let tokenizer = byte_level();
    let (rules, build) = allocations_during(|| StopRules::new(stop_config()));
    let (_, copying) = allocations_during(|| rules.clone());
    assert_eq!(
        copying, 0,
        "a copy of the rules of four stop strings allocated {copying} times: the patterns and the \
         automaton share their tables, the ids live inline"
    );
    assert!(
        build > copying,
        "the build allocated {build} times and the copy {copying}: the automaton's build and the \
         ids' sort are paid once"
    );
    let many = (0..12u32).fold(stop_config(), |config, i| config.with_stop_token(1_000 + i));
    let (rules_many, _) = allocations_during(|| StopRules::new(many));
    let (_, copying_many) = allocations_during(|| rules_many.clone());
    assert_eq!(
        copying_many, 0,
        "a copy of the rules of fourteen stop token ids allocated {copying_many} times: a set past \
         the inline room is shared behind a reference count"
    );
    let (none, no_copy) = allocations_during(|| StopRules::new(StopSequenceConfig::default()));
    let (_, copying_none) = allocations_during(|| none.clone());
    assert_eq!(
        (no_copy, copying_none),
        (0, 0),
        "the rules of no stops allocate nothing to build ({no_copy}) and nothing to copy \
         ({copying_none})"
    );
    let (_, state_only) =
        allocations_during(|| StopSequenceDecoder::with_rules(Arc::clone(&tokenizer), none, false));
    let (_, from_copy) = allocations_during(|| {
        StopSequenceDecoder::with_rules(Arc::clone(&tokenizer), rules.clone(), false)
    });
    let (_, from_config) = allocations_during(|| {
        StopSequenceDecoder::new(Arc::clone(&tokenizer), stop_config(), false)
    });
    assert_eq!(
        from_copy,
        state_only + copying,
        "a decoder over a copy of the rules allocated {from_copy} times: the copy ({copying}) and \
         its own decode state ({state_only})"
    );
    assert!(
        from_config > from_copy,
        "a decoder built from the request's parameters allocated {from_config} times, one over a \
         copy of its rules {from_copy}"
    );
    let (_, first_choice) = allocations_during(|| {
        let rules = StopRules::new(stop_config());
        StopSequenceDecoder::with_rules(Arc::clone(&tokenizer), rules.clone(), false)
    });
    assert_eq!(
        first_choice,
        build + state_only,
        "a stream's first choice, its decoder over the rules built for the request, allocated \
         {first_choice} times: the build ({build}) and its state ({state_only}), what a decoder \
         from the parameters pays ({from_config}); the copy adds nothing"
    );
}

/// An "openai"-format template (every text part followed by a space): the
/// Jinja render gives it each string content as a one-item part list.
const PARTS_TEMPLATE: &str = r"{%- for message in messages -%}
{{- '<|' + message['role'] + '|>' -}}
{%- if message['content'] is string -%}
{{- message['content'] | trim -}}
{%- elif message['content'] is sequence -%}
{%- for item in message['content'] -%}
{{- item['text'] | trim + ' ' -}}
{%- endfor -%}
{%- endif -%}
{{- '<|end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}";

/// The byte-level vocabulary with `PARTS_TEMPLATE` as its chat template, read
/// from files the way a checkpoint is.
fn parts_template_tokenizer() -> (tempfile::TempDir, HuggingFaceTokenizer) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let json = dir.path().join("tokenizer.json");
    common::byte_level_bpe(&common::FEW_MERGES, &[])
        .save(&json, false)
        .expect("the vocabulary file");
    let template = dir.path().join("chat_template.jinja");
    std::fs::write(&template, PARTS_TEMPLATE).expect("the template file");
    let tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
        json.to_str().expect("a utf-8 path"),
        Some(template.to_str().expect("a utf-8 path")),
    )
    .expect("the tokenizer loads");
    (dir, tokenizer)
}

/// Two turns of a tool conversation: a system prompt, a question about
/// `city`, a call, its result, a second question, a second call and its
/// result. Cities of one length keep two conversations one size apart.
fn tool_conversation(city: &str) -> Vec<serde_json::Value> {
    let call = |id: &str, city: &str| {
        serde_json::json!({"role": "assistant", "content": null, "tool_calls": [{"id": id, "type": "function",
            "function": {"name": "weather", "arguments": format!("{{\"city\": \"{city}\"}}")}}]})
    };
    let result = |id: &str, text: &str| serde_json::json!({"role": "tool", "tool_call_id": id, "content": text});
    vec![
        serde_json::json!({"role": "system", "content": "You answer with the tools you have."}),
        serde_json::json!({"role": "user", "content": format!("What is the weather in {city}?")}),
        call("call-1", city),
        result("call-1", "{\"sky\": \"clear\", \"celsius\": 18}"),
        serde_json::json!({"role": "user", "content": "And in Bern?"}),
        call("call-2", "Bern"),
        result("call-2", "{\"sky\": \"rain\", \"celsius\": 7}"),
    ]
}

/// A warm-up render, then the borrowed and the owned render of `tokenizer`
/// on a cold conversation each, one size apart: the allocation calls of the
/// two, the prompts checked equal once the city is set aside.
fn renders_of(tokenizer: &dyn Tokenizer) -> (usize, usize) {
    let params = || ChatTemplateParams {
        add_generation_prompt: true,
        ..ChatTemplateParams::default()
    };
    tokenizer
        .apply_chat_template_with_encoding(&tool_conversation("Rome"), params(), None)
        .expect("the warm-up render");
    let borrowed_messages = tool_conversation("Oslo");
    let (borrowed, borrowed_calls) = allocations_during(|| {
        tokenizer
            .apply_chat_template_with_encoding(&borrowed_messages, params(), None)
            .expect("the borrowed render")
    });
    let owned_messages = tool_conversation("Lima");
    let (owned, owned_calls) = allocations_during(|| {
        tokenizer
            .apply_chat_template_with_encoding_owned(owned_messages, params(), None)
            .expect("the owned render")
    });
    assert_eq!(
        owned.text.replace("Lima", "Oslo"),
        borrowed.text,
        "the same prompt either way"
    );
    assert!(
        borrowed.text.contains("<|tool|>{\"sky\": \"clear\""),
        "the tool results stay strings: {}",
        borrowed.text
    );
    (borrowed_calls, owned_calls)
}

/// The render of an OpenAI-format template wraps every string content as a
/// one-item part list first. On a borrowed slice that is a copy of every
/// message; on a vector the caller hands over it happens in place, the
/// prompt the same. Today: 211 calls against 136 for the seven messages of
/// a two-turn tool conversation, after a warm-up render; a copy added to
/// either path moves its count.
#[test]
fn an_owned_render_of_an_openai_format_template_wraps_in_place_and_copies_no_message() {
    let (_dir, tokenizer) = parts_template_tokenizer();
    assert_eq!(
        tokenizer.chat_template_content_format(),
        llm_tokenizer::chat_template::ChatTemplateContentFormat::OpenAI,
        "the template is detected as the parts format"
    );
    let (borrowed_calls, owned_calls) = renders_of(&tokenizer);
    assert_eq!(
        (borrowed_calls, owned_calls),
        (211, 136),
        "allocation calls of the borrowed and the owned render of seven messages"
    );
}

/// The gateway renders through the cached wrapper, whose owned entry must
/// reach the renderer's for the in-place wrap to hold on the serving path.
/// Today: 211 calls against 136 through the wrapper, as through the renderer.
#[test]
fn the_cached_wrappers_owned_render_wraps_in_place_as_the_renderers_does() {
    let (_dir, tokenizer) = parts_template_tokenizer();
    let inner: Arc<dyn Tokenizer> = Arc::new(tokenizer);
    let cached = CachedTokenizer::new(
        inner,
        CacheConfig {
            enable_l0: true,
            enable_l1: false,
            ..CacheConfig::default()
        },
    );
    let (borrowed_calls, owned_calls) = renders_of(&cached);
    assert_eq!(
        (borrowed_calls, owned_calls),
        (211, 136),
        "allocation calls of the borrowed and the owned render through the cached wrapper"
    );
}
