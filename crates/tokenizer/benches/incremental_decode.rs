//! Per-token cost of turning generated ids into text.
//!
//! `Sequence::append_token` is the gateway's streaming path (through
//! `StopSequenceDecoder`); `decode_step` is the generic algorithm it used for
//! every backend before byte-level vocabularies got a dedicated decoder.
//!
//! Set `SMG_BENCH_TOKENIZER_JSON=/path/to/tokenizer.json` to measure a real
//! vocabulary (Qwen, Llama 3, GPT-2 style); otherwise a small synthetic
//! byte-level BPE is built in memory.
#![allow(clippy::expect_used)]

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use llm_tokenizer::{
    Decoder, Encoder, HuggingFaceTokenizer, Sequence, SequenceDecoderOutput, StopSequenceConfig,
    StopSequenceDecoder, TokenizerTrait,
};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{Merges, Vocab, BPE},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
    AddedToken, Tokenizer as HfTokenizer,
};

const TEXT: &str = "The quick brown fox jumps over the lazy dog. Streaming responses are \
decoded one token at a time, so the per-token cost matters more than the cost of a \
single decode. Café, naïve, 日本語のテキスト, 한국어, emoji 😀👍🏽 and plain ASCII all \
flow through the same path. Numbers like 3.14159 and 2024-01-01, code such as \
`fn main() { println!(\"hello\"); }`, and punctuation!?;: round it out.";

fn synthetic_byte_level() -> HfTokenizer {
    let mut vocab = Vocab::default();
    for ch in ByteLevelPreTokenizer::alphabet() {
        let next = vocab.len() as u32;
        vocab.insert(ch.to_string(), next);
    }
    let pairs = [
        ("t", "h"),
        ("th", "e"),
        ("Ġ", "t"),
        ("Ġt", "h"),
        ("Ġth", "e"),
        ("i", "n"),
        ("o", "n"),
        ("e", "r"),
        ("Ã", "©"),
        ("ð", "Ł"),
    ];
    let mut merges = Merges::new();
    for (a, b) in pairs {
        let next = vocab.len() as u32;
        vocab.insert(format!("{a}{b}"), next);
        merges.push((a.to_string(), b.to_string()));
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("bpe");
    let mut tokenizer = HfTokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::default()));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true)])
        .expect("special token");
    tokenizer
}

fn load_tokenizer() -> (Arc<HuggingFaceTokenizer>, &'static str) {
    match std::env::var("SMG_BENCH_TOKENIZER_JSON") {
        Ok(path) => (
            Arc::new(HuggingFaceTokenizer::from_file(&path).expect("load tokenizer")),
            "real",
        ),
        Err(_) => (
            Arc::new(HuggingFaceTokenizer::from_tokenizer(synthetic_byte_level())),
            "synthetic",
        ),
    }
}

fn bench_decode(c: &mut Criterion) {
    let (tokenizer, kind) = load_tokenizer();
    let ids: Vec<u32> = tokenizer
        .encode(TEXT, false)
        .expect("encode")
        .token_ids()
        .to_vec();
    let dyn_tokenizer: Arc<dyn TokenizerTrait> = tokenizer.clone();

    let mut group = c.benchmark_group(format!("incremental_decode/{kind}"));
    group.throughput(Throughput::Elements(ids.len() as u64));

    group.bench_function("sequence_append_token", |b| {
        b.iter_batched(
            || Sequence::new(dyn_tokenizer.clone()),
            |mut seq| {
                let mut total = 0;
                for &id in &ids {
                    total += seq.append_token(id).expect("step").len();
                }
                total
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("generic_decode_step", |b| {
        b.iter_batched(
            || (Vec::new(), String::new(), 0usize),
            |(mut window, mut prefix, mut prefix_index)| {
                let mut total = 0;
                for &id in &ids {
                    if let Some(text) = tokenizer
                        .decode_step(id, &mut window, &mut prefix, &mut prefix_index, false)
                        .expect("step")
                    {
                        total += text.len();
                    }
                }
                total
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("stop_decoder_process_token", |b| {
        let config = StopSequenceConfig::default()
            .with_stop_sequence("<|im_end|>")
            .with_stop_sequence("\n\nHuman:");
        b.iter_batched(
            || StopSequenceDecoder::new(dyn_tokenizer.clone(), config.clone(), false),
            |mut decoder| {
                let mut total = 0;
                for &id in &ids {
                    total += match decoder.process_token(id).expect("token") {
                        SequenceDecoderOutput::Text(text)
                        | SequenceDecoderOutput::StoppedWithText(text) => text.len(),
                        SequenceDecoderOutput::Held | SequenceDecoderOutput::Stopped => 0,
                    };
                }
                total
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("full_decode_once", |b| {
        b.iter(|| tokenizer.decode(&ids, false).expect("decode").len());
    });
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
