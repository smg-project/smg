//! The cost of streaming a long ByteFallback byte run that contains an invalid
//! byte early: the incremental decoder must not re-decode a window that grows
//! with every byte of the run (one decode per token of linear length made the
//! whole run quadratic). A WordLevel vocabulary with every byte token, no
//! model download.
#![allow(clippy::expect_used)]

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llm_tokenizer::{
    HuggingFaceTokenizer, SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder,
    TokenizerTrait,
};

fn byte_fallback_tokenizer() -> Arc<dyn TokenizerTrait> {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".into(), serde_json::json!(0));
    vocab.insert("x".into(), serde_json::json!(1));
    for byte in 0u32..=255 {
        vocab.insert(format!("<0x{byte:02X}>"), serde_json::json!(byte + 2));
    }
    let definition = serde_json::json!({
        "version": "1.0", "added_tokens": [],
        "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": vocab},
        "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]}
    });
    let native = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&definition).expect("json"))
        .expect("tokenizer");
    Arc::new(HuggingFaceTokenizer::from_tokenizer(native))
}

/// `tokens` byte tokens of CJK text (three bytes per character); with
/// `invalid_at`, the byte at that position is `0xFF`.
fn run(tokens: usize, invalid_at: Option<usize>) -> Vec<u32> {
    let text: String = "日本語".chars().cycle().take(tokens / 3 + 1).collect();
    let mut ids: Vec<u32> = text
        .bytes()
        .take(tokens)
        .map(|b| u32::from(b) + 2)
        .collect();
    if let Some(at) = invalid_at {
        ids[at] = 0xFF + 2;
    }
    ids.push(1);
    ids
}

fn stream(tokenizer: &Arc<dyn TokenizerTrait>, ids: &[u32]) -> usize {
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), true);
    let mut chars = 0;
    for &id in ids {
        if let SequenceDecoderOutput::Text(text) = decoder.process_token(id).expect("step") {
            chars += text.chars().count();
        }
    }
    if let SequenceDecoderOutput::Text(text) = decoder.finish().expect("finish") {
        chars += text.chars().count();
    }
    chars
}

fn bench_invalid_run(c: &mut Criterion) {
    let tokenizer = byte_fallback_tokenizer();
    let mut group = c.benchmark_group("byte_fallback_run");
    for tokens in [300usize, 1_000, 2_000] {
        group.throughput(Throughput::Elements(tokens as u64));
        let valid = run(tokens, None);
        let invalid = run(tokens, Some(9));
        group.bench_with_input(BenchmarkId::new("valid", tokens), &valid, |b, ids| {
            b.iter(|| stream(&tokenizer, ids));
        });
        group.bench_with_input(
            BenchmarkId::new("invalid_byte_early", tokens),
            &invalid,
            |b, ids| b.iter(|| stream(&tokenizer, ids)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_invalid_run);
criterion_main!(benches);
