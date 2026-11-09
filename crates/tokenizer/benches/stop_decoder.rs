//! Micro-benchmarks for StopSequenceDecoder and incremental Sequence decoding.
//!
//! Run on this branch:
//!   OPENSSL_NO_VENDOR=1 cargo bench -p llm-tokenizer --bench stop_decoder
//!
//! Compare against origin/main:
//!   git stash && git checkout origin/main
//!   OPENSSL_NO_VENDOR=1 cargo bench -p llm-tokenizer --bench stop_decoder -- --save-baseline main
//!   git checkout - && git stash pop
//!   OPENSSL_NO_VENDOR=1 cargo bench -p llm-tokenizer --bench stop_decoder -- --baseline main

use std::{hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use llm_tokenizer::{
    mock::MockTokenizer,
    sequence::Sequence,
    stop::{StopRules, StopSequenceConfig, StopSequenceDecoder},
};

/// Benchmark the full stop decoder processing N tokens with varying stop sequence counts.
fn bench_stop_decoder_process_tokens(c: &mut Criterion) {
    let tokenizer = Arc::new(MockTokenizer::new());
    let mut group = c.benchmark_group("stop_decoder_process_tokens");

    // Token cycle: Hello world test token . Hello world test token .
    let token_cycle: Vec<u32> = vec![1, 2, 3, 4, 6];

    for num_stops in [1, 5, 10] {
        let stop_sequences: Vec<String> = (0..num_stops)
            .map(|i| format!("STOP_SEQUENCE_{i}_END"))
            .collect();

        for num_tokens in [100, 500, 2000] {
            let tokens: Vec<u32> = token_cycle
                .iter()
                .cycle()
                .take(num_tokens)
                .copied()
                .collect();

            group.bench_with_input(
                BenchmarkId::new(format!("{num_stops}_stops"), format!("{num_tokens}_tokens")),
                &tokens,
                |b, tokens| {
                    b.iter(|| {
                        let mut config = StopSequenceConfig::default();
                        for s in &stop_sequences {
                            config = config.with_stop_sequence(s);
                        }
                        let mut decoder =
                            StopSequenceDecoder::new(tokenizer.clone(), config, false);
                        for &token_id in tokens {
                            let _ = black_box(decoder.process_token(token_id));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark stop decoder with a stop sequence that triggers near the end.
fn bench_stop_decoder_with_match(c: &mut Criterion) {
    let tokenizer = Arc::new(MockTokenizer::new());
    let mut group = c.benchmark_group("stop_decoder_with_match");

    // "Hello world" is token 1 then 2, mock decodes as "Hello world"
    // Use "Hello world" as the stop sequence — it will match after token 2
    for num_tokens_before_stop in [10, 100, 500] {
        let tokens_before: Vec<u32> = vec![3, 4, 6] // "test token ."
            .into_iter()
            .cycle()
            .take(num_tokens_before_stop)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_tokens_before_stop}_before_match")),
            &tokens_before,
            |b, tokens_before| {
                b.iter(|| {
                    let config = StopSequenceConfig::default().with_stop_sequence("Hello world");
                    let mut decoder = StopSequenceDecoder::new(tokenizer.clone(), config, false);

                    // Process tokens that don't match
                    for &token_id in tokens_before {
                        let _ = black_box(decoder.process_token(token_id));
                    }
                    // Trigger the match
                    let _ = black_box(decoder.process_token(1)); // "Hello"
                    let _ = black_box(decoder.process_token(2)); // "world"
                });
            },
        );
    }

    group.finish();
}

/// Benchmark token-only stop (no string stops — fast path).
fn bench_stop_decoder_token_only(c: &mut Criterion) {
    let tokenizer = Arc::new(MockTokenizer::new());
    let mut group = c.benchmark_group("stop_decoder_token_only");

    let token_cycle: Vec<u32> = vec![1, 2, 3, 4, 6];

    for num_tokens in [100, 500, 2000] {
        let tokens: Vec<u32> = token_cycle
            .iter()
            .cycle()
            .take(num_tokens)
            .copied()
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_tokens}_tokens")),
            &tokens,
            |b, tokens| {
                b.iter(|| {
                    // Only token-level stop, no string stop sequences
                    let config = StopSequenceConfig::default().with_stop_token(999);
                    let mut decoder = StopSequenceDecoder::new(tokenizer.clone(), config, false);
                    for &token_id in tokens {
                        let _ = black_box(decoder.process_token(token_id));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Sequence::append_token incremental decoding.
fn bench_sequence_append_token(c: &mut Criterion) {
    let tokenizer = Arc::new(MockTokenizer::new());
    let mut group = c.benchmark_group("sequence_append_token");

    let token_cycle: Vec<u32> = vec![1, 2, 3, 4, 6];

    for num_tokens in [100, 500, 2000] {
        let tokens: Vec<u32> = token_cycle
            .iter()
            .cycle()
            .take(num_tokens)
            .copied()
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_tokens}_tokens")),
            &tokens,
            |b, tokens| {
                b.iter(|| {
                    let mut seq = Sequence::new(tokenizer.clone());
                    for &token_id in tokens {
                        let _ = black_box(seq.append_token(token_id));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark per-token latency at different sequence positions.
/// Measures whether performance degrades as the sequence grows
/// (the old code had unbounded buffer growth).
fn bench_per_token_at_position(c: &mut Criterion) {
    let tokenizer = Arc::new(MockTokenizer::new());
    let mut group = c.benchmark_group("per_token_at_position");
    let token_cycle: Vec<u32> = vec![1, 2, 3, 4, 6];

    // For each position, pre-fill the decoder to that point, then benchmark
    // a single process_token call.
    for position in [100, 500, 1000, 2000, 5000] {
        let prefill_tokens: Vec<u32> = token_cycle.iter().cycle().take(position).copied().collect();

        // With 5 string stop sequences (typical case)
        let stop_sequences: Vec<String> =
            (0..5).map(|i| format!("STOP_SEQUENCE_{i}_END")).collect();

        group.bench_with_input(
            BenchmarkId::new("5_stops", format!("pos_{position}")),
            &prefill_tokens,
            |b, prefill_tokens| {
                b.iter_batched(
                    || {
                        // Setup: create decoder and prefill to target position
                        let mut config = StopSequenceConfig::default();
                        for s in &stop_sequences {
                            config = config.with_stop_sequence(s);
                        }
                        let mut decoder =
                            StopSequenceDecoder::new(tokenizer.clone(), config, false);
                        for &token_id in prefill_tokens {
                            let _ = decoder.process_token(token_id);
                        }
                        decoder
                    },
                    |mut decoder| {
                        // Measure: single token at this position
                        let _ = black_box(decoder.process_token(3));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Token-only (no string stops)
        group.bench_with_input(
            BenchmarkId::new("token_only", format!("pos_{position}")),
            &prefill_tokens,
            |b, prefill_tokens| {
                b.iter_batched(
                    || {
                        let config = StopSequenceConfig::default().with_stop_token(999);
                        let mut decoder =
                            StopSequenceDecoder::new(tokenizer.clone(), config, false);
                        for &token_id in prefill_tokens {
                            let _ = decoder.process_token(token_id);
                        }
                        decoder
                    },
                    |mut decoder| {
                        let _ = black_box(decoder.process_token(3));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark a choice's decoder construction: from the request's stop parameters, the config
/// built and compiled per decoder, against a copy of rules built once per request.
fn bench_stop_decoder_build(c: &mut Criterion) {
    let tokenizer = Arc::new(MockTokenizer::new());
    let mut group = c.benchmark_group("stop_decoder_build");

    for num_stops in [1, 4, 16] {
        let stop_sequences: Vec<String> = (0..num_stops)
            .map(|i| format!("STOP_SEQUENCE_{i}_END"))
            .collect();
        let config = || {
            let mut config = StopSequenceConfig::default().with_stop_token(2);
            for s in &stop_sequences {
                config = config.with_stop_sequence(s);
            }
            config
        };
        group.bench_with_input(
            BenchmarkId::new("from_config", format!("{num_stops}_stops")),
            &num_stops,
            |b, _| {
                b.iter(|| black_box(StopSequenceDecoder::new(tokenizer.clone(), config(), false)));
            },
        );
        let rules = StopRules::new(config());
        group.bench_with_input(
            BenchmarkId::new("from_rules", format!("{num_stops}_stops")),
            &num_stops,
            |b, _| {
                b.iter(|| {
                    black_box(StopSequenceDecoder::with_rules(
                        tokenizer.clone(),
                        rules.clone(),
                        false,
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_stop_decoder_process_tokens,
    bench_stop_decoder_with_match,
    bench_stop_decoder_token_only,
    bench_sequence_append_token,
    bench_per_token_at_position,
    bench_stop_decoder_build,
);
criterion_main!(benches);
