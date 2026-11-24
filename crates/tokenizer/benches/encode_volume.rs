//! Per-token cost of one encode against the volume of text one thread has
//! encoded: a document of N tokens encoded again after one pass over it
//! (`repeat`: every piece it holds was seen once already) and a document of N
//! tokens no pass has seen, after documents of the same size (`fresh`). The
//! corpus is cache-hostile by default: pseudo-words from a vocabulary of some
//! 170,000 words, most of them seen once in a document, digits and short CJK
//! runs, no special tokens; `VOLUME_BENCH_CORPUS=prose` uses repeated
//! paragraphs instead.
//!
//! `VOLUME_BENCH_TOKENIZERS=name=dir,...` selects the tokenizers (without it
//! the bench says so on one line and exits without measuring, so a plain
//! `cargo bench` of the crate passes it by); `VOLUME_BENCH_TOKENS` the
//! document sizes (default 10000,50000,100000,200000,500000);
//! `VOLUME_BENCH_ROUNDS` the samples per cell (default 3). One JSON line per
//! measurement and a table at the end.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark setup and output"
)]

use std::{hint::black_box, time::Instant};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer};
use serde_json::json;

mod corpora;

const CORPUS_VAR: &str = "VOLUME_BENCH_CORPUS";

fn tokens_of(tokenizer: &dyn Tokenizer, text: &str) -> usize {
    tokenizer
        .encode(text, false)
        .expect("encode")
        .token_ids()
        .len()
}

/// Nanoseconds per token of one encode of `text`.
fn ns_per_token(tokenizer: &dyn Tokenizer, text: &str) -> f64 {
    let t0 = Instant::now();
    let encoding = black_box(tokenizer.encode(text, false).expect("encode"));
    t0.elapsed().as_secs_f64() * 1e9 / encoding.token_ids().len() as f64
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let Ok(specs) = std::env::var("VOLUME_BENCH_TOKENIZERS") else {
        println!(
            "VOLUME_BENCH_TOKENIZERS unset (name=dir,... selects the tokenizers): nothing measured"
        );
        return;
    };
    let sizes: Vec<usize> = std::env::var("VOLUME_BENCH_TOKENS")
        .unwrap_or_else(|_| "10000,50000,100000,200000,500000".into())
        .split(',')
        .map(|s| s.trim().parse().expect("token count"))
        .collect();
    let rounds: usize = std::env::var("VOLUME_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
        .max(1);
    let corpus = std::env::var(CORPUS_VAR).unwrap_or_else(|_| "varied".into());
    let mut table = Vec::new();
    for spec in specs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, dir) = spec.split_once('=').expect("name=dir");
        let tokenizer = create_tokenizer(dir).expect("tokenizer loads");
        let probe = corpora::body(CORPUS_VAR, 7, 40_000);
        let bytes_per_token = probe.len() as f64 / tokens_of(tokenizer.as_ref(), &probe) as f64;
        for &tokens in &sizes {
            let bytes = (tokens as f64 * bytes_per_token) as usize;
            // The same document, encoded once more after one pass over it.
            let text = corpora::body(CORPUS_VAR, tokens as u64, bytes);
            let measured = tokens_of(tokenizer.as_ref(), &text);
            let mut repeat: Vec<f64> = (0..rounds)
                .map(|_| ns_per_token(tokenizer.as_ref(), &text))
                .collect();
            // Documents of the same size no pass has seen, one after another.
            let mut fresh: Vec<f64> = (1..=rounds)
                .map(|round| {
                    let text =
                        corpora::body(CORPUS_VAR, (tokens as u64) << 20 | round as u64, bytes);
                    ns_per_token(tokenizer.as_ref(), &text)
                })
                .collect();
            let (repeat, fresh) = (median(&mut repeat), median(&mut fresh));
            println!(
                "{}",
                json!({"kind":"row","model":name,"corpus":corpus,"tokens":measured,"bytes":text.len(),
                    "rounds":rounds,"repeat_ns_per_token":repeat,"fresh_ns_per_token":fresh})
            );
            table.push((name.to_owned(), measured, repeat, fresh));
        }
    }
    println!(
        "\n{:<24} {:>9} {:>14} {:>14}",
        "model", "tokens", "repeat ns/tok", "fresh ns/tok"
    );
    for (model, tokens, repeat, fresh) in &table {
        println!("{model:<24} {tokens:>9} {repeat:>14.1} {fresh:>14.1}");
    }
}
