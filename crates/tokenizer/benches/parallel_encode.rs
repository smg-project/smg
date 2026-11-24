//! Wall time of one encode of a long prompt, at the thread count of the
//! process's pool: `RAYON_NUM_THREADS` sizes the pool the direct encoder
//! splits a long input over; with `tokenizers-v1`, `SMG_BENCH_THREADS` sets
//! that encoder's internal thread count instead. `PARALLEL_BENCH_TOKENIZERS`
//! = `name=dir,...` (without it the bench says so on one line and exits
//! without measuring, so a plain `cargo bench` of the crate passes it by),
//! `PARALLEL_BENCH_TOKENS` = the prompt sizes (default 131072,262144,1048576),
//! `PARALLEL_BENCH_ROUNDS` (default 5). One JSON line per measurement.
//! `PARALLEL_BENCH_CORPUS=prose` uses repeated paragraphs (the word reuse of
//! ordinary text) instead of the default wide vocabulary.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark setup and output"
)]

use std::{hint::black_box, time::Instant};

use llm_tokenizer::create_tokenizer;
use serde_json::json;

mod corpora;

const CORPUS_VAR: &str = "PARALLEL_BENCH_CORPUS";

fn main() {
    let Ok(specs) = std::env::var("PARALLEL_BENCH_TOKENIZERS") else {
        println!("PARALLEL_BENCH_TOKENIZERS unset (name=dir,... selects the tokenizers): nothing measured");
        return;
    };
    let sizes: Vec<usize> = std::env::var("PARALLEL_BENCH_TOKENS")
        .unwrap_or_else(|_| "131072,262144,1048576".into())
        .split(',')
        .map(|s| s.trim().parse().expect("token count"))
        .collect();
    let rounds: usize = std::env::var("PARALLEL_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let pool_threads = rayon::current_num_threads();
    #[cfg(feature = "tokenizers-v1")]
    let internal: usize = std::env::var("SMG_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    #[cfg(feature = "tokenizers-v1")]
    {
        tokenizers_v1::parallelism::set_num_threads(internal);
        tokenizers_v1::parallelism::set_parallelism(internal > 1);
    }
    #[cfg(not(feature = "tokenizers-v1"))]
    let internal = 1usize;
    for spec in specs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, dir) = spec.split_once('=').expect("name=dir");
        let tokenizer = create_tokenizer(dir).expect("tokenizer loads");
        let probe = corpora::body(CORPUS_VAR, 7, 40_000);
        let bytes_per_token = probe.len() as f64
            / tokenizer
                .encode(&probe, false)
                .expect("probe")
                .token_ids()
                .len() as f64;
        for &tokens in &sizes {
            let text = corpora::body(
                CORPUS_VAR,
                tokens as u64,
                (tokens as f64 * bytes_per_token) as usize,
            );
            let encoding = tokenizer.encode(&text, false).expect("encode");
            let measured = encoding.token_ids().len();
            let mut samples = Vec::with_capacity(rounds);
            for _ in 0..rounds {
                let t0 = Instant::now();
                black_box(tokenizer.encode(&text, false).expect("encode"));
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{}",
                json!({"kind":"row","model":name,"tokens":measured,"bytes":text.len(),
                    "corpus":std::env::var(CORPUS_VAR).unwrap_or_else(|_| "varied".into()),
                    "pool_threads":pool_threads,"internal_threads":internal,
                    "median_ms":samples[samples.len()/2],"min_ms":samples[0],"rounds":rounds,
                    "us_per_token":samples[samples.len()/2]*1e3/measured as f64})
            );
        }
    }
}
