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

const SYLLABLES: &[&str] = &[
    "ka", "ri", "to", "ne", "mu", "sa", "lo", "vi", "dek", "pon", "tra", "sel", "qui", "zam",
    "for", "bli", "ost", "uno", "chi", "wen",
];

const PROSE: &[&str] = &[
    "The harbour wakes before the town does. Fishing boats slip out past the breakwater while the \
     streetlights are still on, and by the time the bakery opens its shutters the first catch is \
     already being sorted on the quay. ",
    "A good map is a story told in one glance: where the rivers bend, which roads were built for \
     carts and which for cars, why the old centre sits on the hill. Reading it slowly rewards you \
     with the reasons behind the shapes. ",
    "清晨的市场里，摊贩们把蔬菜一筐一筐地摆好，水珠还挂在菜叶上。买菜的人不多，大家慢慢地挑选。",
    "fn parse_header(line: &str) -> Option<(&str, &str)> {\n    let (name, value) = line.split_once(':')?;\n    \
     Some((name.trim(), value.trim()))\n}\n",
    "Mountain weather turns in minutes. A clear morning can close into cloud by the time the path \
     reaches the ridge, and the walkers who packed a jacket despite the forecast are the ones \
     still smiling at the hut. ",
];

/// Repeated paragraphs with a running number: the word reuse of ordinary text,
/// which the encoders' piece caches serve well.
fn prose(seed: u64, bytes: usize) -> String {
    let mut text = String::with_capacity(bytes + 256);
    let mut k = seed;
    while text.len() < bytes {
        text.push_str(PROSE[(k % PROSE.len() as u64) as usize]);
        text.push_str(&format!("({k}) "));
        k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1) >> 3;
    }
    text
}

/// Pseudo-words, digits and short CJK runs: a vocabulary wider than a
/// thread's piece-cache share, no special tokens (the cuts come from the
/// pre-tokenizer's pieces alone). `PARALLEL_BENCH_CORPUS=prose` selects
/// [`prose`] instead.
fn body(seed: u64, bytes: usize) -> String {
    if std::env::var("PARALLEL_BENCH_CORPUS").is_ok_and(|v| v == "prose") {
        return prose(seed, bytes);
    }
    let mut text = String::with_capacity(bytes + 64);
    let mut k = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        k = k
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        k >> 33
    };
    let mut in_sentence = 0;
    while text.len() < bytes {
        let roll = next();
        match roll % 11 {
            0 => {
                let n = 3 + (next() % 6) as usize;
                for _ in 0..n {
                    let c = char::from_u32(0x4E00 + (next() % 2000) as u32).unwrap_or('\u{4E2D}');
                    text.push(c);
                }
            }
            1 => text.push_str(&format!("{}", next() % 100_000)),
            _ => {
                let n = 2 + (next() % 3) as usize;
                for _ in 0..n {
                    text.push_str(SYLLABLES[(next() % SYLLABLES.len() as u64) as usize]);
                }
            }
        }
        in_sentence += 1;
        if in_sentence >= 8 + (roll % 7) as usize {
            text.push_str(".\n");
            in_sentence = 0;
        } else {
            text.push(' ');
        }
    }
    text
}

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
        let probe = body(7, 40_000);
        let bytes_per_token = probe.len() as f64
            / tokenizer
                .encode(&probe, false)
                .expect("probe")
                .token_ids()
                .len() as f64;
        for &tokens in &sizes {
            let text = body(tokens as u64, (tokens as f64 * bytes_per_token) as usize);
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
                    "corpus":std::env::var("PARALLEL_BENCH_CORPUS").unwrap_or_else(|_| "varied".into()),
                    "pool_threads":pool_threads,"internal_threads":internal,
                    "median_ms":samples[samples.len()/2],"min_ms":samples[0],"rounds":rounds,
                    "us_per_token":samples[samples.len()/2]*1e3/measured as f64})
            );
        }
    }
}
