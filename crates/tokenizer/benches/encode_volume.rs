//! Per-token cost of one encode against the volume of text one thread has
//! encoded: a document of N tokens encoded again after one pass over it
//! (`repeat`: every piece it holds was seen once already) and a document of N
//! tokens no pass has seen, after documents of the same size (`fresh`). The
//! corpus is cache-hostile by default: pseudo-words from a vocabulary wider
//! than a thread's piece-cache share, digits and short CJK runs, no special
//! tokens; `VOLUME_BENCH_CORPUS=prose` uses repeated paragraphs instead.
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

/// Repeated paragraphs with a running number: the word reuse of ordinary
/// text, which the piece caches serve well.
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

/// Pseudo-words of two to four syllables (a vocabulary of some 170,000
/// words, most of them seen once in a document), digits and short CJK runs
/// that hardly ever repeat; `VOLUME_BENCH_CORPUS=prose` selects [`prose`]
/// instead.
fn body(seed: u64, bytes: usize) -> String {
    if std::env::var("VOLUME_BENCH_CORPUS").is_ok_and(|v| v == "prose") {
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
    let corpus = std::env::var("VOLUME_BENCH_CORPUS").unwrap_or_else(|_| "varied".into());
    let mut table = Vec::new();
    for spec in specs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, dir) = spec.split_once('=').expect("name=dir");
        let tokenizer = create_tokenizer(dir).expect("tokenizer loads");
        let probe = body(7, 40_000);
        let bytes_per_token = probe.len() as f64 / tokens_of(tokenizer.as_ref(), &probe) as f64;
        for &tokens in &sizes {
            let bytes = (tokens as f64 * bytes_per_token) as usize;
            // The same document, encoded once more after one pass over it.
            let text = body(tokens as u64, bytes);
            let measured = tokens_of(tokenizer.as_ref(), &text);
            let mut repeat: Vec<f64> = (0..rounds)
                .map(|_| ns_per_token(tokenizer.as_ref(), &text))
                .collect();
            // Documents of the same size no pass has seen, one after another.
            let mut fresh: Vec<f64> = (1..=rounds)
                .map(|round| {
                    let text = body((tokens as u64) << 20 | round as u64, bytes);
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
