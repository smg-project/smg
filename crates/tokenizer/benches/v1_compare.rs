//! Run the same harness in default and `tokenizers-v1` builds; see v1/README.md.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark setup and JSONL output"
)]

use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Barrier, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use llm_tokenizer::{traits::Encoder, CacheConfig, CachedTokenizer, HuggingFaceTokenizer};
use serde_json::{json, Value};
use tokenizers::{utils::parallelism as legacy_parallelism, Tokenizer};

const MODELS: &str = include_str!("v1/models.json");

fn nonce(n: u64) -> String {
    (0..64)
        .map(|bit| if n & (1 << bit) == 0 { " a" } else { " b" })
        .collect()
}

fn encode(t: &Tokenizer, text: &str) -> Vec<u32> {
    t.encode(text, false)
        .expect("reference encode")
        .get_ids()
        .to_vec()
}

fn body(t: &Tokenizer, tokens: usize) -> String {
    let mut text = String::new();
    for n in 0..tokens {
        text.push_str(&format!(
            "Record {n}: hello world. 请检查分词速度和结果。 fn f_{n}(x: u64) {{}}\n"
        ));
        if text.len() > tokens * 12 {
            break;
        }
    }
    assert!(encode(t, &text).len() > tokens);
    let ends: Vec<_> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain([text.len()])
        .collect();
    let (mut lo, mut hi) = (0, ends.len() - 1);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if encode(t, &text[..ends[mid]]).len() <= tokens - 16 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    text.truncate(ends[lo]);
    text.push_str(&" a".repeat(tokens - encode(t, &text).len()));
    assert_eq!(encode(t, &text).len(), tokens);
    text
}

struct Input {
    text: String,
    offset: Option<usize>,
}
impl Input {
    fn new(t: &Tokenizer, marker: &str, tokens: usize, scenario: &str) -> Self {
        let (text, offset) = if scenario == "flat" {
            (format!("{}\n{}", nonce(0), body(t, tokens - 65)), Some(0))
        } else {
            let prefix = tokens * if scenario == "prefix99" { 99 } else { 90 } / 100;
            let mut text = nonce(0);
            text.push_str(marker);
            for n in 0..7 {
                text.push_str(&body(
                    t,
                    (prefix - 72) / 7 + usize::from(n < (prefix - 72) % 7),
                ));
                text.push_str(marker);
            }
            assert_eq!(encode(t, &text).len(), prefix);
            let offset = text.len();
            text.push_str(&nonce(0));
            text.push('\n');
            text.push_str(&body(t, tokens - prefix - 65));
            let offset = match scenario {
                "repeat" => None,
                "unique" => Some(0),
                _ => Some(offset),
            };
            (text, offset)
        };
        assert_eq!(encode(t, &text).len(), tokens);
        Self { text, offset }
    }
    fn update(&self, text: &mut String, n: u64) {
        if let Some(offset) = self.offset {
            text.replace_range(offset..offset + 128, &nonce(n));
        }
    }
}

fn counters(t: &CachedTokenizer) -> [u64; 4] {
    let l0 = t.cache_stats();
    let l1 = t.l1_cache_stats();
    [
        l0.as_ref().map_or(0, |s| s.hits),
        l0.as_ref().map_or(0, |s| s.misses),
        l1.as_ref().map_or(0, |s| s.hits),
        l1.as_ref().map_or(0, |s| s.misses),
    ]
}

fn measure(
    t: &CachedTokenizer,
    input: &Input,
    tokens: usize,
    concurrency: usize,
    next: &AtomicU64,
) -> Value {
    let ready = Barrier::new(concurrency + 1);
    let go = Barrier::new(concurrency + 1);
    let epoch = OnceLock::new();
    let (mut times, wall, before) = thread::scope(|scope| {
        let handles: Vec<_> = (0..concurrency)
            .map(|_| {
                let ready = &ready;
                let go = &go;
                let epoch = &epoch;
                scope.spawn(move || {
                    let mut text = input.text.clone();
                    for _ in 0..2 {
                        input.update(&mut text, next.fetch_add(1, Ordering::Relaxed));
                        std::hint::black_box(t.encode(&text, false).expect("warm encode"));
                    }
                    ready.wait();
                    go.wait();
                    let start: Instant = *epoch.get().expect("start");
                    let mut samples = Vec::new();
                    while start.elapsed() < Duration::from_millis(300) {
                        input.update(&mut text, next.fetch_add(1, Ordering::Relaxed));
                        let now = Instant::now();
                        let result = t.encode(&text, false).expect("encode");
                        samples.push(now.elapsed().as_secs_f64() * 1000.0);
                        assert_eq!(result.token_ids().len(), tokens);
                        std::hint::black_box(result);
                    }
                    samples
                })
            })
            .collect();
        ready.wait();
        let before = counters(t);
        epoch.set(Instant::now()).expect("set start");
        go.wait();
        let samples = handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker"))
            .collect::<Vec<_>>();
        (
            samples,
            epoch.get().expect("start").elapsed().as_secs_f64(),
            before,
        )
    });
    times.sort_by(f64::total_cmp);
    let q = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
    let after = counters(t);
    let delta: Vec<_> = after.iter().zip(before).map(|(a, b)| a - b).collect();
    json!({"requests":times.len(),"wall_s":wall,"rps":times.len() as f64/wall,
        "p50_ms":q(0.5),"p95_ms":q(0.95),"p99_ms":q(0.99),
        "cache_delta":delta,"l1_budgeted_bytes":t.l1_cache_stats().map_or(0,|s|s.memory_bytes)})
}

fn main() {
    let root = PathBuf::from(std::env::var("SMG_BENCH_MODEL_ROOT").expect("SMG_BENCH_MODEL_ROOT"));
    let internal: usize = std::env::var("SMG_BENCH_THREADS")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .expect("threads");
    legacy_parallelism::set_parallelism(false);
    #[cfg(feature = "tokenizers-v1")]
    {
        tokenizers_v1::parallelism::set_num_threads(internal);
        tokenizers_v1::parallelism::set_parallelism(internal > 1);
    }
    let models: Vec<Value> = serde_json::from_str(MODELS).expect("models");
    for model in models {
        let name = model["name"].as_str().expect("name");
        let path = root.join(name).join("tokenizer.json");
        let reference = Tokenizer::from_file(&path).expect("model");
        let started = Instant::now();
        let ours =
            Arc::new(HuggingFaceTokenizer::from_file(path.to_str().expect("path")).expect("load"));
        let load_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(
            ours.uses_v1_encoder(),
            cfg!(feature = "tokenizers-v1"),
            "{name}"
        );
        let hash = blake3::hash(&fs::read(&path).expect("model bytes"))
            .to_hex()
            .to_string();
        println!(
            "{}",
            json!({"kind":"model","model":name,"revision":model["revision"],
            "blake3":hash,"v1":ours.uses_v1_encoder(),"load_ms":load_ms,"internal_threads":internal})
        );
        for tokens in [128, 2048, 50000, 80000] {
            let scenarios: &[&str] = if tokens < 50000 {
                &["flat"]
            } else {
                &["flat", "unique", "repeat", "prefix90", "prefix99"]
            };
            for &scenario in scenarios {
                let input = Input::new(
                    &reference,
                    model["marker"].as_str().expect("marker"),
                    tokens,
                    scenario,
                );
                let configs: &[&str] = if scenario == "flat" {
                    &["off"]
                } else {
                    &["off", "l0", "l1", "both"]
                };
                for &cache in configs {
                    let concurrencies: &[usize] = if matches!(name, "minimax-m3" | "glm-5.3")
                        && tokens >= 50000
                        && matches!(scenario, "unique" | "repeat" | "prefix90")
                        && matches!(cache, "off" | "both")
                    {
                        &[1, 4, 16, 32]
                    } else {
                        &[1]
                    };
                    for &concurrency in concurrencies {
                        let cached = CachedTokenizer::new(
                            ours.clone(),
                            CacheConfig {
                                enable_l0: matches!(cache, "l0" | "both"),
                                enable_l1: matches!(cache, "l1" | "both"),
                                l1_max_memory: 1024 * 1024 * 1024,
                                ..Default::default()
                            },
                        );
                        // Check full IDs on both miss and hit paths before timing.
                        for n in [0, 1, 17, 65537] {
                            let mut text = input.text.clone();
                            input.update(&mut text, n);
                            let expected = encode(&reference, &text);
                            for _ in 0..2 {
                                assert_eq!(
                                    cached.encode(&text, false).expect("parity").token_ids(),
                                    expected
                                );
                            }
                        }
                        let mut cold = Vec::new();
                        for _ in 0..3 {
                            cached.clear_cache();
                            let now = Instant::now();
                            let ids = cached.encode(&input.text, false).expect("cold");
                            cold.push(now.elapsed().as_secs_f64() * 1000.0);
                            assert_eq!(ids.token_ids().len(), tokens);
                        }
                        let next = AtomicU64::new(1);
                        for round in 0..3 {
                            let mut row = measure(&cached, &input, tokens, concurrency, &next);
                            let obj = row.as_object_mut().expect("row");
                            for (key, value) in [
                                ("kind", json!("timing")),
                                ("model", json!(name)),
                                ("tokens", json!(tokens)),
                                ("scenario", json!(scenario)),
                                ("cache", json!(cache)),
                                ("concurrency", json!(concurrency)),
                                ("round", json!(round)),
                                ("internal_threads", json!(internal)),
                                ("v1", json!(ours.uses_v1_encoder())),
                                ("cold_ms", json!(cold)),
                            ] {
                                obj.insert(key.into(), value);
                            }
                            println!("{row}");
                        }
                    }
                }
            }
            // The embedding/batch API, separate from concurrent requests.
            let input = Input::new(&reference, "", tokens, "flat");
            let inputs = vec![input.text.as_str(); 8];
            let expected = encode(&reference, &input.text);
            for round in 0..6 {
                let now = Instant::now();
                let batch = ours.encode_batch(&inputs, false).expect("batch");
                let ms = now.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(batch.len(), 8);
                for e in batch {
                    assert_eq!(e.token_ids(), expected);
                }
                if round > 0 {
                    println!(
                        "{}",
                        json!({"kind":"batch","model":name,"tokens":tokens,
                    "v1":ours.uses_v1_encoder(),"internal_threads":internal,"ms":ms,"batch_size":8})
                    );
                }
            }
        }
    }
}
