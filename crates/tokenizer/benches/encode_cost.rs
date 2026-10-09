//! Encode cost per prompt token, per tokenizer and prompt length, with the
//! allocations and peak heap of one encode, and a breakdown of the Hugging
//! Face pipeline (normalizer, pre-tokenizer, model).
//!
//! Plain timing loops, no criterion: the numbers are read as a table.
//!
//! Env:
//! - `ENCODE_BENCH_TOKENIZERS`: `name=<dir>[@<prompt set>],...`; each dir is
//!   what the gateway's `--tokenizer` takes (tokenizer.json or tiktoken.model
//!   beside tokenizer_config.json). The prompt set defaults to `name`.
//! - `ENCODE_BENCH_PROMPTS`: a root with `<prompt set>/<N>.txt` (the prompt)
//!   and, optionally, `<N>.ids` (reference ids, one per line) to check.
//! - `ENCODE_BENCH_MAX_TOKENS`: skip prompts above this many reference tokens.
//! - `ENCODE_BENCH_BREAKDOWN=1`: also time the HF stages separately.
//! - `ENCODE_BENCH_THREADS=n`: encode the same prompt on n threads at once
//!   and report the per-thread µs/token (contention check).
#![expect(
    unsafe_code,
    reason = "a counting global allocator needs the unsafe GlobalAlloc trait"
)]
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a benchmark that prints its table and stops on a broken setup"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer};
use tokenizers::{
    Model, NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString,
    PreTokenizer, Tokenizer as HfTokenizer,
};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Heap size when the measured encode started; the peak is reported against it.
static BASELINE: AtomicUsize = AtomicUsize::new(0);

fn note_alloc(size: usize) {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    let cur = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(cur, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        if new_size > layout.size() {
            let cur = CURRENT.fetch_add(new_size - layout.size(), Ordering::Relaxed) + new_size
                - layout.size();
            PEAK.fetch_max(cur, Ordering::Relaxed);
        } else {
            CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn reset_counters() {
    ALLOCS.store(0, Ordering::Relaxed);
    let current = CURRENT.load(Ordering::Relaxed);
    BASELINE.store(current, Ordering::Relaxed);
    PEAK.store(current, Ordering::Relaxed);
}

/// Allocations since the reset, and the peak heap growth over the heap size
/// at the reset (so the result buffer and anything the encode kept count).
fn counters() -> (usize, usize) {
    (
        ALLOCS.load(Ordering::Relaxed),
        PEAK.load(Ordering::Relaxed)
            .saturating_sub(BASELINE.load(Ordering::Relaxed)),
    )
}

struct Prompt {
    name: String,
    text: String,
    reference: Option<Vec<u32>>,
}

fn load_prompts(root: &str, set: &str, max_tokens: usize) -> Vec<Prompt> {
    let dir = std::path::Path::new(root).join(set);
    let mut prompts = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("no prompt set at {}", dir.display());
        return prompts;
    };
    let mut names: Vec<(usize, String)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let stem = name.strip_suffix(".txt")?;
            Some((stem.parse::<usize>().ok()?, stem.to_string()))
        })
        .collect();
    names.sort();
    for (n, stem) in names {
        if n > max_tokens {
            continue;
        }
        let text = std::fs::read_to_string(dir.join(format!("{stem}.txt"))).expect("prompt text");
        let ids_path = dir.join(format!("{stem}.ids"));
        let reference = std::fs::read_to_string(&ids_path)
            .ok()
            .map(|s| reference_ids(&ids_path, &s));
        prompts.push(Prompt {
            name: stem,
            text,
            reference,
        });
    }
    prompts
}

/// The reference ids of a prompt, one per line: a line that is not an id is
/// an error (a silently dropped line could make a truncated reference "exact").
fn reference_ids(path: &std::path::Path, contents: &str) -> Vec<u32> {
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            let context = format!("{}:{}: not a token id: {line:?}", path.display(), index + 1);
            line.trim().parse().expect(&context)
        })
        .collect()
}

fn repetitions(tokens: usize) -> usize {
    (3_000_000 / tokens.max(1)).clamp(1, 2000)
}

fn main() {
    let specs = std::env::var("ENCODE_BENCH_TOKENIZERS").expect("ENCODE_BENCH_TOKENIZERS");
    let root = std::env::var("ENCODE_BENCH_PROMPTS").expect("ENCODE_BENCH_PROMPTS");
    let max_tokens: usize = std::env::var("ENCODE_BENCH_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let breakdown = std::env::var("ENCODE_BENCH_BREAKDOWN").is_ok_and(|v| v == "1");
    let threads: usize = std::env::var("ENCODE_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    println!(
        "{:<22} {:>9} {:>9} {:>11} {:>10} {:>10} {:>9} {:>11} {:>8}",
        "tokenizer/prompt",
        "tokens",
        "bytes",
        "us/tok cold",
        "us/tok min",
        "us/tok avg",
        "ms/req",
        "allocs/req",
        "peak MB"
    );
    for spec in specs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, rest) = spec.split_once('=').expect("name=dir");
        let (dir, set) = rest.split_once('@').unwrap_or((rest, name));
        let tok: Arc<dyn Tokenizer> = create_tokenizer(dir).expect("tokenizer loads");
        let hf = if breakdown {
            HfTokenizer::from_file(format!("{dir}/tokenizer.json")).ok()
        } else {
            None
        };
        for prompt in load_prompts(&root, set, max_tokens) {
            // Exactness against the reference ids, and the token count. This
            // first encode also times the cold path: nothing of this prompt is
            // in any word or piece cache yet.
            let t0 = Instant::now();
            let encoding = tok.encode(&prompt.text, false).expect("encode");
            let cold = t0.elapsed().as_secs_f64();
            let ids = encoding.token_ids();
            let tokens = ids.len();
            let exact = match &prompt.reference {
                Some(reference) if reference.as_slice() == ids => "exact",
                Some(reference) => {
                    let at = reference
                        .iter()
                        .zip(ids)
                        .position(|(a, b)| a != b)
                        .unwrap_or_else(|| reference.len().min(ids.len()));
                    println!(
                        "  !! {name}/{}: ids differ from the reference at token {at} ({} vs {} tokens)",
                        prompt.name,
                        ids.len(),
                        reference.len()
                    );
                    "DIFFERS"
                }
                None => "no-ref",
            };
            drop(encoding);
            // Allocations and peak heap of one encode.
            reset_counters();
            let encoding = tok.encode(&prompt.text, false).expect("encode");
            let (allocs, peak) = counters();
            drop(encoding);
            // Timing.
            let reps = repetitions(tokens);
            let mut best = f64::MAX;
            let mut total = 0.0;
            for _ in 0..reps {
                let t0 = Instant::now();
                let encoding = tok.encode(&prompt.text, false).expect("encode");
                let dt = t0.elapsed().as_secs_f64();
                std::hint::black_box(encoding.token_ids().len());
                best = best.min(dt);
                total += dt;
            }
            let avg = total / reps as f64;
            println!(
                "{:<22} {:>9} {:>9} {:>11.3} {:>10.3} {:>10.3} {:>9.2} {:>11} {:>8.1}  {exact}",
                format!("{name}/{}", prompt.name),
                tokens,
                prompt.text.len(),
                cold * 1e6 / tokens as f64,
                best * 1e6 / tokens as f64,
                avg * 1e6 / tokens as f64,
                avg * 1e3,
                allocs,
                peak as f64 / 1e6
            );
            if threads > 1 {
                let text = Arc::new(prompt.text.clone());
                // Every worker starts its timed loop at the same moment, so
                // the per-thread time is measured under `threads`-way
                // contention from the first encode to the last.
                let start = Arc::new(std::sync::Barrier::new(threads));
                let t0 = Instant::now();
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let tok = Arc::clone(&tok);
                        let text = Arc::clone(&text);
                        let start = Arc::clone(&start);
                        std::thread::spawn(move || {
                            let reps = repetitions(tokens).max(4) / 4;
                            start.wait();
                            let t0 = Instant::now();
                            for _ in 0..reps {
                                let e = tok.encode(&text, false).expect("encode");
                                std::hint::black_box(e.token_ids().len());
                            }
                            t0.elapsed().as_secs_f64() / reps as f64
                        })
                    })
                    .collect();
                let per: Vec<f64> = handles
                    .into_iter()
                    .map(|h| h.join().expect("join"))
                    .collect();
                let wall = t0.elapsed().as_secs_f64();
                let mean = per.iter().sum::<f64>() / per.len() as f64;
                println!(
                    "{:<22} {threads} threads: {:.3} us/tok per thread (wall {:.2} s)",
                    "",
                    mean * 1e6 / tokens as f64,
                    wall
                );
            }
            if let Some(hf) = &hf {
                stage_breakdown(hf, &prompt.text, tokens);
            }
        }
    }
}

fn stage_breakdown(hf: &HfTokenizer, text: &str, tokens: usize) {
    let reps = repetitions(tokens).clamp(1, 200);
    let time = |f: &mut dyn FnMut()| {
        let t0 = Instant::now();
        for _ in 0..reps {
            f();
        }
        t0.elapsed().as_secs_f64() / reps as f64 * 1e6 / tokens as f64
    };
    // 1. normalizer (NormalizedString build + normalize)
    let normalizer = hf.get_normalizer();
    let t_norm = time(&mut || {
        let mut n = NormalizedString::from(text);
        if let Some(norm) = normalizer {
            norm.normalize(&mut n).expect("normalize");
        }
        std::hint::black_box(n.get().len());
    });
    // 2. pre-tokenizer on the normalized string
    let mut normalized = NormalizedString::from(text);
    if let Some(norm) = normalizer {
        norm.normalize(&mut normalized).expect("normalize");
    }
    let pre = hf.get_pre_tokenizer();
    let t_pre = time(&mut || {
        let mut p = PreTokenizedString::from(normalized.clone());
        if let Some(pre) = pre {
            pre.pre_tokenize(&mut p).expect("pre_tokenize");
        }
        std::hint::black_box(
            p.get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .len(),
        );
    });
    let t_clone = time(&mut || {
        std::hint::black_box(normalized.clone().get().len());
    });
    // 3. model on each split
    let mut p = PreTokenizedString::from(normalized.clone());
    if let Some(pre) = pre {
        pre.pre_tokenize(&mut p).expect("pre_tokenize");
    }
    let splits: Vec<String> = p
        .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
        .into_iter()
        .map(|(s, _, _)| s.to_string())
        .collect();
    let model = hf.get_model();
    let t_model = time(&mut || {
        let mut n = 0;
        for s in &splits {
            n += model.tokenize(s).expect("tokenize").len();
        }
        std::hint::black_box(n);
    });
    // 4. the whole encode, and encode_fast (no offsets)
    let t_full = time(&mut || {
        std::hint::black_box(hf.encode(text, false).expect("encode").get_ids().len());
    });
    let t_fast = time(&mut || {
        std::hint::black_box(hf.encode_fast(text, false).expect("encode").get_ids().len());
    });
    println!(
        "{:<22} stages us/tok: normalizer {:.3}  pre-tokenizer {:.3} (of which clone {:.3})  model {:.3} ({} splits)  full encode {:.3}  encode_fast {:.3}",
        "", t_norm, t_pre, t_clone, t_model, splits.len(), t_full, t_fast
    );
}
