//! Opt-in: the L1 prefix cache against the plain encode on recorded render
//! prompts. `SMG_L1_PARITY_DIRS` lists `<name>=<tokenizer directory>` pairs
//! (comma-separated), `SMG_L1_PARITY_ROOT` the fixtures root holding
//! `<name>/render/*.jsonl` with a `reference.text` field per line, and
//! `SMG_L1_PARITY_CASES` the prompts per vocabulary (default 300, spread over
//! the files). Every prompt is encoded cold (a miss that seeds the cache),
//! again (a hit), as the next turn of its conversation (a hit plus a suffix)
//! and as a sibling conversation (its first half plus other text), under both
//! `add_special_tokens` values, and compared with the inner tokenizer's plain
//! encode. Without the environment the test passes without running.
#![expect(
    clippy::print_stdout,
    reason = "an opt-in parity run reports per vocabulary"
)]

use std::{
    io::{BufRead, BufReader},
    sync::Arc,
};

use llm_tokenizer::{
    create_tokenizer,
    traits::{Encoder as _, Tokenizer},
    CacheConfig, CachedTokenizer,
};
use serde_json::Value;

const DIRS_ENV: &str = "SMG_L1_PARITY_DIRS";
const ROOT_ENV: &str = "SMG_L1_PARITY_ROOT";
const CASES_ENV: &str = "SMG_L1_PARITY_CASES";

fn prompts(root: &str, name: &str, limit: usize) -> Vec<String> {
    let dir = std::path::Path::new(root).join(name).join("render");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    let per_file = (limit / files.len().max(1)).max(1);
    let mut out = Vec::new();
    for file in files {
        let Ok(f) = std::fs::File::open(&file) else {
            continue;
        };
        for line in BufReader::new(f)
            .lines()
            .map_while(Result::ok)
            .take(per_file)
        {
            let Ok(v) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if let Some(text) = v["reference"]["text"].as_str() {
                if !text.is_empty() {
                    out.push(text.to_owned());
                }
            }
        }
    }
    out
}

/// The inputs derived from a prompt: itself (cold, then warm), its next turn,
/// and a sibling conversation sharing its first half.
fn derived(text: &str) -> [(&'static str, String); 4] {
    let next_turn = format!(
        "{text}<|im_start|>user\nAnd one more question about the same thing?<|im_end|>\n<|im_start|>assistant\n"
    );
    let mut cut = text.len() / 2;
    while !text.is_char_boundary(cut) {
        cut += 1;
    }
    let sibling = format!(
        "{}Different follow-up text here, with numbers 12345 and symbols #!? and a tail.",
        &text[..cut]
    );
    [
        ("cold", text.to_owned()),
        ("warm", text.to_owned()),
        ("next-turn", next_turn),
        ("sibling", sibling),
    ]
}

#[test]
fn the_l1_prefix_cache_matches_the_plain_encode_on_render_prompts() {
    let Ok(list) = std::env::var(DIRS_ENV) else {
        println!("{DIRS_ENV} unset: nothing to compare");
        return;
    };
    let root = std::env::var(ROOT_ENV).unwrap_or_default();
    let limit: usize = std::env::var(CASES_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let (mut vocabularies, mut compared, mut mismatches) = (0usize, 0usize, 0usize);
    let (mut hits, mut misses) = (0u64, 0u64);
    for item in list.split(',').filter(|s| !s.is_empty()) {
        let Some((name, dir)) = item.split_once('=') else {
            continue;
        };
        let inner: Arc<dyn Tokenizer> = match create_tokenizer(dir) {
            Ok(t) => t,
            Err(e) => {
                println!("L1-PARITY {name}: does not load: {e}");
                continue;
            }
        };
        let texts = prompts(&root, name, limit);
        if texts.is_empty() {
            println!("L1-PARITY {name}: no prompts");
            continue;
        }
        vocabularies += 1;
        let (mut compared_here, mut mismatched_here) = (0usize, 0usize);
        for add_special_tokens in [false, true] {
            let cached = CachedTokenizer::new(
                Arc::clone(&inner),
                CacheConfig {
                    enable_l0: false,
                    enable_l1: true,
                    l1_max_memory: 1 << 30,
                    ..Default::default()
                },
            );
            for text in &texts {
                for (label, input) in derived(text) {
                    let want = inner.encode(&input, add_special_tokens);
                    let got = cached.encode(&input, add_special_tokens);
                    compared_here += 1;
                    let same = match (&want, &got) {
                        (Ok(w), Ok(g)) => w.token_ids() == g.token_ids(),
                        (Err(_), Err(_)) => true,
                        _ => false,
                    };
                    if !same {
                        mismatched_here += 1;
                        if mismatched_here <= 3 {
                            let (w, g) = (
                                want.as_ref().map(|e| e.token_ids().len()).unwrap_or(0),
                                got.as_ref().map(|e| e.token_ids().len()).unwrap_or(0),
                            );
                            println!(
                                "L1-PARITY-MISMATCH {name} add_special_tokens={add_special_tokens} {label}: plain {w} ids, cached {g} ids"
                            );
                        }
                    }
                }
            }
            if let Some(stats) = cached.l1_cache_stats() {
                hits += stats.hits;
                misses += stats.misses;
            }
        }
        compared += compared_here;
        mismatches += mismatched_here;
        println!(
            "L1-PARITY {name}: prompts {} compared {compared_here} mismatches {mismatched_here}",
            texts.len()
        );
    }
    println!(
        "L1-PARITY-TOTAL vocabularies {vocabularies} compared {compared} mismatches {mismatches} l1 hits {hits} misses {misses}"
    );
    assert_eq!(
        mismatches, 0,
        "the L1 prefix cache returned other ids than the plain encode"
    );
}
