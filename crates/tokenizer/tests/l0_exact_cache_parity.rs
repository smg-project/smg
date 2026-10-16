//! Opt-in: the L0 exact-match cache against the plain encode on recorded
//! render prompts. `SMG_L0_PARITY_DIRS` lists `<name>=<tokenizer directory>`
//! pairs (comma-separated), `SMG_L0_PARITY_ROOT` the fixtures root holding
//! `<name>/render/*.jsonl` with a `reference.text` field per line, and
//! `SMG_L0_PARITY_CASES` the prompts per vocabulary (default 300, spread over
//! the files). Every prompt is encoded with the cache off, cold with it on (a
//! miss that fills it), warm (a hit) and as a near twin (its last character
//! changed: a miss that must not return the prompt's ids), under both
//! `add_special_tokens` values, and every result is compared id for id with
//! the inner tokenizer's plain encode. Without the environment the test
//! passes without running.
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

const DIRS_ENV: &str = "SMG_L0_PARITY_DIRS";
const ROOT_ENV: &str = "SMG_L0_PARITY_ROOT";
const CASES_ENV: &str = "SMG_L0_PARITY_CASES";

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

/// The prompt with its last character replaced: the same length (for a
/// one-byte character) and the same text up to the end.
fn near_twin(text: &str) -> String {
    let mut twin: String = text.chars().take(text.chars().count() - 1).collect();
    twin.push(if text.ends_with('\n') { ' ' } else { '\n' });
    twin
}

fn cached(inner: &Arc<dyn Tokenizer>, enable_l0: bool) -> CachedTokenizer {
    CachedTokenizer::new(
        Arc::clone(inner),
        CacheConfig {
            enable_l0,
            l0_max_entries: 10_000,
            l0_max_memory: 1 << 30,
            enable_l1: false,
            ..Default::default()
        },
    )
}

#[test]
fn the_l0_exact_match_cache_matches_the_plain_encode_on_render_prompts() {
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
                println!("L0-PARITY {name}: does not load: {e}");
                continue;
            }
        };
        let texts = prompts(&root, name, limit);
        if texts.is_empty() {
            println!("L0-PARITY {name}: no prompts");
            continue;
        }
        vocabularies += 1;
        let (mut compared_here, mut mismatched_here) = (0usize, 0usize);
        for add_special_tokens in [false, true] {
            let off = cached(&inner, false);
            let on = cached(&inner, true);
            for text in &texts {
                let twin = near_twin(text);
                let want = inner.encode(text, add_special_tokens);
                let want_twin = inner.encode(&twin, add_special_tokens);
                let got = [
                    ("off", off.encode(text, add_special_tokens), &want),
                    ("cold", on.encode(text, add_special_tokens), &want),
                    ("warm", on.encode(text, add_special_tokens), &want),
                    ("twin", on.encode(&twin, add_special_tokens), &want_twin),
                    (
                        "warm-twin",
                        on.encode(&twin, add_special_tokens),
                        &want_twin,
                    ),
                ];
                for (label, got, want) in got {
                    compared_here += 1;
                    let same = match (want, &got) {
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
                                "L0-PARITY-MISMATCH {name} add_special_tokens={add_special_tokens} {label}: plain {w} ids, cached {g} ids"
                            );
                        }
                    }
                }
            }
            if let Some(stats) = on.cache_stats() {
                // Every warm encode hit; a prompt recorded twice hits cold too.
                assert!(
                    stats.hits >= 2 * texts.len() as u64,
                    "{name}: {} hits for {} prompts",
                    stats.hits,
                    texts.len()
                );
                hits += stats.hits;
                misses += stats.misses;
            }
        }
        compared += compared_here;
        mismatches += mismatched_here;
        println!(
            "L0-PARITY {name}: prompts {} compared {compared_here} mismatches {mismatched_here}",
            texts.len()
        );
    }
    println!(
        "L0-PARITY-TOTAL vocabularies {vocabularies} compared {compared} mismatches {mismatches} l0 hits {hits} misses {misses}"
    );
    // The variable is set: something must have been compared, or the run
    // would pass having checked nothing (a vocabulary that does not load, a
    // wrong root, a missing render directory).
    assert!(
        vocabularies > 0 && compared > 0,
        "{DIRS_ENV} is set but no vocabulary produced prompts to compare"
    );
    assert_eq!(
        mismatches, 0,
        "the L0 exact-match cache returned other ids than the plain encode"
    );
}
