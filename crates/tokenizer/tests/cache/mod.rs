//! Encode cache integration tests: the exact-match and prefix caches, their correctness and their
//! parity with the uncached encoder.

use std::io::{BufRead, BufReader};

use serde_json::Value;

mod l0_exact_cache_parity;
mod l1_prefix_cache_parity;
mod tokenizer_cache_correctness_test;

/// The recorded render prompts under `<root>/<name>/render/*.jsonl` (the
/// `reference.text` field of each line), about `limit` of them spread over
/// the files (at least one line read per file); none when the directory is
/// missing.
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
