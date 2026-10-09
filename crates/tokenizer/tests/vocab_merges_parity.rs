//! Opt-in parity of a tokenizer built from a checkpoint's `vocab.json` +
//! `merges.txt` (no `tokenizer.json`) with the reference ids the Python
//! `transformers` tokenizer gives on the same files.
//!
//! Run with `VOCAB_MERGES_PARITY=<dir>=<cases.jsonl>[,<dir>=<cases.jsonl>...]`
//! and `--ignored`: each `dir` holds the checkpoint's tokenizer files and each
//! cases file has one JSON object per line with `text` and `ids`, the ids
//! `AutoTokenizer` on `dir` encodes `text` to with `add_special_tokens=False`.

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::{anyhow, Context, Result};
use llm_tokenizer::create_tokenizer;
use serde::Deserialize;

const CASES_ENV: &str = "VOCAB_MERGES_PARITY";

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
}

#[test]
#[ignore = "requires real vocabulary files and the reference ids the Python tokenizer gives"]
#[expect(
    clippy::print_stderr,
    reason = "the per-directory tally and the first differences are the test's diagnostic output"
)]
fn built_from_vocab_and_merges_matches_the_reference() -> Result<()> {
    let spec = std::env::var(CASES_ENV)
        .map_err(|_| anyhow!("set {CASES_ENV} to <dir>=<cases.jsonl>[,<dir>=<cases.jsonl>...]"))?;
    let mut failures = Vec::new();
    for entry in spec.split(',').filter(|entry| !entry.is_empty()) {
        let (dir, cases) = entry
            .split_once('=')
            .ok_or_else(|| anyhow!("{CASES_ENV} entry {entry:?} is not <dir>=<cases.jsonl>"))?;
        let tokenizer = create_tokenizer(dir)
            .with_context(|| format!("the tokenizer at {dir} does not load"))?;
        let reader = BufReader::new(File::open(cases).with_context(|| format!("open {cases}"))?);
        let (mut texts, mut differences) = (0usize, 0usize);
        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let case: Case = serde_json::from_str(&line)
                .with_context(|| format!("{cases}:{}: not a case", line_no + 1))?;
            texts += 1;
            let ids = tokenizer.encode(&case.text, false)?;
            if ids.token_ids() != case.ids.as_slice() {
                differences += 1;
                if differences <= 10 {
                    let shown: String = case.text.chars().take(80).collect();
                    eprintln!(
                        "{dir}: line {} differs: text {shown:?} ours {:?} reference {:?}",
                        line_no + 1,
                        &ids.token_ids()[..ids.token_ids().len().min(16)],
                        &case.ids[..case.ids.len().min(16)]
                    );
                }
            }
        }
        eprintln!("{dir}: {texts} texts, {differences} differences");
        if differences > 0 {
            failures.push(format!("{dir}: {differences} of {texts} texts differ"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
