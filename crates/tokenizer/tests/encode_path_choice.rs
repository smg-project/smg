//! The encode path each recorded fixture model takes, with the
//! `tokenizers-v1` feature off and on, so that a change to the choice shows
//! up as a changed row rather than as a slower or faster encode.
//!
//! Opt-in: `SMG_ENCODE_PATH_TOKENIZER_DIRS` lists `<slug>=<dir>` pairs
//! (comma-separated; each dir what the gateway's `--tokenizer-path` takes,
//! the recorded revision of the model). Every slug given must be a row of
//! [`EXPECTED`], the fixture models (the bellwether recorded set) with the
//! path each takes in either feature state; the test loads each through the
//! crate's factory, reads the path and compares. Without the variable the
//! test prints a notice and passes; the in-repo unit tests cover the choice
//! on synthetic vocabularies.
#![expect(clippy::print_stdout, reason = "the test reports what it compared")]

use std::collections::BTreeMap;

use llm_tokenizer::{create_tokenizer, EncodePath, HuggingFaceTokenizer};

const DIRS_ENV: &str = "SMG_ENCODE_PATH_TOKENIZER_DIRS";

/// What a fixture model's encodes go through: a Hugging Face tokenizer's
/// path, or a tokenizer outside that wrapper (a tiktoken vocabulary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    Hf(EncodePath),
    Tiktoken,
}

use Expected::{Hf, Tiktoken};

/// Slug, the path with the feature off, the path with the feature on.
const EXPECTED: &[(&str, Expected, Expected)] = &[
    (
        "ai21-jamba2-3b",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "apertus-8b-instruct-2509",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "deepseek-r1",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "deepseek-v3-0324",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "deepseek-v3.1",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "deepseek-v4.1-flash",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "dots3-note-prev",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "ernie-4.5-21b-a3b-thinking",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "gemma-4-e4b-it",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    ("glm-4.6", Hf(EncodePath::Tokenizers), Hf(EncodePath::V1)),
    ("glm-4.7-flash", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    ("glm-5.3-flash", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    ("granite-4.1-3b", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    ("hermes-4-14b", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    (
        "hunyuan-a13b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    ("hy4-preview", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    ("inkling", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    ("iquest-q1", Hf(EncodePath::Native), Hf(EncodePath::Native)),
    (
        "k-exaone-236b-a23b",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::Tokenizers),
    ),
    (
        "k2-horizon-36b",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::Tokenizers),
    ),
    ("kimi-k3", Tiktoken, Tiktoken),
    (
        "laguna-xs.2",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::Tokenizers),
    ),
    (
        "lfm2.5-1.2b-instruct",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "ling-3.0-flash",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::Tokenizers),
    ),
    (
        "llama-xlam-2-8b-fc-r",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "llava-1.5-7b-hf",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    ("mimo-v2.5", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    (
        "minicpm5-2b",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::Tokenizers),
    ),
    ("minimax-m2", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    ("minimax-m2.7", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    ("minimax-m3", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    (
        "mistral-7b-instruct-v0.3",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "muse-glimmer-30b",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "nanbeige4.2-3b",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "nvidia-nemotron-3-nano-30b-a3b-bf16",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "olmo-3-7b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "phi-4-mini-instruct",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "phi-4-multimodal-instruct",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "qwen-agentworld-35b-a3b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "qwen-drive-1.0-4b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "qwen2.5-7b-instruct-1m",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen2.5-omni-7b",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen2.5-vl-32b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen2.5-vl-7b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    ("qwen3-30b-a3b", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    (
        "qwen3-30b-a3b-instruct-2507",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-30b-a3b-thinking-2507",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-4b-instruct-2507",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-4b-saferl",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-4b-thinking-2507",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    ("qwen3-8b", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    (
        "qwen3-coder-30b-a3b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-coder-next",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-next-80b-a3b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-next-80b-a3b-thinking",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-omni-30b-a3b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-omni-30b-a3b-thinking",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-vl-235b-a22b-thinking",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-vl-30b-a3b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-vl-8b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3-vl-8b-thinking",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    (
        "qwen3.5-27b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    ("qwen3.5-2b", Hf(EncodePath::Native), Hf(EncodePath::Native)),
    (
        "qwen3.5-35b-a3b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    ("qwen3.5-9b", Hf(EncodePath::Native), Hf(EncodePath::Native)),
    (
        "qwen3.6-27b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "qwen3.6-35b-a3b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "qwen3.8-2.4t-a95b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "qwen3.8-27b",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "qwen3.8-flash-next",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "qwen3guard-gen-0.6b",
        Hf(EncodePath::Native),
        Hf(EncodePath::V1),
    ),
    ("qwq-32b", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    (
        "seed-oss-36b-instruct",
        Hf(EncodePath::Native),
        Hf(EncodePath::Native),
    ),
    (
        "step-3.5-flash",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    ("step3", Hf(EncodePath::Native), Hf(EncodePath::V1)),
    (
        "tinyllama-1.1b-chat-v1.0",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::V1),
    ),
    (
        "trinity-mini",
        Hf(EncodePath::Tokenizers),
        Hf(EncodePath::Tokenizers),
    ),
    ("webworld-32b", Hf(EncodePath::Native), Hf(EncodePath::V1)),
];

#[test]
fn every_fixture_model_takes_the_path_of_its_row() {
    let Ok(dirs) = std::env::var(DIRS_ENV) else {
        println!("{DIRS_ENV} is not set: skipping the fixture models' encode path check");
        return;
    };
    let feature_on = cfg!(feature = "tokenizers-v1");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut wrong = Vec::new();
    let mut checked = 0usize;
    for pair in dirs.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (slug, dir) = pair
            .split_once('=')
            .unwrap_or_else(|| panic!("{DIRS_ENV}: {pair:?} is not <slug>=<dir>"));
        let &(_, off, on) = EXPECTED
            .iter()
            .find(|(row, ..)| *row == slug)
            .unwrap_or_else(|| panic!("{slug} is not a row of EXPECTED"));
        let expected = if feature_on { on } else { off };
        let tokenizer = create_tokenizer(dir).unwrap_or_else(|e| panic!("{slug}: {e}"));
        let actual = tokenizer
            .as_any()
            .downcast_ref::<HuggingFaceTokenizer>()
            .map_or(Tiktoken, |hf| Hf(hf.encode_path()));
        checked += 1;
        *counts.entry(format!("{actual:?}")).or_default() += 1;
        if actual != expected {
            wrong.push(format!("{slug}: {actual:?}, the row says {expected:?}"));
        }
    }
    println!(
        "feature {}: {checked} of {} fixture models checked; {}",
        if feature_on { "on" } else { "off" },
        EXPECTED.len(),
        counts
            .iter()
            .map(|(path, n)| format!("{path} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(checked > 0, "{DIRS_ENV} names no tokenizer directory");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn the_rows_are_unique_and_in_order() {
    for pair in EXPECTED.windows(2) {
        assert!(pair[0].0 < pair[1].0, "{} before {}", pair[0].0, pair[1].0);
    }
}
