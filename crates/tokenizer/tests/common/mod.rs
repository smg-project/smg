//! Common test utilities for tokenizer tests
#![allow(dead_code)] // each integration test binary uses a subset of these helpers

pub mod counting_alloc;
pub mod hub_fetch;

use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use rustls::crypto::ring;
use serde_json::{json, Map, Value};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{Merges, Vocab, BPE},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
    processors::byte_level::ByteLevel as ByteLevelProcessor,
    AddedToken, Tokenizer,
};

/// The blocking HTTP client of the download helpers. The client takes the
/// process-level TLS crypto provider (reqwest's `rustls-no-provider`
/// feature) and panics at build time when there is none, so `ring` is
/// installed first; a provider installed earlier in the process stays.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
pub fn http_client() -> reqwest::blocking::Client {
    let _ = ring::default_provider().install_default();
    reqwest::blocking::Client::new()
}

/// A builder for a download client with settings of its own, the provider
/// installed the same way.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
pub fn http_client_builder() -> reqwest::blocking::ClientBuilder {
    let _ = ring::default_provider().install_default();
    reqwest::blocking::Client::builder()
}

/// Common test prompts for consistency across tests
pub const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

/// The hashes of the ids [`word_mark_bpe`] with [`WORD_MARK_MERGES`] gives
/// [`TEST_PROMPTS`].
pub const EXPECTED_HASHES: [u64; 4] = [
    5768639699132919811,
    2514674412739152298,
    7348034889433778959,
    3308952177018835361,
];

const KIMI_K3_REPO: &str = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main";
const KIMI_K3_CACHE_DIR: &str = ".tokenizer_cache/kimi_k3";

/// A directory holding the Kimi-K3 tokenizer files (`tiktoken.model`,
/// `tokenizer_config.json`) plus a minimal `config.json` naming the K3
/// architecture so the tiktoken backend selects the XTML renderer.
///
/// `KIMI_K3_MODEL_DIR` points at a full checkpoint directory and skips the
/// download; otherwise the two tokenizer files are fetched once from the
/// public repository into `.tokenizer_cache/kimi_k3/`.
#[expect(clippy::expect_used, reason = "test helper — panics are intentional")]
#[expect(clippy::print_stdout, reason = "test diagnostic output")]
#[expect(clippy::panic, reason = "test helper — panics are intentional")]
pub fn ensure_kimi_k3_cached() -> PathBuf {
    if let Some(dir) = std::env::var_os("KIMI_K3_MODEL_DIR") {
        return PathBuf::from(dir);
    }

    let _guard = hub_fetch::download_lock();

    let cache_dir = PathBuf::from(KIMI_K3_CACHE_DIR);
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    for (file, min_bytes) in [
        ("tiktoken.model", 1_000_000),
        ("tokenizer_config.json", 500),
    ] {
        let path = cache_dir.join(file);
        if path.exists() {
            continue;
        }
        println!("Downloading Kimi-K3 {file} from HuggingFace...");
        let client = http_client();
        let answer = hub_fetch::fetch(&client, &format!("{KIMI_K3_REPO}/{file}"))
            .unwrap_or_else(|e| panic!("Failed to download {file}: {e}"));
        assert!(
            answer.status.is_success(),
            "Failed to download {file}: HTTP {}",
            answer.status
        );
        let content = answer.body;
        assert!(
            content.len() >= min_bytes,
            "Downloaded {file} too small: {} bytes",
            content.len()
        );
        // Rename into place so a failed write never leaves a short file that
        // the next run would trust.
        let part = cache_dir.join(format!("{file}.part"));
        fs::write(&part, content).expect("Failed to write to cache");
        fs::rename(&part, &path).expect("Failed to move download into place");
    }

    // Renderer detection reads `config.json::architectures`; the tokenizer
    // needs nothing else from the checkpoint's 500 KB config.
    let config_path = cache_dir.join("config.json");
    if !config_path.exists() {
        fs::write(
            &config_path,
            r#"{"architectures": ["KimiK3ForConditionalGeneration"], "model_type": "kimi_k3"}"#,
        )
        .expect("Failed to write config.json");
    }

    cache_dir
}

const DEEPSEEK_V41_REPO: &str =
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main";
const DEEPSEEK_V41_CACHE_DIR: &str = ".tokenizer_cache/deepseek_v41";

/// Downloads one tokenizer file of the model `label` from `base` into
/// `dir/file`, mirroring the Kimi-K3 helper's approach: write to a `.part`
/// file and rename into place, so a failed write never leaves a short file
/// that a later run would trust. Only a transport failure that no try got
/// past (no connection, TLS, a timeout before the answer; the GET is sent
/// again a few times first) returns `false` so the caller can skip when
/// offline; a reachable but wrong response (non-success status, short body,
/// a body broken at every try) or an unwritable cache panics with the reason,
/// so a moved or gated file fails loudly instead of silently disabling the
/// parity gate.
#[expect(
    clippy::print_stdout,
    clippy::panic,
    reason = "test helper — diagnostics and loud failures are intentional"
)]
fn download_tokenizer_file(
    label: &str,
    base: &str,
    dir: &Path,
    file: &str,
    min_bytes: usize,
) -> bool {
    println!("Downloading {label} {file} from HuggingFace...");
    let client = http_client();
    let url = format!("{base}/{file}");
    let answer = match hub_fetch::fetch(&client, &url) {
        Ok(answer) => answer,
        Err(error) if !hub_fetch::answered(&error) => {
            println!("{label} download skipped (transport error): {error}");
            return false;
        }
        Err(error) => panic!("{label} {file} download body error: {error}"),
    };
    let status = answer.status;
    assert!(
        status.is_success(),
        "{label} {file} download failed: HTTP {status} from {url}"
    );
    let content = answer.body;
    assert!(
        content.len() >= min_bytes,
        "{label} {file} download is {} bytes, expected at least {min_bytes}",
        content.len()
    );
    let part = dir.join(format!("{file}.part"));
    fs::write(&part, &content)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", part.display()));
    fs::rename(&part, dir.join(file))
        .unwrap_or_else(|error| panic!("cannot rename {}: {error}", part.display()));
    true
}

/// A directory holding the DeepSeek-V4.1-Flash tokenizer files
/// (`tokenizer.json`, `tokenizer_config.json`) plus a minimal `config.json`
/// naming the architecture so renderer detection selects DeepSeek-V4.1.
/// Without the checkpoint's `encoding/` directory the renderer uses the
/// rescaled effort table the checkpoint carries today.
///
/// `DEEPSEEK_V41_MODEL_DIR` points at a full checkpoint directory and skips
/// the download; otherwise the two tokenizer files are fetched once from the
/// public repository into `.tokenizer_cache/deepseek_v41/`. Returns `None`
/// only when offline (the repository is unreachable) and no override is set;
/// a local failure (unwritable cache directory or file) panics so the parity
/// gate cannot be skipped silently.
#[expect(clippy::panic, reason = "test helper — panics are intentional")]
pub fn ensure_deepseek_v41_cached() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("DEEPSEEK_V41_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }

    let _guard = hub_fetch::download_lock();

    let cache_dir = PathBuf::from(DEEPSEEK_V41_CACHE_DIR);
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .unwrap_or_else(|error| panic!("cannot create {}: {error}", cache_dir.display()));
    }

    for (file, min_bytes) in [
        ("tokenizer.json", 1_000_000),
        ("tokenizer_config.json", 500),
    ] {
        let path = cache_dir.join(file);
        if path.exists() {
            continue;
        }
        if !download_tokenizer_file(
            "DeepSeek-V4.1",
            DEEPSEEK_V41_REPO,
            &cache_dir,
            file,
            min_bytes,
        ) {
            return None;
        }
    }

    // Renderer detection reads `config.json::architectures`; the tokenizer
    // needs nothing else from the checkpoint, plus `image_token_id` for the
    // V4.1 image placeholder.
    let config_path = cache_dir.join("config.json");
    if !config_path.exists() {
        fs::write(
            &config_path,
            r#"{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","image_token_id":129264}"#,
        )
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", config_path.display()));
    }

    Some(cache_dir)
}

const DEEPSEEK_V4_REPO: &str =
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/main";
const DEEPSEEK_V4_CACHE_DIR: &str = ".tokenizer_cache/deepseek_v4";

/// A directory holding the DeepSeek-V4 tokenizer files (`tokenizer.json`,
/// `tokenizer_config.json`; identical across the V4 checkpoints) plus a
/// minimal `config.json` naming the architecture so renderer detection
/// selects DeepSeek-V4. Without the checkpoint's `encoding/` directory the
/// renderer uses the effort table the engine's own server renders.
///
/// `DEEPSEEK_V4_MODEL_DIR` points at a full checkpoint directory and skips
/// the download; otherwise the two tokenizer files are fetched once from the
/// public repository into `.tokenizer_cache/deepseek_v4/`. Returns `None`
/// only when offline (the repository is unreachable) and no override is set;
/// a local failure (unwritable cache directory or file) panics so the parity
/// gate cannot be skipped silently.
#[expect(clippy::panic, reason = "test helper — panics are intentional")]
pub fn ensure_deepseek_v4_cached() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("DEEPSEEK_V4_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }

    let _guard = hub_fetch::download_lock();

    let cache_dir = PathBuf::from(DEEPSEEK_V4_CACHE_DIR);
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .unwrap_or_else(|error| panic!("cannot create {}: {error}", cache_dir.display()));
    }

    for (file, min_bytes) in [
        ("tokenizer.json", 1_000_000),
        ("tokenizer_config.json", 500),
    ] {
        let path = cache_dir.join(file);
        if path.exists() {
            continue;
        }
        if !download_tokenizer_file("DeepSeek-V4", DEEPSEEK_V4_REPO, &cache_dir, file, min_bytes) {
            return None;
        }
    }

    let config_path = cache_dir.join("config.json");
    if !config_path.exists() {
        fs::write(
            &config_path,
            r#"{"architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4"}"#,
        )
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", config_path.display()));
    }

    Some(cache_dir)
}

const MINIMAX_M3_REPO: &str =
    "https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/f0e1c1e04d40177e4673a22097036854f536e9c0";
const MINIMAX_M3_CACHE_DIR: &str = ".tokenizer_cache/minimax_m3";

/// A directory holding the MiniMax-M3 tokenizer files (`tokenizer.json`,
/// `tokenizer_config.json`) at the revision the render fixtures were recorded
/// from.
///
/// `MINIMAX_M3_MODEL_DIR` points at a full checkpoint directory and skips the
/// download; otherwise the two files are fetched once from the public
/// repository into `.tokenizer_cache/minimax_m3/`. Returns `None` only when
/// offline (the repository is unreachable) and no override is set; a local
/// failure (unwritable cache directory or file) panics so the parity gate
/// cannot be skipped silently.
#[expect(clippy::panic, reason = "test helper — panics are intentional")]
pub fn ensure_minimax_m3_cached() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MINIMAX_M3_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }

    let _guard = hub_fetch::download_lock();

    let cache_dir = PathBuf::from(MINIMAX_M3_CACHE_DIR);
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .unwrap_or_else(|error| panic!("cannot create {}: {error}", cache_dir.display()));
    }

    for (file, min_bytes) in [
        ("tokenizer.json", 9_000_000),
        ("tokenizer_config.json", 500),
    ] {
        let path = cache_dir.join(file);
        if path.exists() {
            continue;
        }
        if !download_tokenizer_file("MiniMax-M3", MINIMAX_M3_REPO, &cache_dir, file, min_bytes) {
            return None;
        }
    }

    Some(cache_dir)
}

/// The 256 characters a byte-level BPE spells its bytes with: the printable
/// bytes as themselves, the others from U+0100 on.
#[expect(
    clippy::expect_used,
    reason = "every byte maps to a scalar value; a helper of tests that stop on a broken setup"
)]
pub fn byte_level_alphabet() -> Vec<char> {
    let mut next = 0x100;
    (0u32..256)
        .map(|byte| {
            let printable = (0x21..=0x7E).contains(&byte)
                || (0xA1..=0xAC).contains(&byte)
                || (0xAE..=0xFF).contains(&byte);
            let code = if printable {
                byte
            } else {
                next += 1;
                next - 1
            };
            char::from_u32(code).expect("a character of the byte-level alphabet")
        })
        .collect()
}

/// Seven merges of a small byte-level BPE: `the` and ` the` (the space as
/// the byte-level `Ġ`), and `llo`.
pub const FEW_MERGES: [(&str, &str); 7] = [
    ("t", "h"),
    ("th", "e"),
    ("Ġ", "t"),
    ("Ġt", "h"),
    ("Ġth", "e"),
    ("l", "l"),
    ("ll", "o"),
];

/// A byte-level BPE of the common shape: the byte-level alphabet as the
/// vocabulary, `merges` applied in order (each merged pair added to the
/// vocabulary), the byte-level pre-tokenizer without a prefix space, the
/// byte-level decoder and post-processor (which adds no token), and
/// `special_tokens` in the added vocabulary.
#[expect(
    clippy::expect_used,
    reason = "a helper of tests that stop on a broken setup"
)]
pub fn byte_level_bpe(merges: &[(&str, &str)], special_tokens: &[&str]) -> Tokenizer {
    let mut vocab = Vocab::default();
    for (id, letter) in byte_level_alphabet().into_iter().enumerate() {
        vocab.insert(letter.to_string(), id as u32);
    }
    let mut merged = Merges::new();
    for (a, b) in merges {
        let next = vocab.len() as u32;
        vocab.insert(format!("{a}{b}"), next);
        merged.push((a.to_string(), b.to_string()));
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, merged)
        .build()
        .expect("a byte-level BPE");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        ByteLevelPreTokenizer::default().add_prefix_space(false),
    ));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    tokenizer.with_post_processor(Some(ByteLevelProcessor::default()));
    tokenizer
        .add_special_tokens(
            special_tokens
                .iter()
                .map(|token| AddedToken::from(*token, true)),
        )
        .expect("the special tokens");
    tokenizer
}

/// Twelve merges of a small word-mark BPE, in rank order: `▁dee`, `ing`,
/// `▁lea`, `▁is`, `an` and `▁an` with the pairs they are built from.
pub const WORD_MARK_MERGES: [(&str, &str); 12] = [
    ("▁", "d"),
    ("e", "e"),
    ("▁d", "ee"),
    ("i", "n"),
    ("in", "g"),
    ("▁", "l"),
    ("e", "a"),
    ("▁l", "ea"),
    ("▁", "i"),
    ("▁i", "s"),
    ("a", "n"),
    ("▁", "an"),
];

/// A small BPE in the style of the Llama family's `tokenizer.json`, section
/// for section: the normalizer writes the word mark `▁` in front of the text
/// and in place of every space, there is no pre-tokenizer, a character
/// outside the vocabulary is spelled by its UTF-8 bytes (byte fallback), an
/// encoding that asks for special tokens starts with `<s>`, and the decoder
/// turns `▁` back into a space, joins the byte tokens into characters and
/// strips the first space. The vocabulary is `<unk>`, `<s>`, `</s>`, the 256
/// byte tokens, `▁`, the lower-case letters, and `merges` applied in order.
#[expect(
    clippy::expect_used,
    reason = "a helper of tests that stop on a broken setup"
)]
pub fn word_mark_bpe(merges: &[(&str, &str)]) -> Tokenizer {
    let mut vocab = Map::new();
    let mut add = |piece: String| {
        let id = vocab.len();
        assert!(
            !vocab.contains_key(&piece),
            "{piece:?} is in the vocabulary twice"
        );
        vocab.insert(piece, Value::from(id));
    };
    for special in ["<unk>", "<s>", "</s>"] {
        add(special.to_string());
    }
    for byte in 0u32..256 {
        add(format!("<0x{byte:02X}>"));
    }
    add("▁".to_string());
    for letter in 'a'..='z' {
        add(letter.to_string());
    }
    for (a, b) in merges {
        add(format!("{a}{b}"));
    }
    let special = |id: u32, content: &str| {
        json!({
            "id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true,
        })
    };
    let file = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [special(0, "<unk>"), special(1, "<s>"), special(2, "</s>")],
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "▁"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        ]},
        "pre_tokenizer": null,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}},
        },
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ]},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "vocab": vocab,
            "merges": merges.iter().map(|(a, b)| format!("{a} {b}")).collect::<Vec<_>>(),
        },
    });
    Tokenizer::from_str(&file.to_string()).expect("a word-mark BPE")
}

/// A minimal tokenizer.json that loads cleanly: a one-word BPE vocabulary. The only
/// requirement is that it parses; the renderers under test do not call back into it.
pub const MIN_TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "BPE", "vocab": { "hello": 0 }, "merges": [] }
}"#;

/// [`MIN_TOKENIZER_JSON`] with `<s>` and `</s>` in its vocabulary, for the renderers
/// that look the sentinels up.
pub const MIN_TOKENIZER_JSON_WITH_SENTINELS: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "BPE",
        "vocab": { "hello": 0, "<s>": 1, "</s>": 2 },
        "merges": []
    }
}"#;
