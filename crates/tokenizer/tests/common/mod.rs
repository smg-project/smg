//! Common test utilities for tokenizer tests
#![allow(dead_code)] // each integration test binary uses a subset of these helpers

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

// Tokenizer download configuration
const TINYLLAMA_TOKENIZER_URL: &str =
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json";
const CACHE_DIR: &str = ".tokenizer_cache";
const TINYLLAMA_TOKENIZER_FILENAME: &str = "tinyllama_tokenizer.json";

// Global mutex to prevent concurrent downloads
static DOWNLOAD_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

/// Downloads the TinyLlama tokenizer from HuggingFace if not already cached.
/// Returns the path to the cached tokenizer file.
///
/// This function is thread-safe and will only download the tokenizer once
/// even if called from multiple threads concurrently.
#[expect(clippy::unwrap_used, reason = "test helper — panics are intentional")]
#[expect(clippy::expect_used, reason = "test helper — panics are intentional")]
#[expect(clippy::print_stdout, reason = "test diagnostic output")]
pub fn ensure_tokenizer_cached() -> PathBuf {
    // Get or initialize the mutex
    let mutex = DOWNLOAD_MUTEX.get_or_init(|| Mutex::new(()));

    // Lock to ensure only one thread downloads at a time
    let _guard = mutex.lock().unwrap();

    let cache_dir = PathBuf::from(CACHE_DIR);
    let tokenizer_path = cache_dir.join(TINYLLAMA_TOKENIZER_FILENAME);

    // Create cache directory if it doesn't exist
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    // Download tokenizer if not already cached
    if !tokenizer_path.exists() {
        println!("Downloading TinyLlama tokenizer from HuggingFace...");

        // Use blocking reqwest client since we're in tests/benchmarks
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(TINYLLAMA_TOKENIZER_URL)
            .send()
            .expect("Failed to download tokenizer");

        assert!(
            response.status().is_success(),
            "Failed to download tokenizer: HTTP {}",
            response.status()
        );

        let content = response.bytes().expect("Failed to read tokenizer content");

        assert!(
            content.len() >= 100,
            "Downloaded content too small: {} bytes",
            content.len()
        );

        fs::write(&tokenizer_path, content).expect("Failed to write tokenizer to cache");
        println!(
            "Tokenizer downloaded and cached successfully ({} bytes)",
            tokenizer_path.metadata().unwrap().len()
        );
    }

    tokenizer_path
}

/// Common test prompts for consistency across tests
pub const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

/// Pre-computed hashes for verification
pub const EXPECTED_HASHES: [u64; 4] = [
    1209591529327510910,
    4181375434596349981,
    6245658446118930933,
    5097285695902185237,
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
#[expect(clippy::unwrap_used, reason = "test helper — panics are intentional")]
#[expect(clippy::expect_used, reason = "test helper — panics are intentional")]
#[expect(clippy::print_stdout, reason = "test diagnostic output")]
#[expect(clippy::panic, reason = "test helper — panics are intentional")]
pub fn ensure_kimi_k3_cached() -> PathBuf {
    if let Some(dir) = std::env::var_os("KIMI_K3_MODEL_DIR") {
        return PathBuf::from(dir);
    }

    let mutex = DOWNLOAD_MUTEX.get_or_init(|| Mutex::new(()));
    let _guard = mutex.lock().unwrap();

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
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(format!("{KIMI_K3_REPO}/{file}"))
            .send()
            .unwrap_or_else(|e| panic!("Failed to download {file}: {e}"));
        assert!(
            response.status().is_success(),
            "Failed to download {file}: HTTP {}",
            response.status()
        );
        let content = response.bytes().expect("Failed to read download");
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

/// Downloads one DeepSeek-V4.1 tokenizer file from `base` into `dir/file`,
/// mirroring the Kimi-K3 helper's approach: write to a `.part` file and
/// rename into place, so a failed write never leaves a short file that a
/// later run would trust. Only a transport failure (no connection, TLS,
/// timeout) returns `false` so the caller can skip when offline; a reachable
/// but wrong response (non-success status, short body) or an unwritable
/// cache panics with the reason, so a moved or gated file fails loudly
/// instead of silently disabling the parity gate.
#[expect(
    clippy::print_stdout,
    clippy::panic,
    reason = "test helper — diagnostics and loud failures are intentional"
)]
fn download_deepseek_v41_file(base: &str, dir: &Path, file: &str, min_bytes: usize) -> bool {
    println!("Downloading DeepSeek-V4.1 {file} from HuggingFace...");
    let client = reqwest::blocking::Client::new();
    let url = format!("{base}/{file}");
    let response = match client.get(&url).send() {
        Ok(response) => response,
        Err(error) => {
            println!("DeepSeek-V4.1 download skipped (transport error): {error}");
            return false;
        }
    };
    let status = response.status();
    assert!(
        status.is_success(),
        "DeepSeek-V4.1 {file} download failed: HTTP {status} from {url}"
    );
    let content = response
        .bytes()
        .unwrap_or_else(|error| panic!("DeepSeek-V4.1 {file} download body error: {error}"));
    assert!(
        content.len() >= min_bytes,
        "DeepSeek-V4.1 {file} download is {} bytes, expected at least {min_bytes}",
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
///
/// `DEEPSEEK_V41_MODEL_DIR` points at a full checkpoint directory and skips
/// the download; otherwise the two tokenizer files are fetched once from the
/// public repository into `.tokenizer_cache/deepseek_v41/`. Returns `None`
/// only when offline (the repository is unreachable) and no override is set;
/// a local failure (unwritable cache directory or file) panics so the parity
/// gate cannot be skipped silently.
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helper — panics are intentional"
)]
pub fn ensure_deepseek_v41_cached() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("DEEPSEEK_V41_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }

    let mutex = DOWNLOAD_MUTEX.get_or_init(|| Mutex::new(()));
    let _guard = mutex.lock().unwrap();

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
        if !download_deepseek_v41_file(DEEPSEEK_V41_REPO, &cache_dir, file, min_bytes) {
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
