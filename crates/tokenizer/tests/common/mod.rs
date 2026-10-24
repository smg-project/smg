//! Common test utilities for tokenizer tests
#![allow(dead_code)] // each integration test binary uses a subset of these helpers

pub mod hub_fetch;

use std::{
    fs,
    path::{Path, PathBuf},
};

use rustls::crypto::ring;

// Tokenizer download configuration
const TINYLLAMA_TOKENIZER_URL: &str =
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json";
/// Names another source of the TinyLlama tokenizer file: a mirror, or a stub
/// standing in for the hub.
const TINYLLAMA_TOKENIZER_URL_ENV: &str = "TINYLLAMA_TOKENIZER_URL";
const CACHE_DIR: &str = ".tokenizer_cache";
const TINYLLAMA_TOKENIZER_FILENAME: &str = "tinyllama_tokenizer.json";

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

/// The TinyLlama tokenizer file, downloaded into the cache once per process
/// unless it is there already ([`hub_fetch::cached_download`]: one download
/// at a time, a rate-limited hub waited out, a lock a failed download left
/// poisoned taken back). Returns the path to the cached tokenizer file.
pub fn ensure_tokenizer_cached() -> PathBuf {
    let url = std::env::var(TINYLLAMA_TOKENIZER_URL_ENV)
        .unwrap_or_else(|_| TINYLLAMA_TOKENIZER_URL.to_string());
    let path = PathBuf::from(CACHE_DIR).join(TINYLLAMA_TOKENIZER_FILENAME);
    hub_fetch::cached_download(&http_client(), &url, &path, 100);
    path
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
