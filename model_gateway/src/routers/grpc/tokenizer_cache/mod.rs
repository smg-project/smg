//! The tokenizer files the gRPC router's tests pin: public checkpoints fetched
//! once through [`hub_fetch`] into the package's `.tokenizer_cache/`, or read
//! from the directory an environment variable names. Compiled for the crate's
//! test targets only; the test harness re-exports the fetcher.

/// A test binary's downloads from the model hub: the file the tokenizer crate's
/// tests carry too, byte for byte (a parity test checks).
pub mod hub_fetch;

use std::{fs, path::PathBuf};

/// Names a directory that holds the DeepSeek-V4.1 tokenizer files; set,
/// nothing is fetched.
pub const DEEPSEEK_V41_MODEL_DIR_ENV: &str = "DEEPSEEK_V41_MODEL_DIR";
/// Names a directory that holds the Kimi-K3 tokenizer files; set, nothing is
/// fetched.
pub const KIMI_K3_TOKENIZER_DIR_ENV: &str = "SMG_KIMI_K3_TOKENIZER_DIR";

/// The tokenizer files of a public checkpoint, fetched once into the
/// package's `.tokenizer_cache/` for the tests that render through the real
/// tokenizer, as the tokenizer crate's parity tests fetch theirs: the cases
/// run wherever the hub is reachable instead of passing without running.
struct CheckpointTokenizer {
    /// The checkpoint, as the log names it.
    label: &'static str,
    /// The variable naming a directory that already holds the files.
    override_env: &'static str,
    /// The public repository the files are fetched from.
    repo: &'static str,
    /// Where the files are cached, under the package directory the tests
    /// run in.
    cache_dir: &'static str,
    /// The files fetched, each with the least size a whole one has.
    files: &'static [(&'static str, usize)],
    /// The `config.json` written beside them: renderer detection reads the
    /// architecture from it, the multimodal registry the image token.
    config_json: &'static str,
}

const DEEPSEEK_V41: CheckpointTokenizer = CheckpointTokenizer {
    label: "DeepSeek-V4.1",
    override_env: DEEPSEEK_V41_MODEL_DIR_ENV,
    repo: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main",
    cache_dir: ".tokenizer_cache/deepseek_v41",
    files: &[
        ("tokenizer.json", 1_000_000),
        ("tokenizer_config.json", 500),
    ],
    config_json: r#"{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","image_token_id":129264}"#,
};

const KIMI_K3: CheckpointTokenizer = CheckpointTokenizer {
    label: "Kimi-K3",
    override_env: KIMI_K3_TOKENIZER_DIR_ENV,
    repo: "https://huggingface.co/moonshotai/Kimi-K3/resolve/main",
    cache_dir: ".tokenizer_cache/kimi_k3",
    files: &[
        ("tiktoken.model", 1_000_000),
        ("tokenizer_config.json", 500),
    ],
    config_json: r#"{"architectures":["KimiK3ForConditionalGeneration"],"model_type":"kimi_k3"}"#,
};

/// The DeepSeek-V4.1 tokenizer directory: `DEEPSEEK_V41_MODEL_DIR`, else
/// `tokenizer.json` and `tokenizer_config.json` fetched once from the public
/// checkpoint, with a `config.json` naming the architecture and the image
/// token. `None` only when the hub never answered and no override is set; a
/// hub that answers wrongly or a cache that cannot be written panics, so the
/// cases cannot be skipped silently.
pub fn ensure_deepseek_v41_cached() -> Option<PathBuf> {
    DEEPSEEK_V41.ensure_cached()
}

/// The Kimi-K3 tokenizer directory: `SMG_KIMI_K3_TOKENIZER_DIR`, else
/// `tiktoken.model` and `tokenizer_config.json` fetched once from the public
/// checkpoint, with a `config.json` naming the architecture. `None` only
/// when the hub never answered and no override is set; a hub that answers
/// wrongly or a cache that cannot be written panics, so the case cannot be
/// skipped silently.
pub fn ensure_kimi_k3_cached() -> Option<PathBuf> {
    KIMI_K3.ensure_cached()
}

impl CheckpointTokenizer {
    /// The directory holding the files: the override as given, else the
    /// cache, each missing file fetched under the download lock and written
    /// whole or not at all. `None` only when the hub never answered a GET (no
    /// connection, a timeout, the request cut short at every try) and no
    /// override is set, so a box offline skips the cases; a hub that answers
    /// wrongly (a non-success status, a short body, a body broken at every
    /// try) or a cache that cannot be written panics with the reason.
    #[expect(
        clippy::panic,
        clippy::print_stdout,
        reason = "a test helper: a download that goes wrong fails loudly, a skipped one is said"
    )]
    fn ensure_cached(&self) -> Option<PathBuf> {
        if let Some(dir) = std::env::var_os(self.override_env) {
            return Some(PathBuf::from(dir));
        }
        let _guard = hub_fetch::download_lock();
        let dir = PathBuf::from(self.cache_dir);
        fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("cannot create {}: {error}", dir.display()));
        let client = crate::server::blocking_http_client();
        for &(file, min_bytes) in self.files {
            let path = dir.join(file);
            if path.is_file() {
                continue;
            }
            let url = format!("{}/{file}", self.repo);
            println!("Downloading {} {file} from {url}...", self.label);
            let answer = match hub_fetch::fetch(&client, &url) {
                Ok(answer) => answer,
                Err(error) if !hub_fetch::answered(&error) => {
                    println!(
                        "{} {file}: the hub did not answer ({error}); nothing cached",
                        self.label
                    );
                    return None;
                }
                Err(error) => panic!("GET {url}: {error}"),
            };
            assert!(
                answer.status.is_success(),
                "GET {url}: HTTP {}",
                answer.status
            );
            assert!(
                answer.body.len() >= min_bytes,
                "GET {url}: {} bytes, expected at least {min_bytes}",
                answer.body.len()
            );
            // Named for this process, as `hub_fetch::cached_download` names its
            // part file: a second process fetching into the same directory
            // renames its own whole file into place.
            let mut part = path.clone().into_os_string();
            part.push(format!(".{}.part", std::process::id()));
            let part = PathBuf::from(part);
            fs::write(&part, &answer.body)
                .unwrap_or_else(|error| panic!("cannot write {}: {error}", part.display()));
            fs::rename(&part, &path).unwrap_or_else(|error| {
                panic!("cannot rename {} into place: {error}", part.display())
            });
            println!("Cached {} ({} bytes)", path.display(), answer.body.len());
        }
        let config = dir.join("config.json");
        if !config.is_file() {
            fs::write(&config, self.config_json)
                .unwrap_or_else(|error| panic!("cannot write {}: {error}", config.display()));
        }
        Some(dir)
    }
}
