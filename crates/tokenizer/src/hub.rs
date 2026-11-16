use std::path::{Path, PathBuf};

use hf_hub::api::tokio::{Api, ApiBuilder, ApiError};

/// Environment variable for HuggingFace token
/// Note: The hf-hub crate's from_env() doesn't read HF_TOKEN directly,
/// it only reads from the token file. We explicitly read this env var
/// to support CI environments where the token is set as an environment variable.
const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Build an ApiBuilder with token from HF_TOKEN environment variable if set
fn build_api() -> anyhow::Result<Api> {
    let mut builder = ApiBuilder::from_env().with_progress(true);

    // Only override token if HF_TOKEN env var is set and non-empty
    if let Ok(token) = std::env::var(HF_TOKEN_ENV) {
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }
    }

    Ok(builder.build()?)
}

const IGNORED: [&str; 5] = [
    ".gitattributes",
    "LICENSE",
    "LICENSE.txt",
    "README.md",
    "USE_POLICY.md",
];

/// Checks if a file is a model weight file
fn is_weight_file(filename: &str) -> bool {
    filename.ends_with(".bin")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".h5")
        || filename.ends_with(".msgpack")
        || filename.ends_with(".ckpt.index")
}

/// Checks if a file is an image file
fn is_image(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg")
}

/// Checks if a file is a tokenizer file. `pub` so that the Python binding
/// fingerprints a checkpoint by the files this download fetches.
pub fn is_tokenizer_file(filename: &str) -> bool {
    filename.ends_with("tokenizer.json")
        || filename.ends_with("tokenizer_config.json")
        || filename.ends_with("special_tokens_map.json")
        || filename.ends_with("vocab.json")
        || filename.ends_with("merges.txt")
        || filename.ends_with(".model")  // SentencePiece models
        || filename.ends_with(".tiktoken")
        || filename.ends_with("tekken.json") // the tekken vocabulary file
        || is_chat_template_file(filename) // Include chat template files
}

/// Checks if a file is a chat template file
fn is_chat_template_file(filename: &str) -> bool {
    filename.ends_with(".jinja")  // Direct Jinja files
        || filename == "chat_template.json" // JSON file containing Jinja template
}

/// Attempt to download tokenizer files from Hugging Face
/// Returns the directory containing the downloaded tokenizer files
pub async fn download_tokenizer_from_hf(model_id: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
    let model_id = model_id.as_ref();
    let api = build_api()?;
    let model_name = model_id.display().to_string();

    let repo = api.model(model_name.clone());

    let info = match repo.info().await {
        Ok(info) => info,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to fetch model '{model_name}' from HuggingFace: {e}. Is this a valid HuggingFace ID?"
            ));
        }
    };

    if info.siblings.is_empty() {
        return Err(anyhow::anyhow!(
            "Model '{model_name}' exists but contains no downloadable files."
        ));
    }

    let mut cache_dir = None;
    let mut tokenizer_files_found = false;

    // First, identify all tokenizer files to download
    let tokenizer_files: Vec<_> = info
        .siblings
        .iter()
        .filter(|sib| {
            !IGNORED.contains(&sib.rfilename.as_str())
                && !is_image(&sib.rfilename)
                && !is_weight_file(&sib.rfilename)
                && is_tokenizer_file(&sib.rfilename)
        })
        .collect();

    if tokenizer_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No tokenizer files found for model '{model_name}'."
        ));
    }

    // Download all tokenizer files
    for sib in tokenizer_files {
        match repo.get(&sib.rfilename).await {
            Ok(path) => {
                if cache_dir.is_none() {
                    cache_dir = path.parent().map(|p| p.to_path_buf());
                }
                tokenizer_files_found = true;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to download tokenizer file '{}' from model '{}': {}",
                    sib.rfilename,
                    model_name,
                    e
                ));
            }
        }
    }

    if !tokenizer_files_found {
        return Err(anyhow::anyhow!(
            "No tokenizer files could be downloaded for model '{model_name}'."
        ));
    }

    // The config files the loaders read beside the tokenizer (the model type
    // of the pipeline rule, the EOS ids), best-effort: a repository shipped
    // as its tokenizer files alone has neither, and the loaders take a missing
    // file as absent. Any other failure is logged, so a tokenizer loaded as if
    // the file were absent can be traced to the fetch that failed.
    // Downloaded separately by name to avoid matching nested files like
    // 1_Pooling/config.json that would break cache_dir resolution.
    for config_file in ["config.json", "generation_config.json"] {
        if let Err(error) = repo.get(config_file).await {
            if !is_absent(&error) {
                tracing::warn!(
                    model = %model_name,
                    file = config_file,
                    error = %describe(&error),
                    "config file not fetched; the tokenizer loads as if the file were absent"
                );
            }
        }
    }

    match cache_dir {
        Some(dir) => {
            // Ensure we return the correct model directory, not a subfolder
            // Some models have an "original" subfolder for PyTorch weights
            // We want the main model directory that contains tokenizer files
            let final_dir = resolve_model_cache_dir(&dir, &model_name);
            Ok(final_dir)
        }
        None => Err(anyhow::anyhow!(
            "Invalid HF cache path for model '{model_name}'"
        )),
    }
}

/// `true` when the Hub answered that the file is not in the repository (HTTP
/// 404): the expected answer of a repository shipped as its tokenizer files
/// alone. Every other failure (no answer, a 5xx, a proxy error, a denied
/// token) is a fetch that should have succeeded.
fn is_absent(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::RequestError(request) if request.status().is_some_and(|status| status.as_u16() == 404)
    )
}

/// The error's own message followed by the causes below it that the message
/// does not already end with: each `ApiError` variant embeds its source's
/// message in its own (`request error: {0}`), as `io::Error` does for the
/// error it wraps, so a plain walk of the chain would write those twice while
/// the deeper causes (the connect error, the OS error) are what the line is
/// for.
fn describe(error: &ApiError) -> String {
    let mut message = error.to_string();
    let mut cause = std::error::Error::source(error);
    while let Some(next) = cause {
        let text = next.to_string();
        if !message.ends_with(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        cause = next.source();
    }
    message
}

/// One file of a Hub repository: served from the local HF cache when it is
/// there (no network), fetched into it otherwise. Returns the file's path.
pub async fn fetch_file(model_id: &str, filename: &str) -> anyhow::Result<PathBuf> {
    let api = build_api()?;
    let path = api.model(model_id.to_string()).get(filename).await?;
    Ok(path)
}

/// Resolve the correct model cache directory
/// Handles cases where files might be in subfolders (e.g., "original" folder)
fn resolve_model_cache_dir(path: &Path, model_name: &str) -> PathBuf {
    // Check if we're in a subfolder like "original"
    if let Some(parent) = path.parent() {
        if let Some(folder_name) = path.file_name() {
            if folder_name == "original" {
                // We're in the "original" subfolder, go up one level
                return parent.to_path_buf();
            }
        }
    }

    // Check if the current path contains the model name components
    // This helps ensure we're at the right directory level
    let model_parts: Vec<&str> = model_name.split('/').collect();
    if model_parts.len() >= 2 {
        let expected_pattern = format!(
            "models--{}--{}",
            model_parts[0].replace("-", "--"),
            model_parts[1].replace("-", "--")
        );

        if path.to_string_lossy().contains(&expected_pattern) {
            // We're already at the correct level
            return path.to_path_buf();
        }

        let mut current = path.to_path_buf();

        // First check if current path already contains tokenizer files
        if current.join("tokenizer.json").exists() || current.join("tokenizer_config.json").exists()
        {
            return current;
        }

        // If not, traverse up to find the model root, then look in snapshots
        while let Some(parent) = current.parent() {
            if parent.to_string_lossy().contains(&expected_pattern) {
                let snapshots_dir = parent.join("snapshots");
                if snapshots_dir.exists() && snapshots_dir.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snapshot_path = entry.path();
                            if snapshot_path.is_dir()
                                && (snapshot_path.join("tokenizer.json").exists()
                                    || snapshot_path.join("tokenizer_config.json").exists())
                            {
                                return snapshot_path;
                            }
                        }
                    }
                }
                return parent.to_path_buf();
            }
            current = parent.to_path_buf();
        }
    }

    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use std::io;

    use hf_hub::api::tokio::ApiError;

    use super::{describe, is_absent, is_chat_template_file, is_tokenizer_file, is_weight_file};

    /// The error `repo.get` returns when the Hub answers `status` to the
    /// file's request (hf-hub runs `error_for_status` on the answer).
    fn hub_answer(status: u16) -> ApiError {
        let answer = http::Response::builder()
            .status(status)
            .body("")
            .expect("a valid status code");
        let error = reqwest_hub::Response::from(answer)
            .error_for_status()
            .expect_err("a 4xx or 5xx answer is an error");
        ApiError::RequestError(error)
    }

    #[test]
    fn a_404_answer_is_an_absent_file() {
        assert!(is_absent(&hub_answer(404)));
    }

    #[test]
    fn every_other_failure_is_a_fetch_that_failed() {
        for status in [401, 403, 429, 500, 502, 503, 504] {
            assert!(!is_absent(&hub_answer(status)), "status {status}");
        }
        assert!(!is_absent(&ApiError::IoError(io::Error::other(
            "connection reset"
        ))));
        assert!(!is_absent(&ApiError::TooManyRetries(Box::new(hub_answer(
            503
        )))));
    }

    /// An error whose message is its own and whose cause is the next one
    /// down: the shape of a connect failure's chain, without a network.
    #[derive(Debug)]
    struct Layered(&'static str, Option<Box<Layered>>);

    impl std::fmt::Display for Layered {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for Layered {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.1
                .as_deref()
                .map(|cause| cause as &(dyn std::error::Error + 'static))
        }
    }

    /// The logged description writes each message of the chain once: hf-hub's
    /// variant embeds its source's message, `io::Error` embeds the error it
    /// wraps, and a connect error embeds the OS error, so the causes already
    /// at the end of the message are not written again while every deeper
    /// cause is.
    #[test]
    fn the_description_writes_each_cause_once() {
        let status = describe(&hub_answer(503));
        assert!(status.starts_with("request error: "), "{status}");
        assert_eq!(
            status.matches("HTTP status server error (503").count(),
            1,
            "{status}"
        );

        let retries = describe(&ApiError::TooManyRetries(Box::new(hub_answer(503))));
        assert!(
            retries.starts_with("Too many retries: request error: "),
            "{retries}"
        );
        assert_eq!(retries.matches("503").count(), 1, "{retries}");

        let os = Layered("Connection refused (os error 111)", None);
        let connect = Layered(
            "tcp connect error: Connection refused (os error 111)",
            Some(Box::new(os)),
        );
        let client = Layered("client error (Connect)", Some(Box::new(connect)));
        let io = ApiError::IoError(io::Error::other(client));
        assert_eq!(
            describe(&io),
            "I/O error client error (Connect): tcp connect error: Connection refused (os error 111)"
        );
    }

    #[test]
    fn test_is_tokenizer_file() {
        assert!(is_tokenizer_file("tokenizer.json"));
        assert!(is_tokenizer_file("tokenizer_config.json"));
        assert!(is_tokenizer_file("special_tokens_map.json"));
        assert!(is_tokenizer_file("vocab.json"));
        assert!(is_tokenizer_file("merges.txt"));
        assert!(is_tokenizer_file("tekken.json"));
        assert!(is_tokenizer_file("spiece.model"));
        assert!(is_tokenizer_file("chat_template.jinja"));
        assert!(is_tokenizer_file("template.jinja"));
        assert!(!is_tokenizer_file("model.bin"));
        assert!(!is_tokenizer_file("README.md"));
    }

    #[test]
    fn test_is_chat_template_file() {
        assert!(is_chat_template_file("chat_template.jinja"));
        assert!(is_chat_template_file("template.jinja"));
        assert!(is_chat_template_file("any_file.jinja"));
        assert!(is_chat_template_file("chat_template.json"));
        assert!(!is_chat_template_file("tokenizer.json"));
        assert!(!is_chat_template_file("other_file.json"));
        assert!(!is_chat_template_file("chat_template"));
        assert!(!is_chat_template_file("README.md"));
    }

    #[test]
    fn test_is_weight_file() {
        assert!(is_weight_file("model.bin"));
        assert!(is_weight_file("model.safetensors"));
        assert!(is_weight_file("pytorch_model.bin"));
        assert!(!is_weight_file("tokenizer.json"));
        assert!(!is_weight_file("config.json"));
    }
}
