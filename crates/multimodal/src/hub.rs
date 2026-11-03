//! HuggingFace Hub integration for downloading model config files.
//!
//! When the model source is a HuggingFace model ID (e.g. `Qwen/Qwen3-VL-8B-Instruct`)
//! rather than a local directory, this module downloads `config.json` and
//! `preprocessor_config.json` / video processor config files to the local HF
//! cache and returns the resolved path.

use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::api::tokio::{ApiBuilder, ApiError};

const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Resolve a model source to a local directory containing `config.json`.
///
/// If the source is already a local directory with `config.json`, returns it as-is.
/// Otherwise, treats the source as a HuggingFace model ID and downloads
/// `config.json` plus optional processor configs to the local HF cache.
pub async fn resolve_model_config_dir(source: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(source);
    if path.join("config.json").exists() {
        return Ok(path.to_path_buf());
    }

    let mut builder = ApiBuilder::from_env().with_progress(false);
    if let Ok(token) = std::env::var(HF_TOKEN_ENV) {
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }
    }
    let api = builder
        .build()
        .context("Failed to build HuggingFace API client")?;
    let repo = api.model(source.to_string());

    let config_path = repo
        .get("config.json")
        .await
        .with_context(|| format!("Failed to download config.json for model '{source}'"))?;

    // The processor configs beside config.json, best-effort: a checkpoint
    // ships the ones of its modalities, and the processor takes a missing file
    // as absent. Any other failure is logged, so a processor loaded as if the
    // file were absent can be traced to the fetch that failed.
    for config_file in [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
    ] {
        if let Err(error) = repo.get(config_file).await {
            if !is_absent(&error) {
                tracing::warn!(
                    model = source,
                    file = config_file,
                    error = %describe(&error),
                    "processor config not fetched; the processor loads as if the file were absent"
                );
            }
        }
    }

    config_path
        .parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| anyhow::anyhow!("Invalid HF cache path for model '{source}'"))
}

/// `true` when the Hub answered that the file is not in the repository (HTTP
/// 404): the expected answer of a checkpoint without that processor config.
/// Every other failure (no answer, a 5xx, a proxy error, a denied token) is a
/// fetch that should have succeeded.
fn is_absent(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::RequestError(request)
            if request.status().is_some_and(|status| status.as_u16() == 404)
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

#[cfg(test)]
mod tests {
    use std::io;

    use hf_hub::api::tokio::ApiError;

    use super::{describe, is_absent};

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
}
