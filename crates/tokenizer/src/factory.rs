use std::{fs::File, io::Read, path::Path, sync::Arc};

use anyhow::{Error, Result};
use tracing::debug;

use crate::{
    hub::download_tokenizer_from_hf,
    huggingface::HuggingFaceTokenizer,
    tiktoken::{has_tiktoken_file, is_tiktoken_file, TiktokenTokenizer},
    traits,
};

/// Represents the type of tokenizer being used
#[derive(Debug, Clone)]
pub enum TokenizerType {
    HuggingFace(String),
    Mock,
    Tiktoken(String),
    // Future: SentencePiece, GGUF
}

/// Create a tokenizer from a file path to a tokenizer file.
/// The file extension is used to determine the tokenizer type.
/// Supported file types are:
/// - json: HuggingFace tokenizer
/// - For testing: can return mock tokenizer
pub fn create_tokenizer_from_file(file_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    create_tokenizer_with_chat_template(file_path, None)
}

/// Create a tokenizer from a file path with an optional chat template
pub fn create_tokenizer_with_chat_template(
    file_path: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Special case for testing
    if file_path == "mock" || file_path == "test" {
        return Ok(Arc::new(super::mock::MockTokenizer::new()));
    }

    let path = Path::new(file_path);

    // Check if file exists
    if !path.exists() {
        return Err(Error::msg(format!("File not found: {file_path}")));
    }

    // If path is a directory, search for tokenizer files
    if path.is_dir() {
        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists() {
            // Resolve chat template: provided path takes precedence over auto-discovery
            let final_chat_template =
                resolve_and_log_chat_template(chat_template_path, path, file_path);
            let tokenizer_path_str = tokenizer_json.to_str().ok_or_else(|| {
                Error::msg(format!(
                    "Tokenizer path is not valid UTF-8: {tokenizer_json:?}"
                ))
            })?;
            return create_tokenizer_with_chat_template(
                tokenizer_path_str,
                final_chat_template.as_deref(),
            );
        }

        let has_vocab_and_merges =
            path.join("vocab.json").is_file() && path.join("merges.txt").is_file();

        // Priority 2: Hugging Face Qwen2-style byte-level BPE files. Some
        // official checkpoints (including Qwen3-ASR) intentionally omit
        // tokenizer.json and ship only vocab.json + merges.txt.
        if has_vocab_and_merges && has_qwen2_tokenizer_class(path) {
            let final_chat_template =
                resolve_and_log_chat_template(chat_template_path, path, file_path);
            return Ok(Arc::new(
                HuggingFaceTokenizer::from_vocab_and_merges_dir_with_chat_template(
                    path,
                    final_chat_template.as_deref(),
                )?,
            ));
        }

        // Priority 3: tiktoken.model / *.tiktoken
        // Only forward the user's explicit chat_template_path — tiktoken handles
        // its own config/discovery (tokenizer_config.json → directory discovery).
        if has_tiktoken_file(path) {
            return Ok(Arc::new(TiktokenTokenizer::from_dir_with_chat_template(
                path,
                chat_template_path,
            )?));
        }

        // Preserve the specific validation error for vocab + merges layouts
        // when there is no supported tokenizer to fall back to.
        if has_vocab_and_merges {
            let final_chat_template =
                resolve_and_log_chat_template(chat_template_path, path, file_path);
            return Ok(Arc::new(
                HuggingFaceTokenizer::from_vocab_and_merges_dir_with_chat_template(
                    path,
                    final_chat_template.as_deref(),
                )?,
            ));
        }

        return Err(Error::msg(format!(
            "Directory '{file_path}' does not contain a valid tokenizer file (tokenizer.json, tiktoken.model, *.tiktoken, or vocab.json)"
        )));
    }

    // Try to determine tokenizer type from extension
    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|s| s.to_lowercase());

    let result = match extension.as_deref() {
        Some("json") => {
            let tokenizer =
                HuggingFaceTokenizer::from_file_with_chat_template(file_path, chat_template_path)?;

            Ok(Arc::new(tokenizer) as Arc<dyn traits::Tokenizer>)
        }
        Some("model") | Some("tiktoken") => {
            // Check if it's a tiktoken file (tiktoken.model / *.tiktoken) before assuming SentencePiece
            if is_tiktoken_file(path) {
                Ok(Arc::new(TiktokenTokenizer::from_file_with_chat_template(
                    path,
                    chat_template_path,
                )?) as Arc<dyn traits::Tokenizer>)
            } else {
                Err(Error::msg("SentencePiece models not yet supported"))
            }
        }
        Some("gguf") => {
            // GGUF format
            Err(Error::msg("GGUF format not yet supported"))
        }
        _ => {
            // Try to auto-detect by reading file content
            auto_detect_tokenizer(file_path)
        }
    };

    result
}

fn has_qwen2_tokenizer_class(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .and_then(|config| {
            config
                .get("tokenizer_class")
                .and_then(serde_json::Value::as_str)
                .map(|class| matches!(class, "Qwen2Tokenizer" | "Qwen2TokenizerFast"))
        })
        .unwrap_or(false)
}

/// Auto-detect tokenizer type by examining file content
fn auto_detect_tokenizer(file_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    let mut file = File::open(file_path)?;
    let mut buffer = vec![0u8; 512]; // Read first 512 bytes for detection
    let bytes_read = file.read(&mut buffer)?;
    buffer.truncate(bytes_read);

    // Check for JSON (HuggingFace format)
    if is_likely_json(&buffer) {
        let tokenizer = HuggingFaceTokenizer::from_file(file_path)?;
        return Ok(Arc::new(tokenizer));
    }

    // Check for GGUF magic number
    if buffer.len() >= 4 && &buffer[0..4] == b"GGUF" {
        return Err(Error::msg("GGUF format detected but not yet supported"));
    }

    // Check for SentencePiece model
    if is_likely_sentencepiece(&buffer) {
        return Err(Error::msg(
            "SentencePiece model detected but not yet supported",
        ));
    }

    Err(Error::msg(format!(
        "Unable to determine tokenizer type for file: {file_path}"
    )))
}

/// Check if the buffer likely contains JSON data
fn is_likely_json(buffer: &[u8]) -> bool {
    // Skip UTF-8 BOM if present
    let content = if buffer.len() >= 3 && buffer[0..3] == [0xEF, 0xBB, 0xBF] {
        &buffer[3..]
    } else {
        buffer
    };

    // Find first non-whitespace character without allocation
    if let Some(first_byte) = content.iter().find(|&&b| !b.is_ascii_whitespace()) {
        *first_byte == b'{' || *first_byte == b'['
    } else {
        false
    }
}

/// Check if the buffer likely contains a SentencePiece model
fn is_likely_sentencepiece(buffer: &[u8]) -> bool {
    // SentencePiece models often start with specific patterns
    // This is a simplified check
    if buffer.len() < 12 {
        return false;
    }

    // Check header patterns first (cheap)
    if buffer.starts_with(b"\x0a\x09") || buffer.starts_with(b"\x08\x00") {
        return true;
    }

    // Single-pass scan for special token markers
    // Instead of multiple windows() calls, scan once looking for all patterns
    let patterns: &[&[u8]] = &[b"<unk", b"<s>", b"</s>"];
    for window in buffer.windows(4) {
        for pattern in patterns {
            if window.starts_with(pattern) {
                return true;
            }
        }
    }
    false
}

/// The chat template file beside a tokenizer, resolved as the engine's own
/// tokenizer resolves it: `chat_template.jinja` outranks the `chat_template`
/// field of `tokenizer_config.json`, which transformers' tokenizer reads next.
/// `chat_template.json` is the processor's legacy template file, which that
/// tokenizer never reads: a checkpoint may ship it beside a config template
/// and the two can differ, so it serves only a config without one. Any other
/// `.jinja` file comes last. `None` leaves the config's template, or none.
pub fn discover_chat_template_in_dir(dir: &Path) -> Option<String> {
    use std::fs;

    let jinja_path = dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return jinja_path.to_str().map(|s| s.to_string());
    }

    if config_has_chat_template(dir) {
        return None;
    }

    let json_template_path = dir.join("chat_template.json");
    if json_template_path.exists() {
        return json_template_path.to_str().map(|s| s.to_string());
    }

    // Any .jinja file (for models with non-standard naming)
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".jinja") {
                    return entry.path().to_str().map(|s| s.to_string());
                }
            }
        }
    }

    None
}

/// Whether `tokenizer_config.json` in `dir` carries a `chat_template` the
/// loaders read: a string. A named-template list (the form some checkpoints
/// ship) is not read by them, so it must not stand in the way of the file
/// fallbacks.
fn config_has_chat_template(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .is_some_and(|config| {
            config
                .get("chat_template")
                .is_some_and(serde_json::Value::is_string)
        })
}

/// Helper function to resolve and log chat template selection
///
/// Resolves the final chat template to use by prioritizing provided path over auto-discovery,
/// and logs the source for debugging purposes.
fn resolve_and_log_chat_template(
    provided_path: Option<&str>,
    discovery_dir: &Path,
    model_name: &str,
) -> Option<String> {
    let final_chat_template = provided_path
        .map(|s| s.to_string())
        .or_else(|| discover_chat_template_in_dir(discovery_dir));

    match (&provided_path, &final_chat_template) {
        (Some(provided), _) => {
            debug!("Using provided chat template: {}", provided);
        }
        (None, Some(discovered)) => {
            debug!(
                "Auto-discovered chat template in '{}': {}",
                discovery_dir.display(),
                discovered
            );
        }
        (None, None) => {
            debug!(
                "No chat template provided or discovered for model: {}",
                model_name
            );
        }
    }

    final_chat_template
}

/// Factory function to create tokenizer from a model name or path (async version)
pub async fn create_tokenizer_async(
    model_name_or_path: &str,
) -> Result<Arc<dyn traits::Tokenizer>> {
    create_tokenizer_async_with_chat_template(model_name_or_path, None).await
}

/// Check if a model name looks like an OpenAI model that should use tiktoken.
///
/// Uses targeted patterns to minimise false positives.  False negatives are
/// acceptable — the caller falls back to HuggingFace Hub download, which
/// handles non-OpenAI models correctly.  False positives waste time by
/// trying tiktoken for a model that doesn't support it.
///
/// Matched model families:
///   gpt-4, gpt-4o, gpt-4-turbo, gpt-4-32k, gpt-4o-mini, gpt-4.5-preview,
///   gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-instruct,
///   chatgpt-4o-latest,
///   o1, o1-mini, o1-preview, o3, o3-mini, o3-pro, o4-mini,
///   text-davinci-003, code-davinci-002, davinci,
///   text-curie-001, curie, text-babbage-001, babbage,
///   text-ada-001, text-embedding-ada-002, ada
fn is_likely_openai_model(name: &str) -> bool {
    let bare = name.rsplit('/').next().unwrap_or(name);

    // GPT family: gpt-4*, gpt-3.5*, chatgpt-*
    // Require "gpt-" followed by a digit to avoid false positives like "gpt-oss-20b"
    if bare.starts_with("gpt-") && bare.as_bytes().get(4).is_some_and(|b| b.is_ascii_digit()) {
        return true;
    }
    if bare.starts_with("chatgpt-") {
        return true;
    }

    // Reasoning model family (o1, o1-mini, o1-preview, o3, o3-mini, o3-pro, o4-mini, …)
    // Pattern: "o" + digit, optionally followed by "-suffix"
    if bare.starts_with('o')
        && bare.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit())
        && bare.as_bytes().get(2).is_none_or(|b| *b == b'-')
    {
        return true;
    }

    // Legacy completion / embedding / edit models.
    // Use prefix-based checks ("text-", "code-") or exact-match for bare
    // names to avoid matching unrelated models (e.g. "adapter-v2" for "ada",
    // "turbo-llama" for "turbo").
    matches!(bare, "davinci" | "curie" | "babbage" | "ada")
        || bare.starts_with("text-davinci")
        || bare.starts_with("code-davinci")
        || bare.starts_with("text-curie")
        || bare.starts_with("text-babbage")
        || bare.starts_with("text-ada")
        || bare.starts_with("text-embedding-ada")
        || bare.starts_with("code-cushman")
}

/// Factory function to create tokenizer with optional chat template (async version)
pub async fn create_tokenizer_async_with_chat_template(
    model_name_or_path: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Check if it's a file path
    let path = Path::new(model_name_or_path);
    if path.exists() {
        return create_tokenizer_with_chat_template(model_name_or_path, chat_template_path);
    }

    // Check if it's a GPT model name that should use Tiktoken
    if is_likely_openai_model(model_name_or_path) {
        // Try tiktoken first, but fall back to HuggingFace if it fails
        match TiktokenTokenizer::from_model_name(model_name_or_path) {
            Ok(tokenizer) => return Ok(Arc::new(tokenizer)),
            Err(e) => {
                debug!(
                    "Tiktoken failed for '{}': {}, falling back to HuggingFace",
                    model_name_or_path, e
                );
            }
        }
    }

    // Try to download tokenizer files from HuggingFace
    match download_tokenizer_from_hf(model_name_or_path).await {
        Ok(cache_dir) => {
            create_tokenizer_from_downloaded_dir(&cache_dir, model_name_or_path, chat_template_path)
        }
        Err(e) => Err(Error::msg(format!(
            "Failed to download tokenizer from HuggingFace: {e}"
        ))),
    }
}

/// The tokenizer of a checkpoint downloaded from the Hub, from the files in
/// its cache directory: `tokenizer.json`, else a tiktoken vocabulary, else the
/// layouts the directory loader knows (`vocab.json` + `merges.txt`).
fn create_tokenizer_from_downloaded_dir(
    cache_dir: &Path,
    model_name_or_path: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Look for tokenizer.json in the cache directory
    let tokenizer_path = cache_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        // Resolve chat template: provided path takes precedence over auto-discovery
        let final_chat_template =
            resolve_and_log_chat_template(chat_template_path, cache_dir, model_name_or_path);

        let tokenizer_path_str = tokenizer_path.to_str().ok_or_else(|| {
            Error::msg(format!(
                "Tokenizer path is not valid UTF-8: {tokenizer_path:?}"
            ))
        })?;
        create_tokenizer_with_chat_template(tokenizer_path_str, final_chat_template.as_deref())
    } else if has_tiktoken_file(cache_dir) {
        Ok(Arc::new(TiktokenTokenizer::from_dir_with_chat_template(
            cache_dir,
            chat_template_path,
        )?))
    } else {
        // A checkpoint that ships neither, such as the vocab.json + merges.txt
        // pair of the Qwen2 family, is the directory loader's: it gets the
        // directory, not one of the files (a tokenizer_config.json is not a
        // tokenizer), and names the layouts it accepts when none is there.
        let cache_dir_str = cache_dir
            .to_str()
            .ok_or_else(|| Error::msg(format!("Cache path is not valid UTF-8: {cache_dir:?}")))?;
        create_tokenizer_with_chat_template(cache_dir_str, chat_template_path)
    }
}

/// Factory function to create tokenizer from a model name or path (blocking version)
///
/// This delegates to `create_tokenizer_with_chat_template_blocking` with no chat template,
/// which handles both local files and HuggingFace Hub downloads uniformly.
pub fn create_tokenizer(model_name_or_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    create_tokenizer_with_chat_template_blocking(model_name_or_path, None)
}

/// Factory function to create tokenizer with optional chat template (blocking version)
pub fn create_tokenizer_with_chat_template_blocking(
    model_name_or_path: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Check if it's a file path
    let path = Path::new(model_name_or_path);
    if path.exists() {
        return create_tokenizer_with_chat_template(model_name_or_path, chat_template_path);
    }

    // Check if it's a GPT model name that should use Tiktoken
    // Try tiktoken first, but fall back to HuggingFace if it fails
    if is_likely_openai_model(model_name_or_path) {
        match TiktokenTokenizer::from_model_name(model_name_or_path) {
            Ok(tokenizer) => return Ok(Arc::new(tokenizer)),
            Err(e) => {
                debug!(
                    "Tiktoken failed for '{}': {}, falling back to HuggingFace",
                    model_name_or_path, e
                );
            }
        }
    }

    // Fall back to HuggingFace Hub download (requires tokio runtime)
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(create_tokenizer_async_with_chat_template(
                model_name_or_path,
                chat_template_path,
            ))
        })
    } else {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(create_tokenizer_async_with_chat_template(
            model_name_or_path,
            chat_template_path,
        ))
    }
}

/// Get information about a tokenizer file
pub fn get_tokenizer_info(file_path: &str) -> Result<TokenizerType> {
    let path = Path::new(file_path);

    if !path.exists() {
        return Err(Error::msg(format!("File not found: {file_path}")));
    }

    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|s| s.to_lowercase());

    match extension.as_deref() {
        Some("json") => Ok(TokenizerType::HuggingFace(file_path.to_string())),
        _ => {
            // Try auto-detection
            use std::{fs::File, io::Read};

            let mut file = File::open(file_path)?;
            let mut buffer = vec![0u8; 512];
            let bytes_read = file.read(&mut buffer)?;
            buffer.truncate(bytes_read);

            if is_likely_json(&buffer) {
                Ok(TokenizerType::HuggingFace(file_path.to_string()))
            } else {
                Err(Error::msg("Unknown tokenizer type"))
            }
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::print_stdout,
    reason = "diagnostic output in tests for CI skip messages and download results"
)]
mod tests {
    use serde_json::json;

    use super::{
        create_tokenizer, create_tokenizer_async, create_tokenizer_from_downloaded_dir,
        create_tokenizer_from_file, is_likely_json, is_likely_openai_model,
    };

    /// A downloaded checkpoint that ships `vocab.json` + `merges.txt` and no
    /// `tokenizer.json` (Qwen3-Omni) loads from its cache directory, and gives
    /// the ids of a `tokenizer.json` holding the same vocabulary, merges and
    /// added tokens with the family's normalizer and pre-tokenizer.
    #[test]
    fn a_downloaded_vocab_and_merges_pair_loads_like_its_tokenizer_json() {
        let vocab = json!({
            "H": 0, "e": 1, "l": 2, "o": 3, "Ġ": 4, "w": 5, "r": 6, "d": 7, "!": 8,
            "ll": 9, "He": 10, "Hell": 11, "Hello": 12,
            "Ġw": 13, "Ġwo": 14, "Ġwor": 15, "Ġworl": 16, "Ġworld": 17
        });
        let merges = [
            ["l", "l"],
            ["H", "e"],
            ["He", "ll"],
            ["Hell", "o"],
            ["Ġ", "w"],
            ["Ġw", "o"],
            ["Ġwo", "r"],
            ["Ġwor", "l"],
            ["Ġworl", "d"],
        ];
        let added = |id: u32, content: &str, special: bool| {
            json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": special})
        };

        let downloaded = tempfile::tempdir().unwrap();
        std::fs::write(
            downloaded.path().join("vocab.json"),
            serde_json::to_vec(&vocab).unwrap(),
        )
        .unwrap();
        let mut merges_txt = String::from("#version: 0.2\n");
        for [left, right] in &merges {
            merges_txt += &format!("{left} {right}\n");
        }
        std::fs::write(downloaded.path().join("merges.txt"), merges_txt).unwrap();
        std::fs::write(
            downloaded.path().join("tokenizer_config.json"),
            serde_json::to_vec(&json!({
                "tokenizer_class": "Qwen2Tokenizer",
                "add_prefix_space": false,
                "added_tokens_decoder": {
                    "18": added(18, "<|im_start|>", true),
                    "19": added(19, "<tool_call>", false)
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let as_json = tempfile::tempdir().unwrap();
        std::fs::write(
            as_json.path().join("tokenizer.json"),
            serde_json::to_vec(&json!({
                "version": "1.0",
                "added_tokens": [added(18, "<|im_start|>", true), added(19, "<tool_call>", false)],
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                    {"type": "Split",
                     "pattern": {"Regex": crate::huggingface::QWEN2_PRETOKENIZE_REGEX},
                     "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
                     "use_regex": false}
                ]},
                "post_processor": {"type": "ByteLevel", "add_prefix_space": true,
                                   "trim_offsets": false, "use_regex": true},
                "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
                            "use_regex": true},
                "model": {"type": "BPE", "dropout": null, "unk_token": null,
                          "continuing_subword_prefix": "", "end_of_word_suffix": "",
                          "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                          "vocab": vocab, "merges": merges}
            }))
            .unwrap(),
        )
        .unwrap();

        let pair = create_tokenizer_from_downloaded_dir(downloaded.path(), "org/model", None)
            .unwrap_or_else(|e| panic!("the downloaded pair does not load: {e}"));
        let reference = create_tokenizer(as_json.path().to_str().unwrap()).unwrap();
        assert_eq!(
            pair.encode("Hello world!", false).unwrap().token_ids(),
            [12, 17, 8]
        );
        for text in [
            "Hello world!",
            "<|im_start|>Hello<tool_call> world",
            "Hello  world\n\n!",
            "",
        ] {
            assert_eq!(
                pair.encode(text, false).unwrap().token_ids(),
                reference.encode(text, false).unwrap().token_ids(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn test_json_detection() {
        assert!(is_likely_json(b"{\"test\": \"value\"}"));
        assert!(is_likely_json(b"  \n\t{\"test\": \"value\"}"));
        assert!(is_likely_json(b"[1, 2, 3]"));
        assert!(!is_likely_json(b"not json"));
        assert!(!is_likely_json(b""));
    }

    #[test]
    fn test_mock_tokenizer_creation() {
        let tokenizer = create_tokenizer_from_file("mock").unwrap();
        assert_eq!(tokenizer.vocab_size(), 14); // Mock tokenizer has 14 tokens
    }

    #[test]
    fn test_file_not_found() {
        let result = create_tokenizer_from_file("/nonexistent/file.json");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("File not found"));
        }
    }

    #[test]
    fn test_create_tiktoken_tokenizer() {
        let tokenizer = create_tokenizer("gpt-4").unwrap();
        assert!(tokenizer.vocab_size() > 0);

        let text = "Hello, world!";
        let encoding = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }

    #[tokio::test]
    async fn test_download_tokenizer_from_hf() {
        // Skip this test if HF_TOKEN is not set and we're in CI
        if std::env::var("CI").is_ok() && std::env::var("HF_TOKEN").is_err() {
            println!("Skipping HF download test in CI without HF_TOKEN");
            return;
        }

        // Try to create tokenizer for a known small model
        let result = create_tokenizer_async("bert-base-uncased").await;

        // The test might fail due to network issues or rate limiting
        // so we just check that the function executes without panic
        match result {
            Ok(tokenizer) => {
                assert!(tokenizer.vocab_size() > 0);
                println!("Successfully downloaded and created tokenizer");
            }
            Err(e) => {
                println!("Download failed (this might be expected): {e}");
                // Don't fail the test - network issues shouldn't break CI
            }
        }
    }

    #[test]
    fn test_is_likely_openai_model_positives() {
        // GPT-4 family
        assert!(is_likely_openai_model("gpt-4"));
        assert!(is_likely_openai_model("gpt-4o"));
        assert!(is_likely_openai_model("gpt-4o-mini"));
        assert!(is_likely_openai_model("gpt-4-turbo"));
        assert!(is_likely_openai_model("gpt-4-32k"));
        assert!(is_likely_openai_model("gpt-4.5-preview"));

        // GPT-3.5 family
        assert!(is_likely_openai_model("gpt-3.5-turbo"));
        assert!(is_likely_openai_model("gpt-3.5-turbo-16k"));
        assert!(is_likely_openai_model("gpt-3.5-turbo-instruct"));

        // ChatGPT
        assert!(is_likely_openai_model("chatgpt-4o-latest"));

        // Reasoning models
        assert!(is_likely_openai_model("o1"));
        assert!(is_likely_openai_model("o1-mini"));
        assert!(is_likely_openai_model("o1-preview"));
        assert!(is_likely_openai_model("o3"));
        assert!(is_likely_openai_model("o3-mini"));
        assert!(is_likely_openai_model("o3-pro"));
        assert!(is_likely_openai_model("o4-mini"));

        // Legacy models
        assert!(is_likely_openai_model("davinci"));
        assert!(is_likely_openai_model("text-davinci-003"));
        assert!(is_likely_openai_model("code-davinci-002"));
        assert!(is_likely_openai_model("curie"));
        assert!(is_likely_openai_model("text-curie-001"));
        assert!(is_likely_openai_model("babbage"));
        assert!(is_likely_openai_model("text-babbage-001"));
        assert!(is_likely_openai_model("ada"));
        assert!(is_likely_openai_model("text-ada-001"));
        assert!(is_likely_openai_model("text-embedding-ada-002"));
        assert!(is_likely_openai_model("code-cushman-001"));

        // With org prefix
        assert!(is_likely_openai_model("openai/gpt-4"));
        assert!(is_likely_openai_model("openai/o1-mini"));
        assert!(is_likely_openai_model("openai/davinci"));
    }

    #[test]
    fn test_is_likely_openai_model_negatives() {
        // HuggingFace models that should NOT match
        assert!(!is_likely_openai_model("openai/gpt-oss-20b"));
        assert!(!is_likely_openai_model("meta-llama/Llama-3-8B"));
        assert!(!is_likely_openai_model("mistralai/Mistral-7B"));
        assert!(!is_likely_openai_model("bert-base-uncased"));

        // Names that previously caused false positives
        assert!(!is_likely_openai_model("turbo-llama"));
        assert!(!is_likely_openai_model("adapter-v2"));
        assert!(!is_likely_openai_model("oracle-7b"));
        assert!(!is_likely_openai_model("open-llama"));
    }

    /// A checkpoint may ship the processor's legacy `chat_template.json`
    /// beside a `tokenizer_config.json` template, and the two can differ: the
    /// tokenizer's own template is what the engine's tokenizer renders with, a
    /// `chat_template.jinja` outranks it, and the legacy file serves a config
    /// without one.
    #[test]
    fn discovery_prefers_the_tokenizer_s_own_template_over_the_legacy_processor_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let discover = |path: &std::path::Path| super::discover_chat_template_in_dir(path);
        std::fs::write(
            path.join("chat_template.json"),
            r#"{"chat_template": "processor"}"#,
        )
        .unwrap();
        std::fs::write(
            path.join("tokenizer_config.json"),
            r#"{"chat_template": "tokenizer"}"#,
        )
        .unwrap();
        assert_eq!(discover(path), None);

        std::fs::write(path.join("chat_template.jinja"), "file").unwrap();
        assert_eq!(
            discover(path).as_deref(),
            path.join("chat_template.jinja").to_str()
        );

        std::fs::remove_file(path.join("chat_template.jinja")).unwrap();
        std::fs::write(
            path.join("tokenizer_config.json"),
            r#"{"tokenizer_class": "PreTrainedTokenizerFast"}"#,
        )
        .unwrap();
        assert_eq!(
            discover(path).as_deref(),
            path.join("chat_template.json").to_str()
        );

        // A named-template list in the config is not a template the loaders
        // read, so the legacy file still serves such a checkpoint.
        std::fs::write(
            path.join("tokenizer_config.json"),
            r#"{"chat_template": [{"name": "default", "template": "a"}, {"name": "tool_use", "template": "b"}]}"#,
        )
        .unwrap();
        assert_eq!(
            discover(path).as_deref(),
            path.join("chat_template.json").to_str()
        );
    }
}
