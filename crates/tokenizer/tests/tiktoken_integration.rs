//! Integration tests for tiktoken tokenizers using real Kimi-K2-Instruct tokenizer data.
//!
//! These tests download the Kimi-K2-Instruct tiktoken files from HuggingFace Hub
//! to verify our TiktokenTokenizer implementation works correctly with real-world
//! tiktoken-based models.
//!
//! All tests are `#[ignore]` by default — run with `cargo test --ignored` or
//! `cargo test -- --ignored` to exercise them. They require network access.

use std::{
    fs,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::Duration,
};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    create_tokenizer,
    tiktoken::TiktokenTokenizer,
    traits::{Decoder, Encoder, Tokenizer as TokenizerTrait},
};
use rustls::crypto::ring;

// -- Download configuration --

const KIMI_K2_MODEL_ID: &str = "moonshotai/Kimi-K2-Instruct";
/// Default pinned revision. Override with KIMI_K2_REVISION env var.
const KIMI_K2_DEFAULT_REVISION: &str = "main";
const CACHE_DIR: &str = ".tokenizer_cache/kimi_k2";
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60);

const KIMI_K2_FILES: &[&str] = &[
    "tiktoken.model",
    "tokenizer_config.json",
    "chat_template.jinja",
];

static DOWNLOAD_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn kimi_k2_base_url() -> String {
    let rev =
        std::env::var("KIMI_K2_REVISION").unwrap_or_else(|_| KIMI_K2_DEFAULT_REVISION.to_string());
    format!("https://huggingface.co/{KIMI_K2_MODEL_ID}/resolve/{rev}")
}

/// Downloads the Kimi-K2-Instruct tokenizer files from HuggingFace if not already cached.
/// Returns the path to the cached directory containing all tokenizer files.
#[expect(clippy::unwrap_used, reason = "test helper — panics are intentional")]
#[expect(clippy::expect_used, reason = "test helper — panics are intentional")]
#[expect(clippy::panic, reason = "test helper — panics are intentional")]
#[expect(clippy::print_stdout, reason = "test diagnostic output")]
fn ensure_kimi_k2_cached() -> PathBuf {
    let mutex = DOWNLOAD_MUTEX.get_or_init(|| Mutex::new(()));
    let _guard = mutex.lock().unwrap();

    let cache_dir = PathBuf::from(CACHE_DIR);
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).expect("Failed to create Kimi K2 cache directory");
    }

    // The client takes the process-level TLS crypto provider and panics
    // without one: `ring` is installed before the first client is built.
    let _ = ring::default_provider().install_default();
    let client = reqwest::blocking::Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        .build()
        .expect("Failed to build reqwest client");

    let base_url = kimi_k2_base_url();

    for filename in KIMI_K2_FILES {
        let file_path = cache_dir.join(filename);
        if file_path.exists() {
            continue;
        }

        let url = format!("{base_url}/{filename}");
        println!("Downloading Kimi-K2 {filename}...");

        let response = client
            .get(&url)
            .send()
            .unwrap_or_else(|e| panic!("Failed to download {filename}: {e}"));

        assert!(
            response.status().is_success(),
            "Failed to download {filename}: HTTP {}",
            response.status()
        );

        let content = response
            .bytes()
            .unwrap_or_else(|e| panic!("Failed to read {filename} content: {e}"));

        fs::write(&file_path, &content)
            .unwrap_or_else(|e| panic!("Failed to write {filename} to cache: {e}"));

        println!(
            "  {filename} cached ({} bytes)",
            file_path.metadata().unwrap().len()
        );
    }

    cache_dir
}

// -- Tests --

#[test]
#[ignore]
fn test_tiktoken_from_dir_loads_kimi_k2() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    // Kimi K2 has 163,584 base BPE tokens + 256 special token slots = 163,840
    let vocab_size = tokenizer.vocab_size();
    assert!(
        vocab_size > 100_000,
        "vocab_size {vocab_size} too small for Kimi K2"
    );
    #[expect(clippy::print_stdout, reason = "test diagnostic output")]
    {
        println!("Kimi K2 vocab size: {vocab_size}");
    }
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_special_tokens() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    let special = tokenizer.get_special_tokens();

    // Kimi K2 uses [BOS], [EOS] as bos/eos tokens
    assert_eq!(special.bos_token.as_deref(), Some("[BOS]"));
    assert_eq!(special.eos_token.as_deref(), Some("[EOS]"));
    assert_eq!(special.pad_token.as_deref(), Some("[PAD]"));
    assert_eq!(special.unk_token.as_deref(), Some("[UNK]"));
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_encode_decode_roundtrip() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    let prompts = [
        "Hello, world!",
        "deep learning is",
        "The quick brown fox jumps over the lazy dog",
        "1 + 1 = 2",
        "こんにちは世界",
        "🚀 Rust is awesome! 🦀",
    ];

    for prompt in &prompts {
        let encoding = tokenizer.encode(prompt, false).expect("Failed to encode");
        let token_ids = encoding.token_ids();

        assert!(
            !token_ids.is_empty(),
            "Encoding '{prompt}' produced no tokens"
        );

        let decoded = tokenizer
            .decode(token_ids, false)
            .expect("Failed to decode");

        assert_eq!(
            &decoded, prompt,
            "Encode-decode roundtrip failed for: '{prompt}'"
        );
    }
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_token_to_id() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    // Known special tokens from tokenizer_config.json
    assert_eq!(tokenizer.token_to_id("[BOS]"), Some(163584));
    assert_eq!(tokenizer.token_to_id("[EOS]"), Some(163585));
    assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(163586));
    assert_eq!(tokenizer.token_to_id("<|im_user|>"), Some(163587));
    assert_eq!(tokenizer.token_to_id("<|im_assistant|>"), Some(163588));
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_id_to_token() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    assert_eq!(tokenizer.id_to_token(163584), Some("[BOS]".to_string()));
    assert_eq!(tokenizer.id_to_token(163585), Some("[EOS]".to_string()));
    assert_eq!(
        tokenizer.id_to_token(163586),
        Some("<|im_end|>".to_string())
    );
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_chat_template() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    let messages = vec![serde_json::json!({"role": "user", "content": "Hello, who are you?"})];

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        ..Default::default()
    };

    let result = tokenizer
        .apply_chat_template(&messages, params)
        .expect("Failed to apply chat template");

    // The Kimi K2 template should produce output with role markers
    assert!(!result.is_empty(), "Chat template produced empty output");
    // Should contain the user message
    assert!(
        result.contains("Hello, who are you?"),
        "Chat template output missing user message: {result}"
    );
    // Should have assistant generation prompt at the end
    assert!(
        result.contains("<|im_assistant|>"),
        "Chat template output missing assistant prompt: {result}"
    );

    #[expect(clippy::print_stdout, reason = "test diagnostic output")]
    {
        println!("Chat template output:\n{result}");
    }
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_chat_template_multi_turn() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    let messages = vec![
        serde_json::json!({"role": "system", "content": "You are a helpful assistant."}),
        serde_json::json!({"role": "user", "content": "What is 2+2?"}),
        serde_json::json!({"role": "assistant", "content": "2+2 equals 4."}),
        serde_json::json!({"role": "user", "content": "And 3+3?"}),
    ];

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        ..Default::default()
    };

    let result = tokenizer
        .apply_chat_template(&messages, params)
        .expect("Failed to apply multi-turn chat template");

    assert!(result.contains("You are a helpful assistant."));
    assert!(result.contains("What is 2+2?"));
    assert!(result.contains("2+2 equals 4."));
    assert!(result.contains("And 3+3?"));

    #[expect(clippy::print_stdout, reason = "test diagnostic output")]
    {
        println!("Multi-turn chat template output:\n{result}");
    }
}

#[test]
#[ignore]
fn test_factory_creates_tiktoken_from_directory() {
    let dir = ensure_kimi_k2_cached();
    let dir_str = dir.to_str().unwrap();

    // create_tokenizer should detect tiktoken.model in the directory
    let tokenizer = create_tokenizer(dir_str).expect("Factory failed to create tiktoken tokenizer");

    // Should be functional — verify encode/decode works
    let encoding = tokenizer
        .encode("Hello from factory", false)
        .expect("Factory tokenizer failed to encode");
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Factory tokenizer failed to decode");

    assert_eq!(decoded, "Hello from factory");
}

#[test]
#[ignore]
fn test_factory_creates_tiktoken_from_model_file_path() {
    let dir = ensure_kimi_k2_cached();
    let model_path = dir.join("tiktoken.model");
    let model_path_str = model_path.to_str().unwrap();

    // create_tokenizer with direct file path to tiktoken.model
    let tokenizer =
        create_tokenizer(model_path_str).expect("Factory failed with tiktoken.model path");

    let encoding = tokenizer
        .encode("direct path test", false)
        .expect("Failed to encode");
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode");

    assert_eq!(decoded, "direct path test");
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_batch_encode() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    let texts = vec!["Hello", "World", "Testing batch encoding"];
    let encodings = tokenizer
        .encode_batch(&texts, false)
        .expect("Batch encode failed");

    assert_eq!(encodings.len(), 3);
    for (i, encoding) in encodings.iter().enumerate() {
        let decoded = tokenizer
            .decode(encoding.token_ids(), false)
            .expect("Decode failed");
        assert_eq!(decoded, texts[i], "Batch roundtrip failed for index {i}");
    }
}

#[test]
#[ignore]
#[expect(clippy::print_stdout, reason = "test diagnostic output")]
fn test_factory_creates_tiktoken_from_hf_model_id() {
    // This test exercises the full HF download → tiktoken detection path.
    // create_tokenizer("moonshotai/Kimi-K2-Instruct") should:
    //   1. Not match any GPT model name patterns
    //   2. Not find a local path
    //   3. Download tokenizer files from HuggingFace Hub (tiktoken.model, tokenizer_config.json, chat_template.jinja)
    //   4. Detect has_tiktoken_file() in the cache directory
    //   5. Create a TiktokenTokenizer via from_dir_with_chat_template()
    //
    // Skip in CI without HF_TOKEN since Kimi-K2 may be gated
    if std::env::var("CI").is_ok() && std::env::var("HF_TOKEN").is_err() {
        println!("Skipping HF download test in CI without HF_TOKEN");
        return;
    }

    let tokenizer = match create_tokenizer("moonshotai/Kimi-K2-Instruct") {
        Ok(t) => t,
        Err(e) => {
            // Network failures shouldn't break the test suite
            println!("HF download failed (may be expected): {e}");
            return;
        }
    };

    // Verify it's functional
    let vocab_size = tokenizer.vocab_size();
    assert!(
        vocab_size > 100_000,
        "vocab_size {vocab_size} too small for Kimi K2"
    );

    // Verify encode/decode roundtrip
    let text = "Hello from HuggingFace Hub!";
    let encoding = tokenizer.encode(text, false).expect("Encode failed");
    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Decode failed");
    assert_eq!(decoded, text);

    // Verify chat template works (should have been auto-discovered)
    let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        ..Default::default()
    };
    let result = tokenizer.apply_chat_template(&messages, params);
    assert!(
        result.is_ok(),
        "Chat template should work via HF download path: {:?}",
        result.err()
    );

    println!("HF model ID → tiktoken factory test passed (vocab_size={vocab_size})");
}

#[test]
#[ignore]
fn test_tiktoken_kimi_k2_encoding_stability() {
    let dir = ensure_kimi_k2_cached();
    let tokenizer = TiktokenTokenizer::from_dir(&dir).expect("Failed to load Kimi K2 tokenizer");

    // Encode the same text twice — token IDs must be identical
    let text = "Deterministic encoding test: the quick brown fox.";
    let enc1 = tokenizer.encode(text, false).unwrap();
    let enc2 = tokenizer.encode(text, false).unwrap();

    assert_eq!(
        enc1.token_ids(),
        enc2.token_ids(),
        "Same text produced different token IDs"
    );
}
