use std::{fs, path::Path};

use anyhow::Result;
use llm_tokenizer::{chat_template::ChatTemplateParams, create_tokenizer};
use serde_json::json;
use tempfile::TempDir;

const MIN_TIKTOKEN_MODEL: &str = "aGVsbG8= 0\n";

fn write_qwen2_files(dir: &Path, add_prefix_space: bool, first_added_id: u32) -> Result<()> {
    let vocab = json!({
        "H": 0,
        "e": 1,
        "l": 2,
        "o": 3,
        "Ġ": 4,
        "w": 5,
        "r": 6,
        "d": 7,
        "!": 8
    });
    fs::write(dir.join("vocab.json"), serde_json::to_vec(&vocab)?)?;
    fs::write(dir.join("merges.txt"), "#version: 0.2\n")?;

    let config = json!({
        "tokenizer_class": "Qwen2Tokenizer",
        "add_prefix_space": add_prefix_space,
        "add_bos_token": false,
        "eos_token": "<|im_start|>",
        "pad_token": "<|im_start|>",
        "added_tokens_decoder": {
            first_added_id.to_string(): {
                "content": "<|im_start|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            },
            (first_added_id + 1).to_string(): {
                "content": "<tool_call>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": false
            }
        }
    });
    fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec(&config)?,
    )?;
    fs::write(
        dir.join("chat_template.json"),
        r#"{"chat_template":"{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"}"#,
    )?;
    Ok(())
}

#[test]
fn factory_loads_qwen2_vocab_and_merges_directory() -> Result<()> {
    let dir = TempDir::new()?;
    write_qwen2_files(dir.path(), false, 9)?;

    let tokenizer = create_tokenizer(dir.path().to_string_lossy().as_ref())?;
    assert_eq!(tokenizer.vocab_size(), 9);
    assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(9));
    assert_eq!(tokenizer.token_to_id("<tool_call>"), Some(10));

    let text = "Hello world!";
    let encoded = tokenizer.encode(text, false)?;
    assert_eq!(encoded.token_ids(), &[0, 1, 2, 2, 3, 4, 5, 3, 6, 2, 7, 8]);
    assert_eq!(tokenizer.decode(encoded.token_ids(), false)?, text);

    let with_tokens = tokenizer.encode("<|im_start|><tool_call>", false)?;
    assert_eq!(with_tokens.token_ids(), &[9, 10]);
    assert_eq!(
        tokenizer.decode(with_tokens.token_ids(), true)?,
        "<tool_call>"
    );

    let rendered = tokenizer.apply_chat_template(
        &[json!({"role": "user", "content": "Hello"})],
        ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        },
    )?;
    assert_eq!(rendered, "user: Hello\nassistant: ");
    Ok(())
}

#[test]
fn qwen2_loader_honors_add_prefix_space() -> Result<()> {
    let dir = TempDir::new()?;
    write_qwen2_files(dir.path(), true, 9)?;

    let tokenizer = create_tokenizer(dir.path().to_string_lossy().as_ref())?;
    let encoded = tokenizer.encode("Hello", false)?;
    assert_eq!(encoded.token_ids(), &[4, 0, 1, 2, 2, 3]);
    assert_eq!(tokenizer.decode(encoded.token_ids(), false)?, " Hello");
    Ok(())
}

#[test]
fn qwen2_loader_rejects_unrepresentable_added_token_ids() -> Result<()> {
    let dir = TempDir::new()?;
    write_qwen2_files(dir.path(), false, 10)?;

    let error = match create_tokenizer(dir.path().to_string_lossy().as_ref()) {
        Ok(_) => panic!("non-contiguous added token IDs must fail"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("expected ID 10, got Some(9)"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn mixed_non_qwen_directory_falls_back_to_tiktoken() -> Result<()> {
    let dir = TempDir::new()?;
    write_qwen2_files(dir.path(), false, 9)?;
    fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"tokenizer_class":"OtherTokenizer"}"#,
    )?;
    fs::write(dir.path().join("tiktoken.model"), MIN_TIKTOKEN_MODEL)?;

    let tokenizer = create_tokenizer(dir.path().to_string_lossy().as_ref())?;
    assert_eq!(tokenizer.encode("hello", false)?.token_ids(), &[0]);
    Ok(())
}
