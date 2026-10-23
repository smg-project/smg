use anyhow::{anyhow, Result};
use llm_tokenizer::{chat_template::ChatTemplateParams, create_tokenizer};

const TOKENIZER_DIR_ENV: &str = "QWEN3_ASR_TOKENIZER_DIR";

#[test]
#[ignore = "requires an official Qwen3-ASR-1.7B tokenizer snapshot"]
fn official_qwen3_asr_matches_transformers_qwen2_tokenizer() -> Result<()> {
    let dir = std::env::var(TOKENIZER_DIR_ENV)
        .map_err(|_| anyhow!("set {TOKENIZER_DIR_ENV} to the tokenizer snapshot"))?;
    let tokenizer = create_tokenizer(&dir)?;

    let cases: &[(&str, &[u32])] = &[
        ("Hello, world!", &[9707, 11, 1879, 0]),
        (
            "I'm testing Qwen2: 12345\nsecond line.",
            &[
                40, 2776, 7497, 1207, 16948, 17, 25, 220, 16, 17, 18, 19, 20, 198, 5569, 1555, 13,
            ],
        ),
        (
            "Cafe\u{301} 中文🙂 — naïve",
            &[34, 2577, 963, 72858, 16744, 145080, 1959, 94880, 586],
        ),
        (
            "  leading  and trailing  ",
            &[220, 6388, 220, 323, 27748, 256],
        ),
        (
            "<|im_start|>assistant\nHello<|im_end|>",
            &[151644, 77091, 198, 9707, 151645],
        ),
        (
            "<|audio_start|><|audio_pad|><|audio_end|>",
            &[151669, 151676, 151670],
        ),
        (
            "<tool_call>{\"name\":\"x\"}</tool_call>",
            &[151657, 4913, 606, 3252, 87, 9207, 151658],
        ),
        ("", &[]),
    ];

    for (text, expected_ids) in cases {
        let encoding = tokenizer.encode(text, false)?;
        assert_eq!(encoding.token_ids(), *expected_ids, "text={text:?}");
    }

    assert_eq!(tokenizer.vocab_size(), 151643);
    for (token, id) in [
        ("<|endoftext|>", 151643),
        ("<|im_start|>", 151644),
        ("<|im_end|>", 151645),
        ("<tool_call>", 151657),
        ("<|audio_start|>", 151669),
        ("<|audio_end|>", 151670),
        ("<|audio_pad|>", 151676),
        ("<asr_text>", 151704),
    ] {
        assert_eq!(tokenizer.token_to_id(token), Some(id), "token={token}");
    }

    let special = tokenizer.encode("<|im_start|>assistant\nHello<|im_end|>", false)?;
    assert_eq!(
        tokenizer.decode(special.token_ids(), true)?,
        "assistant\nHello"
    );

    let rendered = tokenizer.apply_chat_template(
        &[serde_json::json!({
            "role": "user",
            "content": [{"type": "audio", "audio": "fixture.wav"}]
        })],
        ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        },
    )?;
    assert!(rendered.contains("<|audio_start|><|audio_pad|><|audio_end|>"));
    assert!(rendered.ends_with("<|im_start|>assistant\n"));
    Ok(())
}
