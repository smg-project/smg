//! Verify that `HuggingFaceTokenizer` selects the right chat-template renderer
//! based on `config.json::architectures`.
#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use llm_tokenizer::{
        chat_template::{ChatTemplateParams, ThinkingKeyName, ThinkingToggle},
        huggingface::HuggingFaceTokenizer,
        TokenizerTrait,
    };
    use serde_json::json;
    use tempfile::TempDir;
    /// A minimal tokenizer.json that loads cleanly. The only requirement is that
    /// it parses; the encoder logic does not call back into the tokenizer here.
    const MIN_TOKENIZER_JSON: &str = r#"{
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
    fn write_dir(architectures: Option<&[&str]>) -> (TempDir, String) {
        let temp = TempDir::new().unwrap();
        let tok_path = temp.path().join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        if let Some(archs) = architectures {
            let cfg_path = temp.path().join("config.json");
            let body = json!({ "architectures": archs }).to_string();
            fs::write(&cfg_path, body).unwrap();
        }
        let p = tok_path.to_str().unwrap().to_string();
        (temp, p)
    }

    #[test]
    fn config_with_deepseek_v32_arch_uses_v32_renderer() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV32ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        // V3.2 emits BOS + <｜User｜>Hello<｜Assistant｜></think> in chat mode.
        assert!(out.contains("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"));
        assert!(out.contains("<\u{FF5C}User\u{FF5C}>Hello<\u{FF5C}Assistant\u{FF5C}>"));
        assert!(out.ends_with("</think>"));
    }
    #[test]
    fn config_with_deepseek_v4_arch_uses_v4_renderer() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        // V4 emits BOS + <｜User｜>Hello<｜Assistant｜></think> in chat mode.
        assert!(out.contains("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"));
        assert!(out.contains("<\u{FF5C}User\u{FF5C}>Hello<\u{FF5C}Assistant\u{FF5C}>"));
        assert!(out.ends_with("</think>"));
    }

    #[test]
    fn config_with_unrelated_arch_falls_back_to_jinja() {
        // A non-DeepSeek architecture should keep using the Jinja renderer; with
        // no chat_template set, applying the template should error rather than
        // silently picking a DeepSeek encoder.
        let (_tmp, tok) = write_dir(Some(&["LlamaForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        assert!(
            result.is_err(),
            "expected error from missing Jinja template"
        );
    }
    #[test]
    fn no_config_json_falls_back_to_jinja() {
        // No sibling config.json — must still default to Jinja and not blow up.
        let (_tmp, tok) = write_dir(None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        // Without a chat template registered, the Jinja renderer surfaces an error.
        // The important thing is that we did NOT auto-select a DeepSeek encoder.
        assert!(result.is_err());
    }
    #[test]
    fn malformed_config_json_falls_back_to_jinja() {
        let temp = TempDir::new().unwrap();
        let tok_path = temp.path().join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(temp.path().join("config.json"), "{ this is not json").unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(tok_path.to_str().unwrap()).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        assert!(result.is_err());
    }
    #[test]
    fn deepseek_v4_injects_tools_into_system_message() {
        // Client passes tools at the request level (params.tools), not embedded
        // in messages. The shim must attach them to a system message so the
        // encoder renders the tools block.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("## Tools"), "tools block missing: {out}");
        assert!(
            out.contains("get_weather"),
            "tool name missing from prompt: {out}"
        );
        assert!(
            out.contains("<\u{FF5C}DSML\u{FF5C}tool_calls>"),
            "V4 DSML invocation grammar missing: {out}"
        );
    }

    #[test]
    fn deepseek_v32_injects_tools_into_system_message() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV32ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hi" })];
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "ping", "description": "ping", "parameters": {} }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("## Tools"), "tools block missing: {out}");
        assert!(out.contains("ping"), "tool name missing: {out}");
        // V3.2 uses function_calls (vs V4's tool_calls).
        assert!(
            out.contains("<\u{FF5C}DSML\u{FF5C}function_calls>"),
            "V3.2 DSML invocation grammar missing: {out}"
        );
    }

    #[test]
    fn deepseek_v4_attaches_tools_to_existing_system_message() {
        // When a system message is already present, tools should attach to it
        // rather than inserting a second system block.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![
            json!({ "role": "system", "content": "Be concise." }),
            json!({ "role": "user", "content": "Hi" }),
        ];
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "ping", "description": "ping", "parameters": {} }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(
            out.contains("Be concise."),
            "existing system content lost: {out}"
        );
        assert!(out.contains("ping"), "tool not attached: {out}");
    }

    #[test]
    fn deepseek_renderers_report_thinking_introspection() {
        // V3.2 / V4 inject `<think>` in the prefill when thinking is on, and
        // gate thinking on the `thinking` kwarg. The trait methods must
        // surface this so the gateway can call `mark_reasoning_started` on
        // the conditional reasoning parser (deepseek_v31 etc).
        for arch in &["DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM"] {
            let (_tmp, tok) = write_dir(Some(&[*arch]));
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            assert_eq!(
                tokenizer.thinking_toggle(),
                ThinkingToggle::DefaultOff,
                "{arch}: expected DefaultOff toggle"
            );
            assert_eq!(
                tokenizer.thinking_key_name(),
                Some(ThinkingKeyName::Thinking),
                "{arch}: expected Thinking key name"
            );
            assert!(
                tokenizer.think_in_prefill(),
                "{arch}: expected think_in_prefill=true"
            );
        }
    }

    #[test]
    fn deepseek_renderers_honor_thinking_kwarg_only() {
        // `thinking: true` → prompt ends with <think> (thinking mode).
        // `enable_thinking: true` alone → ignored (chat mode), matching
        // `thinking_key_name() == Some(Thinking)` and sglang's DeepSeek path.
        for arch in &["DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM"] {
            let (_tmp, tok) = write_dir(Some(&[*arch]));
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            let messages = vec![json!({ "role": "user", "content": "Hi" })];

            let mut thinking_kwargs: HashMap<String, serde_json::Value> = HashMap::new();
            thinking_kwargs.insert("thinking".to_string(), serde_json::Value::Bool(true));
            let out_thinking = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        template_kwargs: Some(&thinking_kwargs),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                out_thinking.ends_with("<think>"),
                "{arch}: thinking=true should enter thinking mode: {out_thinking}"
            );

            let mut enable_thinking_kwargs: HashMap<String, serde_json::Value> = HashMap::new();
            enable_thinking_kwargs
                .insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
            let out_enable = tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        template_kwargs: Some(&enable_thinking_kwargs),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                out_enable.ends_with("</think>"),
                "{arch}: enable_thinking alone must NOT enter thinking mode: {out_enable}"
            );
        }
    }

    #[test]
    fn deepseek_v4_renderer_passes_reasoning_effort() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let mut kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        kwargs.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("max".to_string()),
        );
        kwargs.insert("thinking".to_string(), serde_json::Value::Bool(true));
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(
            out.contains("Reasoning Effort: Absolute maximum"),
            "expected reasoning-effort prefix in V4 output"
        );
        assert!(
            out.ends_with("<think>"),
            "thinking mode should leave a <think> token open"
        );
    }

    // -----------------------------------------------------------------------
    // 0731-vs-original effort-encoding identification
    // -----------------------------------------------------------------------

    /// Stand-ins for the `encoding/encoding_dsv4.py` each checkpoint ships;
    /// only the effort block differs between revisions, so the fingerprint is
    /// the 0731-only `max` prompt text.
    const BASE_ENCODER_SNIPPET: &str = r#"REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
)"#;
    const V0731_ENCODER_SNIPPET: &str = r#"REASONING_EFFORT_PROMPTS = {
    "low": "",
    "high": "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "max": "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n",
}"#;

    /// Build `<temp>/<model_name>/` with a V4 config and optionally the
    /// checkpoint's shipped Python encoder.
    fn write_v4_model_dir(model_name: &str, encoder_source: Option<&str>) -> (TempDir, String) {
        let temp = TempDir::new().unwrap();
        let model_dir = temp.path().join(model_name);
        fs::create_dir(&model_dir).unwrap();
        let tok_path = model_dir.join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(
            model_dir.join("config.json"),
            json!({ "architectures": ["DeepseekV4ForCausalLM"] }).to_string(),
        )
        .unwrap();
        if let Some(source) = encoder_source {
            let encoding_dir = model_dir.join("encoding");
            fs::create_dir(&encoding_dir).unwrap();
            fs::write(encoding_dir.join("encoding_dsv4.py"), source).unwrap();
        }
        (temp, tok_path.to_str().unwrap().to_string())
    }

    fn render_with_native_effort(
        tokenizer: &HuggingFaceTokenizer,
        effort: &str,
    ) -> anyhow::Result<String> {
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!(effort))]);
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        tokenizer.apply_chat_template(&messages, params)
    }

    #[test]
    fn shipped_0731_encoder_selects_0731_effort_encoding() {
        // The dir name says nothing about 0731 — the shipped encoder decides.
        let (_tmp, tok) = write_v4_model_dir("my-local-v4-copy", Some(V0731_ENCODER_SNIPPET));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();

        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Beyond maximum"), "{out}");
        let out = render_with_native_effort(&tokenizer, "high").unwrap();
        assert!(out.contains("Reasoning Effort: Absolute maximum"), "{out}");
        assert!(!out.contains("Beyond maximum"), "{out}");
        // `low` is valid in 0731 and renders no prefix.
        let out = render_with_native_effort(&tokenizer, "low").unwrap();
        assert!(!out.contains("Reasoning Effort:"), "{out}");
    }

    #[test]
    fn shipped_base_encoder_wins_over_0731_dir_name() {
        // Fingerprint beats the name: a dir named 0731 that ships the base
        // encoder still gets the original effort encoding.
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", Some(BASE_ENCODER_SNIPPET));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();

        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Absolute maximum"), "{out}");
        assert!(!out.contains("Beyond maximum"), "{out}");
        // The original Python encoder accepts `high` but renders nothing.
        let out = render_with_native_effort(&tokenizer, "high").unwrap();
        assert!(!out.contains("Reasoning Effort:"), "{out}");
        // `low` does not exist in the original revision.
        let err = render_with_native_effort(&tokenizer, "low").unwrap_err();
        assert!(
            err.to_string().contains("must be one of high, max"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_encoder_falls_back_to_dir_name() {
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Beyond maximum"), "{out}");

        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash", None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Absolute maximum"), "{out}");
        assert!(!out.contains("Beyond maximum"), "{out}");
    }

    #[test]
    fn invalid_native_effort_is_rejected_loudly() {
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        for effort in ["medium", "turbo", "MAX"] {
            let err = render_with_native_effort(&tokenizer, effort).unwrap_err();
            assert!(
                err.to_string().contains("must be one of low, high, max"),
                "effort {effort}: unexpected error: {err}"
            );
        }
        // Non-string values are rejected too.
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!(3))]);
        let err = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(err.to_string().contains("must be one of low, high, max"));
    }

    #[test]
    fn native_effort_enables_thinking_unless_overridden() {
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];

        // Effort alone implies thinking mode (the prefix only exists there).
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.ends_with("<think>"), "{out}");

        // An explicit thinking=false still wins and suppresses the prefix.
        let kwargs = HashMap::from([
            ("reasoning_effort".to_string(), json!("max")),
            ("thinking".to_string(), json!(false)),
        ]);
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.ends_with("</think>"), "{out}");
        assert!(!out.contains("Reasoning Effort:"), "{out}");
    }
}
