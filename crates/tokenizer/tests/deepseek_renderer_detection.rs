//! Verify that `HuggingFaceTokenizer` selects the right chat-template renderer
//! based on `config.json::architectures`.
#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use llm_tokenizer::{
        chat_template::{
            ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
        },
        huggingface::HuggingFaceTokenizer,
        TokenizerTrait,
    };
    use openai_protocol::common::{Function, Tool};
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
        write_model_dir(None, architectures, None, None)
    }

    /// Build a model dir (optionally named, under the temp root) with the
    /// minimal tokenizer, an optional config (`architectures` and/or
    /// `model_type`), and an optional shipped Python encoder.
    fn write_model_dir(
        model_name: Option<&str>,
        architectures: Option<&[&str]>,
        model_type: Option<&str>,
        encoder_source: Option<&str>,
    ) -> (TempDir, String) {
        let temp = TempDir::new().unwrap();
        let model_dir = match model_name {
            Some(name) => {
                let dir = temp.path().join(name);
                fs::create_dir(&dir).unwrap();
                dir
            }
            None => temp.path().to_path_buf(),
        };
        let tok_path = model_dir.join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        if architectures.is_some() || model_type.is_some() {
            let mut config = serde_json::Map::new();
            if let Some(archs) = architectures {
                config.insert("architectures".to_string(), json!(archs));
            }
            if let Some(model_type) = model_type {
                config.insert("model_type".to_string(), json!(model_type));
            }
            let body = serde_json::Value::Object(config).to_string();
            fs::write(model_dir.join("config.json"), body).unwrap();
        }
        if let Some(source) = encoder_source {
            let encoding_dir = model_dir.join("encoding");
            fs::create_dir(&encoding_dir).unwrap();
            fs::write(encoding_dir.join("encoding_dsv4.py"), source).unwrap();
        }
        (temp, tok_path.to_str().unwrap().to_string())
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

    fn write_v4_model_dir(model_name: &str, encoder_source: Option<&str>) -> (TempDir, String) {
        write_model_dir(
            Some(model_name),
            Some(&["DeepseekV4ForCausalLM"]),
            None,
            encoder_source,
        )
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
    fn shipped_encoder_fingerprint_beats_dir_name() {
        // Neutral dir name + 0731 encoder -> 0731 encoding.
        let (_tmp, tok) = write_v4_model_dir("my-local-v4-copy", Some(V0731_ENCODER_SNIPPET));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Beyond maximum"), "{out}");

        // 0731 dir name + base encoder -> original encoding, where `low`
        // doesn't exist and is ignored (no thinking, no prefix).
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", Some(BASE_ENCODER_SNIPPET));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Absolute maximum"), "{out}");
        assert!(!out.contains("Beyond maximum"), "{out}");
        let out = render_with_native_effort(&tokenizer, "low").unwrap();
        assert!(!out.contains("Reasoning Effort:"), "{out}");
        assert!(out.ends_with("</think>"), "{out}");
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
    fn unrecognized_native_effort_is_ignored() {
        // The renderer owns interpretation; values outside the revision's set
        // (including merged public efforts like "medium") render chat mode.
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_with_native_effort(&tokenizer, "medium").unwrap();
        assert!(!out.contains("Reasoning Effort:"), "{out}");
        assert!(out.ends_with("</think>"), "{out}");
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

    // -----------------------------------------------------------------------
    // DeepSeek V4.1
    // -----------------------------------------------------------------------

    const V41_BOS: &str = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
    const V41_EOS: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
    /// The system header carrying the default budget (thinking mode only).
    const V41_DEFAULT_EFFORT_HEADER: &str = "<\u{FF5C}System\u{FF5C}>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    /// Tail of a single-user-turn prompt in thinking mode.
    const V41_THINKING_TAIL: &str = "<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}><think>";
    /// Tail of a single-user-turn prompt in chat mode.
    const V41_CHAT_TAIL: &str = "<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}></think>";

    fn v41_tokenizer() -> (TempDir, HuggingFaceTokenizer) {
        let (tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        (tmp, tokenizer)
    }

    /// Render the single user turn `q` with a generation prompt.
    fn render_v41_turn(
        tokenizer: &HuggingFaceTokenizer,
        kwargs: Option<&HashMap<String, serde_json::Value>>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        tokenizer.apply_chat_template(
            &[json!({"role": "user", "content": "q"})],
            ChatTemplateParams {
                add_generation_prompt: true,
                template_kwargs: kwargs,
                thinking,
                ..Default::default()
            },
        )
    }

    #[test]
    fn v41_architecture_selects_the_v41_renderer_with_thinking_on_by_default() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        assert_eq!(tokenizer.thinking_toggle(), ThinkingToggle::DefaultOn);
        assert_eq!(
            tokenizer.native_reasoning_effort_values(),
            &["low", "high", "xhigh", "max"]
        );
        assert!(tokenizer.think_in_prefill());
        let out = tokenizer
            .apply_chat_template(
                &[json!({"role": "user", "content": "q"})],
                ChatTemplateParams {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            out.starts_with("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}System\u{FF5C}>Reasoning Effort: 50 (range"),
            "{out}"
        );
        assert!(
            out.ends_with("<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}><think>"),
            "{out}"
        );
    }

    #[test]
    fn v41_model_type_alone_selects_the_v41_renderer() {
        // Some checkpoints identify themselves only through `model_type`,
        // with an unrelated `architectures` entry.
        let (_tmp, tok) = write_model_dir(
            None,
            Some(&["LlamaForCausalLM"]),
            Some("deepseek_v41"),
            None,
        );
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        assert_eq!(tokenizer.thinking_toggle(), ThinkingToggle::DefaultOn);
        assert_eq!(
            tokenizer.thinking_key_name(),
            Some(ThinkingKeyName::Thinking)
        );
        let out = render_v41_turn(&tokenizer, None, None).unwrap();
        assert_eq!(
            out,
            format!("{V41_BOS}{V41_DEFAULT_EFFORT_HEADER}{V41_THINKING_TAIL}")
        );
    }

    #[test]
    fn v41_reasoning_effort_none_disables_thinking_and_bad_values_error() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({"role": "user", "content": "q"})];

        let none_kwargs = HashMap::from([("reasoning_effort".to_string(), json!("none"))]);
        let off = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    template_kwargs: Some(&none_kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        // Chat mode, and no effort line at all.
        assert_eq!(
            off,
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}></think>"
        );

        let medium_kwargs = HashMap::from([("reasoning_effort".to_string(), json!("medium"))]);
        let err = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    template_kwargs: Some(&medium_kwargs),
                    ..Default::default()
                },
            )
            .expect_err("expected an error for reasoning_effort=medium")
            .to_string();
        assert!(err.contains("Invalid reasoning effort"), "{err}");
    }

    #[test]
    fn v41_integer_effort_strings_are_restored_to_budgets() {
        // The gateway deserialises a top-level `"reasoning_effort": 42` into
        // the string "42" before forwarding it as a template kwarg; the shim
        // must hand `parse_reasoning_effort` the integer budget again.
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("reasoning_effort".to_string(), json!("42"))]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.contains("Reasoning Effort: 42 (range"), "{out}");
        assert!(out.ends_with(V41_THINKING_TAIL), "{out}");
        // Out-of-range integers are still rejected by `parse_reasoning_effort`.
        for bad in ["0", "101"] {
            let kw = HashMap::from([("reasoning_effort".to_string(), json!(bad))]);
            let err = render_v41_turn(&tokenizer, Some(&kw), None)
                .expect_err(bad)
                .to_string();
            assert!(err.contains("Invalid reasoning effort"), "{bad}: {err}");
        }
        // Only the gateway's exact form, ASCII digits, is restored: a sign or
        // surrounding whitespace is not an integer budget and reaches the
        // strict parser as the string it is.
        for bad in ["+42", " 42 "] {
            let kw = HashMap::from([("reasoning_effort".to_string(), json!(bad))]);
            let err = render_v41_turn(&tokenizer, Some(&kw), None)
                .expect_err(bad)
                .to_string();
            assert_eq!(
                err,
                format!("DeepSeek V4.1 reasoning_effort invalid: Invalid reasoning effort `{bad}`: expected an integer within [1, 100] or one of low, high, xhigh, max"),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn v41_explicit_thinking_true_overrides_reasoning_effort_none() {
        // Deliberate divergence from vLLM Python, where "none" forces chat
        // mode: the gateway arms the reasoning parser from the explicit toggle
        // first, so the prompt must enter thinking mode as well. "none"
        // carries no effort level, so the default budget is rendered.
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([
            ("reasoning_effort".to_string(), json!("none")),
            ("thinking".to_string(), json!(true)),
        ]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.ends_with(V41_THINKING_TAIL), "{out}");
        assert!(out.contains("Reasoning Effort: 50 (range"), "{out}");
    }

    #[test]
    fn v41_enable_thinking_alone_is_ignored_until_the_gateway_learns_the_alias() {
        // The gateway reads only the key this tokenizer reports
        // (`thinking_key_name() == Thinking`), so `enable_thinking: false`
        // must not switch the prompt to chat mode while the parser stays
        // armed. vLLM's `enable_thinking` alias is scheduled for the
        // gateway-side task (Task 17); re-enable it in the shim together with
        // that change.
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("enable_thinking".to_string(), json!(false))]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.ends_with(V41_THINKING_TAIL), "{out}");
    }

    #[test]
    fn v41_non_boolean_thinking_kwarg_errors() {
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("thinking".to_string(), json!("yes"))]);
        let err = render_v41_turn(&tokenizer, Some(&kw), None)
            .expect_err("a non-boolean thinking kwarg must error")
            .to_string();
        assert!(err.contains("must be a boolean"), "{err}");
        // The message names the key and the offending value.
        assert!(err.contains("thinking") && err.contains("yes"), "{err}");
    }

    #[test]
    fn v41_non_boolean_drop_thinking_kwarg_errors() {
        // Same rule as `thinking`: present but not a JSON boolean is an error
        // naming the key and the value; JSON null counts as absent.
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("drop_thinking".to_string(), json!("false"))]);
        let err = render_v41_turn(&tokenizer, Some(&kw), None)
            .expect_err("a non-boolean drop_thinking kwarg must error")
            .to_string();
        assert_eq!(
            err,
            "DeepSeek V4.1: template_kwargs[\"drop_thinking\"] must be a boolean, got \"false\""
        );
        let kw = HashMap::from([("drop_thinking".to_string(), serde_json::Value::Null)]);
        assert_eq!(
            render_v41_turn(&tokenizer, Some(&kw), None).unwrap(),
            format!("{V41_BOS}{V41_DEFAULT_EFFORT_HEADER}{V41_THINKING_TAIL}")
        );
    }

    #[test]
    fn v41_params_thinking_false_renders_chat_mode() {
        // The gateway projects a top-level `reasoning_effort` of
        // `none`/`minimal` onto `params.thinking = Some(false)`.
        let (_tmp, tokenizer) = v41_tokenizer();
        let out = render_v41_turn(&tokenizer, None, Some(false)).unwrap();
        assert_eq!(out, format!("{V41_BOS}{V41_CHAT_TAIL}"));
    }

    #[test]
    fn v41_drop_thinking_false_keeps_historical_reasoning() {
        let (_tmp, tokenizer) = v41_tokenizer();
        let messages = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "reasoning_content": "r1", "content": "a1"}),
            json!({"role": "user", "content": "q2"}),
        ];
        let render = |kwargs: Option<&HashMap<String, serde_json::Value>>| {
            tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        add_generation_prompt: true,
                        template_kwargs: kwargs,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        // Default (`drop_thinking: true`): reasoning before the last user
        // turn is dropped.
        assert!(!render(None).contains("r1"));
        let kw = HashMap::from([("drop_thinking".to_string(), json!(false))]);
        let out = render(Some(&kw));
        assert_eq!(
            out,
            format!(
                "{V41_BOS}{V41_DEFAULT_EFFORT_HEADER}<\u{FF5C}User\u{FF5C}>q1<\u{FF5C}Assistant\u{FF5C}><think>r1</think>a1{V41_EOS}<\u{FF5C}User\u{FF5C}>q2<\u{FF5C}Assistant\u{FF5C}><think>"
            )
        );
    }

    #[test]
    fn v41_tools_attach_to_a_mid_conversation_system_message() {
        // vLLM's V4.1 rule: tools attach to the FIRST system message wherever
        // it is, even when the conversation opens with a user turn — unlike
        // V3.2/V4, which only rewrite a *leading* system/developer message.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
            json!({"role": "system", "content": "S"}),
            json!({"role": "user", "content": "q"}),
        ];
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
            add_generation_prompt: true,
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("get_weather"), "tool name missing: {out}");
        // Tools must land on the existing mid-conversation system message's
        // own content, not on a synthesized leading one.
        assert!(
            out.contains("S\n\n## Tools"),
            "tools not attached to the mid-conversation system message: {out}"
        );
        assert_eq!(out.matches("## Tools").count(), 1, "{out}");
    }

    #[test]
    fn v41_typed_function_without_optional_fields_renders_like_raw_json() {
        // D5: `openai_protocol::common::Function` skips its absent optional
        // fields (`description`, `strict`) when serialised, so a typed tool
        // reaches the schema block with the same bytes as the raw JSON a
        // client sends without those keys.
        let (_tmp, tokenizer) = v41_tokenizer();
        let parameters = json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        });
        let typed = serde_json::to_value(Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "lookup".to_string(),
                description: None,
                parameters: parameters.clone(),
                strict: None,
            },
        })
        .unwrap();
        let raw = json!({
            "type": "function",
            "function": { "name": "lookup", "parameters": parameters }
        });
        let render = |tool: serde_json::Value| {
            let tools = [tool];
            tokenizer
                .apply_chat_template(
                    &[json!({"role": "user", "content": "q"})],
                    ChatTemplateParams {
                        add_generation_prompt: true,
                        tools: Some(&tools),
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let from_typed = render(typed);
        assert_eq!(from_typed, render(raw));
        assert!(
            from_typed.contains(
                "\n\n{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}}, \"required\": [\"query\"]}}\n\n"
            ),
            "{from_typed}"
        );
    }

    #[test]
    fn v41_add_generation_prompt_false_continues_the_final_assistant_message() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![
            json!({"role": "user", "content": "q"}),
            json!({"role": "assistant", "reasoning_content": "r", "content": "Sure,"}),
        ];
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: false,
                    ..Default::default()
                },
            )
            .unwrap();
        // Thinking mode (default ON), reasoning kept (the assistant turn is
        // after the last user turn), and no trailing EOS since
        // `add_generation_prompt: false` continues the final message.
        assert_eq!(
            out,
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}System\u{FF5C}>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}><think>r</think>Sure,"
        );
    }

    #[test]
    fn v41_chat_template_content_format_is_openai_v4_stays_string() {
        let (_tmp, tok41) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer41 = HuggingFaceTokenizer::from_file(&tok41).unwrap();
        assert_eq!(
            tokenizer41.chat_template_content_format(),
            ChatTemplateContentFormat::OpenAI
        );

        let (_tmp, tok4) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer4 = HuggingFaceTokenizer::from_file(&tok4).unwrap();
        assert_eq!(
            tokenizer4.chat_template_content_format(),
            ChatTemplateContentFormat::String
        );
    }
}
