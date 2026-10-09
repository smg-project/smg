//! `continue_final_message` on the Jinja renderer: the conversation is
//! rendered with its final message and cut right after that message's text,
//! as transformers' `apply_chat_template(continue_final_message=True)` does.
//! The expected strings are what transformers 5.17 renders for the same
//! templates and messages.

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use llm_tokenizer::{
        chat_template::{ChatTemplateParams, ChatTemplateProcessor},
        huggingface::HuggingFaceTokenizer,
        TokenizerTrait,
    };
    use serde_json::{json, Value};
    use tempfile::TempDir;

    /// An assistant turn opens with a recipient header the generation prompt
    /// does not carry.
    const HEADER: &str = r"
{%- for m in messages -%}
{%- if m.role == 'assistant' -%}
{{- '<|turn|>assistant<|to|>user<|body|>' + m.content + '<|end|>' -}}
{%- else -%}
{{- '<|turn|>' + m.role + '<|body|>' + m.content + '<|end|>' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|turn|>assistant' -}}{%- endif -%}";

    /// The generation prompt opens an empty thought block unless thinking is on,
    /// which no rendered turn has; turns trim their content.
    const THOUGHT: &str = r"
{%- for m in messages -%}
{{- '<|turn|>' + m.role + '\n' + (m.content | trim) + '<|end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn|>assistant\n' -}}
{%- if not enable_thinking -%}{{- '<think></think>\n' -}}{%- endif -%}
{%- endif -%}";

    /// An assistant turn renders its reasoning ahead of the content.
    const REASONING: &str = r"
{%- for m in messages -%}
{%- if m.role == 'assistant' -%}
{{- '<|im_start|>assistant\n<think>\n' + (m.reasoning_content or '') + '\n</think>\n\n' + m.content + '<|im_end|>\n' -}}
{%- else -%}
{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}";

    /// An assistant turn is the generation prompt followed by the content.
    const PLAIN: &str = r"
{%- for m in messages -%}
{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}";

    /// Content is a string or a list of parts.
    const PARTS: &str = r"
{%- for m in messages -%}
{{- '<|' + m.role + '|>' -}}
{%- if m.content is string -%}{{- m.content -}}
{%- else -%}{%- for p in m.content -%}{%- if p.type == 'text' -%}{{- p.text -}}{%- endif -%}{%- endfor -%}
{%- endif -%}
{{- '<|end|>' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}";

    /// Assistant turns are never rendered; only the generation prompt opens one.
    const NO_ASSISTANT_TURNS: &str = r"
{%- for m in messages -%}
{%- if m.role != 'assistant' -%}{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}";

    fn render(
        template: &str,
        messages: &[Value],
        add_generation_prompt: bool,
        continue_final_message: bool,
        kwargs: Option<&HashMap<String, Value>>,
    ) -> anyhow::Result<String> {
        ChatTemplateProcessor::new(template.to_string())?.apply_chat_template(
            messages,
            ChatTemplateParams {
                add_generation_prompt,
                continue_final_message,
                template_kwargs: kwargs,
                ..Default::default()
            },
        )
    }

    fn continue_final(template: &str, messages: &[Value]) -> String {
        render(template, messages, false, true, None).unwrap()
    }

    /// A user turn, then `assistant` as the message to continue.
    fn chat(assistant: Value) -> Vec<Value> {
        let mut assistant = assistant;
        assistant["role"] = json!("assistant");
        vec![json!({"role": "user", "content": "Hi"}), assistant]
    }

    /// What the gateway rendered for a continued message before: the other
    /// messages with a generation prompt, then the message's text.
    fn prompt_then_text(template: &str, messages: &[Value], text: &str) -> String {
        let (_, head) = messages.split_last().unwrap();
        render(template, head, true, false, None).unwrap() + text
    }

    #[test]
    fn keeps_the_header_the_assistant_turn_opens_with() {
        assert_eq!(
            continue_final(HEADER, &chat(json!({"content": "The weather"}))),
            "<|turn|>user<|body|>Hi<|end|><|turn|>assistant<|to|>user<|body|>The weather"
        );
        // Text the template keeps as written is continued as written.
        assert_eq!(
            continue_final(HEADER, &chat(json!({"content": "Cafe\u{301} 날씨 😀 "}))),
            "<|turn|>user<|body|>Hi<|end|><|turn|>assistant<|to|>user<|body|>Cafe\u{301} 날씨 😀 "
        );
    }

    #[test]
    fn leaves_out_what_only_the_generation_prompt_opens() {
        let thinking_off = HashMap::from([("enable_thinking".to_string(), json!(false))]);
        let messages = chat(json!({"content": "The weather"}));
        assert_eq!(
            render(THOUGHT, &messages, false, true, Some(&thinking_off)).unwrap(),
            "<|turn|>user\nHi<|end|>\n<|turn|>assistant\nThe weather"
        );
        // A template that trims the content also trims the marker's space, so
        // the trailing whitespace goes too; with no text that reaches back into
        // the turn header.
        assert_eq!(
            continue_final(THOUGHT, &chat(json!({"content": "The weather is "}))),
            "<|turn|>user\nHi<|end|>\n<|turn|>assistant\nThe weather is"
        );
        assert_eq!(
            continue_final(THOUGHT, &chat(json!({"content": ""}))),
            "<|turn|>user\nHi<|end|>\n<|turn|>assistant"
        );
    }

    #[test]
    fn keeps_the_reasoning_the_turn_renders() {
        let messages =
            chat(json!({"content": "The weather", "reasoning_content": "Asked about it."}));
        assert_eq!(
        continue_final(REASONING, &messages),
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nAsked about it.\n</think>\n\nThe weather"
    );
    }

    #[test]
    fn matches_the_generation_prompt_where_the_turn_is_the_prompt_and_the_content() {
        let messages = chat(json!({"content": "The weather"}));
        assert_eq!(
            continue_final(PLAIN, &messages),
            prompt_then_text(PLAIN, &messages, "The weather")
        );
        // The cut is at the last marker, so text repeated earlier in the chat or
        // the marker's own words in the message do not move it.
        let repeated = vec![
            json!({"role": "user", "content": "The weather"}),
            json!({"role": "assistant", "content": "The weather"}),
        ];
        assert_eq!(
            continue_final(PLAIN, &repeated),
            "<|im_start|>user\nThe weather<|im_end|>\n<|im_start|>assistant\nThe weather"
        );
        let tag = chat(json!({"content": "Say CONTINUE_FINAL_MESSAGE_TAG then"}));
        assert_eq!(
        continue_final(PLAIN, &tag),
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nSay CONTINUE_FINAL_MESSAGE_TAG then"
    );
    }

    #[test]
    fn continues_the_last_text_part() {
        let parts = chat(json!({"content": [
            {"type": "text", "text": "The "},
            {"type": "image"},
            {"type": "text", "text": "weather"}
        ]}));
        assert_eq!(
            continue_final(PARTS, &parts),
            "<|user|>Hi<|end|><|assistant|>The weather"
        );
    }

    /// Where transformers raises (the text never reaches the prompt, or there is
    /// no text to continue) the text follows the generation prompt, as it did
    /// before continuation was rendered by the template.
    #[test]
    fn appends_after_the_generation_prompt_where_transformers_raises() {
        let prefill = chat(json!({"content": "language English"}));
        assert_eq!(
            continue_final(NO_ASSISTANT_TURNS, &prefill),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nlanguage English"
        );
        for (template, assistant) in [
            (PLAIN, json!({"content": null})),
            (PARTS, json!({"content": [{"type": "image"}]})),
        ] {
            let messages = chat(assistant);
            assert_eq!(
                continue_final(template, &messages),
                prompt_then_text(template, &messages, "")
            );
        }

        let both = render(
            PLAIN,
            &chat(json!({"content": "The weather"})),
            true,
            true,
            None,
        );
        assert!(both.is_err());
    }

    /// A tokenizer with a Jinja template declares native continuation and
    /// continues through its template when asked to, and says whether it did:
    /// where the template cannot continue the message, the text follows the
    /// generation prompt instead.
    #[test]
    fn a_jinja_tokenizer_continues_through_its_template() {
        let dir = TempDir::new().unwrap();
        let tokenizer_path = dir.path().join("tokenizer.json");
        let template_path = dir.path().join("chat_template.jinja");
        fs::write(
            &tokenizer_path,
            r#"{"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}"#,
        )
        .unwrap();
        fs::write(&template_path, HEADER).unwrap();
        let mut tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
            tokenizer_path.to_str().unwrap(),
            template_path.to_str(),
        )
        .unwrap();

        assert!(
            tokenizer
                .renderer_capabilities()
                .native_assistant_continuation
        );
        let continue_final = |tokenizer: &HuggingFaceTokenizer| {
            tokenizer
                .apply_chat_template_with_encoding(
                    &chat(json!({"content": "The weather"})),
                    ChatTemplateParams {
                        continue_final_message: true,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        let rendered = continue_final(&tokenizer);
        assert_eq!(
            rendered.text,
            "<|turn|>user<|body|>Hi<|end|><|turn|>assistant<|to|>user<|body|>The weather"
        );
        assert!(rendered.continued_final_message);

        tokenizer
            .set_chat_template(NO_ASSISTANT_TURNS.to_string())
            .unwrap();
        let rendered = continue_final(&tokenizer);
        assert_eq!(
            rendered.text,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nThe weather"
        );
        assert!(!rendered.continued_final_message);
    }
}
