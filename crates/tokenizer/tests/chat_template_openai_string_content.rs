//! String message content under an "openai"-format chat template renders
//! through the template's parts branch, as it does on vLLM's own server.
//!
//! vLLM hands a template it detects as "openai" format every message's string
//! content as a one-item text part list, so such a template takes its parts
//! branch for a plain string too. A template whose two branches differ (a
//! separator after every part, a truthiness check on the content) renders a
//! different prompt for the same request unless the gateway does the same, and
//! the engine then sees one prompt from the gateway and another from its own
//! front end: one token apart per system turn on the Gemma 4 template.
//!
//! The always-on cases use two small templates with the branch shapes found in
//! the wild. With `GEMMA4_MODEL_DIR` (a Gemma 4 checkpoint's `tokenizer.json`,
//! `tokenizer_config.json` and `chat_template.jinja`) the prompts and ids of
//! `tests/fixtures/gemma4/render_ids_fixtures.json`, recorded from the engine's
//! own `/tokenize`, are checked as well; without it that case prints a skip
//! notice and passes.

use std::{collections::HashMap, path::PathBuf};

use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams, ChatTemplateState},
    huggingface::HuggingFaceTokenizer,
    traits::{Encoder, Tokenizer},
};
use serde::Deserialize;
use serde_json::{json, Value};

/// Each part is followed by a space, a string is only trimmed: the first
/// message's branches of the Gemma 4 template.
const PART_SEPARATOR_TEMPLATE: &str = r"{%- for message in messages -%}
{{- '<|' + message['role'] + '|>' -}}
{%- if message['content'] is string -%}
{{- message['content'] | trim -}}
{%- elif message['content'] is sequence -%}
{%- for item in message['content'] -%}
{{- item['text'] | trim + ' ' -}}
{%- endfor -%}
{%- endif -%}
{{- '<|end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}";

/// A truthiness check on the first message's content decides between the
/// request's system prompt and a default one: an empty string is false, a
/// one-item part list is not (the MiniMax templates).
const TRUTHINESS_TEMPLATE: &str = r"{%- if messages[0]['role'] == 'system' -%}
{%- if messages[0]['content'] -%}
{{- '<|system|>' -}}
{%- if messages[0]['content'] is string -%}
{{- messages[0]['content'] -}}
{%- else -%}
{%- for item in messages[0]['content'] -%}{{- item['text'] -}}{%- endfor -%}
{%- endif -%}
{{- '\n' -}}
{%- else -%}
{{- '<|system|>default\n' -}}
{%- endif -%}
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] != 'system' -%}
{{- '<|' + message['role'] + '|>' -}}
{%- if message['content'] is string -%}
{{- message['content'] -}}
{%- else -%}
{%- for item in message['content'] -%}{{- item['text'] -}}{%- endfor -%}
{%- endif -%}
{{- '\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}";

/// Render through a template detected as the parts format, with the
/// generation prompt, like the gateway's chat path.
#[expect(clippy::expect_used, reason = "test helper — panics are intentional")]
fn render_openai(template: &str, messages: &[Value]) -> String {
    let state = ChatTemplateState::new(Some(template.to_string())).expect("template parses");
    assert_eq!(
        state.content_format(),
        ChatTemplateContentFormat::OpenAI,
        "the template must be detected as the parts format"
    );
    state
        .apply(
            messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .expect("template renders")
}

#[test]
fn string_content_takes_the_parts_branch_of_an_openai_template() {
    let as_string = [
        json!({"role": "system", "content": "You are terse."}),
        json!({"role": "user", "content": "Hi"}),
    ];
    let as_parts = [
        json!({"role": "system", "content": [{"type": "text", "text": "You are terse."}]}),
        json!({"role": "user", "content": [{"type": "text", "text": "Hi"}]}),
    ];
    let expected = "<|system|>You are terse. <|end|>\n<|user|>Hi <|end|>\n<|assistant|>";
    assert_eq!(render_openai(PART_SEPARATOR_TEMPLATE, &as_parts), expected);
    assert_eq!(render_openai(PART_SEPARATOR_TEMPLATE, &as_string), expected);
}

#[test]
fn an_empty_string_is_a_one_item_part_list_to_an_openai_template() {
    let empty = [
        json!({"role": "system", "content": ""}),
        json!({"role": "user", "content": "Hi"}),
    ];
    assert_eq!(
        render_openai(TRUTHINESS_TEMPLATE, &empty),
        "<|system|>\n<|user|>Hi\n<|assistant|>"
    );
    let none = [json!({"role": "user", "content": "Hi"})];
    assert_eq!(
        render_openai(TRUTHINESS_TEMPLATE, &none),
        "<|user|>Hi\n<|assistant|>"
    );
}

#[test]
fn a_tool_result_stays_a_string() {
    // vLLM joins a tool message's text parts back into one string, so a
    // template may concatenate it.
    const TEMPLATE: &str = r"{%- for message in messages -%}
{%- if message['role'] == 'tool' -%}
{{- 'tool:' + message['content'] + '\n' -}}
{%- else -%}
{%- for item in message['content'] -%}{{- item['text'] -}}{%- endfor -%}
{{- '\n' -}}
{%- endif -%}
{%- endfor -%}";
    let messages = [
        json!({"role": "user", "content": "Hi"}),
        json!({"role": "tool", "tool_call_id": "c1", "content": "22"}),
    ];
    assert_eq!(render_openai(TEMPLATE, &messages), "Hi\ntool:22\n");
}

#[test]
fn a_string_format_template_keeps_string_content() {
    const TEMPLATE: &str = r"{%- for message in messages -%}{{- message['role'] + ': ' + message['content'] + '\n' -}}{%- endfor -%}";
    let state = ChatTemplateState::new(Some(TEMPLATE.to_string())).expect("template parses");
    assert_eq!(state.content_format(), ChatTemplateContentFormat::String);
    let messages = [json!({"role": "user", "content": "Hi"})];
    assert_eq!(
        state
            .apply(&messages, ChatTemplateParams::default())
            .expect("template renders"),
        "user: Hi\n"
    );
}

/// One recorded request: the engine's prompt token ids for it and the prompt
/// they spell.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    messages: Vec<Value>,
    chat_template_kwargs: HashMap<String, Value>,
    text: String,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Fixtures {
    #[expect(dead_code, reason = "the fixture explains itself in the file")]
    description: String,
    cases: Vec<Case>,
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn gemma4_string_system_turns_render_the_engines_prompt() {
    let Some(dir) = std::env::var_os("GEMMA4_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("skipping: GEMMA4_MODEL_DIR unset (the recorded Gemma 4 prompts need the checkpoint's tokenizer and template)");
        return;
    };
    let tokenizer_path = dir.join("tokenizer.json");
    let template_path = dir.join("chat_template.jinja");
    let tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
        tokenizer_path
            .to_str()
            .expect("tokenizer path must be UTF-8"),
        Some(template_path.to_str().expect("template path must be UTF-8")),
    )
    .expect("the Gemma 4 tokenizer should load");
    assert_eq!(
        tokenizer.chat_template_content_format(),
        ChatTemplateContentFormat::OpenAI
    );
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("fixtures/gemma4/render_ids_fixtures.json"))
            .expect("render_ids_fixtures.json must match the Case schema");
    assert!(!fixtures.cases.is_empty());
    for case in &fixtures.cases {
        let name = &case.name;
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            template_kwargs: Some(&case.chat_template_kwargs),
            ..Default::default()
        };
        let text = tokenizer
            .apply_chat_template(&case.messages, params)
            .unwrap_or_else(|e| panic!("case {name}: render failed: {e}"));
        assert_eq!(
            text, case.text,
            "case {name}: prompt differs from the engine's"
        );
        let ids = tokenizer
            .encode(&text, false)
            .unwrap_or_else(|e| panic!("case {name}: encode failed: {e}"));
        assert_eq!(
            ids.token_ids(),
            case.ids.as_slice(),
            "case {name}: token ids differ from the engine's /tokenize"
        );
    }
}
