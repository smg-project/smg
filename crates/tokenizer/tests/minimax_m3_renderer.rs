//! Render parity of the `root` role with the checkpoint's own chat template.
//!
//! The chat template of the models the MiniMax provider profile targets reads
//! a leading `root` message into its high-priority system slot and a `system`
//! or `developer` message right after it into its developer slot; its role
//! mapping, quoted from `tests/fixtures/minimax_m3/chat_template.jinja`:
//!
//! ```jinja
//! {#- Role mapping: root -> system sp (high priority), system/developer -> developer sp (low priority) -#}
//! {%- if messages and messages[0].role == "root" -%}
//!     {%- set system_message = messages[0] -%}
//!     {%- set conversation_messages = messages[1:] -%}
//!     {%- if conversation_messages and conversation_messages[0].role in ["system", "developer"] -%}
//!         {%- set developer_message = conversation_messages[0] -%}
//!         {%- set conversation_messages = conversation_messages[1:] -%}
//!     {%- endif -%}
//! {%- elif messages and messages[0].role in ["system", "developer"] -%}
//!     {%- set developer_message = messages[0] -%}
//!     {%- set conversation_messages = messages[1:] -%}
//! {%- endif -%}
//! ```
//!
//! A root message that reaches the template as a `system` message lands in
//! the developer slot under the template's default identity, and a system
//! message after it is dropped, since the template reads one leading system
//! or developer message.
//!
//! Every case in `tests/fixtures/minimax_m3/render_fixtures.json` is a chat
//! request as a client sends it, the prompt the checkpoint's own template
//! renders for it through the reference implementation (the renderer of the
//! engine's own server), and the text's ids under the checkpoint's tokenizer.
//! Each request is normalized under its provider profile the way
//! `ValidatedJson` does before the gateway's chat path renders it, its
//! messages are serialized the way that path hands them to the template, and
//! the result is rendered: the text must match (checked with a minimal
//! tokenizer) and so must the ids (checked with the real one:
//! `MINIMAX_M3_MODEL_DIR` or a one-time download into
//! `.tokenizer_cache/minimax_m3/`; offline, the id check prints a skip notice
//! and passes).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "test code: panics and the skip notice are intentional"
)]

use std::{
    fs,
    path::{Path, PathBuf},
};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    huggingface::HuggingFaceTokenizer,
    traits::{Encoder, Tokenizer as TokenizerTrait},
};
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
use serde::Deserialize;
use serde_json::{json, Value};
use tempfile::TempDir;

mod common;

const RENDER_FIXTURES: &str = include_str!("fixtures/minimax_m3/render_fixtures.json");
const TEMPLATE: &str = include_str!("fixtures/minimax_m3/chat_template.jinja");

/// The model id the requests are normalized under: it selects the MiniMax
/// provider profile, the one that accepts the `root` role.
const MODEL: &str = "MiniMax-M3";

/// A tokenizer.json that loads; the text check never calls into it.
const MIN_TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "BPE", "vocab": { "hello": 0 }, "merges": [] }
}"#;

/// One request as a client sends it, with the template's prompt for it.
/// Unknown fields are rejected so a regenerated fixture carrying a new knob
/// fails here instead of rendering with it ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    messages: Vec<Value>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    text: String,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Fixtures {
    #[expect(dead_code, reason = "the fixture explains itself in the file")]
    description: String,
    cases: Vec<Case>,
}

fn fixtures() -> Fixtures {
    let fixtures: Fixtures = serde_json::from_str(RENDER_FIXTURES)
        .expect("render_fixtures.json must match the Case schema");
    assert!(
        !fixtures.cases.is_empty(),
        "render_fixtures.json holds no cases"
    );
    fixtures
}

fn fixture_template_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minimax_m3/chat_template.jinja")
}

/// A tokenizer that renders with the fixture template, whatever template the
/// tokenizer's own directory carries.
fn tokenizer_with_fixture_template(tokenizer_json: &Path) -> HuggingFaceTokenizer {
    HuggingFaceTokenizer::from_file_with_chat_template(
        tokenizer_json.to_str().expect("UTF-8 path"),
        Some(fixture_template_path().to_str().expect("UTF-8 path")),
    )
    .expect("the tokenizer with the fixture template should load")
}

/// The case's messages as the gateway hands them to the template: the
/// request deserialized and normalized under its provider profile (what
/// `ValidatedJson` does before the chat path runs), then every message
/// serialized the way the chat path does.
fn messages_as_rendered(case: &Case) -> Vec<Value> {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": case.messages,
        "tools": case.tools,
    }))
    .unwrap_or_else(|e| {
        panic!(
            "case {}: the gateway does not accept this request: {e}",
            case.name
        )
    });
    request.normalize();
    request
        .messages
        .iter()
        .map(|message| serde_json::to_value(message).expect("a chat message serializes"))
        .collect()
}

fn render(tok: &HuggingFaceTokenizer, case: &Case) -> String {
    let messages = messages_as_rendered(case);
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: case.tools.as_deref(),
        ..Default::default()
    };
    tok.apply_chat_template(&messages, params)
        .unwrap_or_else(|e| panic!("case {}: render failed: {e}", case.name))
}

/// The text of the system or developer message that follows a leading root
/// in the request, when there is one: the template renders it in its
/// developer slot, so it has to be in the prompt.
fn message_after_root(case: &Case) -> Option<&str> {
    let role = |index: usize| -> Option<&str> { case.messages.get(index)?.get("role")?.as_str() };
    if role(0) != Some("root") || !matches!(role(1), Some("system" | "developer")) {
        return None;
    }
    case.messages.get(1)?.get("content")?.as_str()
}

/// The template under test keeps its root branch: a regenerated fixture
/// without it would make the parity below meaningless.
#[test]
fn the_fixture_template_reads_a_leading_root_into_its_system_slot() {
    for line in [
        "{#- Role mapping: root -> system sp (high priority), system/developer -> developer sp (low priority) -#}",
        "{%- if messages and messages[0].role == \"root\" -%}",
        "{%- if conversation_messages and conversation_messages[0].role in [\"system\", \"developer\"] -%}",
    ] {
        assert!(
            TEMPLATE.contains(line),
            "the fixture template has no line {line:?}"
        );
    }
}

/// The rendered text equals the template's own prompt, case by case, with
/// the system or developer message after a root in it.
#[test]
fn minimax_m3_render_text_matches_the_templates_prompts() {
    let temp = TempDir::new().expect("temp dir");
    let tokenizer_json = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_json, MIN_TOKENIZER_JSON).expect("tokenizer.json");
    let tok = tokenizer_with_fixture_template(&tokenizer_json);
    for case in &fixtures().cases {
        let text = render(&tok, case);
        if let Some(following) = message_after_root(case) {
            assert!(
                text.contains(following),
                "case {}: the message after the root is not in the prompt:\n{text}",
                case.name
            );
        }
        assert_eq!(
            text, case.text,
            "case {}: prompt differs from the template's own rendering",
            case.name
        );
    }
}

/// With the checkpoint's tokenizer, the rendered text encodes to the ids of
/// the template's own prompt.
#[test]
fn minimax_m3_render_ids_match_the_templates_prompts() {
    let Some(model_dir) = common::ensure_minimax_m3_cached() else {
        eprintln!(
            "skipping: no MiniMax-M3 tokenizer (set MINIMAX_M3_MODEL_DIR or allow the download)"
        );
        return;
    };
    let tok = tokenizer_with_fixture_template(&model_dir.join("tokenizer.json"));
    for case in &fixtures().cases {
        let name = &case.name;
        let text = render(&tok, case);
        assert_eq!(
            text, case.text,
            "case {name}: prompt differs from the template's own rendering"
        );
        let encoded = tok
            .encode(&text, false)
            .unwrap_or_else(|e| panic!("case {name}: encode failed: {e}"));
        assert_eq!(
            encoded.token_ids(),
            case.ids.as_slice(),
            "case {name}: token ids differ from the template's own prompt"
        );
    }
}
