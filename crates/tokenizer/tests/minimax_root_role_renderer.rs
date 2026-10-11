//! Render parity of the `root` role with the checkpoints' own chat templates.
//!
//! The chat template of the newest MiniMax checkpoint reads a leading `root`
//! message into its high-priority system slot and a `system` or `developer`
//! message right after it into its developer slot; its role mapping, quoted
//! from `tests/fixtures/minimax_m3/chat_template.jinja`:
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
//! A root message that reaches that template as a `system` message lands in
//! the developer slot under the template's default identity, and a system
//! message after it is dropped, since the template reads one leading system
//! or developer message. The templates of the earlier checkpoints have no
//! root branch at all; `tests/fixtures/minimax_m2/chat_template.jinja`:
//!
//! ```jinja
//! {#- Extract system message (only first message if it's system) -#}
//! {%- set system_message = none -%}
//! {%- set conversation_messages = messages -%}
//! {%- if messages and messages[0].role == "system" -%}
//!     {%- set system_message = messages[0] -%}
//!     {%- set conversation_messages = messages[1:] -%}
//! {%- endif -%}
//! ```
//!
//! and their message loop has user, assistant and tool branches only, so a
//! root message handed to them as is would be dropped: the renderer hands
//! such a template every root message as a system message, and merges the
//! system messages at the front, as it does for developer messages.
//!
//! Every case in the two `render_fixtures.json` files is a chat request as a
//! client sends it, the prompt the checkpoint's own template renders for it
//! through the reference implementation (the renderer of the engine's own
//! server), and the text's ids under the checkpoint's tokenizer; a case that
//! the gateway reshapes before the render (several roots merged into one,
//! roots handed to a template without the branch as system messages) also
//! carries `rendered_from`, the list the template was given. Each request is
//! normalized under its provider profile the way `ValidatedJson` does before
//! the gateway's chat path renders it, its messages are serialized the way
//! that path hands them to the template, and the result is rendered: the
//! text must match (checked with a minimal tokenizer) and so must the ids
//! for the template with the root branch (checked with the real one:
//! `MINIMAX_M3_MODEL_DIR` or a one-time download into
//! `.tokenizer_cache/minimax_m3/`; without it the id check prints a skip
//! notice and passes); the cases of the other template record the text only.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "test code: panics and the skip notices are intentional"
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

/// The template with the root branch and its cases.
const M3_TEMPLATE_PATH: &str = "tests/fixtures/minimax_m3/chat_template.jinja";
const M3_TEMPLATE: &str = include_str!("fixtures/minimax_m3/chat_template.jinja");
const M3_RENDER_FIXTURES: &str = include_str!("fixtures/minimax_m3/render_fixtures.json");
/// The template without a root branch and its cases.
const M2_TEMPLATE_PATH: &str = "tests/fixtures/minimax_m2/chat_template.jinja";
const M2_TEMPLATE: &str = include_str!("fixtures/minimax_m2/chat_template.jinja");
const M2_RENDER_FIXTURES: &str = include_str!("fixtures/minimax_m2/render_fixtures.json");

/// The model id the requests are normalized under: it selects the MiniMax
/// provider profile, the one that accepts the `root` role, for both
/// templates (the profile goes by the vendor's name, not the checkpoint).
const MODEL: &str = "MiniMax-M3";

/// A tokenizer.json that loads; the text checks never call into it.
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
    /// The list the template was given when the gateway reshapes the
    /// request before the render; absent when it renders the request as is.
    #[serde(default)]
    rendered_from: Option<Vec<Value>>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    text: String,
    /// The text's ids under the checkpoint's tokenizer; the cases of the
    /// template without a root branch record the text only.
    #[serde(default)]
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Fixtures {
    #[expect(dead_code, reason = "the fixture explains itself in the file")]
    description: String,
    cases: Vec<Case>,
}

fn fixtures(raw: &str) -> Fixtures {
    let fixtures: Fixtures =
        serde_json::from_str(raw).expect("render_fixtures.json must match the Case schema");
    assert!(
        !fixtures.cases.is_empty(),
        "render_fixtures.json holds no cases"
    );
    fixtures
}

fn fixture_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// A tokenizer that renders with the given fixture template, whatever
/// template the tokenizer's own directory carries.
fn tokenizer_with_template(tokenizer_json: &Path, template: &str) -> HuggingFaceTokenizer {
    HuggingFaceTokenizer::from_file_with_chat_template(
        tokenizer_json.to_str().expect("UTF-8 path"),
        Some(fixture_path(template).to_str().expect("UTF-8 path")),
    )
    .expect("the tokenizer with the fixture template should load")
}

/// A minimal tokenizer rendering with the given fixture template.
fn minimal_tokenizer_with_template(temp: &TempDir, template: &str) -> HuggingFaceTokenizer {
    let tokenizer_json = temp.path().join("tokenizer.json");
    fs::write(&tokenizer_json, MIN_TOKENIZER_JSON).expect("tokenizer.json");
    tokenizer_with_template(&tokenizer_json, template)
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

fn params(case: &Case) -> ChatTemplateParams<'_> {
    ChatTemplateParams {
        add_generation_prompt: true,
        tools: case.tools.as_deref(),
        ..Default::default()
    }
}

fn render(tok: &HuggingFaceTokenizer, case: &Case) -> String {
    let messages = messages_as_rendered(case);
    tok.apply_chat_template(&messages, params(case))
        .unwrap_or_else(|e| panic!("case {}: render failed: {e}", case.name))
}

/// The texts of the request's leading instruction messages (root, system
/// and developer messages before the first turn): whatever slot the template
/// gives them, every one of them has to be in the prompt.
fn leading_instruction_texts(case: &Case) -> Vec<&str> {
    case.messages
        .iter()
        .take_while(|m| {
            matches!(
                m.get("role").and_then(Value::as_str),
                Some("root" | "system" | "developer")
            )
        })
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .collect()
}

/// The case's text and ids against the template rendered with the given
/// tokenizer: the request through the profile and the renderer, and, for a
/// reshaped request, the recorded list straight through the template too.
fn check_case(tok: &HuggingFaceTokenizer, case: &Case, check_ids: bool) {
    let name = &case.name;
    let text = render(tok, case);
    for instruction in leading_instruction_texts(case) {
        assert!(
            text.contains(instruction),
            "case {name}: a leading instruction is not in the prompt: {instruction:?}\n{text}"
        );
    }
    assert_eq!(
        text, case.text,
        "case {name}: prompt differs from the template's own rendering"
    );
    if let Some(rendered_from) = &case.rendered_from {
        let direct = tok
            .apply_chat_template(rendered_from, params(case))
            .unwrap_or_else(|e| panic!("case {name}: render of rendered_from failed: {e}"));
        assert_eq!(
            direct, case.text,
            "case {name}: rendered_from does not render to the recorded prompt"
        );
    }
    if check_ids {
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

/// The templates under test keep the shapes the fixtures were recorded
/// from: a regenerated fixture that lost them would make the parity below
/// meaningless.
#[test]
fn the_fixture_templates_keep_their_role_mappings() {
    for line in [
        "{#- Role mapping: root -> system sp (high priority), system/developer -> developer sp (low priority) -#}",
        "{%- if messages and messages[0].role == \"root\" -%}",
        "{%- if conversation_messages and conversation_messages[0].role in [\"system\", \"developer\"] -%}",
    ] {
        assert!(
            M3_TEMPLATE.contains(line),
            "the template with the root branch has no line {line:?}"
        );
    }
    assert!(
        !M2_TEMPLATE.contains("root"),
        "the template without a root branch mentions the role"
    );
    for line in [
        "{#- Extract system message (only first message if it's system) -#}",
        "{%- if messages and messages[0].role == \"system\" -%}",
    ] {
        assert!(
            M2_TEMPLATE.contains(line),
            "the template without a root branch has no line {line:?}"
        );
    }
}

/// Through the template with the root branch, the rendered text equals the
/// template's own prompt, case by case, with every leading instruction in it.
#[test]
fn root_messages_render_as_the_template_with_the_root_branch_does() {
    let temp = TempDir::new().expect("temp dir");
    let tok = minimal_tokenizer_with_template(&temp, M3_TEMPLATE_PATH);
    for case in &fixtures(M3_RENDER_FIXTURES).cases {
        check_case(&tok, case, false);
    }
}

/// Through a template without a root branch, a root message renders as the
/// system message, a system message after it merged into that one, and the
/// text equals the template's own prompt for that list.
#[test]
fn root_messages_render_in_the_system_slot_of_a_template_without_a_root_branch() {
    let temp = TempDir::new().expect("temp dir");
    let tok = minimal_tokenizer_with_template(&temp, M2_TEMPLATE_PATH);
    for case in &fixtures(M2_RENDER_FIXTURES).cases {
        check_case(&tok, case, false);
    }
}

/// With the checkpoint's tokenizer, the rendered text encodes to the ids of
/// the template's own prompt (the template with the root branch).
#[test]
fn root_message_ids_match_the_template_with_the_root_branch() {
    let Some(model_dir) = common::ensure_minimax_m3_cached() else {
        eprintln!(
            "skipping: no MiniMax-M3 tokenizer (set MINIMAX_M3_MODEL_DIR or allow the download)"
        );
        return;
    };
    let tok = tokenizer_with_template(&model_dir.join("tokenizer.json"), M3_TEMPLATE_PATH);
    for case in &fixtures(M3_RENDER_FIXTURES).cases {
        check_case(&tok, case, true);
    }
}
