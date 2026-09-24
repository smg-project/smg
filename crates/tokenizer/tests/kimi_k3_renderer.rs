//! Golden tests for the Kimi-K3 XTML chat-template renderer.
//!
//! Each case builds the equivalent messages/tools/params in Rust and asserts
//! the rendered `String` equals the `text` field of the corresponding entry in
//! `tests/fixtures/kimi_k3/k3_render_fixtures.json` byte-for-byte. Those
//! fixtures are the authoritative expected outputs produced by the upstream
//! Python `encoding_k3.py::build_chat_segments`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::{collections::HashMap, fs};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    encoders::kimi_k3_xtml::apply_kimi_k3_xtml,
    traits::{Encoder, PromptEncoding, Tokenizer as TokenizerTrait},
    TiktokenTokenizer,
};
use serde_json::{json, Value};
use tempfile::TempDir;

mod common;

const MIN_TIKTOKEN_MODEL: &str = "aGVsbG8= 0\n";

/// Load a single fixture's expected `text` by case name.
fn fixture_text(case: &str) -> String {
    let raw = fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kimi_k3/k3_render_fixtures.json"),
    )
    .expect("k3 fixtures must exist");
    let value: Value = serde_json::from_str(&raw).expect("fixtures must be valid JSON");
    value
        .get(case)
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("fixture case `{case}` missing text"))
        .to_string()
}

fn render(messages: &[Value], tools: Option<&[Value]>, thinking: bool) -> String {
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools,
        thinking: Some(thinking),
        ..Default::default()
    };
    apply_kimi_k3_xtml(messages, &params).expect("k3 render should succeed")
}

/// A K3 directory with a stub vocabulary and, intentionally, no
/// chat_template.json / .jinja.
fn k3_tokenizer_without_chat_template() -> (TempDir, TiktokenTokenizer) {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("tiktoken.model"), MIN_TIKTOKEN_MODEL).unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#,
    )
    .unwrap();
    fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
    let tok = TiktokenTokenizer::from_dir(dir.path()).expect("K3 tokenizer should load");
    (dir, tok)
}

fn get_weather_tools() -> Vec<Value> {
    vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }
        }
    })]
}

#[test]
fn plain_user_thinking() {
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    assert_eq!(
        render(&messages, None, true),
        fixture_text("plain_user_thinking")
    );
}

#[test]
fn system_user_thinking() {
    let messages = vec![
        json!({"role": "system", "content": "You are helpful"}),
        json!({"role": "user", "content": "Hi"}),
    ];
    assert_eq!(
        render(&messages, None, true),
        fixture_text("system_user_thinking")
    );
}

#[test]
fn plain_user_no_thinking() {
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    assert_eq!(
        render(&messages, None, false),
        fixture_text("plain_user_no_thinking")
    );
}

#[test]
fn assistant_prior_turn() {
    let messages = vec![
        json!({"role": "user", "content": "Hi"}),
        json!({"role": "assistant", "content": "Hello!"}),
        json!({"role": "user", "content": "Bye"}),
    ];
    assert_eq!(
        render(&messages, None, true),
        fixture_text("assistant_prior_turn")
    );
}

#[test]
fn with_tools() {
    let messages = vec![json!({"role": "user", "content": "weather in Paris?"})];
    let tools = get_weather_tools();
    assert_eq!(
        render(&messages, Some(&tools), true),
        fixture_text("with_tools")
    );
}

#[test]
fn assistant_tool_call_then_result() {
    let messages = vec![
        json!({"role": "user", "content": "weather?"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}}
            }]
        }),
        json!({"role": "tool", "tool": "get_weather", "content": "sunny"}),
    ];
    let tools = get_weather_tools();
    assert_eq!(
        render(&messages, Some(&tools), true),
        fixture_text("assistant_tool_call_then_result")
    );
}

#[test]
fn thinking_effort_low() {
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    let template_kwargs = HashMap::from([("thinking_effort".to_string(), json!("low"))]);
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        thinking: Some(true),
        template_kwargs: Some(&template_kwargs),
        ..Default::default()
    };
    let rendered = apply_kimi_k3_xtml(&messages, &params).expect("k3 render should succeed");
    assert_eq!(rendered, fixture_text("thinking_effort_low"));
}

#[test]
fn thinking_effort_ignored_when_thinking_off() {
    // Reference gates both validation and emission on `thinking`, so an effort
    // provided while thinking is off produces no thinking-effort message.
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    let template_kwargs = HashMap::from([("thinking_effort".to_string(), json!("low"))]);
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        thinking: Some(false),
        template_kwargs: Some(&template_kwargs),
        ..Default::default()
    };
    let rendered = apply_kimi_k3_xtml(&messages, &params).expect("k3 render should succeed");
    assert!(!rendered.contains("thinking-effort"), "got: {rendered}");
    assert_eq!(rendered, fixture_text("plain_user_no_thinking"));
}

#[test]
fn thinking_effort_invalid_is_rejected() {
    // Mirrors the reference `assert thinking_effort in _VALID_THINKING_EFFORTS`;
    // `medium` is described in the body text but not an accepted value.
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    let template_kwargs = HashMap::from([("thinking_effort".to_string(), json!("medium"))]);
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        thinking: Some(true),
        template_kwargs: Some(&template_kwargs),
        ..Default::default()
    };
    assert!(apply_kimi_k3_xtml(&messages, &params).is_err());
}

/// End-to-end: a tokenizer loaded from a K3 directory (no chat template at all)
/// must load successfully, detect the K3 renderer, and render XTML through
/// `apply_chat_template`.
///
/// `apply_chat_template` stands in for the checkpoint's `tokenization_kimi`
/// wrapper, so its output is the plain fixture *plus* the `max` effort
/// directive. The expected bytes are the `thinking_effort_low` fixture with its
/// effort word swapped — the directive is identical at every level.
#[test]
fn tokenizer_loads_and_renders_k3_without_chat_template() {
    let (_dir, tok) = k3_tokenizer_without_chat_template();
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    let rendered = tok
        .apply_chat_template(
            &messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                thinking: Some(true),
                ..Default::default()
            },
        )
        .expect("K3 render should succeed");

    let expected = fixture_text("thinking_effort_low")
        .replace("`thinking_effort=low`", "`thinking_effort=max`");
    assert_eq!(rendered, expected);
    assert!(
        rendered.ends_with(&fixture_text("plain_user_thinking")),
        "the directive is the only addition: {rendered}"
    );
}

/// The K3 encoder coerces tool-call `arguments` itself, so the gateway must
/// forward them as written.
#[test]
fn k3_tokenizer_reports_raw_tool_call_arguments() {
    let (_dir, tok) = k3_tokenizer_without_chat_template();
    let caps = tok.renderer_capabilities();
    assert!(caps.raw_tool_call_arguments, "{caps:?}");
    assert!(!caps.enable_thinking_alias, "{caps:?}");
    assert!(!caps.native_assistant_continuation, "{caps:?}");
}

/// An image part is exactly one `<|media_pad|>` id at its authored position:
/// the text ids on either side are untouched and no separator is added.
#[test]
fn image_part_adds_exactly_one_anchor_token() {
    let model_dir = common::ensure_kimi_k3_cached();
    let tok = TiktokenTokenizer::from_dir(&model_dir).expect("K3 tokenizer should load");
    let anchor = tok.token_to_id("<|media_pad|>").expect("anchor id");
    let sep = tok.token_to_id("<|sep|>").expect("sep id");
    let close = tok.token_to_id("<|close|>").expect("close id");
    let text = "Describe the image set in one short sentence.";
    let ids = |content: Value| {
        let messages = vec![json!({"role": "user", "content": content})];
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            thinking: Some(false),
            ..Default::default()
        };
        let rendered = tok
            .apply_chat_template_with_encoding(&messages, params, None)
            .expect("render should succeed");
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("K3 must defer its encode");
        };
        job.run().expect("encode").token_ids().to_vec()
    };

    let base = ids(json!(text));
    assert!(!base.contains(&anchor));
    let text_first = ids(json!([{"type": "text", "text": text}, {"type": "image"}]));
    let image_first = ids(json!([{"type": "image"}, {"type": "text", "text": text}]));
    for (with_image, neighbour) in [(text_first, close), (image_first, sep)] {
        assert_eq!(with_image.len(), base.len() + 1, "{with_image:?}");
        let positions: Vec<usize> = (0..with_image.len())
            .filter(|&i| with_image[i] == anchor)
            .collect();
        let [p] = positions[..] else {
            panic!("exactly one anchor expected: {positions:?}");
        };
        assert_eq!(&with_image[..p], &base[..p]);
        assert_eq!(&with_image[p + 1..], &base[p..]);
        // Text-first: the anchor precedes `<|close|>`; image-first: it follows `<|sep|>`.
        let beside = if neighbour == close {
            with_image[p + 1]
        } else {
            with_image[p - 1]
        };
        assert_eq!(beside, neighbour);
    }
}

/// Token-id parity with the checkpoint's own `apply_chat_template(tokenize=True)`
/// (`build_chat_segments` + `_encode_chat_segments`), recorded in
/// `tests/fixtures/kimi_k3/k3_render_ids_fixtures.json` from
/// `nvidia/Kimi-K3-NVFP4` (the same tokenizer files as `moonshotai/Kimi-K3`)
/// and reproduced under transformers 4.57.6 and 5.16.1; `tool_call_bad_arguments`
/// was recorded from the same tokenizer files under transformers 5.17.0.
///
/// The real `tiktoken.model` and `tokenizer_config.json` are fetched once
/// into `.tokenizer_cache/kimi_k3/`; `KIMI_K3_MODEL_DIR` points at a local
/// checkpoint directory instead.
#[test]
fn segment_encoding_matches_vendor_token_ids() {
    let model_dir = common::ensure_kimi_k3_cached();
    let tok = TiktokenTokenizer::from_dir(&model_dir).expect("K3 tokenizer should load");

    let raw = fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kimi_k3/k3_render_ids_fixtures.json"),
    )
    .expect("k3 id fixtures must exist");
    let cases: Value = serde_json::from_str(&raw).expect("fixtures must be valid JSON");

    // Cases where a flat encode of the rendered text cannot reproduce the
    // reference ids: marker strings inside message text become control ids,
    // and BPE merges across attribute-piece boundaries (`=".hidden`).
    const FLAT_DIFFERS: &[&str] = &[
        "injected_markers",
        "tool_call_round_trip",
        "punctuation_attribute_value",
    ];

    for (name, case) in cases.as_object().expect("fixture root is an object") {
        let messages = case["messages"].as_array().expect("messages").clone();
        let tools: Option<Vec<Value>> = case["tools"].as_array().cloned();
        let expected: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let params = || ChatTemplateParams {
            add_generation_prompt: true,
            tools: tools.as_deref(),
            ..Default::default()
        };

        let flat = tok
            .apply_chat_template(&messages, params())
            .expect("flat render");
        let rendered = tok
            .apply_chat_template_with_encoding(&messages, params(), None)
            .expect("render should succeed");
        assert_eq!(
            rendered.text, flat,
            "case {name}: text is the flat rendering"
        );
        assert_eq!(
            rendered.text,
            case["text"].as_str().expect("text"),
            "case {name}: text differs from the reference"
        );
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("case {name}: K3 must defer its encode");
        };
        let ids = job.run().expect("deferred encode should succeed");
        assert_eq!(
            ids.token_ids(),
            &expected[..],
            "case {name}: deferred encoding differs from the reference"
        );

        let flat_ids = tok.encode(&flat, false).expect("flat encode");
        assert_eq!(
            flat_ids.token_ids() != &expected[..],
            FLAT_DIFFERS.contains(&name.as_str()),
            "case {name}: flat encoding parity unexpected"
        );
    }
}
