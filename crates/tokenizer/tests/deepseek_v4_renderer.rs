//! DeepSeek-V4 render parity with the engine's own server.
//!
//! Every case in `tests/fixtures/deepseek_v4/render_fixtures.json` is a chat
//! request and the prompt the engine's own server renders for it (its
//! `tokenizers/deepseek_v4.py` over its port of the checkpoint's encoder):
//! the request's `chat_template_kwargs`, top-level `reasoning_effort` and
//! tools, the rendered text, and the text's ids under the checkpoint's
//! `tokenizer.json` without added special tokens, which is what the engine's
//! `/tokenize` returns for the request. Rendering a case the way the gateway's
//! chat path does must reproduce the text (checked with a minimal tokenizer)
//! and the ids (checked with the real one: `DEEPSEEK_V4_MODEL_DIR` or a
//! one-time download into `.tokenizer_cache/deepseek_v4/`; offline, the id
//! check prints a skip notice and passes).

use std::{collections::HashMap, fs};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    huggingface::HuggingFaceTokenizer,
    traits::{Encoder, Tokenizer as TokenizerTrait},
};
use serde::Deserialize;
use serde_json::Value;
use tempfile::TempDir;

mod common;

const RENDER_FIXTURES: &str = include_str!("fixtures/deepseek_v4/render_fixtures.json");

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

/// One request as the engine's own server received it, with its prompt.
/// Unknown fields are rejected so a regenerated fixture carrying a new knob
/// fails here instead of rendering with it ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    messages: Vec<Value>,
    #[serde(default)]
    chat_template_kwargs: Option<HashMap<String, Value>>,
    #[serde(default)]
    reasoning_effort: Option<String>,
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

/// The template kwargs the gateway's chat path forwards for a case: the
/// top-level `reasoning_effort` projected under the key the renderer reads,
/// the request's own `chat_template_kwargs` on top of it.
fn gateway_template_kwargs(case: &Case) -> HashMap<String, Value> {
    let mut kwargs = HashMap::new();
    if let Some(effort) = &case.reasoning_effort {
        kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(effort.clone()),
        );
    }
    if let Some(explicit) = &case.chat_template_kwargs {
        kwargs.extend(explicit.clone());
    }
    kwargs
}

#[expect(clippy::panic, reason = "test helper — panics are intentional")]
fn render(tok: &HuggingFaceTokenizer, case: &Case) -> String {
    let kwargs = gateway_template_kwargs(case);
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: case.tools.as_deref(),
        template_kwargs: (!kwargs.is_empty()).then_some(&kwargs),
        // The gateway's protocol-level thinking preference for the request.
        thinking: openai_protocol::chat::thinking_from_reasoning_effort(
            case.reasoning_effort.as_deref(),
        ),
        ..Default::default()
    };
    tok.apply_chat_template(&case.messages, params)
        .unwrap_or_else(|e| panic!("case {}: render failed: {e}", case.name))
}

#[expect(clippy::expect_used, reason = "test helper — panics are intentional")]
fn fixtures() -> Fixtures {
    let fixtures: Fixtures = serde_json::from_str(RENDER_FIXTURES)
        .expect("render_fixtures.json must match the Case schema");
    assert!(
        !fixtures.cases.is_empty(),
        "render_fixtures.json holds no cases"
    );
    fixtures
}

/// The rendered text equals the engine's prompt, case by case, from a model
/// directory that carries the architecture and no encoder of its own (what a
/// worker streams).
#[test]
fn deepseek_v4_render_text_matches_the_engines_prompts() {
    let temp = TempDir::new().expect("temp dir");
    fs::write(temp.path().join("tokenizer.json"), MIN_TOKENIZER_JSON).expect("tokenizer.json");
    fs::write(
        temp.path().join("config.json"),
        r#"{"architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4"}"#,
    )
    .expect("config.json");
    let tokenizer_path = temp.path().join("tokenizer.json");
    let tok = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().expect("UTF-8 path"))
        .expect("the minimal tokenizer should load");
    for case in &fixtures().cases {
        assert_eq!(
            render(&tok, case),
            case.text,
            "case {}: prompt differs from the engine's",
            case.name
        );
    }
}

/// With the checkpoint's tokenizer, the rendered text encodes to the ids the
/// engine's `/tokenize` returns for the request.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn deepseek_v4_render_ids_match_the_engines_tokenize() {
    let Some(model_dir) = common::ensure_deepseek_v4_cached() else {
        eprintln!(
            "skipping: no DeepSeek-V4 tokenizer (set DEEPSEEK_V4_MODEL_DIR or allow the download)"
        );
        return;
    };
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tok = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().expect("UTF-8 path"))
        .expect("the DeepSeek-V4 tokenizer should load");
    for case in &fixtures().cases {
        let name = &case.name;
        let text = render(&tok, case);
        assert_eq!(
            text, case.text,
            "case {name}: prompt differs from the engine's"
        );
        let encoded = tok
            .encode(&text, false)
            .unwrap_or_else(|e| panic!("case {name}: encode failed: {e}"));
        assert_eq!(
            encoded.token_ids(),
            case.ids.as_slice(),
            "case {name}: token ids differ from the engine's /tokenize"
        );
    }
}
