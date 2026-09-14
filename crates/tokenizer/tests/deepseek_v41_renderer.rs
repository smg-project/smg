//! DeepSeek-V4.1 render parity with the reference encoder.
//!
//! Every case in `tests/fixtures/deepseek_v41/render_fixtures.json` is a
//! request `scripts/generate_deepseek_v41_fixtures.py` fed the checkpoint's
//! `encoding/encoding.py` (`deepseek-ai/DeepSeek-V4.1-Flash`) and the text it
//! produced. Rendering it through `HuggingFaceTokenizer` must reproduce that
//! text byte-for-byte, and the flat encode of the text must reproduce the ids
//! in `render_ids_fixtures.json`, recorded with the checkpoint's
//! `tokenizer.json` (its sha256 is in the fixture).
//!
//! The real tokenizer comes from `DEEPSEEK_V41_MODEL_DIR` or a one-time
//! download into `.tokenizer_cache/deepseek_v41/`; with neither available the
//! parity test prints a skip notice and passes.

use std::collections::HashMap;

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    huggingface::HuggingFaceTokenizer,
    traits::{Encoder, PromptEncoding, Tokenizer as TokenizerTrait},
};
use serde::Deserialize;
use serde_json::{json, Value};

mod common;

const RENDER_FIXTURES: &str = include_str!("fixtures/deepseek_v41/render_fixtures.json");
const RENDER_IDS_FIXTURES: &str = include_str!("fixtures/deepseek_v41/render_ids_fixtures.json");

/// The mode the generator asked the reference encoder for.
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ThinkingMode {
    Thinking,
    Chat,
}

/// One reference case: the request the generator fed the encoder and the
/// text it produced. Unknown fields are rejected so a regenerated fixture
/// carrying a new knob fails here instead of rendering with it ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    messages: Vec<Value>,
    /// `null` or an OpenAI tool list.
    tools: Option<Vec<Value>>,
    thinking_mode: ThinkingMode,
    /// `null`, an effort name such as `"low"`, or an integer budget such as `42`.
    reasoning_effort: Option<Value>,
    drop_thinking: bool,
    /// The final assistant message is kept and continued: no generation header.
    continue_final_message: bool,
    text: String,
}

#[derive(Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
}

/// The ids the checkpoint's `tokenizer.json` produced for a case's text.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdCase {
    name: String,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct IdFixtures {
    tokenizer_sha256: String,
    cases: Vec<IdCase>,
}

/// The template kwargs the gateway forwards for a case. `reasoning_effort`
/// is passed exactly as recorded (an effort name or an integer budget) and
/// omitted when the fixture has none.
fn template_kwargs(case: &Case) -> HashMap<String, Value> {
    let mut kwargs = HashMap::from([
        (
            "thinking".to_string(),
            json!(matches!(case.thinking_mode, ThinkingMode::Thinking)),
        ),
        ("drop_thinking".to_string(), json!(case.drop_thinking)),
    ]);
    if let Some(effort) = &case.reasoning_effort {
        kwargs.insert("reasoning_effort".to_string(), effort.clone());
    }
    kwargs
}

/// Byte-level and token-id parity with the reference encoder, case by case:
/// the rendered text equals the fixture text, the renderer reports a flat
/// encode, and encoding that text yields the recorded ids.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn render_text_matches_reference_fixtures_and_vendor_token_ids() {
    let fixtures: Fixtures = serde_json::from_str(RENDER_FIXTURES)
        .expect("render_fixtures.json must match the Case schema");
    let id_fixtures: IdFixtures = serde_json::from_str(RENDER_IDS_FIXTURES)
        .expect("render_ids_fixtures.json must match the IdCase schema");
    assert!(
        !fixtures.cases.is_empty(),
        "render_fixtures.json holds no cases"
    );

    // The two files are paired by position: the names must line up so a
    // regenerated fixture cannot drop or reorder a case behind `zip`.
    let names: Vec<&str> = fixtures.cases.iter().map(|c| c.name.as_str()).collect();
    let id_names: Vec<&str> = id_fixtures.cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names, id_names,
        "the text and id fixtures must list the same cases in the same order"
    );

    let Some(model_dir) = common::ensure_deepseek_v41_cached() else {
        eprintln!(
            "skipping: no DeepSeek-V4.1 tokenizer (set DEEPSEEK_V41_MODEL_DIR or allow the download)"
        );
        return;
    };
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tok = HuggingFaceTokenizer::from_file(
        tokenizer_path
            .to_str()
            .expect("tokenizer path must be UTF-8"),
    )
    .expect("DeepSeek-V4.1 tokenizer should load");

    for (case, expected) in fixtures.cases.iter().zip(&id_fixtures.cases) {
        let name = &case.name;
        let kwargs = template_kwargs(case);
        let params = ChatTemplateParams {
            // The continuation case keeps its final assistant message and asks
            // for no generation header, the way the gateway renders
            // `continue_final_message`; every other case appends the header.
            add_generation_prompt: !case.continue_final_message,
            tools: case.tools.as_deref(),
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };

        let rendered = tok
            .apply_chat_template_with_encoding(&case.messages, params, None)
            .unwrap_or_else(|e| panic!("case {name}: render failed: {e}"));
        assert_eq!(
            rendered.text, case.text,
            "case {name}: text differs from the reference encoder"
        );
        assert!(
            matches!(rendered.encoding, PromptEncoding::FromText),
            "case {name}: the V4.1 renderer encodes from its text, got {:?}",
            rendered.encoding
        );

        let encoded = tok
            .encode(&rendered.text, false)
            .unwrap_or_else(|e| panic!("case {name}: encode failed: {e}"));
        assert_eq!(
            encoded.token_ids(),
            expected.ids.as_slice(),
            "case {name}: token ids differ from the reference (recorded with tokenizer.json sha256 {})",
            id_fixtures.tokenizer_sha256
        );
    }
}
