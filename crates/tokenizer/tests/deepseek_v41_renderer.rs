//! DeepSeek-V4.1 render parity with the reference encoder.
//!
//! Every case in `tests/fixtures/deepseek_v41/render_fixtures.json` is a
//! request `scripts/generate_deepseek_v41_fixtures.py` fed the checkpoint's
//! `encoding/encoding.py` (`deepseek-ai/DeepSeek-V4.1-Flash`) — or, for the
//! shapes that encoder cannot render, vLLM's port of it — and the text it
//! produced; each case names its oracle. Rendering it through
//! `HuggingFaceTokenizer` must reproduce that text byte-for-byte, and the flat
//! encode of the text must reproduce the ids in `render_ids_fixtures.json`,
//! recorded with the checkpoint's `tokenizer.json` (its sha256 is in the
//! fixture).
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

/// Which encoder recorded a case's text: the checkpoint's own `encoding.py`
/// (`hf`) or vLLM's port (`vllm_python`), used for the shapes the reference
/// cannot render, such as a developer message it keeps.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Oracle {
    Hf,
    VllmPython,
}

/// One reference case: the request the generator fed the encoder and the
/// text it produced. Unknown fields are rejected so a regenerated fixture
/// carrying a new knob fails here instead of rendering with it ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    oracle: Oracle,
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
/// encode, and encoding that text yields the recorded ids. Thinking-mode
/// cases with a native effort name render again without the `thinking`
/// kwarg, and `thinking_int_42` again with the budget as the string the
/// gateway forwards; both must reproduce the text.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn deepseek_v41_render_text_matches_reference_fixtures_and_vendor_token_ids() {
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

    let mut effort_name_second_renders = 0;
    let mut string_budget_second_render = false;
    for (case, expected) in fixtures.cases.iter().zip(&id_fixtures.cases) {
        let name = &case.name;
        let kwargs = template_kwargs(case);
        let render = |kwargs: &HashMap<String, Value>| {
            let params = ChatTemplateParams {
                // A case recorded with `wo_eos` on its final assistant message
                // renders through `add_generation_prompt: false`, which the
                // shim maps to `wo_eos` when the last message is an assistant
                // turn (no EOS, no generation header). The gateway does not
                // send that yet: it renders `continue_final_message` by
                // popping the trailing assistant message and appending its
                // content after the generation header; routing it through
                // `add_generation_prompt: false` is a follow-up. Every other
                // case appends the header.
                add_generation_prompt: !case.continue_final_message,
                tools: case.tools.as_deref(),
                template_kwargs: Some(kwargs),
                ..Default::default()
            };
            tok.apply_chat_template_with_encoding(&case.messages, params, None)
                .unwrap_or_else(|e| panic!("case {name}: render failed: {e}"))
        };

        let rendered = render(&kwargs);
        assert_eq!(
            rendered.text, case.text,
            "case {name}: text differs from the reference encoder (oracle {:?})",
            case.oracle
        );
        assert!(
            matches!(rendered.encoding, PromptEncoding::FromText),
            "case {name}: the V4.1 renderer encodes from its text, got {:?}",
            rendered.encoding
        );

        // A native effort name alone switches thinking on: the same request
        // without the `thinking` kwarg must render the same text.
        let native_effort_name = case
            .reasoning_effort
            .as_ref()
            .and_then(Value::as_str)
            .is_some_and(|effort| tok.native_reasoning_effort_values().contains(&effort));
        if native_effort_name && matches!(case.thinking_mode, ThinkingMode::Thinking) {
            let mut without_toggle = kwargs.clone();
            without_toggle.remove("thinking");
            assert_eq!(
                render(&without_toggle).text,
                case.text,
                "case {name}: a native effort name alone must switch thinking on"
            );
            effort_name_second_renders += 1;
        }
        // The gateway forwards a top-level integer budget as the string
        // "42"; the shim restores the number before parsing it.
        if name == "thinking_int_42" {
            let mut string_budget = kwargs.clone();
            string_budget.insert("reasoning_effort".to_string(), json!("42"));
            assert_eq!(
                render(&string_budget).text,
                case.text,
                "case {name}: the gateway's string form of the budget must render the same text"
            );
            string_budget_second_render = true;
        }

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
    assert!(
        effort_name_second_renders > 0,
        "no thinking-mode case with a native effort name was rendered without the thinking kwarg"
    );
    assert!(
        string_budget_second_render,
        "the thinking_int_42 case was not rendered with the string budget"
    );
}
