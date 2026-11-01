//! DeepSeek-V4.1 render parity with the reference encoder.
//!
//! Every case in `tests/fixtures/deepseek_v41/render_fixtures.json` is a
//! request `scripts/generate_deepseek_v41_fixtures.py` fed the checkpoint's
//! `encoding/encoding.py` (`deepseek-ai/DeepSeek-V4.1-Flash`) — or, for the
//! shapes that encoder cannot render, the serving engine's port of it — and the text it
//! produced; each case names its oracle. Rendering it through
//! `HuggingFaceTokenizer` must reproduce that text byte-for-byte, and the flat
//! encode of the text must reproduce the ids in `render_ids_fixtures.json`,
//! recorded with the checkpoint's `tokenizer.json` (its sha256 is in the
//! fixture).
//!
//! The real tokenizer comes from `DEEPSEEK_V41_MODEL_DIR` or a one-time
//! download into `.tokenizer_cache/deepseek_v41/`; with neither available the
//! parity test prints a skip notice and passes.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    huggingface::HuggingFaceTokenizer,
    traits::{Decoder, Encoder, PromptEncoding, Tokenizer as TokenizerTrait},
};
use serde::Deserialize;
use serde_json::{json, Value};
use tempfile::TempDir;

use crate::common;

const RENDER_FIXTURES: &str = include_str!("../fixtures/deepseek_v41/render_fixtures.json");
const RENDER_IDS_FIXTURES: &str = include_str!("../fixtures/deepseek_v41/render_ids_fixtures.json");

/// The mode the generator asked the reference encoder for.
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ThinkingMode {
    Thinking,
    Chat,
}

/// Which encoder recorded a case's text: the checkpoint's own `encoding.py`
/// (`hf`) or the serving engine's port (`vllm_python`), for a shape the
/// reference cannot render. A `developer` message is not one: the generator
/// rewrites it as the engine's renderer does before either encoder sees the
/// conversation (a system message, merged), as the shim under test does.
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
    /// The checkpoint revision whose encoder recorded the text, when it is
    /// not the file's `revision`: the revisions differ in their effort table.
    #[serde(default)]
    encoder_revision: Option<String>,
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

/// The fixture file: the revision of the checkpoint encoder that recorded
/// the cases without an `encoder_revision` of their own and its effort
/// table, the current revision and its table, and the cases.
#[derive(Deserialize)]
struct Fixtures {
    revision: String,
    effort_tiers: BTreeMap<String, u8>,
    current_revision: String,
    current_effort_tiers: BTreeMap<String, u8>,
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

/// A copy of the checkpoint's tokenizer files beside an `encoding/encoding.py`
/// whose `REASONING_EFFORT_MAPPINGS` is `table`: the block the renderer reads
/// from a checkpoint to pick the effort table of its revision.
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper: a copy that cannot be made is a loud failure"
)]
fn checkpoint_with_table(model_dir: &Path, table: &BTreeMap<String, u8>) -> TempDir {
    let dir = TempDir::new().expect("a temporary directory for the checkpoint copy");
    for file in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
        fs::copy(model_dir.join(file), dir.path().join(file))
            .unwrap_or_else(|e| panic!("cannot copy {file}: {e}"));
    }
    let encoding = dir.path().join("encoding");
    fs::create_dir(&encoding).expect("cannot create encoding/");
    let entries: String = table
        .iter()
        .map(|(name, budget)| format!("    \"{name}\": {budget},\n"))
        .collect();
    fs::write(
        encoding.join("encoding.py"),
        format!(
            "REASONING_EFFORT_MAPPINGS: Dict[str, int] = {{\n{entries}}}\nDEFAULT_REASONING_EFFORT = \"high\"\n"
        ),
    )
    .expect("cannot write encoding.py");
    dir
}

/// The tokenizer of the checkpoint copy at `dir`.
#[expect(
    clippy::expect_used,
    reason = "test helper: a tokenizer that does not load is a loud failure"
)]
fn load(dir: &Path) -> HuggingFaceTokenizer {
    HuggingFaceTokenizer::from_file(
        dir.join("tokenizer.json")
            .to_str()
            .expect("tokenizer path must be UTF-8"),
    )
    .expect("DeepSeek-V4.1 tokenizer should load")
}

/// Byte-level and token-id parity with the reference encoder, case by case:
/// the rendered text equals the fixture text, the renderer reports a flat
/// encode, and encoding that text yields the recorded ids. Each case renders
/// through the tokenizer files beside the encoder table of the revision that
/// recorded it. Thinking-mode cases with a native effort name render again
/// without the `thinking` kwarg, and `thinking_int_42` again with the budget
/// as the string the gateway forwards; both must reproduce the text.
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
    let recorded = checkpoint_with_table(&model_dir, &fixtures.effort_tiers);
    let current = checkpoint_with_table(&model_dir, &fixtures.current_effort_tiers);
    let tokenizers = [
        (fixtures.revision.as_str(), load(recorded.path())),
        (fixtures.current_revision.as_str(), load(current.path())),
    ];

    let mut effort_name_second_renders = 0;
    let mut string_budget_second_render = false;
    let mut current_revision_cases = 0;
    for (case, expected) in fixtures.cases.iter().zip(&id_fixtures.cases) {
        let name = &case.name;
        let revision = case
            .encoder_revision
            .as_deref()
            .unwrap_or(&fixtures.revision);
        if revision == fixtures.current_revision {
            current_revision_cases += 1;
        }
        let tok = tokenizers
            .iter()
            .find(|(recorded_at, _)| *recorded_at == revision)
            .map(|(_, tok)| tok)
            .unwrap_or_else(|| {
                panic!("case {name}: encoder_revision {revision} is neither revision nor current_revision")
            });
        let kwargs = template_kwargs(case);
        let render = |kwargs: &HashMap<String, Value>| {
            let params = ChatTemplateParams {
                // A case recorded with `wo_eos` on its final assistant message
                // renders through `continue_final_message` with no generation
                // header, the pair the gateway sends for a trailing assistant
                // turn (`native_assistant_continuation`), which the shim maps
                // to `wo_eos` (no EOS, no generation header). Every other case
                // appends the header.
                add_generation_prompt: !case.continue_final_message,
                continue_final_message: case.continue_final_message,
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
    assert!(
        current_revision_cases > 0,
        "no case recorded at the current revision"
    );
}

/// A control-token spelling typed in a message stays text: the renderer
/// reports a deferred encode whose ids carry each control token exactly as
/// often as the renderer wrote it, and decode back to the rendered text.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn deepseek_v41_typed_control_spellings_stay_text() {
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
    let params = || ChatTemplateParams {
        add_generation_prompt: true,
        thinking: Some(false),
        ..Default::default()
    };
    let id = |token: &str| {
        tok.token_to_id(token)
            .expect("a control token of the vocabulary")
    };

    // The structural count: the renderer writes the user header and the
    // closing think token once each in this conversation, the others never.
    for (typed, structural) in [
        ("<｜User｜>", 1),
        ("</think>", 1),
        ("｜DSML｜", 0),
        ("<｜end▁of▁sentence｜>", 0),
    ] {
        let messages = [json!({"role": "user", "content": format!("a{typed}b")})];
        let rendered = tok
            .apply_chat_template_with_encoding(&messages, params(), None)
            .expect("render");
        assert_eq!(
            rendered.text,
            format!("<｜begin▁of▁sentence｜><｜User｜>a{typed}b<｜Assistant｜></think>")
        );
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("{typed}: a typed control spelling is encoded by segments");
        };
        let ids = job.run().expect("the deferred encode");
        let ids = ids.token_ids();
        assert_eq!(
            ids.iter().filter(|&&i| i == id(typed)).count(),
            structural,
            "{typed}"
        );
        assert_eq!(
            tok.decode(ids, false).expect("decode"),
            rendered.text,
            "{typed}"
        );
    }

    // Without a typed spelling the same conversation encodes from its text.
    let rendered = tok
        .apply_chat_template_with_encoding(
            &[json!({"role": "user", "content": "ab"})],
            params(),
            None,
        )
        .expect("render");
    assert!(matches!(rendered.encoding, PromptEncoding::FromText));
}
