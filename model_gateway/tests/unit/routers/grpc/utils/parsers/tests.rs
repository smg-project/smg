use std::sync::Arc;

use llm_multimodal::MediaPartOrder;
use smg_worker::identity::basic::WorkerType;

use super::*;
use crate::routers::grpc::{
    backend_client::BasicWorkerBuilder,
    testing::{forced_call_on_a_thinking_prompt, worker},
    utils::{chat_utils, message_utils},
};

/// A parser that registers a reasoning prefix, so a forced call on a
/// thinking prompt takes the shape the worker's engine needs.
const PARSER: &str = "glm47_moe";

/// The worker's constrained-decoding policy comes off its spec first, then
/// its `constrained_decoding` label; the decode worker's under PD; a
/// missing or unreadable value, and no selection, are `requested`.
#[test]
fn constrained_decoding_follows_the_spec_then_the_label_of_the_decode_worker() {
    assert_eq!(
        WorkerDecoding::of(None).policy,
        ConstrainedDecoding::Requested
    );
    assert_eq!(
        WorkerDecoding::of(Some(&worker(&[("constrained_decoding", "always")]))).policy,
        ConstrainedDecoding::Always
    );
    assert_eq!(
        WorkerDecoding::of(Some(&worker(&[("constrained_decoding", "sometimes")]))).policy,
        ConstrainedDecoding::Requested
    );
    assert_eq!(
        WorkerDecoding::of(Some(&worker(&[]))).policy,
        ConstrainedDecoding::Requested
    );
    let spec_wins = BasicWorkerBuilder::new("grpc://engine:50051")
        .worker_type(WorkerType::Regular)
        .label("constrained_decoding", "requested")
        .constrained_decoding(ConstrainedDecoding::Always)
        .build();
    assert_eq!(
        WorkerDecoding::of(Some(&WorkerSelection::Single {
            worker: Arc::new(spec_wins).into(),
        }))
        .policy,
        ConstrainedDecoding::Always
    );
    let prefill = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill:50051")
            .worker_type(WorkerType::Prefill)
            .label("constrained_decoding", "always")
            .build(),
    );
    let decode = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:50051")
            .worker_type(WorkerType::Decode)
            .build(),
    );
    assert_eq!(
        WorkerDecoding::of(Some(&WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill,
            decode: decode.into(),
            runtime_type: openai_protocol::worker::RuntimeType::TokenSpeed,
        }))
        .policy,
        ConstrainedDecoding::Requested
    );
}

/// A retry takes only workers that decide a request's constraint as the
/// first attempt's did: under `requested` whatever the start or the
/// vocabulary, under `always` the whole turn's grammar on both or on
/// neither, and where it is on both, in the same vocabulary (its tag is
/// spelled once, for the first worker's compiler); and any prefill
/// worker, which generates no token a grammar constrains.
#[test]
fn a_retry_takes_only_workers_that_decide_the_constraint_alike() {
    let always = ("constrained_decoding", "always");
    let first_token = ("constrained_decoding_start", "first_token");
    let older = ("structural_tag_vocabulary", "older");
    let labels: [&[(&str, &str)]; 8] = [
        &[],
        &[first_token],
        &[always, first_token],
        &[always, ("constrained_decoding_start", "after_reasoning")],
        &[always],
        &[always, first_token, older],
        &[older],
        &[always, older],
    ];
    // The workers alike: the `requested` ones, whatever their vocabulary;
    // the whole turn's in the current vocabulary; the `always` ones that
    // start the grammar later or do not say, whatever their vocabulary;
    // the whole turn's in the older vocabulary, alone.
    let alike = [0, 0, 1, 2, 2, 3, 0, 2];
    let engine = |labels: &[(&str, &str)]| match worker(labels) {
        WorkerSelection::Single { worker } => worker,
        WorkerSelection::Disaggregated { .. } => panic!("expected one worker"),
    };
    let prefill = BasicWorkerBuilder::new("grpc://prefill:50051")
        .worker_type(WorkerType::Prefill)
        .build();
    for (first, first_labels) in labels.iter().enumerate() {
        let decided_on = WorkerDecoding::of(Some(&worker(first_labels)));
        for (other, other_labels) in labels.iter().enumerate() {
            assert_eq!(
                decided_on.decides_alike(&*engine(other_labels)),
                alike[first] == alike[other],
                "decided on {first_labels:?}, retried on {other_labels:?}"
            );
        }
        assert!(decided_on.decides_alike(&prefill));
    }
}

/// The worker's `structural_tag_vocabulary` label names the vocabulary of
/// structural-tag nodes its grammar compiler takes: the older one on
/// `older`, the current one otherwise, the label's absence and any other
/// word included; under PD the decode worker's label.
#[test]
fn the_structural_tag_vocabulary_follows_the_label_of_the_decode_worker() {
    assert_eq!(WorkerDecoding::of(None).vocabulary, TagVocabulary::Current);
    assert_eq!(
        WorkerDecoding::of(Some(&worker(&[]))).vocabulary,
        TagVocabulary::Current
    );
    assert_eq!(
        WorkerDecoding::of(Some(&worker(&[("structural_tag_vocabulary", "older")]))).vocabulary,
        TagVocabulary::Older
    );
    assert_eq!(
        WorkerDecoding::of(Some(&worker(&[("structural_tag_vocabulary", "current")]))).vocabulary,
        TagVocabulary::Current
    );
    let prefill = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill:50051")
            .worker_type(WorkerType::Prefill)
            .label("structural_tag_vocabulary", "older")
            .build(),
    );
    let decode = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:50051")
            .worker_type(WorkerType::Decode)
            .build(),
    );
    assert_eq!(
        WorkerDecoding::of(Some(&WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill,
            decode: decode.into(),
            runtime_type: openai_protocol::worker::RuntimeType::TokenSpeed,
        }))
        .vocabulary,
        TagVocabulary::Current
    );
}

/// The worker's `constrained_decoding_start` label names where its engine
/// starts the grammar; no selection, no label or another value is unknown.
#[test]
fn grammar_start_comes_from_the_workers_label() {
    assert_eq!(grammar_start_for(None), GrammarStart::Unknown);
    assert_eq!(grammar_start_for(Some(&worker(&[]))), GrammarStart::Unknown);
    assert_eq!(
        grammar_start_for(Some(&worker(&[(
            "constrained_decoding_start",
            "after_reasoning"
        )]))),
        GrammarStart::AfterReasoning
    );
    assert_eq!(
        grammar_start_for(Some(&worker(&[(
            "constrained_decoding_start",
            "first_token"
        )]))),
        GrammarStart::FirstToken
    );
    assert_eq!(
        grammar_start_for(Some(&worker(&[("constrained_decoding_start", "later")]))),
        GrammarStart::Unknown
    );
}

/// A forced call's tag on a thinking prompt takes the shape the selected
/// worker's engine needs: the bare calls behind a reasoning parser's gate,
/// the thought block then the calls from the first token, and the
/// free-text prefix it was built with where the worker does not say.
#[test]
fn tool_constraints_take_the_shape_the_workers_engine_needs() {
    let factory = ToolParserFactory::new();
    let parser = Some(PARSER);
    let built = forced_call_on_a_thinking_prompt(&factory, PARSER);
    let built_tag: Value = serde_json::from_str(&built.1).unwrap();
    assert_eq!(built_tag["format"]["type"], "sequence");

    let shape = |labels: &[(&str, &str)]| -> (Value, bool) {
        let (shaped, first_token) = tool_constraints_for_worker(
            &factory,
            parser,
            Some(built.clone()),
            Some(&worker(labels)),
        )
        .unwrap();
        let (kind, value) = shaped.expect("the constraint stays");
        assert_eq!(kind, "structural_tag");
        (serde_json::from_str(&value).unwrap(), first_token)
    };
    let (deferred, led) = shape(&[("constrained_decoding_start", "after_reasoning")]);
    assert!(!led, "nothing led the bare calls");
    assert_eq!(
        deferred["format"]["type"], "triggered_tags",
        "nothing precedes the calls: {deferred}"
    );
    assert_eq!(deferred["format"]["at_least_one"], true);
    let (first_token, led) = shape(&[("constrained_decoding_start", "first_token")]);
    assert!(led, "the thought block led the calls for the first token");
    assert_eq!(first_token["format"]["type"], "sequence");
    assert_eq!(first_token["format"]["elements"][0]["type"], "tag");
    assert_eq!(first_token["format"]["elements"][0]["end"], "</think>");
    assert_eq!(
        first_token["format"]["elements"][1],
        built_tag["format"]["elements"][1]
    );
    assert_eq!(
        shape(&[]),
        (built_tag, false),
        "a worker that does not say keeps the shape that fits both"
    );

    // A parser without a reasoning prefix leaves the tag as built, on any
    // start: nothing led it, so it starts where the engine does.
    assert_eq!(
        tool_constraints_for_worker(
            &factory,
            Some("mistral"),
            Some(built.clone()),
            Some(&worker(&[("constrained_decoding_start", "first_token")]))
        )
        .unwrap(),
        (Some(built.clone()), false),
        "a parser without a prefix leads nothing"
    );
    // Nothing else is touched: no selection, a JSON-schema constraint, no constraint.
    assert_eq!(
        tool_constraints_for_worker(&factory, parser, Some(built.clone()), None).unwrap(),
        (Some(built.clone()), false)
    );
    let schema = ("json_schema".to_string(), "{}".to_string());
    assert_eq!(
        tool_constraints_for_worker(
            &factory,
            parser,
            Some(schema.clone()),
            Some(&worker(&[("constrained_decoding_start", "first_token")]))
        )
        .unwrap(),
        (Some(schema), false),
        "a schema is not led: it starts where the engine does"
    );
    assert_eq!(
        tool_constraints_for_worker(&factory, parser, None, Some(&worker(&[]))).unwrap(),
        (None, false)
    );
}

#[test]
fn template_without_a_switch_starts_in_reasoning_when_its_prompt_opens_the_think_block() {
    use llm_tokenizer::MockTokenizer;
    // GLM-5.3: no thinking switch, the generation prompt ends in `<think>`.
    // No request field can switch thinking off, so none may disarm the parser.
    let always_thinks = MockTokenizer::new()
        .with_thinking_toggle(ThinkingToggle::None)
        .with_think_in_prefill(true);
    for user_thinking in [None, Some(true), Some(false)] {
        assert!(
            should_mark_reasoning_started(user_thinking, &always_thinks),
            "{user_thinking:?}"
        );
    }
    // A non-thinking template without a switch writes an empty, closed
    // think block: the tag is in the prefill, nothing is open.
    let closed_block = MockTokenizer::new()
        .with_thinking_toggle(ThinkingToggle::None)
        .with_think_in_prefill(true)
        .with_prefill_opens_think_block(false);
    for user_thinking in [None, Some(true), Some(false)] {
        assert!(
            !should_mark_reasoning_started(user_thinking, &closed_block),
            "{user_thinking:?}"
        );
    }
    // A template without a switch that leaves the think block to the model
    // starts in content mode, as before.
    let never_prefills = MockTokenizer::new().with_thinking_toggle(ThinkingToggle::None);
    for user_thinking in [None, Some(true), Some(false)] {
        assert!(!should_mark_reasoning_started(
            user_thinking,
            &never_prefills
        ));
    }
}

#[test]
fn resolve_thinking_pref_precedence() {
    // Explicit toggle > native template effort > typed toggle > the profile's
    // reading of reasoning_effort (here the OpenAI one: none off, high none).
    let effort = |value: &str| ProviderProfile::Generic.thinking_from_reasoning_effort(Some(value));
    assert_eq!(
        resolve_thinking_pref(Some(false), Some(true), Some(true), effort("high")),
        Some(false)
    );
    assert_eq!(
        resolve_thinking_pref(None, Some(true), Some(false), effort("none")),
        Some(true)
    );
    assert_eq!(
        resolve_thinking_pref(None, None, Some(false), effort("high")),
        Some(false)
    );
    assert_eq!(
        resolve_thinking_pref(None, None, Some(true), effort("none")),
        Some(true)
    );
    assert_eq!(
        resolve_thinking_pref(None, None, None, effort("none")),
        Some(false)
    );
    assert_eq!(
        resolve_thinking_pref(None, None, None, effort("high")),
        None
    );
    assert_eq!(resolve_thinking_pref(None, None, None, None), None);
}

/// A kwargs `reasoning_effort` of `"none"` renders chat mode for native
/// renderers, so it must disarm the parser too — even when the top-level
/// field names a native level (the kwargs entry wins in the merge).
#[test]
fn kwargs_none_disarms_like_the_renderer() {
    let tok = T(llm_tokenizer::MockTokenizer::new());
    let none_kw = std::collections::HashMap::from([(
        "reasoning_effort".to_string(),
        Value::String("none".to_string()),
    )]);
    assert_eq!(
        extract_template_effort_thinking(Some(&none_kw), None, &tok),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(Some(&none_kw), None, None, &tok, ProviderProfile::Generic),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
        Some(false)
    );
    // The request-level field is written over the kwargs entry, as the
    // engine renders it: a top-level `high` arms against a kwargs `none`.
    assert_eq!(
        extract_template_effort_thinking(Some(&none_kw), Some("high"), &tok),
        Some(true)
    );
    assert_eq!(
        resolve_user_thinking(
            Some(&none_kw),
            Some("high"),
            None,
            &tok,
            ProviderProfile::Generic
        ),
        Some(true)
    );
    // `minimal` is the switch's other spelling and disarms the same way.
    let minimal_kw = std::collections::HashMap::from([(
        "reasoning_effort".to_string(),
        Value::String("minimal".to_string()),
    )]);
    assert_eq!(
        extract_template_effort_thinking(Some(&minimal_kw), None, &tok),
        Some(false)
    );
}

use llm_tokenizer::traits::{Encoder, Encoding};
struct T(llm_tokenizer::MockTokenizer);
impl Encoder for T {
    fn encode(&self, i: &str, s: bool) -> anyhow::Result<Encoding> {
        self.0.encode(i, s)
    }
    fn encode_batch(&self, i: &[&str], s: bool) -> anyhow::Result<Vec<Encoding>> {
        self.0.encode_batch(i, s)
    }
}
impl llm_tokenizer::traits::Decoder for T {
    fn decode(&self, ids: &[u32], s: bool) -> anyhow::Result<String> {
        self.0.decode(ids, s)
    }
}
impl Tokenizer for T {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn get_special_tokens(&self) -> &llm_tokenizer::traits::SpecialTokens {
        self.0.get_special_tokens()
    }
    fn token_to_id(&self, t: &str) -> Option<u32> {
        self.0.token_to_id(t)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        &["low", "high", "max"]
    }
    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        Some(ThinkingKeyName::ThinkingMode)
    }
}

#[test]
fn sglang_thinking_budget_checks_the_selected_templates_generation_header() {
    use llm_multimodal::{MediaPartOrder, Modality};
    use serde_json::{json, Value};

    use crate::routers::grpc::{
        multimodal::PlaceholderTokens, utils::process_chat_messages_with_placeholders,
    };
    const MODEL: &str = "Qwen/Qwen3-8B";
    const OFF: &str = "<think>\n\n</think>\n\n";
    const TURN: &str =
        "{{ messages[-1].content }}{% if add_generation_prompt %}<|im_start|>assistant\n";
    let normal = format!("{TURN}{{% if enable_thinking|default(true) %}}<think>{{% else %}}{{{{ '<think>\\n\\n</think>\\n\\n' }}}}{{% endif %}}{{% endif %}}");
    let named = normal.replace(
        "enable_thinking|default(true)",
        "thinking_mode != 'disabled'",
    );
    let echo = format!("{{# enable_thinking #}}{TURN}{{{{ messages[-1].content }}}}{{% endif %}}");
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{"hi":0},"merges":[]}}"#,
    )
    .unwrap();
    let allows = |template: Value, tools: bool, kwargs: Value, content: Value| {
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"chat_template":template}).to_string(),
        )
        .unwrap();
        let tokenizer = llm_tokenizer::create_tokenizer(dir.path().to_str().unwrap()).unwrap();
        let request = serde_json::from_value(json!({
            "model":MODEL, "messages":[{"role":"user","content":content}],
            "tools":tools.then(Vec::<Value>::new), "chat_template_kwargs":kwargs, "thinking_token_budget":0
        })).unwrap();
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "<image>".to_owned());
        let render = |chat: &ChatCompletionRequest| {
            process_chat_messages_with_placeholders(
                chat,
                tokenizer.as_ref(),
                Some(&placeholders),
                MediaPartOrder::Authored,
                ProviderProfile::Generic,
            )
            .ok()
            .map(|(rendered, _)| rendered)
        };
        let prompt = render(&request).unwrap().text;
        if content.is_array() {
            assert!(prompt.starts_with("hi\n<image>"));
        }
        chat_budget_has_disabled_thinking(
            &request,
            ProviderProfile::Generic,
            tokenizer.as_ref(),
            &prompt,
            || ::symphony::formats::for_model_shared(MODEL),
            render,
        )
    };
    let off = json!({"enable_thinking":false});
    for tools in [false, true] {
        assert!(allows(json!(normal), tools, off.clone(), json!("hi")));
    }
    assert!(allows(
        json!(normal),
        false,
        off.clone(),
        json!("What does </think> mean?")
    ));
    assert!(allows(
        json!(normal),
        false,
        off.clone(),
        json!([
            {"type":"text","text":"hi"},
            {"type":"image_url","image_url":{"url":"https://example.com/image.png"}}
        ])
    ));
    let templates =
        json!([{"name":"default","template":normal},{"name":"tool_use","template":named}]);
    assert!(!allows(templates.clone(), true, off.clone(), json!("hi")));
    assert!(allows(
        templates,
        true,
        json!({"thinking_mode":"disabled"}),
        json!("hi")
    ));
    assert!(
        !allows(json!(echo), true, off.clone(), json!(OFF)),
        "a user-supplied close inside the header"
    );
    assert!(
        !allows(
            json!("{# enable_thinking #}{{ messages[-1].content }}"),
            false,
            off,
            json!(OFF)
        ),
        "a close in history"
    );
}

/// A tokenizer shaped like the DeepSeek-V4.1 renderer: `thinking` key,
/// native effort names, thinking on by default, and V4.1's rendering
/// capabilities declared.
fn v41_like() -> llm_tokenizer::MockTokenizer {
    llm_tokenizer::MockTokenizer::new()
        .with_thinking_toggle(ThinkingToggle::DefaultOn)
        .with_thinking_key_name(ThinkingKeyName::Thinking)
        .with_native_reasoning_effort_values(&["low", "high", "xhigh", "max"])
        .with_renderer_capabilities(llm_tokenizer::traits::RendererCapabilities {
            media_handling: llm_tokenizer::traits::MediaHandling::Preprocess,
            enable_thinking_alias: true,
            native_assistant_continuation: true,
            raw_tool_call_arguments: true,
            leading_system_only: false,
            json_requires_separate_reasoning: false,
            json_requires_empty_assistant_prefix: false,
        })
}

/// The gateway arms the reasoning parser exactly the way the V4.1
/// renderer picks the mode: `thinking` or its `enable_thinking` alias
/// first, then the effective `reasoning_effort` (`"none"` off, a native
/// name on), then the OpenAI mapping of the top-level field.
#[test]
fn v41_alias_and_kwargs_none_arm_like_the_renderer() {
    let tok = v41_like();
    let alias_off =
        std::collections::HashMap::from([("enable_thinking".to_string(), Value::Bool(false))]);
    assert_eq!(
        extract_thinking_from_kwargs(Some(&alias_off), None, &tok),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(
            Some(&alias_off),
            Some("high"),
            None,
            &tok,
            ProviderProfile::Generic
        ),
        Some(false)
    );
    // `thinking` wins when both keys are present.
    let both = std::collections::HashMap::from([
        ("thinking".to_string(), Value::Bool(true)),
        ("enable_thinking".to_string(), Value::Bool(false)),
    ]);
    assert_eq!(
        extract_thinking_from_kwargs(Some(&both), None, &tok),
        Some(true)
    );
    // Tokenizers that do not declare the alias keep reading their own key only.
    let other = T(llm_tokenizer::MockTokenizer::new());
    assert_eq!(
        extract_thinking_from_kwargs(Some(&alias_off), None, &other),
        None
    );

    // A kwargs `"none"` disarms; a top-level native level is written over
    // it, as the engine renders the request-level field.
    let none_kw = std::collections::HashMap::from([(
        "reasoning_effort".to_string(),
        Value::String("none".to_string()),
    )]);
    assert_eq!(
        extract_template_effort_thinking(Some(&none_kw), None, &tok),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(Some(&none_kw), None, None, &tok, ProviderProfile::Generic),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(
            Some(&none_kw),
            Some("high"),
            None,
            &tok,
            ProviderProfile::Generic
        ),
        Some(true)
    );
    // Native names arm; a top-level "none" disarms; an explicit toggle beats "none".
    assert_eq!(
        resolve_user_thinking(None, Some("xhigh"), None, &tok, ProviderProfile::Generic),
        Some(true)
    );
    assert_eq!(
        resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
        Some(false)
    );
    let explicit_on = std::collections::HashMap::from([
        ("thinking".to_string(), Value::Bool(true)),
        (
            "reasoning_effort".to_string(),
            Value::String("none".to_string()),
        ),
    ]);
    assert_eq!(
        resolve_user_thinking(
            Some(&explicit_on),
            None,
            None,
            &tok,
            ProviderProfile::Generic
        ),
        Some(true)
    );
    assert!(should_mark_reasoning_started(
        resolve_user_thinking(
            Some(&explicit_on),
            None,
            None,
            &tok,
            ProviderProfile::Generic
        ),
        &tok
    ));

    // A native continuation renders the trailing assistant message past
    // its `</think>`, so the parser is not armed for that request even
    // though thinking is on; any other trailing role arms as usual.
    let request = |continue_final: bool, last_role: &str| -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": last_role, "content": "a"}
            ],
            "continue_final_message": continue_final,
        }))
        .expect("chat request")
    };
    assert!(chat_reasoning_starts_in_prefill(
        &request(false, "assistant"),
        &tok
    ));
    assert!(!chat_reasoning_starts_in_prefill(
        &request(true, "assistant"),
        &tok
    ));
    assert!(chat_reasoning_starts_in_prefill(
        &request(true, "user"),
        &tok
    ));
    // Without a generation header nothing opens a `<think>`: a request
    // with `add_generation_prompt: false` and no continued message is not
    // armed; a continued message keeps the rule above.
    let closed = |continue_final: bool, last_role: &str| -> ChatCompletionRequest {
        let mut body = serde_json::to_value(request(continue_final, last_role)).expect("value");
        body["add_generation_prompt"] = Value::Bool(false);
        serde_json::from_value(body).expect("chat request")
    };
    assert!(!chat_reasoning_starts_in_prefill(
        &closed(false, "user"),
        &tok
    ));
    assert!(!chat_reasoning_starts_in_prefill(
        &closed(false, "assistant"),
        &tok
    ));
    assert!(!chat_reasoning_starts_in_prefill(
        &closed(true, "assistant"),
        &tok
    ));
    // The gateway decides from the rendered prompt, not the request: a
    // renderer with an encoder of its own writes the header whatever the
    // request asked (the DeepSeek ports, as the engine's own encoders
    // do), so its prompt arms as before; one that honoured the request
    // rendered no header and does not; a continued message keeps the
    // native rule.
    assert!(chat_reasoning_starts_in_rendered_prefill(
        &closed(false, "user"),
        true,
        false,
        &tok
    ));
    assert!(!chat_reasoning_starts_in_rendered_prefill(
        &closed(false, "user"),
        false,
        false,
        &tok
    ));
    assert!(!chat_reasoning_starts_in_rendered_prefill(
        &closed(true, "assistant"),
        false,
        true,
        &tok
    ));
    // The request-level `high` is written over the kwargs `none`.
    assert!(should_mark_reasoning_started(
        resolve_user_thinking(
            Some(&none_kw),
            Some("high"),
            None,
            &tok,
            ProviderProfile::Generic
        ),
        &tok
    ));
}

/// A Messages prefill (trailing assistant text) continued natively is
/// rendered past the turn's reasoning, so the parser is not armed; a
/// trailing tool call, with or without text, opens a new turn and a
/// popped prefill follows the generation prompt, and both arm as before.
#[test]
fn messages_prefill_continued_natively_does_not_arm() {
    let request = |last: Value| -> openai_protocol::messages::CreateMessageRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "q"}, last]
        }))
        .expect("messages request")
    };
    let prefill = request(serde_json::json!({"role": "assistant", "content": "a"}));
    let tool_use = request(serde_json::json!({"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
    ]}));
    let text_and_tool_use = request(serde_json::json!({"role": "assistant", "content": [
        {"type": "text", "text": "a"},
        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
    ]}));
    let thinking_on =
        || llm_tokenizer::MockTokenizer::new().with_thinking_toggle(ThinkingToggle::DefaultOn);
    let native =
        thinking_on().with_renderer_capabilities(llm_tokenizer::traits::RendererCapabilities {
            native_assistant_continuation: true,
            ..Default::default()
        });
    // Armed from the rendered prompt, as request building arms it.
    let armed = |request: &openai_protocol::messages::CreateMessageRequest,
                 tokenizer: &dyn Tokenizer| {
        let (processed, _) = message_utils::process_messages(
            request,
            tokenizer,
            None,
            None,
            MediaPartOrder::MediaFirst,
            ProviderProfile::for_model(&request.model),
        )
        .expect("rendered");
        messages_reasoning_starts_in_prefill(request, processed.continued_final_message, tokenizer)
    };

    assert!(!armed(&prefill, &native));
    assert!(armed(&tool_use, &native));
    assert!(armed(&text_and_tool_use, &native));
    assert!(armed(&prefill, &thinking_on()));
}

/// A generation prompt that opens `<think>` (thinking on by default) and
/// no rendered assistant turn: a continued final message cannot be cut
/// from the template's own rendering, so its text follows that prompt.
const THINKING_PROMPT_NO_ASSISTANT_TURNS: &str = r"
{%- for m in messages -%}
{%- if m.role != 'assistant' -%}{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}
{%- if enable_thinking is not defined or enable_thinking -%}{{- '<think>\n' -}}{%- endif -%}
{%- endif -%}";

/// The prompt both APIs render for a `Let me` prefill on that template.
const PREFILL_AFTER_THINKING_PROMPT: &str =
    "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nLet me";

fn thinking_prompt_tokenizer() -> llm_tokenizer::TiktokenTokenizer {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
            .expect("tokenizer");
    tokenizer
        .set_chat_template(THINKING_PROMPT_NO_ASSISTANT_TURNS.to_string())
        .expect("template");
    tokenizer
}

/// The reasoning and the answer a fresh qwen3 parser, armed or not, finds
/// in a completion that closes its reasoning and then answers.
fn reasoning_and_answer(armed: bool) -> (String, String) {
    let factory = ReasoningParserFactory::new();
    let mut parser = create_reasoning_parser(&factory, Some("qwen3"), "m").expect("qwen3 parser");
    if armed {
        parser.mark_reasoning_started();
    }
    let parsed = parser
        .detect_and_parse_reasoning(" check.</think>Done")
        .expect("parsed");
    (parsed.reasoning_text, parsed.normal_text)
}

/// Where the template cannot continue a Messages prefill, the prompt ends
/// inside the `<think>` its generation prompt opens. The parser is armed
/// from the rendered prompt, not from the request asking to continue, so
/// the completion's reasoning is parsed as reasoning.
#[test]
fn messages_continuation_after_a_thinking_generation_prompt_arms() {
    let tokenizer = thinking_prompt_tokenizer();
    let request: openai_protocol::messages::CreateMessageRequest =
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "Let me"}
            ]
        }))
        .expect("messages request");
    let (processed, _) = message_utils::process_messages(
        &request,
        &tokenizer,
        None,
        None,
        MediaPartOrder::MediaFirst,
        ProviderProfile::for_model(&request.model),
    )
    .expect("rendered");
    assert_eq!(processed.text, PREFILL_AFTER_THINKING_PROMPT);
    assert!(!processed.continued_final_message);
    let armed = messages_reasoning_starts_in_prefill(
        &request,
        processed.continued_final_message,
        &tokenizer,
    );
    assert_eq!(
        reasoning_and_answer(armed),
        (" check.".to_string(), "Done".to_string())
    );
}

/// The same for a chat `continue_final_message`.
#[test]
fn chat_continuation_after_a_thinking_generation_prompt_arms() {
    let tokenizer = thinking_prompt_tokenizer();
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "Let me"}
        ],
        "continue_final_message": true
    }))
    .expect("chat request");
    let (processed, _) = chat_utils::process_chat_messages_with_placeholders(
        &request,
        &tokenizer,
        None,
        MediaPartOrder::MediaFirst,
        ProviderProfile::for_model(&request.model),
    )
    .expect("rendered");
    assert_eq!(processed.text, PREFILL_AFTER_THINKING_PROMPT);
    assert!(!processed.continued_final_message);
    let armed = chat_reasoning_starts_in_rendered_prefill(
        &request,
        processed.generation_prompt,
        processed.continued_final_message,
        &tokenizer,
    );
    assert_eq!(
        reasoning_and_answer(armed),
        (" check.".to_string(), "Done".to_string())
    );
}

/// Typed toggle: below an explicit kwargs toggle and a native effort, above the OpenAI mapping.
#[test]
fn typed_thinking_toggle_ranks_between_kwargs_and_effort() {
    let k3 = llm_tokenizer::MockTokenizer::new()
        .with_thinking_toggle(ThinkingToggle::DefaultOn)
        .with_thinking_key_name(ThinkingKeyName::Thinking);

    // KVV `thinking:{type:"disabled"}`: chat mode, parser not armed.
    assert_eq!(
        resolve_user_thinking(
            None,
            Some("max"),
            Some(false),
            &k3,
            ProviderProfile::Generic
        ),
        Some(false)
    );
    assert!(!should_mark_reasoning_started(
        resolve_user_thinking(
            None,
            Some("max"),
            Some(false),
            &k3,
            ProviderProfile::Generic
        ),
        &k3
    ));
    // Absent `thinking`, K3 stays thinking-on by default.
    assert!(should_mark_reasoning_started(
        resolve_user_thinking(None, None, None, &k3, ProviderProfile::Generic),
        &k3
    ));
    // An explicit kwargs toggle outranks the typed one.
    let kw_on = std::collections::HashMap::from([("thinking".to_string(), Value::Bool(true))]);
    assert_eq!(
        resolve_user_thinking(
            Some(&kw_on),
            None,
            Some(false),
            &k3,
            ProviderProfile::Generic
        ),
        Some(true)
    );
    // The typed toggle outranks the OpenAI mapping; absent it, the mapping applies.
    assert_eq!(
        resolve_user_thinking(
            None,
            Some("none"),
            Some(true),
            &k3,
            ProviderProfile::Generic
        ),
        Some(true)
    );
    assert_eq!(
        resolve_user_thinking(None, Some("none"), None, &k3, ProviderProfile::Generic),
        Some(false)
    );

    // A native template effort outranks the typed toggle (DeepSeek-V4.1).
    let v41 = v41_like();
    assert_eq!(
        resolve_user_thinking(
            None,
            Some("high"),
            Some(false),
            &v41,
            ProviderProfile::Generic
        ),
        Some(true)
    );
    assert_eq!(
        resolve_user_thinking(
            None,
            Some("medium"),
            Some(false),
            &v41,
            ProviderProfile::Generic
        ),
        Some(false)
    );

    // End to end through the chat request: `thinking.effort` is the effective effort.
    let request = |thinking: Value| -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "q"}],
            "thinking": thinking,
            "reasoning_effort": "none",
        }))
        .expect("chat request")
    };
    assert!(chat_reasoning_starts_in_prefill(
        &request(serde_json::json!({"type": "enabled"})),
        &k3
    ));
    assert!(!chat_reasoning_starts_in_prefill(
        &request(serde_json::json!({"type": "disabled"})),
        &k3
    ));
    assert!(chat_reasoning_starts_in_prefill(
        &request(serde_json::json!({"effort": "high"})),
        &v41
    ));
}

/// A K3 directory with a byte-complete stub vocabulary (every byte one
/// token, no merges) and the real renderer: prompts render and encode
/// without the checkpoint's files.
fn k3_stub() -> (tempfile::TempDir, llm_tokenizer::TiktokenTokenizer) {
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    let dir = tempfile::TempDir::new().unwrap();
    let vocabulary: String = (0..=255u8)
        .map(|byte| format!("{} {byte}\n", BASE64.encode([byte])))
        .collect();
    std::fs::write(dir.path().join("tiktoken.model"), vocabulary).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#,
    )
    .unwrap();
    let tokenizer = llm_tokenizer::TiktokenTokenizer::from_dir(dir.path()).unwrap();
    (dir, tokenizer)
}

/// The K3 parser is armed from the decision the prompt was rendered
/// with, whatever the request's mix of typed toggle, top-level effort and
/// kwargs: the renderer's own rule decides both (an effort word other
/// than `none` leaves the mode to the typed toggle, `none` switches it
/// off, a kwargs `minimal` keeps the default on).
#[test]
fn k3_parser_is_armed_the_way_the_prompt_was_rendered() {
    use crate::routers::grpc::utils::process_chat_messages;

    let (_dir, tokenizer) = k3_stub();
    let request = |fields: Value| -> ChatCompletionRequest {
        let mut body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "q"}],
        });
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).expect("chat request")
    };
    let armed_like_the_prompt = |fields: Value| -> bool {
        let request = request(fields.clone());
        let prompt = process_chat_messages(&request, &tokenizer, None)
            .expect("render")
            .text;
        let opens_think = prompt.ends_with("<|open|>think<|sep|>");
        assert!(
            opens_think || prompt.ends_with("<|open|>response<|sep|>"),
            "{prompt}"
        );
        assert_eq!(
            chat_reasoning_starts_in_prefill(&request, &tokenizer),
            opens_think,
            "{fields}: prompt {prompt:?}"
        );
        opens_think
    };
    // Typed disabled plus an effort word: chat mode on both sides.
    assert!(!armed_like_the_prompt(serde_json::json!({
        "thinking": {"type": "disabled"}, "reasoning_effort": "high"
    })));
    // A kwargs `minimal` is not the renderer's off word: thinking stays on.
    assert!(armed_like_the_prompt(serde_json::json!({
        "chat_template_kwargs": {"reasoning_effort": "minimal"}
    })));
    // `none` switches thinking off, even against a typed enabled.
    assert!(!armed_like_the_prompt(serde_json::json!({
        "chat_template_kwargs": {"reasoning_effort": "none"}, "thinking": {"type": "enabled"}
    })));
    assert!(!armed_like_the_prompt(
        serde_json::json!({"reasoning_effort": "none"})
    ));
    // The top-level `minimal` reaches the renderer as the protocol's off toggle.
    assert!(!armed_like_the_prompt(
        serde_json::json!({"reasoning_effort": "minimal"})
    ));
    assert!(armed_like_the_prompt(
        serde_json::json!({"reasoning_effort": "high"})
    ));
    assert!(!armed_like_the_prompt(serde_json::json!({
        "chat_template_kwargs": {"thinking": false, "reasoning_effort": "high"}
    })));
    // The request-level effort is written over a kwargs entry of any type
    // (the render's merge does the same): the top-level `none` replaces
    // the null entry, so the response channel opens on both sides.
    assert!(!armed_like_the_prompt(serde_json::json!({
        "thinking": {"type": "enabled"}, "reasoning_effort": "none",
        "chat_template_kwargs": {"reasoning_effort": null}
    })));
    assert!(armed_like_the_prompt(serde_json::json!({})));
}

#[test]
fn template_effort_thinking_covers_kwargs_and_top_level_field() {
    let tok = T(llm_tokenizer::MockTokenizer::new());

    // Top-level field arms thinking when the renderer would interpret it.
    assert_eq!(
        extract_template_effort_thinking(None, Some("high"), &tok),
        Some(true)
    );
    // Unrecognized values never arm (the renderer ignores them too).
    assert_eq!(
        extract_template_effort_thinking(None, Some("medium"), &tok),
        None
    );
    // The top-level field wins over a kwargs entry, matching the merge.
    let kwargs = std::collections::HashMap::from([(
        "reasoning_effort".to_string(),
        Value::String("max".to_string()),
    )]);
    assert_eq!(
        extract_template_effort_thinking(Some(&kwargs), Some("medium"), &tok),
        None
    );
    assert_eq!(
        extract_template_effort_thinking(Some(&kwargs), None, &tok),
        Some(true)
    );
    // Renderers without native efforts never arm.
    assert_eq!(
        extract_template_effort_thinking(
            Some(&kwargs),
            Some("high"),
            &llm_tokenizer::MockTokenizer::new()
        ),
        None
    );
}

/// The MiniMax profile reads `reasoning_effort` by its contract: `none`
/// switches thinking off, so the parser is not armed, as the render
/// writes the template's off mode; `minimal` is the lowest level and a
/// level leaves the template's default. The OpenAI reading stays for
/// every other profile.
#[test]
fn minimax_effort_none_disarms_the_parser_as_everywhere() {
    // A tokenizer like the MiniMax M3 template's: the tri-state key, off by
    // default, and no native `reasoning_effort` words of its own.
    let tok = llm_tokenizer::MockTokenizer::new()
        .with_thinking_toggle(ThinkingToggle::DefaultOff)
        .with_thinking_key_name(ThinkingKeyName::ThinkingMode);
    assert_eq!(
        resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Minimax),
        Some(false),
        "MiniMax none"
    );
    for effort in ["minimal", "low", "high", "max"] {
        assert_eq!(
            resolve_user_thinking(None, Some(effort), None, &tok, ProviderProfile::Minimax),
            None,
            "MiniMax {effort}"
        );
    }
    assert_eq!(
        resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Kimi),
        Some(false)
    );
    // The typed toggle still ranks above the profile's reading of the effort.
    assert_eq!(
        resolve_user_thinking(
            None,
            Some("none"),
            Some(true),
            &tok,
            ProviderProfile::Minimax
        ),
        Some(true)
    );
    // A typed `disabled` next to an explicit effort is honoured under
    // MiniMax as everywhere else: the parser is not armed.
    assert_eq!(
        resolve_user_thinking(
            None,
            Some("high"),
            Some(false),
            &tok,
            ProviderProfile::Minimax
        ),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(None, None, Some(false), &tok, ProviderProfile::Minimax),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(
            None,
            Some("high"),
            Some(false),
            &tok,
            ProviderProfile::Generic
        ),
        Some(false)
    );
}

/// An explicit `thinking.type: disabled` wins over `reasoning_effort` at
/// any level, through the request: the prompt renders thinking off and
/// the parser is not armed. The effort alone leaves the template's
/// default; `enabled` arms. (A template with the tri-state toggle, off by
/// default, under the MiniMax profile.)
#[test]
fn thinking_disabled_wins_over_reasoning_effort() {
    let tok = llm_tokenizer::MockTokenizer::new()
        .with_thinking_toggle(ThinkingToggle::DefaultOff)
        .with_thinking_key_name(ThinkingKeyName::ThinkingMode);
    let request = |fields: Value| -> ChatCompletionRequest {
        let mut value = serde_json::json!({
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "Hi"}],
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().cloned().unwrap_or_default());
        serde_json::from_value(value).expect("chat request")
    };
    for (fields, rendered, armed) in [
        (
            serde_json::json!({"thinking": {"type": "disabled"}}),
            Some(false),
            false,
        ),
        (
            serde_json::json!({"reasoning_effort": "high", "thinking": {"type": "disabled"}}),
            Some(false),
            false,
        ),
        (
            serde_json::json!({"reasoning_effort": "low", "thinking": {"type": "disabled"}}),
            Some(false),
            false,
        ),
        (
            serde_json::json!({"thinking": {"type": "disabled", "effort": "high"}}),
            Some(false),
            false,
        ),
        (serde_json::json!({"reasoning_effort": "high"}), None, false),
        (
            serde_json::json!({"thinking": {"type": "enabled"}}),
            Some(true),
            true,
        ),
    ] {
        let request = request(fields.clone());
        assert_eq!(request.template_thinking(), rendered, "{fields}");
        assert_eq!(
            chat_reasoning_starts_in_prefill(&request, &tok),
            armed,
            "{fields}"
        );
    }
}

#[test]
fn thinking_mode_kwargs_map_to_tristate_pref() {
    let tok = T(llm_tokenizer::MockTokenizer::new());
    let kw = |v: Value| std::collections::HashMap::from([("thinking_mode".to_string(), v)]);

    assert_eq!(
        extract_thinking_from_kwargs(Some(&kw(Value::String("enabled".to_string()))), None, &tok),
        Some(true)
    );
    assert_eq!(
        extract_thinking_from_kwargs(Some(&kw(Value::String("disabled".to_string()))), None, &tok),
        Some(false)
    );
    assert_eq!(
        extract_thinking_from_kwargs(Some(&kw(Value::String("adaptive".to_string()))), None, &tok),
        None
    );
    // Non-string values are not a preference for the tri-state key.
    assert_eq!(
        extract_thinking_from_kwargs(Some(&kw(Value::Bool(true))), None, &tok),
        None
    );
    assert_eq!(extract_thinking_from_kwargs(None, None, &tok), None);
}

#[test]
fn create_reasoning_parser_returns_independent_instances() {
    let factory = ReasoningParserFactory::new();

    // qwen3 starts with in_reasoning=false (explicit <think> required).
    let mut a =
        create_reasoning_parser(&factory, None, "qwen3").expect("qwen3 has a reasoning parser");
    let mut b =
        create_reasoning_parser(&factory, None, "qwen3").expect("qwen3 has a reasoning parser");

    // Each call returns an independent instance: state mutated on one parser
    // must not leak into the other (the shared pooled parser the non-streaming
    // path used to take would have violated this).
    a.mark_reasoning_started();
    assert!(a.is_in_reasoning());
    assert!(!b.is_in_reasoning());

    // The untouched instance still parses a full document correctly.
    let rb = b
        .detect_and_parse_reasoning("<think>reasoning</think>answer")
        .unwrap();
    assert_eq!(rb.normal_text, "answer");
    assert_eq!(rb.reasoning_text, "reasoning");
}

#[test]
fn create_reasoning_parser_honors_configured_parser() {
    let factory = ReasoningParserFactory::new();

    let parser = create_reasoning_parser(&factory, Some("qwen3"), "unknown-model")
        .expect("configured qwen3 parser exists");
    assert_eq!(parser.model_type(), "qwen3");
    assert!(check_reasoning_parser_availability(
        &factory,
        Some("qwen3"),
        "unknown-model",
    ));
    for configured in [None, Some("unknown-parser")] {
        assert!(!check_reasoning_parser_availability(
            &factory,
            configured,
            "unknown-model",
        ));
    }
}

#[test]
fn gemma4_rendered_prompt_state_follows_actual_parser_selection() {
    let factory = ReasoningParserFactory::new();
    for configured in [Some("gemma4"), None, Some("unknown-parser")] {
        assert!(check_reasoning_parser_availability(
            &factory,
            configured,
            "google/gemma-4-serve",
        ));
        for (prompt, expected) in [
            ("<|im_start|>model\n", false),
            ("<|im_start|>model<|channel>thought\n", true),
            ("<|im_start|>model<|channel>thought\nplan<channel|>", false),
        ] {
            assert_eq!(
                rendered_prompt_reasoning_in_prefill(
                    &factory,
                    configured,
                    "google/gemma-4-serve",
                    prompt,
                ),
                Some(expected),
                "configured={configured:?}, prompt={prompt:?}"
            );
        }
    }
    // A valid override wins over the model; other models keep their defaults.
    for (configured, model) in [(Some("qwen3"), "gemma-4"), (None, "qwen3")] {
        assert_eq!(
            rendered_prompt_reasoning_in_prefill(
                &factory,
                configured,
                model,
                "<|channel>thought\n",
            ),
            None
        );
    }
}

#[test]
fn inkling_parser_requires_special_tokens() {
    let factory = ReasoningParserFactory::new();

    assert!(reasoning_parser_requires_special_tokens(
        &factory,
        Some("inkling"),
        "served-model"
    ));
    assert!(!reasoning_parser_requires_special_tokens(
        &factory,
        Some("qwen3"),
        "served-model"
    ));
}
