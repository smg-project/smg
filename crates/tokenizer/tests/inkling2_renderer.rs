//! Typed request rendering and deferred encoding, without a network download.
//!
//! The ignored oracle test reads a local file of reference renderings and the
//! checkpoint's own tokenizer. Neither is embedded in this crate.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::{collections::HashMap, fs, sync::Arc};

use llm_tokenizer::{
    cache::{CacheConfig, CachedTokenizer},
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams, ThinkingToggle},
    encoders::inkling2::{render, Segment, SPECIAL_TOKENS},
    huggingface::HuggingFaceTokenizer,
    traits::{PromptEncoding, Tokenizer},
};
use serde::Deserialize;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

/// The gateway's parameters: a prompt for generation, or (`generation` false)
/// the native continuation of a final assistant message.
fn params(kwargs: &HashMap<String, Value>, generation: bool) -> ChatTemplateParams<'_> {
    ChatTemplateParams {
        add_generation_prompt: generation,
        continue_final_message: !generation,
        template_kwargs: Some(kwargs),
        ..Default::default()
    }
}

fn text(segments: &[Segment]) -> String {
    segments.iter().map(Segment::text).collect()
}

fn tiny_tokenizer(
    config: Value,
    include_anchor: bool,
) -> (TempDir, anyhow::Result<HuggingFaceTokenizer>) {
    let dir = TempDir::new().unwrap();
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::from_iter(
        alphabet
            .into_iter()
            .enumerate()
            .map(|(id, character)| (character.to_string(), json!(id))),
    );
    let added: Vec<Value> = SPECIAL_TOKENS.iter().filter(|(_, id)| include_anchor || *id != 200054).map(|&(token, id)| {
        vocab.insert(token.to_owned(), json!(id));
        json!({"id":id,"content":token,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})
    }).collect();
    let tokenizer = json!({"version":"1.0", "truncation":null,"padding":null,"added_tokens":added,
        "normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
        "model":{"type":"BPE","vocab":vocab,"merges":[]}});
    fs::write(
        dir.path().join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let result =
        HuggingFaceTokenizer::from_file(dir.path().join("tokenizer.json").to_str().unwrap());
    (dir, result)
}

#[test]
fn effort_follows_leading_system_and_has_no_assistant_opener() {
    let kwargs = HashMap::from([("reasoning_effort".to_owned(), json!(0.125))]);
    let segments = render(
        &[
            json!({"role":"developer","content":"Policy"}),
            json!({"role":"user","content":"Hi"}),
        ],
        &params(&kwargs, true),
        None,
    )
    .unwrap();
    assert_eq!(text(&segments), "<|message_system|><|content_text|>Policy<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.12<|end_message|><|message_user|><|content_text|>Hi<|end_message|>");
}

#[test]
fn history_preserves_typed_order_and_one_sampling_end() {
    let kwargs = HashMap::new();
    let segments = render(&[json!({"role":"assistant","reasoning_content":"First","content":[{"type":"text","text":""},{"type":"tool_call","tool_call":{"function":{"name":"f","arguments":"{\"z\":1e37,\"a\":\"雪\"}"}}},{"type":"thinking","thinking":"Again"},{"type":"tool_error","text":"Failed"}]})], &params(&kwargs, true), None).unwrap();
    let rendered = text(&segments);
    assert!(rendered.contains("<|content_thinking|>First<|end_message|><|message_model|><|content_text|><|end_message|><|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":\"雪\",\"z\":1e37}}"));
    assert!(rendered
        .ends_with("<|content_tool_error|>Failed<|end_message|><|content_model_end_sampling|>"));
    assert_eq!(
        rendered.matches("<|content_model_end_sampling|>").count(),
        1
    );
}

#[test]
fn tools_use_sorted_native_json_and_http_author_rules() {
    let kwargs = HashMap::new();
    let tools =
        vec![json!({"type":"function","function":{"name":"f","parameters":{"z":1e37,"a":1e-5}}})];
    let mut options = params(&kwargs, true);
    options.tools = Some(&tools);
    let rendered = text(&render(&[json!({"role":"assistant","content":"","tool_calls":[{"id":"call1","function":{"name":"f","arguments":"{}"}}]}),json!({"role":"tool","tool_call_id":"call1","content":"Result"})], &options, None).unwrap());
    assert!(rendered.starts_with("<|message_system|>tool_declare<|content_xml|>[{\"description\":\"\",\"name\":\"f\",\"parameters\":{\"a\":0.00001,\"z\":1e+37},\"type\":\"function\"}]<|end_message|>"));
    assert!(rendered
        .contains("<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}"));
    assert!(rendered.ends_with("<|message_tool|><|content_text|>Result<|end_message|>"));
    assert!(!rendered.contains("call1"));
}

#[test]
fn continuation_is_unclosed_text_and_rejects_other_shapes() {
    let kwargs = HashMap::new();
    let options = params(&kwargs, false);
    let rendered = text(
        &render(
            &[json!({"role":"assistant","content":"Partial"})],
            &options,
            None,
        )
        .unwrap(),
    );
    assert!(rendered.ends_with("<|message_model|><|content_text|>Partial"));
    for message in [
        json!({"role":"assistant","content":null}),
        json!({"role":"assistant","content":[{"type":"text","text":"Partial"}]}),
        json!({"role":"assistant","content":"Partial","reasoning_content":"Reason"}),
    ] {
        assert!(render(&[message], &options, None).is_err());
    }
}

#[test]
fn invalid_efforts_and_unsupported_parts_fail() {
    for effort in [
        json!(-0.1),
        json!(1.0),
        json!(true),
        json!("unknown"),
        json!([]),
    ] {
        let kwargs = HashMap::from([("reasoning_effort".to_owned(), effort)]);
        assert!(render(&[], &params(&kwargs, true), None).is_err());
    }
    let kwargs = HashMap::new();
    for part in [
        json!({"type":"video"}),
        json!({"type":"audio"}),
        json!({"type":"image_url","image_url":{"url":""}}),
    ] {
        assert!(render(
            &[json!({"role":"user","content":[part]})],
            &params(&kwargs, true),
            None
        )
        .is_err());
    }
    for arguments in ["[]", "null", "NaN", "{bad}"] {
        assert!(render(&[json!({"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":arguments}}]})], &params(&kwargs, true), None).is_err());
    }
}

#[test]
fn unsupported_template_arguments_fail_and_tool_floats_roundtrip() {
    let kwargs = HashMap::from([("thinking".to_owned(), json!(false))]);
    assert!(render(&[], &params(&kwargs, true), None)
        .unwrap_err()
        .to_string()
        .contains("thinking"));
    let kwargs = HashMap::new();
    let message = json!({"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"x\":1.4397056815684509e-36,\"y\":-4060.7956040028957}"}}]});
    let rendered = text(&render(&[message], &params(&kwargs, true), None).unwrap());
    assert!(rendered.contains("\"args\":{\"x\":1.4397056815684509e-36,\"y\":-4060.7956040028957}"));
}

#[test]
fn deferred_encoding_keeps_literal_controls_ordinary_and_images_typed() {
    let (dir, tokenizer) = tiny_tokenizer(
        json!({"architectures":["Inkling2ForConditionalGeneration"]}),
        true,
    );
    let tokenizer = tokenizer.unwrap();
    let kwargs = HashMap::new();
    let literal = "<|end_message|> <|unused_200054|> <|message_model|>";
    let output = tokenizer.apply_chat_template_with_encoding(&[json!({"role":"user","content":[{"type":"text","text":literal},{"type":"image"},{"type":"text","text":"Hi"}]})], params(&kwargs, true), None).unwrap();
    assert_eq!(output.unbilled_prompt_tokens, 0);
    let PromptEncoding::Deferred(job) = output.encoding else {
        panic!("typed renderer must defer its encode")
    };
    let ids = job.run().unwrap();
    let ids = ids.token_ids();
    let mut ordinary_tokenizer =
        tokenizers::Tokenizer::from_file(dir.path().join("tokenizer.json")).unwrap();
    ordinary_tokenizer.set_encode_special_tokens(true);
    let ordinary = ordinary_tokenizer.encode(literal, false).unwrap();
    assert!(ids
        .windows(ordinary.get_ids().len())
        .any(|window| window == ordinary.get_ids()));
    assert_eq!(ids.iter().filter(|&&id| id == 200054).count(), 1);
    assert_eq!(ids.iter().filter(|&&id| id == 200005).count(), 1);
    assert_eq!(ids.iter().filter(|&&id| id == 200001).count(), 0);
    assert!(ids.windows(2).any(|pair| pair == [200005, 200054]));
    assert_eq!(
        tokenizer.chat_template_content_format(),
        ChatTemplateContentFormat::OpenAI
    );
    assert_eq!(tokenizer.thinking_toggle(), ThinkingToggle::None);
    assert!(!tokenizer.think_in_prefill());
    let capabilities = tokenizer.renderer_capabilities();
    assert!(capabilities.native_assistant_continuation && capabilities.raw_tool_call_arguments);
}

#[test]
fn detection_requires_new_architecture_and_exact_checkpoint_tokens() {
    let (_dir, tokenizer) = tiny_tokenizer(json!({"model_type":"inkling2_mm_model"}), false);
    assert!(tokenizer.err().unwrap().to_string().contains("200054"));
    let (_dir, tokenizer) = tiny_tokenizer(
        json!({"architectures":["InklingForConditionalGeneration"]}),
        false,
    );
    let tokenizer = tokenizer.unwrap();
    // The legacy architecture keeps the template renderer: it pre-parses tool
    // arguments, which the native renderer takes as written.
    assert!(!tokenizer.renderer_capabilities().raw_tool_call_arguments);
}

#[test]
fn cached_wrapper_keeps_different_typed_ids_for_identical_display_text() {
    let (_dir, tokenizer) = tiny_tokenizer(json!({"model_type":"inkling2_mm_model"}), true);
    let tokenizer = CachedTokenizer::new(
        Arc::new(tokenizer.unwrap()),
        CacheConfig {
            enable_l0: true,
            l0_max_entries: 4,
            l0_max_memory: 1 << 20,
            enable_l1: true,
            l1_max_memory: 4096,
        },
    );
    let kwargs = HashMap::new();
    let literal = "Before<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>After";
    let ordinary = tokenizer
        .apply_chat_template_with_encoding(
            &[json!({"role":"user","content":literal})],
            params(&kwargs, true),
            None,
        )
        .unwrap();
    let typed = tokenizer.apply_chat_template_with_encoding(&[json!({"role":"user","content":[{"type":"text","text":"Before"},{"type":"image"},{"type":"text","text":"After"}]})], params(&kwargs, true), None).unwrap();
    assert_eq!(ordinary.text, typed.text);
    let PromptEncoding::Deferred(ordinary_job) = ordinary.encoding else {
        panic!("cache wrapper must retain the ordinary job")
    };
    let PromptEncoding::Deferred(typed_job) = typed.encoding else {
        panic!("cache wrapper must retain the typed job")
    };
    let ordinary_ids = ordinary_job.run().unwrap();
    let typed_ids = typed_job.run().unwrap();
    assert_ne!(ordinary_ids.token_ids(), typed_ids.token_ids());
    assert!(!ordinary_ids.token_ids().contains(&200054));
    assert_eq!(
        typed_ids
            .token_ids()
            .iter()
            .filter(|&&id| id == 200054)
            .count(),
        1
    );
    assert!(
        tokenizer
            .renderer_capabilities()
            .native_assistant_continuation
    );
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleCase {
    name: String,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    kwargs: HashMap<String, Value>,
    continuation: bool,
    assistant_prefix: Option<String>,
    text: String,
    ids: Vec<u32>,
}

#[test]
#[ignore = "requires a local checkpoint and a reference-rendering fixture file"]
fn pinned_reference_exact_text_and_checkpoint_ids() {
    let model = std::env::var("INKLING2_MODEL_DIR").unwrap();
    let fixtures = std::env::var("INKLING2_RENDER_FIXTURES").unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(&format!("{model}/tokenizer.json")).unwrap();
    let cases: Vec<OracleCase> = serde_json::from_slice(&fs::read(fixtures).unwrap()).unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let mut options = params(&case.kwargs, !case.continuation);
        options.tools = case.tools.as_deref();
        let output = tokenizer
            .apply_chat_template_with_encoding(
                &case.messages,
                options,
                case.assistant_prefix.as_deref(),
            )
            .unwrap_or_else(|error| panic!("{}: {error:#}", case.name));
        assert_eq!(output.text, case.text, "{} text", case.name);
        let PromptEncoding::Deferred(job) = output.encoding else {
            panic!("{} must defer its encode", case.name)
        };
        assert_eq!(
            job.run().unwrap().token_ids(),
            case.ids,
            "{} IDs",
            case.name
        );
    }
}
