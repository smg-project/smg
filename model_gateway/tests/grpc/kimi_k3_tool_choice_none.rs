//! A request that sends tools with `tool_choice: "none"` renders, under the
//! Kimi K3 profile, exactly as the reference renders it: the tool
//! declarations in their system message, then the conversation, then the
//! internal system message that forbids the call. The two cases are the
//! vendor verifier's `k3_tool_none` and `k3_tool_choice_none`
//! (`tests/fixtures/kimi_k3/`), whose prompt-token ground truth the
//! vendor's own service gave.
//!
//! Each case renders on the chat path and on the Messages path (the request
//! as `/v1/messages` carries it, which has no reasoning effort): the Messages
//! prompt is the chat prompt of the same request without its effort.
//!
//! The rendering tests run on a byte-level stand-in vocabulary with the K3
//! control tokens, so they need no download. The count test is opt-in:
//! `SMG_KIMI_K3_TOKENIZER_DIR` names a directory with the checkpoint's
//! `tiktoken.model`, `tokenizer_config.json` and `config.json`; the prompt
//! tokens the gateway bills (the renderer's own ids, less the response-channel
//! stub it does not bill) must then fall in the verifier's accepted range,
//! `expected..=expected + 3`.
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "a test: a failed step panics with its message; the opt-in test reports what it counted"
)]

use std::{fs, path::PathBuf};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use llm_tokenizer::{
    create_tokenizer,
    traits::{Encoding, PromptEncoding, Tokenizer},
};
use openai_protocol::{
    chat::ChatCompletionRequest, messages::CreateMessageRequest, validated::Normalizable,
};
use serde_json::{json, Value};
use smg::routers::grpc::{
    utils::{
        process_chat_messages_with_encoding, process_messages_with_encoding, request_as_rendered,
        template_tools,
    },
    ProcessedMessages,
};
use validator::Validate;

const TOKENIZER_DIR_ENV: &str = "SMG_KIMI_K3_TOKENIZER_DIR";
/// The verifier accepts `expected..=expected + MAX_PROMPT_TOKEN_OVERCOUNT`.
const MAX_PROMPT_TOKEN_OVERCOUNT: u32 = 3;

struct KitCase {
    id: &'static str,
    request: ChatCompletionRequest,
    /// The request without its `reasoning_effort`, which the Messages API
    /// does not carry: what the Messages path must render.
    without_effort: ChatCompletionRequest,
    /// The request as `/v1/messages` carries it: the tool under its
    /// `input_schema`, `tool_choice: {"type": "none"}`, the same turns.
    messages_request: CreateMessageRequest,
    expected_prompt_tokens: u32,
}

/// A chat request as the gateway's HTTP layer would hand it on: a model
/// name, normalized and validated.
fn chat_request(mut body: Value) -> ChatCompletionRequest {
    body["model"] = Value::String("m1".to_string());
    let mut request: ChatCompletionRequest =
        serde_json::from_value(body).expect("the gateway accepts the request");
    request.normalize();
    request
        .validate()
        .expect("the gateway validates the request");
    request
}

/// A verifier case: the recorded request on both paths.
fn kit_case(id: &'static str) -> KitCase {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/kimi_k3")
        .join(format!("{id}.json"));
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut case: Value = serde_json::from_str(&raw).expect("the fixture is JSON");
    let expected_prompt_tokens = case["expected_prompt_tokens"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .expect("expected_prompt_tokens");
    let body = case["request"].take();
    let tools: Vec<Value> = body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|tool| {
            json!({
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            })
        })
        .collect();
    let messages_request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m1",
        "max_tokens": 64,
        "tools": tools,
        "tool_choice": {"type": body["tool_choice"]},
        "messages": body["messages"],
    }))
    .expect("the Messages shape of the request");
    let mut without_effort = body.clone();
    without_effort
        .as_object_mut()
        .expect("an object")
        .remove("reasoning_effort");
    KitCase {
        id,
        request: chat_request(body),
        without_effort: chat_request(without_effort),
        messages_request,
        expected_prompt_tokens,
    }
}

/// A byte-level tiktoken vocabulary with the K3 control tokens and the
/// config that selects the K3 renderer: every string encodes, so the
/// rendering can be read back as text.
fn k3_stand_in_dir() -> tempfile::TempDir {
    let model: String = (0u32..256)
        .map(|b| format!("{} {b}\n", STANDARD.encode([b as u8])))
        .collect();
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join("tiktoken.model"), model).expect("tiktoken.model");
    fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"added_tokens_decoder": {
            "300": {"content": "<|open|>", "special": false},
            "301": {"content": "<|close|>", "special": false},
            "302": {"content": "<|sep|>", "special": false},
            "303": {"content": "<|end_of_msg|>", "special": true}
        }}"#,
    )
    .expect("tokenizer_config.json");
    fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#,
    )
    .expect("config.json");
    dir
}

struct Rendered {
    text: String,
    /// The prompt tokens the gateway bills: the renderer's own ids less the
    /// response-channel stub.
    billed_prompt_tokens: u32,
}

/// The prompt as the chat preparation stage renders it: the tools narrowed
/// by the stage's rule for this renderer, then the template, then the
/// renderer's own encode.
fn render(tok: &dyn Tokenizer, request: &ChatCompletionRequest) -> Rendered {
    let request = request_as_rendered(request, tok.renderer_capabilities());
    let (processed, encoding) =
        process_chat_messages_with_encoding(&request, tok, None).expect("render");
    billed(tok, processed, encoding)
}

/// The prompt as the Messages preparation stage renders it: the tools the
/// template renders for this renderer, then the template, then the
/// renderer's own encode.
fn render_messages(tok: &dyn Tokenizer, request: &CreateMessageRequest) -> Rendered {
    let chat_tools = request.chat_tools();
    let tools = template_tools(
        &chat_tools,
        request.tool_choice.as_ref(),
        tok.renderer_capabilities(),
    );
    let tools_for_template = if tools.is_empty() {
        None
    } else {
        Some(&*tools)
    };
    let (processed, encoding) =
        process_messages_with_encoding(request, tok, tools_for_template).expect("render");
    billed(tok, processed, encoding)
}

fn billed(tok: &dyn Tokenizer, processed: ProcessedMessages, encoding: PromptEncoding) -> Rendered {
    let ids: Encoding = match encoding {
        PromptEncoding::Deferred(job) => job.run().expect("the renderer's encode"),
        PromptEncoding::FromText => tok.encode(&processed.text, false).expect("encode"),
    };
    let total = u32::try_from(ids.token_ids().len()).expect("prompt length");
    Rendered {
        text: processed.text,
        billed_prompt_tokens: total - processed.unbilled_prompt_tokens,
    }
}

const DECLARATION: &str = "Here are the available tools, described in JSONSchema.";
const FORBIDS_THE_CALL: &str = "You MUST NOT call any tools in the next message.";

/// The tool declaration precedes the conversation and the forbidding
/// instruction follows it, as `build_chat_segments` writes them; the model
/// sees the tool it is told not to call.
fn assert_declares_then_forbids(id: &str, text: &str, user_text: &str) {
    let declaration = text
        .find(DECLARATION)
        .unwrap_or_else(|| panic!("{id}: no tool declaration in {text:?}"));
    assert!(
        text.contains(r#""name":"get_weather""#),
        "{id}: the declaration carries the tool: {text:?}"
    );
    let user = text
        .find(user_text)
        .unwrap_or_else(|| panic!("{id}: no user turn in {text:?}"));
    let forbids = text
        .find(FORBIDS_THE_CALL)
        .unwrap_or_else(|| panic!("{id}: no tool_choice=none message in {text:?}"));
    assert!(
        declaration < user && user < forbids,
        "{id}: declaration at {declaration}, user turn at {user}, instruction at {forbids}"
    );
    assert_eq!(
        text.matches(DECLARATION).count(),
        1,
        "{id}: one declaration"
    );
}

/// Both paths, on the stand-in vocabulary: the chat path renders the case
/// as the reference does, and the Messages path renders the chat prompt of
/// the same request without its effort, byte for byte.
fn assert_both_paths(case: &KitCase, user_text: &str) {
    let dir = k3_stand_in_dir();
    let tok = create_tokenizer(dir.path().to_str().expect("utf-8 path")).expect("tokenizer");
    let on_chat = render(tok.as_ref(), &case.request);
    assert_declares_then_forbids(case.id, &on_chat.text, user_text);
    let on_messages = render_messages(tok.as_ref(), &case.messages_request);
    assert_declares_then_forbids(case.id, &on_messages.text, user_text);
    let chat_without_effort = render(tok.as_ref(), &case.without_effort);
    assert_eq!(
        on_messages.text, chat_without_effort.text,
        "{}: the Messages path renders the chat path's prompt",
        case.id
    );
}

#[test]
fn k3_tool_none_declares_the_tool_and_forbids_the_call() {
    assert_both_paths(&kit_case("k3_tool_none"), "What is the weather in Paris?");
}

#[test]
fn k3_tool_choice_none_declares_the_tool_and_forbids_the_call() {
    assert_both_paths(&kit_case("k3_tool_choice_none"), "北京天气怎么样？");
}

#[test]
fn the_kit_cases_count_the_vendors_prompt_tokens() {
    let Ok(dir) = std::env::var(TOKENIZER_DIR_ENV) else {
        println!("{TOKENIZER_DIR_ENV} is not set: skipping the prompt-token count against the vendor ground truth");
        return;
    };
    let tok = create_tokenizer(&dir).expect("the checkpoint's tokenizer");
    for id in ["k3_tool_none", "k3_tool_choice_none"] {
        let case = kit_case(id);
        let rendered = render(tok.as_ref(), &case.request);
        let accepted =
            case.expected_prompt_tokens..=case.expected_prompt_tokens + MAX_PROMPT_TOKEN_OVERCOUNT;
        println!(
            "{id}: {} prompt tokens billed, the vendor accepts {accepted:?}",
            rendered.billed_prompt_tokens
        );
        assert!(
            accepted.contains(&rendered.billed_prompt_tokens),
            "{id}: {} prompt tokens, the vendor accepts {accepted:?}",
            rendered.billed_prompt_tokens
        );
        let on_messages = render_messages(tok.as_ref(), &case.messages_request);
        let chat_without_effort = render(tok.as_ref(), &case.without_effort);
        println!(
            "{id}: {} prompt tokens billed on the Messages path, {} on the chat path without the effort",
            on_messages.billed_prompt_tokens, chat_without_effort.billed_prompt_tokens
        );
        assert_eq!(
            on_messages.billed_prompt_tokens, chat_without_effort.billed_prompt_tokens,
            "{id}: both paths bill the same prompt"
        );
    }
}
