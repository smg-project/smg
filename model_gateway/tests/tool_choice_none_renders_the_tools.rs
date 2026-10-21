//! A request that sends tools with `tool_choice: "none"` renders exactly the
//! prompt the same request renders under `auto`, on the chat path and on the
//! Messages path (`/v1/messages` and its `count_tokens`), so it bills the same
//! prompt tokens and a session that toggles the choice between turns keeps
//! its prefix. The engine's own server renders it so: the tool definitions
//! stay in the prompt under `none` unless an operator excludes them, and the
//! choice reaches the template as a kwarg, where a template that reads it may
//! write an instruction (the Kimi K3 profile's case, `kimi_k3_tool_choice_none`).
//!
//! The renderer is a byte-level vocabulary (one token per UTF-8 byte of the
//! rendered text) with a chat template that writes the tools, so the token
//! counts are exact and the test needs no download.
#![expect(
    clippy::expect_used,
    reason = "a test: a failed step panics with its message"
)]

use std::fs;

use llm_tokenizer::{
    create_tokenizer,
    traits::{PromptEncoding, Tokenizer},
};
use openai_protocol::{
    chat::ChatCompletionRequest, messages::CreateMessageRequest, validated::Normalizable,
};
use serde_json::{json, Map, Value};
use smg::routers::grpc::utils::{
    process_chat_messages_with_encoding, process_messages_with_encoding, request_as_rendered,
    template_tools,
};
use validator::Validate;

/// A template that writes the tools it is given, then the turns: what a
/// checkpoint's own template does, reduced to the bytes that matter here.
const TEMPLATE: &str = "\
{%- if tools %}<|im_start|>system\n# Tools\n\
{%- for tool in tools %}\n{{ tool.function.name }}: {{ tool.function.parameters | tojson }}\
{%- endfor %}\n<|im_end|>\n{%- endif %}\
{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor %}\
{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}";

/// The byte-level pre-tokenizer's character for each byte (GPT-2's table):
/// the printable bytes stand for themselves, the rest take the code points
/// from 256 up, in byte order.
fn byte_level_vocab() -> Map<String, Value> {
    let printable = |b: u8| {
        (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
    };
    let mut vocab = Map::new();
    let mut next = 256u32;
    for byte in 0u32..256 {
        let code = if printable(byte as u8) {
            byte
        } else {
            next += 1;
            next - 1
        };
        let symbol = char::from_u32(code).expect("a code point");
        vocab.insert(symbol.to_string(), json!(byte));
    }
    vocab
}

/// A tokenizer directory whose vocabulary is the 256 byte-level symbols with
/// no merges (every byte is one token) and whose template is [`TEMPLATE`].
fn byte_level_tokenizer() -> (tempfile::TempDir, std::sync::Arc<dyn Tokenizer>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": Value::Object(byte_level_vocab()),
            "merges": []
        }
    });
    fs::write(
        dir.path().join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).expect("json"),
    )
    .expect("tokenizer.json");
    fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"chat_template": TEMPLATE})).expect("json"),
    )
    .expect("tokenizer_config.json");
    let tok = create_tokenizer(dir.path().to_str().expect("utf-8 path")).expect("tokenizer");
    (dir, tok)
}

struct Rendered {
    text: String,
    prompt_tokens: usize,
}

fn rendered(tok: &dyn Tokenizer, text: String, encoding: PromptEncoding) -> Rendered {
    let ids = match encoding {
        PromptEncoding::Deferred(job) => job.run().expect("the renderer's encode"),
        PromptEncoding::FromText => tok.encode(&text, false).expect("encode"),
    };
    let prompt_tokens = ids.token_ids().len();
    assert_eq!(
        prompt_tokens,
        text.len(),
        "the byte-level vocabulary counts one token per byte"
    );
    Rendered {
        text,
        prompt_tokens,
    }
}

fn weather_parameters() -> Value {
    json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
}

/// A chat request as the HTTP layer hands it on, with `tool_choice` as given
/// (`None` leaves the field out: the default is `auto`).
fn chat_request(tools: bool, tool_choice: Option<&str>) -> ChatCompletionRequest {
    let mut body = json!({
        "model": "m1",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    });
    if tools {
        body["tools"] = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "parameters": weather_parameters()
            }
        }]);
    }
    if let Some(choice) = tool_choice {
        body["tool_choice"] = json!(choice);
    }
    let mut request: ChatCompletionRequest =
        serde_json::from_value(body).expect("the gateway accepts the request");
    request.normalize();
    request
        .validate()
        .expect("the gateway validates the request");
    request
}

/// The prompt as the chat preparation stage renders it: the stage's own
/// narrowing rule, the template, the encode the renderer prescribes.
fn render_chat(tok: &dyn Tokenizer, request: &ChatCompletionRequest) -> Rendered {
    let request = request_as_rendered(request);
    let (processed, encoding) =
        process_chat_messages_with_encoding(&request, tok, None).expect("render");
    rendered(tok, processed.text, encoding)
}

/// A Messages request with the same tool and turn, with `tool_choice` as given.
fn messages_request(tool_choice: &str) -> CreateMessageRequest {
    serde_json::from_value(json!({
        "model": "m1",
        "max_tokens": 64,
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "input_schema": weather_parameters()
        }],
        "tool_choice": {"type": tool_choice},
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
    }))
    .expect("the Messages request")
}

/// The prompt as the Messages preparation stage (and `count_tokens`) renders
/// it: the tools the template renders for the choice, then the template.
fn render_messages(tok: &dyn Tokenizer, request: &CreateMessageRequest) -> Rendered {
    let chat_tools = request.chat_tools();
    let tools = template_tools(&chat_tools, request.tool_choice.as_ref());
    let tools_for_template = (!tools.is_empty()).then_some(&*tools);
    let (processed, encoding) =
        process_messages_with_encoding(request, tok, tools_for_template).expect("render");
    rendered(tok, processed.text, encoding)
}

/// On the chat path (the gRPC servicer and the ZMQ link share its preparation
/// stage) `none` renders the tools as `auto` does: the same bytes, the same
/// prompt tokens, more than the request without tools.
#[test]
fn tool_choice_none_renders_the_same_prompt_as_auto_on_the_chat_path() {
    let (_dir, tok) = byte_level_tokenizer();
    let auto = render_chat(tok.as_ref(), &chat_request(true, Some("auto")));
    let none = render_chat(tok.as_ref(), &chat_request(true, Some("none")));
    let unset = render_chat(tok.as_ref(), &chat_request(true, None));
    let without_tools = render_chat(tok.as_ref(), &chat_request(false, None));
    assert!(
        auto.text.contains("get_weather"),
        "the template writes the tool: {:?}",
        auto.text
    );
    assert_eq!(none.text, auto.text, "none renders the bytes auto renders");
    assert_eq!(none.prompt_tokens, auto.prompt_tokens);
    assert_eq!(unset.text, auto.text, "an absent choice is auto");
    assert!(
        none.prompt_tokens > without_tools.prompt_tokens,
        "the tool definitions are in the prompt: {} tokens under none, {} without tools",
        none.prompt_tokens,
        without_tools.prompt_tokens
    );
    assert!(!without_tools.text.contains("get_weather"));
}

/// On the Messages path `{"type": "none"}` renders the tools as `{"type":
/// "auto"}` does, and both render the chat prompt of the same request, so
/// `/v1/messages`, its `count_tokens` and `/v1/chat/completions` bill the
/// same prompt tokens for it.
#[test]
fn tool_choice_none_renders_the_same_prompt_as_auto_on_the_messages_path() {
    let (_dir, tok) = byte_level_tokenizer();
    let auto = render_messages(tok.as_ref(), &messages_request("auto"));
    let none = render_messages(tok.as_ref(), &messages_request("none"));
    assert!(auto.text.contains("get_weather"), "{:?}", auto.text);
    assert_eq!(none.text, auto.text, "none renders the bytes auto renders");
    assert_eq!(none.prompt_tokens, auto.prompt_tokens);
    let chat = render_chat(tok.as_ref(), &chat_request(true, Some("none")));
    assert_eq!(
        none.text, chat.text,
        "the Messages prompt is the chat prompt of the same request"
    );
    assert_eq!(none.prompt_tokens, chat.prompt_tokens);
}
