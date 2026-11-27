//! R1 contract: the OpenAI-compat Responses router MUST forward new
//! content-part variants (`input_image`, `input_file`, `refusal`) to the
//! upstream provider untouched. These tests lock that contract for both
//! the default `OpenAIProvider` (pure pass-through) and the `XAIProvider`
//! (which otherwise rewrites `output_text -> input_text`). The `rid` tests at
//! the end pin the other half of a transform: the gateway-only fields leave
//! the payload before a third-party API reads it.

use openai_protocol::{
    common::Detail,
    model_type::Endpoint,
    responses::{
        Annotation, FileDetail, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponsesRequest,
    },
};
use serde_json::{json, to_value, Value};

use super::{GeminiProvider, OpenAIProvider, Provider, SGLangProvider, XAIProvider};

/// Build a `ResponsesRequest` whose single input message carries every
/// variant `ResponseContentPart` exposes after P1, including a `refusal`
/// that came back from a prior turn via `previous_response_id` replay.
fn request_with_all_content_parts() -> ResponsesRequest {
    ResponsesRequest {
        model: "grok-4".to_string(),
        input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_1".to_string(),
            role: "user".to_string(),
            content: vec![
                ResponseContentPart::InputText {
                    text: "describe this".to_string(),
                },
                ResponseContentPart::InputImage {
                    detail: Some(Detail::High),
                    file_id: Some("file-abc".to_string()),
                    image_url: Some("https://example.com/cat.png".to_string()),
                },
                ResponseContentPart::InputFile {
                    detail: Some(FileDetail::Low),
                    file_data: Some("JVBERi0xLjQK".to_string()),
                    file_id: Some("file-xyz".to_string()),
                    file_url: Some("https://example.com/menu.pdf".to_string()),
                    filename: Some("menu.pdf".to_string()),
                },
                ResponseContentPart::Refusal {
                    refusal: "I can't help with that.".to_string(),
                },
                ResponseContentPart::OutputText {
                    text: "prior assistant turn".to_string(),
                    annotations: vec![Annotation::FileCitation {
                        file_id: "file-abc".to_string(),
                        filename: "cat.png".to_string(),
                        index: 0,
                    }],
                    logprobs: None,
                },
            ],
            status: Some("completed".to_string()),
            phase: None,
        }]),
        ..Default::default()
    }
}

fn first_content_array(payload: &Value) -> &Vec<Value> {
    payload["input"][0]["content"]
        .as_array()
        .expect("content array present on first input item")
}

/// The first input message's content parts of `request_with_all_content_parts`
/// after `provider` transformed the serialized request for `endpoint`.
fn transformed_content(provider: impl Provider, endpoint: Endpoint) -> Vec<Value> {
    let mut payload = to_value(request_with_all_content_parts()).expect("serialize request");
    provider
        .transform_request(&mut payload, endpoint)
        .expect("the provider transform succeeds");
    first_content_array(&payload).clone()
}

#[test]
fn openai_provider_passes_input_image_untouched() {
    let content = transformed_content(OpenAIProvider, Endpoint::Responses);
    assert_eq!(content[1]["type"], json!("input_image"));
    assert_eq!(content[1]["detail"], json!("high"));
    assert_eq!(content[1]["file_id"], json!("file-abc"));
    assert_eq!(
        content[1]["image_url"],
        json!("https://example.com/cat.png")
    );
}

#[test]
fn openai_provider_passes_input_file_untouched() {
    let content = transformed_content(OpenAIProvider, Endpoint::Responses);
    assert_eq!(content[2]["type"], json!("input_file"));
    assert_eq!(content[2]["detail"], json!("low"));
    assert_eq!(content[2]["file_data"], json!("JVBERi0xLjQK"));
    assert_eq!(content[2]["file_id"], json!("file-xyz"));
    assert_eq!(
        content[2]["file_url"],
        json!("https://example.com/menu.pdf")
    );
    assert_eq!(content[2]["filename"], json!("menu.pdf"));
}

#[test]
fn openai_provider_passes_refusal_untouched() {
    let content = transformed_content(OpenAIProvider, Endpoint::Responses);
    assert_eq!(content[3]["type"], json!("refusal"));
    assert_eq!(content[3]["refusal"], json!("I can't help with that."));
}

#[test]
fn openai_provider_preserves_typed_annotations_on_output_text() {
    let content = transformed_content(OpenAIProvider, Endpoint::Responses);
    assert_eq!(content[4]["type"], json!("output_text"));
    let annotations = content[4]["annotations"]
        .as_array()
        .expect("typed annotations survive pass-through");
    assert_eq!(annotations.len(), 1);
    assert_eq!(annotations[0]["type"], json!("file_citation"));
    assert_eq!(annotations[0]["file_id"], json!("file-abc"));
    assert_eq!(annotations[0]["filename"], json!("cat.png"));
    assert_eq!(annotations[0]["index"], json!(0));
}

#[test]
fn xai_provider_passes_input_image_untouched() {
    let content = transformed_content(XAIProvider, Endpoint::Responses);
    assert_eq!(content[1]["type"], json!("input_image"));
    assert_eq!(content[1]["detail"], json!("high"));
    assert_eq!(content[1]["file_id"], json!("file-abc"));
    assert_eq!(
        content[1]["image_url"],
        json!("https://example.com/cat.png")
    );
}

#[test]
fn xai_provider_passes_input_file_untouched() {
    let content = transformed_content(XAIProvider, Endpoint::Responses);
    assert_eq!(content[2]["type"], json!("input_file"));
    assert_eq!(content[2]["detail"], json!("low"));
    assert_eq!(content[2]["file_data"], json!("JVBERi0xLjQK"));
    assert_eq!(content[2]["file_id"], json!("file-xyz"));
    assert_eq!(
        content[2]["file_url"],
        json!("https://example.com/menu.pdf")
    );
    assert_eq!(content[2]["filename"], json!("menu.pdf"));
}

#[test]
fn xai_provider_passes_refusal_untouched() {
    let content = transformed_content(XAIProvider, Endpoint::Responses);
    assert_eq!(content[3]["type"], json!("refusal"));
    assert_eq!(content[3]["refusal"], json!("I can't help with that."));
}

#[test]
fn xai_provider_still_rewrites_output_text_to_input_text_and_keeps_other_variants() {
    // xAI's historical behavior: replayed assistant `output_text` parts
    // are rewritten to `input_text` so xAI's Responses backend accepts
    // them. That rewrite MUST NOT touch `input_image`, `input_file`, or
    // `refusal` parts - this test pins both halves of the contract.
    let content = transformed_content(XAIProvider, Endpoint::Responses);
    assert_eq!(content[0]["type"], json!("input_text"));
    assert_eq!(content[1]["type"], json!("input_image"));
    assert_eq!(content[2]["type"], json!("input_file"));
    assert_eq!(content[3]["type"], json!("refusal"));
    // output_text -> input_text rewrite kicks in on index 4.
    assert_eq!(content[4]["type"], json!("input_text"));
    // The rewritten part still carries its annotations untouched.
    let annotations = content[4]["annotations"]
        .as_array()
        .expect("annotations survive type rewrite");
    assert_eq!(annotations.len(), 1);
    assert_eq!(annotations[0]["type"], json!("file_citation"));
}

#[test]
fn xai_provider_is_noop_for_chat_endpoint() {
    // `input` field is untouched because the Responses-specific rewrite
    // only fires for `Endpoint::Responses`.
    let content = transformed_content(XAIProvider, Endpoint::Chat);
    assert_eq!(content[4]["type"], json!("output_text"));
}

/// A chat request carrying the gateway's own `rid`, the field a client sets
/// to name its request and no third-party API accepts.
fn chat_request_with_rid() -> Value {
    json!({
        "model": "m",
        "rid": "conv_1",
        "messages": [{"role": "user", "content": "hi"}]
    })
}

/// `chat_request_with_rid` after `provider` transformed it for the chat
/// endpoint.
fn transformed_chat(provider: impl Provider) -> Value {
    let mut payload = chat_request_with_rid();
    provider
        .transform_request(&mut payload, Endpoint::Chat)
        .expect("the provider transform succeeds");
    payload
}

#[test]
fn openai_provider_drops_the_gateway_rid() {
    let payload = transformed_chat(OpenAIProvider);
    assert!(payload.get("rid").is_none(), "{payload}");
    assert_eq!(payload["messages"][0]["content"], json!("hi"));
}

#[test]
fn gemini_provider_drops_the_gateway_rid() {
    let payload = transformed_chat(GeminiProvider);
    assert!(payload.get("rid").is_none(), "{payload}");
    assert_eq!(payload["messages"][0]["content"], json!("hi"));
}

#[test]
fn xai_provider_drops_the_gateway_rid() {
    let payload = transformed_chat(XAIProvider);
    assert!(payload.get("rid").is_none(), "{payload}");
    assert_eq!(payload["messages"][0]["content"], json!("hi"));
}

/// The engine-shaped provider forwards the payload as sent: an engine reads
/// `rid` for log correlation.
#[test]
fn sglang_provider_keeps_the_gateway_rid() {
    let payload = transformed_chat(SGLangProvider);
    assert_eq!(payload["rid"], json!("conv_1"));
}
