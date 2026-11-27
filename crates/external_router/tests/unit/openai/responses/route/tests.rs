//! R1 wire-contract tests: the OpenAI-compat Responses router forwards
//! the caller's request body to the upstream provider by serialising the
//! `ResponsesRequest` value (see [`route_responses`] around the
//! `to_value(&request_body)` site). These tests lock the shape that the
//! post-P1 content-part variants produce, so any future change to the
//! serde layer surfaces here before it reaches an upstream.
use openai_protocol::{
    common::{to_value_exact, Detail},
    responses::{
        Annotation, FileDetail, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponsesRequest,
    },
};
use serde_json::json;

fn build_request_with_mixed_content() -> ResponsesRequest {
    ResponsesRequest {
        model: "gpt-5.4".to_string(),
        input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_r1".to_string(),
            role: "user".to_string(),
            content: vec![
                ResponseContentPart::InputText {
                    text: "what is in this image and this file?".to_string(),
                },
                ResponseContentPart::InputImage {
                    detail: Some(Detail::Auto),
                    file_id: Some("file-img".to_string()),
                    image_url: Some("https://example.com/dog.jpg".to_string()),
                },
                ResponseContentPart::InputFile {
                    detail: Some(FileDetail::High),
                    file_data: Some("JVBERi0xLjQK".to_string()),
                    file_id: Some("file-pdf".to_string()),
                    file_url: Some("https://example.com/report.pdf".to_string()),
                    filename: Some("report.pdf".to_string()),
                },
                ResponseContentPart::Refusal {
                    refusal: "I cannot process that request.".to_string(),
                },
            ],
            status: Some("completed".to_string()),
            phase: None,
        }]),
        ..Default::default()
    }
}

/// Exercises the exact `to_value_exact(&request_body)` step `route_responses`
/// uses to build the upstream payload — see `route.rs` handler body.
fn serialize_like_router(req: &ResponsesRequest) -> serde_json::Value {
    to_value_exact(req).expect("router serializes ResponsesRequest without error")
}

#[test]
fn router_serialization_keeps_f32_fields_as_the_client_wrote_them() {
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "temperature": 0.7,
        "top_p": 0.95
    }))
    .expect("responses request");
    let payload = serialize_like_router(&req);
    assert_eq!(payload["top_p"], json!(0.95));
    assert_eq!(payload["temperature"], json!(0.7));
}

#[test]
fn router_serialization_preserves_input_image_fields() {
    let req = build_request_with_mixed_content();
    let payload = serialize_like_router(&req);
    let content = &payload["input"][0]["content"];

    assert_eq!(content[1]["type"], json!("input_image"));
    assert_eq!(content[1]["detail"], json!("auto"));
    assert_eq!(content[1]["file_id"], json!("file-img"));
    assert_eq!(
        content[1]["image_url"],
        json!("https://example.com/dog.jpg")
    );
}

#[test]
fn router_serialization_preserves_input_file_fields() {
    let req = build_request_with_mixed_content();
    let payload = serialize_like_router(&req);
    let content = &payload["input"][0]["content"];

    assert_eq!(content[2]["type"], json!("input_file"));
    assert_eq!(content[2]["detail"], json!("high"));
    assert_eq!(content[2]["file_data"], json!("JVBERi0xLjQK"));
    assert_eq!(content[2]["file_id"], json!("file-pdf"));
    assert_eq!(
        content[2]["file_url"],
        json!("https://example.com/report.pdf")
    );
    assert_eq!(content[2]["filename"], json!("report.pdf"));
}

#[test]
fn router_serialization_preserves_refusal() {
    let req = build_request_with_mixed_content();
    let payload = serialize_like_router(&req);
    let content = &payload["input"][0]["content"];

    assert_eq!(content[3]["type"], json!("refusal"));
    assert_eq!(
        content[3]["refusal"],
        json!("I cannot process that request.")
    );
}

#[test]
fn router_serialization_omits_empty_input_image_fields() {
    // `file_id` / `image_url` / `detail` are all optional; the wire
    // payload must not carry `null`s when the caller leaves them unset
    // (the #[serde(skip_serializing_if = "Option::is_none")] attributes
    // on ResponseContentPart guarantee this).
    let req = ResponsesRequest {
        model: "gpt-5.4".to_string(),
        input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_sparse".to_string(),
            role: "user".to_string(),
            content: vec![ResponseContentPart::InputImage {
                detail: None,
                file_id: Some("file-only".to_string()),
                image_url: None,
            }],
            status: None,
            phase: None,
        }]),
        ..Default::default()
    };
    let payload = serialize_like_router(&req);
    let image = &payload["input"][0]["content"][0];

    assert_eq!(image["type"], json!("input_image"));
    assert_eq!(image["file_id"], json!("file-only"));
    assert!(
        image.get("detail").is_none(),
        "detail should be omitted when None"
    );
    assert!(
        image.get("image_url").is_none(),
        "image_url should be omitted when None"
    );
}

#[test]
fn router_serialization_round_trips_typed_annotations_on_output_text() {
    // Assistant turns replayed from storage carry `OutputText` with typed
    // annotations. R1 must preserve the annotation union end-to-end.
    let req = ResponsesRequest {
        model: "gpt-5.4".to_string(),
        input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_prior".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: "Here are three citations.".to_string(),
                annotations: vec![
                    Annotation::FileCitation {
                        file_id: "file-1".to_string(),
                        filename: "spec.pdf".to_string(),
                        index: 0,
                    },
                    Annotation::UrlCitation {
                        url: "https://example.com".to_string(),
                        title: "Example".to_string(),
                        start_index: 10,
                        end_index: 24,
                    },
                    Annotation::FilePath {
                        file_id: "file-2".to_string(),
                        index: 2,
                    },
                ],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
            phase: None,
        }]),
        ..Default::default()
    };

    let payload = serialize_like_router(&req);
    let annotations = &payload["input"][0]["content"][0]["annotations"];
    assert_eq!(annotations[0]["type"], json!("file_citation"));
    assert_eq!(annotations[0]["filename"], json!("spec.pdf"));
    assert_eq!(annotations[1]["type"], json!("url_citation"));
    assert_eq!(annotations[1]["url"], json!("https://example.com"));
    assert_eq!(annotations[1]["start_index"], json!(10));
    assert_eq!(annotations[2]["type"], json!("file_path"));
    assert_eq!(annotations[2]["index"], json!(2));
}

#[test]
fn router_serialization_preserves_empty_annotations_on_output_text() {
    let req = ResponsesRequest {
        model: "gpt-5.4".to_string(),
        input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_prior".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: "Prior answer.".to_string(),
                annotations: vec![],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
            phase: None,
        }]),
        ..Default::default()
    };

    let payload = serialize_like_router(&req);
    let output_text = &payload["input"][0]["content"][0];

    assert_eq!(output_text["type"], json!("output_text"));
    assert_eq!(output_text["annotations"], json!([]));
}

#[test]
fn router_serialization_preserves_codex_additional_tools_and_agent_message() {
    let input = json!([
        {
            "type": "additional_tools",
            "id": "at_1",
            "role": "developer",
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "parameters": {"type": "object"}
            }]
        },
        {
            "type": "agent_message",
            "id": "am_1",
            "author": "/root",
            "recipient": "/root/sub",
            "content": [
                {"type": "input_text", "text": "Header"},
                {"type": "encrypted_content", "encrypted_content": "gAAAAB-opaque"}
            ]
        }
    ]);
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": input
    }))
    .expect("Codex request parses");

    let payload = serialize_like_router(&req);
    assert_eq!(payload["input"], input);
}
