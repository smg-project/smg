//! Basic tests for Messages API
//!
//! These tests verify request deserialization and the `GenerationRequest`
//! impl (used for routing/streaming on the HTTP backend). End-to-end
//! routing through the HTTP proxy is exercised by
//! `tests/api/messages_api_test.rs`.

use openai_protocol::{
    common::GenerationRequest,
    messages::{CreateMessageRequest, InputContent, Role, SystemContent},
};
use serde_json::json;

#[test]
fn test_create_message_request_deserialization() {
    // Test simple string content
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Hello, Claude!"
            }
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize request");

    assert_eq!(request.model, "claude-sonnet-4-5-20250929");
    assert_eq!(request.max_tokens, 1024);
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].role, Role::User);

    match &request.messages[0].content {
        InputContent::String(s) => assert_eq!(s, "Hello, Claude!"),
        InputContent::Blocks(_) => panic!("Expected string content"),
    }
}

#[test]
fn test_create_message_request_with_system() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert!(request.system.is_some());
    match request.system.unwrap() {
        SystemContent::String(s) => assert_eq!(s, "You are a helpful assistant."),
        SystemContent::Blocks(_) => panic!("Expected string system content"),
    }
}

#[test]
fn test_create_message_request_with_tools() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "test"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert!(request.tools.is_some());
    assert_eq!(request.tools.unwrap().len(), 1);
}

#[test]
fn test_create_message_request_with_stream() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "test"}],
        "stream": true
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert_eq!(request.stream, Some(true));
}

#[test]
fn test_create_message_request_multi_turn() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert_eq!(request.messages.len(), 3);
    assert_eq!(request.messages[0].role, Role::User);
    assert_eq!(request.messages[1].role, Role::Assistant);
    assert_eq!(request.messages[2].role, Role::User);
}

#[test]
fn test_create_message_request_with_temperature() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.7
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert_eq!(request.temperature, Some(0.7));
}

#[test]
fn test_generation_request_impl_is_stream_and_model() {
    let req: CreateMessageRequest = serde_json::from_value(json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();

    assert!(req.is_stream());
    assert_eq!(
        GenerationRequest::get_model(&req),
        Some("claude-sonnet-4-5-20250929")
    );

    let no_stream: CreateMessageRequest = serde_json::from_value(json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    assert!(!no_stream.is_stream());
}

#[test]
fn test_extract_text_for_routing_string_content() {
    let req: CreateMessageRequest = serde_json::from_value(json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
    }))
    .unwrap();

    let text = req.extract_text_for_routing();
    assert_eq!(text, "You are helpful. Hello Hi there How are you?");
}

#[test]
fn test_extract_text_for_routing_text_blocks() {
    let req: CreateMessageRequest = serde_json::from_value(json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16,
        "system": [
            {"type": "text", "text": "Block system 1"},
            {"type": "text", "text": "Block system 2"}
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first chunk"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
                    {"type": "text", "text": "second chunk"}
                ]
            }
        ]
    }))
    .unwrap();

    let text = req.extract_text_for_routing();
    // Image block is skipped; text blocks are concatenated with single-space separators.
    assert_eq!(
        text,
        "Block system 1 Block system 2 first chunk second chunk"
    );
}

#[test]
fn test_extract_text_for_routing_empty_messages_with_no_text() {
    let req: CreateMessageRequest = serde_json::from_value(json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
                ]
            }
        ]
    }))
    .unwrap();

    assert_eq!(req.extract_text_for_routing(), "");
}

#[test]
fn test_create_message_request_with_thinking() {
    let json = json!({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "test"}],
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024
        }
    });

    let request: CreateMessageRequest =
        serde_json::from_value(json).expect("Failed to deserialize");

    assert!(request.thinking.is_some());
}
