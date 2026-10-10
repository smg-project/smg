//! Unit tests for Messages API streaming & response format (Anthropic spec)
//!
//! Reference: https://docs.anthropic.com/en/api/messages
//!
//! Covers:
//! - MessageStreamEvent serialization (all variants)
//! - Non-streaming Message response golden tests
//! - StopReason round-trip
//! - SSE wire format: `event: {type}\ndata: {json}\n\n`
//! - Complete streaming event sequence lifecycle (text + tool_use)
//! - stream field defaults

use openai_protocol::messages::{
    ContentBlock, ContentBlockDelta, CreateMessageRequest, ErrorResponse, Message, MessageDelta,
    MessageDeltaUsage, MessageStreamEvent, StopReason, Usage,
};
use serde_json::{json, Value};

// =============================================================================
// Helper
// =============================================================================

#[expect(
    clippy::expect_used,
    reason = "test helper — panicking on bad input is intentional"
)]
fn parse_sse_events(raw: &[u8]) -> Vec<(String, Value)> {
    let text = std::str::from_utf8(raw).expect("UTF-8");
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let mut etype = String::new();
        let mut data = String::new();
        for line in block.lines() {
            if let Some(t) = line.strip_prefix("event: ") {
                etype = t.to_string();
            } else if let Some(d) = line.strip_prefix("data: ") {
                data = d.to_string();
            }
        }
        if !etype.is_empty() && !data.is_empty() {
            out.push((etype, serde_json::from_str(&data).expect("valid JSON")));
        }
    }
    out
}

fn make_skeleton_message(id: &str) -> Message {
    Message {
        id: id.to_string(),
        message_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![],
        model: "claude-sonnet-4-5-20250929".to_string(),
        stop_reason: None,
        stop_sequence: None,
        stop_details: None,
        usage: Usage {
            input_tokens: 25,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cache_creation: None,
            server_tool_use: None,
            service_tier: None,
            inference_geo: None,
            output_tokens_details: None,
        },
        container: None,
    }
}

// =============================================================================
// 1. MessageStreamEvent serialization — all variants
// =============================================================================

#[test]
fn test_message_start_event() {
    let event = MessageStreamEvent::MessageStart {
        message: make_skeleton_message("msg_01XFDUDYJgAACzvnptvVoYEL"),
    };
    let v: Value = serde_json::to_value(&event).unwrap();
    assert_eq!(v["type"], "message_start");
    assert_eq!(v["message"]["type"], "message");
    assert_eq!(v["message"]["role"], "assistant");
    assert_eq!(v["message"]["id"], "msg_01XFDUDYJgAACzvnptvVoYEL");
    assert!(v["message"]["content"].as_array().unwrap().is_empty());
    assert!(v["message"]["stop_reason"].is_null());
    assert_eq!(v["message"]["usage"]["input_tokens"], 25);
}

#[test]
fn test_content_block_start_text() {
    let v: Value = serde_json::to_value(&MessageStreamEvent::ContentBlockStart {
        index: 0,
        content_block: ContentBlock::Text {
            text: String::new(),
            citations: None,
        },
    })
    .unwrap();
    assert_eq!(v["type"], "content_block_start");
    assert_eq!(v["index"], 0);
    assert_eq!(v["content_block"]["type"], "text");
    assert_eq!(v["content_block"]["text"], "");
}

#[test]
fn test_content_block_start_tool_use() {
    let v: Value = serde_json::to_value(&MessageStreamEvent::ContentBlockStart {
        index: 1,
        content_block: ContentBlock::ToolUse {
            id: "toolu_01A09q90qw90lq917835lq9".to_string(),
            name: "get_weather".to_string(),
            input: json!({}),
        },
    })
    .unwrap();
    assert_eq!(v["content_block"]["type"], "tool_use");
    assert_eq!(v["content_block"]["id"], "toolu_01A09q90qw90lq917835lq9");
    assert_eq!(v["content_block"]["name"], "get_weather");
    assert_eq!(v["content_block"]["input"], json!({}));
}

#[test]
fn test_content_block_start_thinking() {
    let v: Value = serde_json::to_value(&MessageStreamEvent::ContentBlockStart {
        index: 0,
        content_block: ContentBlock::Thinking {
            thinking: String::new(),
            signature: String::new(),
        },
    })
    .unwrap();
    assert_eq!(v["content_block"]["type"], "thinking");
}

#[test]
fn test_content_block_delta_text() {
    let v: Value = serde_json::to_value(&MessageStreamEvent::ContentBlockDelta {
        index: 0,
        delta: ContentBlockDelta::TextDelta {
            text: "Hello".to_string(),
        },
    })
    .unwrap();
    assert_eq!(v["type"], "content_block_delta");
    assert_eq!(v["delta"]["type"], "text_delta");
    assert_eq!(v["delta"]["text"], "Hello");
}

#[test]
fn test_content_block_delta_input_json() {
    let v: Value = serde_json::to_value(&MessageStreamEvent::ContentBlockDelta {
        index: 1,
        delta: ContentBlockDelta::InputJsonDelta {
            partial_json: "{\"location\": \"San".to_string(),
        },
    })
    .unwrap();
    assert_eq!(v["delta"]["type"], "input_json_delta");
    assert_eq!(v["delta"]["partial_json"], "{\"location\": \"San");
}

#[test]
fn test_content_block_delta_thinking() {
    let v: Value = serde_json::to_value(&MessageStreamEvent::ContentBlockDelta {
        index: 0,
        delta: ContentBlockDelta::ThinkingDelta {
            thinking: "reasoning...".to_string(),
        },
    })
    .unwrap();
    assert_eq!(v["delta"]["type"], "thinking_delta");
    assert_eq!(v["delta"]["thinking"], "reasoning...");
}

#[test]
fn test_content_block_stop() {
    let v: Value =
        serde_json::to_value(&MessageStreamEvent::ContentBlockStop { index: 0 }).unwrap();
    assert_eq!(v["type"], "content_block_stop");
    assert_eq!(v["index"], 0);
}

#[test]
fn test_message_delta_end_turn() {
    let v: Value = serde_json::to_value(&MessageStreamEvent::MessageDelta {
        delta: MessageDelta {
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            stop_details: None,
            container: None,
        },
        usage: MessageDeltaUsage {
            output_tokens: 15,
            input_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            server_tool_use: None,
            output_tokens_details: None,
        },
    })
    .unwrap();
    assert_eq!(v["type"], "message_delta");
    assert_eq!(v["delta"]["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["output_tokens"], 15);
}

#[test]
fn test_simple_events() {
    // message_stop, ping, error — compact assertions
    let stop: Value = serde_json::to_value(&MessageStreamEvent::MessageStop).unwrap();
    assert_eq!(stop["type"], "message_stop");

    let ping: Value = serde_json::to_value(&MessageStreamEvent::Ping).unwrap();
    assert_eq!(ping["type"], "ping");

    let err: Value = serde_json::to_value(&MessageStreamEvent::Error {
        error: ErrorResponse {
            error_type: "api_error".to_string(),
            message: "fail".to_string(),
        },
    })
    .unwrap();
    assert_eq!(err["type"], "error");
    assert_eq!(err["error"]["type"], "api_error");
    assert_eq!(err["error"]["message"], "fail");
}

// =============================================================================
// 2. Non-streaming Message response (Anthropic golden tests)
// =============================================================================

#[test]
fn test_non_streaming_text_response() {
    let golden = json!({
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I assist you today?"}],
        "model": "claude-sonnet-4-5-20250929",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 25, "output_tokens": 150}
    });
    let msg: Message = serde_json::from_value(golden).unwrap();
    assert_eq!(msg.id, "msg_01XFDUDYJgAACzvnptvVoYEL");
    assert_eq!(msg.message_type, "message");
    assert_eq!(msg.role, "assistant");
    assert_eq!(msg.stop_reason, Some(StopReason::EndTurn));
    assert!(msg.stop_sequence.is_none());
    assert_eq!(msg.usage.input_tokens, 25);
    assert_eq!(msg.usage.output_tokens, 150);
    match &msg.content[0] {
        ContentBlock::Text { text, .. } => assert_eq!(text, "Hello! How can I assist you today?"),
        other => panic!("Expected Text, got {other:?}"),
    }
    // Round-trip preserves structure
    let v: Value = serde_json::to_value(&msg).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["content"][0]["type"], "text");
}

#[test]
fn test_non_streaming_tool_use_response() {
    let golden = json!({
        "id": "msg_01Aq9w938a90dw8q",
        "type": "message", "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll check the weather."},
            {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
             "name": "get_weather", "input": {"location": "San Francisco, CA"}}
        ],
        "model": "claude-sonnet-4-5-20250929",
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 100, "output_tokens": 50}
    });
    let msg: Message = serde_json::from_value(golden).unwrap();
    assert_eq!(msg.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(msg.content.len(), 2);
    match &msg.content[1] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "San Francisco, CA");
        }
        other => panic!("Expected ToolUse, got {other:?}"),
    }
}

#[test]
fn test_non_streaming_thinking_response() {
    let golden = json!({
        "id": "msg_thinking_01",
        "type": "message", "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me reason...", "signature": "sig_abc123"},
            {"type": "text", "text": "The answer is 42."}
        ],
        "model": "claude-sonnet-4-5-20250929",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 50, "output_tokens": 200}
    });
    let msg: Message = serde_json::from_value(golden).unwrap();
    assert_eq!(msg.content.len(), 2);
    match &msg.content[0] {
        ContentBlock::Thinking {
            thinking,
            signature,
        } => {
            assert_eq!(thinking, "Let me reason...");
            assert_eq!(signature, "sig_abc123");
        }
        other => panic!("Expected Thinking, got {other:?}"),
    }
}

// =============================================================================
// 3. StopReason round-trip
// =============================================================================

#[test]
fn test_stop_reason_values() {
    for (reason, expected) in [
        (StopReason::EndTurn, "end_turn"),
        (StopReason::MaxTokens, "max_tokens"),
        (StopReason::StopSequence, "stop_sequence"),
        (StopReason::ToolUse, "tool_use"),
        (StopReason::PauseTurn, "pause_turn"),
        (StopReason::Refusal, "refusal"),
    ] {
        let s = serde_json::to_string(&reason).unwrap();
        assert_eq!(s, format!("\"{expected}\""));
        let parsed: StopReason = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, reason);
    }
}

// =============================================================================
// 4. SSE wire format
// =============================================================================

#[test]
fn test_sse_wire_format() {
    let event = MessageStreamEvent::ContentBlockDelta {
        index: 0,
        delta: ContentBlockDelta::TextDelta {
            text: "Hello".to_string(),
        },
    };
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(b"event: content_block_delta\ndata: ");
    serde_json::to_writer(&mut buf, &event).unwrap();
    buf.extend_from_slice(b"\n\n");

    let raw = std::str::from_utf8(&buf).unwrap();
    assert!(raw.starts_with("event: content_block_delta\n"));
    assert!(raw.contains("data: {"));
    assert!(raw.ends_with("\n\n"));

    let events = parse_sse_events(&buf);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "content_block_delta");
    assert_eq!(events[0].1["delta"]["text"], "Hello");
}

#[test]
fn test_sse_message_stop_is_minimal() {
    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(b"event: message_stop\ndata: ");
    serde_json::to_writer(&mut buf, &MessageStreamEvent::MessageStop).unwrap();
    buf.extend_from_slice(b"\n\n");

    let events = parse_sse_events(&buf);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "message_stop");
    assert_eq!(events[0].1.as_object().unwrap().len(), 1); // only "type"
}

// =============================================================================
// 5. Complete streaming lifecycle — text
// =============================================================================

#[test]
fn test_text_streaming_sequence() {
    let events = [
        json!({"type": "message_start", "message": {
            "id": "msg_01", "type": "message", "role": "assistant", "content": [],
            "model": "claude-sonnet-4-5-20250929", "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 25, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}}),
        json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": " world!"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 15}}),
        json!({"type": "message_stop"}),
    ];

    // All deserialize
    for (i, ej) in events.iter().enumerate() {
        let _: MessageStreamEvent =
            serde_json::from_value(ej.clone()).unwrap_or_else(|e| panic!("Event {i} failed: {e}"));
    }

    // First: skeleton message
    match serde_json::from_value::<MessageStreamEvent>(events[0].clone()).unwrap() {
        MessageStreamEvent::MessageStart { message } => {
            assert!(message.content.is_empty());
            assert!(message.stop_reason.is_none());
        }
        _ => panic!("Expected MessageStart"),
    }

    // Last meaningful: stop_reason
    match serde_json::from_value::<MessageStreamEvent>(events[5].clone()).unwrap() {
        MessageStreamEvent::MessageDelta { delta, usage } => {
            assert_eq!(delta.stop_reason, Some(StopReason::EndTurn));
            assert_eq!(usage.output_tokens, 15);
        }
        _ => panic!("Expected MessageDelta"),
    }
}

// =============================================================================
// 6. Complete streaming lifecycle — tool use
// =============================================================================

#[test]
fn test_tool_use_streaming_sequence() {
    let events = [
        json!({"type": "message_start", "message": {
            "id": "msg_t01", "type": "message", "role": "assistant", "content": [],
            "model": "claude-sonnet-4-5-20250929", "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 100, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Let me check."}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_01",
                "name": "get_weather", "input": {}}}),
        json!({"type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"location\": \"San"}}),
        json!({"type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": " Francisco\"}"}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
            "usage": {"output_tokens": 30}}),
        json!({"type": "message_stop"}),
    ];

    for (i, ej) in events.iter().enumerate() {
        let _: MessageStreamEvent = serde_json::from_value(ej.clone())
            .unwrap_or_else(|e| panic!("Tool event {i} failed: {e}"));
    }

    // Partial JSON concatenation => valid JSON
    let mut partial = String::new();
    for ej in &events[5..7] {
        if let MessageStreamEvent::ContentBlockDelta {
            delta: ContentBlockDelta::InputJsonDelta { partial_json },
            ..
        } = serde_json::from_value::<MessageStreamEvent>(ej.clone()).unwrap()
        {
            partial.push_str(&partial_json);
        }
    }
    let args: Value = serde_json::from_str(&partial).expect("valid JSON");
    assert_eq!(args["location"], "San Francisco");

    // stop_reason = tool_use
    match serde_json::from_value::<MessageStreamEvent>(events[8].clone()).unwrap() {
        MessageStreamEvent::MessageDelta { delta, .. } => {
            assert_eq!(delta.stop_reason, Some(StopReason::ToolUse));
        }
        _ => panic!("Expected MessageDelta"),
    }
}

// =============================================================================
// 7. stream field defaults
// =============================================================================

#[test]
fn test_stream_field_handling() {
    // stream: false
    let r1: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 100,
        "messages": [{"role": "user", "content": "t"}], "stream": false
    }))
    .unwrap();
    assert!(!r1.is_stream());

    // stream omitted => false
    let r2: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 100,
        "messages": [{"role": "user", "content": "t"}]
    }))
    .unwrap();
    assert!(!r2.is_stream());

    // stream: true
    let r3: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 100,
        "messages": [{"role": "user", "content": "t"}], "stream": true
    }))
    .unwrap();
    assert!(r3.is_stream());
}

// =============================================================================
// 8. message_delta stop_reason variants
// =============================================================================

#[test]
fn test_message_delta_stop_sequence() {
    let event = MessageStreamEvent::MessageDelta {
        delta: MessageDelta {
            stop_reason: Some(StopReason::StopSequence),
            stop_sequence: Some("###END###".to_string()),
            stop_details: None,
            container: None,
        },
        usage: MessageDeltaUsage {
            output_tokens: 42,
            input_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            server_tool_use: None,
            output_tokens_details: None,
        },
    };
    let v: Value = serde_json::to_value(&event).unwrap();
    assert_eq!(v["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(v["delta"]["stop_sequence"], "###END###");

    // Round-trip
    let d: MessageStreamEvent = serde_json::from_value(v).unwrap();
    match d {
        MessageStreamEvent::MessageDelta { delta, .. } => {
            assert_eq!(delta.stop_reason, Some(StopReason::StopSequence));
            assert_eq!(delta.stop_sequence.as_deref(), Some("###END###"));
        }
        _ => panic!("Expected MessageDelta"),
    }
}

#[test]
fn test_message_delta_max_tokens() {
    let d: MessageStreamEvent = serde_json::from_value(json!({
        "type": "message_delta",
        "delta": {"stop_reason": "max_tokens", "stop_sequence": null},
        "usage": {"output_tokens": 1024}
    }))
    .unwrap();
    match d {
        MessageStreamEvent::MessageDelta { delta, usage } => {
            assert_eq!(delta.stop_reason, Some(StopReason::MaxTokens));
            assert!(delta.stop_sequence.is_none());
            assert_eq!(usage.output_tokens, 1024);
        }
        _ => panic!("Expected MessageDelta"),
    }
}
