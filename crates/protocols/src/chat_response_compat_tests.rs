use serde_json::{json, Value};

use super::{ChatCompletionMessage, ChatMessage, ChatMessageDelta};

#[test]
fn tool_response_keeps_explicit_null_content() {
    let message: ChatCompletionMessage = serde_json::from_value(json!({
        "role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}
        }]
    }))
    .unwrap();
    let wire = serde_json::to_value(message).unwrap();
    assert_eq!(wire.get("content"), Some(&Value::Null));
    assert_eq!(wire["tool_calls"][0]["function"]["name"], "lookup");
}

#[test]
fn reasoning_response_fields_are_equivalent_and_round_trip_as_history() {
    let message = ChatCompletionMessage {
        role: "assistant".into(),
        content: Some("answer".into()),
        tool_calls: None,
        reasoning_content: Some("reason".into()),
    };
    let wire = serde_json::to_value(message).unwrap();
    assert_eq!(wire["reasoning"], "reason");
    assert_eq!(wire["reasoning_content"], "reason");
    let read: ChatCompletionMessage = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(read.reasoning_content.as_deref(), Some("reason"));
    let history: ChatMessage = serde_json::from_value(wire).unwrap();
    assert!(
        matches!(history, ChatMessage::Assistant { reasoning_content: Some(ref text), .. } if text == "reason")
    );
    // Request serialization keeps its existing canonical spelling.
    let request_wire = serde_json::to_value(history).unwrap();
    assert_eq!(request_wire["reasoning_content"], "reason");
    assert!(request_wire.get("reasoning").is_none());
}

#[test]
fn reasoning_deltas_omit_absent_fields_and_preserve_each_spelling() {
    for fields in [
        json!({}),
        json!({"reasoning": null}),
        json!({"reasoning_content": null}),
    ] {
        let delta: ChatMessageDelta = serde_json::from_value(fields).unwrap();
        assert_eq!(serde_json::to_value(delta).unwrap(), json!({}));
    }
    for fields in [
        json!({"reasoning": "x"}),
        json!({"reasoning_content": "x"}),
        json!({"reasoning": "x", "reasoning_content": "x"}),
    ] {
        let delta: ChatMessageDelta = serde_json::from_value(fields).unwrap();
        let wire = serde_json::to_value(&delta).unwrap();
        assert_eq!(wire, json!({"reasoning": "x", "reasoning_content": "x"}));
        let read: ChatMessageDelta = serde_json::from_value(wire).unwrap();
        assert_eq!(read.reasoning_content, delta.reasoning_content);
    }
}

#[test]
fn reasoning_conflicts_are_rejected_instead_of_losing_text() {
    let fields = json!({"role": "assistant", "content": "answer", "reasoning": "a", "reasoning_content": "b"});
    assert!(serde_json::from_value::<ChatMessage>(fields.clone()).is_err());
    assert!(serde_json::from_value::<ChatCompletionMessage>(fields.clone()).is_err());
    assert!(serde_json::from_value::<ChatMessageDelta>(fields).is_err());
}

#[test]
fn text_response_omits_both_absent_reasoning_fields() {
    let message = ChatCompletionMessage {
        role: "assistant".into(),
        content: Some("answer".into()),
        tool_calls: None,
        reasoning_content: None,
    };
    assert_eq!(
        serde_json::to_value(message).unwrap(),
        json!({"role": "assistant", "content": "answer"})
    );
}

#[test]
fn response_schema_advertises_both_reasoning_fields() {
    for schema in [
        schemars::schema_for!(ChatCompletionMessage),
        schemars::schema_for!(ChatMessageDelta),
    ] {
        let schema = serde_json::to_value(schema).unwrap();
        assert!(schema["properties"]["reasoning"].is_object(), "{schema}");
        assert!(
            schema["properties"]["reasoning_content"].is_object(),
            "{schema}"
        );
    }
}
