use openai_protocol::chat::{ChatCompletionRequest, ChatMessage};
use serde_json::{json, Value};
use validator::Validate;

#[test]
fn empty_user_content_is_valid() {
    for content in [json!(""), json!([])] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": content}]
        }))
        .unwrap();

        request.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&request).unwrap()["messages"][0]["content"],
            content
        );
    }
}

#[test]
fn empty_messages_are_invalid() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test-model",
        "messages": []
    }))
    .unwrap();

    let errors = request.validate().unwrap_err();
    assert_eq!(errors.field_errors()["messages"][0].code, "empty_array");
}

/// The three rules the public API answers with the bound crossed: the code
/// names it (`error.code`), the field is the param, as the gateway's own
/// range and length rules already do.
#[test]
fn bounds_crossed_are_named_as_the_public_api_names_them() {
    let base = json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]});
    for (field, value, code, excerpt) in [
        (
            "top_p",
            json!(1.5),
            "decimal_above_max_value",
            "Expected a value <= 1.0, but got 1.5",
        ),
        (
            "top_p",
            json!(0.0),
            "decimal_below_min_value",
            "Expected a value > 0.0, but got 0",
        ),
        (
            "stop",
            json!(["a", "b", "c", "d", "e"]),
            "array_above_max_length",
            "Expected an array with maximum length 4, but got an array with length 5",
        ),
        (
            "messages",
            json!([]),
            "empty_array",
            "Expected an array with minimum length 1",
        ),
    ] {
        let mut body = base.clone();
        body[field] = value;
        let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let errors = request.validate().unwrap_err();
        let error = &errors.field_errors()[field][0];
        assert_eq!(error.code, code, "{field}");
        assert!(
            error
                .message
                .as_deref()
                .is_some_and(|m| m.contains(excerpt)),
            "{field}: {:?}",
            error.message
        );
    }
    for (field, value) in [
        ("top_p", json!(1.0)),
        ("top_p", json!(0.1)),
        ("stop", json!(["a", "b", "c", "d"])),
    ] {
        let mut body = base.clone();
        body[field] = value;
        let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert!(request.validate().is_ok(), "{field} within its bound");
    }
}

/// An assistant message with the given reasoning keys: `reasoning_content`,
/// `reasoning`, or both.
fn assistant(spellings: Value) -> Value {
    let mut message = json!({"role": "assistant", "content": ""});
    if let (Value::Object(message), Value::Object(spellings)) = (&mut message, spellings) {
        message.extend(spellings);
    }
    message
}

fn request(messages: Vec<Value>) -> Result<ChatCompletionRequest, serde_json::Error> {
    serde_json::from_value(json!({"model": "test-model", "messages": messages}))
}

/// Issue #5092: an agent client sends the model's reasoning under both keys.
/// A serde alias between the two keys refused the message as a duplicate field.
#[test]
fn assistant_reasoning_is_read_from_either_spelling_or_both() {
    const TEXT: &str = "I should run ls.";
    for (spellings, expected) in [
        (json!({"reasoning_content": TEXT}), Some(TEXT)),
        (json!({"reasoning": TEXT}), Some(TEXT)),
        (
            json!({"reasoning": TEXT, "reasoning_content": TEXT}),
            Some(TEXT),
        ),
        // An empty text does not count and cannot disagree.
        (
            json!({"reasoning": "", "reasoning_content": TEXT}),
            Some(TEXT),
        ),
        (
            json!({"reasoning": TEXT, "reasoning_content": ""}),
            Some(TEXT),
        ),
        (json!({"reasoning": null, "reasoning_content": null}), None),
        (json!({}), None),
    ] {
        let message = assistant(spellings);
        let request = request(vec![message.clone()]).unwrap();
        let ChatMessage::Assistant { reasoning, .. } = &request.messages[0] else {
            panic!("{message}: not an assistant message");
        };
        assert_eq!(reasoning.text.as_deref(), expected, "{message}");
    }
}

#[test]
fn assistant_reasoning_in_both_spellings_with_different_text_is_refused() {
    let error = request(vec![assistant(json!({
        "reasoning": "I should run ls.",
        "reasoning_content": "I should run pwd."
    }))])
    .unwrap_err()
    .to_string();
    // Opens with serde's words, so the public code is `invalid_value`. The
    // HTTP path to the message is tested in `validated.rs`.
    assert!(
        error.starts_with("invalid value: `reasoning_content` and `reasoning` disagree"),
        "{error}"
    );
}

/// A repeated reasoning key is refused, as a repeated key of any other field
/// of the message is. A raw body, because `json!` cannot hold a repeated key.
#[test]
fn a_repeated_reasoning_key_is_refused() {
    for key in ["reasoning_content", "reasoning"] {
        let body = format!(
            r#"{{"model":"test-model","messages":[{{"role":"assistant","content":"","{key}":"a","{key}":"b"}}]}}"#
        );
        let error = serde_json::from_str::<ChatCompletionRequest>(&body)
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with(&format!("duplicate field `{key}`")),
            "{key}: {error}"
        );
    }
}

#[test]
fn assistant_reasoning_round_trips_as_reasoning_content_only() {
    let request = request(vec![
        assistant(json!({"reasoning": "I should run ls."})),
        assistant(json!({})),
    ])
    .unwrap();
    let messages = &serde_json::to_value(&request).unwrap()["messages"];
    assert_eq!(
        messages[0],
        json!({"role": "assistant", "content": "", "reasoning_content": "I should run ls."})
    );
    assert_eq!(messages[1], json!({"role": "assistant", "content": ""}));
}

#[test]
fn a_provider_extension_survives_beside_both_reasoning_spellings() {
    // The reasoning reader takes only its two keys, so the Kimi extension
    // still reads `tools`.
    let request = request(vec![assistant(json!({
        "reasoning": "x",
        "reasoning_content": "x",
        "tools": [{"type": "function", "function": {"name": "f"}}]
    }))])
    .unwrap();
    let ChatMessage::Assistant { reasoning, ext, .. } = &request.messages[0] else {
        panic!("not an assistant message");
    };
    assert_eq!(reasoning.text.as_deref(), Some("x"));
    assert!(ext.tools.is_some(), "{:?}", ext.tools);
}

/// The object schemas in `value` whose `role` constant is `role`.
fn message_schemas<'a>(value: &'a Value, role: &str, found: &mut Vec<&'a Value>) {
    match value {
        Value::Object(object) => {
            if object
                .get("properties")
                .and_then(|p| p["role"]["const"].as_str())
                == Some(role)
            {
                found.push(value);
            }
            for child in object.values() {
                message_schemas(child, role, found);
            }
        }
        Value::Array(items) => {
            for child in items {
                message_schemas(child, role, found);
            }
        }
        _ => {}
    }
}

/// The published schema, the source of the generated clients, keeps
/// `reasoning_content` on the assistant message. It shows neither the
/// `reasoning` key (input only, as before) nor the Rust field name.
#[test]
fn the_schema_keeps_reasoning_content_on_the_assistant_message() {
    let schema = serde_json::to_value(schemars::schema_for!(ChatCompletionRequest)).unwrap();
    let mut assistants = Vec::new();
    message_schemas(&schema, "assistant", &mut assistants);
    assert_eq!(assistants.len(), 1, "{schema}");
    let properties = assistants[0]["properties"].as_object().unwrap();
    assert_eq!(
        properties["reasoning_content"]["type"],
        json!(["string", "null"]),
        "{properties:?}"
    );
    assert!(!properties.contains_key("reasoning"), "{properties:?}");
    assert!(!properties.contains_key("text"), "{properties:?}");
}

/// A replayed assistant turn keeps its refusal, its audio reference and its
/// deprecated function call when the request is written back for an upstream.
/// The crate dropped all three while reading the message, so a provider
/// received a refused turn as a bare `{"role": "assistant"}`.
#[test]
fn an_assistant_turns_refusal_audio_and_function_call_are_written_back() {
    let turn = json!({
        "role": "assistant",
        "content": null,
        "refusal": "I can't help with that.",
        "audio": {"id": "audio_abc123"},
        "function_call": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
    });
    let request = request(vec![
        json!({"role": "user", "content": "What is the weather in Paris?"}),
        turn.clone(),
        json!({"role": "function", "name": "get_weather", "content": "18C"}),
    ])
    .unwrap();
    request.validate().unwrap();

    let ChatMessage::Assistant {
        refusal,
        audio,
        function_call,
        ..
    } = &request.messages[1]
    else {
        panic!("not an assistant message");
    };
    assert_eq!(refusal.as_deref(), Some("I can't help with that."));
    assert_eq!(
        audio.as_ref().map(|audio| audio.id.as_str()),
        Some("audio_abc123")
    );
    assert_eq!(
        function_call.as_ref().map(|call| call.name.as_str()),
        Some("get_weather")
    );

    let written = &serde_json::to_value(&request).unwrap()["messages"][1];
    for key in ["refusal", "audio", "function_call"] {
        assert_eq!(written[key], turn[key], "{key}");
    }
}

/// The public API requires an audio reference's id and a function call's
/// name: a turn that carries either without it is refused while the request
/// is read, where the crate used to drop the key unread.
#[test]
fn an_audio_reference_without_its_id_or_a_function_call_without_its_name_is_refused() {
    for (turn, missing) in [
        (json!({"audio": {}}), "`id`"),
        (json!({"function_call": {"arguments": "{}"}}), "`name`"),
    ] {
        let error = request(vec![
            json!({"role": "user", "content": "hi"}),
            assistant(turn.clone()),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains(missing), "{turn}: {error}");
    }
}

/// An assistant turn without them is written back as before: none of the
/// three keys appears, not even as a null.
#[test]
fn an_assistant_turn_without_them_is_written_back_unchanged() {
    let request = request(vec![
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": "Hello."}),
    ])
    .unwrap();
    assert_eq!(
        serde_json::to_value(&request).unwrap()["messages"][1],
        json!({"role": "assistant", "content": "Hello."})
    );
}

/// The published schema names the three fields on the assistant message, so
/// the generated clients can send them.
#[test]
fn the_schema_carries_the_assistant_turns_refusal_audio_and_function_call() {
    let schema = serde_json::to_value(schemars::schema_for!(ChatCompletionRequest)).unwrap();
    let mut assistants = Vec::new();
    message_schemas(&schema, "assistant", &mut assistants);
    assert_eq!(assistants.len(), 1, "{schema}");
    let properties = assistants[0]["properties"].as_object().unwrap();
    for key in ["refusal", "audio", "function_call"] {
        assert!(properties.contains_key(key), "{key}: {properties:?}");
    }
}

/// A `json_schema` response format keeps its description when the request is
/// written back for an upstream; the crate dropped it while reading the
/// format. Without one, the format is written as before.
#[test]
fn a_json_schema_response_formats_description_is_written_back() {
    let with = json!({"type": "json_schema", "json_schema": {
        "name": "city", "description": "A city and its country.",
        "schema": {"type": "object"}, "strict": true
    }});
    let mut without = with.clone();
    without["json_schema"]
        .as_object_mut()
        .unwrap()
        .remove("description");
    for format in [with, without] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Name a city."}],
            "response_format": format.clone()
        }))
        .unwrap();
        request.validate().unwrap();
        assert!(!request.other.contains_key("response_format"));
        assert_eq!(
            serde_json::to_value(&request).unwrap()["response_format"],
            format
        );
    }
}
