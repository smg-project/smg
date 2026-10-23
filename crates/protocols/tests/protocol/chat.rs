use openai_protocol::chat::ChatCompletionRequest;
use serde_json::json;
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
