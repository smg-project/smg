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
    assert_eq!(
        errors.field_errors()["messages"][0].code,
        "messages cannot be empty"
    );
}
