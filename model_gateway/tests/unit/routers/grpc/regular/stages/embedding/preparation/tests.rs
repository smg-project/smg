use axum::http::StatusCode;
use serde_json::json;

use super::*;

/// The error code of a refusal.
async fn code_of(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .expect("the body");
    let body: Value = serde_json::from_slice(&body).expect("the error envelope");
    (
        status,
        body["error"]["code"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
    )
}

#[tokio::test]
async fn the_four_input_shapes_parse_and_the_rest_are_refused_by_name() {
    let texts = embedding_inputs(&json!(["hello", "world"])).expect("a batch of texts");
    assert_eq!(
        texts,
        CompletionPrompt::Texts(vec!["hello".to_owned(), "world".to_owned()])
    );
    assert_eq!(
        embedding_inputs(&json!("hello")).expect("one text"),
        CompletionPrompt::Text("hello".to_owned())
    );
    assert_eq!(
        embedding_inputs(&json!([1, 2, 3])).expect("one input of ids"),
        CompletionPrompt::TokenIds(vec![1, 2, 3])
    );
    assert_eq!(
        embedding_inputs(&json!([[1, 2], [3]])).expect("a batch of id inputs"),
        CompletionPrompt::TokenIdBatches(vec![vec![1, 2], vec![3]])
    );
    for (shape, code) in [
        (json!([{"text": "a"}]), INVALID_INPUT_CODE),
        (json!({"text": "hello"}), INVALID_INPUT_CODE),
        (json!(["hello", 1]), INVALID_INPUT_CODE),
        (json!([-1, 2]), INVALID_INPUT_CODE),
        (json!(""), EMPTY_INPUT_CODE),
        (json!([]), EMPTY_INPUT_CODE),
        (json!(["hello", ""]), EMPTY_INPUT_CODE),
        (json!([[1, 2], []]), EMPTY_INPUT_CODE),
    ] {
        let refused = embedding_inputs(&shape).expect_err("refused");
        let (status, got) = code_of(refused).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{shape}");
        assert_eq!(got, code, "{shape}");
    }
}

#[tokio::test]
async fn an_empty_member_is_refused_by_its_position() {
    let refused = embedding_inputs(&json!(["hello", "", "world"])).expect_err("refused");
    let body = axum::body::to_bytes(refused.into_body(), 4096)
        .await
        .expect("the body");
    let body: Value = serde_json::from_slice(&body).expect("the error envelope");
    assert_eq!(
        body["error"]["message"], "input[1] cannot be empty",
        "{body}"
    );
}

/// The structured-member array is refused by the stage's own arm, not by
/// the deserializer behind it: the code alone cannot tell the two apart,
/// the message names what this route does not serve.
#[tokio::test]
async fn an_array_of_structured_members_is_refused_as_not_served_here() {
    let refused = embedding_inputs(&json!([{"text": "a"}])).expect_err("refused");
    let body = axum::body::to_bytes(refused.into_body(), 4096)
        .await
        .expect("the body");
    let body: Value = serde_json::from_slice(&body).expect("the error envelope");
    assert_eq!(body["error"]["code"], INVALID_INPUT_CODE, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.starts_with("input as an array of objects (structured members) is not served"),
        "{body}"
    );
}
