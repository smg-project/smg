use llm_tokenizer::create_tokenizer_from_file;
use serde_json::json;

use super::*;

/// The test tokenizer: the renderer that writes one `role: content` line
/// per message.
fn tokenizer() -> Arc<dyn Tokenizer> {
    create_tokenizer_from_file("mock").unwrap()
}

#[test]
fn the_fingerprint_is_the_reported_version_or_the_gateways_default() {
    assert_eq!(fingerprint_of(Some("v7")), "v7");
    assert_eq!(fingerprint_of(Some(" v7 ")), "v7");
    assert_eq!(fingerprint_of(Some("")), "default");
    assert_eq!(fingerprint_of(Some("  ")), "default");
    assert_eq!(fingerprint_of(None), "default");
}

#[test]
fn a_request_is_rendered_and_tokenized() {
    let request = json!({
        "model": "unmapped-model",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let tokenizer = tokenizer();
    let prepared = prepare_chat(&request.to_string(), tokenizer.as_ref()).unwrap();
    assert_eq!(prepared.request.model, "unmapped-model");
    assert!(prepared.text.contains("hi"), "{}", prepared.text);
    assert_eq!(
        prepared.token_ids,
        tokenizer.encode(&prepared.text, false).unwrap().token_ids()
    );
}

/// A request that is not JSON, or not a chat request, is the parsing
/// error of the entry points, worded as before.
#[test]
fn a_request_that_does_not_parse_is_a_parsing_error() {
    for body in ["not json", r#"{"model": "m"}"#] {
        let (code, message) = prepare_chat(body, tokenizer().as_ref())
            .err()
            .expect("a parse error");
        assert_eq!(code, SglErrorCode::ParsingError, "{body}");
        assert!(
            message.starts_with("Failed to parse request JSON: "),
            "{message}"
        );
    }
}

#[test]
fn finite_json_budgets_require_the_selected_workers_confirmation() {
    let cap = |value: i64| {
        let raw = json!({"model":"m","messages":[],"custom_params":{"thinking_budget":value}});
        serde_json::from_str::<ChatCompletionRequest>(&raw.to_string())
            .unwrap()
            .resolved_thinking_token_budget()
    };
    assert!(cap(-2).is_err());
    assert!(cap(i64::from(u32::MAX) + 1).is_err());
    let unavailable = || std::future::ready(Err::<_, ()>(()));
    assert!(verify_thinking_budget(cap(-1).unwrap(), unavailable).is_ok());
    assert_eq!(
        verify_thinking_budget(cap(0).unwrap(), unavailable)
            .unwrap_err()
            .0,
        SglErrorCode::UnknownError
    );
    for value in [0, u32::MAX] {
        let budget = cap(i64::from(value)).unwrap();
        assert_eq!(budget, Some(value));
        for flag in [None, Some(false), Some(true)] {
            let mut info = GetServerInfoResponse::default();
            if let Some(value) = flag {
                info.server_args
                    .get_or_insert_default()
                    .fields
                    .insert(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL.into(), value.into());
            }
            let lookup = || std::future::ready(Ok::<_, ()>(info));
            assert_eq!(
                verify_thinking_budget(budget, lookup).is_ok(),
                flag == Some(true)
            );
        }
    }
}
