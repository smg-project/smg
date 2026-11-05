//! z.ai rules for the GLM-5.3 series. The test normalizes first and then
//! validates, as the HTTP extractor does.
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
use serde_json::json;
use validator::Validate;

/// The request from issue #5181. An agent client sends its default effort
/// to a GLM-5.3 model.
#[expect(clippy::expect_used, reason = "test helper")]
fn request(effort: &str) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "zai-org/GLM-5.3",
        "max_tokens": 8,
        "reasoning_effort": effort,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .expect("request")
}

/// `medium` becomes `high`. `minimal` becomes `low` and thinking stays on.
/// `none` is refused.
#[test]
fn a_glm53_request_with_the_default_medium_effort_is_served() {
    let mut req = request("medium");
    req.normalize();
    assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
    assert!(req.validate().is_ok());

    let mut req = request("high");
    req.normalize();
    assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
    assert!(req.validate().is_ok());

    // The series cannot switch thinking off. `minimal` means `low`, not
    // "no thinking".
    let mut req = request("minimal");
    req.normalize();
    assert_eq!(req.reasoning_effort.as_deref(), Some("low"));
    assert_eq!(req.template_thinking(), None);
    assert!(req.validate().is_ok());

    let mut req = request("none");
    req.normalize();
    assert_eq!(req.reasoning_effort.as_deref(), Some("none"));
    let error = req
        .validate()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(error.contains("invalid reasoning_effort"), "{error}");
}
