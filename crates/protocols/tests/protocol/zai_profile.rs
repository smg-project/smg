//! z.ai rules for the GLM-5.3 series, through normalization and validation
//! in the order the HTTP extractor runs them.
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
use serde_json::json;
use validator::Validate;

/// The request of issue #5181: an agent client's default effort on a
/// GLM-5.3 model.
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

/// `medium` is served as `high`; `minimal` as `low`, with thinking on;
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

    // The series cannot switch thinking off, so `minimal` means `low`,
    // not "do not think".
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
