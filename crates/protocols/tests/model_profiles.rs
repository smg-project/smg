use openai_protocol::{
    chat::ChatCompletionRequest, profile::ModelProfile, validated::Normalizable,
};
use serde_json::json;
use validator::Validate;

#[test]
fn resolved_k3_contract_survives_serving_name_rewrites_and_clones() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"public-ocid","messages":[{"role":"user","content":"Hello"}],"temperature":1.1
    }))
    .unwrap();
    request.resolved_model_profile = Some(ModelProfile::KimiK3);
    request.normalize();
    assert!(request.validate().is_err());
    let mut dispatched = request.clone();
    dispatched.model = "vllm-model".to_string();
    assert!(dispatched.validate().is_err());
    assert!(serde_json::to_value(dispatched)
        .unwrap()
        .get("resolved_model_profile")
        .is_none());
}

#[test]
fn resolved_deepseek_variants_preserve_distinct_reasoning_budgets_on_opaque_names() {
    for (profile, expected) in [
        (ModelProfile::DeepSeekV4, "high"),
        (ModelProfile::DeepSeekV41, "xhigh"),
    ] {
        let mut request: ChatCompletionRequest=serde_json::from_value(json!({
            "model":"vllm-model","messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"xhigh"
        })).unwrap();
        request.resolved_model_profile = Some(profile);
        request.normalize();
        assert_eq!(request.effective_reasoning_effort(), Some(expected));
        assert!(request.validate().is_ok());
    }
}

#[test]
fn resolved_defaults_survive_dispatch_without_exposing_internal_metadata() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"opaque-id","messages":[{"role":"user","content":"Hello"}]
    }))
    .unwrap();
    request.resolved_model_profile = Some(ModelProfile::ZaiWithSamplingDefaults);
    let mut dispatched = request.clone();
    dispatched.model = "another-serving-name".to_string();
    dispatched.normalize();
    assert_eq!(dispatched.temperature, Some(1.0));
    assert_eq!(dispatched.top_p, Some(0.95));
    let json = serde_json::to_value(dispatched).unwrap();
    assert!(json.get("resolved_model_profile").is_none());
}
