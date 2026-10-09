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
fn explicit_deepseek_variants_preserve_distinct_reasoning_budgets_on_opaque_names() {
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
fn profiles_round_trip_through_typed_configuration_and_reject_unknown_values() {
    for (value, expected) in [
        ("openai", ModelProfile::OpenAi),
        ("kimi", ModelProfile::Kimi),
        ("kimi_k3", ModelProfile::KimiK3),
        ("minimax", ModelProfile::Minimax),
        ("zai", ModelProfile::Zai),
        ("zai_glm_5_3", ModelProfile::ZaiGlm53),
        ("deepseek_v4", ModelProfile::DeepSeekV4),
        ("deepseek_v4_1", ModelProfile::DeepSeekV41),
    ] {
        let parsed: ModelProfile = value.parse().unwrap();
        assert_eq!(parsed, expected);
        assert_eq!(serde_json::to_value(parsed).unwrap(), json!(value));
        assert_eq!(
            serde_json::from_value::<ModelProfile>(json!(value)).unwrap(),
            expected
        );
    }
    assert!("unsupported".parse::<ModelProfile>().is_err());
    assert!(serde_json::from_value::<ModelProfile>(json!("unsupported")).is_err());
}

#[test]
fn canonical_defaults_survive_dispatch_without_exposing_internal_metadata() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"opaque-id","messages":[{"role":"user","content":"Hello"}]
    }))
    .unwrap();
    request.resolved_model_profile = Some(ModelProfile::Zai);
    request.resolved_model_id = Some("zai-org/GLM-4.7".to_string());
    let mut dispatched = request.clone();
    dispatched.model = "another-serving-name".to_string();
    dispatched.normalize();
    assert_eq!(dispatched.temperature, Some(1.0));
    assert_eq!(dispatched.top_p, Some(0.95));
    let json = serde_json::to_value(dispatched).unwrap();
    assert!(json.get("resolved_model_profile").is_none());
    assert!(json.get("resolved_model_id").is_none());
}
