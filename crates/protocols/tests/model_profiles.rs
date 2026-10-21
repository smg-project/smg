use openai_protocol::{
    chat::ChatCompletionRequest,
    profile::{ModelProfile, ProviderProfile},
    validated::Normalizable,
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

#[test]
fn configured_deepseek_aliases_preserve_versioned_reasoning() {
    for (alias, canonical, effort, accepts_budget) in [
        (
            "deepseek-v4.1-flash",
            "alias-budget-v41-worker",
            "xhigh",
            true,
        ),
        ("deepseek-v4-pro", "alias-budget-v4-worker", "high", false),
        // A recognized canonical identity overrides a differently versioned alias.
        ("deepseek-v4.1-flash", "deepseek-v4-flash", "high", false),
    ] {
        ProviderProfile::register_model_aliases([(alias, canonical)]);
        for field in ["reasoning_effort", "thinking"] {
            for input in ["xhigh", "80", "101"] {
                let mut body = json!({
                    "model": canonical, "messages": [{"role": "user", "content": "Hello"}]
                });
                body[field] = if field == "thinking" {
                    json!({"effort": input})
                } else {
                    json!(input)
                };
                let mut request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
                request.normalize();
                assert_eq!(
                    request.effective_reasoning_effort(),
                    Some(if input == "xhigh" { effort } else { input }),
                    "{canonical}: {field}={input}"
                );
                assert_eq!(
                    request.validate().is_ok(),
                    input == "xhigh" || (input == "80" && accepts_budget),
                    "{canonical}: {field}={input}"
                );
            }
        }
    }
}

#[test]
fn configured_kimi_and_glm_aliases_preserve_versioned_defaults_and_validation() {
    for (alias, canonical, fields) in [
        (
            "kimi-k3",
            "alias-defaults-k3-worker",
            json!({"temperature": 1.1}),
        ),
        (
            "glm-5.3-flash",
            "alias-defaults-glm53-worker",
            json!({"thinking": {"type": "disabled"}}),
        ),
        ("glm-4.7", "alias-defaults-glm47-worker", json!({})),
    ] {
        ProviderProfile::register_model_aliases([(alias, canonical)]);
        let mut request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": canonical, "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();
        request.normalize();
        assert_eq!(request.temperature, Some(1.0), "{canonical}");
        assert_eq!(request.top_p, Some(0.95), "{canonical}");
        assert!(request.validate().is_ok(), "{canonical}");
        if !fields.as_object().unwrap().is_empty() {
            let mut body = serde_json::to_value(request).unwrap();
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let mut invalid: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            invalid.normalize();
            assert!(invalid.validate().is_err(), "{canonical}");
        }
    }
}
