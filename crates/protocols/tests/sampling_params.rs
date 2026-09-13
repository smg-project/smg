use openai_protocol::{generate::GenerateRequest, sampling_params::SamplingParams};
use serde_json::{json, Value};

fn custom_params_payload() -> Value {
    json!({
        "nested": {
            "list": [1, "two", true, null, {"leaf": null}],
            "object": {"key": "value"}
        },
        "string": "custom",
        "number": 42.5,
        "bool": false,
        "null": null
    })
}

#[test]
fn sampling_params_preserve_custom_params_round_trip() {
    let input = json!({
        "temperature": 0.5,
        "custom_params": custom_params_payload()
    });

    let sampling_params: SamplingParams =
        serde_json::from_value(input.clone()).expect("sampling params should deserialize");
    let output = serde_json::to_value(sampling_params).expect("sampling params should serialize");

    assert_eq!(
        output, input,
        "custom_params must survive the typed round-trip"
    );
}

#[test]
fn generate_request_preserves_nested_custom_params_round_trip() {
    let custom_params = custom_params_payload();
    let input = json!({
        "text": "hello",
        "sampling_params": {
            "temperature": 0.5,
            "custom_params": custom_params.clone()
        }
    });

    let request: GenerateRequest =
        serde_json::from_value(input).expect("generate request should deserialize");
    assert!(
        !request.other.contains_key("custom_params"),
        "nested custom_params must not leak into the request catch-all"
    );

    let output = serde_json::to_value(request).expect("generate request should serialize");
    assert_eq!(
        output.pointer("/sampling_params/custom_params"),
        Some(&custom_params),
        "nested custom_params must survive the typed round-trip"
    );
    assert!(
        output.get("custom_params").is_none(),
        "nested custom_params must not be serialized at the request top level"
    );
}
