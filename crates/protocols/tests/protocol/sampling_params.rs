use openai_protocol::{generate::GenerateRequest, sampling_params::SamplingParams};
use serde_json::{json, Value};
use validator::{Validate, ValidationErrorsKind};

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

/// SGLang's sampling params take seven keys the struct did not type
/// (`sampling_params.py`). A `/generate` body carrying each passes the
/// sampling rules and is written back with it, so the HTTP and PD routers
/// forward it and the gRPC builder sees it.
#[test]
fn generate_request_keeps_sglangs_other_sampling_keys() {
    let sampling_params = json!({
        "max_new_tokens": 8,
        "structural_tag": r#"{"type":"structural_tag","format":{"type":"const_string","value":"ok"}}"#,
        "stop_regex": ["\\d+\\.$", "END"],
        "logit_bias": {"42": -100.0, "7": 2.5},
        "beam_width": 4,
        "spaces_between_special_tokens": false,
        "stream_interval": 2,
        "watermark": {"enabled": true, "context_window": 4}
    });
    let request: GenerateRequest =
        serde_json::from_value(json!({"text": "hello", "sampling_params": sampling_params}))
            .expect("generate request should deserialize");
    assert!(
        request.validate().is_ok(),
        "the keys pass the sampling rules"
    );

    let output = serde_json::to_value(request).expect("generate request should serialize");
    assert_eq!(output["sampling_params"], sampling_params);
    assert!(
        output.get("structural_tag").is_none(),
        "the keys stay under sampling_params"
    );

    let single: SamplingParams = serde_json::from_value(json!({"stop_regex": "END"}))
        .expect("a single stop regex should deserialize");
    assert_eq!(
        serde_json::to_value(single).expect("sampling params should serialize"),
        json!({"stop_regex": "END"})
    );
}

/// A structural tag is the fourth structured output constraint, and SGLang
/// allows one (`sampling_params.py`): beside `regex` it is refused, naming
/// itself. A `beam_width` below one is refused, as SGLang refuses it.
#[test]
fn generate_request_refuses_a_structural_tag_beside_another_constraint() {
    let request = |sampling_params: Value| -> GenerateRequest {
        serde_json::from_value(json!({"text": "hello", "sampling_params": sampling_params}))
            .expect("generate request should deserialize")
    };

    let both = request(json!({"regex": "a+", "structural_tag": "{}"}));
    let errors = both.validate().expect_err("two constraints are refused");
    let Some(ValidationErrorsKind::Struct(nested)) = errors.errors().get("sampling_params") else {
        panic!("the refusal is under sampling_params: {errors:?}");
    };
    let Some(ValidationErrorsKind::Field(found)) = nested.errors().get("__all__") else {
        panic!("the refusal is the one-constraint rule: {nested:?}");
    };
    assert_eq!(found[0].code, "invalid_value");
    assert_eq!(found[0].params["field"], "structural_tag");
    assert_eq!(
        found[0].message.as_deref(),
        Some("only one of regex, ebnf, json_schema, or structural_tag can be set")
    );

    let zero_beams = request(json!({"beam_width": 0}));
    let errors = zero_beams
        .validate()
        .expect_err("a beam width of zero is refused");
    let Some(ValidationErrorsKind::Struct(nested)) = errors.errors().get("sampling_params") else {
        panic!("the refusal is under sampling_params: {errors:?}");
    };
    let Some(ValidationErrorsKind::Field(found)) = nested.errors().get("beam_width") else {
        panic!("the refusal names beam_width: {nested:?}");
    };
    assert_eq!(found[0].code, "range");
}
