use openai_protocol::{
    common::GenerationRequest,
    decisions::{
        DecisionAnswer, DecisionChoice, DecisionQuestion, DecisionResponse, DecisionValue,
        DecisionsRequest,
    },
    validated::Normalizable,
};
use serde_json::{json, Value};
use validator::Validate;

fn request(input: Value) -> Value {
    json!({
        "model": "decision-model",
        "input": input,
        "questions": [{"type": "predicate", "instructions": "Is this positive?"}]
    })
}

#[test]
fn request_round_trips_all_question_types_and_native_extensions() {
    let raw = json!({
        "model": "decision-model",
        "input": [{"role": "user", "type": "message", "native_message": 1,
            "content": [
                {"type": "input_text", "text": "Review this", "native_text": true},
                {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "original", "native_image": {"x": 1}},
                {"type": "input_text", "text": "carefully"}
            ]}],
        "questions": [
            {"type": "predicate", "instructions": "Is it positive?", "native_question": false},
            {"type": "choice", "instructions": "Choose", "name": "same", "choices": [
                {"value": true, "description": "boolean", "native_choice": "keep"},
                {"value": "true"}
            ]},
            {"type": "score", "instructions": "Score", "name": "same", "levels": [
                {"label": "low", "description": "Low quality", "native_level": [1]},
                {"label": "high"}
            ]}
        ],
        "safety_identifier": "opaque-user",
        "native_request": {"feature": true}
    });
    let mut parsed: DecisionsRequest = serde_json::from_value(raw.clone()).unwrap();
    parsed.normalize();
    parsed.validate().unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), raw);
    assert_eq!(parsed.get_model(), Some("decision-model"));
    assert!(!parsed.is_stream());
    assert_eq!(parsed.extract_text_for_routing(), "Review this carefully");
    assert_eq!(parsed.questions[0].name(), None);
    assert_eq!(parsed.questions[1].name(), Some("same"));
    assert_eq!(parsed.questions[2].instructions(), "Score");
    let DecisionQuestion::Choice { choices, .. } = &parsed.questions[1] else {
        panic!("expected choice question");
    };
    assert_eq!(choices[0].value, DecisionValue::Boolean(true));
    assert_eq!(choices[1].value, DecisionValue::String("true".into()));
    assert_ne!(choices[0].value, choices[1].value);
}

#[test]
fn response_round_trips_nullable_names_typed_choices_and_complete_usage() {
    let raw = json!({
        "model": "decision-model",
        "answers": [
            {"type": "predicate", "name": null, "probability": 0.8},
            {"type": "choice", "name": "same", "choice": true, "confidence": 0.7,
                "probabilities": [{"value": true, "probability": 0.7}, {"value": "true", "probability": 0.3}]},
            {"type": "score", "name": "same", "score": 0.25, "confidence": 0.75,
                "probabilities": [{"value": 0, "label": "low", "probability": 0.75}, {"value": 1, "label": "high", "probability": 0.25}]},
            {"type": "refusal", "name": null}
        ],
        "usage": {"input_tokens": 9, "output_tokens": 0, "total_tokens": 9,
            "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 0}}
    });
    let parsed: DecisionResponse = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), raw);
    assert!(matches!(
        &parsed.answers[0],
        DecisionAnswer::Predicate { name: None, .. }
    ));
    assert!(matches!(
        &parsed.answers[1],
        DecisionAnswer::Choice {
            choice: DecisionValue::Boolean(true),
            ..
        }
    ));
    assert!(matches!(
        &parsed.answers[3],
        DecisionAnswer::Refusal { name: None }
    ));
    assert_eq!(parsed.usage.input_tokens_details.cache_write_tokens, 3);
}

#[test]
fn routing_text_preserves_message_and_part_order_without_images() {
    for (input, want) in [
        (json!("A text input"), "A text input"),
        (
            json!([
                {"role": "user", "content": "first"},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "second"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    {"type": "input_text", "text": "third"}
                ]}
            ]),
            "first second third",
        ),
    ] {
        let parsed: DecisionsRequest = serde_json::from_value(request(input)).unwrap();
        parsed.validate().unwrap();
        assert_eq!(parsed.extract_text_for_routing(), want);
    }
}

#[test]
fn rejects_unsupported_input_roles_parts_and_discriminators() {
    for input in [
        json!([{"role": "assistant", "content": "hello"}]),
        json!([{"role": "system", "content": "hello"}]),
        json!([{"role": "developer", "content": "hello"}]),
        json!([{"role": "user", "type": "item_reference", "content": "hello"}]),
        json!([{"role": "user", "type": null, "content": "hello"}]),
        json!([{"role": "user", "content": [{"type": "input_audio", "data": "AA=="}]}]),
        json!([{"role": "user", "content": [{"type": "input_file", "file_id": "file-1"}]}]),
        json!([{"role": "user", "content": [{"type": "input_text"}]}]),
        json!(42),
        json!(null),
    ] {
        assert!(
            serde_json::from_value::<DecisionsRequest>(request(input.clone())).is_err(),
            "accepted {input}"
        );
    }
}

#[test]
fn rejects_non_string_boolean_choice_values_and_null_request_names() {
    for value in [json!(1), json!(null), json!([]), json!({})] {
        let raw =
            json!({"type": "choice", "instructions": "Choose", "choices": [{"value": value}]});
        assert!(serde_json::from_value::<DecisionQuestion>(raw).is_err());
    }
    for raw in [
        json!({"type": "predicate", "instructions": "test", "name": null}),
        json!({"type": "choice", "instructions": "test", "name": null, "choices": []}),
        json!({"type": "score", "instructions": "test", "name": null, "levels": []}),
        json!({"type": "unknown", "instructions": "test"}),
    ] {
        assert!(
            serde_json::from_value::<DecisionQuestion>(raw.clone()).is_err(),
            "accepted {raw}"
        );
    }
}

#[test]
fn validates_data_urls_and_request_wide_image_limit() {
    for url in [
        "https://example.com/image.png",
        "file-1",
        "data:image/png,raw",
        "data:text/plain;base64,AA==",
    ] {
        let parsed: DecisionsRequest =
            serde_json::from_value(request(json!([{"role": "user", "content": [
                {"type": "input_image", "image_url": url}
            ]}])))
            .unwrap();
        assert!(parsed.validate().is_err(), "accepted {url}");
    }
    for (count, valid) in [(128, true), (129, false)] {
        let images =
            vec![json!({"type": "input_image", "image_url": "data:image/png;base64,AA=="}); count];
        let midpoint = count / 2;
        let parsed: DecisionsRequest = serde_json::from_value(request(json!([
            {"role": "user", "content": &images[..midpoint]},
            {"role": "user", "content": &images[midpoint..]}
        ])))
        .unwrap();
        assert_eq!(parsed.validate().is_ok(), valid, "image count {count}");
    }
}

#[test]
fn accepts_all_image_details_and_nullable_safety_identifier() {
    for detail in [
        json!("low"),
        json!("high"),
        json!("auto"),
        json!("original"),
        json!(null),
    ] {
        let mut raw = request(json!([{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/jpeg;base64,AA==", "detail": detail}
        ]}]));
        raw["safety_identifier"] = Value::Null;
        let parsed: DecisionsRequest = serde_json::from_value(raw).unwrap();
        parsed.validate().unwrap();
    }
    let raw = request(json!([{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "medium"}
    ]}]));
    assert!(serde_json::from_value::<DecisionsRequest>(raw).is_err());
}

#[test]
fn validates_safety_identifier_by_character_count() {
    for (count, valid) in [(128, true), (129, false)] {
        let mut raw = request(json!("text"));
        raw["safety_identifier"] = json!("é".repeat(count));
        let parsed: DecisionsRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(parsed.validate().is_ok(), valid);
    }
}

#[test]
fn preserves_openai_empty_strings_and_does_not_apply_backend_option_limits() {
    let raw = json!({"model": "", "input": "", "questions": [
        {"type": "predicate", "instructions": "", "name": ""},
        {"type": "choice", "instructions": "", "choices": (0..27).map(|i| json!({"value": i.to_string()})).collect::<Vec<_>>()},
        {"type": "score", "instructions": "", "levels": (0..11).map(|i| json!({"label": i.to_string()})).collect::<Vec<_>>()}
    ]});
    let parsed: DecisionsRequest = serde_json::from_value(raw.clone()).unwrap();
    parsed.validate().unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), raw);
}

#[test]
fn required_fields_are_not_silently_defaulted() {
    let complete = request(json!("text"));
    for field in ["model", "input", "questions"] {
        let mut raw = complete.clone();
        raw.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<DecisionsRequest>(raw).is_err());
    }
    let usage = json!({"input_tokens": 1, "output_tokens": 0, "total_tokens": 1,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0}});
    for field in [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
    ] {
        let mut raw_usage = usage.clone();
        raw_usage.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<DecisionResponse>(
            json!({"model": "m", "answers": [], "usage": raw_usage})
        )
        .is_err());
    }
    for field in ["cached_tokens", "cache_write_tokens"] {
        let mut raw_usage = usage.clone();
        raw_usage["input_tokens_details"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(serde_json::from_value::<DecisionResponse>(
            json!({"model": "m", "answers": [], "usage": raw_usage})
        )
        .is_err());
    }
}

#[test]
fn answer_names_must_be_present_even_when_null() {
    for mut raw in [
        json!({"type": "predicate", "name": null, "probability": 0.5}),
        json!({"type": "choice", "name": null, "choice": "yes", "confidence": 0.5, "probabilities": []}),
        json!({"type": "score", "name": null, "score": 0.5, "confidence": 0.5, "probabilities": []}),
        json!({"type": "refusal", "name": null}),
    ] {
        serde_json::from_value::<DecisionAnswer>(raw.clone()).unwrap();
        raw.as_object_mut().unwrap().remove("name");
        assert!(
            serde_json::from_value::<DecisionAnswer>(raw.clone()).is_err(),
            "accepted missing name in {raw}"
        );
    }
}

#[test]
fn validates_documented_text_field_maxima() {
    let maximum = "x".repeat(1_048_576);
    let too_long = format!("{maximum}x");
    let mut raw = request(json!("text"));
    raw["model"] = json!(maximum);
    let parsed: DecisionsRequest = serde_json::from_value(raw.clone()).unwrap();
    parsed.validate().unwrap();
    raw["model"] = json!(too_long);
    assert!(serde_json::from_value::<DecisionsRequest>(raw)
        .unwrap()
        .validate()
        .is_err());
    for question in [
        json!({"type": "predicate", "instructions": too_long}),
        json!({"type": "predicate", "instructions": "test", "name": too_long}),
        json!({"type": "choice", "instructions": "test", "choices": [{"value": true, "description": too_long}]}),
        json!({"type": "score", "instructions": "test", "levels": [{"label": too_long}]}),
        json!({"type": "score", "instructions": "test", "levels": [{"label": "ok", "description": too_long}]}),
    ] {
        let mut raw = request(json!("text"));
        raw["questions"] = json!([question]);
        assert!(serde_json::from_value::<DecisionsRequest>(raw)
            .unwrap()
            .validate()
            .is_err());
    }
    let mut raw = request(
        json!([{"role": "user", "content": [{"type": "input_text", "text": "x".repeat(10_485_761)}]}]),
    );
    assert!(serde_json::from_value::<DecisionsRequest>(raw.clone())
        .unwrap()
        .validate()
        .is_err());
    raw["input"][0]["content"][0]["text"] = json!("x".repeat(10_485_760));
    serde_json::from_value::<DecisionsRequest>(raw)
        .unwrap()
        .validate()
        .unwrap();
}

#[test]
fn json_schema_distinguishes_optional_non_null_fields_from_required_nullable_names() {
    let choice = serde_json::to_value(schemars::schema_for!(DecisionChoice)).unwrap();
    assert_eq!(choice["properties"]["description"]["type"], json!("string"));
    assert!(!choice["required"]
        .as_array()
        .unwrap()
        .contains(&json!("description")));
    let questions = serde_json::to_value(schemars::schema_for!(DecisionQuestion)).unwrap();
    for variant in questions["oneOf"].as_array().unwrap() {
        assert_eq!(variant["properties"]["name"]["type"], json!("string"));
        assert!(!variant["required"]
            .as_array()
            .unwrap()
            .contains(&json!("name")));
    }
    let answers = serde_json::to_value(schemars::schema_for!(DecisionAnswer)).unwrap();
    for variant in answers["oneOf"].as_array().unwrap() {
        assert_eq!(
            variant["properties"]["name"]["type"],
            json!(["string", "null"])
        );
        assert!(variant["required"]
            .as_array()
            .unwrap()
            .contains(&json!("name")));
    }
}
