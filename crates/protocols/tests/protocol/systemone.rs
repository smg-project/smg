use openai_protocol::{
    common::GenerationRequest,
    systemone::{SystemOneAnswer, SystemOneQuestion, SystemOneRequest, SystemOneResponse},
    validated::Normalizable,
};
use serde_json::{json, Value};
use validator::Validate;

fn request(state: Value) -> Value {
    json!({
        "model": "jev-latest",
        "state": state,
        "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}
    })
}

#[test]
fn round_trips_native_questions_and_extensions_without_injecting_defaults() {
    let raw = json!({
        "model": "jev-latest",
        "state": {"ticket": "Connection failed", "active": false, "attempts": 0},
        "questions": {
            "urgency": {"type": "noul", "instructions": null, "criteria": {
                "true": {"condition": "Needs attention", "enabled": true},
                "false": null, "native_criterion": false
            }, "native_question": null},
            "department": {"type": "choice", "criteria": {
                "technical": null, "billing": "Payment issue", "sales": ["Pricing", null, false]
            }, "native_question": true},
            "frustration": {"type": "score", "instructions": {"task": "Rate frustration"},
                "criteria": ["Calm", {"tone": "Frustrated", "enabled": false}, ["Angry", 3]]}
        },
        "chat_template_kwargs": {"enable_thinking": false},
        "native_false": false, "native_true": true, "native_null": null,
        "native_default": 0, "native_empty": [], "stream": false
    });
    let mut parsed: SystemOneRequest = serde_json::from_value(raw.clone()).unwrap();
    parsed.normalize();
    parsed.validate().unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), raw);
    assert_eq!(parsed.get_model(), Some("jev-latest"));
    assert!(!parsed.is_stream());
    assert_eq!(
        parsed.extract_text_for_routing(),
        r#"{"ticket":"Connection failed","active":false,"attempts":0}"#
    );
    assert!(matches!(
        parsed.questions["urgency"],
        SystemOneQuestion::Noul { .. }
    ));
    assert!(matches!(
        parsed.questions["department"],
        SystemOneQuestion::Choice { .. }
    ));
    assert!(matches!(
        parsed.questions["frustration"],
        SystemOneQuestion::Score { .. }
    ));
}

#[test]
fn preserves_question_and_choice_order_used_for_native_label_assignment() {
    let raw = r#"{"model":"jev-latest","state":"text","questions":{"zulu":{"type":"choice","criteria":{"z":null,"a":null,"m":null}},"alpha":{"type":"noul","instructions":"True?"}}}"#;
    let parsed: SystemOneRequest = serde_json::from_str(raw).unwrap();
    let serialized = serde_json::to_value(parsed).unwrap();
    let questions = serialized["questions"].as_object().unwrap();
    assert_eq!(
        questions.keys().map(String::as_str).collect::<Vec<_>>(),
        ["zulu", "alpha"]
    );
    let criteria = questions["zulu"]["criteria"].as_object().unwrap();
    assert_eq!(
        criteria.keys().map(String::as_str).collect::<Vec<_>>(),
        ["z", "a", "m"]
    );
}

#[test]
fn distinguishes_optional_fields_from_explicit_nulls() {
    for question in [
        json!({"type": "noul"}),
        json!({"type": "noul", "instructions": null, "criteria": null}),
        json!({"type": "noul", "criteria": {}}),
        json!({"type": "noul", "criteria": {"true": null}}),
        json!({"type": "noul", "criteria": {"false": "No"}}),
        json!({"type": "choice", "instructions": null, "criteria": {"true": null}}),
        json!({"type": "score", "instructions": null, "criteria": ["Low"]}),
    ] {
        let mut raw = request(json!("text"));
        raw["questions"] = json!({"question": question});
        let parsed: SystemOneRequest = serde_json::from_value(raw.clone()).unwrap();
        parsed.validate().unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), raw);
    }
}

#[test]
fn accepts_structured_state_and_routes_on_its_content() {
    for (state, expected) in [
        (json!("Customer message"), "Customer message"),
        (json!({"ticket": "help"}), r#"{"ticket":"help"}"#),
        (
            json!([{"role": "user", "content": "help"}, false]),
            r#"[{"role":"user","content":"help"},false]"#,
        ),
    ] {
        let parsed: SystemOneRequest = serde_json::from_value(request(state)).unwrap();
        assert_eq!(parsed.extract_text_for_routing(), expected);
    }
}

#[test]
fn rejects_missing_model_and_invalid_native_shapes() {
    let mut missing_model = request(json!("text"));
    missing_model.as_object_mut().unwrap().remove("model");
    assert!(serde_json::from_value::<SystemOneRequest>(missing_model).is_err());
    for field in ["model", "state", "questions"] {
        let mut raw = request(json!("text"));
        raw[field] = Value::Null;
        assert!(
            serde_json::from_value::<SystemOneRequest>(raw).is_err(),
            "accepted null {field}"
        );
    }
    for state in [json!(true), json!(1)] {
        assert!(serde_json::from_value::<SystemOneRequest>(request(state)).is_err());
    }
    for question in [
        json!({"type": "predicate", "instructions": "True?"}),
        json!({"type": "noul", "instructions": false}),
        json!({"type": "noul", "criteria": {"true": false}}),
        json!({"type": "choice", "criteria": ["A", "B"]}),
        json!({"type": "choice", "criteria": {"A": true}}),
        json!({"type": "choice"}),
        json!({"type": "score", "criteria": {"0": "Low"}}),
        json!({"type": "score", "criteria": [null]}),
        json!({"type": "score"}),
    ] {
        let mut raw = request(json!("text"));
        raw["questions"] = json!({"question": question});
        assert!(
            serde_json::from_value::<SystemOneRequest>(raw.clone()).is_err(),
            "accepted {raw}"
        );
    }
    let mut raw = request(json!("text"));
    raw["questions"] = json!([{"type": "noul", "instructions": "True?"}]);
    assert!(serde_json::from_value::<SystemOneRequest>(raw).is_err());
}

#[test]
fn validates_documented_nonempty_questions_and_score_criteria() {
    for questions in [
        json!({}),
        json!({"score": {"type": "score", "criteria": []}}),
    ] {
        let mut raw = request(json!("text"));
        raw["questions"] = questions;
        assert!(serde_json::from_value::<SystemOneRequest>(raw)
            .unwrap()
            .validate()
            .is_err());
    }
}

#[test]
fn leaves_engine_specific_limits_to_the_backend() {
    let mut raw = request(json!("text"));
    raw["questions"] = json!({"score": {"type": "score", "criteria": vec!["level"; 11]}});
    serde_json::from_value::<SystemOneRequest>(raw)
        .unwrap()
        .validate()
        .unwrap();
}

#[test]
fn response_round_trips_named_answers_and_backend_extensions() {
    let raw = json!({
        "model": "jev-1.13.0",
        "answers": {
            "department": {"type": "choice", "choice": "technical", "confidence": 0.78,
                "probabilities": {"technical": 0.85, "sales": 0.0, "billing": 0.15}, "x_label_mass": 0.9},
            "frustration": {"type": "score", "score": 1.0, "confidence": 1.0,
                "legend": {"0": "Calm", "1": {"tone": "Frustrated"}, "2": ["Angry"]},
                "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "native_answer": false},
            "is_urgent": {"type": "noul", "noul": 1.0, "native_answer": null}
        },
        "usage": {"input_tokens": 392, "output_tokens": 0, "native_usage": true},
        "native_response": null
    });
    let parsed: SystemOneResponse = serde_json::from_value(raw.clone()).unwrap();
    assert!(matches!(
        parsed.answers["is_urgent"],
        SystemOneAnswer::Noul { noul: 1.0, .. }
    ));
    assert_eq!(serde_json::to_value(parsed).unwrap(), raw);
    assert!(
        serde_json::from_value::<SystemOneAnswer>(json!({"type": "choice", "choice": true,
        "confidence": 1.0, "probabilities": {"true": 1.0}}))
        .is_err()
    );
}

#[test]
fn generated_schema_exposes_named_maps_and_optional_nullable_instructions() {
    let request = serde_json::to_value(schemars::schema_for!(SystemOneRequest)).unwrap();
    assert_eq!(request["properties"]["questions"]["type"], "object");
    assert_eq!(request["properties"]["questions"]["minProperties"], 1);
    for field in ["model", "state", "questions"] {
        assert!(request["required"]
            .as_array()
            .unwrap()
            .contains(&json!(field)));
    }
    let questions = serde_json::to_value(schemars::schema_for!(SystemOneQuestion)).unwrap();
    for variant in questions["oneOf"].as_array().unwrap() {
        assert!(!variant["required"]
            .as_array()
            .unwrap()
            .contains(&json!("instructions")));
        assert!(variant["properties"]["instructions"]["anyOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|schema| schema["type"] == "null"));
        match variant["properties"]["type"]["const"].as_str().unwrap() {
            "choice" => assert_eq!(variant["properties"]["criteria"]["type"], "object"),
            "score" => {
                assert_eq!(variant["properties"]["criteria"]["type"], "array");
                assert_eq!(variant["properties"]["criteria"]["minItems"], 1);
            }
            "noul" => {}
            other => panic!("unexpected question type {other}"),
        }
    }
    let response = serde_json::to_value(schemars::schema_for!(SystemOneResponse)).unwrap();
    assert_eq!(response["properties"]["answers"]["type"], "object");
}
