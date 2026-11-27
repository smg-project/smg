use axum::{body::to_bytes, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;

use super::*;

fn request(input: Value, questions: Value) -> DecisionsRequest {
    let mut fields = Map::new();
    fields.insert("model".into(), json!("test"));
    fields.insert("input".into(), input);
    fields.insert("questions".into(), questions);
    serde_json::from_value(Value::Object(fields)).unwrap()
}

fn all_questions() -> Value {
    json!([
        {"type":"predicate","instructions":"Is it ready?","name":"same"},
        {"type":"choice","instructions":"Select","choices":[{"value":true},{"value":"true","description":"Literal text"}],"name":"same"},
        {"type":"score","instructions":"Rate","levels":[{"label":"low"},{"label":"high","description":"Fully ready"}]}
    ])
}

fn upstream() -> Value {
    json!({"model":"served","answers":{
        "q2":{"type":"score","score":0.75,"confidence":0.5,"probabilities":{"0":0.25,"1":0.75},"legend":{"0":{"label":"low"},"1":{"label":"high","description":"Fully ready"}},"x_label_mass":0.9},
        "q1":{"type":"choice","choice":"o1","confidence":0.6,"probabilities":{"o1":0.8,"o0":0.2},"x_label_mass":0.9},
        "q0":{"type":"noul","noul":0.8,"x_label_mass":0.9}
    },"usage":{"input_tokens":22,"output_tokens":0}})
}

async fn body(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 100_000).await.unwrap()).unwrap()
}

#[test]
fn maps_typed_values_and_duplicate_names_to_private_ids() {
    let adapter = SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
    assert_eq!(
        adapter.request(),
        &json!({"model":"test","state":"Evidence","questions":{
            "q0":{"type":"noul","instructions":"Is it ready?"},
            "q1":{"type":"choice","instructions":"Select","criteria":{"o0":{"value":true},"o1":{"value":"true","description":"Literal text"}}},
            "q2":{"type":"score","instructions":"Rate","criteria":[{"label":"low"},{"label":"high","description":"Fully ready"}]}
        }})
    );
}

#[test]
fn structured_text_evidence_retains_message_and_part_order() {
    let input = json!([
        {"role":"user","type":"message","content":[{"type":"input_text","text":"first"},{"type":"input_text","text":"second"}]},
        {"role":"user","content":"third"}
    ]);
    let adapter = SglangDecisionAdapter::new(&request(input.clone(), all_questions())).unwrap();
    assert_eq!(adapter.request()["state"], input);
}

#[test]
fn rejects_images_and_extensions_instead_of_dropping_them() {
    let image = json!([{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AQ=="}]}]);
    let error = SglangDecisionAdapter::new(&request(image, all_questions()))
        .err()
        .unwrap();
    assert!(error.contains("image"));
    for value in [
        json!({"model":"test","input":"x","questions":[{"type":"predicate","instructions":"x"}],"temperature":1}),
        json!({"model":"test","input":[{"role":"user","content":"x","extra":true}],"questions":[{"type":"predicate","instructions":"x"}]}),
        json!({"model":"test","input":[{"role":"user","content":[{"type":"input_text","text":"x","extra":true}]}],"questions":[{"type":"predicate","instructions":"x"}]}),
        json!({"model":"test","input":"x","questions":[{"type":"predicate","instructions":"x","extra":true}]}),
        json!({"model":"test","input":"x","questions":[{"type":"choice","instructions":"x","choices":[{"value":"a","extra":true}]}]}),
        json!({"model":"test","input":"x","questions":[{"type":"score","instructions":"x","levels":[{"label":"a","extra":true}]}]}),
    ] {
        let request = serde_json::from_value(value).unwrap();
        assert!(
            SglangDecisionAdapter::new(&request)
                .err()
                .unwrap()
                .contains("extra")
                || SglangDecisionAdapter::new(&request)
                    .err()
                    .unwrap()
                    .contains("temperature")
        );
    }
}

#[test]
fn enforces_sglang_bounds_and_accepts_singletons() {
    for (kind, field, item, limit) in [
        ("choice", "choices", json!({"value":"a"}), 255),
        ("score", "levels", json!({"label":"a"}), 10),
    ] {
        for size in [0, limit + 1] {
            let mut question = json!({"type":kind,"instructions":"x"});
            question[field] = json!(vec![item.clone(); size]);
            assert!(SglangDecisionAdapter::new(&request(json!("x"), json!([question]))).is_err());
        }
        let mut question = json!({"type":kind,"instructions":"x"});
        question[field] = json!([item]);
        assert!(SglangDecisionAdapter::new(&request(json!("x"), json!([question]))).is_ok());
    }
    assert!(SglangDecisionAdapter::new(&request(json!(" \n"), all_questions())).is_err());
    assert!(SglangDecisionAdapter::new(&request(
        json!("x"),
        json!([{ "type":"predicate","instructions":" " }])
    ))
    .is_err());
}

#[tokio::test]
async fn converts_answers_in_request_order_with_typed_values_and_usage() {
    let adapter = SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
    let response = adapter
        .convert_response(Json(upstream()).into_response(), 100_000)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body(response).await,
        json!({"model":"served","answers":[
        {"type":"predicate","name":"same","probability":0.8},
        {"type":"choice","name":"same","choice":"true","confidence":0.6,"probabilities":[{"value":true,"probability":0.2},{"value":"true","probability":0.8}]},
        {"type":"score","name":null,"score":0.75,"confidence":0.5,"probabilities":[{"value":0,"label":"low","probability":0.25},{"value":1,"label":"high","probability":0.75}]}
    ],"usage":{"input_tokens":22,"output_tokens":0,"total_tokens":22,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}})
    );
}

#[tokio::test]
async fn preserves_headers_and_status_while_removing_body_validators() {
    let adapter = SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
    let response = Response::builder()
        .status(StatusCode::CREATED)
        .header("x-request-id", "request-42")
        .header("x-smg-routed-worker", "worker")
        .header("content-length", "999")
        .header("content-encoding", "gzip")
        .header("etag", "old")
        .header("content-md5", "old")
        .body(Body::from(upstream().to_string()))
        .unwrap();
    let response = adapter.convert_response(response, 100_000).await;
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()["x-request-id"], "request-42");
    assert_eq!(response.headers()["x-smg-routed-worker"], "worker");
    for header in ["content-length", "content-encoding", "etag", "content-md5"] {
        assert!(!response.headers().contains_key(header));
    }
    assert_eq!(response.headers()["content-type"], "application/json");
}

#[tokio::test]
async fn upstream_validation_errors_remain_errors_with_useful_messages() {
    for fixture in [
        json!({"error":{"message":"unsupported reasoning template","type":"invalid_request_error","code":"invalid_model"}}),
        json!({"detail":"question q0 label is not a single token"}),
    ] {
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(
                (StatusCode::BAD_REQUEST, Json(fixture)).into_response(),
                100_000,
            )
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value = body(response).await;
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unsupported")
                || value["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("single token")
        );
        assert!(value.get("answers").is_none());
    }
}

/// A worker's own error answer, re-minted under its code (or the
/// relay's), carries the worker's status as the error counter's reason:
/// its 500 reads as the engine's internal fault, its 404 as the model
/// not found, its 400 as the request's own fault, never as the gateway's
/// own processing.
#[tokio::test]
async fn a_relayed_worker_error_names_the_workers_status_reason() {
    use crate::{
        observability::metrics::metrics_labels::{
            REASON_ENGINE_INTERNAL, REASON_INVALID_REQUEST, REASON_MODEL_NOT_FOUND,
        },
        routers::{
            common::attempt_ledger::mark_upstream_answer, grpc::utils::error_reason_for_response,
        },
    };
    for (status, fixture, reason) in [
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error":{"message":"boom","type":"server_error","code":"engine_exploded"}}),
            REASON_ENGINE_INTERNAL,
        ),
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail":"no code at all"}),
            REASON_ENGINE_INTERNAL,
        ),
        (
            StatusCode::NOT_FOUND,
            json!({"error":{"message":"no such model","type":"invalid_request_error","code":"model_not_found"}}),
            REASON_MODEL_NOT_FOUND,
        ),
        (
            StatusCode::BAD_REQUEST,
            json!({"detail":"question q0 label is not a single token"}),
            REASON_INVALID_REQUEST,
        ),
    ] {
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let mut answer = (status, Json(fixture)).into_response();
        mark_upstream_answer(&mut answer);
        let response = adapter.convert_response(answer, 100_000).await;
        assert_eq!(response.status(), status);
        assert_eq!(error_reason_for_response(&response), reason, "{status}");
    }
}

/// The converted answer stands on the worker's own parts. So a mark the
/// router left on the worker's answer is still on the converted one,
/// whether the adapter mapped the answer or refused it as malformed.
#[tokio::test]
async fn a_converted_answer_keeps_the_marks_of_the_workers_answer() {
    use crate::routers::common::attempt_ledger::mark_upstream_answer;
    for (answer, status) in [
        (Json(upstream()).into_response(), StatusCode::OK),
        (
            Response::new(Body::from("not json")),
            StatusCode::BAD_GATEWAY,
        ),
    ] {
        let mut answer = answer;
        mark_upstream_answer(&mut answer);
        let adapter =
            SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
        let converted = adapter.convert_response(answer, 100_000).await;
        assert_eq!(converted.status(), status);
        assert!(converted.extensions().get::<UpstreamAnswer>().is_some());
    }
}

/// The gateway's own failed send or body read reaches the adapter as an
/// error without the worker's answer marker: its code keeps naming the
/// reason (a transport fault stays one), the status does not take over.
#[tokio::test]
async fn a_gateway_own_error_keeps_the_reason_of_its_code() {
    use crate::{
        observability::metrics::metrics_labels::REASON_TRANSPORT_DISCONNECT,
        routers::grpc::utils::error_reason_for_response,
    };
    let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
    let response = adapter
        .convert_response(
            error::internal_error("call_upstream_connection_failed", "refused"),
            100_000,
        )
        .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        error_reason_for_response(&response),
        REASON_TRANSPORT_DISCONNECT
    );
}

#[tokio::test]
async fn malformed_successes_fail_as_bad_gateway() {
    let mutations = [
        ("/answers/q0/noul", json!(1.1)),
        ("/answers/q0/type", json!("predicate")),
        ("/answers/q1/choice", json!("unknown")),
        ("/answers/q1/confidence", json!(-0.1)),
        ("/answers/q1/probabilities/o0", Value::Null),
        ("/answers/q1/probabilities", json!({"o0":0.2,"other":0.8})),
        ("/answers/q2/score", json!(2)),
        ("/answers/q2/score", json!("NaN")),
        ("/answers/q2/probabilities", json!({"0":0.2})),
        ("/answers/q0", Value::Null),
        ("/usage/input_tokens", json!(-1)),
        ("/usage/output_tokens", json!(1)),
        ("/model", Value::Null),
    ];
    for (path, replacement) in mutations {
        let mut value = upstream();
        *value.pointer_mut(path).unwrap() = replacement;
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(Json(value).into_response(), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{path}");
        assert!(body(response).await["error"]["message"].is_string());
    }
    for text in [
        "{",
        "null",
        r#"{"model":"served","answers":{"q0":{"type":"noul","noul":0.2,"noul":0.9}},"usage":{"input_tokens":1,"output_tokens":0}}"#,
    ] {
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(Response::new(Body::from(text)), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }
}

#[tokio::test]
async fn conversion_enforces_the_response_buffer_limit() {
    let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
    let response = adapter
        .convert_response(Json(upstream()).into_response(), 10)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn duplicate_backend_keys_cannot_overwrite_valid_answer_probabilities() {
    for text in [
        upstream()
            .to_string()
            .replace("\"noul\":0.8", "\"noul\":0.2,\"noul\":0.8"),
        upstream()
            .to_string()
            .replace("\"o0\":0.2", "\"o0\":0.4,\"o0\":0.2"),
    ] {
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(Response::new(Body::from(text)), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(body(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("duplicate key"));
    }
}

#[tokio::test]
async fn singleton_answers_preserve_original_strings_and_score_labels() {
    let questions = json!([
        {"type":"choice","instructions":"Select","choices":[{"value":"</s>\nA: true"}]},
        {"type":"score","instructions":"Rate","levels":[{"label":"A: low"}]}
    ]);
    let adapter = SglangDecisionAdapter::new(&request(json!("x"), questions)).unwrap();
    assert_eq!(
        adapter.request()["questions"]["q0"]["criteria"],
        json!({"o0":{"value":"</s>\nA: true"}})
    );
    let fixture = json!({"model":"served","answers":{
        "q0":{"type":"choice","choice":"o0","confidence":1,"probabilities":{"o0":1},"x_label_mass":0.6},
        "q1":{"type":"score","score":0,"confidence":1,"probabilities":{"0":1},"legend":{"0":{"label":"A: low"}},"x_label_mass":0.7}
    },"usage":{"input_tokens":1,"output_tokens":0}});
    let response = adapter
        .convert_response(Json(fixture).into_response(), 100_000)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body(response).await["answers"],
        json!([
            {"type":"choice","name":null,"choice":"</s>\nA: true","confidence":1.0,"probabilities":[{"value":"</s>\nA: true","probability":1.0}]},
            {"type":"score","name":null,"score":0.0,"confidence":1.0,"probabilities":[{"value":0,"label":"A: low","probability":1.0}]}
        ])
    );
}

#[test]
fn gateway_safety_identifier_is_not_forwarded_as_a_backend_extension() {
    let mut value = json!({"model":"test","input":"x","questions":[{"type":"predicate","instructions":"x"}],"safety_identifier":"user-42"});
    let request = serde_json::from_value(value.take()).unwrap();
    let adapter = SglangDecisionAdapter::new(&request).unwrap();
    assert_eq!(
        adapter.request(),
        &json!({"model":"test","state":"x","questions":{"q0":{"type":"noul","instructions":"x"}}})
    );
}

#[tokio::test]
async fn rejects_inconsistent_distributions_scores_and_choice_winners() {
    for (path, replacement) in [
        ("/answers/q1/probabilities", json!({"o0":0.8,"o1":0.8})),
        ("/answers/q2/probabilities", json!({"0":0.5,"1":0.75})),
        ("/answers/q2/score", json!(0.1)),
        ("/answers/q1/choice", json!("o0")),
    ] {
        let mut fixture = upstream();
        *fixture.pointer_mut(path).unwrap() = replacement;
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(Json(fixture).into_response(), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{path}");
    }
}

#[tokio::test]
async fn accepts_tied_winners_and_probability_roundoff() {
    for winner in ["o0", "o1"] {
        let mut fixture = upstream();
        fixture["answers"]["q1"]["probabilities"] = json!({"o0":0.5,"o1":0.5});
        fixture["answers"]["q1"]["choice"] = json!(winner);
        fixture["answers"]["q2"]["probabilities"] = json!({"0":0.25,"1":0.75000001});
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(Json(fixture).into_response(), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn preserves_existing_openai_error_envelopes() {
    let fixture = json!({"error":{"message":"unsupported template","type":"invalid_request_error","code":"unsupported_model","param":"model","backend_context":{"retry":false}},"trace":"abc"});
    let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
    let response = adapter
        .convert_response(
            (StatusCode::BAD_REQUEST, Json(fixture.clone())).into_response(),
            100_000,
        )
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(body(response).await, fixture);
}

#[tokio::test]
async fn bounds_the_encoded_public_response_including_original_labels() {
    let questions = json!([{ "type":"choice", "instructions":"Select", "choices":[{"value":"x".repeat(2000)}]}]);
    let fixture = json!({"model":"served","answers":{"q0":{"type":"choice","choice":"o0","confidence":1,"probabilities":{"o0":1},"x_label_mass":0.5}},"usage":{"input_tokens":1,"output_tokens":0}});
    let limit = fixture.to_string().len() + 20;
    let adapter = SglangDecisionAdapter::new(&request(json!("x"), questions)).unwrap();
    let response = adapter
        .convert_response(Json(fixture).into_response(), limit)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn releasing_dispatch_evidence_retains_answer_mapping() {
    let mut adapter =
        SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
    adapter.release_request();
    assert!(adapter.request().is_null());
    let response = adapter
        .convert_response(Json(upstream()).into_response(), 100_000)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let answers = body(response).await["answers"].clone();
    assert_eq!(answers[0]["name"], "same");
    assert_eq!(answers[1]["choice"], "true");
    assert_eq!(answers[2]["probabilities"][1]["label"], "high");
}
