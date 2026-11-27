use std::sync::Arc;

use axum::http::StatusCode;
use serde_json::json;
use smg_grpc_client::{sglang_proto as sglang, tokenspeed_proto as tokenspeed, vllm_proto as vllm};
use smg_worker::identity::basic::{RuntimeType, Worker};

use super::*;
use crate::routers::grpc::{
    backend_client::BasicWorkerBuilder,
    context::ExecutionPlanKind,
    proto_wrapper::{ProtoGenerateRequest, ProtoRequest},
};

fn chat(mut body: Value) -> RequestType {
    body["model"] = json!("m");
    body["messages"] = json!([]);
    RequestType::Chat(Arc::new(serde_json::from_value(body).unwrap()))
}

fn embedding(mut body: Value) -> RequestType {
    body["model"] = json!("m");
    body["input"] = json!("hi");
    RequestType::Embedding(Arc::new(serde_json::from_value(body).unwrap()))
}

/// An SGLang worker: its wire carries both fields, so only the label decides.
fn worker(label: Option<&str>) -> impl Worker {
    let builder =
        BasicWorkerBuilder::new("grpc://127.0.0.1:9000").runtime_type(RuntimeType::Sglang);
    match label {
        Some(value) => builder.label(CACHE_PARTITION_FIELDS_LABEL, value),
        None => builder,
    }
    .build()
}

async fn error_of(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

#[test]
fn the_partition_holds_the_body_fields_and_the_forwarded_adapter() {
    let messages = RequestType::Messages(Arc::new(
        serde_json::from_value(json!({
            "model": "m", "max_tokens": 8, "messages": [],
            "cache_salt": "s", "extra_key": "k", "lora_path": "unforwarded"
        }))
        .unwrap(),
    ));
    assert_eq!(
        request_partition(&messages),
        CachePartition {
            cache_salt: Some("s"),
            extra_key: Some("k"),
            lora_path: None,
        }
    );
    let generate = RequestType::Generate(Arc::new(
        serde_json::from_value(json!({
            "model": "m", "text": "hi", "cache_salt": "s", "extra_key": "k", "lora_path": "a"
        }))
        .unwrap(),
    ));
    assert_eq!(
        request_partition(&generate),
        CachePartition {
            cache_salt: Some("s"),
            extra_key: Some("k"),
            lora_path: Some("a"),
        }
    );
    let empty = chat(json!({"cache_salt": "", "extra_key": ""}));
    assert_eq!(request_partition(&empty), CachePartition::default());
    let embed = embedding(json!({"cache_salt": "s"}));
    assert_eq!(request_partition(&embed), CachePartition::default());
}

#[tokio::test]
async fn the_ingress_check_refuses_a_value_the_wire_cannot_carry_and_hides_it() {
    for (request, code) in [
        (chat(json!({"cache_salt": 7})), "invalid_request_parameters"),
        (chat(json!({"extra_key": ["secret"]})), "invalid_request_parameters"),
        (
            chat(json!({"cache_salt": {"secret": 1}})),
            "invalid_request_parameters",
        ),
        (
            embedding(json!({"cache_salt": "secret"})),
            NOT_SUPPORTED_CODE,
        ),
        (
            RequestType::Classify(Arc::new(
                serde_json::from_value(json!({
                    "model": "m", "input": "hi", "extra_key": "secret"
                }))
                .unwrap(),
            )),
            NOT_SUPPORTED_CODE,
        ),
        (
            RequestType::Decisions(Arc::new(
                serde_json::from_value(json!({
                    "model": "m", "input": "Evidence.",
                    "questions": [{"type": "predicate", "name": "q", "instructions": "Is it true?"}],
                    "cache_salt": "secret"
                }))
                .unwrap(),
            )),
            NOT_SUPPORTED_CODE,
        ),
    ] {
        let (status, body) = error_of(refuse_unforwarded(&request).unwrap_err()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], code, "{body}");
        assert!(!body.to_string().contains("secret"), "{body}");
    }
    for request in [
        chat(json!({"cache_salt": "s", "extra_key": "k"})),
        chat(json!({"cache_salt": null, "extra_key": ""})),
        embedding(json!({"cache_salt": ""})),
        embedding(json!({})),
    ] {
        assert!(refuse_unforwarded(&request).is_ok());
    }
}

#[test]
fn a_worker_meets_the_demand_when_its_label_lists_each_field() {
    let salt = CachePartitionDemand::of(&CachePartition {
        cache_salt: Some("s"),
        ..Default::default()
    });
    let both = CachePartitionDemand::of(&CachePartition {
        cache_salt: Some("s"),
        extra_key: Some("k"),
        lora_path: None,
    });
    let none = CachePartitionDemand::of(&CachePartition {
        cache_salt: Some(""),
        extra_key: None,
        lora_path: Some("adapter"),
    });
    assert!(!none.any());
    for (label, takes_salt, takes_both) in [
        (None, false, false),
        (Some(""), false, false),
        (Some("extra_key"), false, false),
        (Some("cache_salt"), true, false),
        (Some("cache_salt,extra_key"), true, true),
        (Some(" cache_salt , extra_key "), true, true),
    ] {
        let worker = worker(label);
        assert_eq!(salt.met_by(&worker), takes_salt, "{label:?}");
        assert_eq!(both.met_by(&worker), takes_both, "{label:?}");
        assert!(none.met_by(&worker), "{label:?}");
    }
}

/// A label counts only for a field that the gateway's wire for the worker's
/// runtime carries. An operator label cannot open a wire without the field.
#[test]
fn a_label_counts_only_for_a_field_that_the_wire_carries() {
    let salt = CachePartitionDemand::of(&CachePartition {
        cache_salt: Some("s"),
        ..Default::default()
    });
    let key = CachePartitionDemand::of(&CachePartition {
        extra_key: Some("k"),
        ..Default::default()
    });
    for (runtime, takes_salt, takes_key) in [
        (RuntimeType::Sglang, true, true),
        (RuntimeType::Vllm, true, false),
        (RuntimeType::TokenSpeed, false, false),
        (RuntimeType::Trtllm, false, false),
        (RuntimeType::Mlx, false, false),
        (RuntimeType::Unspecified, false, false),
    ] {
        let worker = BasicWorkerBuilder::new("grpc://127.0.0.1:9000")
            .runtime_type(runtime)
            .label(CACHE_PARTITION_FIELDS_LABEL, "cache_salt,extra_key")
            .build();
        assert_eq!(salt.met_by(&worker), takes_salt, "{runtime:?}");
        assert_eq!(key.met_by(&worker), takes_key, "{runtime:?}");
    }
}

#[tokio::test]
async fn the_refusal_names_the_fields() {
    let both = CachePartitionDemand {
        cache_salt: true,
        extra_key: true,
    };
    let (status, body) = error_of(both.refusal("m", "workers")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], NOT_SUPPORTED_CODE);
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("apply cache_salt and extra_key"),
        "{message}"
    );
}

fn vllm_request() -> ProtoGenerateRequest {
    ProtoGenerateRequest::Vllm(Box::default())
}

fn sglang_request() -> ProtoGenerateRequest {
    ProtoGenerateRequest::Sglang(Box::default())
}

/// The salt and the extra key of each generate request, as its wire holds them.
fn stamped(plan: &mut ExecutionPlan) -> Vec<(Option<String>, Option<String>)> {
    plan.generate_requests_mut()
        .map(|request| match request {
            ProtoGenerateRequest::Vllm(req) => (req.cache_salt.clone(), None),
            ProtoGenerateRequest::Sglang(req) => req.cache_partition.as_deref().map_or(
                (None, None),
                |partition: &sglang::CachePartition| {
                    (partition.cache_salt.clone(), partition.extra_key.clone())
                },
            ),
            _ => (None, None),
        })
        .collect()
}

#[test]
fn the_stamp_reaches_every_generate_request_of_the_plan() {
    let partition = CachePartition {
        cache_salt: Some("s"),
        extra_key: Some("k"),
        lora_path: Some("adapter"),
    };
    let salt_and_key = (Some("s".to_string()), Some("k".to_string()));
    let salt_only = (Some("s".to_string()), None);
    for (mut plan, expected) in [
        (
            ExecutionPlan::Single(ProtoRequest::Generate(sglang_request())),
            vec![salt_and_key.clone()],
        ),
        (
            ExecutionPlan::PrefillDecode(vllm_request()),
            vec![salt_only.clone()],
        ),
        (
            ExecutionPlan::EncodePrefillDecode {
                request: sglang_request(),
            },
            vec![salt_and_key.clone()],
        ),
        (
            ExecutionPlan::Batch {
                kind: ExecutionPlanKind::Single,
                shared_request_id: "b".to_string(),
                requests: vec![vllm_request(), vllm_request()],
            },
            vec![salt_only.clone(), salt_only],
        ),
    ] {
        stamp(&mut plan, partition);
        assert_eq!(stamped(&mut plan), expected);
    }

    // A wire without the fields keeps its request as it was built.
    let mut token_speed = ExecutionPlan::Single(ProtoRequest::Generate(
        ProtoGenerateRequest::TokenSpeed(Box::default()),
    ));
    stamp(&mut token_speed, partition);
    let ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::TokenSpeed(req))) =
        &token_speed
    else {
        panic!("the plan holds one generate request of that wire");
    };
    assert_eq!(**req, tokenspeed::GenerateRequest::default());
}

/// The fields that the gate counts for a runtime are the fields that the
/// stamp writes on the wire of that runtime.
#[test]
fn the_gate_counts_the_fields_that_the_stamp_writes() {
    let both = CachePartition {
        cache_salt: Some("s"),
        extra_key: Some("k"),
        lora_path: None,
    };
    for (runtime, request) in [
        (RuntimeType::Vllm, vllm_request()),
        (RuntimeType::Sglang, sglang_request()),
        (
            RuntimeType::Trtllm,
            ProtoGenerateRequest::Trtllm(Box::default()),
        ),
        (RuntimeType::Mlx, ProtoGenerateRequest::Mlx(Box::default())),
        (
            RuntimeType::TokenSpeed,
            ProtoGenerateRequest::TokenSpeed(Box::default()),
        ),
    ] {
        let mut plan = ExecutionPlan::Single(ProtoRequest::Generate(request));
        stamp(&mut plan, both);
        let (salt, key) = stamped(&mut plan).remove(0);
        let written: Vec<&str> = [(CACHE_SALT, salt.is_some()), (EXTRA_KEY, key.is_some())]
            .into_iter()
            .filter_map(|(field, set)| set.then_some(field))
            .collect();
        assert_eq!(
            written,
            ProtoGenerateRequest::cache_partition_fields(runtime),
            "{runtime:?}"
        );
    }
}

#[test]
fn a_request_without_the_fields_keeps_its_wire_form() {
    let mut plan = ExecutionPlan::Single(ProtoRequest::Generate(sglang_request()));
    stamp(
        &mut plan,
        CachePartition {
            lora_path: Some("adapter"),
            ..Default::default()
        },
    );
    let ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::Sglang(req))) = &plan
    else {
        panic!("the plan holds one generate request of that wire");
    };
    assert!(req.cache_partition.is_none());
    let mut plan = ExecutionPlan::Single(ProtoRequest::Generate(vllm_request()));
    stamp(&mut plan, CachePartition::default());
    let ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::Vllm(req))) = &plan
    else {
        panic!("the plan holds one generate request of that wire");
    };
    assert_eq!(**req, vllm::GenerateRequest::default());
}
