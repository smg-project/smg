//! A chat request's cache partition fields through the pipeline. The
//! pipeline refuses a value that the wire cannot carry before any stage
//! runs. It sends the request only to a worker that applies its fields.

use smg_worker::labels::CACHE_PARTITION_FIELDS_LABEL;

use super::*;
use crate::routers::grpc::proto_wrapper::ProtoGenerateRequest;

fn chat_with(fields: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    });
    for (key, value) in fields.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

fn calls(seen: &Mutex<Vec<Vec<u32>>>) -> usize {
    seen.lock().unwrap_or_else(PoisonError::into_inner).len()
}

#[tokio::test]
async fn the_pipeline_refuses_a_salted_chat_that_no_worker_applies() {
    let model_dir = unreadable_model_dir();
    let (pipeline, components, seen) =
        chat_pipeline_over(RuntimeType::TokenSpeed, &[], model_dir.path()).await;
    let (status, body) = chat_error_body(
        &pipeline,
        components,
        chat_with(serde_json::json!({"cache_salt": "tenant-secret"})),
    )
    .await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"], "cache_partition_not_supported",
        "{body}"
    );
    assert!(!body.to_string().contains("tenant-secret"), "{body}");
    assert_eq!(calls(&seen), 0);
}

/// The TokenSpeed wire has no salt field. A label that lists the salt does
/// not change that, so the pipeline refuses the salted chat.
#[tokio::test]
async fn the_pipeline_refuses_a_salt_that_the_wire_cannot_carry() {
    let model_dir = unreadable_model_dir();
    let (pipeline, components, seen) = chat_pipeline_over(
        RuntimeType::TokenSpeed,
        &[(CACHE_PARTITION_FIELDS_LABEL, "cache_salt")],
        model_dir.path(),
    )
    .await;
    let (status, body) = chat_error_body(
        &pipeline,
        components,
        chat_with(serde_json::json!({"cache_salt": "tenant-secret"})),
    )
    .await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"], "cache_partition_not_supported",
        "{body}"
    );
    assert!(!body.to_string().contains("tenant-secret"), "{body}");
    assert_eq!(calls(&seen), 0);
}

/// An empty value is unset: any worker takes the request.
#[tokio::test]
async fn a_chat_with_an_empty_salt_reaches_any_worker() {
    let model_dir = unreadable_model_dir();
    let (pipeline, components, seen) =
        chat_pipeline_over(RuntimeType::TokenSpeed, &[], model_dir.path()).await;
    let fields = serde_json::json!({"cache_salt": "", "extra_key": null});
    let response = chat_response(&pipeline, components, chat_with(fields)).await;
    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(calls(&seen), 1);
}

#[tokio::test]
async fn an_extra_key_needs_a_worker_that_lists_it() {
    // The SGLang wire carries the extra key, but the label lists only the salt.
    let model_dir = unreadable_model_dir();
    let (pipeline, components, seen) = chat_pipeline_over(
        RuntimeType::Sglang,
        &[(CACHE_PARTITION_FIELDS_LABEL, "cache_salt")],
        model_dir.path(),
    )
    .await;
    let (status, body) = chat_error_body(
        &pipeline,
        components,
        chat_with(serde_json::json!({"cache_salt": "s", "extra_key": "k"})),
    )
    .await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"], "cache_partition_not_supported",
        "{body}"
    );
    assert_eq!(calls(&seen), 0);
}

#[tokio::test]
async fn the_pipeline_refuses_a_field_that_is_not_a_string() {
    for fields in [
        serde_json::json!({"cache_salt": 7}),
        serde_json::json!({"extra_key": ["a", "b"]}),
    ] {
        let model_dir = unreadable_model_dir();
        let (pipeline, components, seen) = chat_pipeline_over(
            RuntimeType::TokenSpeed,
            &[(CACHE_PARTITION_FIELDS_LABEL, "cache_salt,extra_key")],
            model_dir.path(),
        )
        .await;
        let (status, body) =
            chat_error_body(&pipeline, components, chat_with(fields.clone())).await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{fields}: {body}");
        assert_eq!(
            body["error"]["code"], "invalid_request_parameters",
            "{fields}: {body}"
        );
        assert_eq!(calls(&seen), 0, "{fields}");
    }
}

/// The pipeline stamps the salt on the request that it builds for a vLLM
/// worker. Ingress runs to the built plan. The test makes no engine call.
#[tokio::test]
async fn the_built_request_carries_the_salt() {
    let model_dir = unreadable_model_dir();
    let (pipeline, components, _) = chat_pipeline_over(
        RuntimeType::Vllm,
        &[(CACHE_PARTITION_FIELDS_LABEL, "cache_salt")],
        model_dir.path(),
    )
    .await;
    let request: ChatCompletionRequest =
        serde_json::from_value(chat_with(serde_json::json!({"cache_salt": "tenant-a"})))
            .expect("chat request");
    let mut ctx = RequestContext::for_chat(Arc::new(request), None, MODEL.into(), components);
    let Ok(mut build) = pipeline.run_ingress(&mut ctx).await else {
        panic!("ingress builds the plan");
    };
    let salts: Vec<Option<String>> = build
        .plan
        .generate_requests_mut()
        .map(|request| match request {
            ProtoGenerateRequest::Vllm(request) => request.cache_salt.clone(),
            _ => panic!("a vLLM worker gets a vLLM request"),
        })
        .collect();
    assert_eq!(salts, [Some("tenant-a".to_string())]);
}
