//! A served model whose only worker leaves the registry (an engine restart
//! under discovery removes the pod's worker for as long as the pod is not
//! ready) is refused as a pool with no worker, the transient 503 a client
//! retries, not as a model that does not exist; a name nobody has served
//! stays the 404; the model serves again the moment a worker is registered
//! for it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

use std::{sync::Arc, time::Duration};

use axum::{
    body::to_bytes,
    http::{header::RETRY_AFTER, StatusCode},
};
use common::grpc_harness::{
    grpc_router, grpc_router_config, grpc_test_context, grpc_worker, serve_canned_worker, Canned,
};
use openai_protocol::{
    completion::CompletionRequest,
    model_card::ModelCard,
    worker::{RuntimeType, WorkerType},
};
use smg::{middleware::TenantRequestMeta, routers::RouterTrait, tenant::TenantKey, worker::Worker};

use crate::common;

const MODEL: &str = "served-model";

fn completion(model: &str) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "prompt": "Hello world",
        "max_tokens": 4,
        "stream": false,
    }))
    .expect("completion request")
}

/// The status, the `Retry-After` header and the JSON body of a completion
/// for `model` through `router`.
async fn request(
    router: &dyn RouterTrait,
    model: &str,
) -> (StatusCode, Option<String>, serde_json::Value) {
    let tenant = TenantRequestMeta::new(TenantKey::new("served-model-tenant"));
    let response = router
        .route_completion(None, &tenant, completion(model), model)
        .await;
    let status = response.status();
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
    (status, retry_after, body)
}

fn worker_at(port: u16) -> Arc<dyn Worker> {
    Arc::new(
        grpc_worker(
            port,
            WorkerType::Regular,
            RuntimeType::TokenSpeed,
            ModelCard::new(MODEL),
        )
        .build(),
    )
}

#[tokio::test]
async fn a_served_model_whose_only_worker_left_is_refused_as_a_pool_with_no_worker() {
    let port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 3,
    })
    .await;
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .build_unchecked();
    let app_context = grpc_test_context(config, MODEL).await;
    let worker = worker_at(port);
    let router = grpc_router(&app_context, vec![Arc::clone(&worker)]).await;
    let registry = &app_context.worker_registry;

    let (status, _, body) = request(router.as_ref(), MODEL).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The worker leaves, as discovery removes a pod's worker while the pod
    // is not ready.
    let worker_id = registry.get_id_by_url(worker.url()).unwrap();
    assert!(registry.remove(&worker_id).is_some());

    let (status, retry_after, body) = request(router.as_ref(), MODEL).await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "a served model with no worker is a pool with no worker: {body}"
    );
    assert_eq!(body["error"]["code"], "no_available_workers", "{body}");
    assert_eq!(
        retry_after.as_deref(),
        Some("1"),
        "paced like the pool's other refusals"
    );

    let (status, _, body) = request(router.as_ref(), "never-served").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");

    // The pod is ready again: a worker is registered for the model and it
    // serves as before.
    registry
        .register(worker_at(port))
        .expect("register the returning worker");
    let (status, _, body) = request(router.as_ref(), MODEL).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
