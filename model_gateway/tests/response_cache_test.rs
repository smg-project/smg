//! The exact-match response cache through the real app (its middleware
//! resolves the tenant from the trusted header) over an in-process HTTP mock
//! engine: the second identical deterministic request is answered from
//! memory with the engine's first answer, another tenant's is generated,
//! sampled and streamed requests and `x-smg-cache: no-store` bypass, and
//! with the cache off (the default) nothing is tagged.

#[path = "common/mod.rs"]
mod common;

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Request, StatusCode},
    Router,
};
use serde_json::{json, Value};
use smg::config::{ResponseCacheBackend, ResponseCacheConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

const CACHE_STATUS: &str = "x-smg-cache-status";
const TENANT_HEADER: &str = "x-smg-tenant-id";

fn config(cache: bool) -> RouterConfig {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(3002)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config.tenant_resolution.trust_tenant_header = true;
    if cache {
        config.response_cache = ResponseCacheConfig {
            backend: ResponseCacheBackend::Memory,
            ttl_secs: 60,
            max_entries: 64,
            max_bytes: 1 << 20,
            max_entry_bytes: 1 << 16,
        };
    }
    config
}

async fn context(cache: bool) -> AppTestContext {
    AppTestContext::new_with_config(
        config(cache),
        vec![MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }],
    )
    .await
}

/// One POST as `tenant`, with an `x-smg-cache` header when given: the
/// status, the `x-smg-cache-status` header and the JSON body.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn post(
    app: &Router,
    path: &str,
    tenant: &str,
    cache_header: Option<&str>,
    body: &Value,
) -> (StatusCode, Option<String>, Value) {
    let mut request = Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .header(TENANT_HEADER, tenant);
    if let Some(value) = cache_header {
        request = request.header("x-smg-cache", value);
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let cache_status = response
        .headers()
        .get(CACHE_STATUS)
        .map(|value| value.to_str().unwrap().to_owned());
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, cache_status, json)
}

fn chat(prompt: &str, temperature: f64) -> Value {
    json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 8,
    })
}

#[tokio::test]
async fn the_second_identical_deterministic_request_is_answered_from_memory() {
    let ctx = context(true).await;
    let app = ctx.create_app();

    let request = chat("What is the capital of France?", 0.0);
    let (status, cache, first) = post(&app, "/v1/chat/completions", "a", None, &request).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(cache.as_deref(), Some("miss"));
    let (status, cache, second) = post(&app, "/v1/chat/completions", "a", None, &request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cache.as_deref(), Some("hit"));
    assert_eq!(
        second, first,
        "a hit replays the engine's answer, id and all"
    );

    // Another tenant with the same request is generated anew.
    let (status, cache, other) = post(&app, "/v1/chat/completions", "b", None, &request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cache.as_deref(), Some("miss"));
    assert_ne!(other["id"], first["id"]);

    // The same request in another key order is the same request.
    let reordered = json!({
        "max_tokens": 8,
        "temperature": 0,
        "messages": [{"content": "What is the capital of France?", "role": "user"}],
        "model": "mock-model",
        "user": "someone",
    });
    let (_, cache, replay) = post(&app, "/v1/chat/completions", "a", None, &reordered).await;
    assert_eq!(cache.as_deref(), Some("hit"));
    assert_eq!(replay, first);

    // Completions and messages have caches of their own.
    let completion =
        json!({"model": "mock-model", "prompt": "Paris is", "temperature": 0, "max_tokens": 8});
    let (status, cache, first) = post(&app, "/v1/completions", "a", None, &completion).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(cache.as_deref(), Some("miss"));
    let (_, cache, second) = post(&app, "/v1/completions", "a", None, &completion).await;
    assert_eq!(cache.as_deref(), Some("hit"));
    assert_eq!(second, first);

    let message = json!({
        "model": "mock-model",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0,
    });
    let (status, cache, first) = post(&app, "/v1/messages", "a", None, &message).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(cache.as_deref(), Some("miss"));
    let (_, cache, second) = post(&app, "/v1/messages", "a", None, &message).await;
    assert_eq!(cache.as_deref(), Some("hit"));
    assert_eq!(second, first);

    ctx.shutdown().await;
}

#[tokio::test]
async fn sampled_streamed_and_no_store_requests_bypass_the_cache() {
    let ctx = context(true).await;
    let app = ctx.create_app();

    let sampled = chat("Tell me a story", 0.7);
    let (status, cache, first) = post(&app, "/v1/chat/completions", "a", None, &sampled).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(cache.as_deref(), Some("bypass"));
    let (_, cache, second) = post(&app, "/v1/chat/completions", "a", None, &sampled).await;
    assert_eq!(cache.as_deref(), Some("bypass"));
    assert_ne!(
        second["id"], first["id"],
        "a sampled request is generated every time"
    );

    let mut streamed = chat("What is the capital of France?", 0.0);
    streamed["stream"] = json!(true);
    let (status, cache, _) = post(&app, "/v1/chat/completions", "a", None, &streamed).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cache.as_deref(), Some("bypass"));

    // no-store neither reads nor writes: the next plain request still misses.
    let request = chat("What is the capital of Spain?", 0.0);
    let (_, cache, _) = post(
        &app,
        "/v1/chat/completions",
        "a",
        Some("no-store"),
        &request,
    )
    .await;
    assert_eq!(cache.as_deref(), Some("bypass"));
    let (_, cache, _) = post(&app, "/v1/chat/completions", "a", None, &request).await;
    assert_eq!(cache.as_deref(), Some("miss"));
    let (_, cache, _) = post(
        &app,
        "/v1/chat/completions",
        "a",
        Some("no-store"),
        &request,
    )
    .await;
    assert_eq!(cache.as_deref(), Some("bypass"));

    ctx.shutdown().await;
}

#[tokio::test]
async fn with_the_cache_off_every_request_is_generated_and_nothing_is_tagged() {
    let ctx = context(false).await;
    let app = ctx.create_app();

    let request = chat("What is the capital of France?", 0.0);
    let (status, cache, first) = post(&app, "/v1/chat/completions", "a", None, &request).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(cache, None);
    let (_, cache, second) = post(&app, "/v1/chat/completions", "a", None, &request).await;
    assert_eq!(cache, None);
    assert_ne!(second["id"], first["id"]);

    ctx.shutdown().await;
}
