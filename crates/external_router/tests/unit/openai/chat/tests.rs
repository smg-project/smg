use async_trait::async_trait;
use axum::body::to_bytes;

use super::*;
use crate::{
    error::extract_error_code_from_response,
    retry::REQUEST_TIMEOUT_CODE,
    tenant::TenantKey,
    test_support::{
        answering, counting_alloc, http_client, text_of, upstream, RefusedAddress, StubWorker,
    },
    worker::{ExternalWorker, WorkerStats},
};

async fn bytes_of(response: Response) -> Vec<u8> {
    to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a body")
        .to_vec()
}

fn content_type_of(response: &Response) -> Option<String> {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// A DeepSeek V4.1 `medium` effort goes to the upstream as the vendor's
/// `high`, in both top-level spellings; a chat-template kwarg and another
/// vendor's request go out as sent.
#[test]
fn a_v41_medium_effort_reaches_the_upstream_as_the_vendors_high() {
    let request = |model: &str| {
        serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "medium",
            "thinking": {"effort": "medium"},
            "chat_template_kwargs": {"reasoning_effort": "medium"},
        }))
        .unwrap()
    };
    let body = chat_for_upstream(request("deepseek-v4.1-flash"));
    assert_eq!(body.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(
        body.thinking
            .as_ref()
            .and_then(|thinking| thinking.effort.as_deref()),
        Some("high")
    );
    assert_eq!(
        body.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("reasoning_effort")),
        Some(&serde_json::json!("medium"))
    );
    let body = chat_for_upstream(request("gpt-4o"));
    assert_eq!(body.reasoning_effort.as_deref(), Some("medium"));
}

#[tokio::test]
async fn a_buffered_body_keeps_the_upstreams_status_content_type_and_bytes() {
    let response = buffered_body(
        upstream(201, Some("application/json"), "{\"ok\":true}"),
        StatusCode::CREATED,
    )
    .await;
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(
        content_type_of(&response).as_deref(),
        Some("application/json")
    );
    assert_eq!(bytes_of(response).await, b"{\"ok\":true}");
}

/// A stub whose answer's head and first chunk leave at once and whose
/// body then never ends, as a provider that stalls mid-answer does.
#[expect(
    clippy::disallowed_methods,
    reason = "test stub server lives for the duration of the test process"
)]
async fn stalled_upstream() -> String {
    use futures_util::StreamExt;
    let app = axum::Router::new().fallback(axum::routing::any(|| async {
        let head = futures_util::stream::iter([Ok::<_, std::io::Error>(
            bytes::Bytes::from_static(b"{\"partial\":"),
        )]);
        (
            [(CONTENT_TYPE, "application/json")],
            Body::from_stream(head.chain(futures_util::stream::pending())),
        )
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A client whose budget for a call is `timeout`, the process TLS
/// crypto provider installed first.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
fn client_with_timeout(timeout: std::time::Duration) -> reqwest::Client {
    use rustls::crypto::ring::default_provider;
    let _ = default_provider().install_default();
    reqwest::Client::builder().timeout(timeout).build().unwrap()
}

/// The request's own timeout running out while the body is read is the
/// request's end: the final 504, not the replayable 500 of a read failure.
#[tokio::test]
async fn a_buffered_body_the_request_timeout_cuts_is_the_final_504() {
    let url = stalled_upstream().await;
    let client = client_with_timeout(std::time::Duration::from_millis(200));
    let answer = client.post(url).send().await.unwrap();

    let response = buffered_body(answer, StatusCode::OK).await;

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        extract_error_code_from_response(&response),
        REQUEST_TIMEOUT_CODE
    );
    assert!(!is_retryable_response(&response));
}

/// A body lost mid-answer is the 500 of a read failure, and it says why
/// without the worker's address. The HTTP client's text of a failed body
/// read ends with the request URL.
#[tokio::test]
async fn a_buffered_body_cut_short_is_answered_without_the_workers_address() {
    let addr = answering(200, "short", true).await;
    let answer = http_client()
        .post(format!("http://{addr}"))
        .send()
        .await
        .expect("the answer's head");

    let response = buffered_body(answer, StatusCode::OK).await;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = text_of(response).await;
    assert!(!body.contains(&addr.to_string()), "{body}");
}

#[tokio::test]
async fn a_relayed_body_forwards_the_bytes_and_is_an_event_stream_on_a_success_without_a_type() {
    let response = relayed_body(upstream(200, None, "data: x\n\n"), StatusCode::OK);
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        content_type_of(&response).as_deref(),
        Some("text/event-stream")
    );
    assert_eq!(bytes_of(response).await, b"data: x\n\n");
}

/// A worker source that selects its one worker (the crate's stub source
/// selects none).
#[derive(Debug)]
struct OneWorker(Arc<dyn ExternalWorker>);

#[async_trait]
impl WorkerSource for OneWorker {
    async fn select(
        &self,
        _req: &SelectWorkerRequest<'_>,
    ) -> Result<Arc<dyn ExternalWorker>, Response> {
        Ok(Arc::clone(&self.0))
    }

    fn retry_config(&self, _model_id: &str) -> Option<RetryConfig> {
        None
    }

    fn stats(&self) -> WorkerStats {
        WorkerStats::default()
    }

    fn external_workers(&self) -> Vec<Arc<dyn ExternalWorker>> {
        vec![Arc::clone(&self.0)]
    }
}

/// A chat request whose one message carries `content_bytes` of text.
fn chat_request(content_bytes: usize) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "x".repeat(content_bytes)}],
    }))
    .expect("a chat request")
}

/// The bytes this thread allocates while the route takes a request of
/// `content_bytes` of message text to its one attempt, against a worker
/// that refuses the connection: the serialized body is built before the
/// connect, and no socket's timing enters the count.
async fn bytes_allocated_by_the_route(
    deps: &ChatRouterContext<'_>,
    tenant: &TenantRequestMeta,
    content_bytes: usize,
) -> usize {
    let body = chat_request(content_bytes);
    let before = counting_alloc::bytes();
    let response = route_chat(deps, None, tenant, body, "m").await;
    let allocated = counting_alloc::bytes() - before;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        extract_error_code_from_response(&response),
        "upstream_error",
        "the answer is the one attempt's refused connect"
    );
    allocated
}

/// The serialized request is built once and moves into the share the
/// attempts send: the route does not copy it a second time. Read as the
/// bytes the route allocates per byte of message text between two sizes
/// in the same growth band of the serializers' buffers (the fixed costs
/// cancel), up to the one attempt's connect: the exact serialization's
/// text and value and the attempt's own serialization remain, five bytes
/// per byte at these sizes; the route before this change copied the value
/// once more and read six. The window's floor catches a route that answers
/// before it serializes; the error code ties the answer to the attempt.
#[expect(
    clippy::print_stderr,
    reason = "the numbers a pin reads are printed for the Numbers table"
)]
#[tokio::test]
async fn the_serialized_request_is_not_copied_a_second_time_for_the_attempts() {
    let refused = RefusedAddress::reserve();
    let url = format!("http://{}", refused.addr());
    let workers: Arc<dyn WorkerSource> = Arc::new(OneWorker(StubWorker::at(&url)));
    let provider_registry = ProviderRegistry::new();
    let shared_components = Arc::new(SharedComponents {
        client: http_client(),
    });
    let retry_config = RetryConfig {
        max_retries: 0,
        initial_backoff_ms: 0,
        max_backoff_ms: 0,
        backoff_multiplier: 1.0,
        jitter_factor: 0.0,
    };
    let deps = ChatRouterContext {
        workers: &workers,
        provider_registry: &provider_registry,
        shared_components: &shared_components,
        retry_config: &retry_config,
    };
    let tenant = TenantRequestMeta::new(TenantKey::new("t"));
    // Whatever a first request builds once is not the pin's.
    bytes_allocated_by_the_route(&deps, &tenant, 1024).await;
    let (small, large) = (48 * 1024, 96 * 1024);
    let at_small = bytes_allocated_by_the_route(&deps, &tenant, small).await;
    let at_large = bytes_allocated_by_the_route(&deps, &tenant, large).await;
    let per_thousand = at_large.saturating_sub(at_small) * 1000 / (large - small);
    eprintln!(
        "bytes allocated by the chat route: {at_small} for {small} bytes of message text, {at_large} for {large}: {per_thousand} per thousand bytes of text"
    );
    assert!(
        (4000..5500).contains(&per_thousand),
        "bytes allocated per thousand bytes of message text: {per_thousand}"
    );
}

#[tokio::test]
async fn external_chat_forwards_thinking_token_budget_fields_as_supplied() {
    use axum::{routing::post, Json, Router};
    use serde_json::{json, Value};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let workers: Arc<dyn WorkerSource> = Arc::new(OneWorker(StubWorker::at(&url)));
    let deps = ChatRouterContext {
        workers: &workers,
        provider_registry: &ProviderRegistry::new(),
        shared_components: &Arc::new(SharedComponents {
            client: http_client(),
        }),
        retry_config: &RetryConfig::default(),
    };
    // Return the body as received by the external provider, after its request transform.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|body: Json<Value>| async { body }),
    );
    let requests = async {
        let tenant = TenantRequestMeta::new(TenantKey::new("t"));
        for fields in [
            json!({"thinking_token_budget": 0}),
            json!({"custom_params": {"thinking_budget": 8, "other": true}}),
            json!({"thinking_token_budget": 0, "custom_params": {"thinking_budget": 8}}),
            json!({"thinking_token_budget": -1}),
            json!({"custom_params": {"thinking_budget": -1}}),
            json!({}),
        ] {
            let mut body = json!({"model":"m","messages":[{"role":"user","content":"hi"}]});
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let response = route_chat(
                &deps,
                None,
                &tenant,
                serde_json::from_value(body).unwrap(),
                "m",
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let received: Value = serde_json::from_slice(&bytes_of(response).await).unwrap();
            for key in ["thinking_token_budget", "custom_params"] {
                assert_eq!(
                    received.get(key),
                    fields.get(key),
                    "external request {fields}"
                );
            }
        }
    };
    tokio::select! {
        result = axum::serve(listener, app) => panic!("provider exited early: {result:?}"),
        () = requests => {}
    }
}

/// A chat request nothing answers: the caller's 503 says why without the
/// worker's address. The HTTP client's own error text ends with the
/// request URL.
#[tokio::test]
async fn a_failed_chat_send_answers_without_the_workers_address() {
    let refused = RefusedAddress::reserve();
    let url = format!("http://{}", refused.addr());
    let workers: Arc<dyn WorkerSource> = Arc::new(OneWorker(StubWorker::at(&url)));
    let provider_registry = ProviderRegistry::new();
    let shared_components = Arc::new(SharedComponents {
        client: http_client(),
    });
    let retry_config = RetryConfig {
        max_retries: 0,
        initial_backoff_ms: 0,
        max_backoff_ms: 0,
        backoff_multiplier: 1.0,
        jitter_factor: 0.0,
    };
    let deps = ChatRouterContext {
        workers: &workers,
        provider_registry: &provider_registry,
        shared_components: &shared_components,
        retry_config: &retry_config,
    };
    let tenant = TenantRequestMeta::new(TenantKey::new("t"));

    let response = route_chat(&deps, None, &tenant, chat_request(1), "m").await;

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = text_of(response).await;
    assert!(body.contains("Failed to contact upstream"), "{body}");
    assert!(!body.contains(&refused.addr().to_string()), "{body}");
}
