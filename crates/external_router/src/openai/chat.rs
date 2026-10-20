//! Chat completion routing for the OpenAI router.

use std::{sync::Arc, time::Instant};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use futures_util::StreamExt;
use openai_protocol::{chat::ChatCompletionRequest, common::to_value_exact, model_type::Endpoint};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    context::{ComponentRefs, PayloadState, RequestContext, SharedComponents, WorkerSelection},
    provider::ProviderRegistry,
    router::resolve_provider,
};
use crate::{
    error,
    header_utils::{apply_provider_headers, extract_auth_header},
    known,
    metrics::{bool_to_static_str, metrics_labels, Metrics},
    retry::{is_retryable_response, RetryExecutor},
    sse::SSE_CHANNEL_BUFFER,
    tenant::TenantRequestMeta,
    upstream_body,
    worker::{SelectWorkerRequest, WorkerSource},
    RetryConfig,
};

/// Shared context passed to chat routing functions.
pub(super) struct ChatRouterContext<'a> {
    pub workers: &'a Arc<dyn WorkerSource>,
    pub provider_registry: &'a ProviderRegistry,
    pub shared_components: &'a Arc<SharedComponents>,
    pub retry_config: &'a RetryConfig,
}

/// One attempt against the worker: the payload posted with the caller's auth
/// and the provider's headers, as an event stream when the request streams.
async fn send_attempt(
    client: &reqwest::Client,
    url: &str,
    payload: &serde_json::Value,
    headers: Option<&HeaderMap>,
    worker_api_key: Option<&String>,
    streaming: bool,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut req = client.post(url).json(payload);
    let auth_header = extract_auth_header(headers, worker_api_key);
    req = apply_provider_headers(req, url, auth_header.as_ref());
    if streaming {
        req = req.header("Accept", "text/event-stream");
    }
    req.send().await
}

/// The upstream's success stream relayed to the client frame by frame; an
/// error on it aborts the client's stream (the error's text is not
/// delivered). The content type is the upstream's, else `text/event-stream`.
/// A refusal never comes here: [`upstream_body::answer_error`] answers it
/// whole.
fn relayed_body(resp: reqwest::Response, status: StatusCode) -> Response {
    let content_type = resp.headers().get(CONTENT_TYPE).cloned();
    let stream = resp.bytes_stream();
    let (tx, rx) = mpsc::channel(SSE_CHANNEL_BUFFER);
    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding"
    )]
    tokio::spawn(async move {
        let mut s = stream;
        while let Some(chunk) = s.next().await {
            match chunk {
                Ok(bytes) => {
                    if tx.send(Ok(bytes)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(format!("Stream error: {e}"))).await;
                    break;
                }
            }
        }
    });
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    *response.status_mut() = status;
    response.headers_mut().insert(
        CONTENT_TYPE,
        content_type.unwrap_or_else(|| HeaderValue::from_static("text/event-stream")),
    );
    response
}

/// The upstream's success body read whole under its status and content type;
/// a read failure is a 500. A refusal never comes here:
/// [`upstream_body::answer_error`] answers it.
async fn buffered_body(resp: reqwest::Response, status: StatusCode) -> Response {
    let content_type = resp.headers().get(CONTENT_TYPE).cloned();
    match resp.bytes().await {
        Ok(body) => {
            let mut response = Response::new(Body::from(body));
            *response.status_mut() = status;
            if let Some(ct) = content_type {
                response.headers_mut().insert(CONTENT_TYPE, ct);
            }
            response
        }
        Err(e) => error::internal_error("upstream_error", format!("Failed to read response: {e}")),
    }
}

/// Route a chat completion request to the appropriate upstream worker.
pub(super) async fn route_chat(
    deps: &ChatRouterContext<'_>,
    headers: Option<&HeaderMap>,
    tenant_meta: &TenantRequestMeta,
    body: ChatCompletionRequest,
    model_id: &str,
) -> Response {
    let start = Instant::now();
    let model = model_id;
    let streaming = body.stream;

    Metrics::record_router_request(
        metrics_labels::ROUTER_OPENAI,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        metrics_labels::ENDPOINT_CHAT,
        bool_to_static_str(streaming),
    );

    let worker = match deps
        .workers
        .select(&SelectWorkerRequest {
            model_id: model,
            headers,
            router: Some(known::OPENAI),
            ..Default::default()
        })
        .await
    {
        Ok(w) => w,
        Err(response) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_CHAT,
                metrics_labels::ERROR_NO_WORKERS,
            );
            return response;
        }
    };

    let mut payload = match to_value_exact(&body) {
        Ok(v) => v,
        Err(e) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_CHAT,
                metrics_labels::ERROR_VALIDATION,
            );
            return error::bad_request(
                "invalid_request",
                format!("Failed to serialize request: {e}"),
            );
        }
    };

    // Patch the serialized payload to use the effective model consistently.
    payload["model"] = serde_json::Value::String(model.to_owned());

    let provider = resolve_provider(deps.provider_registry, worker.as_ref(), model);
    if let Err(e) = provider.transform_request(&mut payload, Endpoint::Chat) {
        Metrics::record_router_error(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_CHAT,
            metrics_labels::ERROR_VALIDATION,
        );
        return error::bad_request("invalid_request", format!("Provider transform error: {e}"));
    }

    let mut ctx = RequestContext::for_chat(
        Arc::new(body),
        headers.cloned(),
        Some(model_id.to_string()),
        ComponentRefs::Shared(Arc::clone(deps.shared_components)),
    );
    ctx.tenant_request_meta = Some(tenant_meta.clone());

    ctx.state.worker = Some(WorkerSelection {
        worker: Arc::clone(&worker),
        provider,
    });

    let url = format!("{}/v1/chat/completions", worker.url());
    ctx.state.payload = Some(PayloadState {
        json: payload,
        url: url.clone(),
    });

    // Wrap values in Arc to avoid cloning large objects on each retry attempt
    #[expect(
        clippy::expect_used,
        reason = "payload is set earlier in this function; absence is a logic error"
    )]
    let payload_ref = ctx.payload().expect("Payload not prepared");
    let payload_json = Arc::new(payload_ref.json.clone());
    let client = ctx.components.client().clone();
    let headers_cloned = Arc::new(ctx.headers().cloned());
    let worker_api_key = Arc::new(worker.api_key().cloned());
    let is_streaming = ctx.is_streaming();
    let model_name: Arc<str> = Arc::from(model);

    let response = RetryExecutor::execute_response_with_retry(
        deps.retry_config,
        |_attempt| {
            let client = client.clone();
            let url = url.clone();
            let payload = Arc::clone(&payload_json);
            let headers = Arc::clone(&headers_cloned);
            let worker_api_key = Arc::clone(&worker_api_key);
            let worker = Arc::clone(&worker);
            let model_name = Arc::clone(&model_name);

            async move {
                let sent = send_attempt(
                    &client,
                    &url,
                    &payload,
                    (*headers).as_ref(),
                    (*worker_api_key).as_ref(),
                    is_streaming,
                )
                .await;
                let resp = match sent {
                    Ok(r) => r,
                    Err(e) => {
                        worker.record_outcome(503);
                        return error::service_unavailable(
                            "upstream_error",
                            format!("Failed to contact upstream: {e}"),
                        );
                    }
                };

                let status = StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                // Record CB outcome based on HTTP status.
                // For streaming: status is known upfront (200 = success).
                // For non-streaming: we record here too — body read errors
                // are connection issues, not worker health issues.
                worker.record_outcome(status.as_u16());

                if !status.is_success() {
                    // A refusal is answered whole and sanitized whether or
                    // not the request streams: the provider's account
                    // identifiers in its message are the operator's, not
                    // the caller's (as the Responses paths answer theirs).
                    upstream_body::answer_error(resp, &model_name).await
                } else if is_streaming {
                    relayed_body(resp, status)
                } else {
                    buffered_body(resp, status).await
                }
            }
        },
        |res, _attempt| is_retryable_response(res),
        |delay, attempt| {
            Metrics::record_worker_retry(
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::ENDPOINT_CHAT,
            );
            Metrics::record_worker_retry_backoff(attempt, delay);
        },
        || {
            Metrics::record_worker_retries_exhausted(
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::ENDPOINT_CHAT,
            );
        },
    )
    .await;

    if response.status().is_success() {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_CHAT,
            start.elapsed(),
        );
    } else {
        Metrics::record_router_error(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_CHAT,
            metrics_labels::ERROR_BACKEND,
        );
    }

    response
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;

    use super::*;

    fn upstream(
        status: u16,
        content_type: Option<&str>,
        body: impl Into<reqwest::Body>,
    ) -> reqwest::Response {
        let mut builder = http::Response::builder().status(status);
        if let Some(ct) = content_type {
            builder = builder.header(CONTENT_TYPE, ct);
        }
        reqwest::Response::from(builder.body(body.into()).expect("a response"))
    }

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

    #[tokio::test]
    async fn a_relayed_body_forwards_the_bytes_and_is_an_event_stream_on_a_success_without_a_type()
    {
        let response = relayed_body(upstream(200, None, "data: x\n\n"), StatusCode::OK);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            content_type_of(&response).as_deref(),
            Some("text/event-stream")
        );
        assert_eq!(bytes_of(response).await, b"data: x\n\n");
    }
}
