//! Chat completion routing for the OpenAI router.

use std::{sync::Arc, time::Instant};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use futures_util::StreamExt;
use openai_protocol::chat::ChatCompletionRequest;
use serde_json::to_value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    context::{ComponentRefs, PayloadState, RequestContext, SharedComponents, WorkerSelection},
    provider::ProviderRegistry,
    router::resolve_provider,
};
use crate::{
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        common::{
            header_utils::{apply_provider_headers, extract_auth_header},
            retry::{is_retryable_response, RetryExecutor},
            sse::SSE_CHANNEL_BUFFER,
            worker_selection::{SelectWorkerRequest, WorkerSelector},
        },
        error,
    },
    worker::{Endpoint, ProviderType, WorkerRegistry},
};

/// Shared context passed to chat routing functions.
pub(super) struct ChatRouterContext<'a> {
    pub worker_registry: &'a WorkerRegistry,
    pub provider_registry: &'a ProviderRegistry,
    pub shared_components: &'a Arc<SharedComponents>,
    pub retry_config: &'a RetryConfig,
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

    let selector = WorkerSelector::new(deps.worker_registry, &deps.shared_components.client);
    let worker = match selector
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers,
            provider: Some(ProviderType::OpenAI),
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

    let mut payload = match to_value(&body) {
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

    let response = RetryExecutor::execute_response_with_retry(
        deps.retry_config,
        |_attempt| {
            let client = client.clone();
            let url = url.clone();
            let payload = Arc::clone(&payload_json);
            let headers = Arc::clone(&headers_cloned);
            let worker_api_key = Arc::clone(&worker_api_key);
            let worker = Arc::clone(&worker);

            async move {
                let mut req = client.post(&url).json(&*payload);
                let auth_header =
                    extract_auth_header((*headers).as_ref(), (*worker_api_key).as_ref());
                req = apply_provider_headers(req, &url, auth_header.as_ref());

                if is_streaming {
                    req = req.header("Accept", "text/event-stream");
                }

                let resp = match req.send().await {
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

                if is_streaming {
                    let stream = resp.bytes_stream();
                    let (tx, rx) = mpsc::channel(SSE_CHANNEL_BUFFER);
                    #[expect(clippy::disallowed_methods, reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding")]
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
                    let mut response =
                        Response::new(Body::from_stream(ReceiverStream::new(rx)));
                    *response.status_mut() = status;
                    response
                        .headers_mut()
                        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                    response
                } else {
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
                        Err(e) => {
                            error::internal_error(
                                "upstream_error",
                                format!("Failed to read response: {e}"),
                            )
                        }
                    }
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
