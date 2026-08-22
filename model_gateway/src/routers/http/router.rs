use std::{
    error::Error as _,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        HeaderMap, HeaderValue, Method, StatusCode,
    },
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use bytes::{Bytes, BytesMut};
use futures_util::{stream, Stream, StreamExt};
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    common::GenerationRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::{RerankRequest, RerankResponse, RerankResult},
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use reqwest::{
    multipart::{Form, Part},
    Client,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, warn};

use crate::{
    app_context::AppContext,
    config::types::{RetryConfig, RouterConfig},
    middleware::{scheduler::PreemptionGuard, TenantRequestMeta},
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::{
        common::{
            attach_sized_body, header_utils, overload,
            realtime::{
                rest::forward_realtime_rest, webrtc, webrtc::handle_realtime_webrtc,
                ws::handle_realtime_ws, RealtimeLabels, RealtimeRegistry,
            },
            request_lease::{ReleasePoint, RequestLease, RoutingDerivatives},
            retry::{is_retryable_response, is_retryable_status, RetryExecutor},
            sse::SSE_CHANNEL_BUFFER,
            worker_selection::{SelectWorkerRequest, WorkerSelector},
        },
        error::{self, extract_error_code_from_response},
        grpc::utils::{error_type_from_status, route_to_endpoint},
        http::{
            request_body::{serialize_request_body, RequestBodyError},
            request_stream::{CappedBodyStream, StreamProgress},
        },
        router_manager::RouterManager,
        RouterTrait,
    },
    worker::{AttachedBody, ConnectionMode, Worker, WorkerLoadGuard, WorkerRegistry, WorkerType},
};

/// Max body size for a WebRTC `/v1/realtime/calls` SDP offer (10 MiB).
const WEBRTC_REQUEST_BODY_LIMIT: usize = 10 * 1024 * 1024;

/// Error codes for streamed-body aborts the client caused. They carry no
/// worker verdict: recording a circuit-breaker sample for them would let slow
/// or oversized uploaders open a healthy worker's breaker.
const STREAMED_BODY_STALLED: &str = "request_body_stalled";
const STREAMED_BODY_TOO_LARGE: &str = "request_body_too_large";
const STREAMED_BODY_ABORTED: &str = "request_body_aborted";

/// Regular router that uses injected load balancing policies
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    retry_config: RetryConfig,
    /// Cap on buffered worker response bodies, mirroring the ingress limit.
    max_payload_size: usize,
    /// Streamed-body stall watchdog: abort a dispatch once a single client
    /// wait lasts this long. `None` disables.
    stream_stall_timeout: Option<Duration>,
    realtime_registry: Arc<RealtimeRegistry>,
    webrtc_bind_addr: Option<std::net::IpAddr>,
    webrtc_stun_server: Option<String>,
}

trait RealtimeRestRequest: serde::Serialize + Clone {
    fn model(&self) -> Option<&str>;
    fn set_model(&mut self, model: String);
}

impl RealtimeRestRequest for RealtimeSessionCreateRequest {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn set_model(&mut self, model: String) {
        self.model = Some(model);
    }
}

impl RealtimeRestRequest for RealtimeClientSecretCreateRequest {
    fn model(&self) -> Option<&str> {
        self.session.model.as_deref()
    }

    fn set_model(&mut self, model: String) {
        self.session.model = Some(model);
    }
}

impl RealtimeRestRequest for RealtimeTranscriptionSessionCreateRequest {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn set_model(&mut self, model: String) {
        self.model = Some(model);
    }
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("policy_registry", &self.policy_registry)
            .field("client", &self.client)
            .field("retry_config", &self.retry_config)
            .finish_non_exhaustive()
    }
}

impl Router {
    /// Create a new router with injected policy and client
    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other router constructors"
    )]
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            max_payload_size: ctx.router_config.max_payload_size,
            stream_stall_timeout: match ctx.router_config.stream_body_stall_timeout_secs {
                0 => None,
                secs => Some(Duration::from_secs(secs)),
            },
            realtime_registry: ctx.realtime_registry.clone(),
            webrtc_bind_addr: ctx.webrtc_bind_addr,
            webrtc_stun_server: ctx.webrtc_stun_server.clone(),
        })
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{worker_url}/{endpoint}"));
                for (name, value) in headers {
                    if header_utils::should_forward_request_header(&name) {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match send_with_stale_conn_retry(request_builder).await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {e}"),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    /// Select worker considering circuit breaker state.
    /// Filters to workers serving the specified model. When model is "unknown"
    /// (generate endpoint without model), considers all HTTP workers.
    fn select_worker_for_model(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
    ) -> Option<Arc<dyn Worker>> {
        // UNKNOWN_MODEL_ID means caller didn't specify a model — find any available worker
        let model_filter = if model_id == crate::worker::UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };
        let workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = self.policy_registry.get_policy_or_default(model_id);

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let idx = self.policy_registry.select_worker(
            &policy,
            &available,
            &SelectWorkerInfo {
                request_text: text,
                tokens,
                headers,
                routing_key: self.policy_registry.resolve_routing_key(headers),
                rid_key,
                hash_ring,
                leg: crate::policies::WorkerLeg::Single,
            },
        )?;

        // Record worker selection metric (Layer 3)
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            policy.name(),
        );

        Some(available[idx].clone())
    }

    /// Select a local, realtime-capable worker for the given model.
    ///
    /// Uses the shared [`WorkerSelector`] (least-loaded) filtered to regular
    /// HTTP workers advertising the `realtime` label, so realtime traffic
    /// never lands on a worker that can't serve it.
    async fn select_realtime_worker(
        &self,
        model_id: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Arc<dyn Worker>, Response> {
        WorkerSelector::new(&self.worker_registry, &self.client)
            .select_worker(&SelectWorkerRequest {
                model_id,
                headers,
                worker_type: Some(WorkerType::Regular),
                connection_mode: Some(ConnectionMode::Http),
                require_realtime_capable: true,
                ..Default::default()
            })
            .await
    }

    async fn route_realtime_rest<T: RealtimeRestRequest + Sync>(
        &self,
        headers: Option<&HeaderMap>,
        body: &T,
        endpoint: &'static str,
        endpoint_label: &'static str,
    ) -> Response {
        let requested_model = body.model().unwrap_or_default();
        let canonical_model = self.worker_registry.resolve_model_alias(requested_model);
        let model = canonical_model.as_deref().unwrap_or(requested_model);
        let worker = self.select_realtime_worker(model, headers).await;

        if let Some(canonical_model) = canonical_model.as_deref() {
            let mut body = body.clone();
            body.set_model(canonical_model.to_string());
            forward_realtime_rest(
                RealtimeLabels::HTTP,
                &self.client,
                worker,
                headers,
                &body,
                model,
                endpoint,
                endpoint_label,
            )
            .await
        } else {
            forward_realtime_rest(
                RealtimeLabels::HTTP,
                &self.client,
                worker,
                headers,
                body,
                model,
                endpoint,
                endpoint_label,
            )
            .await
        }
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: T,
        route: &'static str,
        model_id: &str,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        // Pre-tokenized requests route on the token tree; the decimal-string
        // rendering is only materialized when there are no tokens. A valid
        // x-smg-routing-tokens hint wins over body-derived tokens/text.
        let routing_tokens: Option<Vec<u32>> = header_utils::parse_routing_tokens_hint(headers)
            .or_else(|| {
                typed_req
                    .routing_tokens()
                    .map(|ids| ids.iter().map(|&id| id as u32).collect())
            });
        let text = routing_tokens
            .is_none()
            .then(|| typed_req.extract_text_for_routing());
        let rid_key = self
            .policy_registry
            .derive_rid_key(typed_req.rid())
            .map(str::to_string);
        // Resolve once, here, so every registry, policy and metrics lookup
        // below is keyed by the canonical model ID. Only `get_by_model`
        // understands aliases; retry configs, hash rings and policies do not,
        // and an alias would silently fall back to router defaults.
        let canonical_model = self.worker_registry.resolve_model_alias(model_id);
        let model_id = canonical_model.as_deref().unwrap_or(model_id);
        let model = model_id;
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(is_stream),
        );

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model_id);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        // The lease owns the parsed request and its routing derivatives for
        // the dispatch phase; its release point encodes the retry policy.
        let lease = RequestLease::new(
            typed_req,
            RoutingDerivatives {
                tokens: routing_tokens,
                text,
                rid_key,
            },
            ReleasePoint::from_retry_config(retry_config),
        );

        let response = if lease.release_point() == ReleasePoint::AfterDispatch {
            // Retries disabled: one dispatch; the lease frees the parsed
            // request the moment the upstream bytes are serialized.
            let res = self
                .route_typed_request_once(
                    headers,
                    &lease,
                    route,
                    model_id,
                    canonical_model.as_deref(),
                    is_stream,
                )
                .await;
            Metrics::record_router_upstream_response(
                metrics_labels::ROUTER_HTTP,
                res.status().as_u16(),
                extract_error_code_from_response(&res),
            );
            // Mirror the retry executor's exhaustion accounting for a
            // retryable response that gets no retry.
            if is_retryable_response(&res) {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_REGULAR, endpoint);
            }
            res
        } else {
            RetryExecutor::execute_response_with_retry(
                retry_config,
                // operation per attempt; the lease keeps the request alive
                // for replay until the retry window closes (first
                // non-retryable response).
                |_: u32| async {
                    let res = self
                        .route_typed_request_once(
                            headers,
                            &lease,
                            route,
                            model_id,
                            canonical_model.as_deref(),
                            is_stream,
                        )
                        .await;

                    // Need to be outside `route_typed_request_once` because that function has multiple return paths
                    Metrics::record_router_upstream_response(
                        metrics_labels::ROUTER_HTTP,
                        res.status().as_u16(),
                        extract_error_code_from_response(&res),
                    );

                    res
                },
                // should_retry predicate
                |res, _attempt| is_retryable_response(res),
                // on_backoff hook
                |delay, attempt| {
                    // Layer 3 worker metrics
                    Metrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                    Metrics::record_worker_retry_backoff(attempt, delay);
                },
                // on_exhausted hook
                || {
                    Metrics::record_worker_retries_exhausted(
                        metrics_labels::WORKER_REGULAR,
                        endpoint,
                    );
                },
            )
            .await
        };

        if response.status().is_success() {
            let duration = start.elapsed();
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_response(&response) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn route_typed_request_once<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        lease: &RequestLease<T>,
        route: &'static str,
        model_id: &str,
        canonical_model: Option<&str>,
        is_stream: bool,
    ) -> Response {
        let worker = match lease.with_view(|view| {
            self.select_worker_for_model(model_id, view.text, view.tokens, headers, view.rid_key)
        }) {
            Some(w) => w,
            None => {
                // Distinguish "no workers for this model" from "workers exist but unavailable"
                let model_filter = if model_id == crate::worker::UNKNOWN_MODEL_ID {
                    None
                } else {
                    Some(model_id)
                };
                let total = self.worker_registry.get_workers_filtered(
                    model_filter,
                    Some(WorkerType::Regular),
                    Some(ConnectionMode::Http),
                    None,
                    false,
                );
                // `total` is exactly the pool selection drew from, wildcard
                // model included — classifying from it rather than from the
                // model index is what makes the shed fire for a model-less
                // `/generate` and for a model that is also served over gRPC.
                return if total.is_empty() {
                    error::model_not_found(model_id)
                } else if let Some(shed) = overload::shed_if_all_overloaded(&total, model_id) {
                    shed
                } else {
                    error::service_unavailable(
                        "no_available_workers",
                        "All workers are unavailable (circuit breaker open or unhealthy)",
                    )
                };
            }
        };

        // Dispatch-time re-check of the one chosen worker: O(1), and the only
        // thing that closes the window between selection and dispatch in which
        // a load report can flip the veto.
        if let Some(shed) = overload::shed_if_worker_overloaded(worker.as_ref(), model_id) {
            return shed;
        }

        // Keyed-load accounting uses the same effective key as selection:
        // rid-derived first, header fallback.
        let load_guard = lease.with_view(|view| {
            WorkerLoadGuard::with_key(
                worker.clone(),
                view.rid_key
                    .or_else(|| self.policy_registry.sticky_header_key(headers)),
            )
        });

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();
        let raw_body_len = header_utils::content_length(headers);
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let response = match lease.serialize_with(|view| {
            serialize_request_body(view.request, canonical_model, worker.as_ref(), raw_body_len)
        }) {
            Ok(body) => {
                // Past this point dispatch needs only the serialized bytes;
                // the lease frees the parsed request and its routing
                // derivatives now when retries are disabled.
                lease.release_dispatch();
                self.send_serialized_request(
                    headers,
                    body,
                    route,
                    worker.as_ref(),
                    is_stream,
                    load_guard,
                )
                .await
            }
            Err(RequestBodyError::Serialize(e)) => error::bad_request(
                "serialization_failed",
                format!("Failed to serialize request body: {e}"),
            ),
            Err(RequestBodyError::Prepare(e)) => error::bad_request(
                "request_preparation_failed",
                format!("Failed to prepare request: {e}"),
            ),
        };

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        worker.record_outcome(status.as_u16());

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            Metrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the sglang worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }

        // Caller Authorization takes precedence over each worker's API key;
        // forward all other allow-listed headers without duplicating auth.
        let client_auth = header_utils::extract_auth_header(headers, None);
        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| {
                        !name.as_str().eq_ignore_ascii_case("authorization")
                            && header_utils::should_forward_request_header(name.as_str())
                    })
                    .collect()
            })
            .unwrap_or_default();

        let futures: Vec<_> = workers
            .into_iter()
            .map(|worker| {
                let url = format!("{}/{}", worker.base_url(), endpoint);
                let client = self.client.clone();
                let method = method.clone();

                let headers = filtered_headers.clone();
                let client_auth = client_auth.clone();
                let api_key = worker.api_key().cloned();

                async move {
                    let mut request_builder = match method {
                        Method::GET => client.get(url),
                        Method::POST => client.post(url),
                        _ => {
                            return Err(error::method_not_allowed(
                                "unsupported_method",
                                "Unsupported method for simple routing",
                            ))
                        }
                    };

                    // Caller header wins; fall back to the worker's API key.
                    let auth = client_auth.or_else(|| {
                        api_key.and_then(|k| HeaderValue::from_str(&format!("Bearer {k}")).ok())
                    });
                    if let Some(auth) = auth {
                        request_builder = request_builder.header("Authorization", auth);
                    }

                    for (name, value) in headers {
                        request_builder = request_builder.header(name.clone(), value.clone());
                    }

                    send_with_stale_conn_retry(request_builder)
                        .await
                        .map_err(convert_reqwest_error)
                }
            })
            .collect();

        // Now execute the collected futures concurrently
        let mut stream = stream::iter(futures).buffer_unordered(32);
        let mut last_response: Option<Response> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;

                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {e}"),
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(e);
                }
            }
        }

        last_response
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    /// Forward an audio transcription request to an audio-capable worker as
    /// `multipart/form-data`. Separate from `route_typed_request` because the
    /// endpoint is not JSON-bodied.
    async fn route_multipart_transcription(
        &self,
        headers: Option<&HeaderMap>,
        body: &TranscriptionRequest,
        audio: AudioFile,
        route: &'static str,
        model_id: &str,
    ) -> Response {
        let start = Instant::now();
        let is_stream = body.is_stream();
        // A valid x-smg-routing-tokens hint wins over body-derived text.
        let hinted_tokens = header_utils::parse_routing_tokens_hint(headers);
        let text = hinted_tokens
            .is_none()
            .then(|| body.extract_text_for_routing());
        // Resolve once, here, for the same reason as `route_typed_request`:
        // only `get_by_model` understands aliases, so the policy and hash ring
        // lookups below would silently fall back to router defaults on an
        // alias. This path cannot reuse that resolution because multipart
        // never goes through `route_typed_request`.
        let canonical_model = self.worker_registry.resolve_model_alias(model_id);
        let model_id = canonical_model.as_deref().unwrap_or(model_id);
        let endpoint = route_to_endpoint(route);

        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            endpoint,
            bool_to_static_str(is_stream),
        );

        // Finalize router metrics for an early error that never reached an
        // upstream worker (model_not_found, dp_aware_not_supported, no
        // available workers, build failure). Without this, pre-send failures
        // silently disappear from router_upstream_responses / router_error.
        let record_pre_send_error = |response: &Response| {
            let rstatus = response.status();
            Metrics::record_router_upstream_response(
                metrics_labels::ROUTER_HTTP,
                rstatus.as_u16(),
                extract_error_code_from_response(response),
            );
            // Response-aware: a terminal shed carries a retryable status but
            // must still count as a router error.
            if !is_retryable_response(response) {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    error_type_from_status(rstatus),
                );
            }
        };

        // Multipart transcription can't route through `worker.prepare_request`,
        // which is the hook that injects `data_parallel_rank` for DP-aware
        // workers. Pre-filter DP-aware workers out of the candidate pool so
        // the policy can pick a non-DP worker when one exists; only fall back
        // to model_not_found / 400 when every candidate is DP-aware.
        let model_filter = if model_id == crate::worker::UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };
        let all_workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,
            false,
        );
        if all_workers.is_empty() {
            let resp = error::model_not_found(model_id);
            record_pre_send_error(&resp);
            return resp;
        }
        let non_dp_workers: Vec<Arc<dyn Worker>> = all_workers
            .iter()
            .filter(|w| !w.is_dp_aware())
            .cloned()
            .collect();
        if non_dp_workers.is_empty() {
            let resp = error::bad_request(
                "dp_aware_not_supported",
                "/v1/audio/transcriptions does not yet support DP-aware workers",
            );
            record_pre_send_error(&resp);
            return resp;
        }
        let available: Vec<Arc<dyn Worker>> = non_dp_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            let resp =
                overload::shed_if_all_overloaded(&non_dp_workers, model_id).unwrap_or_else(|| {
                    error::service_unavailable(
                        "no_available_workers",
                        "All workers are unavailable (circuit breaker open or unhealthy)",
                    )
                });
            record_pre_send_error(&resp);
            return resp;
        }

        let policy = self.policy_registry.get_policy_or_default(model_id);
        let hash_ring = self.worker_registry.get_hash_ring(model_id);
        let idx = match self.policy_registry.select_worker(
            &policy,
            &available,
            &SelectWorkerInfo {
                request_text: text.as_deref(),
                tokens: hinted_tokens.as_deref(),
                headers,
                routing_key: self.policy_registry.resolve_routing_key(headers),
                rid_key: None,
                hash_ring,
                leg: crate::policies::WorkerLeg::Single,
            },
        ) {
            Some(i) => i,
            None => {
                let resp = error::service_unavailable(
                    "no_available_workers",
                    "Policy returned no eligible worker",
                );
                record_pre_send_error(&resp);
                return resp;
            }
        };
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            policy.name(),
        );
        let worker = available[idx].clone();

        // Same dispatch-time re-check the regular path takes. A transcription
        // occupies its worker for far longer than a chat completion, so a
        // report landing in the selection→dispatch window is the one case where
        // dispatching anyway is measurably worse.
        if let Some(resp) = overload::shed_if_worker_overloaded(worker.as_ref(), model_id) {
            record_pre_send_error(&resp);
            return resp;
        }

        // Streamed requests have no rid; the header is the whole sticky key.
        let load_guard = WorkerLoadGuard::with_key(
            worker.clone(),
            self.policy_registry.sticky_header_key(headers),
        );

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        events::RequestSentEvent { url: worker.url() }.emit();

        let form = match build_transcription_form(body, audio, canonical_model.as_deref()) {
            Ok(f) => f,
            Err(e) => {
                let resp = error::bad_request("multipart_build_failed", e);
                record_pre_send_error(&resp);
                return resp;
            }
        };

        let endpoint_url = worker.endpoint_url(route);
        let mut request_builder = self.client.post(&endpoint_url).multipart(form);

        // reqwest sets the multipart Content-Type (with boundary) itself; the
        // forward allow-list already excludes Content-Type/Content-Length.
        request_builder = header_utils::apply_forwarded_request_headers(
            request_builder,
            headers,
            worker.api_key(),
        );

        let res = match send_with_stale_conn_retry(request_builder).await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send multipart transcription request worker_url={} route={} error={}",
                    worker.url(),
                    route,
                    e
                );
                let err_resp = convert_reqwest_error(e);
                let err_status = err_resp.status();
                // Feed the synthetic status into the worker circuit breaker
                // and worker-error metric; transport failures (timeouts,
                // connect errors) must be visible to health tracking so the
                // same bad worker isn't picked repeatedly.
                worker.record_outcome(err_status.as_u16());
                if err_status.is_server_error() {
                    Metrics::record_worker_error(
                        metrics_labels::WORKER_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        error_type_from_status(err_status),
                    );
                }
                Metrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    err_status.as_u16(),
                    extract_error_code_from_response(&err_resp),
                );
                // Mirror route_typed_request: a send failure must still bump
                // the terminal router_error counter, not just upstream_response.
                Metrics::record_router_error(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    error_type_from_status(err_status),
                );
                return err_resp;
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        Metrics::record_router_upstream_response(metrics_labels::ROUTER_HTTP, status.as_u16(), "");

        events::RequestReceivedEvent {}.emit();

        let response = if is_stream {
            // Preserve the upstream content-type verbatim. A `stream=true`
            // hint from the client doesn't guarantee the worker actually
            // streams — whisper backends may ignore it and return a normal
            // JSON body (success or 4xx error). Don't relabel non-SSE
            // responses as SSE; leave that judgment to whatever the worker
            // set.
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            header_utils::insert_routed_worker_id(&mut response_headers, worker.url());
            let stream = res.bytes_stream();
            // Bounded channel applies backpressure: if the downstream client
            // is slow, the upstream relay awaits on `send` rather than piling
            // chunks in memory.
            const STREAM_RELAY_BUFFER: usize = 32;
            let (tx, rx) = mpsc::channel::<Result<Bytes, String>>(STREAM_RELAY_BUFFER);
            // Attribute worker-level and router-level outcomes to the actual
            // stream completion from inside the relay task: a mid-stream error
            // after a 2xx header, or a non-streaming 5xx header returned under
            // `stream=true`, must be visible to circuit-breaker + worker-error
            // + router-error metrics. Recording only at header time would mis-
            // classify those.
            let worker_for_stream = worker.clone();
            let stream_header_status = status;
            let stream_model_id = model_id.to_string();
            let stream_endpoint = endpoint;
            let stream_start = start;
            #[expect(
                clippy::disallowed_methods,
                reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding"
            )]
            tokio::spawn(async move {
                let mut stream = stream;
                let mut stream_failed = false;
                let mut client_disconnected = false;
                loop {
                    tokio::select! {
                        chunk = stream.next() => match chunk {
                            Some(Ok(bytes)) => {
                                if tx.send(Ok(bytes)).await.is_err() {
                                    client_disconnected = true;
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                stream_failed = true;
                                let _ = tx.send(Err(format!("Stream error: {e}"))).await;
                                break;
                            }
                            None => break,
                        },
                        // Client gone with no chunk in flight (long prefill,
                        // stalled upstream): break so the reqwest stream drops,
                        // closing the upstream connection and letting the
                        // engine abort generation.
                        () = tx.closed() => {
                            client_disconnected = true;
                            break;
                        }
                    }
                }
                // Effective status = BAD_GATEWAY if the relay failed, else the
                // worker's header status. Covers both "5xx header returned
                // while stream=true" and "200 header then mid-stream break".
                let effective_status = if stream_failed {
                    StatusCode::BAD_GATEWAY
                } else {
                    stream_header_status
                };
                worker_for_stream.record_outcome(effective_status.as_u16());
                if effective_status.is_server_error() {
                    Metrics::record_worker_error(
                        metrics_labels::WORKER_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        error_type_from_status(effective_status),
                    );
                }
                // A client disconnect gets no terminal router metric: it is
                // neither a completed request (duration) nor a router or
                // worker failure (error). Worker-level attribution above
                // still applies — the header status is a worker fact.
                if client_disconnected {
                    return;
                }
                if effective_status.is_success() {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_HTTP,
                        metrics_labels::BACKEND_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        &stream_model_id,
                        stream_endpoint,
                        stream_start.elapsed(),
                    );
                } else {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_HTTP,
                        metrics_labels::BACKEND_REGULAR,
                        metrics_labels::CONNECTION_HTTP,
                        &stream_model_id,
                        stream_endpoint,
                        error_type_from_status(effective_status),
                    );
                }
            });
            let stream = ReceiverStream::new(rx);
            let body = Body::from_stream(stream);
            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;
            response = AttachedBody::wrap_response(response, load_guard);
            response
        } else {
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            header_utils::insert_routed_worker_id(&mut response_headers, worker.url());
            match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => error::internal_error(
                    "read_response_body_failed",
                    format!("Failed to read response body: {e}"),
                ),
            }
        };

        // Non-streaming: classify metrics off the final response the client
        // will actually see. A body-read failure can rewrite a 2xx upstream
        // into a local 5xx, and we want the circuit breaker + metrics to see
        // that. Streaming outcomes are owned by the relay task above.
        if !is_stream {
            let final_status = response.status();
            worker.record_outcome(final_status.as_u16());
            if final_status.is_server_error() {
                Metrics::record_worker_error(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    error_type_from_status(final_status),
                );
            }
            if final_status.is_success() {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    start.elapsed(),
                );
            } else {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_HTTP,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    model_id,
                    endpoint,
                    error_type_from_status(final_status),
                );
            }
        }

        response
    }

    /// Buffer a worker response body, capped at `limit` bytes; a larger body
    /// is a misbehaving worker and yields a 502 before memory balloons.
    async fn read_worker_body_capped<S, E>(mut stream: S, limit: usize) -> Result<Bytes, Response>
    where
        S: Stream<Item = Result<Bytes, E>> + Unpin,
        E: std::fmt::Display,
    {
        let mut body = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    return Err(error::internal_error(
                        "read_response_body_failed",
                        format!("Failed to get response body: {e}"),
                    ));
                }
            };
            if body.len().saturating_add(chunk.len()) > limit {
                warn!(limit, "Worker response exceeded the body limit");
                return Err(error::bad_gateway(
                    "upstream_response_too_large",
                    format!("Response from worker exceeded {limit} bytes"),
                ));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body.freeze())
    }

    // Send an already-serialized request body. The stale-connection resend
    // guard inside `send_with_stale_conn_retry` shares the body allocation by
    // refcount, so the bytes live exactly until the response head arrives.
    async fn send_serialized_request(
        &self,
        headers: Option<&HeaderMap>,
        body: Bytes,
        route: &'static str,
        worker: &dyn Worker,
        is_stream: bool,
        load_guard: WorkerLoadGuard,
    ) -> Response {
        let api_key = worker.api_key().cloned();
        let endpoint_url = worker.endpoint_url(route);

        let mut request_builder = attach_sized_body(
            self.client
                .post(&endpoint_url)
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json")),
            body,
        );

        request_builder = header_utils::apply_forwarded_request_headers(
            request_builder,
            headers,
            api_key.as_ref(),
        );

        let res = match send_with_stale_conn_retry(request_builder).await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker.url(),
                    route,
                    e
                );

                return convert_reqwest_error(e);
            }
        };

        self.forward_worker_response(res, is_stream, worker.url(), load_guard)
            .await
    }

    /// Relay a worker response to the client. A streaming response flows
    /// through a bounded channel with the load guard attached to the body; a
    /// buffered response is read capped at the ingress payload limit.
    async fn forward_worker_response(
        &self,
        res: reqwest::Response,
        is_stream: bool,
        worker_url: &str,
        load_guard: WorkerLoadGuard,
    ) -> Response {
        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if is_stream {
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            header_utils::insert_routed_worker_id(&mut response_headers, worker_url);
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            let stream = res.bytes_stream();
            // Bounded channel applies backpressure: a slow client makes the
            // relay await on `send` instead of buffering the whole response.
            let (tx, rx) = mpsc::channel(SSE_CHANNEL_BUFFER);

            // Spawn task to forward stream
            #[expect(
                clippy::disallowed_methods,
                reason = "fire-and-forget stream relay; gateway shutdown need not wait for individual stream forwarding"
            )]
            tokio::spawn(async move {
                let mut stream = stream;
                loop {
                    tokio::select! {
                        chunk = stream.next() => match chunk {
                            Some(Ok(bytes)) => {
                                if tx.send(Ok(bytes)).await.is_err() {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                let _ = tx.send(Err(format!("Stream error: {e}"))).await;
                                break;
                            }
                            None => break,
                        },
                        // Client gone with no chunk in flight (long prefill,
                        // stalled upstream): break so the reqwest stream drops,
                        // closing the upstream connection and letting the
                        // engine abort generation.
                        () = tx.closed() => break,
                    }
                }
            });

            let stream = ReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;

            // Attach load guard to response body for proper RAII lifecycle
            // Guard is dropped when response body is consumed or client disconnects
            response = AttachedBody::wrap_response(response, load_guard);
            response
        } else {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            // Cap the buffered read at the ingress payload limit; this is the
            // point where an upstream body is first pulled into memory.
            let mut response = match Self::read_worker_body_capped(
                res.bytes_stream(),
                self.max_payload_size,
            )
            .await
            {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(error_response) => error_response,
            };
            header_utils::insert_routed_worker_id(response.headers_mut(), worker_url);

            // load_guard dropped here automatically after response body is read
            response
        }
    }

    /// Forward a raw request body to `worker` as a chunked stream.
    ///
    /// The body is never buffered: a counting wrapper caps it at the ingress
    /// payload limit (over-limit → 413, upstream send aborted) and a watchdog
    /// aborts the dispatch once the sender waits on the client for
    /// `stream_stall_timeout` (→ 408; the clock pauses under worker
    /// backpressure). The response relay is the buffered path's, with SSE
    /// detected from the worker's Content-Type because the request's `stream`
    /// flag is unread.
    async fn send_streamed_request(
        &self,
        headers: &HeaderMap,
        body: Body,
        route: &'static str,
        worker: &dyn Worker,
        load_guard: WorkerLoadGuard,
    ) -> Response {
        let api_key = worker.api_key().cloned();
        let endpoint_url = worker.endpoint_url(route);

        let progress = Arc::new(StreamProgress::new());
        let capped = CappedBodyStream::new(
            body.into_data_stream(),
            self.max_payload_size,
            Arc::clone(&progress),
        );

        let mut request_builder = self
            .client
            .post(&endpoint_url)
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .body(reqwest::Body::wrap_stream(capped));
        request_builder = header_utils::apply_forwarded_request_headers(
            request_builder,
            Some(headers),
            api_key.as_ref(),
        );

        let send = send_with_stale_conn_retry(request_builder);
        tokio::pin!(send);
        let sent = tokio::select! {
            sent = &mut send => sent,
            () = progress.stalled(self.stream_stall_timeout) => {
                let timeout_secs = self.stream_stall_timeout.map_or(0, |d| d.as_secs());
                warn!(
                    timeout_secs,
                    "Streamed request body stalled waiting on the client; aborting dispatch"
                );
                return error::create_error(
                    StatusCode::REQUEST_TIMEOUT,
                    STREAMED_BODY_STALLED,
                    format!(
                        "No request body bytes arrived from the client for {timeout_secs} seconds"
                    ),
                );
            }
        };

        let res = match sent {
            Ok(res) => res,
            Err(_) if progress.limit_exceeded() => {
                warn!(
                    limit = self.max_payload_size,
                    "Streamed request body exceeded the payload limit"
                );
                return error::create_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    STREAMED_BODY_TOO_LARGE,
                    format!("Request body exceeded {} bytes", self.max_payload_size),
                );
            }
            Err(_) if progress.inbound_error() => {
                return error::create_error(
                    StatusCode::BAD_REQUEST,
                    STREAMED_BODY_ABORTED,
                    "The request body stream failed before it was fully forwarded",
                );
            }
            Err(e) => {
                error!(
                    "Failed to send streamed request worker_url={} route={} error={}",
                    worker.url(),
                    route,
                    e
                );
                return convert_reqwest_error(e);
            }
        };

        let is_stream = res
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|ct| ct.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        self.forward_worker_response(res, is_stream, worker.url(), load_guard)
            .await
    }

    /// Build the public rerank response.
    ///
    /// Rerank is the one HTTP route whose response the gateway constructs
    /// itself instead of passing the worker's through, so the model it reports
    /// has to be canonicalized here. `canonical_model` is set only when the
    /// client addressed the model by an alias; reporting the alias would make
    /// this route disagree with every other one about which model ran.
    ///
    /// The worker body read is capped at `max_body_bytes`; a larger body is a
    /// misbehaving worker and yields a 502.
    async fn build_rerank_response(
        req: RerankResponseSpec,
        canonical_model: Option<&str>,
        response: Response,
        max_body_bytes: usize,
    ) -> Response {
        let (_, response_body) = response.into_parts();
        let body_bytes = match to_bytes(response_body, max_body_bytes).await {
            Ok(bytes) => bytes,
            Err(e) => {
                if e.source()
                    .and_then(|s| s.downcast_ref::<http_body_util::LengthLimitError>())
                    .is_some()
                {
                    warn!(
                        limit = max_body_bytes,
                        "Rerank worker response exceeded the body limit"
                    );
                    return error::bad_gateway(
                        "upstream_response_too_large",
                        format!("Rerank response from worker exceeded {max_body_bytes} bytes"),
                    );
                }
                error!("Failed to read rerank worker response: {e}");
                return error::internal_error(
                    "rerank_response_build_failed",
                    "Failed to build rerank response",
                );
            }
        };
        let rerank_results = match serde_json::from_slice::<Vec<RerankResult>>(&body_bytes) {
            Ok(results) => results,
            Err(e) => {
                error!("Failed to build rerank response: {e}");
                return error::internal_error(
                    "rerank_response_build_failed",
                    "Failed to build rerank response",
                );
            }
        };
        let model = canonical_model.map_or(req.model, ToOwned::to_owned);
        let mut rerank_response = RerankResponse::new(rerank_results, model, req.rid);
        // Sorting is handled by Python worker (serving_rerank.py)
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Json(rerank_response).into_response()
    }
}

/// Post-dispatch inputs for the rerank response builder; the request itself
/// (documents included) is released at dispatch.
struct RerankResponseSpec {
    model: String,
    rid: Option<openai_protocol::common::StringOrArray>,
    top_k: Option<usize>,
    return_documents: bool,
}

impl From<&RerankRequest> for RerankResponseSpec {
    fn from(req: &RerankRequest) -> Self {
        Self {
            model: req.model.clone(),
            rid: req.rid.clone(),
            top_k: req.top_k,
            return_documents: req.return_documents,
        }
    }
}

/// Build the multipart body forwarded to the worker.
///
/// `canonical_model` is set only when the client addressed the model by an
/// alias. The worker was registered under the canonical ID and has never heard
/// of the alias, so that is the name the form carries.
fn build_transcription_form(
    body: &TranscriptionRequest,
    audio: AudioFile,
    canonical_model: Option<&str>,
) -> Result<Form, String> {
    let AudioFile {
        bytes,
        file_name,
        content_type,
    } = audio;

    // Wrap the already-buffered Bytes in a reqwest Body (Arc refcount, no
    // additional copy) instead of Part::bytes, which would force a Vec copy.
    let file_len = bytes.len() as u64;
    let mut file_part =
        Part::stream_with_length(reqwest::Body::from(bytes), file_len).file_name(file_name);
    if let Some(ct) = content_type.as_deref() {
        file_part = file_part
            .mime_str(ct)
            .map_err(|e| format!("Invalid audio content-type '{ct}': {e}"))?;
    }

    let mut form = Form::new().part("file", file_part).text(
        "model",
        canonical_model.map_or_else(|| body.model.clone(), ToOwned::to_owned),
    );

    if let Some(ref language) = body.language {
        form = form.text("language", language.clone());
    }
    if let Some(ref prompt) = body.prompt {
        form = form.text("prompt", prompt.clone());
    }
    if let Some(ref fmt) = body.response_format {
        form = form.text("response_format", fmt.clone());
    }
    if let Some(temp) = body.temperature {
        form = form.text("temperature", temp.to_string());
    }
    if let Some(ref grans) = body.timestamp_granularities {
        for g in grans {
            form = form.text("timestamp_granularities[]", g.clone());
        }
    }
    if let Some(stream) = body.stream {
        form = form.text("stream", stream.to_string());
    }

    Ok(form)
}

/// True for transport failures surfaced without any response: no upstream
/// status, not a timeout, and not a response-phase (body/decode) or local
/// (builder/redirect) error. The dominant producer is a pooled connection
/// the backend closed while idle; the backend never processed the request,
/// so one resend is safe for any route.
fn is_pre_response_transport_error(e: &reqwest::Error) -> bool {
    e.status().is_none()
        && !e.is_timeout()
        && !e.is_body()
        && !e.is_decode()
        && !e.is_builder()
        && !e.is_redirect()
}

/// Send with a single retry on pre-response transport failures. Requests
/// whose body cannot be cloned (multipart streams) fail through unchanged.
pub(crate) async fn send_with_stale_conn_retry(
    builder: reqwest::RequestBuilder,
) -> Result<reqwest::Response, reqwest::Error> {
    let retry = builder.try_clone();
    match builder.send().await {
        Err(e) if is_pre_response_transport_error(&e) => match retry {
            Some(retry) => {
                Metrics::record_upstream_send_retry(metrics_labels::ROUTER_HTTP);
                retry.send().await
            }
            None => Err(e),
        },
        other => other,
    }
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{e}. URL: {url}");

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

impl Router {
    /// Streamed pass-through: select a worker from headers alone and
    /// forward the raw body verbatim (JSON validation and normalization
    /// defer to the worker). `Err` hands the request back untouched —
    /// body unconsumed — for the buffered typed path. Callers gate on
    /// the size threshold via [`stream_large_request_bodies`].
    pub(crate) async fn route_streaming_request(
        &self,
        req: Request<Body>,
        route: &'static str,
    ) -> Result<Response, Request<Body>> {
        // Content-blind eligibility: the model and stream flag sit inside the
        // unread body, so any registered policy that routes on request text
        // (the body's model could select it) forces the buffered path unless
        // a valid routing hint header stands in for the text. Every fallback
        // hands the request back with its body unconsumed.
        let model_id = crate::worker::UNKNOWN_MODEL_ID;
        if self
            .policy_registry
            .any_policy_needs_request_text(Some(req.headers()))
        {
            return Err(req);
        }
        // A body-mutating worker (`prepare_request`) anywhere in the fleet
        // forces the buffered path: the mutation needs the parsed body.
        let candidates = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,
            false,
        );
        if candidates.iter().any(|w| w.mutates_request()) {
            return Err(req);
        }
        // Buffered-path parity: a valid tokens hint is exactly what selection
        // would have received there (text is never extracted alongside it).
        // Streamed requests have no readable body, hence no rid key; the
        // sticky override keys them by the header alone.
        let hinted_tokens = header_utils::parse_routing_tokens_hint(Some(req.headers()));
        let Some(worker) = self.select_worker_for_model(
            model_id,
            None,
            hinted_tokens.as_deref(),
            Some(req.headers()),
            None,
        ) else {
            return Err(req);
        };
        // Guards a registration race: a mutating worker that joined after the
        // fleet check above must still not receive an unmutated stream.
        if worker.mutates_request() {
            return Err(req);
        }

        let start = Instant::now();
        let endpoint = route_to_endpoint(route);
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            endpoint,
            "false",
        );

        let load_guard = WorkerLoadGuard::with_key(
            worker.clone(),
            self.policy_registry.sticky_header_key(Some(req.headers())),
        );
        events::RequestSentEvent { url: worker.url() }.emit();

        let (parts, body) = req.into_parts();
        let mut headers_with_trace = parts.headers;
        inject_trace_context_http(&mut headers_with_trace);

        let response = self
            .send_streamed_request(
                &headers_with_trace,
                body,
                route,
                worker.as_ref(),
                load_guard,
            )
            .await;

        events::RequestReceivedEvent {}.emit();
        let status = response.status();
        let error_code = extract_error_code_from_response(&response);
        if error_code != STREAMED_BODY_STALLED
            && error_code != STREAMED_BODY_TOO_LARGE
            && error_code != STREAMED_BODY_ABORTED
        {
            worker.record_outcome(status.as_u16());
            if status.is_server_error() {
                Metrics::record_worker_error(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::CONNECTION_HTTP,
                    error_type_from_status(status),
                );
            }
        }
        Metrics::record_router_upstream_response(
            metrics_labels::ROUTER_HTTP,
            status.as_u16(),
            error_code,
        );
        if status.is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                endpoint,
                start.elapsed(),
            );
        } else if !is_retryable_status(status) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model_id,
                endpoint,
                error_type_from_status(status),
            );
        }
        Ok(response)
    }
}

/// Whether the request qualifies for the streamed body pass-through:
/// `--stream-request-bodies-over` is on and the declared Content-Length
/// exceeds it. Chunked uploads carry no Content-Length and always buffer.
fn exceeds_stream_threshold(threshold: u64, req: &Request<Body>) -> bool {
    threshold > 0
        && req
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .is_some_and(|len| len > threshold)
}

/// State for [`stream_large_request_bodies`]: the app-level router handle
/// (the manager in production, a concrete router in tests) and the streaming
/// threshold.
#[derive(Clone)]
pub struct StreamBodyState {
    router: Arc<dyn RouterTrait>,
    threshold: u64,
}

impl StreamBodyState {
    pub fn new(router: Arc<dyn RouterTrait>, config: &RouterConfig) -> Self {
        Self {
            router,
            threshold: config.stream_request_bodies_over,
        }
    }
}

/// Route-layer middleware: a typed-JSON request whose declared Content-Length
/// exceeds the threshold is offered to the HTTP regular router's streamed
/// pass-through before the handler's `Json`/`ValidatedJson` extractor can
/// buffer it. Any decline — other route, below threshold, no HTTP regular
/// router, router-side ineligibility — falls through to the untouched
/// handler with the body unconsumed. Sits inside the admission layer, so
/// streamed requests hold an admission permit and race the preemption token.
pub async fn stream_large_request_bodies(
    State(state): State<StreamBodyState>,
    cancel: PreemptionGuard,
    req: Request<Body>,
    next: Next,
) -> Response {
    let route: &'static str = match req.uri().path() {
        "/generate" => "/generate",
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/completions" => "/v1/completions",
        "/v1/messages" => "/v1/messages",
        "/v1/embeddings" => "/v1/embeddings",
        "/v1/classify" => "/v1/classify",
        _ => return next.run(req).await,
    };
    if !exceeds_stream_threshold(state.threshold, &req) {
        return next.run(req).await;
    }
    let resolved = match state.router.as_any().downcast_ref::<RouterManager>() {
        // Multi-router deployments route by the model inside the body; a
        // content-blind dispatch could pick a router that does not serve it.
        Some(manager) if manager.router_count() > 1 => return next.run(req).await,
        Some(manager) => match manager.select_router_for_request(None) {
            Some(selected) => selected,
            None => return next.run(req).await,
        },
        None => Arc::clone(&state.router),
    };
    let Some(http) = resolved.as_any().downcast_ref::<Router>() else {
        return next.run(req).await;
    };
    cancel
        .guard(async move {
            match http.route_streaming_request(req, route).await {
                Ok(response) => response,
                Err(req) => next.run(req).await,
            }
        })
        .await
}

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: GenerateRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/messages", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: CompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{response_id}/cancel");
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: ClassifyRequest,
        model_id: &str,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/classify", model_id)
            .await
    }

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        self.route_multipart_transcription(
            headers,
            body,
            audio,
            "/v1/audio/transcriptions",
            model_id,
        )
        .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: RerankRequest,
        model_id: &str,
    ) -> Response {
        let canonical_model = self.worker_registry.resolve_model_alias(model_id);
        let response_spec = RerankResponseSpec::from(&body);
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            Self::build_rerank_response(
                response_spec,
                canonical_model.as_deref(),
                response,
                self.max_payload_size,
            )
            .await
        } else {
            response
        }
    }

    async fn route_realtime_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeSessionCreateRequest,
    ) -> Response {
        self.route_realtime_rest(
            headers,
            body,
            "/v1/realtime/sessions",
            metrics_labels::ENDPOINT_REALTIME_SESSIONS,
        )
        .await
    }

    async fn route_realtime_client_secret(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        self.route_realtime_rest(
            headers,
            body,
            "/v1/realtime/client_secrets",
            metrics_labels::ENDPOINT_REALTIME_CLIENT_SECRETS,
        )
        .await
    }

    async fn route_realtime_transcription_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        self.route_realtime_rest(
            headers,
            body,
            "/v1/realtime/transcription_sessions",
            metrics_labels::ENDPOINT_REALTIME_TRANSCRIPTION,
        )
        .await
    }

    async fn route_realtime_ws(&self, req: Request<Body>, model: &str) -> Response {
        let (parts, _body) = req.into_parts();

        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_WEBSOCKET,
            model,
            metrics_labels::ENDPOINT_REALTIME,
            "false",
        );

        let auth_header = header_utils::extract_auth_header(Some(&parts.headers), None);
        let worker = self
            .select_realtime_worker(model, Some(&parts.headers))
            .await;

        handle_realtime_ws(
            RealtimeLabels::HTTP,
            parts,
            model.to_owned(),
            worker,
            auth_header,
            Arc::clone(&self.realtime_registry),
        )
        .await
    }

    async fn route_realtime_webrtc(&self, req: Request<Body>, model: &str) -> Response {
        let (parts, body) = req.into_parts();
        let body = match to_bytes(body, WEBRTC_REQUEST_BODY_LIMIT).await {
            Ok(b) => b,
            Err(e) => {
                if e.source()
                    .and_then(|s| s.downcast_ref::<http_body_util::LengthLimitError>())
                    .is_some()
                {
                    return StatusCode::PAYLOAD_TOO_LARGE.into_response();
                }
                return error::bad_request("invalid_body", format!("Failed to read body: {e}"));
            }
        };

        let parsed = match webrtc::parse_webrtc_request(&parts, &body, model).await {
            Ok(p) => p,
            Err(resp) => return resp,
        };

        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_WEBRTC,
            &parsed.model,
            metrics_labels::ENDPOINT_REALTIME,
            "false",
        );

        let auth_header = header_utils::extract_auth_header(Some(&parts.headers), None);
        let worker = self
            .select_realtime_worker(&parsed.model, Some(&parts.headers))
            .await;

        let bind_addr = self
            .webrtc_bind_addr
            .unwrap_or_else(|| std::net::Ipv4Addr::UNSPECIFIED.into());

        handle_realtime_webrtc(
            RealtimeLabels::HTTP,
            parts.headers,
            parsed,
            worker,
            auth_header,
            self.client.clone(),
            bind_addr,
            self.webrtc_stun_server.clone(),
            Arc::clone(&self.realtime_registry),
        )
        .await
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use std::{
        net::SocketAddr,
        sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
    };

    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        config::types::{PolicyConfig, RoutingKeyOverrideConfig},
        policies::{CacheAwareLengthPolicy, CacheAwarePolicy},
        routers::common::request_lease::test_probe::{spawn_release_gated_stub, DropProbeRequest},
        worker::BasicWorkerBuilder,
    };

    /// Accepts `kill_first` connections and closes them before any response
    /// bytes, then serves a minimal 200 on every later connection. Returns
    /// (addr, accepted-connection counter).
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only server task; lives no longer than the test"
    )]
    async fn flaky_upstream(kill_first: usize) -> (SocketAddr, Arc<AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_clone = Arc::clone(&accepted);
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let n = accepted_clone.fetch_add(1, AtomicOrdering::SeqCst);
                if n < kill_first {
                    drop(sock);
                    continue;
                }
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                    .await;
            }
        });
        (addr, accepted)
    }

    #[tokio::test]
    async fn stale_conn_retry_recovers_on_second_connection() {
        let (addr, accepted) = flaky_upstream(1).await;
        let client = Client::new();
        let builder = client.post(format!("http://{addr}/generate")).body("{}");

        let res = send_with_stale_conn_retry(builder).await.unwrap();
        assert_eq!(res.status().as_u16(), 200);
        assert_eq!(accepted.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn stale_conn_retry_is_bounded_to_one() {
        let (addr, accepted) = flaky_upstream(usize::MAX).await;
        let client = Client::new();
        let builder = client.post(format!("http://{addr}/generate")).body("{}");

        let err = send_with_stale_conn_retry(builder).await.unwrap_err();
        assert!(is_pre_response_transport_error(&err));
        assert_eq!(accepted.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn stale_conn_retry_skips_unclonable_bodies() {
        let (addr, accepted) = flaky_upstream(usize::MAX).await;
        let client = Client::new();
        let stream_body = reqwest::Body::wrap_stream(stream::once(async {
            Ok::<_, std::io::Error>(Bytes::from_static(b"{}"))
        }));
        let builder = client
            .post(format!("http://{addr}/generate"))
            .body(stream_body);

        send_with_stale_conn_retry(builder).await.unwrap_err();
        assert_eq!(accepted.load(AtomicOrdering::SeqCst), 1);
    }

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));

        // Register test workers
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        worker_registry.register_or_replace(Arc::new(worker1));
        worker_registry.register_or_replace(Arc::new(worker2));

        Router {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            max_payload_size: 536_870_912,
            stream_stall_timeout: Some(Duration::from_secs(60)),
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_status(openai_protocol::worker::WorkerStatus::NotReady);
        router
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_with_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }

    fn rerank_request() -> RerankRequest {
        RerankRequest {
            query: "q".to_string(),
            documents: vec!["d1".to_string(), "d2".to_string()],
            model: "test-model".to_string(),
            top_k: Some(1),
            return_documents: false,
            rid: None,
            user: None,
        }
    }

    fn rerank_worker_body() -> Vec<u8> {
        serde_json::to_vec(&vec![
            RerankResult {
                score: 0.9,
                document: Some("d1".to_string()),
                index: 0,
                meta_info: None,
            },
            RerankResult {
                score: 0.5,
                document: Some("d2".to_string()),
                index: 1,
                meta_info: None,
            },
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn build_rerank_response_accepts_body_at_limit() {
        let req = rerank_request();
        let body = rerank_worker_body();
        let limit = body.len();
        let upstream = Response::new(Body::from(body));

        let response =
            Router::build_rerank_response(RerankResponseSpec::from(&req), None, upstream, limit)
                .await;

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let rerank: RerankResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(rerank.model, "test-model");
        assert_eq!(rerank.results.len(), 1);
        assert_eq!(rerank.results[0].document, None);
    }

    #[tokio::test]
    async fn build_rerank_response_caps_oversized_body_with_502() {
        let req = rerank_request();
        let body = rerank_worker_body();
        let limit = body.len() - 1;
        let upstream = Response::new(Body::from(body));

        let response =
            Router::build_rerank_response(RerankResponseSpec::from(&req), None, upstream, limit)
                .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            extract_error_code_from_response(&response),
            "upstream_response_too_large"
        );
    }

    fn body_chunks(chunks: &[&'static [u8]]) -> Vec<Result<Bytes, String>> {
        chunks.iter().map(|c| Ok(Bytes::from_static(c))).collect()
    }

    #[tokio::test]
    async fn read_worker_body_capped_accepts_body_at_limit() {
        let chunks = body_chunks(&[b"abc", b"def", b"gh"]);

        let body = Router::read_worker_body_capped(stream::iter(chunks), 8)
            .await
            .unwrap();

        assert_eq!(body.as_ref(), b"abcdefgh");
    }

    #[tokio::test]
    async fn read_worker_body_capped_rejects_oversized_body_with_502() {
        let chunks = body_chunks(&[b"abc", b"def", b"gh"]);

        let response = Router::read_worker_body_capped(stream::iter(chunks), 7)
            .await
            .unwrap_err();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            extract_error_code_from_response(&response),
            "upstream_response_too_large"
        );
    }

    #[tokio::test]
    async fn read_worker_body_capped_maps_read_failure_to_500() {
        let chunks = vec![Ok(Bytes::from_static(b"abc")), Err("boom".to_string())];

        let response = Router::read_worker_body_capped(stream::iter(chunks), 8)
            .await
            .unwrap_err();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            extract_error_code_from_response(&response),
            "read_response_body_failed"
        );
    }

    fn least_load_policy() -> PolicyConfig {
        PolicyConfig::LeastLoad {
            load_check_interval_secs: 10,
            kv_pressure_weight: 0.15,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
            max_waiting_requests: 0,
        }
    }

    fn cache_aware_policy() -> PolicyConfig {
        PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        }
    }

    fn streaming_router(
        policy: PolicyConfig,
        max_payload_size: usize,
        workers: Vec<crate::worker::BasicWorker>,
    ) -> Router {
        streaming_router_with_registry(
            Arc::new(PolicyRegistry::new(policy)),
            max_payload_size,
            workers,
        )
    }

    fn streaming_router_with_key_override(
        policy: PolicyConfig,
        max_payload_size: usize,
        workers: Vec<crate::worker::BasicWorker>,
    ) -> Router {
        streaming_router_with_registry(
            Arc::new(PolicyRegistry::with_override(
                policy,
                RoutingKeyOverrideConfig {
                    enabled: true,
                    ..Default::default()
                },
            )),
            max_payload_size,
            workers,
        )
    }

    fn streaming_router_with_registry(
        policy_registry: Arc<PolicyRegistry>,
        max_payload_size: usize,
        workers: Vec<crate::worker::BasicWorker>,
    ) -> Router {
        let worker_registry = Arc::new(WorkerRegistry::new());
        for worker in workers {
            worker_registry.register_or_replace(Arc::new(worker));
        }
        Router {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            max_payload_size,
            stream_stall_timeout: Some(Duration::from_secs(60)),
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        }
    }

    fn plain_worker(url: &str) -> crate::worker::BasicWorker {
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build()
    }

    /// Like `plain_worker` but attaches a `pool=<pool>` label so the
    /// cache_aware_length policy can split workers into long/short pools.
    fn labeled_worker(url: &str, pool: Option<&str>) -> crate::worker::BasicWorker {
        let mut builder = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check());
        if let Some(p) = pool {
            builder = builder.label("pool", p);
        }
        builder.build()
    }

    fn cache_aware_length_policy() -> PolicyConfig {
        PolicyConfig::CacheAwareLength {
            cache_threshold: 0.3,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            chars_per_token: 4,
            long_prefill_threshold: 100_000,
            long_pool_max_load: 2,
            short_pool_max_load: 2,
        }
    }

    type CapturedUpstreamRequest = Arc<tokio::sync::Mutex<Option<(HeaderMap, Bytes)>>>;

    /// Loopback engine stub: captures the forwarded `/generate` request and
    /// answers with the given content type and body.
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    async fn spawn_capture_stub(
        content_type: &'static str,
        response_body: &'static str,
    ) -> (String, CapturedUpstreamRequest) {
        let captured: CapturedUpstreamRequest = Arc::new(tokio::sync::Mutex::new(None));
        let sink = Arc::clone(&captured);
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(move |headers: HeaderMap, body: Bytes| {
                let sink = Arc::clone(&sink);
                async move {
                    *sink.lock().await = Some((headers, body));
                    ([(CONTENT_TYPE, content_type)], response_body)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), captured)
    }

    fn streamed_request(chunks: &[&'static [u8]]) -> Request<Body> {
        let chunks: Vec<Result<Bytes, std::io::Error>> =
            chunks.iter().map(|c| Ok(Bytes::from_static(c))).collect();
        Request::builder()
            .method(Method::POST)
            .uri("/generate")
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .body(Body::from_stream(stream::iter(chunks)))
            .unwrap()
    }

    fn request_with_content_length(len: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder().method(Method::POST).uri("/generate");
        if let Some(len) = len {
            builder = builder.header(CONTENT_LENGTH, len);
        }
        builder.body(Body::empty()).unwrap()
    }

    #[test]
    fn stream_threshold_requires_flag_and_larger_content_length() {
        assert_eq!(RouterConfig::default().stream_request_bodies_over, 0);
        assert!(!exceeds_stream_threshold(
            0,
            &request_with_content_length(Some("10000"))
        ));

        assert!(!exceeds_stream_threshold(
            1024,
            &request_with_content_length(None)
        ));
        assert!(!exceeds_stream_threshold(
            1024,
            &request_with_content_length(Some("1024"))
        ));
        assert!(!exceeds_stream_threshold(
            1024,
            &request_with_content_length(Some("not-a-number"))
        ));
        assert!(exceeds_stream_threshold(
            1024,
            &request_with_content_length(Some("1025"))
        ));
    }

    #[tokio::test]
    async fn streamed_request_forwards_chunked_and_relays_response() {
        let (url, captured) = spawn_capture_stub("application/json", r#"{"text":"ok"}"#).await;
        let router = streaming_router(least_load_policy(), 1024 * 1024, vec![plain_worker(&url)]);

        let response = router
            .route_streaming_request(
                streamed_request(&[b"{\"text\":\"", b"hello\"}"]),
                "/generate",
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get("x-smg-routed-worker-id").is_some());
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), br#"{"text":"ok"}"#);

        let (headers, body) = captured.lock().await.take().unwrap();
        assert_eq!(body.as_ref(), b"{\"text\":\"hello\"}");
        assert!(
            headers.get(CONTENT_LENGTH).is_none(),
            "streamed forward must not carry a Content-Length"
        );
        assert_eq!(
            headers
                .get(http::header::TRANSFER_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("chunked")
        );
        assert_eq!(
            headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[tokio::test]
    async fn streamed_sse_response_relays_as_event_stream() {
        let (url, _captured) = spawn_capture_stub("text/event-stream", "data: hi\n\n").await;
        let router = streaming_router(least_load_policy(), 1024 * 1024, vec![plain_worker(&url)]);

        let response = router
            .route_streaming_request(streamed_request(&[b"{\"stream\":true}"]), "/generate")
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), b"data: hi\n\n");
    }

    #[tokio::test]
    async fn oversized_streamed_body_yields_413() {
        let (url, _captured) = spawn_capture_stub("application/json", "{}").await;
        let router = streaming_router(least_load_policy(), 8, vec![plain_worker(&url)]);

        let response = router
            .route_streaming_request(streamed_request(&[b"12345678", b"9"]), "/generate")
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "request_body_too_large"
        );
    }

    /// A stalled uploader must abort with 408 and leave the worker's circuit
    /// breaker untouched: 408 is a retryable status, so recording it as a
    /// worker outcome would trip a threshold-1 breaker here.
    #[tokio::test(start_paused = true)]
    async fn stalled_streamed_body_yields_408_without_breaker_sample() {
        let (url, _captured) = spawn_capture_stub("application/json", "{}").await;
        let worker = BasicWorkerBuilder::new(&url)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .circuit_breaker_config(crate::worker::CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            })
            .build();
        let router = streaming_router(least_load_policy(), 1024 * 1024, vec![worker]);

        let stalled_body = Body::from_stream(
            stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"{\"text\":\""))])
                .chain(stream::pending()),
        );
        let req = Request::builder()
            .method(Method::POST)
            .uri("/generate")
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .body(stalled_body)
            .unwrap();

        let response = router
            .route_streaming_request(req, "/generate")
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(
            extract_error_code_from_response(&response),
            "request_body_stalled"
        );
        let worker = &router.worker_registry.get_all()[0];
        assert!(
            worker.circuit_breaker_can_execute(),
            "a client-caused stall must not record a breaker failure"
        );
    }

    /// A client abort mid-upload must map to the excluded 400 code and leave
    /// the worker's threshold-1 circuit breaker untouched.
    #[tokio::test]
    async fn aborted_streamed_body_yields_400_without_breaker_sample() {
        let (url, _captured) = spawn_capture_stub("application/json", "{}").await;
        let worker = BasicWorkerBuilder::new(&url)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .circuit_breaker_config(crate::worker::CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            })
            .build();
        let router = streaming_router(least_load_policy(), 1024 * 1024, vec![worker]);

        let aborted_body = Body::from_stream(stream::iter([
            Ok(Bytes::from_static(b"{\"text\":\"")),
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "client reset",
            )),
        ]));
        let req = Request::builder()
            .method(Method::POST)
            .uri("/generate")
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .body(aborted_body)
            .unwrap();

        let response = router
            .route_streaming_request(req, "/generate")
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            extract_error_code_from_response(&response),
            "request_body_aborted"
        );
        let worker = &router.worker_registry.get_all()[0];
        assert!(
            worker.circuit_breaker_can_execute(),
            "a client abort must not record a breaker failure"
        );
    }

    async fn assert_falls_back_with_body_intact(router: &Router) {
        let req = router
            .route_streaming_request(streamed_request(&[b"{\"text\":\"hello\"}"]), "/generate")
            .await
            .unwrap_err();

        let body = to_bytes(req.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), b"{\"text\":\"hello\"}");
    }

    #[tokio::test]
    async fn text_needing_policy_falls_back_to_buffered() {
        let router = streaming_router(
            cache_aware_policy(),
            1024 * 1024,
            vec![plain_worker("http://worker1:8080")],
        );
        assert_falls_back_with_body_intact(&router).await;
    }

    fn with_header(mut req: Request<Body>, name: &'static str, value: &str) -> Request<Body> {
        req.headers_mut().insert(name, value.parse().unwrap());
        req
    }

    async fn routed_worker_id(router: &Router, req: Request<Body>) -> String {
        let response = router
            .route_streaming_request(req, "/generate")
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response
            .headers()
            .get("x-smg-routed-worker-id")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn tokens_hint_streams_under_cache_aware_with_tree_affinity() {
        let (url_a, _cap_a) = spawn_capture_stub("application/json", "{}").await;
        let (url_b, _cap_b) = spawn_capture_stub("application/json", "{}").await;
        let router = streaming_router(
            cache_aware_policy(),
            1024 * 1024,
            vec![plain_worker(&url_a), plain_worker(&url_b)],
        );
        let workers = router.worker_registry.get_all();
        router
            .policy_registry
            .get_default_policy()
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .unwrap()
            .init_workers(&workers);

        // The token tree pages by 16, so shorter hints train no affinity.
        let hint: String = (1..=32u32)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let first = routed_worker_id(
            &router,
            with_header(
                streamed_request(&[b"{\"text\":\"hello\"}"]),
                "x-smg-routing-tokens",
                &hint,
            ),
        )
        .await;
        let second = routed_worker_id(
            &router,
            with_header(
                streamed_request(&[b"{\"text\":\"other\"}"]),
                "x-smg-routing-tokens",
                &hint,
            ),
        )
        .await;
        assert_eq!(first, second, "same hint must stick to the tree tenant");
    }

    #[tokio::test]
    async fn invalid_tokens_hint_keeps_text_needing_policy_buffered() {
        let router = streaming_router(
            cache_aware_policy(),
            1024 * 1024,
            vec![plain_worker("http://worker1:8080")],
        );
        let over_cap = vec!["7"; 513].join(",");
        for hint in ["1,,3", over_cap.as_str()] {
            let req = with_header(
                streamed_request(&[b"{\"text\":\"hello\"}"]),
                "x-smg-routing-tokens",
                hint,
            );
            let req = router
                .route_streaming_request(req, "/generate")
                .await
                .unwrap_err();
            let body = to_bytes(req.into_body(), usize::MAX).await.unwrap();
            assert_eq!(body.as_ref(), b"{\"text\":\"hello\"}");
        }
    }

    #[tokio::test]
    async fn key_hint_streams_under_cache_aware_only_with_override() {
        // Without the sticky override nothing consumes the key content-blind.
        let router = streaming_router(
            cache_aware_policy(),
            1024 * 1024,
            vec![plain_worker("http://worker1:8080")],
        );
        let req = with_header(
            streamed_request(&[b"{\"text\":\"hello\"}"]),
            "x-smg-routing-key",
            "media-1",
        );
        assert!(router
            .route_streaming_request(req, "/generate")
            .await
            .is_err());

        let (url_a, _cap_a) = spawn_capture_stub("application/json", "{}").await;
        let (url_b, _cap_b) = spawn_capture_stub("application/json", "{}").await;
        let router = streaming_router_with_key_override(
            cache_aware_policy(),
            1024 * 1024,
            vec![plain_worker(&url_a), plain_worker(&url_b)],
        );
        let first = routed_worker_id(
            &router,
            with_header(
                streamed_request(&[b"{\"text\":\"hello\"}"]),
                "x-smg-routing-key",
                "media-1",
            ),
        )
        .await;
        let second = routed_worker_id(
            &router,
            with_header(
                streamed_request(&[b"{\"text\":\"other\"}"]),
                "x-smg-routing-key",
                "media-1",
            ),
        )
        .await;
        assert_eq!(first, second, "keyed requests must stick to one worker");

        // Over-cap keys are ignored by the same extractor selection uses.
        let req = with_header(
            streamed_request(&[b"{\"text\":\"hello\"}"]),
            "x-smg-routing-key",
            &"k".repeat(129),
        );
        assert!(router
            .route_streaming_request(req, "/generate")
            .await
            .is_err());
    }

    /// With retries disabled the parsed request must be freed at dispatch:
    /// the upstream stub refuses to answer until the probe's only remaining
    /// holder is the test itself.
    #[tokio::test]
    async fn disabled_retries_release_parsed_request_before_upstream_responds() {
        let probe = Arc::new(());
        let (url, released) = spawn_release_gated_stub(Arc::downgrade(&probe)).await;
        let mut router =
            streaming_router(least_load_policy(), 1024 * 1024, vec![plain_worker(&url)]);
        router.retry_config = RetryConfig {
            max_retries: 1,
            ..Default::default()
        };

        let req = DropProbeRequest {
            text: "hello".to_string(),
            _probe: Arc::clone(&probe),
        };
        let response = router
            .route_typed_request(None, req, "/generate", crate::worker::UNKNOWN_MODEL_ID)
            .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            released.load(AtomicOrdering::SeqCst),
            "the parsed request must be freed before the upstream answers"
        );
    }

    /// With retries enabled the request must survive for replay: a 503 on the
    /// first attempt is retried with an identical body.
    #[tokio::test]
    async fn enabled_retries_replay_an_identical_body() {
        use tokio::sync::Mutex;

        let bodies: Arc<Mutex<Vec<Bytes>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&bodies);
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_clone = Arc::clone(&hits);
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(move |body: Bytes| {
                let sink = Arc::clone(&sink);
                let hits = Arc::clone(&hits_clone);
                async move {
                    sink.lock().await.push(body);
                    if hits.fetch_add(1, AtomicOrdering::SeqCst) == 0 {
                        (StatusCode::SERVICE_UNAVAILABLE, "busy").into_response()
                    } else {
                        (StatusCode::OK, "{}").into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let mut router = streaming_router(
            least_load_policy(),
            1024 * 1024,
            vec![plain_worker(&format!("http://{addr}"))],
        );
        router.retry_config = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            ..Default::default()
        };

        let req = DropProbeRequest {
            text: "replay me".to_string(),
            _probe: Arc::new(()),
        };
        let response = router
            .route_typed_request(None, req, "/generate", crate::worker::UNKNOWN_MODEL_ID)
            .await;

        assert_eq!(response.status(), StatusCode::OK);
        let bodies = bodies.lock().await;
        assert_eq!(bodies.len(), 2, "503 then 200 must mean two attempts");
        assert_eq!(bodies[0], bodies[1], "the retry must replay the same body");
    }

    #[tokio::test]
    async fn mutating_worker_falls_back_to_buffered() {
        let worker = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .dp_config(3, 8)
            .build();
        let router = streaming_router(least_load_policy(), 1024 * 1024, vec![worker]);
        assert_falls_back_with_body_intact(&router).await;
    }

    #[tokio::test]
    async fn missing_worker_falls_back_to_buffered() {
        let router = streaming_router(least_load_policy(), 1024 * 1024, vec![]);
        assert_falls_back_with_body_intact(&router).await;
    }

    // ===== cache_aware_length E2E: all 16 decision-table scenarios =====
    //
    // Each test drives the real routing decision point
    // (`select_worker_for_model`), which builds the exact `SelectWorkerInfo`
    // (request_text + headers) the policy receives in production — the same
    // path the buffered HTTP request takes. Workers carry `pool=long` labels;
    // pre-set loads use WorkerLoadGuard + mem::forget (same as the unit tests).

    /// Pin a worker's in-flight load at `load` for the lifetime of the test by
    /// leaking the RAII guard (the process tears down on test exit). Accepts
    /// the registered `Arc<dyn Worker>` directly — no downcast needed since
    /// `WorkerLoadGuard::new` takes `Arc<dyn Worker>`.
    fn pin_load(worker: &Arc<dyn Worker>, load: usize) {
        for _ in 0..load {
            std::mem::forget(WorkerLoadGuard::new(Arc::clone(worker), None));
        }
    }

    /// Spawn a two-pool router: short workers (no label) + long workers
    /// (`pool=long`), with cache_aware_length as the default policy and trees
    /// seeded. Returns the router so tests can assert routed worker.
    async fn length_router(short_urls: &[&str], long_urls: &[&str]) -> Router {
        let mut workers: Vec<crate::worker::BasicWorker> = Vec::new();
        for u in short_urls {
            workers.push(labeled_worker(u, None));
        }
        for u in long_urls {
            workers.push(labeled_worker(u, Some("long")));
        }
        let router = streaming_router(cache_aware_length_policy(), 1024 * 1024, workers);
        let live = router.worker_registry.get_all();
        router
            .policy_registry
            .get_default_policy()
            .as_any()
            .downcast_ref::<CacheAwareLengthPolicy>()
            .unwrap()
            .init_workers(&live);
        router
    }

    /// Route `prompt` through the real `select_worker_for_model` (the exact
    /// path a buffered HTTP request takes), returning the selected worker's
    /// URL — the value the router writes to `x-smg-routed-worker-id`.
    /// Returns `None` when the fleet rejects (503). Panics if the selection
    /// unexpectedly fails (use `route_or_none` for the reject case).
    fn route_to_url(router: &Router, prompt: &str, headers: Option<&HeaderMap>) -> String {
        router
            .select_worker_for_model(
                crate::worker::UNKNOWN_MODEL_ID,
                Some(prompt),
                None,
                headers,
                None,
            )
            .map(|w| w.url().to_string())
            .expect("selection should succeed with healthy workers")
    }

    /// Like `route_to_url` but returns `None` when the fleet rejects (503).
    fn route_or_none(router: &Router, prompt: &str, headers: Option<&HeaderMap>) -> Option<String> {
        router
            .select_worker_for_model(
                crate::worker::UNKNOWN_MODEL_ID,
                Some(prompt),
                None,
                headers,
                None,
            )
            .map(|w| w.url().to_string())
    }

    /// Build a HeaderMap with `X-Prompt-Tokens: <tokens>`.
    fn tokens_header(tokens: usize) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            "x-prompt-tokens",
            http::HeaderValue::from_str(&tokens.to_string()).unwrap(),
        );
        h
    }

    /// Find a live worker by URL in the router's registry and pin its load.
    fn pin_worker(router: &Router, url: &str, load: usize) {
        let worker = router
            .worker_registry
            .get_all()
            .iter()
            .find(|w| w.url() == url)
            .cloned()
            .unwrap();
        pin_load(&worker, load);
    }

    /// Mark a worker unhealthy by URL.
    fn mark_unhealthy(router: &Router, url: &str) {
        router
            .worker_registry
            .get_all()
            .iter()
            .find(|w| w.url() == url)
            .cloned()
            .unwrap()
            .set_status(openai_protocol::worker::WorkerStatus::NotReady);
    }

    // --- Step 1: health filter ---

    #[tokio::test]
    async fn cal_step1_all_unhealthy_returns_503() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        mark_unhealthy(&router, &url_s);
        mark_unhealthy(&router, &url_l);
        let h = tokens_header(200_000);
        let routed = route_or_none(&router, "hello", Some(&h));
        assert!(
            routed.is_none(),
            "all unhealthy → 503 (None), got {routed:?}"
        );
    }

    // --- Step 2: global imbalance ---

    #[tokio::test]
    async fn cal_step2_global_imbalance_picks_min_load() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        // Pin the long-pool worker high so the fleet is imbalanced (100 vs 0).
        pin_worker(&router, &url_l, 100);
        let h = tokens_header(200_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_s, "imbalanced fleet → healthy min-load worker");
    }

    // --- Step 3: cache hit ---

    #[tokio::test]
    async fn cal_step3_cache_hit_sticks_regardless_of_pool() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let prompt = "shared long prompt prefix that builds cache affinity";

        // Seed the cache with a long-request header so the first request goes
        // to the long-pool worker (url_l), not the default short-pool routing.
        let h = tokens_header(200_000);
        let first = route_to_url(&router, prompt, Some(&h));
        assert_eq!(first, url_l, "header-classified long request → long pool");

        // Same prompt without the header: cache lookup (Step 3) takes
        // precedence over pool selection (Step 4), so it stays on url_l.
        let second = route_to_url(&router, prompt, None);
        assert_eq!(second, url_l, "cache hit overrides short-pool routing");
    }

    #[tokio::test]
    async fn cal_step3_no_tree_random_healthy_does_not_panic() {
        // Do NOT call init_workers → no tree → random healthy fallback.
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = streaming_router(
            cache_aware_length_policy(),
            1024 * 1024,
            vec![
                labeled_worker(&url_s, None),
                labeled_worker(&url_l, Some("long")),
            ],
        );
        let routed = route_to_url(&router, "novel", None);
        assert!(
            routed == url_s || routed == url_l,
            "random fallback picks a healthy worker: {routed}"
        );
    }

    // --- Step 4 long request (uncached ≥ 100K): 4 paths ---

    #[tokio::test]
    async fn cal_step4_long_uses_long_pool_when_free() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let h = tokens_header(200_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_l, "long request, free long pool → long pool");
    }

    #[tokio::test]
    async fn cal_step4_long_overflows_to_idle_short() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        pin_worker(&router, &url_l, 2); // long pool full (load = max)
        let h = tokens_header(200_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_s, "full long pool → idle (load 0) short worker");
    }

    #[tokio::test]
    async fn cal_step4_long_queues_on_long_when_short_busy() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        pin_worker(&router, &url_l, 2); // long full
        pin_worker(&router, &url_s, 1); // short busy (load > 0, not idle)
        let h = tokens_header(200_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_l, "long full + short busy → queue on long pool");
    }

    // Step 4: long pool all unhealthy + short pool all load>0 → all-healthy min-load.
    #[tokio::test]
    async fn cal_step4_long_unhealthy_short_busy_all_healthy_min_load() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        mark_unhealthy(&router, &url_l); // long pool unhealthy
        pin_worker(&router, &url_s, 1); // short busy (load > 0)
        let h = tokens_header(200_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(
            routed, url_s,
            "long unhealthy + short busy → all-healthy min-load (short worker)"
        );
    }

    // --- Step 4 short request (uncached < 100K): 5 paths ---

    #[tokio::test]
    async fn cal_step4_short_uses_short_pool_when_free() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let h = tokens_header(1_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_s, "short request, free short pool → short pool");
    }

    #[tokio::test]
    async fn cal_step4_short_overflows_to_long_when_short_full() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        pin_worker(&router, &url_s, 2); // short pool full
        let h = tokens_header(1_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_l, "short pool full → overflow to long pool");
    }

    #[tokio::test]
    async fn cal_step4_short_falls_back_to_short_when_both_full() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        pin_worker(&router, &url_s, 2); // short full
        pin_worker(&router, &url_l, 2); // long full
        let h = tokens_header(1_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_s, "both full → queue on short pool min-load");
    }

    #[tokio::test]
    async fn cal_step4_short_uses_long_when_short_pool_empty() {
        // Only long-pool workers configured; short pool is empty.
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[], &[&url_l]).await;
        let h = tokens_header(1_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_l, "no short pool → long pool min-load");
    }

    #[tokio::test]
    async fn cal_step4_short_both_pools_empty_all_healthy_min_load() {
        // Only one healthy short worker (long pool empty); it is the sole
        // all-healthy min-load candidate.
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[]).await;
        let h = tokens_header(1_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(
            routed, url_s,
            "single healthy worker → all-healthy min-load"
        );
    }

    // --- Step 4: token source priority (header vs char estimate vs none) ---

    #[tokio::test]
    async fn cal_step4_char_estimate_fallback_when_no_header() {
        // No X-Prompt-Tokens header → char-level estimate. A 400-char novel
        // prompt → 100 tokens < 100K threshold → short request → short pool.
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let prompt = "a".repeat(400);
        let routed = route_to_url(&router, &prompt, None);
        assert_eq!(routed, url_s, "char estimate < threshold → short pool");
    }

    #[tokio::test]
    async fn cal_step4_uncached_unknown_all_healthy_min_load() {
        // Empty prompt, no header, no tokens → uncached not computable →
        // all-healthy min-load. Pin the long worker higher so the short worker
        // is the unique minimum-load candidate.
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        pin_worker(&router, &url_l, 10); // long worker at load 10
        let routed = route_to_url(&router, "", None);
        assert_eq!(
            routed, url_s,
            "uncached unknown → all-healthy min-load (short worker at load 0)"
        );
    }

    // --- Step 5: tree recording (cache hit after pool routing) ---

    #[tokio::test]
    async fn cal_step5_pool_routing_records_tree_for_future_hit() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let prompt = "novel pool-routed prompt that will be recorded";
        // First request: novel → pool split (short, since <100K by char estimate)
        let first = route_to_url(&router, prompt, None);
        // Second request: same prompt → now a cache hit → same worker.
        let second = route_to_url(&router, prompt, None);
        assert_eq!(first, second, "pool routing recorded tree → subsequent hit");
    }

    // --- Additional scenarios ---

    // Step 3: cache hit but matched worker unhealthy → clean stale + first healthy
    #[tokio::test]
    async fn cal_step3_hit_unhealthy_falls_back_to_first_healthy() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let prompt = "shared cache-building prompt for affinity test";
        // First request: seed the cache
        let first = route_to_url(&router, prompt, None);
        // Mark the selected worker unhealthy
        mark_unhealthy(&router, &first);
        // Second request: same prompt hits cache, but matched worker is
        // unhealthy → fall back to another healthy worker
        let second = route_to_url(&router, prompt, None);
        assert_ne!(
            second, first,
            "unhealthy matched worker must not be selected"
        );
    }

    // Step 4 ≥100K: long pool all unhealthy + short pool has load=0 worker → long→short overflow to idle worker
    #[tokio::test]
    async fn cal_step4_long_pool_unhealthy_overflows_to_idle_short() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        mark_unhealthy(&router, &url_l); // long pool all unhealthy
                                         // short pool worker idle (load 0)
        let h = tokens_header(200_000);
        let routed = route_to_url(&router, "novel", Some(&h));
        assert_eq!(routed, url_s, "long pool all unhealthy → idle short worker");
    }

    // Step 4 token source priority: header exact value overrides char estimate
    #[tokio::test]
    async fn cal_step4_header_overrides_char_estimate() {
        // Short prompt (char estimate → 1 token → short request → short pool),
        // but header says 200000 (→ long request → long pool).
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let h = tokens_header(200_000);
        let routed = route_to_url(&router, "short", Some(&h));
        assert_eq!(
            routed, url_l,
            "header (200K) must override char estimate (1 token) → long pool"
        );
    }

    // --- Full HTTP path via route_typed_request ---

    /// Construct a minimal CompletionRequest for routing tests.
    fn completion_request(prompt: &str) -> CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "",
            "prompt": prompt,
            "stream": false,
        }))
        .unwrap()
    }

    /// E2E via route_typed_request: a header-classified long request routes
    /// to the long-pool worker, and the response carries x-smg-routed-worker-id.
    #[tokio::test]
    async fn cal_http_long_request_routes_to_long_pool() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        let headers = tokens_header(200_000);
        let resp = router
            .route_typed_request(
                Some(&headers),
                completion_request("novel prompt"),
                "/generate",
                crate::worker::UNKNOWN_MODEL_ID,
            )
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let routed = resp
            .headers()
            .get("x-smg-routed-worker-id")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(routed, url_l, "header-classified long request → long pool");
    }

    /// E2E via route_typed_request: an all-unhealthy fleet returns 503.
    #[tokio::test]
    async fn cal_http_all_unhealthy_returns_503() {
        let (url_s, _cap_s) = spawn_capture_stub("application/json", "{}").await;
        let (url_l, _cap_l) = spawn_capture_stub("application/json", "{}").await;
        let router = length_router(&[&url_s], &[&url_l]).await;
        mark_unhealthy(&router, &url_s);
        mark_unhealthy(&router, &url_l);
        let resp = router
            .route_typed_request(
                None,
                completion_request("hello"),
                "/generate",
                crate::worker::UNKNOWN_MODEL_ID,
            )
            .await;
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "all unhealthy → 503"
        );
    }
}
