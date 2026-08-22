use std::{
    any::Any,
    error::Error as _,
    sync::{atomic::AtomicBool, Arc},
};

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    responses::ResponsesRequest,
};

use super::{
    chat::{self, ChatRouterContext},
    context::{ResponsesComponents, SharedComponents},
    health,
    provider::ProviderRegistry,
    responses::route::{self as responses_route, ResponsesRouterContext},
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::{metrics_labels, Metrics},
    routers::common::{
        header_utils::extract_auth_header,
        realtime::{
            rest::forward_realtime_rest, webrtc, webrtc::handle_realtime_webrtc,
            ws::handle_realtime_ws, RealtimeLabels, RealtimeRegistry,
        },
        worker_selection::{SelectWorkerRequest, WorkerSelector},
    },
    worker::{ProviderType, Worker, WorkerRegistry},
};

const WEBRTC_REQUEST_BODY_LIMIT: usize = 10 * 1024 * 1024;

/// Resolve the provider implementation for a given worker and model.
///
/// Checks (in order): worker's per-model provider, model name heuristic,
/// then falls back to the default provider.
pub(super) fn resolve_provider(
    registry: &ProviderRegistry,
    worker: &dyn Worker,
    model: &str,
) -> Arc<dyn super::provider::Provider> {
    if let Some(pt) = worker.provider_for_model(model) {
        return registry.get_arc(pt);
    }
    if let Some(pt) = ProviderType::from_model_name(model) {
        return registry.get_arc(&pt);
    }
    registry.default_provider_arc()
}

pub struct OpenAIRouter {
    worker_registry: Arc<WorkerRegistry>,
    provider_registry: ProviderRegistry,
    healthy: AtomicBool,
    shared_components: Arc<SharedComponents>,
    responses_components: Arc<ResponsesComponents>,
    retry_config: RetryConfig,
    realtime_registry: Arc<RealtimeRegistry>,
    context: Arc<AppContext>,
}

impl std::fmt::Debug for OpenAIRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let registry_stats = self.worker_registry.stats();
        f.debug_struct("OpenAIRouter")
            .field("registered_workers", &registry_stats.total_workers)
            .field("registered_models", &registry_stats.total_models)
            .field("healthy_workers", &registry_stats.healthy_workers)
            .field("healthy", &self.healthy)
            .finish()
    }
}

impl OpenAIRouter {
    #[expect(
        clippy::unused_async,
        reason = "async for API consistency with other router constructors"
    )]
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        let worker_registry = ctx.worker_registry.clone();
        let mcp_orchestrator = ctx
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "MCP manager not initialized in AppContext".to_string())?
            .clone();

        let shared_components = Arc::new(SharedComponents {
            client: ctx.client.clone(),
            router_config: Arc::new(ctx.router_config.clone()),
        });

        let responses_components = Arc::new(ResponsesComponents {
            shared: Arc::clone(&shared_components),
            mcp_orchestrator: mcp_orchestrator.clone(),
            mcp_format_registry: ctx.mcp_format_registry.clone(),
            response_storage: ctx.response_storage.clone(),
            conversation_storage: ctx.conversation_storage.clone(),
            conversation_item_storage: ctx.conversation_item_storage.clone(),
        });

        Ok(Self {
            worker_registry,
            provider_registry: ProviderRegistry::new(),
            healthy: AtomicBool::new(true),
            shared_components,
            responses_components,
            retry_config: ctx.router_config.effective_retry_config(),
            realtime_registry: ctx.realtime_registry.clone(),
            context: Arc::clone(ctx),
        })
    }

    async fn select_worker(
        &self,
        model_id: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Arc<dyn Worker>, Response> {
        WorkerSelector::new(&self.worker_registry, &self.shared_components.client)
            .select_worker(&SelectWorkerRequest {
                model_id,
                headers,
                provider: Some(ProviderType::OpenAI),
                ..Default::default()
            })
            .await
    }
}

#[async_trait::async_trait]
impl crate::routers::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        health::health_generate(&self.worker_registry)
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        health::get_server_info(&self.worker_registry)
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config = self.worker_registry.get_retry_config(model_id);
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        let deps = ChatRouterContext {
            worker_registry: &self.worker_registry,
            provider_registry: &self.provider_registry,
            shared_components: &self.shared_components,
            retry_config,
        };
        chat::route_chat(&deps, headers, tenant_meta, body, model_id).await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let deps = ResponsesRouterContext {
            worker_registry: &self.worker_registry,
            provider_registry: &self.provider_registry,
            responses_components: &self.responses_components,
        };
        responses_route::route_responses(&deps, headers, tenant_meta, &body, model_id).await
    }

    async fn route_realtime_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeSessionCreateRequest,
    ) -> Response {
        // TODO(Phase 3): Inject MCP tool definitions into body.tools
        let model = body.model.as_deref().unwrap_or_default();
        let worker = self.select_worker(model, headers).await;
        forward_realtime_rest(
            RealtimeLabels::OPENAI,
            &self.shared_components.client,
            worker,
            headers,
            body,
            model,
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
        // TODO(Phase 3): Inject MCP tool definitions into body.session.tools
        let model = body.session.model.as_deref().unwrap_or_default();
        let worker = self.select_worker(model, headers).await;
        forward_realtime_rest(
            RealtimeLabels::OPENAI,
            &self.shared_components.client,
            worker,
            headers,
            body,
            model,
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
        let model = body.model.as_deref().unwrap_or_default();
        let worker = self.select_worker(model, headers).await;
        forward_realtime_rest(
            RealtimeLabels::OPENAI,
            &self.shared_components.client,
            worker,
            headers,
            body,
            model,
            "/v1/realtime/transcription_sessions",
            metrics_labels::ENDPOINT_REALTIME_TRANSCRIPTION,
        )
        .await
    }

    async fn route_realtime_ws(&self, req: Request<Body>, model: &str) -> Response {
        let (parts, _body) = req.into_parts();

        Metrics::record_router_request(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_WEBSOCKET,
            model,
            metrics_labels::ENDPOINT_REALTIME,
            "false",
        );

        let auth_header = extract_auth_header(Some(&parts.headers), None);
        let worker = self.select_worker(model, Some(&parts.headers)).await;

        handle_realtime_ws(
            RealtimeLabels::OPENAI,
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
        let body = match axum::body::to_bytes(body, WEBRTC_REQUEST_BODY_LIMIT).await {
            Ok(b) => b,
            Err(e) => {
                if e.source()
                    .and_then(|s| s.downcast_ref::<http_body_util::LengthLimitError>())
                    .is_some()
                {
                    return StatusCode::PAYLOAD_TOO_LARGE.into_response();
                }
                return super::super::error::bad_request(
                    "invalid_body",
                    format!("Failed to read body: {e}"),
                );
            }
        };

        // Parse body once to extract model, SDP, and session config.
        // For multipart, the model comes from the session JSON body.
        // For application/sdp, the model comes from the query parameter.
        let parsed = match webrtc::parse_webrtc_request(&parts, &body, model).await {
            Ok(p) => p,
            Err(resp) => return resp,
        };

        Metrics::record_router_request(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_WEBRTC,
            &parsed.model,
            metrics_labels::ENDPOINT_REALTIME,
            "false",
        );

        let auth_header = extract_auth_header(Some(&parts.headers), None);
        let worker = self
            .select_worker(&parsed.model, Some(&parts.headers))
            .await;

        let bind_addr = self
            .context
            .webrtc_bind_addr
            .unwrap_or_else(|| std::net::Ipv4Addr::UNSPECIFIED.into());

        handle_realtime_webrtc(
            RealtimeLabels::OPENAI,
            parts.headers,
            parsed,
            worker,
            auth_header,
            self.shared_components.client.clone(),
            bind_addr,
            self.context.webrtc_stun_server.clone(),
            Arc::clone(&self.realtime_registry),
        )
        .await
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }
}
