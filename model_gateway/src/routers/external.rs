//! The gateway's side of the external-router contract: what a third-party
//! router gets to see of the gateway, and how one is mounted as a router.

use std::{any::Any, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{body::Body, extract::Request, http::HeaderMap, response::Response};
use openai_protocol::{
    chat::ChatCompletionRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    responses::ResponsesRequest,
    worker::ProviderType,
};
use smg_external_router::{
    worker::{ExternalWorker, LoadHold, SelectWorkerRequest, WorkerSource, WorkerStats},
    ExternalContext, ExternalRouter, ExternalRouterSpec, RetryConfig,
};

use super::{common::worker_selection::WorkerSelector, RouterTrait};
use crate::{
    app_context::AppContext,
    middleware::TenantRequestMeta,
    worker::{RuntimeType, Worker, WorkerLoadGuard, WorkerRegistry},
};

/// A gateway worker as an external router sees it.
pub struct GatewayWorker(Arc<dyn Worker>);

impl GatewayWorker {
    pub fn handle(worker: Arc<dyn Worker>) -> Arc<dyn ExternalWorker> {
        Arc::new(Self(worker))
    }
}

impl fmt::Debug for GatewayWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatewayWorker")
            .field("url", &self.0.url())
            .finish()
    }
}

impl ExternalWorker for GatewayWorker {
    fn url(&self) -> &str {
        self.0.url()
    }

    fn api_key(&self) -> Option<&String> {
        self.0.api_key()
    }

    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    fn is_healthy(&self) -> bool {
        self.0.is_healthy()
    }

    fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.0.provider_for_model(model_id)
    }

    fn record_outcome(&self, status_code: u16) {
        self.0.record_outcome(status_code);
    }

    fn http_client(&self) -> &reqwest::Client {
        self.0.http_client()
    }

    fn hold_load(&self, headers: Option<&HeaderMap>) -> LoadHold {
        Box::new(WorkerLoadGuard::new(Arc::clone(&self.0), headers))
    }
}

/// The worker registry as the external routers' worker source.
pub struct GatewayWorkers {
    registry: Arc<WorkerRegistry>,
}

impl GatewayWorkers {
    /// The registry as a shared worker source.
    pub fn source(registry: Arc<WorkerRegistry>) -> Arc<dyn WorkerSource> {
        Arc::new(Self { registry })
    }
}

impl fmt::Debug for GatewayWorkers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatewayWorkers")
            .field("workers", &self.registry.stats().total_workers)
            .finish()
    }
}

#[async_trait]
impl WorkerSource for GatewayWorkers {
    async fn select(
        &self,
        req: &SelectWorkerRequest<'_>,
    ) -> Result<Arc<dyn ExternalWorker>, Response> {
        WorkerSelector::new(&self.registry)
            .select_worker(req)
            .await
            .map(GatewayWorker::handle)
    }

    fn retry_config(&self, model_id: &str) -> Option<RetryConfig> {
        self.registry.get_retry_config(model_id)
    }

    fn stats(&self) -> WorkerStats {
        let stats = self.registry.stats();
        WorkerStats {
            total_workers: stats.total_workers,
            total_models: stats.total_models,
            healthy_workers: stats.healthy_workers,
            unhealthy_workers: stats.unhealthy_workers,
        }
    }

    fn external_workers(&self) -> Vec<Arc<dyn ExternalWorker>> {
        self.registry
            .get_all()
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == RuntimeType::External)
            .map(GatewayWorker::handle)
            .collect()
    }
}

/// Everything an external router may use, drawn from the app context.
pub fn external_context(ctx: &AppContext) -> ExternalContext {
    ExternalContext {
        client: ctx.client.clone(),
        request_timeout: Duration::from_secs(ctx.router_config.request_timeout_secs),
        retry: ctx.router_config.effective_retry_config(),
        workers: GatewayWorkers::source(ctx.worker_registry.clone()),
        mcp: ctx.mcp_orchestrator.clone(),
        mcp_formats: ctx.mcp_format_registry.clone(),
        responses: ctx.response_storage.clone(),
        conversations: ctx.conversation_storage.clone(),
        conversation_items: ctx.conversation_item_storage.clone(),
        realtime: ctx.realtime_registry.clone(),
        webrtc_bind_addr: ctx.webrtc_bind_addr,
        webrtc_stun_server: ctx.webrtc_stun_server.clone(),
    }
}

/// An external router mounted as a gateway router.
pub struct ExternalRouterAdapter {
    spec: ExternalRouterSpec,
    inner: Arc<dyn ExternalRouter>,
}

impl ExternalRouterAdapter {
    /// Build the router `spec` describes against this gateway.
    pub async fn mount(
        spec: ExternalRouterSpec,
        ctx: &AppContext,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let inner = (spec.build)(external_context(ctx)).await?;
        Ok(Box::new(Self { spec, inner }))
    }

    pub fn spec(&self) -> &ExternalRouterSpec {
        &self.spec
    }
}

impl fmt::Debug for ExternalRouterAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalRouterAdapter")
            .field("router_id", &self.spec.router_id)
            .field("inner", &self.inner)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for ExternalRouterAdapter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn router_type(&self) -> &'static str {
        self.inner.router_type()
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.inner.health_generate(req).await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.inner.get_server_info(req).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        self.inner
            .route_chat(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        self.inner
            .route_responses(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        self.inner
            .route_messages(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_interactions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: InteractionsRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.inner
            .route_interactions(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_realtime_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeSessionCreateRequest,
    ) -> Response {
        self.inner.route_realtime_session(headers, body).await
    }

    async fn route_realtime_client_secret(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        self.inner.route_realtime_client_secret(headers, body).await
    }

    async fn route_realtime_transcription_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        self.inner
            .route_realtime_transcription_session(headers, body)
            .await
    }

    async fn route_realtime_ws(&self, req: Request<Body>, model_id: &str) -> Response {
        self.inner.route_realtime_ws(req, model_id).await
    }

    async fn route_realtime_webrtc(&self, req: Request<Body>, model_id: &str) -> Response {
        self.inner.route_realtime_webrtc(req, model_id).await
    }
}
