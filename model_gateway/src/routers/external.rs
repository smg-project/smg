//! The gateway's side of the external-router contract: what a third-party
//! router gets to see of the gateway, and how one is mounted as a router.

use std::{any::Any, fmt, future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{body::Body, extract::Request, http::HeaderMap, response::Response};
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::GenerationRequest,
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

use super::{
    common::worker_selection::WorkerSelector,
    grpc::utils::{error_reason_for_response, error_type_for_response, route_to_endpoint},
    http::tenant_rate_limit::{self, TenantCharge},
    RouterTrait,
};
use crate::{
    app_context::AppContext,
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    rate_limit::RateLimitManager,
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
            .field("url", &self.0.redacted_url())
            .finish()
    }
}

impl ExternalWorker for GatewayWorker {
    fn url(&self) -> &str {
        self.0.url()
    }

    fn redacted_url(&self) -> &str {
        self.0.redacted_url()
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
    /// The tenant rate limiter, when configured: every route that charges a
    /// tenant reserves through it before the router dispatches
    /// ([`Self::route_reserved`]).
    rate_limit_manager: Option<Arc<RateLimitManager>>,
    /// The registry, for the canonical name of the model a route names
    /// ([`Self::route_reserved`]).
    worker_registry: Arc<WorkerRegistry>,
}

impl ExternalRouterAdapter {
    /// Build the router `spec` describes against this gateway.
    pub async fn mount(
        spec: ExternalRouterSpec,
        ctx: &AppContext,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let inner = (spec.build)(external_context(ctx)).await?;
        Ok(Box::new(Self {
            spec,
            inner,
            rate_limit_manager: ctx.rate_limit_manager.clone(),
            worker_registry: ctx.worker_registry.clone(),
        }))
    }

    pub fn spec(&self) -> &ExternalRouterSpec {
        &self.spec
    }

    /// Route `request` through the provider router under the tenant's
    /// reservation, as the gateway's own HTTP routers route theirs: reserved
    /// once before the dispatch, resolved once on the router's answer
    /// (`tenant_rate_limit`). A denied request never reaches the provider
    /// router, so its request and its refusal are counted here. The
    /// reservation is keyed by the canonical name of `model_id`, an alias
    /// resolved through the registry as the other routers resolve theirs;
    /// the count carries the name as the route gave it, like the provider
    /// router's own series.
    async fn route_reserved<T, F, Fut>(
        &self,
        tenant_meta: &TenantRequestMeta,
        request: T,
        route: &'static str,
        model_id: &str,
        dispatch: F,
    ) -> Response
    where
        T: TenantCharge + Send,
        F: FnOnce(T) -> Fut + Send,
        Fut: Future<Output = Response> + Send,
    {
        let is_stream = request.is_stream();
        // Resolve once, here, so the reservation is keyed by the canonical
        // model ID, as on the regular and the disaggregated routers: the
        // limiter's per-model rules match the name, and an alias (configured,
        // or a provider's discovered variant) is registered against the
        // served one. The provider router gets the name the route gave it,
        // as before, and so does a refusal's count, which sits next to that
        // router's own series.
        let requested_model = model_id;
        let canonical_model = self.worker_registry.resolve_model_alias(model_id);
        let model_id = canonical_model.as_deref().unwrap_or(model_id);
        let (reservation, denial) = tenant_rate_limit::reserve(
            tenant_rate_limit::estimate(
                self.rate_limit_manager.as_deref(),
                Some(tenant_meta),
                &request,
                None,
            ),
            model_id,
        )
        .await;
        let response = match denial {
            Some(denied) => {
                self.record_denied(requested_model, route, is_stream, &denied);
                denied
            }
            None => dispatch(request).await,
        };
        // A buffered answer is the provider's body, held whole already (the
        // router read it without a bound), so the re-read is unbounded.
        tenant_rate_limit::resolve(reservation.as_ref(), route, is_stream, response, usize::MAX)
            .await
    }

    /// Count a request the tenant limit refused ahead of the provider
    /// router, which counts the ones it dispatches: the request under the
    /// router's type (`router_type`, the name the router gives itself) with
    /// backend `external` and the route's endpoint, and the refusal under its
    /// class and reason. The OpenAI-compatible router counts its dispatched
    /// traffic under that same type; the Anthropic router counts its own
    /// under the HTTP router's type and the Gemini router counts none, so a
    /// refusal of theirs is the first series under their own name. `model_id`
    /// is the name the route gave, as those routers label their own series
    /// (an alias stays an alias), under the gateway's label bound: a name the
    /// registry does not know counts as `other`, where the provider router's
    /// own bound keeps the name.
    fn record_denied(&self, model_id: &str, route: &str, is_stream: bool, denied: &Response) {
        let router_type = self.inner.router_type();
        let endpoint = route_to_endpoint(route);
        let model_label = self.worker_registry.metric_model_label(model_id);
        Metrics::record_router_request(
            router_type,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_label,
            endpoint,
            bool_to_static_str(is_stream),
        );
        Metrics::record_router_error(
            router_type,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_label,
            endpoint,
            error_type_for_response(denied),
            error_reason_for_response(denied),
        );
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
        self.route_reserved(
            tenant_meta,
            body,
            "/v1/chat/completions",
            model_id,
            |body| self.inner.route_chat(headers, tenant_meta, body, model_id),
        )
        .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        self.route_reserved(tenant_meta, body, "/v1/responses", model_id, |body| {
            self.inner
                .route_responses(headers, tenant_meta, body, model_id)
        })
        .await
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        // `rid` is this gateway's extension, forwarded to an engine for log
        // correlation; a provider refuses an unknown top-level field. The
        // OpenAI-compatible providers drop the gateway-only fields from
        // their JSON payload (`strip_sglang_fields`); a Messages provider
        // posts the typed request, so the field is cleared here.
        let mut body = body;
        body.rid = None;
        self.route_reserved(tenant_meta, body, "/v1/messages", model_id, |body| {
            self.inner
                .route_messages(headers, tenant_meta, body, model_id)
        })
        .await
    }

    async fn route_interactions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: InteractionsRequest,
        model_id: Option<&str>,
    ) -> Response {
        // The charge is keyed by the model the route named, else the one the
        // request names.
        let model = model_id
            .or_else(|| body.get_model())
            .unwrap_or_default()
            .to_owned();
        self.route_reserved(tenant_meta, body, "/v1/interactions", &model, |body| {
            self.inner
                .route_interactions(headers, tenant_meta, body, model_id)
        })
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::{http::StatusCode, response::IntoResponse};
    use serde_json::{json, Value};
    use smg_external_router::spec_for_provider;

    use super::*;
    use crate::middleware::{RouteRequestMeta, TenantKey};

    /// A provider router of the kind the adapter mounts, keeping the
    /// Messages request it was handed as the JSON it would post: the trait's
    /// two required methods, every other route its default.
    #[derive(Debug, Default)]
    struct RecordingProvider {
        posted: Mutex<Option<Value>>,
    }

    #[async_trait]
    impl ExternalRouter for RecordingProvider {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn router_type(&self) -> &'static str {
            "stub"
        }

        async fn route_messages(
            &self,
            _headers: Option<&HeaderMap>,
            _tenant_meta: &TenantRequestMeta,
            body: CreateMessageRequest,
            _model_id: &str,
        ) -> Response {
            let posted = serde_json::to_value(&body).expect("a request serializes");
            *self.posted.lock().expect("the recorder") = Some(posted);
            StatusCode::OK.into_response()
        }
    }

    /// A Messages request reaches its provider without the gateway's `rid`:
    /// a thread turn carries the id the gateway minted, a plain request the
    /// client's own, and a provider refuses the unknown field.
    #[tokio::test]
    async fn a_provider_gets_no_rid() {
        let provider = Arc::new(RecordingProvider::default());
        let adapter = ExternalRouterAdapter {
            spec: spec_for_provider(None).expect("the OpenAI-compatible spec"),
            inner: provider.clone(),
            rate_limit_manager: None,
            worker_registry: Arc::new(WorkerRegistry::new()),
        };
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 1,
            "rid": "msg_minted",
            "messages": [{"role": "user", "content": "q"}],
        }))
        .expect("a request");
        let tenant_meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
        let response = adapter
            .route_messages(None, &tenant_meta, request, "m")
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let posted = provider
            .posted
            .lock()
            .expect("the recorder")
            .take()
            .expect("a posted request");
        assert!(posted.get("rid").is_none(), "{posted}");
        assert_eq!(posted["model"], "m");
        assert_eq!(posted["messages"][0]["content"], "q");
    }
}
