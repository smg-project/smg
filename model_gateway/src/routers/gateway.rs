//! The gateway: one router-shaped front that owns the family routers and
//! hands each request to the one whose workers serve the model.
//!
//! In single-router mode every request goes to the one router the config
//! built. In IGW mode the choice is made per request from the model's
//! routing snapshot: an external worker sends the request to the provider
//! router that takes it; otherwise the family routers are weighted by the
//! size of the pools they would select from, so traffic can migrate between
//! HTTP and gRPC and between regular and disaggregated fleets gradually.

use std::{future::Future, sync::Arc};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::RerankRequest,
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
    UNKNOWN_MODEL_ID,
};
use smg_external_router::spec_for_provider;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    middleware::TenantRequestMeta,
    routers::{
        common::body_policy::REASON_MODEL_SELECTION,
        error as route_error,
        factory::{router_ids, RouterId},
        BodyPolicy, RouterFactory, RouterTrait,
    },
    server::ServerConfig,
    worker::{ConnectionMode, ProviderType, RoutingPool, WorkerRegistry},
};

pub struct Gateway {
    worker_registry: Arc<WorkerRegistry>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    enable_igw: bool,
}

/// The answer when no router serves a request.
const NO_ROUTER: (StatusCode, &str) = (
    StatusCode::NOT_FOUND,
    "No router available for this request",
);

impl Gateway {
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            enable_igw: false,
        }
    }

    fn try_register(
        &self,
        id: RouterId,
        label: &str,
        result: Result<Box<dyn RouterTrait>, String>,
    ) {
        match result {
            Ok(router) => {
                info!("Created {label} router");
                self.register_router(id, Arc::from(router));
            }
            Err(e) => {
                warn!("Failed to create {label} router: {e}");
            }
        }
    }

    pub async fn from_config(
        config: &ServerConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        let mut gateway = Self::new(app_context.worker_registry.clone());
        gateway.enable_igw = config.router_config.enable_igw;
        let gateway = Arc::new(gateway);

        if config.router_config.enable_igw {
            info!("Initializing the gateway in multi-router mode (IGW)");
            let routers =
                RouterFactory::create_igw_routers(&config.router_config.policy, app_context).await;
            for (id, label, result) in routers {
                gateway.try_register(id, label, result);
            }
            info!(
                "Gateway initialized with {} routers for multi-router mode",
                gateway.router_count(),
            );
        } else {
            info!("Initializing the gateway in single-router mode");
            let single_router = Arc::from(RouterFactory::create_router(app_context).await?);
            let router_id = Self::determine_router_id(
                &config.router_config.mode,
                config.router_config.connection_mode,
            );
            info!("Created single router with ID: {}", router_id.as_str());
            gateway.register_router(router_id.clone(), single_router);
            gateway.set_default_router(router_id);
        }

        if gateway.router_count() == 0 {
            return Err("No routers could be initialized".to_string());
        }
        Ok(gateway)
    }

    pub fn determine_router_id(
        routing_mode: &RoutingMode,
        connection_mode: ConnectionMode,
    ) -> RouterId {
        match (connection_mode, routing_mode) {
            (ConnectionMode::Http, RoutingMode::Regular { .. }) => router_ids::HTTP_REGULAR,
            (ConnectionMode::Http, RoutingMode::PrefillDecode { .. }) => router_ids::HTTP_PD,
            (ConnectionMode::Http, RoutingMode::OpenAI { .. }) => router_ids::HTTP_OPENAI,
            (ConnectionMode::Http, RoutingMode::Anthropic { .. }) => router_ids::HTTP_ANTHROPIC,
            (ConnectionMode::Grpc | ConnectionMode::Zmq, RoutingMode::Regular { .. }) => {
                router_ids::GRPC_REGULAR
            }
            (ConnectionMode::Grpc | ConnectionMode::Zmq, RoutingMode::PrefillDecode { .. }) => {
                router_ids::GRPC_PD
            }
            // EPD only runs on gRPC; the HTTP arm never reaches a real router
            // (the factory errors), but the match must stay exhaustive.
            (_, RoutingMode::EncodePrefillDecode { .. }) => router_ids::GRPC_EPD,
            (ConnectionMode::Http, RoutingMode::Gemini { .. }) => router_ids::HTTP_GEMINI,
            (ConnectionMode::Grpc | ConnectionMode::Zmq, RoutingMode::OpenAI { .. }) => {
                router_ids::GRPC_REGULAR
            }
            (ConnectionMode::Grpc | ConnectionMode::Zmq, RoutingMode::Anthropic { .. }) => {
                router_ids::GRPC_REGULAR
            }
            (ConnectionMode::Grpc | ConnectionMode::Zmq, RoutingMode::Gemini { .. }) => {
                router_ids::GRPC_REGULAR
            }
        }
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// Selects a router by weighting available router types by their worker counts.
    /// PD routers only receive weight when both prefill and decode workers are
    /// present on the same protocol; EPD requires encode, prefill, and decode
    /// workers over gRPC. Incomplete role sets contribute 0.
    ///
    /// Weighting the router selection lets operators gradually migrate traffic between
    /// HTTP / gRPC and regular / prefill-decode disaggregation workers.
    fn pick_router_by_weights(
        &self,
        grpc_epd: usize,
        grpc_pd: usize,
        http_pd: usize,
        grpc_regular: usize,
        http_regular: usize,
    ) -> Option<Arc<dyn RouterTrait>> {
        let options: [(usize, &RouterId); 5] = [
            (grpc_epd, &router_ids::GRPC_EPD),
            (grpc_pd, &router_ids::GRPC_PD),
            (http_pd, &router_ids::HTTP_PD),
            (grpc_regular, &router_ids::GRPC_REGULAR),
            (http_regular, &router_ids::HTTP_REGULAR),
        ];

        let total: usize = options
            .iter()
            .filter(|(weight, router_id)| *weight > 0 && self.routers.contains_key(*router_id))
            .map(|(weight, _)| *weight)
            .sum();
        if total == 0 {
            return None;
        }

        let pick = ((rand::random::<f64>() * total as f64) as usize).min(total - 1);
        let mut cum = 0usize;
        for (weight, router_id) in &options {
            if *weight == 0 || !self.routers.contains_key(*router_id) {
                continue;
            }
            cum += weight;
            if pick < cum {
                return self.routers.get(*router_id).map(|r| r.clone());
            }
        }
        None
    }

    /// The mounted external router that takes workers of `provider`, resolved
    /// through the crate's identity table so dispatch and admission agree.
    fn external_router_for(&self, provider: Option<&ProviderType>) -> Option<Arc<dyn RouterTrait>> {
        let spec = spec_for_provider(provider)?;
        self.routers
            .get(&RouterId::new(spec.router_id))
            .map(|router| Arc::clone(router.value()))
    }

    /// The router for `model_id` (the whole fleet when `None`), read from the
    /// model's routing snapshot: the pools are the same projections the
    /// routers select from, cached across requests, so this costs a few
    /// pool-length reads rather than a walk over every worker.
    ///
    /// An external worker sends the request to its provider's router. A
    /// disaggregated family is weighted only when both of its legs exist on
    /// the wire the family selects over, which is how a leg on a transport
    /// the family cannot use (ZMQ prefill or decode) no longer counts.
    fn select_router_for_model(&self, model_id: Option<&str>) -> Option<Arc<dyn RouterTrait>> {
        let snapshot = self
            .worker_registry
            .get_routing_snapshot(model_id.unwrap_or(UNKNOWN_MODEL_ID));
        if let Some(model) = model_id {
            if let Some(external) = snapshot.pool(RoutingPool::External).first() {
                return self.external_router_for(external.provider_for_model(model));
            }
        }
        let size = |pool: RoutingPool| snapshot.pool(pool).len();
        let grpc_encode = size(RoutingPool::GrpcEncode);
        let grpc_prefill = size(RoutingPool::GrpcPrefill);
        let grpc_decode = size(RoutingPool::GrpcDecode);
        let http_prefill = size(RoutingPool::HttpPrefill);
        let http_decode = size(RoutingPool::HttpDecode);
        let grpc_regular = size(RoutingPool::GrpcPipelineRegular);
        let http_regular = size(RoutingPool::HttpRegular);

        let grpc_epd_ready = grpc_encode > 0
            && grpc_prefill > 0
            && grpc_decode > 0
            && self.routers.contains_key(&router_ids::GRPC_EPD);
        let grpc_epd = if grpc_epd_ready {
            grpc_encode + grpc_prefill + grpc_decode
        } else {
            0
        };
        // A disaggregated family needs both legs before it takes any weight.
        let grpc_pd = if !grpc_epd_ready && grpc_prefill > 0 && grpc_decode > 0 {
            grpc_prefill + grpc_decode
        } else {
            0
        };
        let http_pd = if http_prefill > 0 && http_decode > 0 {
            http_prefill + http_decode
        } else {
            0
        };
        self.pick_router_by_weights(grpc_epd, grpc_pd, http_pd, grpc_regular, http_regular)
    }

    /// Whether an external worker serves `model` through a provider router
    /// this build does not carry. Such a request must fail rather than fall
    /// through to a self-hosted router that would proxy it untranslated.
    fn external_router_missing(&self, model: &str) -> bool {
        self.worker_registry
            .get_routing_snapshot(model)
            .pool(RoutingPool::External)
            .iter()
            .any(|w| {
                self.external_router_for(w.provider_for_model(model))
                    .is_none()
            })
    }

    fn requires_explicit_generate_model(&self, model_id: &str) -> bool {
        self.enable_igw && (model_id.trim().is_empty() || model_id == UNKNOWN_MODEL_ID)
    }

    /// The router that serves `model_id`, or the default router when the
    /// fleet gives no answer.
    pub fn select_router_for_request(
        &self,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // In single-router mode (enable_igw=false), always use the default router
        if !self.enable_igw {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(ref default_id) = *default_router {
                debug!(
                    "Single-router mode: using default router {} for model {:?}",
                    default_id.as_str(),
                    model_id
                );
                return self.routers.get(default_id).map(|r| r.clone());
            }
        }

        self.select_router_for_model(model_id).or_else(|| {
            if let Some(model) = model_id {
                if self.external_router_missing(model) {
                    warn!(
                        model = %model,
                        "No provider router compiled in for this model's external worker"
                    );
                    return None;
                }
            }
            let default = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            default
                .as_ref()
                .and_then(|id| self.routers.get(id).map(|r| r.clone()))
        })
    }

    /// Hand a request to the router that serves `model`, or answer `miss`.
    async fn dispatch<F, Fut>(
        &self,
        model: Option<&str>,
        miss: impl IntoResponse,
        run: F,
    ) -> Response
    where
        F: FnOnce(Arc<dyn RouterTrait>) -> Fut,
        Fut: Future<Output = Response>,
    {
        match self.select_router_for_request(model) {
            Some(router) => run(router).await,
            None => miss.into_response(),
        }
    }
}

#[async_trait]
impl RouterTrait for Gateway {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn request_body_policy(&self) -> BodyPolicy {
        if self.router_count() == 1 {
            if let Some(router) = self.select_router_for_request(None) {
                return router.request_body_policy();
            }
        }
        BodyPolicy::MustBuffer(REASON_MODEL_SELECTION)
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        let miss = (
            StatusCode::SERVICE_UNAVAILABLE,
            "No routers with healthy workers available",
        );
        self.dispatch(None, miss, |router| async move {
            router.health_generate(req).await
        })
        .await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        let miss = (StatusCode::SERVICE_UNAVAILABLE, "No routers available");
        self.dispatch(None, miss, |router| async move {
            router.get_server_info(req).await
        })
        .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Model info is fleet-wide: the default router answers, else any.
        let default_id = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let router = match default_id {
            Some(id) => self.routers.get(&id).map(|r| r.clone()),
            None => self.routers.iter().next().map(|r| r.value().clone()),
        };
        match router {
            Some(router) => router.get_model_info(req).await,
            None => (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response(),
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: GenerateRequest,
        model_id: &str,
    ) -> Response {
        if self.requires_explicit_generate_model(model_id) {
            return route_error::bad_request(
                "missing_model",
                "/generate requests must include a model when IGW routing is enabled",
            );
        }
        self.dispatch(Some(model_id), NO_ROUTER, |router| async move {
            router
                .route_generate(headers, tenant_meta, body, model_id)
                .await
        })
        .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        self.dispatch(Some(model_id), NO_ROUTER, |router| async move {
            router
                .route_chat(headers, tenant_meta, body, model_id)
                .await
        })
        .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CompletionRequest,
        model_id: &str,
    ) -> Response {
        self.dispatch(Some(model_id), NO_ROUTER, |router| async move {
            router
                .route_completion(headers, tenant_meta, body, model_id)
                .await
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
        self.dispatch(Some(model_id), NO_ROUTER, |router| async move {
            router
                .route_messages(headers, tenant_meta, body, model_id)
                .await
        })
        .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available to handle responses request",
        );
        self.dispatch(Some(model_id), miss, |router| async move {
            router
                .route_responses(headers, tenant_meta, body, model_id)
                .await
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
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available to handle interactions request",
        );
        self.dispatch(model_id, miss, |router| async move {
            router
                .route_interactions(headers, tenant_meta, body, model_id)
                .await
        })
        .await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            format!("No router available to cancel response '{response_id}'"),
        );
        self.dispatch(None, miss, |router| async move {
            router.cancel_response(headers, response_id).await
        })
        .await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        self.dispatch(Some(model_id), NO_ROUTER, |router| async move {
            router
                .route_embeddings(headers, tenant_meta, body, model_id)
                .await
        })
        .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ClassifyRequest,
        model_id: &str,
    ) -> Response {
        self.dispatch(Some(model_id), NO_ROUTER, |router| async move {
            router
                .route_classify(headers, tenant_meta, body, model_id)
                .await
        })
        .await
    }

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        self.dispatch(Some(model_id), NO_ROUTER, |router| async move {
            router
                .route_audio_transcriptions(headers, tenant_meta, body, audio, model_id)
                .await
        })
        .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: RerankRequest,
        model_id: &str,
    ) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available for rerank request",
        );
        self.dispatch(Some(model_id), miss, |router| async move {
            router
                .route_rerank(headers, tenant_meta, body, model_id)
                .await
        })
        .await
    }

    async fn route_realtime_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeSessionCreateRequest,
    ) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available for realtime session request",
        );
        let model = body.model.as_deref();
        self.dispatch(model, miss, |router| async move {
            router.route_realtime_session(headers, body).await
        })
        .await
    }

    async fn route_realtime_client_secret(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available for realtime client secret request",
        );
        let model = body.session.model.as_deref();
        self.dispatch(model, miss, |router| async move {
            router.route_realtime_client_secret(headers, body).await
        })
        .await
    }

    async fn route_realtime_transcription_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available for realtime transcription request",
        );
        let model = body.model.as_deref();
        self.dispatch(model, miss, |router| async move {
            router
                .route_realtime_transcription_session(headers, body)
                .await
        })
        .await
    }

    async fn route_realtime_ws(&self, req: Request<Body>, model: &str) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available for realtime WebSocket request",
        );
        self.dispatch(Some(model), miss, |router| async move {
            router.route_realtime_ws(req, model).await
        })
        .await
    }

    async fn route_realtime_webrtc(&self, req: Request<Body>, model: &str) -> Response {
        let miss = (
            StatusCode::NOT_FOUND,
            "No router available for realtime WebRTC request",
        );
        self.dispatch(Some(model), miss, |router| async move {
            router.route_realtime_webrtc(req, model).await
        })
        .await
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for Gateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gateway")
            .field("routers", &self.routers.len())
            .field("enable_igw", &self.enable_igw)
            .field("workers_count", &self.worker_registry.get_all().len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use openai_protocol::model_card::ModelCard;

    use super::*;
    use crate::{
        middleware::{RouteRequestMeta, TenantKey},
        routers::factory::router_ids,
        worker::{BasicWorkerBuilder, CircuitBreakerConfig, WorkerRegistry, WorkerType},
    };

    #[derive(Debug)]
    struct StubRouter;

    #[async_trait]
    impl RouterTrait for StubRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn route_generate(
            &self,
            _headers: Option<&HeaderMap>,
            _tenant_meta: &TenantRequestMeta,
            _body: GenerateRequest,
            _model_id: &str,
        ) -> Response {
            (StatusCode::OK, "routed").into_response()
        }

        fn router_type(&self) -> &'static str {
            "stub"
        }
    }

    #[derive(Debug)]
    struct PdStubRouter;

    #[async_trait]
    impl RouterTrait for PdStubRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn route_generate(
            &self,
            _headers: Option<&HeaderMap>,
            _tenant_meta: &TenantRequestMeta,
            _body: GenerateRequest,
            _model_id: &str,
        ) -> Response {
            (StatusCode::OK, "pd-routed").into_response()
        }

        fn router_type(&self) -> &'static str {
            "pd"
        }
    }

    #[derive(Debug)]
    struct EpdStubRouter;

    #[async_trait]
    impl RouterTrait for EpdStubRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn route_generate(
            &self,
            _headers: Option<&HeaderMap>,
            _tenant_meta: &TenantRequestMeta,
            _body: GenerateRequest,
            _model_id: &str,
        ) -> Response {
            (StatusCode::OK, "epd-routed").into_response()
        }

        fn router_type(&self) -> &'static str {
            "epd"
        }
    }

    fn test_gateway(enable_igw: bool) -> Arc<Gateway> {
        let mut manager = Gateway::new(Arc::new(WorkerRegistry::new()));
        manager.enable_igw = enable_igw;
        let manager = Arc::new(manager);
        manager.register_router(router_ids::HTTP_REGULAR, Arc::new(StubRouter));
        manager
    }

    /// A lone router speaks for itself — a forward-capable one keeps
    /// streaming enabled, a buffering one shows its derived reason — and
    /// more than one router makes dispatch model-addressed.
    #[test]
    fn body_policy_delegates_to_a_lone_router_and_buffers_multi_router() {
        #[derive(Debug)]
        struct ForwardStubRouter;

        #[async_trait]
        impl RouterTrait for ForwardStubRouter {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn request_body_policy(&self) -> BodyPolicy {
                BodyPolicy::ForwardCapable
            }

            fn router_type(&self) -> &'static str {
                "stub_forward"
            }
        }

        let forward_manager = {
            let mut m = Gateway::new(Arc::new(WorkerRegistry::new()));
            m.enable_igw = false;
            let m = Arc::new(m);
            m.register_router(router_ids::HTTP_REGULAR, Arc::new(ForwardStubRouter));
            m
        };
        assert_eq!(
            forward_manager.request_body_policy(),
            BodyPolicy::ForwardCapable
        );

        let manager = test_gateway(false);
        assert_eq!(
            manager.request_body_policy(),
            BodyPolicy::MustBuffer("stub")
        );

        let pd_manager = {
            let mut m = Gateway::new(Arc::new(WorkerRegistry::new()));
            m.enable_igw = false;
            let m = Arc::new(m);
            m.register_router(router_ids::HTTP_PD, Arc::new(PdStubRouter));
            m
        };
        assert_eq!(
            pd_manager.request_body_policy(),
            BodyPolicy::MustBuffer("pd")
        );

        let manager = test_gateway(true);
        manager.register_router(router_ids::HTTP_PD, Arc::new(PdStubRouter));
        assert_eq!(
            manager.request_body_policy(),
            BodyPolicy::MustBuffer(REASON_MODEL_SELECTION)
        );
    }

    fn test_tenant_meta() -> TenantRequestMeta {
        RouteRequestMeta::new(TenantKey::from("test-tenant"))
    }

    fn generate_request_without_model() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({ "text": "hello" })).unwrap()
    }

    #[tokio::test]
    async fn igw_generate_rejects_default_unknown_model() {
        let manager = test_gateway(true);
        let request = generate_request_without_model();

        assert_eq!(request.model, UNKNOWN_MODEL_ID);

        let response = manager
            .route_generate(None, &test_tenant_meta(), request.clone(), &request.model)
            .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            route_error::extract_error_code_from_response(&response),
            "missing_model"
        );
    }

    #[tokio::test]
    async fn single_router_generate_keeps_default_unknown_model_behavior() {
        let manager = test_gateway(false);
        let request = generate_request_without_model();

        assert_eq!(request.model, UNKNOWN_MODEL_ID);

        let response = manager
            .route_generate(None, &test_tenant_meta(), request.clone(), &request.model)
            .await;

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn weighted_routing_splits_40_pd_60_regular() {
        let registry = Arc::new(WorkerRegistry::new());

        let mut url_idx = 0;
        let mut add_workers = |wtype: WorkerType, count: usize| {
            for _ in 0..count {
                let mut labels = HashMap::new();
                labels.insert("model_id".to_string(), "model-x".to_string());
                let worker = BasicWorkerBuilder::new(format!("http://w{url_idx}:8080"))
                    .worker_type(wtype)
                    .connection_mode(ConnectionMode::Http)
                    .labels(labels)
                    .circuit_breaker_config(CircuitBreakerConfig::default())
                    .build();
                registry.register(Arc::new(worker)).unwrap();
                url_idx += 1;
            }
        };

        // Try adding 2 prefill, 2 decode workers, and 6 regular workers.
        // We should send 40% of traffic to PD and 60% to regular.
        add_workers(WorkerType::Prefill, 2);
        add_workers(WorkerType::Decode, 2);
        add_workers(WorkerType::Regular, 6);

        let mut manager = Gateway::new(registry);
        manager.enable_igw = true;
        let manager = Arc::new(manager);
        manager.register_router(router_ids::HTTP_PD, Arc::new(PdStubRouter));
        manager.register_router(router_ids::HTTP_REGULAR, Arc::new(StubRouter));

        let n = 10_000;
        let pd_count = (0..n)
            .filter(|_| {
                manager
                    .select_router_for_request(Some("model-x"))
                    .map(|r| r.router_type() == "pd")
                    .unwrap_or(false)
            })
            .count();

        let pd_ratio = pd_count as f64 / n as f64;
        let expected = 0.4; // 4 PD workers / 10 total
        let tolerance = 0.05;
        assert!(
            (pd_ratio - expected).abs() < tolerance,
            "PD ratio {pd_ratio:.3} was outside expected {expected} ± {tolerance}",
        );
    }

    #[test]
    fn weighted_routing_selects_epd_for_grpc_epd_workers() {
        let registry = Arc::new(WorkerRegistry::new());

        for (idx, wtype) in [WorkerType::Encode, WorkerType::Prefill, WorkerType::Decode]
            .into_iter()
            .enumerate()
        {
            let mut labels = HashMap::new();
            labels.insert("model_id".to_string(), "model-x".to_string());
            let worker = BasicWorkerBuilder::new(format!("http://epd-w{idx}:8080"))
                .worker_type(wtype)
                .connection_mode(ConnectionMode::Grpc)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build();
            registry.register(Arc::new(worker)).unwrap();
        }

        let mut manager = Gateway::new(registry);
        manager.enable_igw = true;
        let manager = Arc::new(manager);
        manager.register_router(router_ids::GRPC_PD, Arc::new(PdStubRouter));
        manager.register_router(router_ids::GRPC_EPD, Arc::new(EpdStubRouter));

        for _ in 0..100 {
            let router = manager
                .select_router_for_request(Some("model-x"))
                .expect("expected EPD router");

            assert_eq!(router.router_type(), "epd");
        }
    }

    #[test]
    fn a_leg_on_a_transport_the_family_cannot_use_does_not_count() {
        // A gRPC prefill worker and a ZMQ decode worker: the gRPC PD family
        // selects from gRPC-only pools, so the fleet is not PD-ready and the
        // request must not be steered to a family that cannot select a pair.
        let registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, connection) in [
            (
                "grpc://prefill:8080",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
            ),
            ("zmq://decode:8080", WorkerType::Decode, ConnectionMode::Zmq),
            (
                "grpc://regular:8080",
                WorkerType::Regular,
                ConnectionMode::Grpc,
            ),
        ] {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .connection_mode(connection)
                .model(ModelCard::new("model-x"))
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build();
            registry.register(Arc::new(worker)).unwrap();
        }
        let mut gateway = Gateway::new(registry);
        gateway.enable_igw = true;
        let gateway = Arc::new(gateway);
        gateway.register_router(router_ids::GRPC_PD, Arc::new(PdStubRouter));
        gateway.register_router(router_ids::GRPC_REGULAR, Arc::new(StubRouter));

        for _ in 0..50 {
            let router = gateway
                .select_router_for_request(Some("model-x"))
                .expect("the regular family serves the model");
            assert_eq!(router.router_type(), "stub");
        }
    }

    #[test]
    fn an_external_worker_sends_the_request_to_its_provider_router() {
        let registry = Arc::new(WorkerRegistry::new());
        let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
            "url": "https://api.anthropic.com",
            "runtime_type": "external",
            "provider": "anthropic",
            "models": [{"id": "claude-3-5-sonnet"}],
        }))
        .unwrap();
        let worker = BasicWorkerBuilder::from_spec(spec)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build();
        registry.register(Arc::new(worker)).unwrap();
        let mut gateway = Gateway::new(registry);
        gateway.enable_igw = true;
        let gateway = Arc::new(gateway);
        gateway.register_router(router_ids::HTTP_REGULAR, Arc::new(StubRouter));

        // Without the provider router the request fails instead of falling
        // through to the self-hosted default.
        assert!(gateway
            .select_router_for_request(Some("claude-3-5-sonnet"))
            .is_none());

        gateway.register_router(router_ids::HTTP_ANTHROPIC, Arc::new(PdStubRouter));
        let router = gateway
            .select_router_for_request(Some("claude-3-5-sonnet"))
            .expect("the Anthropic router is mounted");
        assert_eq!(router.router_type(), "pd");
    }

    #[test]
    fn weighted_routing_accepts_model_alias() {
        let registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type) in [
            ("http://prefill:8080", WorkerType::Prefill),
            ("http://decode:8080", WorkerType::Decode),
        ] {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Http)
                .model(ModelCard::new("canonical-model").with_alias("model-alias"))
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build();
            registry.register(Arc::new(worker)).unwrap();
        }

        let mut manager = Gateway::new(registry);
        manager.enable_igw = true;
        let manager = Arc::new(manager);
        manager.register_router(router_ids::HTTP_REGULAR, Arc::new(StubRouter));
        manager.register_router(router_ids::HTTP_PD, Arc::new(PdStubRouter));

        let router = manager
            .select_router_for_request(Some("model-alias"))
            .expect("alias should select the PD router");
        assert_eq!(router.router_type(), "pd");
    }
}
