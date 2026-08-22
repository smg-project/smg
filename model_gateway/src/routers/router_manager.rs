//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use dashmap::DashMap;
use futures::future::select_all;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    model_card::ModelCard,
    models::ListModelsResponse,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::RerankRequest,
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
    UNKNOWN_MODEL_ID,
};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    middleware::{AuthConfig, TenantRequestMeta},
    routers::{
        common::header_utils::apply_provider_headers,
        error as route_error,
        factory::{router_ids, RouterId},
        RouterFactory, RouterTrait,
    },
    server::ServerConfig,
    worker::{ConnectionMode, ProviderType, RuntimeType, Worker, WorkerRegistry, WorkerType},
};

pub struct RouterManager {
    worker_registry: Arc<WorkerRegistry>,
    client: reqwest::Client,
    /// Every credential that authenticates as *this gateway* (shared
    /// `api_key` plus any per-tenant keys) — not just the shared key. Used
    /// to keep `/v1/models`' BYOK short-circuit from mistaking a valid
    /// tenant-scoped key for a foreign upstream-provider credential and
    /// forwarding it externally.
    gateway_auth: AuthConfig,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    enable_igw: bool,
}

impl RouterManager {
    pub fn new(worker_registry: Arc<WorkerRegistry>, client: reqwest::Client) -> Self {
        Self {
            worker_registry,
            client,
            gateway_auth: AuthConfig::new(None),
            routers: Arc::new(DashMap::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            enable_igw: false,
        }
    }

    /// Register a router if creation succeeded, log either way.
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
        let mut manager = Self::new(
            app_context.worker_registry.clone(),
            app_context.client.clone(),
        );
        manager.enable_igw = config.router_config.enable_igw;
        manager.gateway_auth = AuthConfig::with_tenant_keys(
            config.router_config.api_key.clone(),
            &config.router_config.tenant_api_keys,
        );
        let manager = Arc::new(manager);

        if config.router_config.enable_igw {
            info!("Initializing RouterManager in multi-router mode (IGW)");

            let routers =
                RouterFactory::create_igw_routers(&config.router_config.policy, app_context).await;

            for (id, label, result) in routers {
                manager.try_register(id, label, result);
            }

            info!(
                "RouterManager initialized with {} routers for multi-router mode",
                manager.router_count(),
            );
        } else {
            info!("Initializing RouterManager in single-router mode");

            let single_router = Arc::from(RouterFactory::create_router(app_context).await?);
            let router_id = Self::determine_router_id(
                &config.router_config.mode,
                config.router_config.connection_mode,
            );

            info!("Created single router with ID: {}", router_id.as_str());
            manager.register_router(router_id.clone(), single_router);
            manager.set_default_router(router_id);
        }

        if manager.router_count() == 0 {
            return Err("No routers could be initialized".to_string());
        }

        Ok(manager)
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

    fn select_router_for_workers(
        &self,
        workers: &[Arc<dyn Worker>],
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // External workers take highest priority when a model is known.
        if let Some(model) = model_id {
            for w in workers {
                if matches!(w.metadata().spec.runtime_type, RuntimeType::External) {
                    let router_id = match w.provider_for_model(model) {
                        Some(ProviderType::Gemini) => &router_ids::HTTP_GEMINI,
                        Some(ProviderType::Anthropic) => &router_ids::HTTP_ANTHROPIC,
                        _ => &router_ids::HTTP_OPENAI,
                    };
                    return self.routers.get(router_id).map(|r| r.clone());
                }
            }
        }

        let mut grpc_encode = 0;
        let mut grpc_prefill = 0;
        let mut http_prefill = 0;
        let mut grpc_decode = 0;
        let mut http_decode = 0;
        let mut grpc_regular = 0;
        let mut http_regular = 0;

        for w in workers {
            match (w.worker_type(), w.connection_mode()) {
                (WorkerType::Encode, ConnectionMode::Grpc | ConnectionMode::Zmq) => {
                    grpc_encode += 1;
                }
                (WorkerType::Encode, ConnectionMode::Http) => {}
                (WorkerType::Prefill, ConnectionMode::Grpc | ConnectionMode::Zmq) => {
                    grpc_prefill += 1;
                }
                (WorkerType::Prefill, ConnectionMode::Http) => http_prefill += 1,
                (WorkerType::Decode, ConnectionMode::Grpc | ConnectionMode::Zmq) => {
                    grpc_decode += 1;
                }
                (WorkerType::Decode, ConnectionMode::Http) => http_decode += 1,
                (WorkerType::Regular, ConnectionMode::Grpc | ConnectionMode::Zmq) => {
                    grpc_regular += 1;
                }
                (WorkerType::Regular, ConnectionMode::Http) => http_regular += 1,
            }
        }

        let grpc_epd_ready = grpc_encode > 0
            && grpc_prefill > 0
            && grpc_decode > 0
            && self.routers.contains_key(&router_ids::GRPC_EPD);
        let grpc_epd = if grpc_epd_ready {
            grpc_encode + grpc_prefill + grpc_decode
        } else {
            0
        };

        // We need at least one prefill and one decode worker to handle requests
        // in PD disaggregation mode.
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

    fn requires_explicit_generate_model(&self, model_id: &str) -> bool {
        self.enable_igw && (model_id.trim().is_empty() || model_id == UNKNOWN_MODEL_ID)
    }

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

        let workers = if let Some(model) = model_id {
            self.worker_registry.get_by_model(model).to_vec()
        } else {
            self.worker_registry.get_all()
        };

        self.select_router_for_workers(&workers, model_id)
            .or_else(|| {
                let default = self
                    .default_router
                    .read()
                    .unwrap_or_else(|e| e.into_inner());
                default
                    .as_ref()
                    .and_then(|id| self.routers.get(id).map(|r| r.clone()))
            })
    }

    /// Build a response from self-hosted registry models (excludes external workers).
    fn registry_models_response(&self) -> Response {
        let cards: Vec<_> = self
            .worker_registry
            .get_all()
            .iter()
            .filter(|w| !matches!(w.metadata().spec.runtime_type, RuntimeType::External))
            .flat_map(|w| w.models())
            .collect();
        if cards.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            let resp = ListModelsResponse::from_model_cards(cards);
            (StatusCode::OK, Json(resp)).into_response()
        }
    }

    /// Fan out to all healthy external upstreams concurrently with the caller's
    /// bearer token and return the first successful model inventory. Returns an
    /// empty vec on total failure.
    async fn fetch_upstream_models(&self, bearer_token: &str) -> Vec<ModelCard> {
        let unique_urls: Vec<_> = self
            .worker_registry
            .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
            .iter()
            .map(|w| w.url().to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        if unique_urls.is_empty() {
            return Vec::new();
        }

        debug!(
            "Trying {} upstream(s) for model discovery",
            unique_urls.len()
        );

        let auth = match HeaderValue::from_str(&format!("Bearer {bearer_token}")) {
            Ok(v) => Some(v),
            Err(e) => {
                warn!("Bearer token contains invalid header characters: {e}");
                return Vec::new();
            }
        };

        // Fan out concurrently; return first non-empty result.
        let mut pending: Vec<_> = unique_urls
            .into_iter()
            .map(|url| {
                Box::pin(Self::fetch_models_from(
                    self.client.clone(),
                    url,
                    auth.clone(),
                ))
            })
            .collect();

        while !pending.is_empty() {
            let (cards, _index, remaining) = select_all(pending).await;
            if !cards.is_empty() {
                return cards;
            }
            pending = remaining;
        }

        Vec::new()
    }

    /// Fetch models from a single upstream endpoint.
    async fn fetch_models_from(
        client: reqwest::Client,
        base_url: String,
        auth: Option<HeaderValue>,
    ) -> Vec<ModelCard> {
        let url = format!("{base_url}/v1/models");
        let req = apply_provider_headers(client.get(&url), &url, auth.as_ref());

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                debug!("Failed to reach upstream {url}: {e}");
                return Vec::new();
            }
        };

        if !resp.status().is_success() {
            debug!(
                "Upstream {url} returned {} for model discovery",
                resp.status()
            );
            return Vec::new();
        }

        match resp.json::<Value>().await {
            Ok(json) => ListModelsResponse::parse_upstream(&json, ProviderType::from_url(&url)),
            Err(e) => {
                warn!("Failed to parse upstream models from {url}: {e}");
                Vec::new()
            }
        }
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        let router = self.select_router_for_request(None);
        if let Some(router) = router {
            router.health_generate(_req).await
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No routers with healthy workers available",
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        let router = self.select_router_for_request(None);
        if let Some(router) = router {
            router.get_server_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract token from Authorization header (case-insensitive "Bearer " prefix
        // per RFC 7235) or Anthropic-style x-api-key header.
        let bearer_token = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| {
                let lower = h.to_ascii_lowercase();
                lower.starts_with("bearer ").then(|| h[7..].to_string())
            })
            .or_else(|| {
                req.headers()
                    .get("x-api-key")
                    .and_then(|h| h.to_str().ok())
                    .map(String::from)
            });

        // Short-circuit: if the token matches any of the gateway's own
        // credentials (shared or per-tenant), skip upstream fan-out and
        // return registry models directly. A tenant-scoped key must never
        // reach the BYOK fan-out below — that would forward it externally.
        if let Some(ref token) = bearer_token {
            if self.gateway_auth.contains_token(token) {
                return self.registry_models_response();
            }
        }

        // If the caller sent a provider token, try to discover models from
        // upstream providers. This enables BYOK (bring your own key) flows.
        if let Some(ref token) = bearer_token {
            let upstream_cards = self.fetch_upstream_models(token).await;
            if !upstream_cards.is_empty() {
                let resp = ListModelsResponse::from_model_cards(upstream_cards);
                return (StatusCode::OK, Json(resp)).into_response();
            }
            // All upstreams failed or returned nothing — fall through to registry.
        }

        self.registry_models_response()
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Route to default router or first available router
        let router_id = {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            default_router.clone()
        };

        let router = if let Some(id) = router_id {
            self.routers.get(&id).map(|r| r.clone())
        } else {
            // If no default, use first available router
            self.routers.iter().next().map(|r| r.value().clone())
        };

        if let Some(router) = router {
            router.get_model_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
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

        let router = self.select_router_for_request(Some(model_id));

        if let Some(router) = router {
            router
                .route_generate(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));

        if let Some(router) = router {
            router
                .route_chat(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CompletionRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));

        if let Some(router) = router {
            router
                .route_completion(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));
        if let Some(router) = router {
            router
                .route_messages(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));
        if let Some(router) = router {
            router
                .route_responses(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle responses request",
            )
                .into_response()
        }
    }

    async fn route_interactions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: InteractionsRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Owned so it can outlive `body`, which moves into the routed call.
        let selected_model = model_id
            .map(str::to_string)
            .or_else(|| body.model.clone())
            .or_else(|| body.agent.clone());
        let router = self.select_router_for_request(selected_model.as_deref());

        if let Some(router) = router {
            router
                .route_interactions(headers, tenant_meta, body, selected_model.as_deref())
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle interactions request",
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{response_id}'"),
            )
                .into_response()
        }
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));

        if let Some(router) = router {
            router
                .route_embeddings(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ClassifyRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));

        if let Some(router) = router {
            router
                .route_classify(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));

        if let Some(router) = router {
            router
                .route_audio_transcriptions(headers, tenant_meta, body, audio, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: RerankRequest,
        model_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(Some(model_id));

        if let Some(router) = router {
            router
                .route_rerank(headers, tenant_meta, body, model_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for rerank request",
            )
                .into_response()
        }
    }

    async fn route_realtime_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeSessionCreateRequest,
    ) -> Response {
        let model = body.model.as_deref();
        let router = self.select_router_for_request(model);
        if let Some(router) = router {
            router.route_realtime_session(headers, body).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime session request",
            )
                .into_response()
        }
    }

    async fn route_realtime_client_secret(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        let model = body.session.model.as_deref();
        let router = self.select_router_for_request(model);
        if let Some(router) = router {
            router.route_realtime_client_secret(headers, body).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime client secret request",
            )
                .into_response()
        }
    }

    async fn route_realtime_transcription_session(
        &self,
        headers: Option<&HeaderMap>,
        body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        let model = body.model.as_deref();
        let router = self.select_router_for_request(model);
        if let Some(router) = router {
            router
                .route_realtime_transcription_session(headers, body)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime transcription request",
            )
                .into_response()
        }
    }

    async fn route_realtime_ws(&self, req: Request<Body>, model: &str) -> Response {
        let router = self.select_router_for_request(Some(model));
        if let Some(router) = router {
            router.route_realtime_ws(req, model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime WebSocket request",
            )
                .into_response()
        }
    }

    async fn route_realtime_webrtc(&self, req: Request<Body>, model: &str) -> Response {
        let router = self.select_router_for_request(Some(model));
        if let Some(router) = router {
            router.route_realtime_webrtc(req, model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for realtime WebRTC request",
            )
                .into_response()
        }
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*default_router)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;

    use super::*;
    use crate::{
        config::TenantApiKeyEntry,
        middleware::{RouteRequestMeta, TenantKey},
        routers::factory::router_ids,
        worker::{BasicWorkerBuilder, CircuitBreakerConfig, WorkerRegistry},
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

    fn test_manager(enable_igw: bool) -> Arc<RouterManager> {
        let mut manager =
            RouterManager::new(Arc::new(WorkerRegistry::new()), reqwest::Client::new());
        manager.enable_igw = enable_igw;
        let manager = Arc::new(manager);
        manager.register_router(router_ids::HTTP_REGULAR, Arc::new(StubRouter));
        manager
    }

    fn test_tenant_meta() -> TenantRequestMeta {
        RouteRequestMeta::new(TenantKey::from("test-tenant"))
    }

    /// A tenant-scoped credential must never reach `fetch_upstream_models`:
    /// that would forward the gateway's own secret to an external provider
    /// as if it were the caller's BYOK token. Verified against a real mock
    /// upstream that counts hits, since a stubbed HTTP client can't
    /// distinguish "short-circuited" from "fell through and failed" by
    /// response alone — both end up returning registry models on failure.
    #[tokio::test]
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    async fn get_models_short_circuits_for_tenant_key_without_forwarding_upstream() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let hit_count = Arc::new(AtomicUsize::new(0));
        let hit_count_clone = hit_count.clone();
        let mock_app = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(move || {
                let hit_count = hit_count_clone.clone();
                async move {
                    hit_count.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({"object": "list", "data": []}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let registry = Arc::new(WorkerRegistry::new());
        let external_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{addr}"))
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(external_worker);
        // A non-External worker so registry_models_response() (the
        // short-circuit path) has something to return besides 503.
        let internal_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("internal-model")])
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(internal_worker);

        let mut manager = RouterManager::new(registry, reqwest::Client::new());
        manager.gateway_auth = AuthConfig::with_tenant_keys(
            Some("shared-secret".to_string()),
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header(header::AUTHORIZATION, "Bearer team-red-secret")
            .body(Body::empty())
            .unwrap();
        let response = manager.get_models(req).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            0,
            "a tenant-scoped key must never trigger upstream BYOK fan-out"
        );

        // Control: a genuinely unrecognized token should still trigger BYOK
        // fan-out — confirms the short-circuit is keyed on the gateway's own
        // credentials, not disabled entirely.
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header(header::AUTHORIZATION, "Bearer not-a-gateway-key")
            .body(Body::empty())
            .unwrap();
        let _ = manager.get_models(req).await;
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            1,
            "an unrecognized token should still fan out to upstream providers"
        );
    }

    fn generate_request_without_model() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({ "text": "hello" })).unwrap()
    }

    #[tokio::test]
    async fn igw_generate_rejects_default_unknown_model() {
        let manager = test_manager(true);
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
        let manager = test_manager(false);
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

        let mut manager = RouterManager::new(registry, reqwest::Client::new());
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

        let mut manager = RouterManager::new(registry, reqwest::Client::new());
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

        let mut manager = RouterManager::new(registry, reqwest::Client::new());
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
