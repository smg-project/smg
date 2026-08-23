use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    extract::{Extension, Path, Query, Request, State},
    http::{header::InvalidHeaderName, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use llm_tokenizer::TokenizerRegistry;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    multipart::AudioTranscriptionMultipart,
    parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::{RerankRequest, V1RerankReqInput},
    responses::ResponsesRequest,
    tokenize::{AddTokenizerRequest, DetokenizeRequest, TokenizeRequest},
    validated::ValidatedJson,
    worker::{
        ListWorkersQuery, StartProfileRequest, StopProfileRequest, WorkerSpec, WorkerUpdateRequest,
    },
};
use rustls::crypto::ring;
use serde::Deserialize;
use serde_json::Value;
use smg_mesh::{MeshServerBuilder, MeshServerConfig, MeshServerHandler};
use tokio::{signal, spawn};
use tracing::{debug, error, info, warn, Level};
use wfaas::LoggingSubscriber;

use crate::{
    app_context::AppContext,
    config::RouterConfig,
    mesh::MeshAdapters,
    middleware::{self, AdmissionQueue, AuthConfig},
    observability::{
        logging::{self, LoggingConfig},
        metrics::{self, PrometheusConfig},
        metrics_server, otel_trace, runtime_metrics,
    },
    routers::{
        common::realtime::ws::RealtimeQueryParams,
        conversations,
        http::router::{stream_eligible_request_bodies, StreamBodyState},
        parse, responses as response_handlers,
        router_manager::RouterManager,
        tokenize, RouterTrait,
    },
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    wasm::route::{add_wasm_module, list_wasm_modules, remove_wasm_module},
    worker::{
        manager::{WorkerManager, WorkerManagerConfig},
        ConnectionMode,
    },
    workflow::{
        job_queue::{JobQueue, JobQueueConfig},
        Job, TokenizerConfigRequest, WorkflowEngines,
    },
};
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub admission_queue: Option<Arc<AdmissionQueue>>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub mesh_handler: Option<Arc<MeshServerHandler>>,
    pub mesh_adapters: Option<Arc<MeshAdapters>>,
    /// Cached O(1) readiness state shared with the optional dedicated
    /// probe listener. Maintained event-driven by
    /// [`crate::health::spawn_readiness_maintainer`].
    pub probe_state: Arc<crate::health::ProbeState>,
}

async fn parse_function_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ParseFunctionCallRequest>,
) -> Response {
    parse::parse_function_call(&state.context, &req).await
}

async fn parse_reasoning(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SeparateReasoningRequest>,
) -> Response {
    parse::parse_reasoning(&state.context, &req).await
}

async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

async fn liveness() -> Response {
    crate::health::liveness_response()
}

/// O(1) readiness: reads the event-maintained snapshot (see
/// [`crate::health::spawn_readiness_maintainer`]) and the drain flag — no registry
/// scan per probe. The decision logic lives in
/// [`crate::health::ProbeState::recompute`] and is unchanged from the previous
/// inline implementation, plus the drain gate (not-ready while draining).
async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    state.probe_state.readiness_response()
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    WorkerManager::get_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    Json(body): Json<GenerateRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_generate(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_chat(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_completion(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    ValidatedJson(body): ValidatedJson<RerankRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_rerank(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    Json(body): Json<V1RerankReqInput>,
) -> Response {
    let rerank_body: RerankRequest = body.into();
    let model = rerank_body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_rerank(Some(&headers), &tenant_meta, rerank_body, &model),
        )
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_responses(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn v1_interactions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    ValidatedJson(body): ValidatedJson<InteractionsRequest>,
) -> Response {
    let model_id = body.model.clone().or_else(|| body.agent.clone());
    cancel
        .guard(state.router.route_interactions(
            Some(&headers),
            &tenant_meta,
            body,
            model_id.as_deref(),
        ))
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_embeddings(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn v1_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    ValidatedJson(body): ValidatedJson<CreateMessageRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_messages(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn v1_classify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    Json(body): Json<ClassifyRequest>,
) -> Response {
    let model = body.model.clone();
    cancel
        .guard(
            state
                .router
                .route_classify(Some(&headers), &tenant_meta, body, &model),
        )
        .await
}

async fn v1_audio_transcriptions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Extension(tenant_meta): Extension<middleware::TenantRequestMeta>,
    cancel: middleware::scheduler::PreemptionGuard,
    AudioTranscriptionMultipart { request, audio }: AudioTranscriptionMultipart,
) -> Response {
    cancel
        .guard(state.router.route_audio_transcriptions(
            Some(&headers),
            &tenant_meta,
            &request,
            audio,
            &request.model,
        ))
        .await
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::get_response(&state.context.response_storage, &response_id).await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::delete_response(&state.context.response_storage, &response_id).await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Response {
    response_handlers::list_response_input_items(&state.context.response_storage, &response_id)
        .await
}

async fn v1_conversations_create(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation(&state.context.conversation_storage, body).await
}

async fn v1_conversations_get(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::get_conversation(&state.context.conversation_storage, &conversation_id).await
}

async fn v1_conversations_update(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::update_conversation(&state.context.conversation_storage, &conversation_id, body)
        .await
}

async fn v1_conversations_delete(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::delete_conversation(&state.context.conversation_storage, &conversation_id).await
}

#[derive(Deserialize, Default)]
struct ListItemsQuery {
    limit: Option<usize>,
    order: Option<String>,
    after: Option<String>,
}

async fn v1_conversations_list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(ListItemsQuery {
        limit,
        order,
        after,
    }): Query<ListItemsQuery>,
) -> Response {
    conversations::list_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        limit,
        order.as_deref(),
        after.as_deref(),
    )
    .await
}

#[derive(Deserialize, Default)]
struct GetItemQuery {
    /// Additional fields to include in response (not yet implemented)
    include: Option<Vec<String>>,
}

async fn v1_conversations_create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        body,
    )
    .await
}

async fn v1_conversations_get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    Query(query): Query<GetItemQuery>,
) -> Response {
    conversations::get_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
        query.include,
    )
    .await
}

async fn v1_conversations_delete_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> Response {
    conversations::delete_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
    )
    .await
}

async fn v1_realtime_webrtc(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RealtimeQueryParams>,
    req: Request,
) -> Response {
    // Model may come from query param (application/sdp) or session body
    // (multipart/form-data). Let the handler validate per content type.
    let model = params.model.unwrap_or_default();
    state.router.route_realtime_webrtc(req, &model).await
}

async fn v1_realtime_ws(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RealtimeQueryParams>,
    req: Request,
) -> Response {
    let model = match params.model {
        Some(m) if !m.trim().is_empty() => m,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "Missing required 'model' query parameter",
            )
                .into_response();
        }
    };
    state.router.route_realtime_ws(req, &model).await
}

async fn v1_realtime_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeSessionCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_session(Some(&headers), &body)
        .await
}

async fn v1_realtime_client_secret(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeClientSecretCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_client_secret(Some(&headers), &body)
        .await
}

async fn v1_realtime_transcription_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RealtimeTranscriptionSessionCreateRequest>,
) -> Response {
    state
        .router
        .route_realtime_transcription_session(Some(&headers), &body)
        .await
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::flush_cache_all(&state.context.worker_registry)
        .await
        .into_response()
}

async fn start_profile(
    State(state): State<Arc<AppState>>,
    body: Option<Json<StartProfileRequest>>,
) -> Response {
    let body = body.map_or_else(StartProfileRequest::default, |Json(body)| body);
    WorkerManager::start_profile_all(
        &state.context.worker_registry,
        &body.options,
        body.url.as_deref(),
    )
    .await
    .into_response()
}

async fn stop_profile(
    State(state): State<Arc<AppState>>,
    body: Option<Json<StopProfileRequest>>,
) -> Response {
    let body = body.map_or_else(StopProfileRequest::default, |Json(body)| body);
    WorkerManager::stop_profile_all(&state.context.worker_registry, body.url.as_deref())
        .await
        .into_response()
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::get_all_worker_loads(
        &state.context.worker_registry,
        &state.context.client,
        state
            .context
            .worker_monitor
            .as_ref()
            .map(|monitor| monitor.native_loads_absent()),
    )
    .await
    .into_response()
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerSpec>,
) -> Response {
    match state.context.worker_service.create_worker(config).await {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn list_workers_rest(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListWorkersQuery>,
) -> Response {
    state
        .context
        .worker_service
        .list_workers(query.model.as_deref())
        .into_response()
}

async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state.context.worker_service.get_worker(&worker_id_raw) {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state
        .context
        .worker_service
        .delete_worker(&worker_id_raw)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn update_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
    Json(update): Json<WorkerUpdateRequest>,
) -> Response {
    match state
        .context
        .worker_service
        .update_worker(&worker_id_raw, update)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn replace_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
    Json(config): Json<WorkerSpec>,
) -> Response {
    match state
        .context
        .worker_service
        .replace_worker(&worker_id_raw, config)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

// ============================================================================
// Tokenize / Detokenize Handlers
// ============================================================================

async fn v1_tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    tokenize::tokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    tokenize::detokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_tokenizers_add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddTokenizerRequest>,
) -> Response {
    tokenize::add_tokenizer(&state.context, request).await
}

async fn v1_tokenizers_list(State(state): State<Arc<AppState>>) -> Response {
    tokenize::list_tokenizers(&state.context.tokenizer_registry).await
}

async fn v1_tokenizers_get(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_info(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_status(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_status(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_remove(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::remove_tokenizer(&state.context, &tokenizer_id).await
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Dedicated port for the isolated liveness/readiness/health probe listener. `None`
    /// leaves the dedicated listener off; probe routes always stay on the
    /// main `port` regardless.
    pub health_check_port: Option<u16>,
    /// Explicit async runtime worker-thread count. `None` uses tokio's
    /// container-aware default (`available_parallelism()`).
    pub runtime_worker_threads: Option<usize>,
    pub router_config: RouterConfig,
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub log_json: bool,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
    pub request_id_headers: Option<Vec<String>>,
    pub shutdown_grace_period_secs: u64,
    /// Control plane authentication configuration
    pub control_plane_auth: Option<smg_auth::ControlPlaneAuthConfig>,
    pub mesh_server_config: Option<MeshServerConfig>,
    /// Bind address for WebRTC UDP sockets.
    /// `None` means use the default (0.0.0.0, auto-detect candidate IP).
    pub webrtc_bind_addr: Option<std::net::IpAddr>,
    /// STUN server for ICE candidate gathering (host:port).
    /// Defaults to `stun.l.google.com:19302`; `"none"` to disable.
    pub webrtc_stun_server: Option<String>,
}

/// Apply the request-admission layer to a protected route group.
///
/// `AdmissionMode::Priority` installs the priority scheduler middleware
/// (carrying its own `Arc<SchedulerState>`); `AdmissionMode::Legacy` installs
/// the original `concurrency_limit_middleware`. Either runs innermost of the
/// protective layers (closest to the handler), after tenant resolution has
/// populated `RouteRequestMeta`.
///
/// Invariant: a request parked at admission keeps its body unread — bodies
/// are collected only at handler extraction, after a permit is granted.
fn with_admission_layer(
    router: Router<Arc<AppState>>,
    admission_mode: &middleware::scheduler::AdmissionMode,
    app_state: Arc<AppState>,
) -> Router<Arc<AppState>> {
    match admission_mode {
        middleware::scheduler::AdmissionMode::Priority(scheduler_state) => {
            router.route_layer(axum::middleware::from_fn_with_state(
                scheduler_state.clone(),
                middleware::scheduler::priority_admission_middleware,
            ))
        }
        middleware::scheduler::AdmissionMode::Legacy => {
            router.route_layer(axum::middleware::from_fn_with_state(
                app_state,
                middleware::concurrency_limit_middleware,
            ))
        }
    }
}

/// `serving_auth_config` covers inference-serving routes and may include
/// per-tenant keys. `admin_auth_config` is the admin/worker-management
/// fallback (used when `control_plane_auth_state` is `None`) and must
/// contain only the shared gateway-wide key, never per-tenant keys.
pub fn build_app(
    app_state: Arc<AppState>,
    serving_auth_config: AuthConfig,
    admin_auth_config: AuthConfig,
    control_plane_auth_state: Option<smg_auth::ControlPlaneAuthState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Result<Router, InvalidHeaderName> {
    // Pending (upgrade not completed): 30s TTL
    // Disconnected: 60 min TTL
    app_state.context.realtime_registry.start_reaper(
        Duration::from_secs(3600),
        Duration::from_secs(30),
        Duration::from_secs(60),
    );

    let tenant_resolution_state =
        middleware::TenantResolutionState::new(&app_state.context.router_config)?;

    // Choose the admission path once at startup: priority scheduler when
    // enabled (and it starts cleanly), otherwise the legacy concurrency limit.
    let admission_mode = middleware::scheduler::AdmissionMode::from_config(
        &app_state.context.router_config,
        app_state.context.worker_registry.clone(),
        app_state.context.rate_limiter.clone(),
    );

    let protected_routes = with_admission_layer(
        Router::new()
            .route("/v1/responses", post(v1_responses))
            .route("/v1/responses/{response_id}", get(v1_responses_get))
            .route(
                "/v1/responses/{response_id}/cancel",
                post(v1_responses_cancel),
            )
            .route("/v1/responses/{response_id}", delete(v1_responses_delete))
            .route(
                "/v1/responses/{response_id}/input_items",
                get(v1_responses_list_input_items),
            )
            .route("/v1/conversations", post(v1_conversations_create))
            .route(
                "/v1/conversations/{conversation_id}",
                get(v1_conversations_get)
                    .post(v1_conversations_update)
                    .delete(v1_conversations_delete),
            )
            .route(
                "/v1/conversations/{conversation_id}/items",
                get(v1_conversations_list_items).post(v1_conversations_create_items),
            )
            .route(
                "/v1/conversations/{conversation_id}/items/{item_id}",
                get(v1_conversations_get_item).delete(v1_conversations_delete_item),
            )
            .route_layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                middleware::storage_context_middleware,
            ))
            .route("/generate", post(generate))
            .route("/v1/chat/completions", post(v1_chat_completions))
            .route("/v1/completions", post(v1_completions))
            .route("/rerank", post(rerank))
            .route("/v1/rerank", post(v1_rerank))
            .route("/v1/embeddings", post(v1_embeddings))
            .route("/v1/messages", post(v1_messages))
            .route("/v1/interactions", post(v1_interactions))
            .route("/v1/classify", post(v1_classify))
            // Per-request buffer-vs-stream decision for typed-JSON bodies;
            // declined requests pass to the handlers untouched.
            .route_layer(axum::middleware::from_fn_with_state(
                StreamBodyState::new(app_state.router.clone(), app_state.context.clone()),
                stream_eligible_request_bodies,
            ))
            // Tokenize / Detokenize endpoints
            .route("/v1/tokenize", post(v1_tokenize))
            .route("/v1/detokenize", post(v1_detokenize))
            // Realtime REST endpoints (same middleware as other protected routes)
            .route("/v1/realtime/sessions", post(v1_realtime_session))
            .route(
                "/v1/realtime/client_secrets",
                post(v1_realtime_client_secret),
            )
            .route(
                "/v1/realtime/transcription_sessions",
                post(v1_realtime_transcription_session),
            ),
        &admission_mode,
        app_state.clone(),
    )
    .route_layer(axum::middleware::from_fn_with_state(
        tenant_resolution_state.clone(),
        middleware::route_request_meta_middleware,
    ))
    .route_layer(axum::middleware::from_fn_with_state(
        serving_auth_config.clone(),
        middleware::auth_middleware,
    ))
    .route_layer(axum::middleware::from_fn_with_state(
        app_state.clone(),
        middleware::wasm_middleware,
    ));

    // WebSocket and WebRTC routes: auth + concurrency but NO WASM middleware.
    // WASM OnResponse reconstructs the response from status/headers/body,
    // dropping the response extensions that carry the WebSocket upgrade future.
    let realtime_routes = with_admission_layer(
        Router::new()
            .route("/v1/realtime", get(v1_realtime_ws))
            .route("/v1/realtime/calls", post(v1_realtime_webrtc)),
        &admission_mode,
        app_state.clone(),
    )
    .route_layer(axum::middleware::from_fn_with_state(
        tenant_resolution_state.clone(),
        middleware::route_request_meta_middleware,
    ))
    .route_layer(axum::middleware::from_fn_with_state(
        serving_auth_config.clone(),
        middleware::auth_middleware,
    ));

    // Multipart upload routes: auth + concurrency but NO WASM middleware.
    // The WASM OnRequest phase buffers the full body into a `Vec<u8>` subject
    // to the WASM manager's `max_body_size` (10MB default). Audio uploads
    // routinely exceed that, so WASM middleware would reject them with 400
    // before reaching the handler.
    let multipart_upload_routes = with_admission_layer(
        Router::new().route("/v1/audio/transcriptions", post(v1_audio_transcriptions)),
        &admission_mode,
        app_state.clone(),
    )
    .route_layer(axum::middleware::from_fn_with_state(
        tenant_resolution_state,
        middleware::route_request_meta_middleware,
    ))
    .route_layer(axum::middleware::from_fn_with_state(
        serving_auth_config.clone(),
        middleware::auth_middleware,
    ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/engine_metrics", get(engine_metrics))
        .route("/v1/models", get(v1_models))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info));

    // Build admin routes with control plane auth if configured, otherwise use simple API key auth
    let admin_routes = Router::new()
        .route("/flush_cache", post(flush_cache))
        .route("/start_profile", post(start_profile))
        .route("/stop_profile", post(stop_profile))
        .route("/get_loads", get(get_loads))
        .route("/parse/function_call", post(parse_function_call))
        .route("/parse/reasoning", post(parse_reasoning))
        .route("/wasm", post(add_wasm_module))
        .route("/wasm/{module_uuid}", delete(remove_wasm_module))
        .route("/wasm", get(list_wasm_modules))
        // Tokenizer management endpoints
        .route(
            "/v1/tokenizers",
            post(v1_tokenizers_add).get(v1_tokenizers_list),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}",
            get(v1_tokenizers_get).delete(v1_tokenizers_remove),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}/status",
            get(v1_tokenizers_status),
        );

    // Build worker routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker).get(list_workers_rest))
        .route(
            "/workers/{worker_id}",
            get(get_worker)
                .put(replace_worker)
                .patch(update_worker)
                .delete(delete_worker),
        );

    // Fallback (no control-plane auth) normally uses `admin_auth_config`.
    // If only tenant keys are configured (no shared `--api-key`), there's no
    // credential that should reach these routes — deny outright instead of
    // falling back to `auth_middleware`, which would treat the empty config
    // as "open". Neither config having any key at all is a fully open
    // dev/test deployment, which keeps its legacy pass-through behavior.
    let apply_control_plane_auth = |routes: Router<Arc<AppState>>| {
        if let Some(ref cp_state) = control_plane_auth_state {
            routes.route_layer(axum::middleware::from_fn_with_state(
                cp_state.clone(),
                smg_auth::control_plane_auth_middleware,
            ))
        } else if !admin_auth_config.is_enabled() && serving_auth_config.is_enabled() {
            routes.route_layer(axum::middleware::from_fn(middleware::deny_all_middleware))
        } else {
            routes.route_layer(axum::middleware::from_fn_with_state(
                admin_auth_config.clone(),
                middleware::auth_middleware,
            ))
        }
    };
    let admin_routes = apply_control_plane_auth(admin_routes);
    let worker_routes = apply_control_plane_auth(worker_routes);

    // `/ha/*` management routes (routers/mesh handlers) are removed
    // in this PR — they all read/write through the v1
    // `MeshSyncManager` and don't map cleanly onto the v2 adapters.
    // A v2-aware admin surface will return in a follow-up PR once
    // adapters are production-wired.

    Ok(Router::new()
        .merge(protected_routes)
        .merge(realtime_routes)
        .merge(multipart_upload_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::HttpMetricsLayer::new(
            app_state.context.inflight_tracker.clone(),
        ))
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state))
}

pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Defense in depth: the CLI and Python bindings both validate via
    // `RouterConfigBuilder::build()` before reaching here, but `RouterConfig`
    // is public and `Deserialize`, so a Rust library caller can construct
    // `ServerConfig` directly (or deserialize it) and call `startup` without
    // ever going through the builder. Validate here too so `tenant_api_keys`
    // (and everything else `ConfigValidator` checks) is enforced regardless
    // of construction path — a bypassed empty/duplicate tenant key would
    // otherwise silently reach `AuthConfig`.
    config.router_config.validate()?;

    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    if let Some(trace_config) = &config.router_config.trace_config {
        otel_trace::otel_tracing_init(
            trace_config.enable_trace,
            Some(&trace_config.otlp_traces_endpoint),
        )?;
    }

    let _log_guard = if LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        None
    } else {
        Some(logging::init_logging(
            LoggingConfig {
                level: config
                    .log_level
                    .as_deref()
                    .and_then(|s| match s.to_uppercase().parse::<Level>() {
                        Ok(l) => Some(l),
                        Err(_) => {
                            warn!("Invalid log level string: '{s}'. Defaulting to INFO.");
                            None
                        }
                    })
                    .unwrap_or(Level::INFO),
                json_format: config.log_json,
                log_dir: config.log_dir.clone(),
                colorize: true,
                log_file_name: "smg".to_string(),
                log_targets: None,
            },
            config.router_config.trace_config.clone(),
        ))
    };

    // Seed the process-wide multimodal tensor transport defaults from the
    // resolved router config; per-worker specs still override at request time.
    use crate::routers::grpc::multimodal::init_mm_transport_defaults;
    init_mm_transport_defaults(
        config.router_config.multimodal_tensor_transport,
        config.router_config.multimodal_shm_min_bytes,
    );

    // Start the metrics server. It binds the port eagerly so we fail fast on
    // port conflicts or bad addresses.
    if let Some(prometheus_config) = &config.prometheus_config {
        let handle = metrics::start_prometheus(prometheus_config.clone());
        let (_metrics_addr, _server_handle) = metrics_server::start_metrics_server(
            handle,
            prometheus_config.host.clone(),
            prometheus_config.port,
        )
        .await?;
        // Tokio runtime self-observability (event-loop canary + sampler).
        // `startup` runs on the main runtime, so the observer lands on —
        // and therefore measures — the runtime that serves requests.
        runtime_metrics::spawn_observer();
    }

    // Build the mesh server if configured. Starting gossip is deferred until
    // MeshAdapters has registered the `worker:`/`rl:` CRDT namespaces below —
    // a remote op arriving for an unregistered prefix would merge through the
    // default last-writer-wins engine with the wrong semantics.
    let (mesh_server, mesh_handler) = match &config.mesh_server_config {
        Some(mesh_server_config) => {
            let (server, handler) = MeshServerBuilder::from(mesh_server_config).build();
            (Some(server), Some(Arc::new(handler)))
        }
        None => (None, None),
    };

    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | max_payload: {}MB",
        config.host,
        config.port,
        config.router_config.mode,
        config.router_config.policy,
        config.max_payload_size / (1024 * 1024)
    );

    let app_context = Arc::new(
        AppContext::from_config(
            config.router_config.clone(),
            config.request_timeout_secs,
            config.webrtc_bind_addr,
            config.webrtc_stun_server.clone(),
        )
        .await?,
    );

    // Register the CRDT namespaces and start the inbound sync adapters, then
    // start gossip. Order matters: see the note at the mesh build above.
    let mesh_adapters = mesh_handler.as_ref().map(|handler| {
        MeshAdapters::start(
            handler.mesh_kv(),
            handler.self_name.clone(),
            app_context.worker_registry.clone(),
            handler.state.clone(),
            app_context.policy_registry.clone(),
        )
    });
    if let Some(mesh_server) = mesh_server {
        #[expect(
            clippy::disallowed_methods,
            reason = "mesh server runs for the lifetime of the process; shutdown is handled by the mesh handler"
        )]
        spawn(async move {
            if let Err(e) = mesh_server.start().await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
    }

    if config.prometheus_config.is_some() {
        app_context.inflight_tracker.start_sampler(20);
    }

    let weak_context = Arc::downgrade(&app_context);
    let worker_job_queue = JobQueue::new(
        JobQueueConfig {
            queue_capacity: config.router_config.job_queue_capacity,
            max_concurrent_jobs: config.router_config.job_queue_concurrency,
        },
        weak_context,
    );
    #[expect(
        clippy::expect_used,
        reason = "OnceLock initialization during startup; double-init is a fatal bug"
    )]
    app_context
        .worker_job_queue
        .set(worker_job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize typed workflow engines
    let engines = WorkflowEngines::new(&config.router_config);

    // Subscribe logging to all workflow engines
    engines.subscribe_all(Arc::new(LoggingSubscriber)).await;

    #[expect(
        clippy::expect_used,
        reason = "OnceLock initialization during startup; double-init is a fatal bug"
    )]
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");
    debug!(
        "Workflow engines initialized (health check timeout: {}s)",
        config.router_config.health_check.timeout_secs
    );

    // Submit startup tokenizer job if tokenizer path is configured
    // This runs before worker initialization to ensure tokenizer is available
    if config.router_config.disable_tokenizer_autoload {
        info!("Tokenizer autoload disabled via config; skipping startup tokenizer load");
    } else if let Some(tokenizer_source) = config
        .router_config
        .tokenizer_path
        .as_ref()
        .or(config.router_config.model_path.as_ref())
    {
        info!("Loading startup tokenizer from: {}", tokenizer_source);

        #[expect(
            clippy::expect_used,
            reason = "JobQueue was just initialized above; absence is unreachable"
        )]
        let job_queue = app_context
            .worker_job_queue
            .get()
            .expect("JobQueue should be initialized");

        let tokenizer_config = TokenizerConfigRequest {
            id: TokenizerRegistry::generate_id(),
            name: tokenizer_source.clone(),
            source: tokenizer_source.clone(),
            chat_template_path: config.router_config.chat_template.clone(),
            cache_config: config.router_config.tokenizer_cache.to_option(),
            fail_on_duplicate: false,
        };

        let job = Job::AddTokenizer {
            config: Box::new(tokenizer_config),
        };

        job_queue
            .submit(job)
            .await
            .map_err(|e| format!("Failed to submit startup tokenizer job: {e}"))?;

        info!("Startup tokenizer job submitted (will complete in background)");
    }

    info!(
        "Initializing workers for routing mode: {:?}",
        config.router_config.mode
    );

    // Submit worker initialization job to queue
    #[expect(
        clippy::expect_used,
        reason = "JobQueue was initialized above; absence is unreachable"
    )]
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");
    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(config.router_config.clone()),
    };
    job_queue
        .submit(job)
        .await
        .map_err(|e| format!("Failed to submit worker initialization job: {e}"))?;

    info!("Worker initialization job submitted (will complete in background)");

    if let Some(mcp_config) = &config.router_config.mcp_config {
        info!("Found {} MCP server(s) in config", mcp_config.servers.len());
        let mcp_job = Job::InitializeMcpServers {
            mcp_config: Box::new(mcp_config.clone()),
        };
        job_queue
            .submit(mcp_job)
            .await
            .map_err(|e| format!("Failed to submit MCP initialization job: {e}"))?;
    } else {
        info!("No MCP config provided, skipping MCP server initialization");
    }

    // Note: MCP orchestrator handles background refresh internally via refresh channel
    // configured by inventory.refresh_interval in mcp.yaml

    let worker_stats = app_context.worker_registry.stats();
    info!(
        "Workers initialized: {} total, {} healthy",
        worker_stats.total_workers, worker_stats.healthy_workers
    );

    let router_manager = RouterManager::from_config(&config, &app_context).await?;
    let router: Arc<dyn RouterTrait> = router_manager.clone();

    // WorkerManager owns the background health check loop. Its handle must
    // outlive the server to keep the task alive — bind it here so its Drop
    // (which aborts the task) runs at server shutdown. `maybe_start` decides
    // whether the loop has any work to do (health polling and/or promoting
    // workers that wait on the connect signal) and skips it otherwise. ZMQ
    // workers are promoted only by the connect signal, so pass the configured
    // transport up front: config workers register in the background after this
    // point, so the registry cannot yet reveal them.
    let _worker_manager = WorkerManager::maybe_start(
        app_context.worker_registry.clone(),
        WorkerManagerConfig {
            default_check_interval_secs: config.router_config.health_check.check_interval_secs,
            remove_unhealthy: config.router_config.health_check.remove_unhealthy_workers,
        },
        app_context.worker_job_queue.get().cloned(),
        !config.router_config.health_check.disable_health_check,
        config.router_config.connection_mode == ConnectionMode::Zmq,
    );

    // WorkerMonitor subscribes to registry events. Starting its event
    // loop here (after the synchronous worker population in
    // RouterManager::from_config above) means the bootstrap reconcile
    // captures every worker that exists at this point and the event
    // task picks up everything registered afterwards.
    if let Some(ref worker_monitor) = app_context.worker_monitor {
        worker_monitor.start_event_loop();
        debug!("Started WorkerMonitor event loop");
    }

    let admission_queue =
        if app_context.rate_limiter.is_some() && config.router_config.queue_size > 0 {
            debug!(
                "Admission queue enabled (size: {}, timeout: {}s)",
                config.router_config.queue_size, config.router_config.queue_timeout_secs
            );
            Some(Arc::new(AdmissionQueue::new(
                config.router_config.queue_size,
                Duration::from_secs(config.router_config.queue_timeout_secs),
            )))
        } else {
            None
        };

    if app_context.rate_limiter.is_none() {
        info!("Rate limiting is disabled (max_concurrent_requests = -1)");
    } else if admission_queue.is_none() {
        debug!(
            "Rate limiting enabled (max_concurrent_requests = {}, queue disabled)",
            config.router_config.max_concurrent_requests
        );
    }

    // Get mesh cluster state and port before moving mesh_handler into app_state
    let mesh_cluster_state = mesh_handler.as_ref().map(|h| h.state.clone());
    let mesh_port = config
        .mesh_server_config
        .as_ref()
        .map(|c| c.advertise_addr.port());

    // O(1) readiness state: maintained from WorkerRegistry events (plus a
    // short checkpoint for broadcast-bypassing mutations), read by
    // `/readiness` on the main listener and on the optional dedicated
    // probe listener below. Dropping the JoinHandle detaches the task.
    let probe_state = crate::health::ProbeState::new(app_context.inflight_tracker.clone());
    let _readiness_maintainer = crate::health::spawn_readiness_maintainer(
        probe_state.clone(),
        app_context.worker_registry.clone(),
        app_context.tokenizer_registry.clone(),
        config.router_config.clone(),
    );

    // Optional isolated probe listener (additive): when `--health-check-port`
    // is set, serves /liveness, /readiness, /health on that port from a
    // dedicated single-worker runtime on its own OS thread, so probes
    // cannot be starved by the request runtime. The same routes always remain
    // on the main listener.
    if let Some(probe_port) = config.health_check_port {
        let probe_addr =
            crate::health::start_probe_listener(&config.host, probe_port, probe_state.clone())?;
        info!("Probe listener started on {probe_addr} (--health-check-port {probe_port})");
    }

    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        admission_queue,
        router_manager: Some(router_manager),
        mesh_handler,
        mesh_adapters,
        probe_state,
    });
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            let app_context_arc = Arc::clone(&app_state.context);

            match start_service_discovery(
                service_discovery_config,
                app_context_arc,
                mesh_cluster_state,
                mesh_port,
            )
            .await
            {
                Ok(handle) => {
                    info!("Service discovery started");
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "service discovery runs for the lifetime of the server"
                    )]
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {e}");
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

    info!(
        "Router ready | workers: {:?}",
        WorkerManager::get_worker_urls(&app_state.context.worker_registry)
    );

    let request_id_headers = config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    // Serving routes accept both the shared key and per-tenant keys, so the
    // rate limiter (and anything else keyed on tenant identity) can tell
    // tenants apart. Admin/worker-management routes must NOT accept tenant
    // keys when falling back to simple API-key auth (no control-plane auth
    // configured) — a tenant credential must not be able to reach
    // `/workers`, `/flush_cache`, etc. Only the shared gateway-wide key does.
    let serving_auth_config = AuthConfig::with_tenant_keys(
        config.router_config.api_key.clone(),
        &config.router_config.tenant_api_keys,
    );
    let admin_auth_config = AuthConfig::new(config.router_config.api_key.clone());

    // Initialize control plane authentication if configured
    let control_plane_auth_state =
        smg_auth::ControlPlaneAuthState::try_init(config.control_plane_auth.as_ref()).await;

    let app = build_app(
        app_state,
        serving_auth_config,
        admin_auth_config,
        control_plane_auth_state,
        config.max_payload_size,
        request_id_headers,
        config.router_config.cors_allowed_origins.clone(),
    )?;

    // TcpListener::bind accepts &str and handles IPv4/IPv6 via ToSocketAddrs
    let bind_addr = format!("{}:{}", config.host, config.port);
    info!("Starting server on {}", bind_addr);

    // Parse address and set up graceful shutdown (common to both TLS and non-TLS)
    let addr: std::net::SocketAddr = bind_addr
        .parse()
        .map_err(|e| format!("Invalid address: {e}"))?;

    let handle = axum_server::Handle::new();
    let handle_clone = handle.clone();
    let inflight_tracker = app_context.inflight_tracker.clone();
    let grace = Duration::from_secs(config.shutdown_grace_period_secs);
    // Keep accepting for a short settle window before gating, so requests still
    // routed here during load-balancer / EndpointSlice propagation lag are
    // served instead of connection-refused — readiness already reports 503 by
    // then (#1694), so the orchestrator is removing this endpoint meanwhile.
    // Carved out of the grace budget (never more than half, capped at 5s — the
    // per-worker `drain_settle_secs` default) so total shutdown stays within
    // `shutdown_grace_period_secs` and never overruns terminationGracePeriod.
    let settle = (grace / 2).min(Duration::from_secs(5));
    let drain_timeout = grace.saturating_sub(settle);
    #[expect(
        clippy::disallowed_methods,
        reason = "shutdown signal handler must outlive the server to trigger graceful shutdown"
    )]
    spawn(async move {
        shutdown_signal().await;

        // Phase 1: Flip readiness to 503, hold the listener open through the
        // settle window so propagation-lagged requests still land, then gate
        // new connections.
        info!(
            in_flight = inflight_tracker.len(),
            "Beginning graceful shutdown: readiness draining"
        );
        inflight_tracker.begin_drain();
        if !settle.is_zero() {
            info!(
                settle_secs = settle.as_secs(),
                "Keeping listener open during load-balancer propagation window"
            );
            tokio::time::sleep(settle).await;
        }
        handle_clone.graceful_shutdown(Some(drain_timeout));

        // Phase 2: Drain — wait for in-flight requests to complete
        // Re-check after gating to catch requests that arrived between the
        // snapshot and graceful_shutdown stopping the accept loop.
        if !inflight_tracker.is_empty() {
            let drained = inflight_tracker.wait_for_drain(drain_timeout).await;
            if drained {
                info!("All in-flight requests drained");
            } else {
                warn!(
                    remaining = inflight_tracker.len(),
                    timeout_secs = drain_timeout.as_secs(),
                    "Drain timed out, forcing shutdown with requests still in-flight"
                );
            }
        }
        // Phase 3: Teardown proceeds after axum server stops (in the main task)
    });

    let server_result = if let (Some(cert), Some(key)) = (
        &config.router_config.server_cert,
        &config.router_config.server_key,
    ) {
        info!("TLS enabled");
        ring::default_provider()
            .install_default()
            .map_err(|e| format!("Failed to install rustls ring provider: {e:?}"))?;

        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem(cert.clone(), key.clone())
            .await
            .map_err(|e| format!("Failed to create TLS config: {e}"))?;

        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await
    } else {
        axum_server::bind(addr)
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await
    };

    // Graceful Shutdown

    info!("HTTP server stopped. Starting component cleanup...");

    // This triggers background task cancellation, waits for tools, and denies approvals
    if let Some(orchestrator) = app_context.mcp_orchestrator.get() {
        orchestrator.shutdown().await;
    }

    info!("Cleanup complete. Process exiting.");

    // Return original server error if any, otherwise Ok
    server_result.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

#[expect(
    clippy::expect_used,
    reason = "signal handler installation is infallible on supported platforms; failure is fatal"
)]
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        () = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([
                http::Method::GET,
                http::Method::POST,
                http::Method::PATCH,
                http::Method::DELETE,
                http::Method::OPTIONS,
            ])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TenantApiKeyEntry;

    fn minimal_server_config(router_config: RouterConfig) -> ServerConfig {
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            health_check_port: None,
            runtime_worker_threads: None,
            router_config,
            max_payload_size: 1024,
            log_dir: None,
            log_level: None,
            log_json: false,
            service_discovery_config: None,
            prometheus_config: None,
            request_timeout_secs: 60,
            request_id_headers: None,
            shutdown_grace_period_secs: 5,
            control_plane_auth: None,
            mesh_server_config: None,
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        }
    }

    /// `startup` must reject an invalid config even when `RouterConfig` was
    /// built directly (struct literal or `Deserialize`), not via
    /// `RouterConfigBuilder::build()` — a Rust library caller can construct
    /// `ServerConfig` this way and bypass the builder's validation entirely.
    /// Regression test for a reviewer-flagged gap: an empty tenant_api_keys
    /// entry reaching `AuthConfig` unvalidated would make `Authorization:
    /// Bearer ` (empty token) a valid serving credential.
    #[tokio::test]
    async fn startup_validates_directly_constructed_router_config() {
        let router_config = RouterConfig {
            tenant_api_keys: vec![TenantApiKeyEntry {
                tenant_id: "team-a".to_string(),
                key: String::new(),
            }],
            ..Default::default()
        };

        let result = startup(minimal_server_config(router_config)).await;
        assert!(
            result.is_err(),
            "startup must validate router_config even when constructed directly"
        );
    }
}
