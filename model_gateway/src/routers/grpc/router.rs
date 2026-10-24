use std::{borrow::Cow, future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    common::Tool as ChatTool,
    completion::CompletionRequest,
    decisions::DecisionsRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::{
        CountMessageTokensRequest, CountMessageTokensResponse, CreateMessageRequest, Message,
        MessageDelta, MessageDeltaUsage, MessageStreamEvent, StopReason, ToolChoice, Usage,
    },
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use tracing::{debug, warn};
use uuid::Uuid;

use super::{
    common::{
        responses::{
            background, handlers::cancel_response_impl, streaming::build_sse_response_from_stream,
            utils::validate_worker_availability, BackgroundResponses, BackgroundStream,
            ResponsesContext,
        },
        stages::{
            context_length_exceeded, helpers::resolve_request_id_stamp, smallest_context_length,
            RateLimitCell, RateLimitOutcome,
        },
    },
    context::{RequestType, SharedComponents},
    harmony::{serve_harmony_responses, serve_harmony_responses_stream, HarmonyDetector},
    mode::Mode,
    multimodal::{mm_settings, MultimodalComponents},
    pipeline::{Endpoint, PipelineDeps, RequestPipeline},
    regular::{
        responses, stages::decisions::supports_worker as supports_decisions_worker,
        streaming::StreamingProcessor,
    },
    utils::{message_utils, missing_tokenizer_response, ParserResolver},
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::metrics_labels,
    routers::{error, grpc::regular::symphony::template::Templates, RouterTrait},
    worker::{RoutingPool, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
};

/// gRPC router implementation for SGLang.
///
/// A single `Mode`-parameterized router serving Regular, PrefillDecode, and
/// EncodePrefillDecode. `mode` selects the disaggregation params baked into
/// every pipeline and drives the per-mode retry-metric labels, `Debug` output,
/// and `router_type`. Optional members are `None` when the mode doesn't serve
/// them and the corresponding endpoints 501: `embedding_pipeline` and
/// `classify_pipeline` are Regular-only; `harmony_pipeline`,
/// `responses_context`, and `harmony_responses_context` exist in every mode
/// but EPD.
#[derive(Clone)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    mode: Mode,
    pipeline: RequestPipeline,
    harmony_pipeline: Option<RequestPipeline>,
    embedding_pipeline: Option<RequestPipeline>,
    classify_pipeline: Option<RequestPipeline>,
    decisions_pipeline: Option<RequestPipeline>,
    transcription_pipeline: Option<RequestPipeline>,
    messages_pipeline: RequestPipeline,
    completion_pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
    responses_context: Option<ResponsesContext>,
    harmony_responses_context: Option<ResponsesContext>,
    retry_config: RetryConfig,
}

impl GrpcRouter {
    /// Regular and PD build the Harmony pipeline and responses contexts and
    /// require the MCP orchestrator; EPD leaves them `None` and 501s those
    /// endpoints. Embedding/classify pipelines are Regular-only.
    pub fn new(ctx: &Arc<AppContext>, mode: Mode) -> Result<Self, String> {
        // Get tokenizer registry (no longer requires pre-loaded tokenizer)
        let tokenizer_registry = ctx.tokenizer_registry.clone();

        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires tool parser factory".to_string())?
            .clone();

        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        // What can fail here is the operator's own setting, so the router
        // stops rather than coming up with media handling quietly switched
        // off and every media request failing later for no stated reason.
        let multimodal = Some(Arc::new(
            MultimodalComponents::new(
                ctx.multimodal_config_registry.clone(),
                ctx.router_config.mm_per_request_image_limit,
                ctx.router_config.multimodal_max_inflight_bytes,
                mm_settings(),
            )
            .map_err(|e| format!("multimodal components: {e:#}"))?,
        ));

        // Create shared components for pipeline
        let shared_components = Arc::new(SharedComponents {
            tokenizer_registry: tokenizer_registry.clone(),
            worker_registry: worker_registry.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
            parser_resolver: ParserResolver::new(
                worker_registry.clone(),
                ctx.configured_tool_parser.clone(),
                ctx.configured_reasoning_parser.clone(),
                &tool_parser_factory,
                &reasoning_parser_factory,
            ),
            multimodal,
            tokenizer_autoload: !ctx.router_config.disable_tokenizer_autoload,
            tokenizer_load_failures: ctx.tokenizer_load_failures.clone(),
            parser: ctx.router_config.parser,
            templates: Arc::new(Templates::new(tokenizer_registry.clone())),
        });

        // The admission queue's timeout also bounds how long an accepted
        // request may sit ahead of worker selection.
        let queue_timeout = Duration::from_secs(ctx.router_config.queue_timeout_secs);
        // The request timeout bounds each attempt's calls to its workers and
        // the streams they open, as it bounds the HTTP path's calls; the
        // silence bounds (off unless set) bound the wait for the first
        // message and between messages.
        let request_timeout = Duration::from_secs(ctx.router_config.request_timeout_secs);
        let first_token_timeout = Duration::from_secs(ctx.router_config.first_token_timeout_secs);
        let inter_token_timeout = Duration::from_secs(ctx.router_config.inter_token_timeout_secs);
        // Deps for the parser-consuming endpoints (chat/messages/harmony).
        let configured_deps = PipelineDeps::new(
            worker_registry.clone(),
            policy_registry.clone(),
            ctx.prefill_admission.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
            ctx.rate_limit_manager.clone(),
            queue_timeout,
        )
        .with_request_timeout(request_timeout)
        .with_silence_bounds(first_token_timeout, inter_token_timeout);
        // Deps for the parser-free endpoints (completion/embeddings/classify/
        // decisions/transcription).
        let pair_deps = PipelineDeps::pair(
            worker_registry.clone(),
            policy_registry.clone(),
            ctx.prefill_admission.clone(),
            ctx.rate_limit_manager.clone(),
            queue_timeout,
        )
        .with_request_timeout(request_timeout)
        .with_silence_bounds(first_token_timeout, inter_token_timeout);

        // Present in every mode: chat/generate, messages, completion.
        let pipeline = RequestPipeline::build(Endpoint::Chat, mode, &configured_deps)
            .ok_or_else(|| format!("gRPC router: no chat pipeline for mode {mode:?}"))?;
        let messages_pipeline = RequestPipeline::build(Endpoint::Messages, mode, &configured_deps)
            .ok_or_else(|| format!("gRPC router: no messages pipeline for mode {mode:?}"))?;
        let completion_pipeline = RequestPipeline::build(Endpoint::Completion, mode, &pair_deps)
            .ok_or_else(|| format!("gRPC router: no completion pipeline for mode {mode:?}"))?;

        // `None` when the (endpoint, mode) combo is unsupported; those endpoints 501.
        let harmony_pipeline = RequestPipeline::build(Endpoint::Harmony, mode, &configured_deps);
        let embedding_pipeline = RequestPipeline::build(Endpoint::Embeddings, mode, &pair_deps);
        let classify_pipeline = RequestPipeline::build(Endpoint::Classify, mode, &pair_deps);
        let decisions_pipeline = RequestPipeline::build(Endpoint::Decisions, mode, &pair_deps);
        let transcription_pipeline =
            RequestPipeline::build(Endpoint::Transcription, mode, &pair_deps);

        // Responses contexts are the sole consumer of the MCP orchestrator; EPD
        // builds neither (it doesn't serve /v1/responses).
        let (responses_context, harmony_responses_context) = if mode == Mode::EncodePrefillDecode {
            (None, None)
        } else {
            let mcp_orchestrator = ctx
                .mcp_orchestrator
                .get()
                .ok_or_else(|| "gRPC router requires MCP manager".to_string())?
                .clone();

            // Capture storage request context from middleware task-local (before any spawn)
            let storage_request_context = smg_data_connector::current_request_context();
            // One registry of background responses for both contexts: one cap,
            // one place a cancel looks a running id up.
            let background_responses = Arc::new(BackgroundResponses::new(
                ctx.router_config.max_background_responses,
            ));

            // Helper closure to create responses context with a given pipeline
            let create_responses_context = |pipeline: &RequestPipeline| {
                ResponsesContext::new(
                    Arc::new(pipeline.clone()),
                    shared_components.clone(),
                    ctx.response_storage.clone(),
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    mcp_orchestrator.clone(),
                    ctx.mcp_format_registry.clone(),
                    storage_request_context.clone(),
                    background_responses.clone(),
                )
            };

            let responses_context = create_responses_context(&pipeline);
            let harmony_responses_context = harmony_pipeline
                .as_ref()
                .map(&create_responses_context)
                .ok_or_else(|| {
                    format!("gRPC router: mode {mode:?} must build a harmony pipeline")
                })?;
            (Some(responses_context), Some(harmony_responses_context))
        };

        Ok(GrpcRouter {
            worker_registry,
            mode,
            pipeline,
            harmony_pipeline,
            embedding_pipeline,
            classify_pipeline,
            decisions_pipeline,
            transcription_pipeline,
            messages_pipeline,
            completion_pipeline,
            shared_components,
            responses_context,
            harmony_responses_context,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    /// The per-model retry override registered by a worker, else the router
    /// default. Applied at every retrying endpoint
    /// (chat/generate/messages/completion) in every mode.
    ///
    /// Retry overrides are keyed by canonical model ID, so `model_id` must
    /// already be canonical (e.g. via [`Self::resolve_canonical_model_id`]) --
    /// every `route_*_impl` resolves that once before the retry loop starts,
    /// so this never needs to resolve an alias itself. Retry overrides are
    /// read before the pipeline canonicalizes in `RequestContext::new`, so an
    /// alias reaching this unresolved would miss the override and silently
    /// fall back to the router default.
    fn resolve_retry_config_for_canonical(&self, canonical_model_id: &str) -> RetryConfig {
        self.worker_registry
            .get_retry_config(canonical_model_id)
            .unwrap_or_else(|| self.retry_config.clone())
    }

    /// Resolve `model_id` to its canonical form once, before the pipeline
    /// runs. Every dispatch attempt for that logical request reuses this
    /// value -- re-resolving mid-request would let an alias repointed during
    /// the retry window dispatch a different model than whatever the tenant
    /// rate-limit reservation was actually made for, bypassing that model's
    /// own policy and settling its usage against the wrong budget.
    fn resolve_canonical_model_id(&self, model_id: &str) -> String {
        self.worker_registry
            .resolve_model_alias(model_id)
            .as_deref()
            .unwrap_or(model_id)
            .to_string()
    }

    /// Close a reservation that a non-2xx final response never got the
    /// chance to settle (prep failure repeated across every attempt,
    /// retries exhausted, a non-retryable dispatch failure). No-op if
    /// nothing was ever reserved. Safe to call unconditionally alongside a
    /// success path's own inline `settle_success`/streaming `settle_success`+
    /// `ReservationAttachment` -- `close_reserved_only` is CAS-guarded, only
    /// the first resolution of a handle wins.
    async fn close_reservation_if_unsettled(cell: &RateLimitCell, status: StatusCode) {
        if status.is_success() {
            return;
        }
        if let Some(RateLimitOutcome::Admitted(handle)) = cell.peek() {
            handle.close_reserved_only().await;
        }
    }

    /// Run a pipeline call for the canonical `model_id` under a tenant
    /// rate-limit reservation: the cell the pipeline reserves into, the
    /// model's retry config (keyed by the canonical id, see
    /// [`Self::resolve_retry_config_for_canonical`]), `run` with those and
    /// the id, then the close of a reservation a non-2xx answer never
    /// settled. Every typed route runs its pipeline through here; the
    /// Responses iterations, which reserve nothing today, call theirs
    /// directly (see `common/stages/rate_limit.rs`).
    async fn execute_reserved<Fut>(
        &self,
        model_id: String,
        run: impl FnOnce(String, Arc<RateLimitCell>, RetryConfig) -> Fut,
    ) -> Response
    where
        Fut: Future<Output = Response>,
    {
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let retry_config = self.resolve_retry_config_for_canonical(&model_id);
        let response = run(model_id, rate_limit_cell.clone(), retry_config).await;
        Self::close_reservation_if_unsettled(&rate_limit_cell, response.status()).await;
        response
    }

    /// The chat request as the pipeline takes it: validated, its `model`
    /// field canonical, in the `Arc` the request context holds, with the
    /// pipeline that serves it. Done before the async hops of
    /// [`Self::route_chat_impl`], so that only the handle travels through
    /// their state machines, not a copy of the 1 KiB request per hop.
    fn chat_request_for_pipeline(
        &self,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Result<(Arc<ChatCompletionRequest>, String, &RequestPipeline), Response> {
        if let Err(response) = super::validate_text_only_output(&body) {
            return Err(*response);
        }

        // EPD has no Harmony pipeline, so its chat requests all use the single
        // chat/generate pipeline.
        let is_harmony = self.harmony_pipeline.is_some()
            && HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        debug!(
            "Processing chat completion request for model: {}, using_harmony={}",
            model_id, is_harmony
        );

        let pipeline = match self.harmony_pipeline.as_ref() {
            Some(harmony_pipeline) if is_harmony => harmony_pipeline,
            _ => &self.pipeline,
        };

        // Canonicalize once, up front, so every dispatch attempt (and the
        // reservation) targets the same model -- see
        // `resolve_canonical_model_id`'s doc comment. The body's own `model`
        // field is rewritten to match: by this point `model_id_cloned` is
        // already canonical, so `RequestContext::new`'s own alias resolve
        // would no-op and otherwise leave the alias sitting in the body for
        // response metadata and parser selection to read.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model.clone_from(&model_id_cloned);
        Ok((Arc::new(canonical_body), model_id_cloned, pipeline))
    }

    /// Main route_chat implementation, over the request
    /// [`Self::chat_request_for_pipeline`] prepared.
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: Arc<ChatCompletionRequest>,
        model_id: String,
        pipeline: &RequestPipeline,
    ) -> Response {
        // The request Arc moves into the pipeline: no handle survives out
        // here, so the parsed request frees at the build boundary.
        self.execute_reserved(
            model_id,
            |model_id, rate_limit_cell, retry_config| async move {
                pipeline
                    .execute_chat(
                        body,
                        headers.cloned(),
                        model_id,
                        self.shared_components.clone(),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                        Some(&retry_config),
                        None,
                    )
                    .await
            },
        )
        .await
    }

    /// Main route_generate implementation
    async fn route_generate_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: GenerateRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing generate request for model: {}", model_id);

        // slime (and other native RL clients) never send `model` on
        // `/generate`, so the request lands here carrying the wildcard
        // placeholder. When the fleet serves exactly one real model there is
        // only one sane target, so default to it instead of 404ing a
        // request the registry could otherwise route without ambiguity.
        // Zero or several served models keep the current behavior:
        // model_not_found. An untagged worker (no model card, no
        // `model_id` label) registers under the wildcard itself
        // (`Worker::model_id`'s fallback), so it is filtered out here:
        // otherwise it would mask a real single-model fleet behind a
        // spurious second entry.
        let resolved_model_id: Cow<'_, str> = if model_id == UNKNOWN_MODEL_ID {
            let served_models: Vec<String> = self
                .worker_registry
                .get_models()
                .into_iter()
                .filter(|served| served != UNKNOWN_MODEL_ID)
                .collect();
            match served_models.as_slice() {
                [only] => {
                    debug!(
                        "Defaulting model-less generate request to the single served model: {}",
                        only
                    );
                    Cow::Owned(only.clone())
                }
                _ => Cow::Borrowed(model_id),
            }
        } else {
            Cow::Borrowed(model_id)
        };

        // Canonicalize once, up front -- see `resolve_canonical_model_id`'s
        // doc comment. Rewrite the body's `model` field to match; see
        // `chat_request_for_pipeline`.
        let model_id_cloned = self.resolve_canonical_model_id(&resolved_model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();

        self.execute_reserved(
            model_id_cloned,
            |model_id, rate_limit_cell, retry_config| async move {
                self.pipeline
                    .execute_generate(
                        Arc::new(canonical_body),
                        headers.cloned(),
                        model_id,
                        self.shared_components.clone(),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                        Some(&retry_config),
                    )
                    .await
            },
        )
        .await
    }

    /// Main route_responses implementation
    ///
    /// Routes to either Harmony or regular responses implementation based on model detection
    async fn route_responses_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let (Some(responses_context), Some(harmony_responses_context)) =
            (&self.responses_context, &self.harmony_responses_context)
        else {
            return error::not_implemented(
                "responses_not_implemented",
                "Responses endpoint not implemented",
            );
        };
        let Some(harmony_pipeline) = self.harmony_pipeline.as_ref() else {
            return error::not_implemented(
                "responses_not_implemented",
                "Responses endpoint not implemented",
            );
        };

        // 0. Fast worker validation (fail-fast before expensive operations)
        if let Some(error_response) = validate_worker_availability(&self.worker_registry, model_id)
        {
            return error_response;
        }

        let (body, canonical_model_id) =
            canonicalize_responses_request(&self.worker_registry, &body, model_id);
        let model_id = canonical_model_id.as_ref();

        // Choose implementation based on Harmony model detection (checks worker metadata)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        // `background: true` answers the queued object at once and runs the
        // work behind the response id; with the mode switched off such a
        // request runs in the foreground, and its object says so.
        let mut body = body.into_owned();
        if body.background == Some(true) {
            if responses_context.background.enabled() {
                let request_context = smg_data_connector::current_request_context();
                let ctx = if is_harmony {
                    self.harmony_responses_context(
                        harmony_pipeline,
                        harmony_responses_context,
                        request_context,
                    )
                } else {
                    ResponsesContext {
                        request_context,
                        ..responses_context.clone()
                    }
                };
                return self
                    .route_background_responses(
                        &ctx,
                        is_harmony,
                        headers,
                        tenant_meta,
                        body,
                        model_id,
                    )
                    .await;
            }
            body.background = None;
        }

        if is_harmony {
            debug!(
                "Processing Harmony responses request for model: {}, streaming: {}",
                model_id,
                body.stream.unwrap_or(false)
            );
            let harmony_ctx = self.harmony_responses_context(
                harmony_pipeline,
                harmony_responses_context,
                smg_data_connector::current_request_context(),
            );

            if body.stream.unwrap_or(false) {
                serve_harmony_responses_stream(&harmony_ctx, body, tenant_meta.clone(), None).await
            } else {
                match serve_harmony_responses(&harmony_ctx, body, tenant_meta.clone(), None).await {
                    Ok(response) => Json(response).into_response(),
                    Err(error_response) => error_response,
                }
            }
        } else {
            responses::route_responses(
                responses_context,
                Arc::new(body),
                headers.cloned(),
                tenant_meta.clone(),
                model_id.to_string(),
            )
            .await
        }
    }

    /// The per-request context of the Harmony Responses path: the Harmony
    /// pipeline with the shared storage and MCP handles and this request's
    /// storage context.
    fn harmony_responses_context(
        &self,
        harmony_pipeline: &RequestPipeline,
        harmony_responses_context: &ResponsesContext,
        request_context: Option<smg_data_connector::RequestContext>,
    ) -> ResponsesContext {
        ResponsesContext::new(
            Arc::new(harmony_pipeline.clone()),
            self.shared_components.clone(),
            harmony_responses_context.response_storage.clone(),
            harmony_responses_context.conversation_storage.clone(),
            harmony_responses_context.conversation_item_storage.clone(),
            harmony_responses_context.mcp_orchestrator.clone(),
            harmony_responses_context.mcp_format_registry.clone(),
            request_context,
            harmony_responses_context.background.clone(),
        )
    }

    /// `background: true`: the `queued` Response object is stored under a new
    /// id and answered (or streamed) at once; the work runs in a task of its
    /// own behind that id, on the Harmony or the regular path. One beyond
    /// the configured number in flight answers 429.
    async fn route_background_responses(
        &self,
        ctx: &ResponsesContext,
        is_harmony: bool,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        request: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let id = format!("resp_{}", Uuid::now_v7());
        let Some(slot) = ctx.background.admit(&id) else {
            warn!(
                in_flight = ctx.background.in_flight(),
                max_in_flight = ctx.background.max_in_flight(),
                "Background response refused: the limit is reached"
            );
            return background::limit_reached(ctx.background.max_in_flight());
        };
        let request = Arc::new(request);
        let queued = background::queued_response(&id, &request);
        if let Err(e) = background::store_record(
            &ctx.response_storage,
            &queued,
            &request,
            crate::tenant::storage_tenant(tenant_meta).as_deref(),
            ctx.request_context.clone(),
        )
        .await
        {
            return error::storage_failure(
                "store_response_failed",
                "Failed to store the queued response",
                &e,
            );
        }
        debug!(response_id = %id, harmony = is_harmony, "Background response queued");

        let tenant_meta = tenant_meta.clone();
        let headers = headers.cloned();
        let model_id = model_id.to_string();
        if request.stream.unwrap_or(false) {
            let stream = BackgroundStream::new(slot);
            let answer = if is_harmony {
                serve_harmony_responses_stream(
                    ctx,
                    Arc::unwrap_or_clone(request),
                    tenant_meta,
                    Some(stream),
                )
                .await
            } else {
                responses::route_responses_background_stream(
                    ctx,
                    request,
                    headers,
                    tenant_meta,
                    model_id,
                    stream,
                )
                .await
            };
            // A plain error before the stream opened closes the record as
            // failed; the stream itself runs behind the id.
            return background::close_before_stream(
                &ctx.response_storage,
                ctx.request_context.clone(),
                &id,
                answer,
            )
            .await;
        }

        if is_harmony {
            background::spawn(
                ctx.clone(),
                slot,
                request,
                Box::new(move |ctx, request| {
                    Box::pin(async move {
                        serve_harmony_responses(
                            ctx,
                            Arc::unwrap_or_clone(request),
                            tenant_meta,
                            Some(id),
                        )
                        .await
                    })
                }),
            );
        } else {
            background::spawn(
                ctx.clone(),
                slot,
                request,
                Box::new(move |ctx, request| {
                    Box::pin(async move {
                        responses::route_responses_background(
                            ctx,
                            request,
                            headers,
                            tenant_meta,
                            model_id,
                            id,
                        )
                        .await
                    })
                }),
            );
        }
        (StatusCode::OK, Json(queued)).into_response()
    }

    /// Main route_embeddings implementation
    async fn route_embeddings_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        let Some(embedding_pipeline) = self.embedding_pipeline.as_ref() else {
            return error::not_implemented(
                "embeddings_not_implemented",
                "Embeddings not implemented",
            );
        };
        debug!("Processing embedding request for model: {}", model_id);

        // One dispatch, no retry loop, as before: the retry config the
        // reserved execution resolves is not read here.
        self.execute_reserved(
            model_id.to_string(),
            |model_id, rate_limit_cell, _retry_config| async move {
                embedding_pipeline
                    .execute_embeddings(
                        Arc::new(body),
                        headers.cloned(),
                        model_id,
                        self.shared_components.clone(),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                    )
                    .await
            },
        )
        .await
    }

    /// Main route_messages implementation
    async fn route_messages_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing messages request for model: {}", model_id);

        // Canonicalize once, up front -- see `resolve_canonical_model_id`'s
        // doc comment. Rewrite the body's `model` field to match; see
        // `chat_request_for_pipeline`.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let canonical_body = Arc::new(canonical_body);

        // `max_tokens: 0` is answered as the public API answers it: an empty
        // message that reports the prompt's token count; nothing is generated.
        // No worker takes part, so the answer carries no rate-limit
        // reservation, as count_tokens reserves none; it is counted as a
        // Messages request, refusals included, where the pipeline counts
        // its own, so the traffic stays visible.
        if canonical_body.max_tokens == 0 {
            const ENDPOINT: &str = metrics_labels::ENDPOINT_MESSAGES;
            self.messages_pipeline.record_request(
                ENDPOINT,
                &model_id_cloned,
                canonical_body.stream.unwrap_or(false),
            );
            let response = self
                .empty_message_response(tenant_meta, canonical_body, &model_id_cloned)
                .await;
            if !response.status().is_success() {
                self.messages_pipeline
                    .record_error(Some(ENDPOINT), &model_id_cloned, &response);
            }
            return response;
        }

        self.execute_reserved(
            model_id_cloned,
            |model_id, rate_limit_cell, retry_config| async move {
                self.messages_pipeline
                    .execute_messages(
                        canonical_body,
                        headers.cloned(),
                        model_id,
                        self.shared_components.clone(),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                        Some(&retry_config),
                    )
                    .await
            },
        )
        .await
    }

    /// `/v1/messages/count_tokens`, answered from the gateway's own
    /// tokenizer: the request is rendered through the model's chat template
    /// exactly as `/v1/messages` renders it and the prompt's token count is
    /// returned. No worker takes part, so every worker type is covered.
    async fn route_messages_count_tokens_impl(
        &self,
        body: CountMessageTokensRequest,
        model_id: &str,
    ) -> Response {
        let model_id = self.resolve_canonical_model_id(model_id);
        match self
            .count_prompt_tokens(
                &model_id,
                message_utils::MessagesPrompt::from(&body),
                &body.chat_tools(),
                body.tool_choice.as_ref(),
                false,
            )
            .await
        {
            Ok(input_tokens) => (
                StatusCode::OK,
                Json(CountMessageTokensResponse { input_tokens }),
            )
                .into_response(),
            Err(refused) => refused,
        }
    }

    /// The prompt's token count, as `/v1/messages/count_tokens` reports it:
    /// the registry's tokenizer for the model, the tools the template renders,
    /// the request's media and the effort its `output_config` states (in
    /// `other`, the request's untyped fields) resolved as the preparation
    /// stage resolves them; no worker takes part. The refusals are
    /// count_tokens': a model nobody serves is the client's mistake (404),
    /// whatever a removed worker left in the tokenizer registry; a served
    /// model whose tokenizer the gateway is still loading is not ready yet
    /// (503 with a `Retry-After`); one whose tokenizer load was skipped or
    /// failed for good will not get one on its own (503 without); a served
    /// model with no tokenizer when the gateway does not load tokenizers
    /// itself is a gateway fault (500), as is a prompt that does not encode;
    /// a prompt that does not render is a 400.
    /// `record_media` says whether the prompt's media counts in the
    /// per-request media series: the `max_tokens: 0` answer is a Messages
    /// request, a token count is not.
    async fn count_prompt_tokens(
        &self,
        model_id: &str,
        prompt: message_utils::MessagesPrompt<'_>,
        chat_tools: &[ChatTool],
        tool_choice: Option<&ToolChoice>,
        record_media: bool,
    ) -> Result<u32, Response> {
        if !self.worker_registry.contains_model(model_id) {
            return Err(error::model_not_found(model_id));
        }
        let Some(tokenizer) = self.shared_components.tokenizer_registry.get(model_id) else {
            return Err(missing_tokenizer_response(
                "GrpcRouter::count_prompt_tokens",
                model_id,
                true,
                self.shared_components.tokenizer_autoload,
                self.shared_components
                    .tokenizer_load_failures
                    .get(model_id)
                    .as_ref(),
            ));
        };
        let tools = message_utils::template_tools(
            chat_tools,
            tool_choice,
            tokenizer.renderer_capabilities(),
        );
        // A token count is not a request the media series count; the
        // `max_tokens: 0` answer is one, and the caller says which.
        let media = message_utils::resolve_messages_media(
            &self.shared_components,
            model_id,
            &*tokenizer,
            prompt.messages,
            record_media,
        )
        .await?;
        message_utils::count_input_tokens(
            prompt,
            tokenizer,
            &tools,
            media.placeholders(),
            media.media_order,
        )
        .await
        .map_err(|e| match e {
            message_utils::CountError::Render(e) => {
                debug!(model = %model_id, error = %e, "the request does not render");
                error::bad_request("process_messages_failed", e)
            }
            message_utils::CountError::Encode(e) => {
                tracing::error!(model = %model_id, error = %e, "tokenization failed");
                error::internal_error("tokenization_failed", format!("Tokenization failed: {e}"))
            }
        })
    }

    /// The answer to `max_tokens: 0`, as the public API answers it: a message
    /// with no content, `stop_reason: max_tokens` and the prompt's token count
    /// in `usage`, counted as `/v1/messages/count_tokens` counts it; no worker
    /// takes part, and the message carries the id the pipeline would have
    /// given it. A streaming request gets that message as `message_start`,
    /// then `message_delta` with the stop reason and `message_stop`. A request
    /// with media is counted at the placeholders the template writes for it,
    /// as count_tokens counts it: the media is neither fetched nor decoded, so
    /// the count is the template's, not the expanded prompt's; an image or a
    /// video to a pool whose every worker is language model only, and media
    /// no worker of the model takes (an image to a pool of engines without
    /// image input), are refused with `multimodal_not_supported` here as on
    /// every Messages path. A
    /// prompt longer than the smallest context window the model's workers
    /// advertise is refused as the pipeline refuses it after worker selection
    /// (`context_length_exceeded`), so a 200 here means the prompt fits
    /// whichever worker a real call lands on; a deployment that advertises no
    /// window leaves the check to the engine, as the pipeline does.
    async fn empty_message_response(
        &self,
        tenant_meta: &TenantRequestMeta,
        body: Arc<CreateMessageRequest>,
        model_id: &str,
    ) -> Response {
        let input_tokens = match self
            .count_prompt_tokens(
                model_id,
                message_utils::MessagesPrompt::from(&*body),
                &body.chat_tools(),
                body.tool_choice.as_ref(),
                true,
            )
            .await
        {
            Ok(input_tokens) => input_tokens,
            Err(refused) => return refused,
        };
        if let Some(limit) =
            smallest_context_length(&self.worker_registry.get_by_model(model_id), model_id)
        {
            if input_tokens > limit {
                return context_length_exceeded(input_tokens as usize, limit, model_id);
            }
        }
        let stream = body.stream.unwrap_or(false);
        let (id, _) = resolve_request_id_stamp(
            &RequestType::Messages(body),
            Some(tenant_meta),
            "msg_",
            self.mode != Mode::Regular,
        );
        let message = Message {
            id,
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: Vec::new(),
            model: model_id.to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            stop_sequence: None,
            stop_details: None,
            usage: Usage::from_counts(input_tokens, 0),
            container: None,
        };
        if !stream {
            return (StatusCode::OK, Json(message)).into_response();
        }

        let events = [
            MessageStreamEvent::MessageStart {
                message: Message {
                    stop_reason: None,
                    ..message
                },
            },
            MessageStreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason: Some(StopReason::MaxTokens),
                    ..MessageDelta::default()
                },
                usage: MessageDeltaUsage {
                    output_tokens: 0,
                    input_tokens: Some(input_tokens),
                    cache_creation_input_tokens: Some(0),
                    cache_read_input_tokens: Some(0),
                    server_tool_use: None,
                    output_tokens_details: None,
                },
            },
            MessageStreamEvent::MessageStop,
        ];
        let mut frames: Vec<Result<bytes::Bytes, std::io::Error>> =
            Vec::with_capacity(events.len());
        let mut buffer = Vec::new();
        for event in &events {
            if let Err(e) = StreamingProcessor::format_messages_sse_into(&mut buffer, event) {
                tracing::error!(model = %model_id, error = %e, "max_tokens 0: a stream event does not serialize");
                return error::internal_error("messages_event_serialization_failed", e);
            }
            frames.push(Ok(bytes::Bytes::from(std::mem::take(&mut buffer))));
        }
        build_sse_response_from_stream(futures::stream::iter(frames))
    }

    /// Main route_completion implementation
    async fn route_completion_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CompletionRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing completion request for model: {}", model_id);

        // Canonicalize once, up front -- see `resolve_canonical_model_id`'s
        // doc comment. Rewrite the body's `model` field to match; see
        // `chat_request_for_pipeline`.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();

        self.execute_reserved(
            model_id_cloned,
            |model_id, rate_limit_cell, retry_config| async move {
                self.completion_pipeline
                    .execute_completion(
                        Arc::new(canonical_body),
                        headers.cloned(),
                        model_id,
                        self.shared_components.clone(),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                        Some(&retry_config),
                    )
                    .await
            },
        )
        .await
    }

    /// Score Decisions through the regular SGLang pipeline.
    async fn route_decisions_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        mut body: DecisionsRequest,
        model_id: &str,
    ) -> Response {
        let Some(pipeline) = self.decisions_pipeline.as_ref() else {
            return error::not_implemented(
                "decisions_not_supported",
                "Decisions requires a regular SGLang gRPC router",
            );
        };
        let canonical = self.resolve_canonical_model_id(model_id);
        if canonical == UNKNOWN_MODEL_ID {
            return error::model_not_found(model_id);
        }
        let Some(snapshot) = self.worker_registry.model_routing_snapshot(&canonical) else {
            return error::model_not_found(model_id);
        };
        if !snapshot
            .pool(RoutingPool::GrpcPipelineRegular)
            .iter()
            .any(|worker| supports_decisions_worker(worker.as_ref()))
        {
            return error::not_implemented(
                "decisions_not_supported",
                "Decisions scoring requires a regular SGLang gRPC worker",
            );
        }
        body.model = canonical.clone();
        self.execute_reserved(
            canonical,
            |model_id, rate_limit_cell, retry_config| async move {
                pipeline
                    .execute_decisions(
                        Arc::new(body),
                        headers.cloned(),
                        model_id,
                        self.shared_components.clone(),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                        Some(&retry_config),
                    )
                    .await
            },
        )
        .await
    }

    /// Main route_classify implementation
    async fn route_classify_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ClassifyRequest,
        model_id: &str,
    ) -> Response {
        let Some(classify_pipeline) = self.classify_pipeline.as_ref() else {
            return error::not_implemented("classify_not_implemented", "Classify not implemented");
        };
        debug!("Processing classify request for model: {}", model_id);

        // One dispatch, no retry loop, as before.
        self.execute_reserved(
            model_id.to_string(),
            |model_id, rate_limit_cell, _retry_config| async move {
                classify_pipeline
                    .execute_classify(
                        Arc::new(body),
                        headers.cloned(),
                        model_id,
                        self.shared_components.clone(),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                    )
                    .await
            },
        )
        .await
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.mode {
            Mode::Regular => {
                let stats = self.worker_registry.stats();
                f.debug_struct("GrpcRouter")
                    .field("workers_count", &stats.total_workers)
                    .finish()
            }
            Mode::PrefillDecode | Mode::EncodePrefillDecode => {
                // Count every worker this router can serve (gRPC and ZMQ both
                // ride the gRPC pipeline), not just ConnectionMode::Grpc.
                let count_pipeline_workers = |worker_type| {
                    self.worker_registry
                        .get_workers_filtered(None, Some(worker_type), None, None, false)
                        .iter()
                        .filter(|w| w.connection_mode().uses_grpc_pipeline())
                        .count()
                };
                let prefill_workers = count_pipeline_workers(WorkerType::Prefill);
                let decode_workers = count_pipeline_workers(WorkerType::Decode);
                f.debug_struct("GrpcRouter")
                    .field("prefill_workers_count", &prefill_workers)
                    .field("decode_workers_count", &decode_workers)
                    .finish()
            }
        }
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: GenerateRequest,
        model_id: &str,
    ) -> Response {
        self.route_generate_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        match self.chat_request_for_pipeline(body, model_id) {
            Ok((body, model_id, pipeline)) => {
                self.route_chat_impl(headers, tenant_meta, body, model_id, pipeline)
                    .await
            }
            Err(response) => response,
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        self.route_responses_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn cancel_response(
        &self,
        _headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        response_id: &str,
    ) -> Response {
        let Some(responses_context) = self.responses_context.as_ref() else {
            return error::not_implemented(
                "cancel_response_not_implemented",
                "Cancel response not implemented",
            );
        };
        let tenant = crate::tenant::storage_tenant(tenant_meta);
        cancel_response_impl(responses_context, tenant.as_deref(), response_id).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        self.route_embeddings_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ClassifyRequest,
        model_id: &str,
    ) -> Response {
        self.route_classify_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_decisions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: DecisionsRequest,
        model_id: &str,
    ) -> Response {
        self.route_decisions_impl(headers, tenant_meta, body, model_id)
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
        // Transcription is a first-class pipeline endpoint (Regular-only, so
        // `None` in PD/EPD). Detection, the chat-shaped request synthesis, and
        // output parsing all live inside the pipeline's transcription stages.
        let Some(pipeline) = self.transcription_pipeline.as_ref() else {
            return error::not_implemented(
                "audio_transcriptions_not_implemented",
                "Audio transcriptions not implemented",
            );
        };
        // One dispatch, no retry loop, as before.
        self.execute_reserved(
            model_id.to_string(),
            |model_id, rate_limit_cell, _retry_config| async move {
                pipeline
                    .execute_transcription(
                        Arc::new(body.clone()),
                        Arc::new(audio),
                        headers.cloned(),
                        model_id,
                        Arc::clone(&self.shared_components),
                        Some(tenant_meta.clone()),
                        Some(rate_limit_cell),
                    )
                    .await
            },
        )
        .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_completion_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        self.route_messages_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_messages_count_tokens(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: CountMessageTokensRequest,
        model_id: &str,
    ) -> Response {
        self.route_messages_count_tokens_impl(body, model_id).await
    }

    fn router_type(&self) -> &'static str {
        self.mode.router_type()
    }
}

/// Resolve a Responses request's model alias, for both the routing decisions
/// and the request the Responses layer will read.
///
/// The Responses layer builds its SSE events and its tool call responses from
/// the request rather than from the pipeline context, so the alias has to be
/// gone before dispatch. Otherwise a plain answer reports the canonical ID
/// while a tool call answer reports the alias.
///
/// Returns both halves together so a caller cannot take the canonical model ID
/// and forget the body, or the reverse. The common path borrows both untouched
/// and allocates nothing.
fn canonicalize_responses_request<'a>(
    worker_registry: &WorkerRegistry,
    body: &'a ResponsesRequest,
    model_id: &'a str,
) -> (Cow<'a, ResponsesRequest>, Cow<'a, str>) {
    let Some(canonical_model) = worker_registry.resolve_model_alias(model_id) else {
        return (Cow::Borrowed(body), Cow::Borrowed(model_id));
    };
    let mut canonical_body = body.clone();
    canonical_body.model.clear();
    canonical_body.model.push_str(&canonical_model);
    (
        Cow::Owned(canonical_body),
        Cow::Owned(canonical_model.to_string()),
    )
}

#[cfg(test)]
mod pd_tests;
